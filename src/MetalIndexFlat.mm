#import <faiss-metal/MetalIndexFlat.h>
#import <faiss-metal/MetalResources.h>
#import <faiss/IndexFlat.h>
#import "MetalDistance.h"
#import "MetalL2Norm.h"
#import "MetalSelect.h"
#import "MetalTensor.h"
#include <faiss/impl/FaissAssert.h>
#include <cstring>

namespace faiss_metal {

struct MetalIndexFlat::Impl {
    std::shared_ptr<MetalResources> resources;
    MetalTensor<float, 2> vectors;   // (ntotal x d) stored vectors
    MetalTensor<float, 1> norms;     // (ntotal) precomputed ||v||^2
    size_t capacity = 0;             // allocated capacity in vectors

    std::unique_ptr<MetalDistance> distance;
    std::unique_ptr<MetalL2Norm> l2norm;
    std::unique_ptr<MetalSelect> selector;

    Impl(std::shared_ptr<MetalResources> res)
            : resources(std::move(res)),
              distance(std::make_unique<MetalDistance>(resources.get())),
              l2norm(std::make_unique<MetalL2Norm>(resources.get())),
              selector(std::make_unique<MetalSelect>(resources.get())) {}
};

MetalIndexFlat::MetalIndexFlat(
        std::shared_ptr<MetalResources> resources,
        int d,
        faiss::MetricType metric)
        : faiss::Index(d, metric),
          impl_(std::make_unique<Impl>(std::move(resources))) {
    is_trained = true;
}

MetalIndexFlat::~MetalIndexFlat() = default;

void MetalIndexFlat::add(faiss::idx_t n, const float* x) {
    if (n == 0) return;

    id<MTLDevice> device = impl_->resources->getDevice();
    id<MTLCommandQueue> queue = impl_->resources->getDefaultCommandQueue();
    size_t newTotal = ntotal + n;

    // Grow capacity if needed (2x growth strategy)
    if (newTotal > impl_->capacity) {
        size_t newCap = std::max(newTotal, impl_->capacity * 2);
        newCap = std::max(newCap, (size_t)1024); // minimum 1024 vectors

        MetalTensor<float, 2> newVectors(device, {newCap, (size_t)d});
        MetalTensor<float, 1> newNorms(device, {newCap});

        // Copy existing data
        if (ntotal > 0) {
            memcpy(newVectors.data(), impl_->vectors.data(),
                   ntotal * d * sizeof(float));
            memcpy(newNorms.data(), impl_->norms.data(),
                   ntotal * sizeof(float));
        }

        impl_->vectors = std::move(newVectors);
        impl_->norms = std::move(newNorms);
        impl_->capacity = newCap;
    }

    // Copy new vectors
    memcpy(impl_->vectors.data() + ntotal * d, x, n * d * sizeof(float));

    // Compute norms for new vectors (for L2)
    if (metric_type == faiss::METRIC_L2) {
        // Create a temporary buffer pointing to the new vectors region
        id<MTLBuffer> newVecBuf = [device
                newBufferWithBytesNoCopy:(void*)(impl_->vectors.data() + ntotal * d)
                                  length:n * d * sizeof(float)
                                 options:MTLResourceStorageModeShared
                             deallocator:nil];

        // If NoCopy failed (alignment), fall back
        if (!newVecBuf) {
            newVecBuf = [device newBufferWithBytes:(impl_->vectors.data() + ntotal * d)
                                            length:n * d * sizeof(float)
                                           options:MTLResourceStorageModeShared];
        }

        id<MTLBuffer> newNormBuf = [device
                newBufferWithBytesNoCopy:(void*)(impl_->norms.data() + ntotal)
                                  length:n * sizeof(float)
                                 options:MTLResourceStorageModeShared
                             deallocator:nil];
        if (!newNormBuf) {
            // Compute to temp buffer, then copy
            id<MTLBuffer> tmpNorms = [device newBufferWithLength:n * sizeof(float)
                                                         options:MTLResourceStorageModeShared];
            impl_->l2norm->compute(newVecBuf, tmpNorms, n, d, queue);
            memcpy(impl_->norms.data() + ntotal,
                   [tmpNorms contents], n * sizeof(float));
        } else {
            impl_->l2norm->compute(newVecBuf, newNormBuf, n, d, queue);
        }
    }

    ntotal = newTotal;
}

void MetalIndexFlat::search(
        faiss::idx_t n,
        const float* x,
        faiss::idx_t k,
        float* distances,
        faiss::idx_t* labels,
        const faiss::SearchParameters* params) const {

    if (n == 0 || ntotal == 0) {
        // Fill with sentinel values
        for (faiss::idx_t i = 0; i < n * k; i++) {
            distances[i] = (metric_type == faiss::METRIC_L2) ? INFINITY : -INFINITY;
            labels[i] = -1;
        }
        return;
    }

    FAISS_THROW_IF_NOT_MSG(
            k > 0 && k <= ntotal,
            "k must be > 0 and <= ntotal");

    id<MTLDevice> device = impl_->resources->getDevice();
    id<MTLCommandQueue> queue = impl_->resources->getDefaultCommandQueue();

    // Create query buffer (zero-copy if page-aligned, otherwise copy)
    id<MTLBuffer> queryBuf = [device
            newBufferWithBytes:x
                        length:n * d * sizeof(float)
                       options:MTLResourceStorageModeShared];

    // Vectors buffer: use underlying MTLBuffer, but only ntotal rows
    // We can create a view by using offset in the encoder, but for simplicity
    // we'll create a wrapper pointing to the right region
    id<MTLBuffer> vecBuf = impl_->vectors.buffer();

    // Norms buffer
    id<MTLBuffer> normBuf = impl_->norms.buffer();

    // Distance output: (n x ntotal) -- can be large!
    size_t distBytes = n * ntotal * sizeof(float);
    id<MTLBuffer> distBuf = [device newBufferWithLength:distBytes
                                                options:MTLResourceStorageModeShared];

    // Step 1: Compute distances
    impl_->distance->compute(
            queryBuf, vecBuf, normBuf, distBuf,
            n, ntotal, d, metric_type, queue);

    // Step 2: Top-k selection
    // Output buffers for top-k (int32_t indices from Metal)
    id<MTLBuffer> outDistBuf = [device newBufferWithLength:n * k * sizeof(float)
                                                    options:MTLResourceStorageModeShared];
    id<MTLBuffer> outIdxBuf = [device newBufferWithLength:n * k * sizeof(int32_t)
                                                   options:MTLResourceStorageModeShared];

    impl_->selector->select(
            distBuf, outDistBuf, outIdxBuf,
            n, ntotal, k, metric_type, queue);

    // Copy results back
    float* outDistPtr = (float*)[outDistBuf contents];
    int32_t* outIdxPtr = (int32_t*)[outIdxBuf contents];

    memcpy(distances, outDistPtr, n * k * sizeof(float));

    // Convert int32_t → idx_t (int64_t)
    for (faiss::idx_t i = 0; i < n * k; i++) {
        labels[i] = (faiss::idx_t)outIdxPtr[i];
    }
}

void MetalIndexFlat::reset() {
    ntotal = 0;
    impl_->capacity = 0;
    impl_->vectors = MetalTensor<float, 2>();
    impl_->norms = MetalTensor<float, 1>();
}

void MetalIndexFlat::reconstruct(faiss::idx_t key, float* recons) const {
    FAISS_THROW_IF_NOT_MSG(
            key >= 0 && key < ntotal,
            "reconstruct: key out of range");
    memcpy(recons, impl_->vectors.data() + key * d, d * sizeof(float));
}

const float* MetalIndexFlat::getVectorsData() const {
    return impl_->vectors.data();
}

// --- Conversion helpers ---

std::unique_ptr<MetalIndexFlat> index_cpu_to_metal(
        std::shared_ptr<MetalResources> resources,
        const faiss::IndexFlat* cpu_index) {

    auto metal_index = std::make_unique<MetalIndexFlat>(
            resources, cpu_index->d, cpu_index->metric_type);

    if (cpu_index->ntotal > 0) {
        const float* data = cpu_index->get_xb();
        metal_index->add(cpu_index->ntotal, data);
    }

    return metal_index;
}

std::unique_ptr<faiss::IndexFlat> index_metal_to_cpu(
        const MetalIndexFlat* metal_index) {

    auto cpu_index = std::make_unique<faiss::IndexFlat>(
            metal_index->d, metal_index->metric_type);

    if (metal_index->ntotal > 0) {
        cpu_index->add(metal_index->ntotal, metal_index->getVectorsData());
    }

    return cpu_index;
}

} // namespace faiss_metal
