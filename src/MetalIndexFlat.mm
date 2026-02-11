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
    bool useFloat16Storage = false;

    // FP32 storage (default)
    MetalTensor<float, 2> vectors;   // (ntotal x d) stored vectors

    // FP16 storage (when useFloat16Storage == true)
    id<MTLBuffer> vectorsF16Buf = nil;
    size_t vectorsF16Capacity = 0;   // capacity in number of vectors

    MetalTensor<float, 1> norms;     // (ntotal) precomputed ||v||^2, always FP32
    size_t capacity = 0;             // allocated capacity in vectors (FP32 mode)

    std::unique_ptr<MetalDistance> distance;
    std::unique_ptr<MetalL2Norm> l2norm;
    std::unique_ptr<MetalSelect> selector;

    // Scratch buffers reused across search() calls to avoid per-call allocation
    id<MTLBuffer> scratchDistBuf = nil;
    size_t scratchDistBytes = 0;
    id<MTLBuffer> scratchOutDistBuf = nil;
    id<MTLBuffer> scratchOutIdxBuf = nil;
    size_t scratchTopkElems = 0;
    id<MTLBuffer> scratchQueryNormBuf = nil;
    size_t scratchQueryNormElems = 0;

    id<MTLBuffer> getScratchDist(id<MTLDevice> device, size_t bytes) {
        if (bytes > scratchDistBytes) {
            scratchDistBytes = bytes;
            // Private: GPU-only intermediate (GEMM out → Select in), no CPU access needed
            scratchDistBuf = [device newBufferWithLength:bytes
                                                options:MTLResourceStorageModePrivate];
        }
        return scratchDistBuf;
    }

    void getScratchTopk(id<MTLDevice> device, size_t elems,
                        id<MTLBuffer>& outDist, id<MTLBuffer>& outIdx) {
        if (elems > scratchTopkElems) {
            scratchTopkElems = elems;
            scratchOutDistBuf = [device newBufferWithLength:elems * sizeof(float)
                                                    options:MTLResourceStorageModeShared];
            scratchOutIdxBuf = [device newBufferWithLength:elems * sizeof(int32_t)
                                                   options:MTLResourceStorageModeShared];
        }
        outDist = scratchOutDistBuf;
        outIdx = scratchOutIdxBuf;
    }

    id<MTLBuffer> getScratchQueryNorms(id<MTLDevice> device, size_t nq) {
        if (nq > scratchQueryNormElems) {
            scratchQueryNormElems = nq;
            // Private: GPU-only intermediate (l2_norm out → GEMM in), no CPU access needed
            scratchQueryNormBuf = [device newBufferWithLength:nq * sizeof(float)
                                                      options:MTLResourceStorageModePrivate];
        }
        return scratchQueryNormBuf;
    }

    id<MTLBuffer> getVectorsBuf() const {
        return useFloat16Storage ? vectorsF16Buf : vectors.buffer();
    }

    Impl(std::shared_ptr<MetalResources> res, bool f16)
            : resources(std::move(res)),
              useFloat16Storage(f16),
              distance(std::make_unique<MetalDistance>(resources.get())),
              l2norm(std::make_unique<MetalL2Norm>(resources.get())),
              selector(std::make_unique<MetalSelect>(resources.get())) {
        distance->setVectorsFloat16(f16);
    }
};

MetalIndexFlat::MetalIndexFlat(
        std::shared_ptr<MetalResources> resources,
        int d,
        faiss::MetricType metric,
        bool useFloat16Storage)
        : faiss::Index(d, metric),
          impl_(std::make_unique<Impl>(std::move(resources), useFloat16Storage)) {
    is_trained = true;
}

MetalIndexFlat::~MetalIndexFlat() = default;

void MetalIndexFlat::add(faiss::idx_t n, const float* x) {
    if (n == 0) return;

    id<MTLDevice> device = impl_->resources->getDevice();
    id<MTLCommandQueue> queue = impl_->resources->getDefaultCommandQueue();
    size_t newTotal = ntotal + n;

    if (impl_->useFloat16Storage) {
        // --- FP16 storage path ---
        size_t f16Cap = impl_->vectorsF16Capacity;
        if (newTotal > f16Cap) {
            size_t newCap = std::max(newTotal, f16Cap * 2);
            newCap = std::max(newCap, (size_t)1024);

            id<MTLBuffer> newBuf = [device newBufferWithLength:newCap * d * sizeof(uint16_t)
                                                       options:MTLResourceStorageModeShared];
            MetalTensor<float, 1> newNorms(device, {newCap});

            if (ntotal > 0) {
                memcpy([newBuf contents], [impl_->vectorsF16Buf contents],
                       ntotal * d * sizeof(uint16_t));
                memcpy(newNorms.data(), impl_->norms.data(),
                       ntotal * sizeof(float));
            }

            impl_->vectorsF16Buf = newBuf;
            impl_->vectorsF16Capacity = newCap;
            impl_->norms = std::move(newNorms);
        }

        // Convert float32 → float16 and store
        __fp16* dst = (__fp16*)[impl_->vectorsF16Buf contents] + ntotal * d;
        for (size_t i = 0; i < (size_t)n * d; i++) {
            dst[i] = (__fp16)x[i];
        }

        // Compute norms from float32 input (more accurate than from half)
        if (metric_type == faiss::METRIC_L2) {
            id<MTLBuffer> newVecBuf = [device newBufferWithBytes:x
                                                          length:n * d * sizeof(float)
                                                         options:MTLResourceStorageModeShared];
            id<MTLBuffer> tmpNorms = [device newBufferWithLength:n * sizeof(float)
                                                         options:MTLResourceStorageModeShared];
            impl_->l2norm->compute(newVecBuf, tmpNorms, n, d, queue);
            memcpy(impl_->norms.data() + ntotal,
                   [tmpNorms contents], n * sizeof(float));
        }
    } else {
        // --- FP32 storage path (original) ---
        if (newTotal > impl_->capacity) {
            size_t newCap = std::max(newTotal, impl_->capacity * 2);
            newCap = std::max(newCap, (size_t)1024);

            MetalTensor<float, 2> newVectors(device, {newCap, (size_t)d});
            MetalTensor<float, 1> newNorms(device, {newCap});

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

        memcpy(impl_->vectors.data() + ntotal * d, x, n * d * sizeof(float));

        if (metric_type == faiss::METRIC_L2) {
            id<MTLBuffer> newVecBuf = [device newBufferWithBytes:(impl_->vectors.data() + ntotal * d)
                                                          length:n * d * sizeof(float)
                                                         options:MTLResourceStorageModeShared];
            id<MTLBuffer> tmpNorms = [device newBufferWithLength:n * sizeof(float)
                                                         options:MTLResourceStorageModeShared];
            impl_->l2norm->compute(newVecBuf, tmpNorms, n, d, queue);
            memcpy(impl_->norms.data() + ntotal,
                   [tmpNorms contents], n * sizeof(float));
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

    FAISS_THROW_IF_NOT_MSG(k > 0, "k must be > 0");

    // Metal shaders use int32_t indices — cap at 2^31-1 vectors
    FAISS_THROW_IF_NOT_MSG(
            ntotal <= (faiss::idx_t)INT32_MAX,
            "MetalIndexFlat supports at most 2^31-1 vectors (int32 indices)");

    if (n == 0 || ntotal == 0) {
        for (faiss::idx_t i = 0; i < n * k; i++) {
            distances[i] = (metric_type == faiss::METRIC_L2) ? INFINITY : -INFINITY;
            labels[i] = -1;
        }
        return;
    }

    // Clamp k to ntotal (matches FAISS CPU behavior)
    faiss::idx_t effective_k = std::min(k, (faiss::idx_t)ntotal);

    id<MTLDevice> device = impl_->resources->getDevice();
    id<MTLCommandQueue> queue = impl_->resources->getDefaultCommandQueue();

    id<MTLBuffer> queryBuf = [device
            newBufferWithBytes:x
                        length:n * d * sizeof(float)
                       options:MTLResourceStorageModeShared];

    id<MTLBuffer> vecBuf = impl_->getVectorsBuf();
    id<MTLBuffer> normBuf = impl_->norms.buffer();

    // Reuse scratch buffers across calls
    size_t distBytes = n * ntotal * sizeof(float);
    id<MTLBuffer> distBuf = impl_->getScratchDist(device, distBytes);

    id<MTLBuffer> outDistBuf, outIdxBuf;
    impl_->getScratchTopk(device, n * effective_k, outDistBuf, outIdxBuf);

    // Single command buffer: distance + top-k in one GPU submission.
    // All k values use GPU-native selection (no MPS, no CPU fallback).
    id<MTLBuffer> queryNormsBuf = nil;
    if (metric_type == faiss::METRIC_L2) {
        queryNormsBuf = impl_->getScratchQueryNorms(device, n);
    }

    id<MTLCommandBuffer> cmdBuf = [queue commandBuffer];
    impl_->distance->encode(
            cmdBuf, queryBuf, vecBuf, normBuf, queryNormsBuf,
            distBuf, n, ntotal, d, metric_type);
    impl_->selector->encode(
            cmdBuf, distBuf, outDistBuf, outIdxBuf,
            nil, nil, nil, n, ntotal, effective_k, metric_type);
    [cmdBuf commit];
    [cmdBuf waitUntilCompleted];

    // Copy results back
    float* outDistPtr = (float*)[outDistBuf contents];
    int32_t* outIdxPtr = (int32_t*)[outIdxBuf contents];

    float sentinel_dist = (metric_type == faiss::METRIC_L2) ? INFINITY : -INFINITY;

    for (faiss::idx_t i = 0; i < n; i++) {
        // Copy effective_k results
        for (faiss::idx_t j = 0; j < effective_k; j++) {
            distances[i * k + j] = outDistPtr[i * effective_k + j];
            labels[i * k + j] = (faiss::idx_t)outIdxPtr[i * effective_k + j];
        }
        // Fill remaining slots if k > ntotal
        for (faiss::idx_t j = effective_k; j < k; j++) {
            distances[i * k + j] = sentinel_dist;
            labels[i * k + j] = -1;
        }
    }
}

void MetalIndexFlat::reset() {
    ntotal = 0;
    impl_->capacity = 0;
    impl_->vectors = MetalTensor<float, 2>();
    impl_->vectorsF16Buf = nil;
    impl_->vectorsF16Capacity = 0;
    impl_->norms = MetalTensor<float, 1>();
    impl_->scratchDistBuf = nil;
    impl_->scratchDistBytes = 0;
    impl_->scratchOutDistBuf = nil;
    impl_->scratchOutIdxBuf = nil;
    impl_->scratchTopkElems = 0;
    impl_->scratchQueryNormBuf = nil;
    impl_->scratchQueryNormElems = 0;
}

void MetalIndexFlat::reconstruct(faiss::idx_t key, float* recons) const {
    FAISS_THROW_IF_NOT_MSG(
            key >= 0 && key < ntotal,
            "reconstruct: key out of range");
    if (impl_->useFloat16Storage) {
        const __fp16* src = (const __fp16*)[impl_->vectorsF16Buf contents] + key * d;
        for (int i = 0; i < d; i++) {
            recons[i] = (float)src[i];
        }
    } else {
        memcpy(recons, impl_->vectors.data() + key * d, d * sizeof(float));
    }
}

const float* MetalIndexFlat::getVectorsData() const {
    if (impl_->useFloat16Storage) return nullptr;
    return impl_->vectors.data();
}

bool MetalIndexFlat::isFloat16Storage() const {
    return impl_->useFloat16Storage;
}

void MetalIndexFlat::setForceMPS(bool force) {
    impl_->distance->setForceMPS(force);
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
        if (metal_index->isFloat16Storage()) {
            // Reconstruct each vector from FP16 → FP32
            std::vector<float> tmp(metal_index->d);
            for (faiss::idx_t i = 0; i < metal_index->ntotal; i++) {
                metal_index->reconstruct(i, tmp.data());
                cpu_index->add(1, tmp.data());
            }
        } else {
            cpu_index->add(metal_index->ntotal, metal_index->getVectorsData());
        }
    }

    return cpu_index;
}

} // namespace faiss_metal
