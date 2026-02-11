#import <Foundation/Foundation.h>
#import <faiss-metal/StandardMetalResources.h>
#import <faiss-metal/MetalDeviceCapabilities.h>

#include <cassert>
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

// Internal headers for direct testing
#include "../src/MetalL2Norm.h"
#include "../src/MetalDistance.h"

using namespace faiss_metal;

static void test_l2_norm() {
    printf("test_l2_norm... ");

    auto res = std::make_shared<StandardMetalResources>();
    MetalL2Norm norm(res.get());

    const size_t n = 100;
    const size_t d = 128;

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    // Generate random vectors
    std::vector<float> data(n * d);
    for (auto& v : data) v = dist(rng);

    // Compute on Metal
    id<MTLDevice> device = res->getDevice();
    id<MTLBuffer> inputBuf = [device newBufferWithBytes:data.data()
                                                length:n * d * sizeof(float)
                                               options:MTLResourceStorageModeShared];
    id<MTLBuffer> outputBuf = [device newBufferWithLength:n * sizeof(float)
                                                  options:MTLResourceStorageModeShared];

    norm.compute(inputBuf, outputBuf, n, d, res->getDefaultCommandQueue());

    float* metalNorms = (float*)[outputBuf contents];

    // Compute reference on CPU
    for (size_t i = 0; i < n; i++) {
        float cpuNorm = 0.0f;
        for (size_t j = 0; j < d; j++) {
            float v = data[i * d + j];
            cpuNorm += v * v;
        }
        float diff = std::abs(metalNorms[i] - cpuNorm);
        assert(diff < 1e-3f && "L2 norm mismatch");
    }

    printf("PASS\n");
}

static void test_l2_norm_large_dim() {
    printf("test_l2_norm_large_dim... ");

    auto res = std::make_shared<StandardMetalResources>();
    MetalL2Norm norm(res.get());

    const size_t n = 10;
    const size_t d = 1536; // large dimension

    std::mt19937 rng(123);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    std::vector<float> data(n * d);
    for (auto& v : data) v = dist(rng);

    id<MTLDevice> device = res->getDevice();
    id<MTLBuffer> inputBuf = [device newBufferWithBytes:data.data()
                                                length:n * d * sizeof(float)
                                               options:MTLResourceStorageModeShared];
    id<MTLBuffer> outputBuf = [device newBufferWithLength:n * sizeof(float)
                                                  options:MTLResourceStorageModeShared];

    norm.compute(inputBuf, outputBuf, n, d, res->getDefaultCommandQueue());

    float* metalNorms = (float*)[outputBuf contents];

    for (size_t i = 0; i < n; i++) {
        float cpuNorm = 0.0f;
        for (size_t j = 0; j < d; j++) {
            float v = data[i * d + j];
            cpuNorm += v * v;
        }
        float relDiff = std::abs(metalNorms[i] - cpuNorm) / std::max(cpuNorm, 1e-6f);
        assert(relDiff < 1e-4f && "L2 norm (large dim) mismatch");
    }

    printf("PASS\n");
}

static void test_l2_distance() {
    printf("test_l2_distance... ");

    auto res = std::make_shared<StandardMetalResources>();
    MetalDistance distCalc(res.get());
    MetalL2Norm normCalc(res.get());

    const size_t nq = 5;
    const size_t nv = 50;
    const size_t d = 64;

    std::mt19937 rng(99);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    std::vector<float> queries(nq * d);
    std::vector<float> vectors(nv * d);
    for (auto& v : queries) v = dist(rng);
    for (auto& v : vectors) v = dist(rng);

    id<MTLDevice> device = res->getDevice();
    id<MTLCommandQueue> queue = res->getDefaultCommandQueue();

    id<MTLBuffer> queryBuf = [device newBufferWithBytes:queries.data()
                                                length:nq * d * sizeof(float)
                                               options:MTLResourceStorageModeShared];
    id<MTLBuffer> vecBuf = [device newBufferWithBytes:vectors.data()
                                              length:nv * d * sizeof(float)
                                             options:MTLResourceStorageModeShared];
    id<MTLBuffer> normBuf = [device newBufferWithLength:nv * sizeof(float)
                                                options:MTLResourceStorageModeShared];
    id<MTLBuffer> distBuf = [device newBufferWithLength:nq * nv * sizeof(float)
                                                options:MTLResourceStorageModeShared];

    // Precompute vector norms
    normCalc.compute(vecBuf, normBuf, nv, d, queue);

    // Compute L2 distances
    distCalc.compute(queryBuf, vecBuf, normBuf, distBuf,
                     nq, nv, d, faiss::METRIC_L2, queue);

    float* metalDist = (float*)[distBuf contents];

    // Reference CPU computation
    for (size_t i = 0; i < nq; i++) {
        for (size_t j = 0; j < nv; j++) {
            float cpuDist = 0.0f;
            for (size_t k = 0; k < d; k++) {
                float diff = queries[i * d + k] - vectors[j * d + k];
                cpuDist += diff * diff;
            }
            float absDiff = std::abs(metalDist[i * nv + j] - cpuDist);
            float relDiff = absDiff / std::max(cpuDist, 1e-6f);
            // FP16 GEMM path (M2+) has ~1e-2 relative error; FP32 MPS path ~1e-4
            assert(relDiff < 5e-2f && "L2 distance mismatch");
        }
    }

    printf("PASS\n");
}

static void test_ip_distance() {
    printf("test_ip_distance... ");

    auto res = std::make_shared<StandardMetalResources>();
    MetalDistance distCalc(res.get());

    const size_t nq = 5;
    const size_t nv = 50;
    const size_t d = 64;

    std::mt19937 rng(77);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    std::vector<float> queries(nq * d);
    std::vector<float> vectors(nv * d);
    for (auto& v : queries) v = dist(rng);
    for (auto& v : vectors) v = dist(rng);

    id<MTLDevice> device = res->getDevice();

    id<MTLBuffer> queryBuf = [device newBufferWithBytes:queries.data()
                                                length:nq * d * sizeof(float)
                                               options:MTLResourceStorageModeShared];
    id<MTLBuffer> vecBuf = [device newBufferWithBytes:vectors.data()
                                              length:nv * d * sizeof(float)
                                             options:MTLResourceStorageModeShared];
    id<MTLBuffer> distBuf = [device newBufferWithLength:nq * nv * sizeof(float)
                                                options:MTLResourceStorageModeShared];

    distCalc.compute(queryBuf, vecBuf, nil, distBuf,
                     nq, nv, d, faiss::METRIC_INNER_PRODUCT,
                     res->getDefaultCommandQueue());

    float* metalDist = (float*)[distBuf contents];

    for (size_t i = 0; i < nq; i++) {
        for (size_t j = 0; j < nv; j++) {
            float cpuDist = 0.0f;
            for (size_t k = 0; k < d; k++) {
                cpuDist += queries[i * d + k] * vectors[j * d + k];
            }
            float absDiff = std::abs(metalDist[i * nv + j] - cpuDist);
            // FP16 GEMM path (M2+) has larger absolute error
            assert(absDiff < 5e-1f && "IP distance mismatch");
        }
    }

    printf("PASS\n");
}

int main() {
    @autoreleasepool {
        // Print device info
        auto res = std::make_shared<StandardMetalResources>();
        const auto& caps = res->getCapabilities();
        printf("=== Metal Distance Tests ===\n");
        printf("%s\n", faiss_metal::describeCapabilities(caps).c_str());

        test_l2_norm();
        test_l2_norm_large_dim();
        test_l2_distance();
        test_ip_distance();
        printf("All distance tests passed!\n");
    }
    return 0;
}
