#pragma once

#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
#include <faiss/MetricType.h>
#include <cstddef>
#include <memory>

namespace faiss_metal {

class MetalResources;
class MetalL2Norm;

/// Distance computation on Metal.
///
/// Path selection based on device generation:
///   M1 (Apple7): MPS GEMM + custom norm/broadcast shaders
///   M2+ (Apple8+): simdgroup_matrix GEMM (avoids MPS overhead, 2x FP16 throughput)
///   M3+ (Apple9+): same as M2 but with dynamic threadgroup memory caching
///
/// L2: dist = ||q||^2 + ||v||^2 - 2*q·v
/// IP: dist = Q * V^T
class MetalDistance {
   public:
    MetalDistance(MetalResources* resources);
    ~MetalDistance();

    /// Force MPS GEMM path even on M2+ hardware. For testing both paths.
    void setForceMPS(bool force) { forceMPS_ = force; }

    /// Tell distance kernels that the vectors buffer contains half (FP16) data.
    void setVectorsFloat16(bool f16) { vectorsFloat16_ = f16; }

    /// Encode distance computation into an existing command buffer.
    /// For L2, also needs a scratch buffer for query norms (queryNormsBuf).
    /// Caller is responsible for committing.
    void encode(
            id<MTLCommandBuffer> cmdBuf,
            id<MTLBuffer> queries,
            id<MTLBuffer> vectors,
            id<MTLBuffer> vecNorms,
            id<MTLBuffer> queryNormsBuf,
            id<MTLBuffer> distOutput,
            size_t nq,
            size_t nv,
            size_t d,
            faiss::MetricType metric);

    /// Convenience: create command buffer, encode, commit, wait.
    void compute(
            id<MTLBuffer> queries,
            id<MTLBuffer> vectors,
            id<MTLBuffer> vecNorms,
            id<MTLBuffer> distOutput,
            size_t nq,
            size_t nv,
            size_t d,
            faiss::MetricType metric,
            id<MTLCommandQueue> queue);

   private:
    MetalResources* resources_;
    bool useSimdGroupGemm_; // M2+: use custom GEMM instead of MPS
    bool forceMPS_ = false; // override: force MPS path for testing
    bool vectorsFloat16_ = false; // vectors stored as half
    std::unique_ptr<MetalL2Norm> l2norm_; // reused across calls

    id<MTLComputePipelineState> broadcastSumPipeline_;
    id<MTLComputePipelineState> simdgroupGemmPipeline_;
    id<MTLComputePipelineState> simdgroupGemmL2FusedPipeline_;
    id<MTLComputePipelineState> directL2Pipeline_;

    // FP16 storage variants (vectors stored as half)
    id<MTLComputePipelineState> simdgroupGemmF16StoragePipeline_;
    id<MTLComputePipelineState> simdgroupGemmL2FusedF16StoragePipeline_;
    id<MTLComputePipelineState> directL2F16StoragePipeline_;

    void encodeMPS(
            id<MTLCommandBuffer> cmdBuf,
            id<MTLBuffer> queries, id<MTLBuffer> vectors,
            id<MTLBuffer> vecNorms, id<MTLBuffer> queryNormsBuf,
            id<MTLBuffer> distOutput,
            size_t nq, size_t nv, size_t d,
            faiss::MetricType metric);

    void encodeSimdGroup(
            id<MTLCommandBuffer> cmdBuf,
            id<MTLBuffer> queries, id<MTLBuffer> vectors,
            id<MTLBuffer> vecNorms, id<MTLBuffer> queryNormsBuf,
            id<MTLBuffer> distOutput,
            size_t nq, size_t nv, size_t d,
            faiss::MetricType metric);
};

} // namespace faiss_metal
