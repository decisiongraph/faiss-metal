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

    /// Compute all-pairs distances between queries and database vectors.
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
    std::unique_ptr<MetalL2Norm> l2norm_; // reused across calls

    id<MTLComputePipelineState> broadcastSumPipeline_;
    id<MTLComputePipelineState> simdgroupGemmPipeline_; // simdgroup_gemm_f32_via_f16
    id<MTLComputePipelineState> directL2Pipeline_;      // l2_distance_direct_f16

    void computeMPS(
            id<MTLBuffer> queries, id<MTLBuffer> vectors,
            id<MTLBuffer> vecNorms, id<MTLBuffer> distOutput,
            size_t nq, size_t nv, size_t d,
            faiss::MetricType metric, id<MTLCommandQueue> queue);

    void computeSimdGroup(
            id<MTLBuffer> queries, id<MTLBuffer> vectors,
            id<MTLBuffer> vecNorms, id<MTLBuffer> distOutput,
            size_t nq, size_t nv, size_t d,
            faiss::MetricType metric, id<MTLCommandQueue> queue);
};

} // namespace faiss_metal
