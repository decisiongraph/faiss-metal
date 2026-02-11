#pragma once

#import <Metal/Metal.h>
#include <cstddef>

namespace faiss_metal {

class MetalResources;

/// Dispatch L2 norm computation on Metal GPU.
/// Computes ||v||^2 for each row of an (n x d) matrix.
/// Automatically selects FP32 or FP16 fast path based on device capabilities.
class MetalL2Norm {
   public:
    MetalL2Norm(MetalResources* resources);

    /// Compute squared L2 norms.
    /// @param input   MTLBuffer containing (n x d) float matrix
    /// @param output  MTLBuffer for n floats (preallocated)
    /// @param n       Number of rows
    /// @param d       Dimension
    /// @param queue   Command queue to use
    void compute(
            id<MTLBuffer> input,
            id<MTLBuffer> output,
            size_t n,
            size_t d,
            id<MTLCommandQueue> queue);

   private:
    bool useFP16_;

    // FP32 pipelines (M1 baseline)
    id<MTLComputePipelineState> pipeline_;
    id<MTLComputePipelineState> pipelineLarge_;

    // FP16 pipelines (M2+ fast path)
    id<MTLComputePipelineState> pipelineF16_;
    id<MTLComputePipelineState> pipelineF16Large_;
};

} // namespace faiss_metal
