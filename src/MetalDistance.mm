#import "MetalDistance.h"
#import "MetalL2Norm.h"
#import <faiss-metal/MetalResources.h>
#import <faiss-metal/MetalDeviceCapabilities.h>
#include <faiss/impl/FaissAssert.h>
#include <algorithm>

namespace faiss_metal {

// Must match struct GemmParams in shaders/simdgroup_gemm.metal
struct GemmParams {
    uint32_t M, N, K;
    float alpha, beta;
};

MetalDistance::MetalDistance(MetalResources* resources)
        : resources_(resources),
          l2norm_(std::make_unique<MetalL2Norm>(resources)) {
    id<MTLLibrary> lib = resources->getMetalLibrary();
    id<MTLDevice> device = resources->getDevice();
    NSError* error = nil;

    const auto& caps = resources->getCapabilities();
    useSimdGroupGemm_ = caps.hasFastFP16; // M2+ gets simdgroup_matrix path

    // Always build broadcast_sum (needed for L2 in all paths)
    id<MTLFunction> bsFn = [lib newFunctionWithName:@"broadcast_sum_l2"];
    FAISS_THROW_IF_NOT_MSG(bsFn, "Metal function 'broadcast_sum_l2' not found");
    broadcastSumPipeline_ = [device newComputePipelineStateWithFunction:bsFn error:&error];
    FAISS_THROW_IF_NOT_MSG(broadcastSumPipeline_, "Failed to create broadcast_sum pipeline");

    if (useSimdGroupGemm_) {
        // simdgroup_matrix GEMM: float32 input → half compute → float32 output
        id<MTLFunction> gemmFn = [lib newFunctionWithName:@"simdgroup_gemm_f32_via_f16"];
        FAISS_THROW_IF_NOT_MSG(gemmFn, "Metal function 'simdgroup_gemm_f32_via_f16' not found");
        simdgroupGemmPipeline_ = [device newComputePipelineStateWithFunction:gemmFn error:&error];
        FAISS_THROW_IF_NOT_MSG(simdgroupGemmPipeline_, "Failed to create simdgroup_gemm pipeline");

        // Direct L2 for small nv (fused, avoids GEMM + broadcast)
        id<MTLFunction> directFn = [lib newFunctionWithName:@"l2_distance_direct_f16"];
        FAISS_THROW_IF_NOT_MSG(directFn, "Metal function 'l2_distance_direct_f16' not found");
        directL2Pipeline_ = [device newComputePipelineStateWithFunction:directFn error:&error];
        FAISS_THROW_IF_NOT_MSG(directL2Pipeline_, "Failed to create direct L2 pipeline");
    }
}

MetalDistance::~MetalDistance() = default;

void MetalDistance::compute(
        id<MTLBuffer> queries,
        id<MTLBuffer> vectors,
        id<MTLBuffer> vecNorms,
        id<MTLBuffer> distOutput,
        size_t nq,
        size_t nv,
        size_t d,
        faiss::MetricType metric,
        id<MTLCommandQueue> queue) {

    if (nq == 0 || nv == 0) return;

    if (useSimdGroupGemm_) {
        computeSimdGroup(queries, vectors, vecNorms, distOutput,
                         nq, nv, d, metric, queue);
    } else {
        computeMPS(queries, vectors, vecNorms, distOutput,
                   nq, nv, d, metric, queue);
    }
}

// --- M1 path: MPS GEMM ---

void MetalDistance::computeMPS(
        id<MTLBuffer> queries,
        id<MTLBuffer> vectors,
        id<MTLBuffer> vecNorms,
        id<MTLBuffer> distOutput,
        size_t nq,
        size_t nv,
        size_t d,
        faiss::MetricType metric,
        id<MTLCommandQueue> queue) {

    id<MTLDevice> device = resources_->getDevice();

    MPSMatrixDescriptor* queryDesc = [MPSMatrixDescriptor
            matrixDescriptorWithRows:nq columns:d
                            rowBytes:d * sizeof(float) dataType:MPSDataTypeFloat32];
    MPSMatrixDescriptor* vecDesc = [MPSMatrixDescriptor
            matrixDescriptorWithRows:nv columns:d
                            rowBytes:d * sizeof(float) dataType:MPSDataTypeFloat32];
    MPSMatrixDescriptor* outDesc = [MPSMatrixDescriptor
            matrixDescriptorWithRows:nq columns:nv
                            rowBytes:nv * sizeof(float) dataType:MPSDataTypeFloat32];

    MPSMatrix* queryMat = [[MPSMatrix alloc] initWithBuffer:queries descriptor:queryDesc];
    MPSMatrix* vecMat = [[MPSMatrix alloc] initWithBuffer:vectors descriptor:vecDesc];
    MPSMatrix* outMat = [[MPSMatrix alloc] initWithBuffer:distOutput descriptor:outDesc];

    if (metric == faiss::METRIC_INNER_PRODUCT) {
        MPSMatrixMultiplication* gemm = [[MPSMatrixMultiplication alloc]
                initWithDevice:device transposeLeft:false transposeRight:true
                    resultRows:nq resultColumns:nv interiorColumns:d
                         alpha:1.0 beta:0.0];

        id<MTLCommandBuffer> cmdBuf = [queue commandBuffer];
        [gemm encodeToCommandBuffer:cmdBuf leftMatrix:queryMat
                        rightMatrix:vecMat resultMatrix:outMat];
        [cmdBuf commit];
        [cmdBuf waitUntilCompleted];
    } else {
        // L2: -2*Q*V^T, then add norms
        id<MTLBuffer> queryNorms = [device newBufferWithLength:nq * sizeof(float)
                                                       options:MTLResourceStorageModeShared];
        l2norm_->compute(queries, queryNorms, nq, d, queue);

        MPSMatrixMultiplication* gemm = [[MPSMatrixMultiplication alloc]
                initWithDevice:device transposeLeft:false transposeRight:true
                    resultRows:nq resultColumns:nv interiorColumns:d
                         alpha:-2.0 beta:0.0];

        id<MTLCommandBuffer> cmdBuf = [queue commandBuffer];
        [gemm encodeToCommandBuffer:cmdBuf leftMatrix:queryMat
                        rightMatrix:vecMat resultMatrix:outMat];

        // Encode broadcast_sum into same command buffer
        id<MTLComputeCommandEncoder> enc = [cmdBuf computeCommandEncoder];
        uint32_t nv32 = (uint32_t)nv;
        [enc setComputePipelineState:broadcastSumPipeline_];
        [enc setBuffer:distOutput offset:0 atIndex:0];
        [enc setBuffer:queryNorms offset:0 atIndex:1];
        [enc setBuffer:vecNorms offset:0 atIndex:2];
        [enc setBytes:&nv32 length:sizeof(nv32) atIndex:3];
        MTLSize gridSize = MTLSizeMake(nv, nq, 1);
        MTLSize groupSize = MTLSizeMake(std::min((size_t)256, nv),
                                        std::min((size_t)4, nq), 1);
        [enc dispatchThreads:gridSize threadsPerThreadgroup:groupSize];
        [enc endEncoding];

        [cmdBuf commit];
        [cmdBuf waitUntilCompleted];
    }
}

// --- M2+ path: simdgroup_matrix GEMM ---

void MetalDistance::computeSimdGroup(
        id<MTLBuffer> queries,
        id<MTLBuffer> vectors,
        id<MTLBuffer> vecNorms,
        id<MTLBuffer> distOutput,
        size_t nq,
        size_t nv,
        size_t d,
        faiss::MetricType metric,
        id<MTLCommandQueue> queue) {

    id<MTLDevice> device = resources_->getDevice();

    // For small nv with L2, use fused direct distance (avoids GEMM overhead)
    if (metric == faiss::METRIC_L2 && nv <= 256) {
        uint32_t d32 = (uint32_t)d;
        uint32_t nv32 = (uint32_t)nv;
        id<MTLCommandBuffer> cmdBuf = [queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cmdBuf computeCommandEncoder];
        [enc setComputePipelineState:directL2Pipeline_];
        [enc setBuffer:queries offset:0 atIndex:0];
        [enc setBuffer:vectors offset:0 atIndex:1];
        [enc setBuffer:distOutput offset:0 atIndex:2];
        [enc setBytes:&d32 length:sizeof(d32) atIndex:3];
        [enc setBytes:&nv32 length:sizeof(nv32) atIndex:4];
        // Grid: (nv, nq), threadgroup: (32, 1) -- one SIMD group per (query, vector) pair
        [enc dispatchThreadgroups:MTLSizeMake(nv, nq, 1)
            threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
        [enc endEncoding];
        [cmdBuf commit];
        [cmdBuf waitUntilCompleted];
        return;
    }

    // simdgroup_matrix GEMM for Q * V^T
    float alpha = (metric == faiss::METRIC_L2) ? -2.0f : 1.0f;
    GemmParams params{(uint32_t)nq, (uint32_t)nv, (uint32_t)d, alpha, 0.0f};

    size_t gridX = (nv + 31) / 32;
    size_t gridY = (nq + 31) / 32;

    id<MTLCommandBuffer> cmdBuf = [queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cmdBuf computeCommandEncoder];
    [enc setComputePipelineState:simdgroupGemmPipeline_];
    [enc setBuffer:queries offset:0 atIndex:0];
    [enc setBuffer:vectors offset:0 atIndex:1];
    [enc setBuffer:distOutput offset:0 atIndex:2];
    [enc setBytes:&params length:sizeof(params) atIndex:3];
    [enc dispatchThreadgroups:MTLSizeMake(gridX, gridY, 1)
        threadsPerThreadgroup:MTLSizeMake(32, 4, 1)];
    [enc endEncoding];

    if (metric == faiss::METRIC_L2) {
        // Add norms: dist[i][j] += ||q_i||^2 + ||v_j||^2
        id<MTLBuffer> queryNorms = [device newBufferWithLength:nq * sizeof(float)
                                                       options:MTLResourceStorageModeShared];
        // Norm computation needs its own command buffer (separate encode)
        l2norm_->compute(queries, queryNorms, nq, d, queue);

        uint32_t nv32 = (uint32_t)nv;
        // Encode broadcast_sum - must wait for GEMM first
        [cmdBuf commit];
        [cmdBuf waitUntilCompleted];

        id<MTLCommandBuffer> cmdBuf2 = [queue commandBuffer];
        id<MTLComputeCommandEncoder> enc2 = [cmdBuf2 computeCommandEncoder];
        [enc2 setComputePipelineState:broadcastSumPipeline_];
        [enc2 setBuffer:distOutput offset:0 atIndex:0];
        [enc2 setBuffer:queryNorms offset:0 atIndex:1];
        [enc2 setBuffer:vecNorms offset:0 atIndex:2];
        [enc2 setBytes:&nv32 length:sizeof(nv32) atIndex:3];
        MTLSize gridSize = MTLSizeMake(nv, nq, 1);
        MTLSize groupSize = MTLSizeMake(std::min((size_t)256, nv),
                                        std::min((size_t)4, nq), 1);
        [enc2 dispatchThreads:gridSize threadsPerThreadgroup:groupSize];
        [enc2 endEncoding];
        [cmdBuf2 commit];
        [cmdBuf2 waitUntilCompleted];
    } else {
        [cmdBuf commit];
        [cmdBuf waitUntilCompleted];
    }
}

} // namespace faiss_metal
