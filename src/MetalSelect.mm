#import "MetalSelect.h"
#import <faiss-metal/MetalResources.h>
#import <faiss-metal/MetalDeviceCapabilities.h>
#include <faiss/impl/FaissAssert.h>
#include <algorithm>
#include <cstring>

namespace faiss_metal {

MetalSelect::MetalSelect(MetalResources* resources) : resources_(resources) {
    id<MTLLibrary> lib = resources->getMetalLibrary();
    NSError* error = nil;

    const auto& caps = resources->getCapabilities();
    hasDynamicThreadgroupMem_ = caps.hasDynamicThreadgroupMemory; // M3+

    id<MTLFunction> blockFn = [lib newFunctionWithName:@"block_select_min"];
    FAISS_THROW_IF_NOT_MSG(blockFn, "Metal function 'block_select_min' not found");
    blockSelectMinPipeline_ = [resources->getDevice()
            newComputePipelineStateWithFunction:blockFn
                                         error:&error];
    FAISS_THROW_IF_NOT_MSG(blockSelectMinPipeline_, "Failed to create block_select_min pipeline");

    id<MTLFunction> warpMinFn = [lib newFunctionWithName:@"warp_select_min"];
    FAISS_THROW_IF_NOT_MSG(warpMinFn, "Metal function 'warp_select_min' not found");
    warpSelectMinPipeline_ = [resources->getDevice()
            newComputePipelineStateWithFunction:warpMinFn
                                         error:&error];
    FAISS_THROW_IF_NOT_MSG(warpSelectMinPipeline_, "Failed to create warp_select_min pipeline");

    id<MTLFunction> warpMaxFn = [lib newFunctionWithName:@"warp_select_max"];
    FAISS_THROW_IF_NOT_MSG(warpMaxFn, "Metal function 'warp_select_max' not found");
    warpSelectMaxPipeline_ = [resources->getDevice()
            newComputePipelineStateWithFunction:warpMaxFn
                                         error:&error];
    FAISS_THROW_IF_NOT_MSG(warpSelectMaxPipeline_, "Failed to create warp_select_max pipeline");
}

void MetalSelect::select(
        id<MTLBuffer> distances,
        id<MTLBuffer> outDistances,
        id<MTLBuffer> outIndices,
        size_t nq,
        size_t nv,
        size_t k,
        faiss::MetricType metric,
        id<MTLCommandQueue> queue) {

    if (nq == 0 || nv == 0 || k == 0) return;

    bool wantMin = (metric == faiss::METRIC_L2);
    uint32_t nv32 = (uint32_t)nv;
    uint32_t k32 = (uint32_t)k;

    // Use MPS for small k where available
    if (k <= 16) {
        // MPSMatrixFindTopK finds largest; for L2 we need smallest.
        // Strategy: negate for L2, use TopK, negate back.
        id<MTLDevice> device = resources_->getDevice();

        // For L2, negate distances first
        id<MTLBuffer> workDistances = distances;
        if (wantMin) {
            // Negate in-place is risky for the caller's buffer, so copy+negate
            size_t distBytes = nq * nv * sizeof(float);
            workDistances = [device newBufferWithLength:distBytes
                                                options:MTLResourceStorageModeShared];
            float* src = (float*)[distances contents];
            float* dst = (float*)[workDistances contents];
            for (size_t i = 0; i < nq * nv; i++) {
                dst[i] = -src[i];
            }
        }

        // Use MPS TopK
        MPSMatrixDescriptor* distDesc = [MPSMatrixDescriptor
                matrixDescriptorWithRows:nq
                                 columns:nv
                                rowBytes:nv * sizeof(float)
                                dataType:MPSDataTypeFloat32];

        MPSMatrixDescriptor* outDistDesc = [MPSMatrixDescriptor
                matrixDescriptorWithRows:nq
                                 columns:k
                                rowBytes:k * sizeof(float)
                                dataType:MPSDataTypeFloat32];

        MPSMatrixDescriptor* outIdxDesc = [MPSMatrixDescriptor
                matrixDescriptorWithRows:nq
                                 columns:k
                                rowBytes:k * sizeof(float) // MPSMatrixFindTopK uses float for indices
                                dataType:MPSDataTypeFloat32];

        MPSMatrix* distMat = [[MPSMatrix alloc] initWithBuffer:workDistances descriptor:distDesc];

        // MPS outputs indices as float, we need a temp buffer
        id<MTLBuffer> mpsOutDist = [device newBufferWithLength:nq * k * sizeof(float)
                                                        options:MTLResourceStorageModeShared];
        id<MTLBuffer> mpsOutIdx = [device newBufferWithLength:nq * k * sizeof(float)
                                                       options:MTLResourceStorageModeShared];

        MPSMatrix* outDistMat = [[MPSMatrix alloc] initWithBuffer:mpsOutDist descriptor:outDistDesc];
        MPSMatrix* outIdxMat = [[MPSMatrix alloc] initWithBuffer:mpsOutIdx descriptor:outIdxDesc];

        MPSMatrixFindTopK* topk = [[MPSMatrixFindTopK alloc]
                initWithDevice:device numberOfTopKValues:k];

        id<MTLCommandBuffer> cmdBuf = [queue commandBuffer];
        [topk encodeToCommandBuffer:cmdBuf
                       inputMatrix:distMat
                      resultMatrix:outDistMat
                resultIndexMatrix:outIdxMat];
        [cmdBuf commit];
        [cmdBuf waitUntilCompleted];

        // Copy results, converting indices from float to int32_t
        float* mpsDistPtr = (float*)[mpsOutDist contents];
        float* mpsIdxPtr = (float*)[mpsOutIdx contents];
        float* outDistPtr = (float*)[outDistances contents];
        int32_t* outIdxPtr = (int32_t*)[outIndices contents];

        for (size_t i = 0; i < nq * k; i++) {
            outDistPtr[i] = wantMin ? -mpsDistPtr[i] : mpsDistPtr[i];
            outIdxPtr[i] = (int32_t)mpsIdxPtr[i];
        }

        return;
    }

    // Custom shader path for larger k
    if (k <= 32) {
        // SIMD-group level selection
        auto pipeline = wantMin ? warpSelectMinPipeline_ : warpSelectMaxPipeline_;

        id<MTLCommandBuffer> cmdBuf = [queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [cmdBuf computeCommandEncoder];
        [encoder setComputePipelineState:pipeline];
        [encoder setBuffer:distances offset:0 atIndex:0];
        [encoder setBuffer:outDistances offset:0 atIndex:1];
        [encoder setBuffer:outIndices offset:0 atIndex:2];
        [encoder setBytes:&nv32 length:sizeof(nv32) atIndex:3];
        [encoder setBytes:&k32 length:sizeof(k32) atIndex:4];
        [encoder dispatchThreadgroups:MTLSizeMake(nq, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
        [encoder endEncoding];
        [cmdBuf commit];
        [cmdBuf waitUntilCompleted];
    } else {
        // Block-level selection for large k
        // Currently only min (L2) implemented in shader; for IP, negate + use min
        id<MTLBuffer> workDistances = distances;
        id<MTLDevice> device = resources_->getDevice();

        if (!wantMin) {
            size_t distBytes = nq * nv * sizeof(float);
            workDistances = [device newBufferWithLength:distBytes
                                                options:MTLResourceStorageModeShared];
            float* src = (float*)[distances contents];
            float* dst = (float*)[workDistances contents];
            for (size_t i = 0; i < nq * nv; i++) {
                dst[i] = -src[i];
            }
        }

        id<MTLCommandBuffer> cmdBuf = [queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [cmdBuf computeCommandEncoder];
        [encoder setComputePipelineState:blockSelectMinPipeline_];
        [encoder setBuffer:workDistances offset:0 atIndex:0];
        [encoder setBuffer:outDistances offset:0 atIndex:1];
        [encoder setBuffer:outIndices offset:0 atIndex:2];
        [encoder setBytes:&nv32 length:sizeof(nv32) atIndex:3];
        [encoder setBytes:&k32 length:sizeof(k32) atIndex:4];
        // M3+ dynamic threadgroup memory caching allows higher occupancy at 512 threads.
        // M1/M2: 256 threads (fits within 32KB threadgroup memory comfortably).
        size_t blockThreads = hasDynamicThreadgroupMem_ ? 256 : 256;
        // Note: keeping 256 for now since the shader's shared memory is sized for 256.
        // TODO: add a 512-thread variant of block_select for M3+.

        [encoder dispatchThreadgroups:MTLSizeMake(nq, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(blockThreads, 1, 1)];
        [encoder endEncoding];
        [cmdBuf commit];
        [cmdBuf waitUntilCompleted];

        // Negate back for IP
        if (!wantMin) {
            float* outDistPtr = (float*)[outDistances contents];
            for (size_t i = 0; i < nq * k; i++) {
                outDistPtr[i] = -outDistPtr[i];
            }
        }
    }
}

} // namespace faiss_metal
