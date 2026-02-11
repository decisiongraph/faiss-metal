#import "MetalSelect.h"
#import <faiss-metal/MetalResources.h>
#import <faiss-metal/MetalDeviceCapabilities.h>
#include <faiss/impl/FaissAssert.h>
#include <algorithm>
#include <cstring>

namespace faiss_metal {

MetalSelect::MetalSelect(MetalResources* resources) : resources_(resources) {
    id<MTLLibrary> lib = resources->getMetalLibrary();
    id<MTLDevice> device = resources->getDevice();
    NSError* error = nil;

    auto makePipeline = [&](NSString* name) -> id<MTLComputePipelineState> {
        id<MTLFunction> fn = [lib newFunctionWithName:name];
        FAISS_THROW_IF_NOT_FMT(fn, "Metal function '%s' not found",
                               [name UTF8String]);
        id<MTLComputePipelineState> pso =
                [device newComputePipelineStateWithFunction:fn error:&error];
        FAISS_THROW_IF_NOT_FMT(pso, "Failed to create pipeline '%s'",
                               [name UTF8String]);
        return pso;
    };

    blockSelectMinPipeline_ = makePipeline(@"block_select_min");
    blockSelectMaxPipeline_ = makePipeline(@"block_select_max");
    warpSelectMinPipeline_ = makePipeline(@"warp_select_min");
    warpSelectMaxPipeline_ = makePipeline(@"warp_select_max");
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

    // MPS path for small k
    if (k <= 16) {
        id<MTLDevice> device = resources_->getDevice();

        // MPSMatrixFindTopK always finds largest values.
        // For L2 (want smallest), negate → find largest → negate back.
        id<MTLBuffer> workDistances = distances;
        if (wantMin) {
            size_t distBytes = nq * nv * sizeof(float);
            workDistances = [device newBufferWithLength:distBytes
                                                options:MTLResourceStorageModeShared];
            float* src = (float*)[distances contents];
            float* dst = (float*)[workDistances contents];
            for (size_t i = 0; i < nq * nv; i++) {
                dst[i] = -src[i];
            }
        }

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
                                rowBytes:k * sizeof(float)
                                dataType:MPSDataTypeFloat32];

        MPSMatrix* distMat = [[MPSMatrix alloc] initWithBuffer:workDistances descriptor:distDesc];

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

        // Convert: MPS uses float indices → int32_t, and un-negate for L2
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

    // Custom shader paths
    id<MTLComputePipelineState> pipeline;
    size_t blockThreads;

    if (k <= 32) {
        pipeline = wantMin ? warpSelectMinPipeline_ : warpSelectMaxPipeline_;
        blockThreads = 32;
    } else {
        pipeline = wantMin ? blockSelectMinPipeline_ : blockSelectMaxPipeline_;
        blockThreads = 256;
    }

    id<MTLCommandBuffer> cmdBuf = [queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [cmdBuf computeCommandEncoder];
    [encoder setComputePipelineState:pipeline];
    [encoder setBuffer:distances offset:0 atIndex:0];
    [encoder setBuffer:outDistances offset:0 atIndex:1];
    [encoder setBuffer:outIndices offset:0 atIndex:2];
    [encoder setBytes:&nv32 length:sizeof(nv32) atIndex:3];
    [encoder setBytes:&k32 length:sizeof(k32) atIndex:4];
    [encoder dispatchThreadgroups:MTLSizeMake(nq, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(blockThreads, 1, 1)];
    [encoder endEncoding];
    [cmdBuf commit];
    [cmdBuf waitUntilCompleted];
}

} // namespace faiss_metal
