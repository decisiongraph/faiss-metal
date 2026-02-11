# faiss-metal

Metal GPU backend for [FAISS](https://github.com/facebookresearch/faiss) vector similarity search on Apple Silicon.

FAISS only supports NVIDIA CUDA for GPU acceleration. faiss-metal fills this gap for macOS by implementing brute-force index operations on Apple Silicon GPUs via Metal compute shaders. It subclasses `faiss::Index` directly so it works as a drop-in replacement.

**Key design advantage:** Apple Silicon's unified memory means zero CPU-GPU transfer overhead. Vectors stored in MTLBuffers are accessible by both CPU and GPU without copies.

## Features

- `MetalIndexFlat` with L2 and inner product metrics
- Automatic code path selection per Apple Silicon generation (M1/M2/M3/M4)
- CPU-to-Metal and Metal-to-CPU index conversion
- Scratch buffer reuse across search calls
- Custom SIMD-group GEMM, top-k selection, and L2 norm shaders

## Hardware support

| Generation | Metal Family | GEMM path | Top-k selection | Block select threads |
|:---|:---|:---|:---|:---|
| M1 | Apple7 | MPS MatrixMultiplication (FP32) | MPS/warp/block | 256 |
| M2 | Apple8 | simdgroup_matrix GEMM (FP16, 2x throughput) | MPS/warp/block | 256 |
| M3 | Apple9 | simdgroup_matrix GEMM (FP16) | MPS/warp/block | 512 (dynamic threadgroup mem) |
| M4 | Apple9+ | simdgroup_matrix GEMM (FP16) | MPS/warp/block | 512 (dynamic threadgroup mem) |

Top-k routing: `k <= 16` uses MPSMatrixFindTopK, `k <= 32` uses warp_select (SIMD-group), `k > 32` uses block_select (threadgroup).

## Usage

```cpp
#include <faiss-metal/faiss_metal.h>

auto res = std::make_shared<faiss_metal::StandardMetalResources>();
faiss_metal::MetalIndexFlat index(res, 128, faiss::METRIC_L2);

// Add vectors
index.add(n, vectors);

// Search
std::vector<float> distances(nq * k);
std::vector<faiss::idx_t> labels(nq * k);
index.search(nq, queries, k, distances.data(), labels.data());
```

### Convert from CPU FAISS index

```cpp
faiss::IndexFlatL2 cpu_index(128);
cpu_index.add(n, vectors);

auto res = std::make_shared<faiss_metal::StandardMetalResources>();
auto metal_index = faiss_metal::index_cpu_to_metal(res, &cpu_index);

// Search on GPU
metal_index->search(nq, queries, k, distances, labels);

// Convert back if needed
auto cpu_index2 = faiss_metal::index_metal_to_cpu(metal_index.get());
```

## Build

Requires macOS 12+, CMake 3.24+, Xcode CLI tools, and FAISS installed.

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

### Run tests

```bash
cmake -B build -DFAISS_METAL_BUILD_TESTS=ON
cmake --build build -j
cd build && ctest --output-on-failure
```

### Run benchmarks

```bash
./build/bench_metal_flat
```

Benchmarks compare Metal vs CPU FAISS across dataset sizes (10K-100K vectors), dimensions (32-1536), and k values.

## Project structure

```
include/faiss-metal/           Public C++ headers
  MetalIndexFlat.h             Flat index (subclasses faiss::Index)
  MetalResources.h             Abstract resource manager
  StandardMetalResources.h     Default resource manager
  MetalDeviceCapabilities.h    Runtime Apple Silicon detection
  faiss_metal.h                Convenience header (includes all)

src/                           Implementation (ObjC++)
  MetalDistance.h/.mm          L2/IP distance via MPS or simdgroup GEMM
  MetalL2Norm.h/.mm            L2 norm computation
  MetalSelect.h/.mm            Top-k selection dispatch
  MetalTensor.h                Tensor wrapping MTLBuffer
  MetalContext.h/.mm           Metal device/library init
  MetalIndexFlat.mm            Index add/search/reset
  StandardMetalResources.mm    Default resources

shaders/                       Metal compute shaders
  simdgroup_gemm.metal         Custom FP16 GEMM (M2+)
  l2_norm.metal                FP32 L2 norm
  l2_norm_f16.metal            FP16 L2 norm (M2+)
  broadcast_sum.metal          L2 distance norm broadcast
  warp_select.metal            SIMD-group top-k (k <= 32)
  block_select.metal           Threadgroup top-k (k > 32)
  GemmParams.h                 Shared struct (Metal + C++)

tests/
  test_metal_flat.mm           Index correctness vs CPU FAISS
  test_metal_distance.mm       Distance kernel accuracy
  bench_metal_flat.mm          Performance benchmarks
```

## Numerical precision

The FP16 GEMM path (M2+) trades some precision for 2x throughput. Distances have ~1% relative error vs FP32, but top-1 results match exactly. The MPS path (M1 or forced) uses FP32 throughout.

## Limitations

- macOS only (Apple Silicon required)
- Flat index only (IVF planned)
- Maximum 2^31-1 vectors (Metal shaders use int32 indices)
- No Python bindings yet

## License

MIT
