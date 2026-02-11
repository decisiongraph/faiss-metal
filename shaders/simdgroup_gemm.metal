#include <metal_stdlib>
#include <metal_simdgroup_matrix>
#include "GemmParams.h"
using namespace metal;

/// Half-precision GEMM using simdgroup_matrix_multiply_accumulate.
/// Available on Apple7+ (M1+), but Apple8+ (M2+) has 2x FP16 throughput.
///
/// Computes C = alpha * A * B^T + beta * C
/// A: (M x K), B: (N x K) -- note B is stored row-major, transposed during compute
/// C: (M x N)
///
/// This replaces MPSMatrixMultiplication for distance computation, avoiding
/// MPS dispatch overhead and enabling fusion with subsequent operations.
///
/// Tile sizes: BM=32, BN=32, BK=32 (fits 32KB threadgroup memory)
/// Dispatch: grid = (ceil(N/32), ceil(M/32)), threadgroup = (32, 4) = 128 threads

constant uint BM = 32;
constant uint BN = 32;
constant uint BK = 32;
constant uint SIMD_TILE = 8;

/// Float32-input variant: casts to half on load for simdgroup_matrix,
/// accumulates in half, outputs float32. Slight precision loss but 2x throughput.
kernel void simdgroup_gemm_f32_via_f16(
    device const float* A [[buffer(0)]],
    device const float* B [[buffer(1)]],
    device float* C [[buffer(2)]],
    constant GemmParams& params [[buffer(3)]],
    uint2 gid [[threadgroup_position_in_grid]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]) {

    const uint m_start = gid.y * BM;
    const uint n_start = gid.x * BN;

    if (m_start >= params.M || n_start >= params.N) return;

    // Load float32 data, convert to half in shared memory
    threadgroup half shared_a[2][BM][BK + 4] __attribute__((aligned(16)));
    threadgroup half shared_b[2][BN][BK + 4] __attribute__((aligned(16)));

    const uint simd_m = (simd_group / 2) * 16;
    const uint simd_n = (simd_group % 2) * 16;

    simdgroup_half8x8 c_frag[2][2];
    for (uint i = 0; i < 2; i++)
        for (uint j = 0; j < 2; j++)
            c_frag[i][j] = simdgroup_half8x8(0.0h);

    const uint num_k_tiles = (params.K + BK - 1) / BK;
    const uint thread_id = simd_group * 32 + simd_lane;
    const uint total_threads = 128;
    uint buf = 0;

    // Preload first tile (float32 → half conversion during load)
    {
        for (uint i = thread_id; i < BM * BK; i += total_threads) {
            uint r = i / BK, c = i % BK;
            uint row = m_start + r, col = c;
            shared_a[0][r][c] = (row < params.M && col < params.K)
                ? half(A[row * params.K + col]) : half(0.0h);
        }
        for (uint i = thread_id; i < BN * BK; i += total_threads) {
            uint r = i / BK, c = i % BK;
            uint row = n_start + r, col = c;
            shared_b[0][r][c] = (row < params.N && col < params.K)
                ? half(B[row * params.K + col]) : half(0.0h);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint tile_k = 0; tile_k < num_k_tiles; tile_k++) {
        uint next_buf = 1 - buf;

        if (tile_k + 1 < num_k_tiles) {
            uint k_next = (tile_k + 1) * BK;
            for (uint i = thread_id; i < BM * BK; i += total_threads) {
                uint r = i / BK, c = i % BK;
                uint row = m_start + r, col = k_next + c;
                shared_a[next_buf][r][c] = (row < params.M && col < params.K)
                    ? half(A[row * params.K + col]) : half(0.0h);
            }
            for (uint i = thread_id; i < BN * BK; i += total_threads) {
                uint r = i / BK, c = i % BK;
                uint row = n_start + r, col = k_next + c;
                shared_b[next_buf][r][c] = (row < params.N && col < params.K)
                    ? half(B[row * params.K + col]) : half(0.0h);
            }
        }

        for (uint k = 0; k < BK; k += SIMD_TILE) {
            simdgroup_half8x8 a_frag[2];
            for (uint i = 0; i < 2; i++)
                simdgroup_load(a_frag[i], &shared_a[buf][simd_m + i * 8][k], BK + 4);

            for (uint i = 0; i < 2; i++) {
                for (uint j = 0; j < 2; j++) {
                    simdgroup_half8x8 bt;
                    simdgroup_load(bt, &shared_b[buf][simd_n + j * 8][k], BK + 4, true);
                    simdgroup_multiply_accumulate(c_frag[i][j], a_frag[i], bt, c_frag[i][j]);
                }
            }
        }

        buf = next_buf;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Store: half accumulator → threadgroup half → float32 output
    threadgroup half store_buf[BM][BN + 4] __attribute__((aligned(16)));

    for (uint i = 0; i < 2; i++) {
        for (uint j = 0; j < 2; j++) {
            simdgroup_store(c_frag[i][j],
                            &store_buf[simd_m + i * 8][simd_n + j * 8],
                            BN + 4);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Convert from threadgroup half to device float with alpha/beta scaling
    for (uint idx = thread_id; idx < BM * BN; idx += total_threads) {
        uint r = idx / BN;
        uint c = idx % BN;
        uint out_row = m_start + r;
        uint out_col = n_start + c;
        if (out_row < params.M && out_col < params.N) {
            uint oidx = out_row * params.N + out_col;
            float val = float(store_buf[r][c]) * params.alpha;
            if (params.beta != 0.0f) {
                val += params.beta * C[oidx];
            }
            C[oidx] = val;
        }
    }
}
