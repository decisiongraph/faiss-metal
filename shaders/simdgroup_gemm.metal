#include <metal_stdlib>
#include <metal_simdgroup_matrix>
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

struct GemmParams {
    uint M;       // rows of A / rows of C
    uint N;       // rows of B / cols of C  (B is transposed)
    uint K;       // cols of A / cols of B
    float alpha;
    float beta;
};

kernel void simdgroup_gemm_f16(
    device const half* A [[buffer(0)]],       // (M x K) row-major
    device const half* B [[buffer(1)]],       // (N x K) row-major (transposed in compute)
    device float* C [[buffer(2)]],            // (M x N) row-major, output in float32
    constant GemmParams& params [[buffer(3)]],
    uint2 gid [[threadgroup_position_in_grid]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]) {

    const uint m_start = gid.y * BM;
    const uint n_start = gid.x * BN;

    if (m_start >= params.M || n_start >= params.N) return;

    // Shared memory: double-buffered, with padding to avoid bank conflicts
    threadgroup half shared_a[2][BM][BK + 4] __attribute__((aligned(16)));
    threadgroup half shared_b[2][BN][BK + 4] __attribute__((aligned(16)));  // B stored as (N x K)

    // 4 SIMD groups, each computes a 16x16 block of the 32x32 tile
    const uint simd_m = (simd_group / 2) * 16;
    const uint simd_n = (simd_group % 2) * 16;

    // 2x2 grid of 8x8 accumulators per SIMD group = 16x16
    simdgroup_half8x8 c_frag[2][2];
    for (uint i = 0; i < 2; i++)
        for (uint j = 0; j < 2; j++)
            c_frag[i][j] = simdgroup_half8x8(0.0h);

    const uint num_k_tiles = (params.K + BK - 1) / BK;
    const uint thread_id = simd_group * 32 + simd_lane;
    const uint total_threads = 128;
    uint buf = 0;

    // Preload first tile
    {
        for (uint i = thread_id; i < BM * BK; i += total_threads) {
            uint r = i / BK, c = i % BK;
            uint row = m_start + r, col = c;
            shared_a[0][r][c] = (row < params.M && col < params.K) ? A[row * params.K + col] : half(0.0h);
        }
        for (uint i = thread_id; i < BN * BK; i += total_threads) {
            uint r = i / BK, c = i % BK;
            uint row = n_start + r, col = c;
            shared_b[0][r][c] = (row < params.N && col < params.K) ? B[row * params.K + col] : half(0.0h);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint tile_k = 0; tile_k < num_k_tiles; tile_k++) {
        uint next_buf = 1 - buf;

        // Prefetch next tile
        if (tile_k + 1 < num_k_tiles) {
            uint k_next = (tile_k + 1) * BK;
            for (uint i = thread_id; i < BM * BK; i += total_threads) {
                uint r = i / BK, c = i % BK;
                uint row = m_start + r, col = k_next + c;
                shared_a[next_buf][r][c] = (row < params.M && col < params.K)
                    ? A[row * params.K + col] : half(0.0h);
            }
            for (uint i = thread_id; i < BN * BK; i += total_threads) {
                uint r = i / BK, c = i % BK;
                uint row = n_start + r, col = k_next + c;
                shared_b[next_buf][r][c] = (row < params.N && col < params.K)
                    ? B[row * params.K + col] : half(0.0h);
            }
        }

        // Compute: iterate over k in steps of 8 (SIMD tile size)
        for (uint k = 0; k < BK; k += SIMD_TILE) {
            simdgroup_half8x8 a_frag[2];
            for (uint i = 0; i < 2; i++) {
                simdgroup_load(a_frag[i], &shared_a[buf][simd_m + i * 8][k], BK + 4);
            }

            // B is stored as (N x K), load B^T tiles
            // simdgroup_load reads 8x8 from (row, col) with stride
            simdgroup_half8x8 b_frag[2];
            for (uint j = 0; j < 2; j++) {
                simdgroup_load(b_frag[j], &shared_b[buf][simd_n + j * 8][k], BK + 4);
            }

            for (uint i = 0; i < 2; i++) {
                for (uint j = 0; j < 2; j++) {
                    // A * B^T: we have A rows and B rows (which are B^T columns)
                    // simdgroup_multiply_accumulate(C, A, B, C) computes C += A * B
                    // Since B is stored transposed, we need to transpose b_frag
                    simdgroup_half8x8 bt;
                    simdgroup_load(bt, &shared_b[buf][simd_n + j * 8][k], BK + 4, true);
                    // Actually: we want C[i][j] += A_frag[i] * B_frag[j]^T
                    // The "true" parameter in simdgroup_load transposes the load
                    simdgroup_multiply_accumulate(c_frag[i][j], a_frag[i], bt, c_frag[i][j]);
                }
            }
        }

        buf = next_buf;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Store results: convert half accumulators to float32 output
    const half alpha_h = half(params.alpha);
    const half beta_h = half(params.beta);

    for (uint i = 0; i < 2; i++) {
        for (uint j = 0; j < 2; j++) {
            uint out_row_base = m_start + simd_m + i * 8;
            uint out_col_base = n_start + simd_n + j * 8;

            for (uint r = 0; r < 8; r++) {
                for (uint c_idx = 0; c_idx < 8; c_idx++) {
                    uint out_row = out_row_base + r;
                    uint out_col = out_col_base + c_idx;
                    if (out_row < params.M && out_col < params.N) {
                        uint idx = out_row * params.N + out_col;
                        float val = float(alpha_h * c_frag[i][j][r][c_idx]);
                        if (params.beta != 0.0f) {
                            val += params.beta * C[idx];
                        }
                        C[idx] = val;
                    }
                }
            }
        }
    }
}

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

    // Store: half accumulator → float32 output
    for (uint i = 0; i < 2; i++) {
        for (uint j = 0; j < 2; j++) {
            uint out_row_base = m_start + simd_m + i * 8;
            uint out_col_base = n_start + simd_n + j * 8;

            for (uint r = 0; r < 8; r++) {
                for (uint c_idx = 0; c_idx < 8; c_idx++) {
                    uint out_row = out_row_base + r;
                    uint out_col = out_col_base + c_idx;
                    if (out_row < params.M && out_col < params.N) {
                        uint idx = out_row * params.N + out_col;
                        float val = float(half(params.alpha) * c_frag[i][j][r][c_idx]);
                        if (params.beta != 0.0f) {
                            val += params.beta * C[idx];
                        }
                        C[idx] = val;
                    }
                }
            }
        }
    }
}
