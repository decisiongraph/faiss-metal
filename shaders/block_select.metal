#include <metal_stdlib>
using namespace metal;

/// Block-level top-k selection for k > 32.
/// Each threadgroup processes one query row, finding the k smallest (L2) or largest (IP) values.
///
/// Uses a register-based heap per thread, then merges across the threadgroup.
///
/// Dispatch: threadgroups = nq, threads_per_threadgroup = 256 (or 512)
/// k must be <= 2048

// Per-thread heap insertion (min-heap for keeping top-k smallest)
// We keep a local sorted buffer of size LOCAL_K per thread
#define LOCAL_K 8

kernel void block_select_min(
    device const float* distances [[buffer(0)]],   // (nq x nv)
    device float* out_distances [[buffer(1)]],      // (nq x k)
    device int32_t* out_indices [[buffer(2)]],      // (nq x k)
    constant uint& nv [[buffer(3)]],
    constant uint& k [[buffer(4)]],
    uint row [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]]) {

    // Each thread scans a stripe of the row and keeps its LOCAL_K best
    float local_dist[LOCAL_K];
    int32_t local_idx[LOCAL_K];
    uint local_count = 0;
    float local_max = INFINITY; // worst value in our local buffer

    device const float* row_data = distances + row * nv;

    for (uint i = tid; i < nv; i += tg_size) {
        float d = row_data[i];
        if (local_count < LOCAL_K) {
            // Fill phase
            local_dist[local_count] = d;
            local_idx[local_count] = (int32_t)i;
            local_count++;
            if (local_count == LOCAL_K) {
                // Find max in buffer
                local_max = local_dist[0];
                for (uint j = 1; j < LOCAL_K; j++) {
                    local_max = max(local_max, local_dist[j]);
                }
            }
        } else if (d < local_max) {
            // Replace the worst element
            uint worst = 0;
            for (uint j = 1; j < LOCAL_K; j++) {
                if (local_dist[j] > local_dist[worst]) worst = j;
            }
            local_dist[worst] = d;
            local_idx[worst] = (int32_t)i;
            local_max = local_dist[0];
            for (uint j = 1; j < LOCAL_K; j++) {
                local_max = max(local_max, local_dist[j]);
            }
        }
    }

    // Write local results to shared memory for merging
    // 256 threads * 8 entries * (4+4) bytes = 16KB < 32KB threadgroup limit
    threadgroup float shared_dist[256 * LOCAL_K];
    threadgroup int32_t shared_idx[256 * LOCAL_K];

    uint base = tid * LOCAL_K;
    for (uint i = 0; i < local_count; i++) {
        shared_dist[base + i] = local_dist[i];
        shared_idx[base + i] = local_idx[i];
    }
    for (uint i = local_count; i < LOCAL_K; i++) {
        shared_dist[base + i] = INFINITY;
        shared_idx[base + i] = -1;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Thread 0 merges all local buffers and writes final top-k
    if (tid == 0) {
        uint total_candidates = tg_size * LOCAL_K;
        device float* out_d = out_distances + row * k;
        device int32_t* out_i = out_indices + row * k;

        // Simple selection: find k smallest from all candidates
        for (uint ki = 0; ki < k; ki++) {
            float best_d = INFINITY;
            uint best_pos = 0;
            for (uint j = 0; j < total_candidates; j++) {
                if (shared_dist[j] < best_d) {
                    best_d = shared_dist[j];
                    best_pos = j;
                }
            }
            out_d[ki] = best_d;
            out_i[ki] = shared_idx[best_pos];
            shared_dist[best_pos] = INFINITY; // mark as used
        }
    }
}
