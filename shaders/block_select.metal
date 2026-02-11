#include <metal_stdlib>
using namespace metal;

/// Block-level top-k selection for k > 32.
/// Each threadgroup processes one query row.
///
/// Each thread scans a stripe, keeps LOCAL_K best in a sorted buffer,
/// then all threads write to shared memory for a final merge.
///
/// Dispatch: threadgroups = nq, threads_per_threadgroup = 256
/// k must be <= 2048

#define LOCAL_K 8

// Insertion sort a small buffer (ascending order for min, descending for max)
inline void sort_local_asc(thread float* dist, thread int32_t* idx, uint count) {
    for (uint i = 1; i < count; i++) {
        float d = dist[i];
        int32_t ix = idx[i];
        uint j = i;
        while (j > 0 && dist[j - 1] > d) {
            dist[j] = dist[j - 1];
            idx[j] = idx[j - 1];
            j--;
        }
        dist[j] = d;
        idx[j] = ix;
    }
}

inline void sort_local_desc(thread float* dist, thread int32_t* idx, uint count) {
    for (uint i = 1; i < count; i++) {
        float d = dist[i];
        int32_t ix = idx[i];
        uint j = i;
        while (j > 0 && dist[j - 1] < d) {
            dist[j] = dist[j - 1];
            idx[j] = idx[j - 1];
            j--;
        }
        dist[j] = d;
        idx[j] = ix;
    }
}

kernel void block_select_min(
    device const float* distances [[buffer(0)]],   // (nq x nv)
    device float* out_distances [[buffer(1)]],      // (nq x k)
    device int32_t* out_indices [[buffer(2)]],      // (nq x k)
    constant uint& nv [[buffer(3)]],
    constant uint& k [[buffer(4)]],
    uint row [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]]) {

    float local_dist[LOCAL_K];
    int32_t local_idx[LOCAL_K];
    uint local_count = 0;
    float local_max = INFINITY;

    device const float* row_data = distances + row * nv;

    for (uint i = tid; i < nv; i += tg_size) {
        float d = row_data[i];
        if (local_count < LOCAL_K) {
            local_dist[local_count] = d;
            local_idx[local_count] = (int32_t)i;
            local_count++;
            if (local_count == LOCAL_K) {
                local_max = local_dist[0];
                for (uint j = 1; j < LOCAL_K; j++)
                    local_max = max(local_max, local_dist[j]);
            }
        } else if (d < local_max) {
            uint worst = 0;
            for (uint j = 1; j < LOCAL_K; j++) {
                if (local_dist[j] > local_dist[worst]) worst = j;
            }
            local_dist[worst] = d;
            local_idx[worst] = (int32_t)i;
            local_max = local_dist[0];
            for (uint j = 1; j < LOCAL_K; j++)
                local_max = max(local_max, local_dist[j]);
        }
    }

    // Sort each thread's local buffer so the merge can early-exit
    sort_local_asc(local_dist, local_idx, local_count);

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

    // Thread 0: k-way merge from sorted per-thread buffers
    if (tid == 0) {
        device float* out_d = out_distances + row * k;
        device int32_t* out_i = out_indices + row * k;

        // Track current position in each thread's sorted buffer
        // Use simple repeated-minimum approach but skip INFINITY entries
        uint total_candidates = tg_size * LOCAL_K;

        for (uint ki = 0; ki < k; ki++) {
            float best_d = INFINITY;
            uint best_pos = 0;
            for (uint j = 0; j < total_candidates; j++) {
                if (shared_dist[j] < best_d) {
                    best_d = shared_dist[j];
                    best_pos = j;
                }
            }
            if (best_d == INFINITY) {
                // No more valid candidates
                for (uint fill = ki; fill < k; fill++) {
                    out_d[fill] = INFINITY;
                    out_i[fill] = -1;
                }
                break;
            }
            out_d[ki] = best_d;
            out_i[ki] = shared_idx[best_pos];
            shared_dist[best_pos] = INFINITY;
        }
    }
}

kernel void block_select_max(
    device const float* distances [[buffer(0)]],
    device float* out_distances [[buffer(1)]],
    device int32_t* out_indices [[buffer(2)]],
    constant uint& nv [[buffer(3)]],
    constant uint& k [[buffer(4)]],
    uint row [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]]) {

    float local_dist[LOCAL_K];
    int32_t local_idx[LOCAL_K];
    uint local_count = 0;
    float local_min = -INFINITY;

    device const float* row_data = distances + row * nv;

    for (uint i = tid; i < nv; i += tg_size) {
        float d = row_data[i];
        if (local_count < LOCAL_K) {
            local_dist[local_count] = d;
            local_idx[local_count] = (int32_t)i;
            local_count++;
            if (local_count == LOCAL_K) {
                local_min = local_dist[0];
                for (uint j = 1; j < LOCAL_K; j++)
                    local_min = min(local_min, local_dist[j]);
            }
        } else if (d > local_min) {
            uint worst = 0;
            for (uint j = 1; j < LOCAL_K; j++) {
                if (local_dist[j] < local_dist[worst]) worst = j;
            }
            local_dist[worst] = d;
            local_idx[worst] = (int32_t)i;
            local_min = local_dist[0];
            for (uint j = 1; j < LOCAL_K; j++)
                local_min = min(local_min, local_dist[j]);
        }
    }

    sort_local_desc(local_dist, local_idx, local_count);

    threadgroup float shared_dist[256 * LOCAL_K];
    threadgroup int32_t shared_idx[256 * LOCAL_K];

    uint base = tid * LOCAL_K;
    for (uint i = 0; i < local_count; i++) {
        shared_dist[base + i] = local_dist[i];
        shared_idx[base + i] = local_idx[i];
    }
    for (uint i = local_count; i < LOCAL_K; i++) {
        shared_dist[base + i] = -INFINITY;
        shared_idx[base + i] = -1;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid == 0) {
        device float* out_d = out_distances + row * k;
        device int32_t* out_i = out_indices + row * k;

        uint total_candidates = tg_size * LOCAL_K;

        for (uint ki = 0; ki < k; ki++) {
            float best_d = -INFINITY;
            uint best_pos = 0;
            for (uint j = 0; j < total_candidates; j++) {
                if (shared_dist[j] > best_d) {
                    best_d = shared_dist[j];
                    best_pos = j;
                }
            }
            if (best_d == -INFINITY) {
                for (uint fill = ki; fill < k; fill++) {
                    out_d[fill] = -INFINITY;
                    out_i[fill] = -1;
                }
                break;
            }
            out_d[ki] = best_d;
            out_i[ki] = shared_idx[best_pos];
            shared_dist[best_pos] = -INFINITY;
        }
    }
}
