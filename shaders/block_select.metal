#include <metal_stdlib>
using namespace metal;

/// Block-level top-k selection for k > 32.
/// Each threadgroup processes one query row.
///
/// Phase 1: Each thread scans a stripe, keeps LOCAL_K best in a sorted buffer.
/// Phase 2: Parallel pairwise merge reduction in shared memory.
/// Phase 3: Thread 0 extracts final top-k from merged result.
///
/// Dispatch: threadgroups = nq, threads_per_threadgroup = 256 (or 512 on M3+)
/// k must be <= LOCAL_K * threads_per_threadgroup

#define LOCAL_K 8

// Insertion sort a small buffer (ascending for min)
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

/// Merge two sorted runs of LOCAL_K elements in shared memory (ascending).
/// Each thread in pair handles one side. Winner keeps best LOCAL_K from both.
inline void merge_pair_asc(
    threadgroup float* shared_dist,
    threadgroup int32_t* shared_idx,
    uint my_base,
    uint other_base)
{
    // Two-pointer merge: both runs are sorted ascending.
    // Keep the smallest LOCAL_K from 2*LOCAL_K candidates.
    float merged_d[LOCAL_K];
    int32_t merged_i[LOCAL_K];
    uint a = 0, b = 0;

    for (uint out = 0; out < LOCAL_K; out++) {
        float da = (a < LOCAL_K) ? shared_dist[my_base + a] : INFINITY;
        float db = (b < LOCAL_K) ? shared_dist[other_base + b] : INFINITY;
        if (da <= db) {
            merged_d[out] = da;
            merged_i[out] = shared_idx[my_base + a];
            a++;
        } else {
            merged_d[out] = db;
            merged_i[out] = shared_idx[other_base + b];
            b++;
        }
    }

    for (uint i = 0; i < LOCAL_K; i++) {
        shared_dist[my_base + i] = merged_d[i];
        shared_idx[my_base + i] = merged_i[i];
    }
}

inline void merge_pair_desc(
    threadgroup float* shared_dist,
    threadgroup int32_t* shared_idx,
    uint my_base,
    uint other_base)
{
    float merged_d[LOCAL_K];
    int32_t merged_i[LOCAL_K];
    uint a = 0, b = 0;

    for (uint out = 0; out < LOCAL_K; out++) {
        float da = (a < LOCAL_K) ? shared_dist[my_base + a] : -INFINITY;
        float db = (b < LOCAL_K) ? shared_dist[other_base + b] : -INFINITY;
        if (da >= db) {
            merged_d[out] = da;
            merged_i[out] = shared_idx[my_base + a];
            a++;
        } else {
            merged_d[out] = db;
            merged_i[out] = shared_idx[other_base + b];
            b++;
        }
    }

    for (uint i = 0; i < LOCAL_K; i++) {
        shared_dist[my_base + i] = merged_d[i];
        shared_idx[my_base + i] = merged_i[i];
    }
}

kernel void block_select_min(
    device const float* distances [[buffer(0)]],   // (nq x nv)
    device float* out_distances [[buffer(1)]],      // (nq x k)
    device int32_t* out_indices [[buffer(2)]],      // (nq x k)
    constant uint& nv [[buffer(3)]],
    constant uint& k [[buffer(4)]],
    threadgroup float* shared_dist [[threadgroup(0)]],
    threadgroup int32_t* shared_idx [[threadgroup(1)]],
    uint row [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]]) {

    float local_dist[LOCAL_K];
    int32_t local_idx[LOCAL_K];
    uint local_count = 0;
    float local_max = INFINITY;

    device const float* row_data = distances + row * nv;

    // Phase 1: Each thread scans its stripe, keeping LOCAL_K best
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

    sort_local_asc(local_dist, local_idx, local_count);

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

    // Phase 2: Parallel pairwise merge reduction.
    // Each round, thread i merges with thread i+stride, halving active threads.
    for (uint stride = 1; stride < tg_size; stride *= 2) {
        if ((tid % (stride * 2)) == 0 && (tid + stride) < tg_size) {
            uint my_base = tid * LOCAL_K;
            uint other_base = (tid + stride) * LOCAL_K;
            merge_pair_asc(shared_dist, shared_idx, my_base, other_base);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Phase 3: Thread 0 writes final top-k from merged result (already sorted)
    if (tid == 0) {
        device float* out_d = out_distances + row * k;
        device int32_t* out_i = out_indices + row * k;
        uint count = min(k, (uint)LOCAL_K);
        for (uint i = 0; i < count; i++) {
            out_d[i] = shared_dist[i];
            out_i[i] = shared_idx[i];
        }
        for (uint i = count; i < k; i++) {
            out_d[i] = INFINITY;
            out_i[i] = -1;
        }
    }
}

kernel void block_select_max(
    device const float* distances [[buffer(0)]],
    device float* out_distances [[buffer(1)]],
    device int32_t* out_indices [[buffer(2)]],
    constant uint& nv [[buffer(3)]],
    constant uint& k [[buffer(4)]],
    threadgroup float* shared_dist [[threadgroup(0)]],
    threadgroup int32_t* shared_idx [[threadgroup(1)]],
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

    // Parallel pairwise merge reduction
    for (uint stride = 1; stride < tg_size; stride *= 2) {
        if ((tid % (stride * 2)) == 0 && (tid + stride) < tg_size) {
            uint my_base = tid * LOCAL_K;
            uint other_base = (tid + stride) * LOCAL_K;
            merge_pair_desc(shared_dist, shared_idx, my_base, other_base);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (tid == 0) {
        device float* out_d = out_distances + row * k;
        device int32_t* out_i = out_indices + row * k;
        uint count = min(k, (uint)LOCAL_K);
        for (uint i = 0; i < count; i++) {
            out_d[i] = shared_dist[i];
            out_i[i] = shared_idx[i];
        }
        for (uint i = count; i < k; i++) {
            out_d[i] = -INFINITY;
            out_i[i] = -1;
        }
    }
}
