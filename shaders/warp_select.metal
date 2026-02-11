#include <metal_stdlib>
using namespace metal;

/// SIMD-group level top-k selection for small k (k <= 32).
/// Each SIMD group processes one query row.
/// Each lane maintains one slot in a sorted top-k buffer.
///
/// Dispatch: threadgroups = nq, threads_per_threadgroup = 32
/// Output is sorted (lane 0 = best, lane k-1 = worst of top-k).

kernel void warp_select_min(
    device const float* distances [[buffer(0)]],   // (nq x nv)
    device float* out_distances [[buffer(1)]],      // (nq x k)
    device int32_t* out_indices [[buffer(2)]],      // (nq x k)
    constant uint& nv [[buffer(3)]],
    constant uint& k [[buffer(4)]],
    uint row [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]) {

    // Each lane holds one slot: lane 0 = best (smallest), lane k-1 = k-th best
    float my_dist = INFINITY;
    int32_t my_idx = -1;

    device const float* row_data = distances + row * nv;

    for (uint i = lane; i < nv; i += 32) {
        float d = row_data[i];

        // Threshold: worst value currently in top-k (lane k-1)
        float threshold = simd_broadcast(my_dist, k - 1);

        if (d < threshold) {
            // Find insertion point: count how many current top-k values are better
            uint pos = 0;
            for (uint j = 0; j < k; j++) {
                float other = simd_broadcast(my_dist, j);
                if (d >= other) pos++;
            }

            if (pos < k) {
                // Snapshot current values across ALL lanes before any mutation
                // Use simd_shuffle to read lane-1's value from the snapshot
                float old_dist = my_dist;
                int32_t old_idx = my_idx;

                if (lane == pos) {
                    // This lane takes the new value
                    my_dist = d;
                    my_idx = (int32_t)i;
                } else if (lane > pos && lane < k) {
                    // Shift down: this lane takes what was in lane-1
                    // simd_shuffle reads from the pre-mutation snapshot (old_dist)
                    // which is correct since all lanes captured their values above
                    my_dist = simd_shuffle(old_dist, lane - 1);
                    my_idx = simd_shuffle(old_idx, lane - 1);
                }
                // Lanes < pos and lanes >= k: unchanged
            }
        }
    }

    // Write sorted results
    if (lane < k) {
        out_distances[row * k + lane] = my_dist;
        out_indices[row * k + lane] = my_idx;
    }
}

/// Max variant for inner product (find k largest)
kernel void warp_select_max(
    device const float* distances [[buffer(0)]],
    device float* out_distances [[buffer(1)]],
    device int32_t* out_indices [[buffer(2)]],
    constant uint& nv [[buffer(3)]],
    constant uint& k [[buffer(4)]],
    uint row [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]) {

    float my_dist = -INFINITY;
    int32_t my_idx = -1;

    device const float* row_data = distances + row * nv;

    for (uint i = lane; i < nv; i += 32) {
        float d = row_data[i];
        float threshold = simd_broadcast(my_dist, k - 1);

        if (d > threshold) {
            uint pos = 0;
            for (uint j = 0; j < k; j++) {
                float other = simd_broadcast(my_dist, j);
                if (d <= other) pos++;
            }

            if (pos < k) {
                float old_dist = my_dist;
                int32_t old_idx = my_idx;

                if (lane == pos) {
                    my_dist = d;
                    my_idx = (int32_t)i;
                } else if (lane > pos && lane < k) {
                    my_dist = simd_shuffle(old_dist, lane - 1);
                    my_idx = simd_shuffle(old_idx, lane - 1);
                }
            }
        }
    }

    if (lane < k) {
        out_distances[row * k + lane] = my_dist;
        out_indices[row * k + lane] = my_idx;
    }
}
