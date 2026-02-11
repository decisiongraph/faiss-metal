#include <metal_stdlib>
using namespace metal;

/// SIMD-group level top-k selection for small k (k <= 32).
/// Each SIMD group processes one query row using a bitonic sorting network.
/// This is the fast path -- for k <= 32, we can keep the entire top-k in registers.
///
/// Dispatch: threadgroups = nq, threads_per_threadgroup = 32

kernel void warp_select_min(
    device const float* distances [[buffer(0)]],   // (nq x nv)
    device float* out_distances [[buffer(1)]],      // (nq x k)
    device int32_t* out_indices [[buffer(2)]],      // (nq x k)
    constant uint& nv [[buffer(3)]],
    constant uint& k [[buffer(4)]],
    uint row [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]) {

    // Each lane maintains one slot in the top-k buffer
    // lane i holds the i-th best value (lane 0 = best, lane k-1 = worst of top-k)
    float my_dist = INFINITY;
    int32_t my_idx = -1;

    device const float* row_data = distances + row * nv;

    // Phase 1: scan all values
    for (uint i = lane; i < nv; i += 32) {
        float d = row_data[i];

        // Check if this value should enter the top-k
        // The k-th lane holds the current threshold
        float threshold = simd_broadcast(my_dist, k - 1);

        if (d < threshold) {
            // Insert into sorted position using SIMD shuffle
            // Find insertion point
            uint pos = 0;
            for (uint j = 0; j < k; j++) {
                float other = simd_broadcast(my_dist, j);
                if (d >= other) pos++;
            }

            if (pos < k) {
                // Shift elements down and insert
                float prev_d = my_dist;
                int32_t prev_i = my_idx;

                float new_d = simd_broadcast(d, 0); // broadcast doesn't change value
                int32_t new_i = (int32_t)i;

                // Each lane decides what value to hold
                if (lane == pos) {
                    // This lane takes the new value
                    my_dist = d;
                    my_idx = (int32_t)i;
                } else if (lane > pos && lane < k) {
                    // Shift: take value from lane-1
                    my_dist = simd_shuffle_up(prev_d, 1);
                    my_idx = simd_shuffle_up(prev_i, 1);
                }
            }
        }
    }

    // Phase 2: write results
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
                float prev_d = my_dist;
                int32_t prev_i = my_idx;

                if (lane == pos) {
                    my_dist = d;
                    my_idx = (int32_t)i;
                } else if (lane > pos && lane < k) {
                    my_dist = simd_shuffle_up(prev_d, 1);
                    my_idx = simd_shuffle_up(prev_i, 1);
                }
            }
        }
    }

    if (lane < k) {
        out_distances[row * k + lane] = my_dist;
        out_indices[row * k + lane] = my_idx;
    }
}
