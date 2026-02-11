#import <Foundation/Foundation.h>
#import <faiss-metal/MetalIndexFlat.h>
#import <faiss-metal/StandardMetalResources.h>
#import <faiss-metal/MetalDeviceCapabilities.h>
#import <faiss/IndexFlat.h>

#include <cassert>
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

using namespace faiss_metal;

/// Compare Metal vs CPU FAISS results. Allow some tolerance for floating point.
static void compare_results(
        const char* label,
        faiss::idx_t n,
        faiss::idx_t k,
        const float* metal_distances,
        const faiss::idx_t* metal_labels,
        const float* cpu_distances,
        const faiss::idx_t* cpu_labels,
        float dist_tol = 1e-3f) {

    int mismatches = 0;
    for (faiss::idx_t i = 0; i < n; i++) {
        for (faiss::idx_t j = 0; j < k; j++) {
            faiss::idx_t idx = i * k + j;
            float dDiff = std::abs(metal_distances[idx] - cpu_distances[idx]);
            float relDiff = dDiff / std::max(std::abs(cpu_distances[idx]), 1e-6f);

            if (relDiff > dist_tol) {
                if (mismatches < 5) {
                    printf("  [%s] query=%lld rank=%lld: Metal dist=%.6f label=%lld, "
                           "CPU dist=%.6f label=%lld (relDiff=%.6f)\n",
                           label, (long long)i, (long long)j,
                           metal_distances[idx], (long long)metal_labels[idx],
                           cpu_distances[idx], (long long)cpu_labels[idx],
                           relDiff);
                }
                mismatches++;
            }
        }
    }
    if (mismatches > 0) {
        printf("  [%s] WARNING: %d/%lld distance mismatches (tol=%.0e)\n",
               label, mismatches, (long long)(n * k), dist_tol);
    }

    // Check that top-1 labels match (most important)
    int top1_mismatches = 0;
    for (faiss::idx_t i = 0; i < n; i++) {
        if (metal_labels[i * k] != cpu_labels[i * k]) {
            top1_mismatches++;
        }
    }
    assert(top1_mismatches == 0 && "Top-1 labels must match exactly");
}

static void test_flat_l2(size_t nv, size_t nq, size_t d, size_t k) {
    printf("test_flat_l2 (nv=%zu, nq=%zu, d=%zu, k=%zu)... ", nv, nq, d, k);

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    std::vector<float> vectors(nv * d);
    std::vector<float> queries(nq * d);
    for (auto& v : vectors) v = dist(rng);
    for (auto& v : queries) v = dist(rng);

    // CPU reference
    faiss::IndexFlatL2 cpu_index(d);
    cpu_index.add(nv, vectors.data());

    std::vector<float> cpu_distances(nq * k);
    std::vector<faiss::idx_t> cpu_labels(nq * k);
    cpu_index.search(nq, queries.data(), k, cpu_distances.data(), cpu_labels.data());

    // Metal
    auto res = std::make_shared<StandardMetalResources>();
    MetalIndexFlat metal_index(res, d, faiss::METRIC_L2);
    metal_index.add(nv, vectors.data());

    std::vector<float> metal_distances(nq * k);
    std::vector<faiss::idx_t> metal_labels(nq * k);
    metal_index.search(nq, queries.data(), k, metal_distances.data(), metal_labels.data());

    compare_results("L2", nq, k,
                    metal_distances.data(), metal_labels.data(),
                    cpu_distances.data(), cpu_labels.data());

    printf("PASS\n");
}

static void test_flat_ip(size_t nv, size_t nq, size_t d, size_t k) {
    printf("test_flat_ip (nv=%zu, nq=%zu, d=%zu, k=%zu)... ", nv, nq, d, k);

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    std::vector<float> vectors(nv * d);
    std::vector<float> queries(nq * d);
    for (auto& v : vectors) v = dist(rng);
    for (auto& v : queries) v = dist(rng);

    // CPU reference
    faiss::IndexFlatIP cpu_index(d);
    cpu_index.add(nv, vectors.data());

    std::vector<float> cpu_distances(nq * k);
    std::vector<faiss::idx_t> cpu_labels(nq * k);
    cpu_index.search(nq, queries.data(), k, cpu_distances.data(), cpu_labels.data());

    // Metal
    auto res = std::make_shared<StandardMetalResources>();
    MetalIndexFlat metal_index(res, d, faiss::METRIC_INNER_PRODUCT);
    metal_index.add(nv, vectors.data());

    std::vector<float> metal_distances(nq * k);
    std::vector<faiss::idx_t> metal_labels(nq * k);
    metal_index.search(nq, queries.data(), k, metal_distances.data(), metal_labels.data());

    compare_results("IP", nq, k,
                    metal_distances.data(), metal_labels.data(),
                    cpu_distances.data(), cpu_labels.data(),
                    1e-2f); // IP can have slightly larger numerical differences

    printf("PASS\n");
}

static void test_conversion() {
    printf("test_conversion (cpu→metal→cpu)... ");

    const size_t nv = 500;
    const size_t d = 128;
    const size_t nq = 10;
    const size_t k = 5;

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    std::vector<float> vectors(nv * d);
    std::vector<float> queries(nq * d);
    for (auto& v : vectors) v = dist(rng);
    for (auto& v : queries) v = dist(rng);

    // Build CPU index
    faiss::IndexFlatL2 cpu_index(d);
    cpu_index.add(nv, vectors.data());

    // Convert to Metal
    auto res = std::make_shared<StandardMetalResources>();
    auto metal_index = index_cpu_to_metal(res, &cpu_index);

    assert(metal_index->ntotal == cpu_index.ntotal);
    assert(metal_index->d == cpu_index.d);

    // Convert back to CPU
    auto cpu_index2 = index_metal_to_cpu(metal_index.get());
    assert(cpu_index2->ntotal == cpu_index.ntotal);

    // Search both and compare
    std::vector<float> d1(nq * k), d2(nq * k);
    std::vector<faiss::idx_t> l1(nq * k), l2(nq * k);

    cpu_index.search(nq, queries.data(), k, d1.data(), l1.data());
    cpu_index2->search(nq, queries.data(), k, d2.data(), l2.data());

    for (size_t i = 0; i < nq * k; i++) {
        assert(l1[i] == l2[i] && "Round-trip label mismatch");
        assert(std::abs(d1[i] - d2[i]) < 1e-5f && "Round-trip distance mismatch");
    }

    printf("PASS\n");
}

static void test_reset() {
    printf("test_reset... ");

    auto res = std::make_shared<StandardMetalResources>();
    MetalIndexFlat index(res, 32, faiss::METRIC_L2);

    std::vector<float> data(100 * 32, 1.0f);
    index.add(100, data.data());
    assert(index.ntotal == 100);

    index.reset();
    assert(index.ntotal == 0);

    // Add again after reset
    index.add(50, data.data());
    assert(index.ntotal == 50);

    printf("PASS\n");
}

static void test_reconstruct() {
    printf("test_reconstruct... ");

    auto res = std::make_shared<StandardMetalResources>();
    const int d = 64;
    MetalIndexFlat index(res, d, faiss::METRIC_L2);

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<float> data(10 * d);
    for (auto& v : data) v = dist(rng);

    index.add(10, data.data());

    std::vector<float> recons(d);
    for (int i = 0; i < 10; i++) {
        index.reconstruct(i, recons.data());
        for (int j = 0; j < d; j++) {
            assert(recons[j] == data[i * d + j] && "Reconstruct mismatch");
        }
    }

    printf("PASS\n");
}

static void test_forced_mps_search() {
    printf("test_forced_mps_search (L2+IP via MPS path)... ");

    const size_t nv = 500;
    const size_t nq = 5;
    const size_t d = 128;
    const size_t k = 5;

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<float> vectors(nv * d);
    std::vector<float> queries(nq * d);
    for (auto& v : vectors) v = dist(rng);
    for (auto& v : queries) v = dist(rng);

    // CPU reference
    faiss::IndexFlatL2 cpu_index(d);
    cpu_index.add(nv, vectors.data());
    std::vector<float> cpu_distances(nq * k);
    std::vector<faiss::idx_t> cpu_labels(nq * k);
    cpu_index.search(nq, queries.data(), k, cpu_distances.data(), cpu_labels.data());

    // Metal with forced MPS path
    auto res = std::make_shared<StandardMetalResources>();
    MetalIndexFlat metal_index(res, d, faiss::METRIC_L2);
    metal_index.setForceMPS(true);
    metal_index.add(nv, vectors.data());

    std::vector<float> metal_distances(nq * k);
    std::vector<faiss::idx_t> metal_labels(nq * k);
    metal_index.search(nq, queries.data(), k, metal_distances.data(), metal_labels.data());

    compare_results("ForcedMPS-L2", nq, k,
                    metal_distances.data(), metal_labels.data(),
                    cpu_distances.data(), cpu_labels.data());

    printf("PASS\n");
}

int main() {
    @autoreleasepool {
        auto res = std::make_shared<StandardMetalResources>();
        const auto& caps = res->getCapabilities();

        printf("=== MetalIndexFlat Tests ===\n");
        printf("%s\n", faiss_metal::describeCapabilities(caps).c_str());

        // Show active code paths
        printf("GEMM: %s\n", caps.hasFastFP16 ? "simdgroup_matrix (FP16)" : "MPS (FP32)");
        printf("L2 norm: %s\n\n", caps.hasFastFP16 ? "FP16 fast path" : "FP32");

        // Various dimensions
        test_flat_l2(1000, 10, 32, 5);
        test_flat_l2(1000, 10, 128, 10);
        test_flat_l2(500, 5, 768, 5);
        test_flat_l2(500, 5, 1536, 5);

        // Inner product
        test_flat_ip(1000, 10, 128, 10);

        // Edge cases
        test_flat_l2(100, 1, 32, 1);   // single query, k=1

        // Forced MPS path (tests both code paths on M2+)
        test_forced_mps_search();

        // Conversion
        test_conversion();

        // Reset
        test_reset();

        // Reconstruct
        test_reconstruct();

        printf("All MetalIndexFlat tests passed!\n");
    }
    return 0;
}
