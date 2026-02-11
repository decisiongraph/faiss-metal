#pragma once

#include <faiss/Index.h>
#include <faiss/MetricType.h>
#include <memory>

namespace faiss_metal {

class MetalResources;

/// Flat (brute-force) index on Metal GPU.
/// Subclasses faiss::Index directly -- all data lives in MTLBuffers (unified memory).
class MetalIndexFlat : public faiss::Index {
   public:
    /// @param resources  Metal resource manager (device, queue, shaders)
    /// @param d          Vector dimension
    /// @param metric     METRIC_L2 or METRIC_INNER_PRODUCT
    MetalIndexFlat(
            std::shared_ptr<MetalResources> resources,
            int d,
            faiss::MetricType metric = faiss::METRIC_L2);

    ~MetalIndexFlat() override;

    // --- faiss::Index interface ---
    void add(faiss::idx_t n, const float* x) override;
    void search(
            faiss::idx_t n,
            const float* x,
            faiss::idx_t k,
            float* distances,
            faiss::idx_t* labels,
            const faiss::SearchParameters* params = nullptr) const override;
    void reset() override;
    void reconstruct(faiss::idx_t key, float* recons) const override;

    /// Direct access to stored vectors (CPU pointer into unified memory)
    const float* getVectorsData() const;

    /// Force MPS GEMM path even on M2+ hardware. For testing both paths.
    void setForceMPS(bool force);

   private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

/// Convert CPU IndexFlat → MetalIndexFlat (copies vectors into MTLBuffer)
std::unique_ptr<MetalIndexFlat> index_cpu_to_metal(
        std::shared_ptr<MetalResources> resources,
        const faiss::IndexFlat* cpu_index);

/// Convert MetalIndexFlat → CPU IndexFlat
std::unique_ptr<faiss::IndexFlat> index_metal_to_cpu(
        const MetalIndexFlat* metal_index);

} // namespace faiss_metal
