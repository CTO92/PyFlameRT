#include "pyflame_rt/pruning/pruning.hpp"
#include <algorithm>
#include <numeric>
#include <cmath>

namespace pyflame_rt {
namespace pruning {

NMSparsityPruning::NMSparsityPruning(size_t n, size_t m)
    : n_(n), m_(m)
{
    if (n_ > m_) {
        throw std::invalid_argument("N must be <= M in N:M sparsity");
    }
    if (n_ == 0 || m_ == 0) {
        throw std::invalid_argument("N and M must be > 0");
    }
}

Tensor NMSparsityPruning::compute_importance(const Tensor& weights) {
    // For N:M sparsity, importance is just absolute value
    size_t num_elements = weights.num_elements();

    // Security: handle empty tensors (HIGH-P4 fix)
    if (num_elements == 0) {
        return Tensor(weights.shape(), weights.dtype());
    }

    Tensor importance(weights.shape(), weights.dtype());
    const float* src = static_cast<const float*>(weights.data());
    float* dst = static_cast<float*>(importance.data());

    // Security: null pointer checks (HIGH-P4 fix)
    if (!src || !dst) {
        throw std::runtime_error("Null data pointer in tensor");
    }

    for (size_t i = 0; i < num_elements; ++i) {
        dst[i] = std::abs(src[i]);
    }

    return importance;
}

PruningMask NMSparsityPruning::compute_mask(
    const std::string& name,
    const Tensor& weights,
    float target_sparsity)
{
    // N:M sparsity: keep N largest values in every M consecutive elements
    // e.g., 2:4 keeps 2 out of every 4 elements (50% sparsity)
    // The target_sparsity parameter is ignored as N:M has fixed sparsity

    PruningMask mask;
    mask.tensor_name = name;
    mask.sparsity = 1.0f - static_cast<float>(n_) / m_;

    size_t total = weights.num_elements();

    // Security: handle empty tensors (HIGH-P4 fix)
    if (total == 0) {
        mask.mask = Tensor(weights.shape(), DType::Float32);
        return mask;
    }

    mask.mask = Tensor(weights.shape(), DType::Float32);
    float* mask_data = static_cast<float*>(mask.mask.data());
    const float* w = static_cast<const float*>(weights.data());

    // Security: null pointer checks (HIGH-P4 fix)
    if (!mask_data || !w) {
        throw std::runtime_error("Null data pointer in tensor");
    }

    // Initialize mask to zeros
    std::fill(mask_data, mask_data + total, 0.0f);

    // Process in groups of M
    for (size_t i = 0; i < total; i += m_) {
        size_t group_size = std::min(m_, total - i);

        if (group_size < m_) {
            // Last partial group: keep proportionally
            size_t keep = static_cast<size_t>(
                std::ceil(static_cast<float>(group_size * n_) / m_));
            keep = std::min(keep, group_size);

            // Find indices of largest values in this partial group
            std::vector<size_t> indices(group_size);
            std::iota(indices.begin(), indices.end(), 0);

            std::partial_sort(indices.begin(), indices.begin() + keep, indices.end(),
                [&w, i](size_t a, size_t b) {
                    return std::abs(w[i + a]) > std::abs(w[i + b]);
                });

            // Set mask for kept elements
            for (size_t j = 0; j < keep; ++j) {
                mask_data[i + indices[j]] = 1.0f;
            }
        } else {
            // Full group: find indices of N largest absolute values
            std::vector<size_t> indices(m_);
            std::iota(indices.begin(), indices.end(), 0);

            // Partial sort to find N largest
            std::partial_sort(indices.begin(), indices.begin() + n_, indices.end(),
                [&w, i](size_t a, size_t b) {
                    return std::abs(w[i + a]) > std::abs(w[i + b]);
                });

            // Set mask for N largest elements
            for (size_t j = 0; j < n_; ++j) {
                mask_data[i + indices[j]] = 1.0f;
            }
        }
    }

    return mask;
}

} // namespace pruning
} // namespace pyflame_rt
