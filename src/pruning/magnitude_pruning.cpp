#include "pyflame_rt/pruning/pruning.hpp"
#include <algorithm>
#include <numeric>
#include <cmath>

namespace pyflame_rt {
namespace pruning {

Tensor MagnitudePruning::compute_importance(const Tensor& weights) {
    // Importance = absolute value of weights
    Tensor importance(weights.shape(), weights.dtype());
    const float* src = static_cast<const float*>(weights.data());
    float* dst = static_cast<float*>(importance.data());

    for (size_t i = 0; i < weights.num_elements(); ++i) {
        dst[i] = std::abs(src[i]);
    }

    return importance;
}

PruningMask MagnitudePruning::compute_mask(
    const std::string& name,
    const Tensor& weights,
    float target_sparsity)
{
    PruningMask mask;
    mask.tensor_name = name;
    mask.sparsity = target_sparsity;

    if (target_sparsity <= 0.0f) {
        // No pruning - all ones mask
        mask.mask = Tensor(weights.shape(), DType::Float32);
        float* mask_data = static_cast<float*>(mask.mask.data());
        std::fill(mask_data, mask_data + weights.num_elements(), 1.0f);
        return mask;
    }

    if (target_sparsity >= 1.0f) {
        // Full pruning - all zeros mask
        mask.mask = Tensor(weights.shape(), DType::Float32);
        float* mask_data = static_cast<float*>(mask.mask.data());
        std::fill(mask_data, mask_data + weights.num_elements(), 0.0f);
        return mask;
    }

    // Compute importance scores
    Tensor importance = compute_importance(weights);

    // Security: validate importance tensor has data (HIGH-P3 fix)
    size_t num_elements = importance.num_elements();
    if (num_elements == 0) {
        throw std::runtime_error("Cannot prune empty weight tensor");
    }

    // Gather all importance scores
    std::vector<float> scores(num_elements);
    const float* src = static_cast<const float*>(importance.data());
    if (!src) {
        throw std::runtime_error("Importance tensor has null data pointer");
    }
    std::copy(src, src + num_elements, scores.begin());

    // Sort to find threshold
    std::sort(scores.begin(), scores.end());

    // Security: safe threshold index calculation (HIGH-P3 fix)
    // Use double for intermediate calculation to avoid precision loss
    size_t threshold_idx = static_cast<size_t>(
        std::floor(static_cast<double>(scores.size()) * target_sparsity));
    // Ensure we don't go past the last valid index
    threshold_idx = std::min(threshold_idx, scores.size() - 1);
    float threshold = scores[threshold_idx];

    // Create binary mask: keep weights with importance > threshold
    mask.mask = Tensor(weights.shape(), DType::Float32);
    float* mask_data = static_cast<float*>(mask.mask.data());

    for (size_t i = 0; i < weights.num_elements(); ++i) {
        mask_data[i] = (src[i] > threshold) ? 1.0f : 0.0f;
    }

    // Handle tie-breaking: if we pruned too many, restore some
    size_t target_zeros = static_cast<size_t>(weights.num_elements() * target_sparsity);
    size_t actual_zeros = 0;
    for (size_t i = 0; i < weights.num_elements(); ++i) {
        if (mask_data[i] == 0.0f) actual_zeros++;
    }

    // If we pruned too many (due to ties at threshold), restore some
    if (actual_zeros > target_zeros) {
        size_t to_restore = actual_zeros - target_zeros;
        for (size_t i = 0; i < weights.num_elements() && to_restore > 0; ++i) {
            if (mask_data[i] == 0.0f && src[i] == threshold) {
                mask_data[i] = 1.0f;
                to_restore--;
            }
        }
    }

    return mask;
}

} // namespace pruning
} // namespace pyflame_rt
