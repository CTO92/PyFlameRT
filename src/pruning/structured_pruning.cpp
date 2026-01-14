#include "pyflame_rt/pruning/pruning.hpp"
#include <algorithm>
#include <numeric>
#include <cmath>
#include <unordered_set>
#include <limits>
#include <stdexcept>

namespace pyflame_rt {
namespace pruning {

namespace {
/// Safe index calculation with overflow check (MED-P2 fix)
inline size_t safe_index_nchw(size_t n, size_t c, size_t s,
                               size_t C, size_t spatial, size_t total_elements) {
    // Check each multiplication for overflow
    if (C > 0 && n > std::numeric_limits<size_t>::max() / C) {
        throw std::overflow_error("Index calculation overflow in n*C");
    }
    size_t n_c = n * C;

    if (spatial > 0 && n_c > std::numeric_limits<size_t>::max() / spatial) {
        throw std::overflow_error("Index calculation overflow in n*C*spatial");
    }

    if (spatial > 0 && c > std::numeric_limits<size_t>::max() / spatial) {
        throw std::overflow_error("Index calculation overflow in c*spatial");
    }

    size_t idx = n_c * spatial + c * spatial + s;

    // Final bounds check
    if (idx >= total_elements) {
        throw std::out_of_range("Computed index exceeds tensor bounds");
    }

    return idx;
}
} // anonymous namespace

StructuredPruning::StructuredPruning(Dimension dim)
    : dimension_(dim)
{
}

Tensor StructuredPruning::compute_importance(const Tensor& weights) {
    auto shape = weights.shape();
    const float* w = static_cast<const float*>(weights.data());

    if (shape.size() < 2) {
        // 1D tensor - return as-is
        Tensor importance(weights.shape(), DType::Float32);
        float* imp = static_cast<float*>(importance.data());
        for (size_t i = 0; i < weights.num_elements(); ++i) {
            imp[i] = std::abs(w[i]);
        }
        return importance;
    }

    size_t num_units = 0;
    size_t unit_size = 0;

    if (dimension_ == Dimension::Filter) {
        // For Conv weights: [out_channels, in_channels, H, W]
        // Importance per output filter (first dimension)
        num_units = shape[0];
        unit_size = weights.num_elements() / num_units;
    } else if (dimension_ == Dimension::Channel) {
        // Importance per input channel (second dimension)
        if (shape.size() >= 2) {
            num_units = shape[1];
            unit_size = 1;
            // For multi-dimensional, calculate properly
            for (size_t i = 2; i < shape.size(); ++i) {
                unit_size *= shape[i];
            }
            unit_size *= shape[0];  // Include batch dimension in stride
        }
    } else {
        // Head dimension - typically for attention
        num_units = shape[0];
        unit_size = weights.num_elements() / num_units;
    }

    Tensor importance({static_cast<int64_t>(num_units)}, DType::Float32);
    float* imp = static_cast<float*>(importance.data());
    std::fill(imp, imp + num_units, 0.0f);

    if (dimension_ == Dimension::Filter || dimension_ == Dimension::Head) {
        // Sum absolute values for each filter/head
        for (size_t u = 0; u < num_units; ++u) {
            for (size_t i = 0; i < unit_size; ++i) {
                imp[u] += std::abs(w[u * unit_size + i]);
            }
            imp[u] /= unit_size;  // Normalize by unit size
        }
    } else {
        // Channel dimension - more complex indexing
        // For shape [N, C, H, W], compute importance per channel C
        size_t N = static_cast<size_t>(shape[0]);
        size_t C = static_cast<size_t>(shape[1]);

        // Security: check for division by zero (MED-P2 fix)
        if (N == 0 || C == 0) {
            throw std::invalid_argument("Shape dimensions N and C cannot be zero");
        }

        size_t total_elements = weights.num_elements();
        size_t spatial = total_elements / (N * C);

        if (spatial == 0) {
            throw std::invalid_argument("Spatial dimension cannot be zero");
        }

        for (size_t c = 0; c < C; ++c) {
            for (size_t n = 0; n < N; ++n) {
                for (size_t s = 0; s < spatial; ++s) {
                    // Security: use safe index calculation (MED-P2 fix)
                    size_t idx = safe_index_nchw(n, c, s, C, spatial, total_elements);
                    imp[c] += std::abs(w[idx]);
                }
            }
            // Division is safe since we already checked N and spatial > 0
            imp[c] /= static_cast<float>(N * spatial);
        }
    }

    return importance;
}

PruningMask StructuredPruning::compute_mask(
    const std::string& name,
    const Tensor& weights,
    float target_sparsity)
{
    PruningMask mask;
    mask.tensor_name = name;
    mask.sparsity = target_sparsity;

    auto shape = weights.shape();

    if (target_sparsity <= 0.0f || shape.size() < 2) {
        // No pruning - all ones mask
        mask.mask = Tensor(weights.shape(), DType::Float32);
        float* mask_data = static_cast<float*>(mask.mask.data());
        std::fill(mask_data, mask_data + weights.num_elements(), 1.0f);
        return mask;
    }

    // Compute per-unit importance
    Tensor importance = compute_importance(weights);
    size_t num_units = importance.num_elements();
    size_t num_to_prune = static_cast<size_t>(num_units * target_sparsity);

    // Don't prune all units
    num_to_prune = std::min(num_to_prune, num_units - 1);

    // Find indices to prune (lowest importance)
    std::vector<size_t> indices(num_units);
    std::iota(indices.begin(), indices.end(), 0);

    const float* imp = static_cast<const float*>(importance.data());
    std::sort(indices.begin(), indices.end(),
              [imp](size_t a, size_t b) { return imp[a] < imp[b]; });

    std::unordered_set<size_t> prune_set(indices.begin(),
                                          indices.begin() + num_to_prune);

    // Create structured mask
    mask.mask = Tensor(weights.shape(), DType::Float32);
    float* mask_data = static_cast<float*>(mask.mask.data());
    std::fill(mask_data, mask_data + weights.num_elements(), 1.0f);

    if (dimension_ == Dimension::Filter || dimension_ == Dimension::Head) {
        size_t unit_size = weights.num_elements() / num_units;
        for (size_t u = 0; u < num_units; ++u) {
            if (prune_set.count(u)) {
                // Zero out entire filter/head
                for (size_t i = 0; i < unit_size; ++i) {
                    mask_data[u * unit_size + i] = 0.0f;
                }
            }
        }
    } else {
        // Channel dimension
        size_t N = shape[0];
        size_t C = shape[1];
        size_t spatial = weights.num_elements() / (N * C);

        for (size_t c = 0; c < C; ++c) {
            if (prune_set.count(c)) {
                // Zero out this channel across all batch elements
                for (size_t n = 0; n < N; ++n) {
                    for (size_t s = 0; s < spatial; ++s) {
                        size_t idx = n * C * spatial + c * spatial + s;
                        mask_data[idx] = 0.0f;
                    }
                }
            }
        }
    }

    return mask;
}

} // namespace pruning
} // namespace pyflame_rt
