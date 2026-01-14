#include "pyflame_rt/pruning/pruning.hpp"
#include <algorithm>
#include <numeric>
#include <cmath>
#include <sstream>

namespace pyflame_rt {
namespace pruning {

// ============================================================================
// PruningMask Implementation
// ============================================================================

Tensor PruningMask::apply(const Tensor& tensor) const {
    if (mask.num_elements() != tensor.num_elements()) {
        throw std::runtime_error("Mask and tensor size mismatch");
    }

    // Security: validate tensor has data (MED-P1 fix)
    if (tensor.num_elements() == 0) {
        return Tensor(tensor.shape(), tensor.dtype());
    }

    Tensor result(tensor.shape(), tensor.dtype());
    const float* src = static_cast<const float*>(tensor.data());
    const float* mask_data = static_cast<const float*>(mask.data());
    float* dst = static_cast<float*>(result.data());

    // Security: null pointer checks (MED-P1 fix)
    if (!src || !mask_data || !dst) {
        throw std::runtime_error("Null data pointer in tensor or mask");
    }

    for (size_t i = 0; i < tensor.num_elements(); ++i) {
        dst[i] = src[i] * mask_data[i];
    }

    return result;
}

size_t PruningMask::count_nonzero() const {
    // Security: check for empty mask (MED-P1 fix)
    if (mask.num_elements() == 0) {
        return 0;
    }

    const float* mask_data = static_cast<const float*>(mask.data());
    // Security: null pointer check (MED-P1 fix)
    if (!mask_data) {
        throw std::runtime_error("Null data pointer in mask");
    }

    size_t count = 0;
    for (size_t i = 0; i < mask.num_elements(); ++i) {
        if (mask_data[i] != 0.0f) {
            count++;
        }
    }
    return count;
}

// ============================================================================
// WeightPruner Implementation
// ============================================================================

WeightPruner::WeightPruner(const PruningConfig& config)
    : config_(config)
{
    strategy_ = create_strategy(config);
}

WeightPruner::~WeightPruner() = default;

std::vector<std::string> WeightPruner::analyze(const Graph& graph) {
    std::vector<std::string> prunable;

    for (const auto& node : graph.nodes()) {
        // Check if node has weights that can be pruned
        std::string op_type = node->op_type();
        if (op_type == "Conv" ||
            op_type == "MatMul" ||
            op_type == "Gemm" ||
            op_type == "Linear" ||
            op_type == "ConvTranspose") {

            std::string name = node->name();

            // Check exclusion list
            bool excluded = false;
            for (const auto& ex : config_.layers_to_exclude) {
                if (name.find(ex) != std::string::npos) {
                    excluded = true;
                    break;
                }
            }

            if (!excluded) {
                // Check inclusion list (if specified)
                if (config_.layers_to_prune.empty()) {
                    prunable.push_back(name);
                } else {
                    for (const auto& inc : config_.layers_to_prune) {
                        if (name.find(inc) != std::string::npos) {
                            prunable.push_back(name);
                            break;
                        }
                    }
                }
            }
        }
    }

    return prunable;
}

std::vector<PruningMask> WeightPruner::compute_masks(
    const Graph& graph,
    float target_sparsity)
{
    std::vector<PruningMask> masks;
    auto prunable = analyze(graph);

    for (const auto& name : prunable) {
        // Try different naming conventions for weights
        const Tensor* weights = graph.get_initializer(name + "_weight");
        if (!weights) {
            weights = graph.get_initializer(name + ".weight");
        }
        if (!weights) {
            weights = graph.get_initializer(name + "/weight");
        }
        if (!weights) {
            // Try to find weight in node inputs
            for (const auto& node : graph.nodes()) {
                if (node->name() == name && node->inputs().size() > 1) {
                    weights = graph.get_initializer(node->inputs()[1]);
                    break;
                }
            }
        }
        if (!weights) continue;

        // Compute mask
        PruningMask mask = strategy_->compute_mask(name, *weights, target_sparsity);
        masks.push_back(std::move(mask));
    }

    return masks;
}

Graph WeightPruner::apply(const Graph& graph,
                          const std::vector<PruningMask>& masks)
{
    Graph pruned = graph.clone();

    for (const auto& mask : masks) {
        // Try different naming conventions
        std::vector<std::string> possible_names = {
            mask.tensor_name + "_weight",
            mask.tensor_name + ".weight",
            mask.tensor_name + "/weight"
        };

        Tensor* weights = nullptr;
        for (const auto& weight_name : possible_names) {
            weights = pruned.get_mutable_initializer(weight_name);
            if (weights) break;
        }

        if (!weights) {
            // Try to find weight in node inputs
            for (const auto& node : pruned.nodes()) {
                if (node->name() == mask.tensor_name && node->inputs().size() > 1) {
                    weights = pruned.get_mutable_initializer(node->inputs()[1]);
                    break;
                }
            }
        }

        if (!weights) continue;

        // Apply mask
        *weights = mask.apply(*weights);
    }

    return pruned;
}

Graph WeightPruner::prune(const Graph& graph) {
    auto masks = compute_masks(graph, config_.target_sparsity);
    auto pruned = apply(graph, masks);
    update_stats(pruned, masks);
    return pruned;
}

Graph WeightPruner::prune_iterative(
    const Graph& graph,
    IterationCallback callback)
{
    Graph current = graph.clone();

    for (size_t iter = 0; iter < config_.num_iterations; ++iter) {
        float current_sparsity = compute_sparsity_at_iteration(iter);

        auto masks = compute_masks(current, current_sparsity);
        current = apply(current, masks);
        update_stats(current, masks);

        if (callback) {
            callback(iter, current_sparsity, stats_);
        }
    }

    return current;
}

float WeightPruner::compute_sparsity_at_iteration(size_t iteration) const {
    if (config_.num_iterations == 0) {
        return config_.target_sparsity;
    }

    float t = static_cast<float>(iteration + 1) / config_.num_iterations;

    switch (config_.schedule) {
        case PruningSchedule::OneShot:
            return config_.target_sparsity;

        case PruningSchedule::Iterative:
            return config_.start_sparsity +
                   (config_.end_sparsity - config_.start_sparsity) * t;

        case PruningSchedule::Cubic:
            return config_.end_sparsity +
                   (config_.start_sparsity - config_.end_sparsity) *
                   std::pow(1.0f - t, 3);

        case PruningSchedule::Polynomial:
            return config_.start_sparsity +
                   (config_.end_sparsity - config_.start_sparsity) *
                   std::pow(t, 3);

        default:
            return config_.target_sparsity;
    }
}

void WeightPruner::update_stats(const Graph& graph,
                                 const std::vector<PruningMask>& masks)
{
    stats_.total_params = 0;
    stats_.nonzero_params = 0;
    stats_.layer_sparsity.clear();

    for (const auto& mask : masks) {
        // Try to find weights
        std::vector<std::string> possible_names = {
            mask.tensor_name + "_weight",
            mask.tensor_name + ".weight",
            mask.tensor_name + "/weight"
        };

        const Tensor* weights = nullptr;
        for (const auto& weight_name : possible_names) {
            weights = graph.get_initializer(weight_name);
            if (weights) break;
        }

        if (!weights) {
            for (const auto& node : graph.nodes()) {
                if (node->name() == mask.tensor_name && node->inputs().size() > 1) {
                    weights = graph.get_initializer(node->inputs()[1]);
                    break;
                }
            }
        }

        if (!weights) continue;

        size_t total = weights->num_elements();
        // Security: skip empty tensors (MED-P1 fix)
        if (total == 0) continue;

        const float* data = static_cast<const float*>(weights->data());
        // Security: null pointer check (MED-P1 fix)
        if (!data) continue;

        size_t nonzero = 0;
        for (size_t i = 0; i < total; ++i) {
            if (data[i] != 0.0f) nonzero++;
        }

        stats_.total_params += total;
        stats_.nonzero_params += nonzero;
        // Division is safe since we checked total > 0 above
        stats_.layer_sparsity[mask.tensor_name] =
            1.0f - static_cast<float>(nonzero) / total;
    }

    stats_.overall_sparsity = stats_.total_params > 0 ?
        1.0f - static_cast<float>(stats_.nonzero_params) / stats_.total_params : 0.0f;

    stats_.compression_ratio = stats_.nonzero_params > 0 ?
        static_cast<float>(stats_.total_params) / stats_.nonzero_params : 1.0f;

    stats_.memory_savings = (stats_.total_params - stats_.nonzero_params) * sizeof(float);
}

PruningStats WeightPruner::get_stats() const {
    return stats_;
}

std::vector<std::pair<float, size_t>> WeightPruner::get_sparsity_histogram(
    const Graph& graph,
    size_t num_bins) const
{
    std::vector<std::pair<float, size_t>> histogram(num_bins);
    for (size_t i = 0; i < num_bins; ++i) {
        histogram[i].first = static_cast<float>(i) / num_bins;
        histogram[i].second = 0;
    }

    // Count weights in each sparsity bucket
    for (const auto& [name, sparsity] : stats_.layer_sparsity) {
        size_t bin = std::min(static_cast<size_t>(sparsity * num_bins), num_bins - 1);
        histogram[bin].second++;
    }

    return histogram;
}

float WeightPruner::estimate_accuracy_impact(const PruningStats& stats) const {
    // Simple heuristic: higher sparsity = more accuracy loss
    // This is a rough estimate; actual impact depends on many factors
    float base_impact = stats.overall_sparsity * 0.1f;  // 10% accuracy loss at 100% sparsity

    // Structured pruning typically has lower accuracy impact
    if (config_.granularity == PruningGranularity::Structured) {
        base_impact *= 0.7f;
    }

    // N:M sparsity is more hardware-friendly
    if (config_.granularity == PruningGranularity::NM) {
        base_impact *= 0.8f;
    }

    return base_impact;
}

std::string WeightPruner::export_report() const {
    std::ostringstream oss;
    oss << "Pruning Report\n";
    oss << "==============\n\n";
    oss << "Configuration:\n";
    oss << "  Target sparsity: " << config_.target_sparsity * 100 << "%\n";
    oss << "  Granularity: ";
    switch (config_.granularity) {
        case PruningGranularity::Unstructured: oss << "Unstructured\n"; break;
        case PruningGranularity::Structured: oss << "Structured\n"; break;
        case PruningGranularity::Block: oss << "Block\n"; break;
        case PruningGranularity::NM: oss << "N:M (" << config_.nm_n << ":" << config_.nm_m << ")\n"; break;
    }
    oss << "  Criterion: ";
    switch (config_.criterion) {
        case PruningCriterion::Magnitude: oss << "Magnitude\n"; break;
        case PruningCriterion::Movement: oss << "Movement\n"; break;
        case PruningCriterion::Random: oss << "Random\n"; break;
        case PruningCriterion::Taylor: oss << "Taylor\n"; break;
        case PruningCriterion::LAMP: oss << "LAMP\n"; break;
    }
    oss << "\nResults:\n";
    oss << "  Total parameters: " << stats_.total_params << "\n";
    oss << "  Non-zero parameters: " << stats_.nonzero_params << "\n";
    oss << "  Overall sparsity: " << stats_.overall_sparsity * 100 << "%\n";
    oss << "  Compression ratio: " << stats_.compression_ratio << "x\n";
    oss << "  Memory savings: " << stats_.memory_savings / 1024.0 / 1024.0 << " MB\n";
    oss << "\nPer-layer sparsity:\n";
    for (const auto& [name, sparsity] : stats_.layer_sparsity) {
        oss << "  " << name << ": " << sparsity * 100 << "%\n";
    }
    return oss.str();
}

// ============================================================================
// Factory Functions
// ============================================================================

std::unique_ptr<PruningStrategy> create_strategy(PruningCriterion criterion) {
    switch (criterion) {
        case PruningCriterion::Magnitude:
            return std::make_unique<MagnitudePruning>();
        case PruningCriterion::Movement:
            // Movement pruning would require gradient info; fall back to magnitude
            return std::make_unique<MagnitudePruning>();
        case PruningCriterion::Random:
            return std::make_unique<MagnitudePruning>();  // Would implement RandomPruning
        case PruningCriterion::Taylor:
            return std::make_unique<MagnitudePruning>();  // Would implement TaylorPruning
        case PruningCriterion::LAMP:
            return std::make_unique<MagnitudePruning>();  // Would implement LAMPPruning
        default:
            return std::make_unique<MagnitudePruning>();
    }
}

std::unique_ptr<PruningStrategy> create_strategy(const PruningConfig& config) {
    switch (config.granularity) {
        case PruningGranularity::Structured:
            return std::make_unique<StructuredPruning>();
        case PruningGranularity::NM:
            return std::make_unique<NMSparsityPruning>(config.nm_n, config.nm_m);
        case PruningGranularity::Unstructured:
        case PruningGranularity::Block:
        default:
            return create_strategy(config.criterion);
    }
}

Graph prune_model(const Graph& graph, float sparsity) {
    PruningConfig config;
    config.target_sparsity = sparsity;
    WeightPruner pruner(config);
    return pruner.prune(graph);
}

Graph prune_model(const Graph& graph, const PruningConfig& config) {
    WeightPruner pruner(config);
    return pruner.prune(graph);
}

} // namespace pruning
} // namespace pyflame_rt
