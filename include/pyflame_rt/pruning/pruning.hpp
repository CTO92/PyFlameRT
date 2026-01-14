#pragma once

#include "pyflame_rt/tensor.hpp"
#include "pyflame_rt/graph.hpp"
#include <memory>
#include <vector>
#include <functional>
#include <unordered_map>
#include <unordered_set>

namespace pyflame_rt {
namespace pruning {

/// Pruning granularity
enum class PruningGranularity {
    Unstructured,    // Individual weight pruning
    Structured,      // Channel/filter pruning
    Block,           // Block-wise pruning (e.g., 4x4 blocks)
    NM              // N:M sparsity pattern (e.g., 2:4)
};

/// Pruning criterion
enum class PruningCriterion {
    Magnitude,       // Prune smallest absolute values
    Movement,        // Prune based on gradient movement
    Random,          // Random pruning (baseline)
    Taylor,          // Taylor expansion importance
    LAMP            // Layer-adaptive magnitude pruning
};

/// Pruning schedule
enum class PruningSchedule {
    OneShot,         // Single pruning step
    Iterative,       // Gradual pruning over iterations
    Cubic,           // Cubic sparsity schedule
    Polynomial       // Polynomial schedule
};

/// Configuration for weight pruning
struct PruningConfig {
    /// Target sparsity (0.0 = no pruning, 1.0 = all pruned)
    float target_sparsity = 0.5f;

    /// Pruning granularity
    PruningGranularity granularity = PruningGranularity::Unstructured;

    /// Pruning criterion
    PruningCriterion criterion = PruningCriterion::Magnitude;

    /// Pruning schedule
    PruningSchedule schedule = PruningSchedule::OneShot;

    /// Number of pruning iterations (for iterative schedules)
    size_t num_iterations = 10;

    /// Start sparsity (for iterative schedules)
    float start_sparsity = 0.0f;

    /// End sparsity (for iterative schedules)
    float end_sparsity = 0.9f;

    /// Layers to prune (empty = all layers)
    std::vector<std::string> layers_to_prune;

    /// Layers to exclude from pruning
    std::vector<std::string> layers_to_exclude;

    /// N value for N:M sparsity
    size_t nm_n = 2;

    /// M value for N:M sparsity
    size_t nm_m = 4;

    /// Block size for block pruning
    std::vector<size_t> block_size = {4, 4};

    /// Fine-tune after pruning
    bool fine_tune = true;

    /// Number of fine-tuning epochs
    size_t fine_tune_epochs = 5;

    /// Learning rate for fine-tuning
    float fine_tune_lr = 1e-5f;
};

/// Pruning mask for a single tensor
struct PruningMask {
    /// Tensor name
    std::string tensor_name;

    /// Binary mask (1 = keep, 0 = prune)
    Tensor mask;

    /// Current sparsity level
    float sparsity = 0.0f;

    /// Apply mask to tensor
    Tensor apply(const Tensor& tensor) const;

    /// Count non-zero elements
    size_t count_nonzero() const;
};

/// Pruning statistics
struct PruningStats {
    /// Total parameters before pruning
    size_t total_params = 0;

    /// Non-zero parameters after pruning
    size_t nonzero_params = 0;

    /// Overall sparsity
    float overall_sparsity = 0.0f;

    /// Per-layer sparsity
    std::unordered_map<std::string, float> layer_sparsity;

    /// Compression ratio
    float compression_ratio = 1.0f;

    /// Estimated memory savings (bytes)
    size_t memory_savings = 0;
};

/// Abstract base class for pruning strategies
class PruningStrategy {
public:
    virtual ~PruningStrategy() = default;

    /// Compute pruning mask for a tensor
    virtual PruningMask compute_mask(
        const std::string& name,
        const Tensor& weights,
        float target_sparsity) = 0;

    /// Compute importance scores for weights
    virtual Tensor compute_importance(const Tensor& weights) = 0;

    /// Get strategy name
    virtual std::string name() const = 0;
};

/// Magnitude-based pruning
class MagnitudePruning : public PruningStrategy {
public:
    PruningMask compute_mask(
        const std::string& name,
        const Tensor& weights,
        float target_sparsity) override;

    Tensor compute_importance(const Tensor& weights) override;

    std::string name() const override { return "magnitude"; }
};

/// Structured pruning (channel/filter)
class StructuredPruning : public PruningStrategy {
public:
    enum class Dimension { Channel, Filter, Head };

    explicit StructuredPruning(Dimension dim = Dimension::Channel);

    PruningMask compute_mask(
        const std::string& name,
        const Tensor& weights,
        float target_sparsity) override;

    Tensor compute_importance(const Tensor& weights) override;

    std::string name() const override { return "structured"; }

private:
    Dimension dimension_;
};

/// N:M sparsity pruning
class NMSparsityPruning : public PruningStrategy {
public:
    NMSparsityPruning(size_t n = 2, size_t m = 4);

    PruningMask compute_mask(
        const std::string& name,
        const Tensor& weights,
        float target_sparsity) override;

    Tensor compute_importance(const Tensor& weights) override;

    std::string name() const override { return "nm_sparsity"; }

private:
    size_t n_, m_;
};

/// Weight pruner - main interface
class WeightPruner {
public:
    explicit WeightPruner(const PruningConfig& config);
    ~WeightPruner();

    // ========================================================================
    // Pruning Operations
    // ========================================================================

    /// Analyze graph and determine prunable layers
    std::vector<std::string> analyze(const Graph& graph);

    /// Compute pruning masks for all target layers
    std::vector<PruningMask> compute_masks(
        const Graph& graph,
        float target_sparsity);

    /// Apply pruning masks to graph weights
    Graph apply(const Graph& graph,
                const std::vector<PruningMask>& masks);

    /// One-shot pruning: analyze, compute masks, apply
    Graph prune(const Graph& graph);

    /// Iterative pruning with callback
    using IterationCallback = std::function<void(
        size_t iteration,
        float current_sparsity,
        const PruningStats& stats)>;

    Graph prune_iterative(
        const Graph& graph,
        IterationCallback callback = nullptr);

    // ========================================================================
    // Statistics and Analysis
    // ========================================================================

    /// Get pruning statistics
    PruningStats get_stats() const;

    /// Analyze sparsity distribution
    std::vector<std::pair<float, size_t>> get_sparsity_histogram(
        const Graph& graph,
        size_t num_bins = 10) const;

    /// Estimate accuracy impact (heuristic)
    float estimate_accuracy_impact(const PruningStats& stats) const;

    // ========================================================================
    // Export
    // ========================================================================

    /// Export pruning report
    std::string export_report() const;

    /// Get configuration
    const PruningConfig& config() const { return config_; }

private:
    PruningConfig config_;
    std::unique_ptr<PruningStrategy> strategy_;
    PruningStats stats_;

    float compute_sparsity_at_iteration(size_t iteration) const;
    void update_stats(const Graph& graph, const std::vector<PruningMask>& masks);
};

/// Convenience functions
std::unique_ptr<PruningStrategy> create_strategy(PruningCriterion criterion);
std::unique_ptr<PruningStrategy> create_strategy(const PruningConfig& config);
Graph prune_model(const Graph& graph, float sparsity);
Graph prune_model(const Graph& graph, const PruningConfig& config);

} // namespace pruning
} // namespace pyflame_rt
