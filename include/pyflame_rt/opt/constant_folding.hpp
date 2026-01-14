#pragma once

#include "pyflame_rt/opt/pass.hpp"

#include <unordered_set>

namespace pyflame_rt {
namespace opt {

/// Constant folding pass configuration
struct ConstantFoldingConfig {
    /// Maximum tensor size to fold (bytes)
    size_t max_tensor_bytes = 16 * 1024 * 1024;  // 16 MB

    /// Fold shape-computing ops (Shape, Size)
    bool fold_shape_ops = true;

    /// Fold expensive ops (MatMul, Conv) - usually false
    bool fold_expensive_ops = false;

    /// Operators to never fold
    std::vector<std::string> exclude_ops;
};

/// Constant folding optimization pass
///
/// Pre-computes nodes where all inputs are constants.
///
/// Supports:
/// - Arithmetic operations (Add, Sub, Mul, Div, etc.)
/// - Unary operations (Neg, Abs, Sqrt, etc.)
/// - Shape operations with constant shapes (Reshape, Transpose)
/// - Reduction operations on constant tensors
///
class ConstantFoldingPass : public Pass {
public:
    ConstantFoldingPass() = default;
    explicit ConstantFoldingPass(ConstantFoldingConfig config);

    const char* name() const override { return "ConstantFolding"; }

    const char* description() const override {
        return "Pre-compute operations with constant inputs";
    }

    PassResult run(Graph& graph) override;

    bool should_run(OptLevel level) const override {
        return level >= OptLevel::Basic;
    }

    void set_config(const ConstantFoldingConfig& config) { config_ = config; }
    const ConstantFoldingConfig& config() const { return config_; }

private:
    ConstantFoldingConfig config_;

    /// Set of tensor names known to be constant
    std::unordered_set<std::string> constant_tensors_;

    /// Check if node can be constant folded
    bool can_fold(const Graph& graph, const Node& node) const;

    /// Check if tensor is constant
    bool is_constant_tensor(const Graph& graph, const std::string& name) const;

    /// Evaluate node and create initializers for outputs
    bool fold_node(Graph& graph, const Node& node);

    /// Check if operator is in exclude list
    bool is_excluded(const std::string& op_type) const;

    /// Estimate output size for a node
    std::optional<size_t> estimate_output_size(
        const Graph& graph, const Node& node) const;
};

} // namespace opt
} // namespace pyflame_rt
