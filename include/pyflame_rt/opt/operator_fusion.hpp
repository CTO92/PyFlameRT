#pragma once

#include "pyflame_rt/opt/pass.hpp"
#include "pyflame_rt/opt/pattern_matcher.hpp"

#include <functional>

namespace pyflame_rt {
namespace opt {

/// Fusion pattern definition
struct FusionPattern {
    /// Pattern name (e.g., "ConvBNRelu")
    std::string name;

    /// Graph pattern to match
    Pattern pattern;

    /// Fused operator type
    std::string fused_op;

    /// Function to create fused node from matched nodes
    using FuseFunc = std::function<std::shared_ptr<Node>(
        const PatternMatch& match,
        Graph& graph
    )>;
    FuseFunc fuse_func;

    /// Constraint function (optional)
    using ConstraintFunc = std::function<bool(const PatternMatch&, const Graph&)>;
    ConstraintFunc constraint;
};

/// Fusion pass configuration
struct FusionConfig {
    /// Enable Conv + BatchNorm fusion
    bool fuse_conv_bn = true;

    /// Enable Conv + BatchNorm + Activation fusion
    bool fuse_conv_bn_activation = true;

    /// Enable MatMul + Add (bias) fusion
    bool fuse_matmul_add = true;

    /// Enable element-wise + activation fusion
    bool fuse_elementwise_activation = true;

    /// Only fuse if backend supports the fused op
    bool check_backend_support = false;

    /// Target backend name
    std::string target_backend = "cpu";
};

/// Operator fusion pass
///
/// Identifies sequences of operators that can be combined into
/// single fused operations for better performance.
///
/// Supported fusion patterns:
/// - Conv + BatchNorm + ReLU -> FusedConvBNRelu
/// - Conv + BatchNorm -> FusedConvBN
/// - MatMul + Add (bias) -> Gemm
/// - Add + ReLU -> FusedAddRelu
/// - BatchNorm + ReLU -> FusedBNRelu
///
class OperatorFusionPass : public Pass {
public:
    OperatorFusionPass();
    explicit OperatorFusionPass(FusionConfig config);

    const char* name() const override { return "OperatorFusion"; }

    const char* description() const override {
        return "Fuse operator sequences into optimized kernels";
    }

    PassResult run(Graph& graph) override;

    bool should_run(OptLevel level) const override {
        return level >= OptLevel::Extended;
    }

    std::vector<std::string> dependencies() const override {
        return {"ConstantFolding"};
    }

    /// Register custom fusion pattern
    void register_pattern(FusionPattern pattern);

    /// Get registered patterns
    const std::vector<FusionPattern>& patterns() const { return patterns_; }

    void set_config(const FusionConfig& config) { config_ = config; }
    const FusionConfig& config() const { return config_; }

private:
    FusionConfig config_;
    std::vector<FusionPattern> patterns_;

    /// Apply single fusion to graph
    bool apply_fusion(
        Graph& graph,
        const FusionPattern& pattern,
        const PatternMatch& match
    );

    /// Register built-in patterns
    void register_builtin_patterns();
};

} // namespace opt
} // namespace pyflame_rt
