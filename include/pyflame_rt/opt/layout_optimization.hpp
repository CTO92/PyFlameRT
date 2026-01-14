#pragma once

#include "pyflame_rt/opt/pass.hpp"

namespace pyflame_rt {
namespace opt {

/// Tensor layout format
enum class Layout {
    NCHW,    // Batch, Channel, Height, Width (PyTorch default)
    NHWC,    // Batch, Height, Width, Channel (TensorFlow default)
    NC4HW4,  // Blocked format for SIMD
    Unknown
};

/// Layout optimization configuration
struct LayoutConfig {
    /// Target layout for convolutions
    Layout conv_layout = Layout::NCHW;

    /// Insert explicit transpose nodes when needed
    bool insert_transposes = true;

    /// Propagate layout through graph
    bool propagate_layout = true;

    /// Target backend name
    std::string target_backend = "cpu";
};

/// Layout optimization pass
///
/// Converts tensor layouts to optimal format for the target backend.
/// Inserts layout conversion operations where necessary.
///
/// For CPU backend:
/// - Uses NCHW for most operations
/// - May convert to blocked format for vectorization
///
class LayoutOptimizationPass : public Pass {
public:
    LayoutOptimizationPass() = default;
    explicit LayoutOptimizationPass(LayoutConfig config);

    const char* name() const override { return "LayoutOptimization"; }

    const char* description() const override {
        return "Optimize tensor memory layouts for target hardware";
    }

    PassResult run(Graph& graph) override;

    bool should_run(OptLevel level) const override {
        return level >= OptLevel::All;
    }

    std::vector<std::string> dependencies() const override {
        return {"OperatorFusion"};  // Run after fusion
    }

    void set_config(const LayoutConfig& config) { config_ = config; }
    const LayoutConfig& config() const { return config_; }

private:
    LayoutConfig config_;

    /// Get preferred layout for an operator
    Layout get_preferred_layout(const Node& node) const;

    /// Insert layout conversion node
    void insert_transpose(
        Graph& graph,
        const std::string& input_name,
        const std::string& output_name,
        Layout from,
        Layout to
    );

    /// Analyze and annotate layouts in the graph
    void analyze_layouts(Graph& graph);
};

/// Convert layout enum to string
const char* layout_to_string(Layout layout);

/// Parse layout from string
Layout layout_from_string(const std::string& str);

} // namespace opt
} // namespace pyflame_rt
