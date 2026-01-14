#include "pyflame_rt/opt/layout_optimization.hpp"

#include <unordered_map>

namespace pyflame_rt {
namespace opt {

// ============================================================================
// Layout Utilities
// ============================================================================

const char* layout_to_string(Layout layout) {
    switch (layout) {
        case Layout::NCHW: return "NCHW";
        case Layout::NHWC: return "NHWC";
        case Layout::NC4HW4: return "NC4HW4";
        case Layout::Unknown: return "Unknown";
    }
    return "Unknown";
}

Layout layout_from_string(const std::string& str) {
    if (str == "NCHW") return Layout::NCHW;
    if (str == "NHWC") return Layout::NHWC;
    if (str == "NC4HW4") return Layout::NC4HW4;
    return Layout::Unknown;
}

// ============================================================================
// LayoutOptimizationPass
// ============================================================================

LayoutOptimizationPass::LayoutOptimizationPass(LayoutConfig config)
    : config_(std::move(config)) {}

Layout LayoutOptimizationPass::get_preferred_layout(const Node& node) const {
    // Convolution ops prefer specific layout
    if (node.op_type() == "Conv" ||
        node.op_type() == "ConvTranspose" ||
        node.op_type() == "MaxPool" ||
        node.op_type() == "AveragePool" ||
        node.op_type() == "GlobalAveragePool" ||
        node.op_type() == "BatchNormalization") {
        return config_.conv_layout;
    }

    // Element-wise ops don't care about layout
    if (node.op_type() == "Add" ||
        node.op_type() == "Sub" ||
        node.op_type() == "Mul" ||
        node.op_type() == "Div" ||
        node.op_type() == "Relu" ||
        node.op_type() == "Sigmoid" ||
        node.op_type() == "Tanh") {
        return Layout::Unknown;  // Any layout is fine
    }

    // MatMul doesn't have a spatial layout
    if (node.op_type() == "MatMul" ||
        node.op_type() == "Gemm") {
        return Layout::Unknown;
    }

    return Layout::Unknown;
}

void LayoutOptimizationPass::insert_transpose(
    Graph& graph,
    const std::string& input_name,
    const std::string& output_name,
    Layout from,
    Layout to)
{
    // Generate permutation based on layout conversion
    std::vector<int64_t> perm;

    if (from == Layout::NCHW && to == Layout::NHWC) {
        perm = {0, 2, 3, 1};  // NCHW -> NHWC
    } else if (from == Layout::NHWC && to == Layout::NCHW) {
        perm = {0, 3, 1, 2};  // NHWC -> NCHW
    } else {
        return;  // Unsupported conversion
    }

    // Create transpose node
    auto transpose = std::make_unique<Node>(
        output_name + "_transpose",
        "Transpose",
        std::vector<std::string>{input_name},
        std::vector<std::string>{output_name}
    );
    transpose->set_attr("perm", perm);

    graph.add_node(std::move(transpose));
}

void LayoutOptimizationPass::analyze_layouts(Graph& graph) {
    // Annotate each tensor with its layout
    // This is a placeholder - full implementation would track layouts
    // through the graph and insert transposes where needed
}

PassResult LayoutOptimizationPass::run(Graph& graph) {
    PassResult result;

    // Phase 1: Analyze layouts
    analyze_layouts(graph);

    // Phase 2: Propagate layouts and insert transposes
    // For now, this is a no-op - layouts are already consistent
    // in most models (either all NCHW or all NHWC)

    if (config_.propagate_layout) {
        // Count nodes that prefer specific layouts
        std::unordered_map<Layout, int> layout_counts;

        for (const auto& node : graph.nodes()) {
            Layout preferred = get_preferred_layout(*node);
            if (preferred != Layout::Unknown) {
                layout_counts[preferred]++;
            }
        }

        // If all layout-sensitive ops agree, no changes needed
        // Otherwise, we'd need to insert transposes
        if (layout_counts.size() > 1) {
            result.warnings.push_back(
                "Graph has mixed layout preferences - optimization skipped"
            );
        }
    }

    return result;
}

// Register pass with global registry
REGISTER_OPT_PASS(LayoutOptimizationPass)

} // namespace opt
} // namespace pyflame_rt
