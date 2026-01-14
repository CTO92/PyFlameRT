#include "pyflame_rt/opt/constant_folding.hpp"
#include "pyflame_rt/registry.hpp"
#include "pyflame_rt/errors.hpp"

#include <algorithm>

namespace pyflame_rt {
namespace opt {

ConstantFoldingPass::ConstantFoldingPass(ConstantFoldingConfig config)
    : config_(std::move(config)) {}

bool ConstantFoldingPass::is_excluded(const std::string& op_type) const {
    return std::find(config_.exclude_ops.begin(),
                     config_.exclude_ops.end(),
                     op_type) != config_.exclude_ops.end();
}

bool ConstantFoldingPass::is_constant_tensor(
    const Graph& graph, const std::string& name) const
{
    // Check if it's an initializer
    if (graph.has_initializer(name)) {
        return true;
    }

    // Check if produced by a node we've already folded
    if (constant_tensors_.count(name)) {
        return true;
    }

    return false;
}

bool ConstantFoldingPass::can_fold(const Graph& graph, const Node& node) const {
    // Check exclusion list
    if (is_excluded(node.op_type())) {
        return false;
    }

    // Check expensive ops
    if (!config_.fold_expensive_ops) {
        static const std::unordered_set<std::string> expensive_ops = {
            "MatMul", "Gemm", "Conv", "ConvTranspose",
            "BatchNormalization", "LayerNormalization"
        };
        if (expensive_ops.count(node.op_type())) {
            return false;
        }
    }

    // Check shape ops
    if (!config_.fold_shape_ops) {
        static const std::unordered_set<std::string> shape_ops = {
            "Shape", "Size", "ConstantOfShape"
        };
        if (shape_ops.count(node.op_type())) {
            return false;
        }
    }

    // Node must have inputs (otherwise nothing to fold)
    if (node.inputs().empty()) {
        return false;
    }

    // All inputs must be constant
    for (const auto& input : node.inputs()) {
        if (!is_constant_tensor(graph, input)) {
            return false;
        }
    }

    // Check if we have an implementation for this op
    if (!OperatorRegistry::instance().has(node.op_type())) {
        return false;
    }

    // Check output size estimate
    auto size = estimate_output_size(graph, node);
    if (size && *size > config_.max_tensor_bytes) {
        return false;
    }

    return true;
}

std::optional<size_t> ConstantFoldingPass::estimate_output_size(
    const Graph& graph, const Node& node) const
{
    // For element-wise ops, output size equals first input size
    if (node.inputs().empty()) {
        return std::nullopt;
    }

    const std::string& first_input = node.inputs()[0];

    if (graph.has_initializer(first_input)) {
        auto tensor = graph.get_initializer(first_input);
        if (tensor) {
            return tensor->size_bytes();
        }
    }

    return std::nullopt;
}

bool ConstantFoldingPass::fold_node(Graph& graph, const Node& node) {
    // Gather input tensors
    std::vector<const Tensor*> inputs;
    inputs.reserve(node.inputs().size());

    for (const auto& input_name : node.inputs()) {
        auto tensor = graph.get_initializer(input_name);
        if (!tensor) {
            return false;  // Should have been checked by can_fold
        }
        inputs.push_back(tensor.get());
    }

    // Get operator implementation
    const OpFunc* op_func = OperatorRegistry::instance().get(node.op_type());
    if (!op_func) {
        return false;
    }

    // Create execution context
    OpContext ctx{&node};

    // Execute the operation
    std::vector<Tensor> outputs;
    try {
        outputs = (*op_func)(inputs, ctx);
    } catch (const std::exception&) {
        // Evaluation failed - don't fold this node
        return false;
    }

    // Verify output count matches
    if (outputs.size() != node.outputs().size()) {
        return false;
    }

    // Store outputs as initializers
    for (size_t i = 0; i < outputs.size(); ++i) {
        const std::string& output_name = node.outputs()[i];

        // Clone the output tensor and add as initializer
        graph.add_initializer(output_name, outputs[i].clone());

        // Mark this tensor as constant for subsequent folding
        constant_tensors_.insert(output_name);
    }

    return true;
}

PassResult ConstantFoldingPass::run(Graph& graph) {
    PassResult result;
    constant_tensors_.clear();

    // Get nodes in topological order
    auto nodes = graph.topological_sort();

    std::vector<std::string> nodes_to_remove;

    for (const auto& node : nodes) {
        if (can_fold(graph, *node)) {
            if (fold_node(graph, *node)) {
                nodes_to_remove.push_back(node->name());
                result.stats.constants_folded++;
                result.modified = true;
            }
        }
    }

    // Remove folded nodes
    for (const auto& name : nodes_to_remove) {
        graph.remove_node(name);
        result.stats.nodes_removed++;
    }

    return result;
}

// Register pass with global registry
REGISTER_OPT_PASS(ConstantFoldingPass)

} // namespace opt
} // namespace pyflame_rt
