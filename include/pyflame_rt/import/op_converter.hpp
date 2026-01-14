#pragma once

#include "pyflame_rt/node.hpp"
#include "pyflame_rt/graph.hpp"

#include <any>
#include <functional>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace pyflame_rt {
namespace import {

/// Context provided to operator converters
struct OpConversionContext {
    /// Source operator name (e.g., "Conv" from ONNX)
    std::string source_op;

    /// Source framework identifier (e.g., "onnx", "pytorch")
    std::string framework;

    /// Opset version (for ONNX)
    int opset_version = 0;

    /// Input tensor names
    std::vector<std::string> inputs;

    /// Output tensor names
    std::vector<std::string> outputs;

    /// Source attributes (framework-specific types stored as std::any)
    std::unordered_map<std::string, std::any> attrs;

    /// Optional access to graph for complex conversions (may be nullptr)
    Graph* graph = nullptr;

    /// Node name hint (may be empty)
    std::string node_name;

    /// Helper to get attribute with default
    template<typename T>
    T get_attr(const std::string& name, T default_value = T{}) const {
        auto it = attrs.find(name);
        if (it == attrs.end()) {
            return default_value;
        }
        try {
            return std::any_cast<T>(it->second);
        } catch (const std::bad_any_cast&) {
            return default_value;
        }
    }

    /// Helper to check if attribute exists
    bool has_attr(const std::string& name) const {
        return attrs.find(name) != attrs.end();
    }
};

/// Result of operator conversion
struct OpConversionResult {
    /// Converted node(s) - may produce multiple nodes for decomposed ops
    std::vector<std::shared_ptr<Node>> nodes;

    /// Warning messages generated during conversion
    std::vector<std::string> warnings;

    /// Whether conversion was successful
    bool success = true;

    /// Error message if not successful
    std::string error;

    /// Create a successful result with a single node
    static OpConversionResult single(std::shared_ptr<Node> node) {
        OpConversionResult result;
        result.nodes.push_back(std::move(node));
        return result;
    }

    /// Create a failed result
    static OpConversionResult failure(const std::string& error_msg) {
        OpConversionResult result;
        result.success = false;
        result.error = error_msg;
        return result;
    }

    /// Create a successful result with multiple nodes
    static OpConversionResult multi(std::vector<std::shared_ptr<Node>> nodes) {
        OpConversionResult result;
        result.nodes = std::move(nodes);
        return result;
    }
};

/// Function type for operator converters
using OpConverter = std::function<OpConversionResult(const OpConversionContext&)>;

/// Registry of operator converters
///
/// Maps (framework, op_name) pairs to converter functions.
/// Used during model import to convert source operators to PyFlameRT operators.
class OpConverterRegistry {
public:
    /// Get the singleton instance
    static OpConverterRegistry& instance();

    /// Register a converter for an operator
    /// @param framework Source framework name (e.g., "onnx", "pytorch")
    /// @param op_name Source operator name
    /// @param converter Conversion function
    void register_converter(
        const std::string& framework,
        const std::string& op_name,
        OpConverter converter
    );

    /// Get converter for operator
    /// @return Converter function, or nullptr if not found
    OpConverter get(
        const std::string& framework,
        const std::string& op_name
    ) const;

    /// Check if operator has a converter
    bool has(
        const std::string& framework,
        const std::string& op_name
    ) const;

    /// List all supported operators for a framework
    std::vector<std::string> supported_ops(const std::string& framework) const;

    /// List all supported frameworks
    std::vector<std::string> supported_frameworks() const;

private:
    OpConverterRegistry() = default;
    OpConverterRegistry(const OpConverterRegistry&) = delete;
    OpConverterRegistry& operator=(const OpConverterRegistry&) = delete;

    // Map: framework -> op_name -> converter
    std::unordered_map<std::string,
        std::unordered_map<std::string, OpConverter>> converters_;
};

// ============================================================================
// Helper macros for operator registration
// ============================================================================

/// Register an ONNX operator converter
#define REGISTER_ONNX_OP_CONVERTER(op_name, converter_func) \
    namespace { \
        static const bool _reg_onnx_##op_name = [] { \
            ::pyflame_rt::import::OpConverterRegistry::instance() \
                .register_converter("onnx", #op_name, converter_func); \
            return true; \
        }(); \
    }

/// Register a PyTorch operator converter
#define REGISTER_PYTORCH_OP_CONVERTER(op_name, converter_func) \
    namespace { \
        static const bool _reg_pytorch_##op_name = [] { \
            ::pyflame_rt::import::OpConverterRegistry::instance() \
                .register_converter("pytorch", #op_name, converter_func); \
            return true; \
        }(); \
    }

/// Register a TorchScript operator converter
#define REGISTER_TORCHSCRIPT_OP_CONVERTER(op_name, converter_func) \
    namespace { \
        static const bool _reg_ts_##op_name = [] { \
            ::pyflame_rt::import::OpConverterRegistry::instance() \
                .register_converter("torchscript", #op_name, converter_func); \
            return true; \
        }(); \
    }

// ============================================================================
// Common converter helpers
// ============================================================================

namespace converters {

/// Create a simple 1-to-1 operator mapping
/// Source and target operator have the same name and attributes pass through
inline OpConversionResult direct_mapping(
    const OpConversionContext& ctx,
    const std::string& target_op = ""
) {
    std::string op_type = target_op.empty() ? ctx.source_op : target_op;
    std::string name = ctx.node_name.empty()
        ? (ctx.outputs.empty() ? op_type + "_node" : ctx.outputs[0] + "_" + op_type)
        : ctx.node_name;

    auto node = std::make_shared<Node>(name, op_type, ctx.inputs, ctx.outputs);

    // Copy all attributes
    for (const auto& [attr_name, attr_value] : ctx.attrs) {
        // Try common types
        if (auto* v = std::any_cast<int64_t>(&attr_value)) {
            node->set_attr(attr_name, *v);
        } else if (auto* v = std::any_cast<float>(&attr_value)) {
            node->set_attr(attr_name, *v);
        } else if (auto* v = std::any_cast<std::string>(&attr_value)) {
            node->set_attr(attr_name, *v);
        } else if (auto* v = std::any_cast<std::vector<int64_t>>(&attr_value)) {
            node->set_attr(attr_name, *v);
        } else if (auto* v = std::any_cast<std::vector<float>>(&attr_value)) {
            node->set_attr(attr_name, *v);
        }
        // Other types are silently ignored
    }

    return OpConversionResult::single(std::move(node));
}

/// Create an elementwise/activation operator (single input, single output, no special attrs)
inline OpConversionResult elementwise_op(
    const OpConversionContext& ctx,
    const std::string& target_op = ""
) {
    std::string op_type = target_op.empty() ? ctx.source_op : target_op;
    std::string name = ctx.node_name.empty()
        ? (ctx.outputs.empty() ? op_type + "_node" : ctx.outputs[0] + "_" + op_type)
        : ctx.node_name;

    auto node = std::make_shared<Node>(name, op_type, ctx.inputs, ctx.outputs);
    return OpConversionResult::single(std::move(node));
}

} // namespace converters

} // namespace import
} // namespace pyflame_rt
