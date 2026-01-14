#pragma once

#include "pyflame_rt/graph.hpp"
#include "pyflame_rt/types.hpp"
#include "pyflame_rt/errors.hpp"

#include <functional>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

namespace pyflame_rt {

/// Result of shape inference for a graph
struct ShapeInferenceResult {
    /// True if all tensor shapes were successfully inferred
    bool complete = false;

    /// Inferred shapes by tensor name
    std::unordered_map<std::string, std::vector<int64_t>> shapes;

    /// Inferred dtypes by tensor name
    std::unordered_map<std::string, DType> dtypes;

    /// Tensors that couldn't be fully resolved (dynamic dimensions)
    std::vector<std::string> unresolved;

    /// Errors encountered during inference
    std::vector<std::string> errors;

    /// Get shape for a tensor (empty if not inferred)
    std::vector<int64_t> get_shape(const std::string& name) const {
        auto it = shapes.find(name);
        return (it != shapes.end()) ? it->second : std::vector<int64_t>{};
    }

    /// Check if tensor has inferred shape
    bool has_shape(const std::string& name) const {
        return shapes.find(name) != shapes.end();
    }
};

/// Context passed to shape inference functions
struct ShapeContext {
    /// Reference to current node being processed
    const Node& node;

    /// Access to all known shapes
    const std::unordered_map<std::string, std::vector<int64_t>>& shapes;

    /// Access to all known dtypes
    const std::unordered_map<std::string, DType>& dtypes;

    /// Get input shape by index
    std::optional<std::vector<int64_t>> input_shape(size_t idx) const {
        if (idx >= node.inputs().size()) return std::nullopt;
        const std::string& name = node.inputs()[idx];
        auto it = shapes.find(name);
        if (it == shapes.end()) return std::nullopt;
        return it->second;
    }

    /// Get input dtype by index
    std::optional<DType> input_dtype(size_t idx) const {
        if (idx >= node.inputs().size()) return std::nullopt;
        const std::string& name = node.inputs()[idx];
        auto it = dtypes.find(name);
        if (it == dtypes.end()) return std::nullopt;
        return it->second;
    }

    /// Get number of inputs
    size_t num_inputs() const { return node.inputs().size(); }

    /// Get number of outputs
    size_t num_outputs() const { return node.outputs().size(); }
};

/// Shape inference function type
/// @param ctx Context with node info and known shapes
/// @param out_shapes Output shapes (one per output)
/// @param out_dtypes Output dtypes (one per output)
/// @return true if inference succeeded, false otherwise
using ShapeFunc = std::function<bool(
    const ShapeContext& ctx,
    std::vector<std::vector<int64_t>>& out_shapes,
    std::vector<DType>& out_dtypes
)>;

/// Shape inference engine
///
/// Propagates tensor shapes through a computational graph based on
/// operator-specific inference rules.
class ShapeInference {
public:
    /// Construct shape inference engine for a graph
    /// @param graph The graph to analyze
    explicit ShapeInference(Graph& graph);

    /// Set input shapes for the graph
    /// @param shapes Map of input name to concrete shape
    void set_input_shapes(
        const std::unordered_map<std::string, std::vector<int64_t>>& shapes
    );

    /// Set a single input shape
    void set_input_shape(const std::string& name, const std::vector<int64_t>& shape);

    /// Set input dtypes (optional, can also infer from graph inputs)
    void set_input_dtypes(
        const std::unordered_map<std::string, DType>& dtypes
    );

    /// Run shape inference through the graph
    /// @return Inference result with all inferred shapes
    ShapeInferenceResult run();

    /// Get inferred shape for a tensor (after running inference)
    std::optional<std::vector<int64_t>> get_shape(const std::string& name) const;

    /// Get inferred dtype for a tensor (after running inference)
    std::optional<DType> get_dtype(const std::string& name) const;

    /// Register a custom shape function for an operator
    /// @param op_type Operator type name
    /// @param func Shape inference function
    static void register_shape_func(const std::string& op_type, ShapeFunc func);

    /// Check if shape function exists for operator
    static bool has_shape_func(const std::string& op_type);

private:
    Graph& graph_;
    std::unordered_map<std::string, std::vector<int64_t>> shapes_;
    std::unordered_map<std::string, DType> dtypes_;

    /// Infer output shapes for a single node
    bool infer_node(const Node& node);

    /// Initialize shapes from graph inputs and initializers
    void initialize_shapes();
};

/// Shape inference exception
class ShapeInferenceError : public PyFlameRTError {
public:
    ShapeInferenceError(const std::string& tensor,
                        const std::string& reason)
        : PyFlameRTError("Shape inference failed for '" + tensor + "': " + reason)
        , tensor_(tensor) {}

    const std::string& tensor() const { return tensor_; }

private:
    std::string tensor_;
};

// ============================================================================
// Shape inference utility functions
// ============================================================================

namespace shape_utils {

/// Compute broadcast shape for two shapes
/// @return Broadcast result, or empty if not broadcastable
std::vector<int64_t> broadcast_shapes(
    const std::vector<int64_t>& a,
    const std::vector<int64_t>& b
);

/// Check if two shapes are broadcastable
bool are_broadcastable(
    const std::vector<int64_t>& a,
    const std::vector<int64_t>& b
);

/// Compute total number of elements in a shape
int64_t num_elements(const std::vector<int64_t>& shape);

/// Normalize axis (handle negative values)
int64_t normalize_axis(int64_t axis, size_t ndim);

/// Compute output shape for matmul
std::optional<std::vector<int64_t>> matmul_shape(
    const std::vector<int64_t>& a,
    const std::vector<int64_t>& b
);

/// Compute output shape for convolution
std::optional<std::vector<int64_t>> conv_output_shape(
    const std::vector<int64_t>& input,   // N, C, H, W
    const std::vector<int64_t>& weight,  // O, C, kH, kW
    const std::vector<int64_t>& strides,
    const std::vector<int64_t>& pads,
    const std::vector<int64_t>& dilations
);

/// Compute output shape for pooling
std::optional<std::vector<int64_t>> pool_output_shape(
    const std::vector<int64_t>& input,
    const std::vector<int64_t>& kernel,
    const std::vector<int64_t>& strides,
    const std::vector<int64_t>& pads
);

} // namespace shape_utils

} // namespace pyflame_rt
