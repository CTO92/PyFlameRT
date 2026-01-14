#include "pyflame_rt/import/shape_inference.hpp"
#include <algorithm>
#include <numeric>
#include <cmath>

namespace pyflame_rt {

namespace {

// ============================================================================
// Shape Function Registry
// ============================================================================

std::unordered_map<std::string, ShapeFunc>& get_shape_func_registry() {
    static std::unordered_map<std::string, ShapeFunc> registry;
    return registry;
}

// ============================================================================
// Built-in Shape Functions
// ============================================================================

/// Elementwise operations: output shape = broadcast(input shapes)
bool shape_elementwise(const ShapeContext& ctx,
                       std::vector<std::vector<int64_t>>& out_shapes,
                       std::vector<DType>& out_dtypes) {
    auto shape0 = ctx.input_shape(0);
    if (!shape0) return false;

    std::vector<int64_t> result = *shape0;

    // Broadcast with all other inputs
    for (size_t i = 1; i < ctx.num_inputs(); ++i) {
        auto shape_i = ctx.input_shape(i);
        if (!shape_i) return false;

        auto broadcast = shape_utils::broadcast_shapes(result, *shape_i);
        if (broadcast.empty() && (!result.empty() || !shape_i->empty())) {
            return false; // Not broadcastable
        }
        result = broadcast;
    }

    out_shapes.push_back(result);
    if (auto dt = ctx.input_dtype(0)) {
        out_dtypes.push_back(*dt);
    }
    return true;
}

/// Unary operations: output shape = input shape
bool shape_unary(const ShapeContext& ctx,
                 std::vector<std::vector<int64_t>>& out_shapes,
                 std::vector<DType>& out_dtypes) {
    auto input_shape = ctx.input_shape(0);
    if (!input_shape) return false;

    out_shapes.push_back(*input_shape);
    if (auto dt = ctx.input_dtype(0)) {
        out_dtypes.push_back(*dt);
    }
    return true;
}

/// MatMul shape inference
bool shape_matmul(const ShapeContext& ctx,
                  std::vector<std::vector<int64_t>>& out_shapes,
                  std::vector<DType>& out_dtypes) {
    auto shape_a = ctx.input_shape(0);
    auto shape_b = ctx.input_shape(1);
    if (!shape_a || !shape_b) return false;

    auto result = shape_utils::matmul_shape(*shape_a, *shape_b);
    if (!result) return false;

    out_shapes.push_back(*result);
    if (auto dt = ctx.input_dtype(0)) {
        out_dtypes.push_back(*dt);
    }
    return true;
}

/// Gemm shape inference (2D only)
bool shape_gemm(const ShapeContext& ctx,
                std::vector<std::vector<int64_t>>& out_shapes,
                std::vector<DType>& out_dtypes) {
    auto shape_a = ctx.input_shape(0);
    auto shape_b = ctx.input_shape(1);
    if (!shape_a || !shape_b) return false;
    if (shape_a->size() != 2 || shape_b->size() != 2) return false;

    int64_t trans_a = ctx.node.get_attr<int64_t>("transA", 0);
    int64_t trans_b = ctx.node.get_attr<int64_t>("transB", 0);

    int64_t M = trans_a ? (*shape_a)[1] : (*shape_a)[0];
    int64_t N = trans_b ? (*shape_b)[0] : (*shape_b)[1];

    out_shapes.push_back({M, N});
    if (auto dt = ctx.input_dtype(0)) {
        out_dtypes.push_back(*dt);
    }
    return true;
}

/// Reshape shape inference
bool shape_reshape(const ShapeContext& ctx,
                   std::vector<std::vector<int64_t>>& out_shapes,
                   std::vector<DType>& out_dtypes) {
    auto input_shape = ctx.input_shape(0);
    if (!input_shape) return false;

    // Get target shape from attribute
    auto target = ctx.node.get_attr<std::vector<int64_t>>("shape", {});

    // If shape is empty, try to get from second input (ONNX style)
    // This requires runtime knowledge, so we can't infer statically
    if (target.empty()) {
        return false;
    }

    int64_t input_elements = shape_utils::num_elements(*input_shape);

    // Handle special values: -1 (infer) and 0 (copy from input)
    int unknown_idx = -1;
    int64_t known_product = 1;

    for (size_t i = 0; i < target.size(); ++i) {
        if (target[i] == -1) {
            if (unknown_idx >= 0) {
                return false; // Only one -1 allowed
            }
            unknown_idx = static_cast<int>(i);
        } else if (target[i] == 0) {
            // Copy dimension from input
            if (i >= input_shape->size()) {
                return false;
            }
            target[i] = (*input_shape)[i];
            known_product *= target[i];
        } else if (target[i] > 0) {
            known_product *= target[i];
        } else {
            return false; // Invalid dimension
        }
    }

    // Infer unknown dimension
    if (unknown_idx >= 0) {
        if (known_product == 0) {
            return false;
        }
        target[unknown_idx] = input_elements / known_product;
    }

    out_shapes.push_back(target);
    if (auto dt = ctx.input_dtype(0)) {
        out_dtypes.push_back(*dt);
    }
    return true;
}

/// Transpose shape inference
bool shape_transpose(const ShapeContext& ctx,
                     std::vector<std::vector<int64_t>>& out_shapes,
                     std::vector<DType>& out_dtypes) {
    auto input_shape = ctx.input_shape(0);
    if (!input_shape) return false;

    auto perm = ctx.node.get_attr<std::vector<int64_t>>("perm", {});

    // Default permutation: reverse all dimensions
    if (perm.empty()) {
        perm.resize(input_shape->size());
        for (size_t i = 0; i < perm.size(); ++i) {
            perm[i] = static_cast<int64_t>(perm.size() - 1 - i);
        }
    }

    if (perm.size() != input_shape->size()) {
        return false;
    }

    std::vector<int64_t> result(input_shape->size());
    for (size_t i = 0; i < perm.size(); ++i) {
        if (perm[i] < 0 || perm[i] >= static_cast<int64_t>(input_shape->size())) {
            return false;
        }
        result[i] = (*input_shape)[static_cast<size_t>(perm[i])];
    }

    out_shapes.push_back(result);
    if (auto dt = ctx.input_dtype(0)) {
        out_dtypes.push_back(*dt);
    }
    return true;
}

/// Concat shape inference
bool shape_concat(const ShapeContext& ctx,
                  std::vector<std::vector<int64_t>>& out_shapes,
                  std::vector<DType>& out_dtypes) {
    if (ctx.num_inputs() == 0) return false;

    auto first_shape = ctx.input_shape(0);
    if (!first_shape) return false;

    int64_t axis = ctx.node.get_attr<int64_t>("axis", 0);
    axis = shape_utils::normalize_axis(axis, first_shape->size());

    std::vector<int64_t> result = *first_shape;

    // Sum along concatenation axis
    for (size_t i = 1; i < ctx.num_inputs(); ++i) {
        auto shape_i = ctx.input_shape(i);
        if (!shape_i) return false;
        if (shape_i->size() != result.size()) return false;

        // Check non-concat dimensions match
        for (size_t d = 0; d < result.size(); ++d) {
            if (d == static_cast<size_t>(axis)) continue;
            if ((*shape_i)[d] != result[d]) return false;
        }

        result[axis] += (*shape_i)[axis];
    }

    out_shapes.push_back(result);
    if (auto dt = ctx.input_dtype(0)) {
        out_dtypes.push_back(*dt);
    }
    return true;
}

/// Squeeze shape inference
bool shape_squeeze(const ShapeContext& ctx,
                   std::vector<std::vector<int64_t>>& out_shapes,
                   std::vector<DType>& out_dtypes) {
    auto input_shape = ctx.input_shape(0);
    if (!input_shape) return false;

    auto axes = ctx.node.get_attr<std::vector<int64_t>>("axes", {});

    std::vector<int64_t> result;

    if (axes.empty()) {
        // Squeeze all dimensions of size 1
        for (int64_t dim : *input_shape) {
            if (dim != 1) {
                result.push_back(dim);
            }
        }
    } else {
        // Squeeze specified axes
        std::vector<bool> squeeze_dim(input_shape->size(), false);
        for (int64_t axis : axes) {
            axis = shape_utils::normalize_axis(axis, input_shape->size());
            squeeze_dim[axis] = true;
        }

        for (size_t i = 0; i < input_shape->size(); ++i) {
            if (!squeeze_dim[i]) {
                result.push_back((*input_shape)[i]);
            }
        }
    }

    out_shapes.push_back(result);
    if (auto dt = ctx.input_dtype(0)) {
        out_dtypes.push_back(*dt);
    }
    return true;
}

/// Unsqueeze shape inference
bool shape_unsqueeze(const ShapeContext& ctx,
                     std::vector<std::vector<int64_t>>& out_shapes,
                     std::vector<DType>& out_dtypes) {
    auto input_shape = ctx.input_shape(0);
    if (!input_shape) return false;

    auto axes = ctx.node.get_attr<std::vector<int64_t>>("axes", {});
    if (axes.empty()) return false;

    size_t output_rank = input_shape->size() + axes.size();
    std::vector<int64_t> result(output_rank);

    // Normalize and sort axes
    std::vector<int64_t> sorted_axes = axes;
    for (auto& axis : sorted_axes) {
        if (axis < 0) axis += static_cast<int64_t>(output_rank);
    }
    std::sort(sorted_axes.begin(), sorted_axes.end());

    // Place 1s at unsqueeze positions, copy input elsewhere
    size_t input_idx = 0;
    size_t axis_idx = 0;
    for (size_t i = 0; i < output_rank; ++i) {
        if (axis_idx < sorted_axes.size() &&
            static_cast<int64_t>(i) == sorted_axes[axis_idx]) {
            result[i] = 1;
            ++axis_idx;
        } else {
            result[i] = (*input_shape)[input_idx++];
        }
    }

    out_shapes.push_back(result);
    if (auto dt = ctx.input_dtype(0)) {
        out_dtypes.push_back(*dt);
    }
    return true;
}

/// Flatten shape inference
bool shape_flatten(const ShapeContext& ctx,
                   std::vector<std::vector<int64_t>>& out_shapes,
                   std::vector<DType>& out_dtypes) {
    auto input_shape = ctx.input_shape(0);
    if (!input_shape) return false;

    int64_t axis = ctx.node.get_attr<int64_t>("axis", 1);
    axis = shape_utils::normalize_axis(axis, input_shape->size());

    int64_t dim0 = 1, dim1 = 1;
    for (size_t i = 0; i < input_shape->size(); ++i) {
        if (static_cast<int64_t>(i) < axis) {
            dim0 *= (*input_shape)[i];
        } else {
            dim1 *= (*input_shape)[i];
        }
    }

    out_shapes.push_back({dim0, dim1});
    if (auto dt = ctx.input_dtype(0)) {
        out_dtypes.push_back(*dt);
    }
    return true;
}

/// Reduction operations shape inference
bool shape_reduce(const ShapeContext& ctx,
                  std::vector<std::vector<int64_t>>& out_shapes,
                  std::vector<DType>& out_dtypes) {
    auto input_shape = ctx.input_shape(0);
    if (!input_shape) return false;

    auto axes = ctx.node.get_attr<std::vector<int64_t>>("axes", {});
    bool keepdims = ctx.node.get_attr<int64_t>("keepdims", 1) != 0;

    // Default: reduce all axes
    if (axes.empty()) {
        for (size_t i = 0; i < input_shape->size(); ++i) {
            axes.push_back(static_cast<int64_t>(i));
        }
    }

    // Normalize axes
    for (auto& axis : axes) {
        axis = shape_utils::normalize_axis(axis, input_shape->size());
    }

    std::vector<int64_t> result;
    for (size_t i = 0; i < input_shape->size(); ++i) {
        bool is_reduced = std::find(axes.begin(), axes.end(),
                                    static_cast<int64_t>(i)) != axes.end();
        if (is_reduced) {
            if (keepdims) result.push_back(1);
        } else {
            result.push_back((*input_shape)[i]);
        }
    }

    if (result.empty()) result.push_back(1); // Scalar case

    out_shapes.push_back(result);
    if (auto dt = ctx.input_dtype(0)) {
        out_dtypes.push_back(*dt);
    }
    return true;
}

/// ArgMax/ArgMin shape inference
bool shape_argmax(const ShapeContext& ctx,
                  std::vector<std::vector<int64_t>>& out_shapes,
                  std::vector<DType>& out_dtypes) {
    auto input_shape = ctx.input_shape(0);
    if (!input_shape) return false;

    int64_t axis = ctx.node.get_attr<int64_t>("axis", 0);
    bool keepdims = ctx.node.get_attr<int64_t>("keepdims", 1) != 0;

    axis = shape_utils::normalize_axis(axis, input_shape->size());

    std::vector<int64_t> result;
    for (size_t i = 0; i < input_shape->size(); ++i) {
        if (static_cast<int64_t>(i) == axis) {
            if (keepdims) result.push_back(1);
        } else {
            result.push_back((*input_shape)[i]);
        }
    }

    if (result.empty()) result.push_back(1);

    out_shapes.push_back(result);
    out_dtypes.push_back(DType::Int64); // ArgMax always returns int64
    return true;
}

/// Conv2D shape inference
bool shape_conv(const ShapeContext& ctx,
                std::vector<std::vector<int64_t>>& out_shapes,
                std::vector<DType>& out_dtypes) {
    auto input_shape = ctx.input_shape(0);
    auto weight_shape = ctx.input_shape(1);
    if (!input_shape || !weight_shape) return false;

    auto strides = ctx.node.get_attr<std::vector<int64_t>>("strides", {1, 1});
    auto pads = ctx.node.get_attr<std::vector<int64_t>>("pads", {0, 0, 0, 0});
    auto dilations = ctx.node.get_attr<std::vector<int64_t>>("dilations", {1, 1});

    auto result = shape_utils::conv_output_shape(
        *input_shape, *weight_shape, strides, pads, dilations);

    if (!result) return false;

    out_shapes.push_back(*result);
    if (auto dt = ctx.input_dtype(0)) {
        out_dtypes.push_back(*dt);
    }
    return true;
}

/// Pooling shape inference
bool shape_pool(const ShapeContext& ctx,
                std::vector<std::vector<int64_t>>& out_shapes,
                std::vector<DType>& out_dtypes) {
    auto input_shape = ctx.input_shape(0);
    if (!input_shape) return false;

    auto kernel = ctx.node.get_attr<std::vector<int64_t>>("kernel_shape", {2, 2});
    auto strides = ctx.node.get_attr<std::vector<int64_t>>("strides", {1, 1});
    auto pads = ctx.node.get_attr<std::vector<int64_t>>("pads", {0, 0, 0, 0});

    auto result = shape_utils::pool_output_shape(
        *input_shape, kernel, strides, pads);

    if (!result) return false;

    out_shapes.push_back(*result);
    if (auto dt = ctx.input_dtype(0)) {
        out_dtypes.push_back(*dt);
    }
    return true;
}

/// Global pooling shape inference
bool shape_global_pool(const ShapeContext& ctx,
                       std::vector<std::vector<int64_t>>& out_shapes,
                       std::vector<DType>& out_dtypes) {
    auto input_shape = ctx.input_shape(0);
    if (!input_shape || input_shape->size() != 4) return false;

    out_shapes.push_back({(*input_shape)[0], (*input_shape)[1], 1, 1});
    if (auto dt = ctx.input_dtype(0)) {
        out_dtypes.push_back(*dt);
    }
    return true;
}

/// BatchNorm/LayerNorm shape (identity)
bool shape_norm(const ShapeContext& ctx,
                std::vector<std::vector<int64_t>>& out_shapes,
                std::vector<DType>& out_dtypes) {
    return shape_unary(ctx, out_shapes, out_dtypes);
}

/// Softmax shape (identity)
bool shape_softmax(const ShapeContext& ctx,
                   std::vector<std::vector<int64_t>>& out_shapes,
                   std::vector<DType>& out_dtypes) {
    return shape_unary(ctx, out_shapes, out_dtypes);
}

/// Slice shape inference
bool shape_slice(const ShapeContext& ctx,
                 std::vector<std::vector<int64_t>>& out_shapes,
                 std::vector<DType>& out_dtypes) {
    auto input_shape = ctx.input_shape(0);
    if (!input_shape) return false;

    auto starts = ctx.node.get_attr<std::vector<int64_t>>("starts", {});
    auto ends = ctx.node.get_attr<std::vector<int64_t>>("ends", {});
    auto axes = ctx.node.get_attr<std::vector<int64_t>>("axes", {});
    auto steps = ctx.node.get_attr<std::vector<int64_t>>("steps", {});

    if (starts.empty() || ends.empty()) {
        // Can't infer without starts/ends
        return false;
    }

    // Default axes
    if (axes.empty()) {
        axes.resize(starts.size());
        std::iota(axes.begin(), axes.end(), 0);
    }

    // Default steps
    if (steps.empty()) {
        steps.resize(starts.size(), 1);
    }

    std::vector<int64_t> result = *input_shape;

    for (size_t i = 0; i < axes.size(); ++i) {
        int64_t axis = shape_utils::normalize_axis(axes[i], input_shape->size());
        int64_t dim_size = (*input_shape)[axis];

        int64_t start = starts[i];
        int64_t end = ends[i];
        int64_t step = steps[i];

        // Handle negative indices
        if (start < 0) start += dim_size;
        if (end < 0) end += dim_size;

        // Clamp to valid range
        start = std::max(int64_t(0), std::min(start, dim_size));
        end = std::max(int64_t(0), std::min(end, dim_size));

        // Calculate output size
        int64_t length = (step > 0) ? (end - start) : (start - end);
        result[axis] = (length + std::abs(step) - 1) / std::abs(step);
        if (result[axis] < 0) result[axis] = 0;
    }

    out_shapes.push_back(result);
    if (auto dt = ctx.input_dtype(0)) {
        out_dtypes.push_back(*dt);
    }
    return true;
}

// ============================================================================
// Shape Function Registration
// ============================================================================

struct ShapeFuncRegistrar {
    ShapeFuncRegistrar() {
        auto& reg = get_shape_func_registry();

        // Elementwise binary operations
        for (const char* op : {"Add", "Sub", "Mul", "Div", "Pow"}) {
            reg[op] = shape_elementwise;
        }

        // Unary operations
        for (const char* op : {"Relu", "Sigmoid", "Tanh", "LeakyRelu",
                               "Elu", "Selu", "Gelu", "HardSwish",
                               "Sqrt", "Exp", "Log", "Neg", "Abs",
                               "Dropout", "Identity"}) {
            reg[op] = shape_unary;
        }

        // Matrix operations
        reg["MatMul"] = shape_matmul;
        reg["Gemm"] = shape_gemm;

        // Tensor manipulation
        reg["Reshape"] = shape_reshape;
        reg["Transpose"] = shape_transpose;
        reg["Concat"] = shape_concat;
        reg["Squeeze"] = shape_squeeze;
        reg["Unsqueeze"] = shape_unsqueeze;
        reg["Flatten"] = shape_flatten;
        reg["Slice"] = shape_slice;

        // Normalization
        reg["BatchNormalization"] = shape_norm;
        reg["LayerNormalization"] = shape_norm;
        reg["Softmax"] = shape_softmax;

        // Reduction operations
        for (const char* op : {"ReduceSum", "ReduceMean", "ReduceMax",
                               "ReduceMin", "ReduceProd"}) {
            reg[op] = shape_reduce;
        }
        reg["ArgMax"] = shape_argmax;
        reg["ArgMin"] = shape_argmax;

        // Convolution and pooling
        reg["Conv"] = shape_conv;
        reg["MaxPool"] = shape_pool;
        reg["AveragePool"] = shape_pool;
        reg["GlobalAveragePool"] = shape_global_pool;
    }
};

static ShapeFuncRegistrar shape_func_registrar;

} // anonymous namespace

// ============================================================================
// Shape Utility Functions
// ============================================================================

namespace shape_utils {

std::vector<int64_t> broadcast_shapes(
    const std::vector<int64_t>& a,
    const std::vector<int64_t>& b
) {
    size_t max_dims = std::max(a.size(), b.size());
    std::vector<int64_t> result(max_dims);

    for (size_t i = 0; i < max_dims; ++i) {
        int64_t dim_a = (i < a.size()) ? a[a.size() - 1 - i] : 1;
        int64_t dim_b = (i < b.size()) ? b[b.size() - 1 - i] : 1;

        if (dim_a != dim_b && dim_a != 1 && dim_b != 1) {
            return {}; // Not broadcastable
        }
        result[max_dims - 1 - i] = std::max(dim_a, dim_b);
    }

    return result;
}

bool are_broadcastable(
    const std::vector<int64_t>& a,
    const std::vector<int64_t>& b
) {
    size_t max_dims = std::max(a.size(), b.size());

    for (size_t i = 0; i < max_dims; ++i) {
        int64_t dim_a = (i < a.size()) ? a[a.size() - 1 - i] : 1;
        int64_t dim_b = (i < b.size()) ? b[b.size() - 1 - i] : 1;

        if (dim_a != dim_b && dim_a != 1 && dim_b != 1) {
            return false;
        }
    }

    return true;
}

int64_t num_elements(const std::vector<int64_t>& shape) {
    if (shape.empty()) return 0;
    int64_t total = 1;
    for (int64_t dim : shape) {
        total *= dim;
    }
    return total;
}

int64_t normalize_axis(int64_t axis, size_t ndim) {
    if (axis < 0) {
        axis += static_cast<int64_t>(ndim);
    }
    return axis;
}

std::optional<std::vector<int64_t>> matmul_shape(
    const std::vector<int64_t>& a,
    const std::vector<int64_t>& b
) {
    if (a.size() < 2 || b.size() < 2) {
        return std::nullopt;
    }

    int64_t M = a[a.size() - 2];
    int64_t K_a = a[a.size() - 1];
    int64_t K_b = b[b.size() - 2];
    int64_t N = b[b.size() - 1];

    if (K_a != K_b) {
        return std::nullopt;
    }

    // Build output shape with batch dimensions
    std::vector<int64_t> result;
    for (size_t i = 0; i < a.size() - 2; ++i) {
        result.push_back(a[i]);
    }
    result.push_back(M);
    result.push_back(N);

    return result;
}

std::optional<std::vector<int64_t>> conv_output_shape(
    const std::vector<int64_t>& input,
    const std::vector<int64_t>& weight,
    const std::vector<int64_t>& strides,
    const std::vector<int64_t>& pads,
    const std::vector<int64_t>& dilations
) {
    if (input.size() != 4 || weight.size() != 4) {
        return std::nullopt;
    }

    int64_t N = input[0];
    int64_t H = input[2];
    int64_t W = input[3];
    int64_t O = weight[0];
    int64_t kH = weight[2];
    int64_t kW = weight[3];

    int64_t pad_h = (pads.size() >= 2) ? pads[0] + (pads.size() > 2 ? pads[2] : pads[0]) : 0;
    int64_t pad_w = (pads.size() >= 2) ? pads[1] + (pads.size() > 2 ? pads[3] : pads[1]) : 0;
    int64_t stride_h = strides.size() > 0 ? strides[0] : 1;
    int64_t stride_w = strides.size() > 1 ? strides[1] : 1;
    int64_t dil_h = dilations.size() > 0 ? dilations[0] : 1;
    int64_t dil_w = dilations.size() > 1 ? dilations[1] : 1;

    int64_t out_h = (H + pad_h - dil_h * (kH - 1) - 1) / stride_h + 1;
    int64_t out_w = (W + pad_w - dil_w * (kW - 1) - 1) / stride_w + 1;

    return std::vector<int64_t>{N, O, out_h, out_w};
}

std::optional<std::vector<int64_t>> pool_output_shape(
    const std::vector<int64_t>& input,
    const std::vector<int64_t>& kernel,
    const std::vector<int64_t>& strides,
    const std::vector<int64_t>& pads
) {
    if (input.size() != 4) {
        return std::nullopt;
    }

    int64_t N = input[0];
    int64_t C = input[1];
    int64_t H = input[2];
    int64_t W = input[3];

    int64_t kH = kernel.size() > 0 ? kernel[0] : 2;
    int64_t kW = kernel.size() > 1 ? kernel[1] : 2;

    int64_t pad_h = (pads.size() >= 2) ? pads[0] + (pads.size() > 2 ? pads[2] : pads[0]) : 0;
    int64_t pad_w = (pads.size() >= 2) ? pads[1] + (pads.size() > 2 ? pads[3] : pads[1]) : 0;
    int64_t stride_h = strides.size() > 0 ? strides[0] : 1;
    int64_t stride_w = strides.size() > 1 ? strides[1] : 1;

    int64_t out_h = (H + pad_h - kH) / stride_h + 1;
    int64_t out_w = (W + pad_w - kW) / stride_w + 1;

    return std::vector<int64_t>{N, C, out_h, out_w};
}

} // namespace shape_utils

// ============================================================================
// ShapeInference Implementation
// ============================================================================

ShapeInference::ShapeInference(Graph& graph) : graph_(graph) {}

void ShapeInference::set_input_shapes(
    const std::unordered_map<std::string, std::vector<int64_t>>& shapes
) {
    for (const auto& [name, shape] : shapes) {
        shapes_[name] = shape;
    }
}

void ShapeInference::set_input_shape(
    const std::string& name,
    const std::vector<int64_t>& shape
) {
    shapes_[name] = shape;
}

void ShapeInference::set_input_dtypes(
    const std::unordered_map<std::string, DType>& dtypes
) {
    for (const auto& [name, dtype] : dtypes) {
        dtypes_[name] = dtype;
    }
}

void ShapeInference::initialize_shapes() {
    // Initialize from graph inputs
    for (const auto& input : graph_.inputs()) {
        // Only set if not already set by user
        if (shapes_.find(input.name) == shapes_.end()) {
            if (!is_dynamic_shape(input.shape)) {
                std::vector<int64_t> concrete;
                for (const auto& dim : input.shape) {
                    concrete.push_back(dim.value());
                }
                shapes_[input.name] = concrete;
            }
        }
        if (dtypes_.find(input.name) == dtypes_.end()) {
            dtypes_[input.name] = input.dtype;
        }
    }

    // Initialize from initializers (weights)
    for (const auto& name : graph_.initializer_names()) {
        auto tensor = graph_.get_initializer(name);
        if (tensor) {
            shapes_[name] = tensor->shape();
            dtypes_[name] = tensor->dtype();
        }
    }
}

bool ShapeInference::infer_node(const Node& node) {
    auto& registry = get_shape_func_registry();

    auto it = registry.find(node.op_type());
    if (it == registry.end()) {
        return false; // Unknown operator
    }

    ShapeContext ctx{node, shapes_, dtypes_};

    std::vector<std::vector<int64_t>> out_shapes;
    std::vector<DType> out_dtypes;

    if (!it->second(ctx, out_shapes, out_dtypes)) {
        return false;
    }

    // Store results for each output
    for (size_t i = 0; i < node.outputs().size() && i < out_shapes.size(); ++i) {
        shapes_[node.outputs()[i]] = out_shapes[i];
    }
    for (size_t i = 0; i < node.outputs().size() && i < out_dtypes.size(); ++i) {
        dtypes_[node.outputs()[i]] = out_dtypes[i];
    }

    return true;
}

ShapeInferenceResult ShapeInference::run() {
    ShapeInferenceResult result;

    // Initialize from graph structure
    initialize_shapes();

    // Get topologically sorted nodes
    std::vector<Node*> nodes;
    try {
        nodes = graph_.topological_sort();
    } catch (const std::exception& e) {
        result.errors.push_back(std::string("Topological sort failed: ") + e.what());
        return result;
    }

    // Propagate shapes through nodes
    for (const Node* node : nodes) {
        if (!infer_node(*node)) {
            // Record unresolved outputs
            for (const auto& output : node->outputs()) {
                result.unresolved.push_back(output);
            }
        }
    }

    result.shapes = shapes_;
    result.dtypes = dtypes_;
    result.complete = result.unresolved.empty() && result.errors.empty();

    return result;
}

std::optional<std::vector<int64_t>> ShapeInference::get_shape(
    const std::string& name
) const {
    auto it = shapes_.find(name);
    if (it == shapes_.end()) return std::nullopt;
    return it->second;
}

std::optional<DType> ShapeInference::get_dtype(const std::string& name) const {
    auto it = dtypes_.find(name);
    if (it == dtypes_.end()) return std::nullopt;
    return it->second;
}

void ShapeInference::register_shape_func(
    const std::string& op_type,
    ShapeFunc func
) {
    get_shape_func_registry()[op_type] = std::move(func);
}

bool ShapeInference::has_shape_func(const std::string& op_type) {
    return get_shape_func_registry().find(op_type) != get_shape_func_registry().end();
}

} // namespace pyflame_rt
