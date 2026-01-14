#include "pyflame_rt/import/op_converter.hpp"
#include "pyflame_rt/node.hpp"

namespace pyflame_rt {
namespace import {

// ============================================================================
// Helper Functions
// ============================================================================

namespace {

/// Create a simple node with direct mapping (same op name, copy all attrs)
std::shared_ptr<Node> make_simple_node(
    const OpConversionContext& ctx,
    const std::string& target_op = ""
) {
    std::string op = target_op.empty() ? ctx.source_op : target_op;
    std::string name = ctx.node_name.empty()
        ? (ctx.outputs.empty() ? op + "_node" : ctx.outputs[0])
        : ctx.node_name;

    auto node = std::make_shared<Node>(name, op, ctx.inputs, ctx.outputs);

    // Copy attributes
    for (const auto& [attr_name, attr_value] : ctx.attrs) {
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
    }

    return node;
}

/// Create a simple elementwise node (no attributes needed)
std::shared_ptr<Node> make_elementwise_node(
    const OpConversionContext& ctx,
    const std::string& target_op = ""
) {
    std::string op = target_op.empty() ? ctx.source_op : target_op;
    std::string name = ctx.node_name.empty()
        ? (ctx.outputs.empty() ? op + "_node" : ctx.outputs[0])
        : ctx.node_name;

    return std::make_shared<Node>(name, op, ctx.inputs, ctx.outputs);
}

} // anonymous namespace

// ============================================================================
// Math Operators
// ============================================================================

OpConversionResult convert_add(const OpConversionContext& ctx) {
    return OpConversionResult::single(make_elementwise_node(ctx, "Add"));
}
REGISTER_ONNX_OP_CONVERTER(Add, convert_add)

OpConversionResult convert_sub(const OpConversionContext& ctx) {
    return OpConversionResult::single(make_elementwise_node(ctx, "Sub"));
}
REGISTER_ONNX_OP_CONVERTER(Sub, convert_sub)

OpConversionResult convert_mul(const OpConversionContext& ctx) {
    return OpConversionResult::single(make_elementwise_node(ctx, "Mul"));
}
REGISTER_ONNX_OP_CONVERTER(Mul, convert_mul)

OpConversionResult convert_div(const OpConversionContext& ctx) {
    return OpConversionResult::single(make_elementwise_node(ctx, "Div"));
}
REGISTER_ONNX_OP_CONVERTER(Div, convert_div)

OpConversionResult convert_neg(const OpConversionContext& ctx) {
    return OpConversionResult::single(make_elementwise_node(ctx, "Neg"));
}
REGISTER_ONNX_OP_CONVERTER(Neg, convert_neg)

OpConversionResult convert_abs(const OpConversionContext& ctx) {
    return OpConversionResult::single(make_elementwise_node(ctx, "Abs"));
}
REGISTER_ONNX_OP_CONVERTER(Abs, convert_abs)

OpConversionResult convert_sqrt(const OpConversionContext& ctx) {
    return OpConversionResult::single(make_elementwise_node(ctx, "Sqrt"));
}
REGISTER_ONNX_OP_CONVERTER(Sqrt, convert_sqrt)

OpConversionResult convert_exp(const OpConversionContext& ctx) {
    return OpConversionResult::single(make_elementwise_node(ctx, "Exp"));
}
REGISTER_ONNX_OP_CONVERTER(Exp, convert_exp)

OpConversionResult convert_log(const OpConversionContext& ctx) {
    return OpConversionResult::single(make_elementwise_node(ctx, "Log"));
}
REGISTER_ONNX_OP_CONVERTER(Log, convert_log)

OpConversionResult convert_pow(const OpConversionContext& ctx) {
    return OpConversionResult::single(make_elementwise_node(ctx, "Pow"));
}
REGISTER_ONNX_OP_CONVERTER(Pow, convert_pow)

OpConversionResult convert_matmul(const OpConversionContext& ctx) {
    return OpConversionResult::single(make_elementwise_node(ctx, "MatMul"));
}
REGISTER_ONNX_OP_CONVERTER(MatMul, convert_matmul)

OpConversionResult convert_gemm(const OpConversionContext& ctx) {
    auto node = make_simple_node(ctx, "Gemm");

    // ONNX Gemm has: Y = alpha * A' * B' + beta * C
    // Set defaults if not present
    if (!node->has_attr("alpha")) {
        node->set_attr("alpha", 1.0f);
    }
    if (!node->has_attr("beta")) {
        node->set_attr("beta", 1.0f);
    }
    if (!node->has_attr("transA")) {
        node->set_attr("transA", int64_t{0});
    }
    if (!node->has_attr("transB")) {
        node->set_attr("transB", int64_t{0});
    }

    return OpConversionResult::single(std::move(node));
}
REGISTER_ONNX_OP_CONVERTER(Gemm, convert_gemm)

// ============================================================================
// Activation Functions
// ============================================================================

OpConversionResult convert_relu(const OpConversionContext& ctx) {
    return OpConversionResult::single(make_elementwise_node(ctx, "Relu"));
}
REGISTER_ONNX_OP_CONVERTER(Relu, convert_relu)

OpConversionResult convert_sigmoid(const OpConversionContext& ctx) {
    return OpConversionResult::single(make_elementwise_node(ctx, "Sigmoid"));
}
REGISTER_ONNX_OP_CONVERTER(Sigmoid, convert_sigmoid)

OpConversionResult convert_tanh(const OpConversionContext& ctx) {
    return OpConversionResult::single(make_elementwise_node(ctx, "Tanh"));
}
REGISTER_ONNX_OP_CONVERTER(Tanh, convert_tanh)

OpConversionResult convert_softmax(const OpConversionContext& ctx) {
    auto node = make_simple_node(ctx, "Softmax");

    // ONNX Softmax axis defaults to -1 in opset 13+, 1 in earlier opsets
    if (!node->has_attr("axis")) {
        int64_t default_axis = (ctx.opset_version >= 13) ? -1 : 1;
        node->set_attr("axis", default_axis);
    }

    return OpConversionResult::single(std::move(node));
}
REGISTER_ONNX_OP_CONVERTER(Softmax, convert_softmax)

OpConversionResult convert_leaky_relu(const OpConversionContext& ctx) {
    auto node = make_simple_node(ctx, "LeakyRelu");

    // Default alpha = 0.01
    if (!node->has_attr("alpha")) {
        node->set_attr("alpha", 0.01f);
    }

    return OpConversionResult::single(std::move(node));
}
REGISTER_ONNX_OP_CONVERTER(LeakyRelu, convert_leaky_relu)

OpConversionResult convert_elu(const OpConversionContext& ctx) {
    auto node = make_simple_node(ctx, "Elu");

    // Default alpha = 1.0
    if (!node->has_attr("alpha")) {
        node->set_attr("alpha", 1.0f);
    }

    return OpConversionResult::single(std::move(node));
}
REGISTER_ONNX_OP_CONVERTER(Elu, convert_elu)

OpConversionResult convert_selu(const OpConversionContext& ctx) {
    auto node = make_simple_node(ctx, "Selu");

    // SELU has fixed constants
    node->set_attr("alpha", 1.6732632423543772848170429916717f);
    node->set_attr("gamma", 1.0507009873554804934193349852946f);

    return OpConversionResult::single(std::move(node));
}
REGISTER_ONNX_OP_CONVERTER(Selu, convert_selu)

OpConversionResult convert_softplus(const OpConversionContext& ctx) {
    return OpConversionResult::single(make_elementwise_node(ctx, "Softplus"));
}
REGISTER_ONNX_OP_CONVERTER(Softplus, convert_softplus)

OpConversionResult convert_softsign(const OpConversionContext& ctx) {
    return OpConversionResult::single(make_elementwise_node(ctx, "Softsign"));
}
REGISTER_ONNX_OP_CONVERTER(Softsign, convert_softsign)

OpConversionResult convert_prelu(const OpConversionContext& ctx) {
    // PRelu takes slope as second input
    return OpConversionResult::single(make_elementwise_node(ctx, "PRelu"));
}
REGISTER_ONNX_OP_CONVERTER(PRelu, convert_prelu)

// ============================================================================
// Tensor Operations
// ============================================================================

OpConversionResult convert_reshape(const OpConversionContext& ctx) {
    // ONNX Reshape takes shape as second input in opset 5+
    // We keep the same structure
    return OpConversionResult::single(make_simple_node(ctx, "Reshape"));
}
REGISTER_ONNX_OP_CONVERTER(Reshape, convert_reshape)

OpConversionResult convert_transpose(const OpConversionContext& ctx) {
    auto node = make_simple_node(ctx, "Transpose");

    // perm attribute specifies the permutation
    // If not present, reverse the dimensions
    // We keep the attribute as-is

    return OpConversionResult::single(std::move(node));
}
REGISTER_ONNX_OP_CONVERTER(Transpose, convert_transpose)

OpConversionResult convert_concat(const OpConversionContext& ctx) {
    auto node = make_simple_node(ctx, "Concat");

    // axis is required in ONNX
    if (!node->has_attr("axis")) {
        return OpConversionResult::failure("Concat requires 'axis' attribute");
    }

    return OpConversionResult::single(std::move(node));
}
REGISTER_ONNX_OP_CONVERTER(Concat, convert_concat)

OpConversionResult convert_split(const OpConversionContext& ctx) {
    auto node = make_simple_node(ctx, "Split");

    // Default axis = 0
    if (!node->has_attr("axis")) {
        node->set_attr("axis", int64_t{0});
    }

    return OpConversionResult::single(std::move(node));
}
REGISTER_ONNX_OP_CONVERTER(Split, convert_split)

OpConversionResult convert_squeeze(const OpConversionContext& ctx) {
    auto node = make_simple_node(ctx, "Squeeze");

    // axes specifies which dimensions to squeeze
    // If not present, squeeze all size-1 dimensions

    return OpConversionResult::single(std::move(node));
}
REGISTER_ONNX_OP_CONVERTER(Squeeze, convert_squeeze)

OpConversionResult convert_unsqueeze(const OpConversionContext& ctx) {
    auto node = make_simple_node(ctx, "Unsqueeze");

    // axes is required (which dimensions to add)

    return OpConversionResult::single(std::move(node));
}
REGISTER_ONNX_OP_CONVERTER(Unsqueeze, convert_unsqueeze)

OpConversionResult convert_flatten(const OpConversionContext& ctx) {
    auto node = make_simple_node(ctx, "Flatten");

    // Default axis = 1
    if (!node->has_attr("axis")) {
        node->set_attr("axis", int64_t{1});
    }

    return OpConversionResult::single(std::move(node));
}
REGISTER_ONNX_OP_CONVERTER(Flatten, convert_flatten)

OpConversionResult convert_slice(const OpConversionContext& ctx) {
    // ONNX Slice has different signatures across opsets
    // In opset 10+, starts/ends/axes/steps are inputs
    // In earlier opsets, they are attributes
    auto node = make_simple_node(ctx, "Slice");

    return OpConversionResult::single(std::move(node));
}
REGISTER_ONNX_OP_CONVERTER(Slice, convert_slice)

OpConversionResult convert_gather(const OpConversionContext& ctx) {
    auto node = make_simple_node(ctx, "Gather");

    // Default axis = 0
    if (!node->has_attr("axis")) {
        node->set_attr("axis", int64_t{0});
    }

    return OpConversionResult::single(std::move(node));
}
REGISTER_ONNX_OP_CONVERTER(Gather, convert_gather)

OpConversionResult convert_expand(const OpConversionContext& ctx) {
    // Expand broadcasts tensor to a larger shape
    return OpConversionResult::single(make_simple_node(ctx, "Expand"));
}
REGISTER_ONNX_OP_CONVERTER(Expand, convert_expand)

// ============================================================================
// Reduction Operations
// ============================================================================

OpConversionResult convert_reduce_sum(const OpConversionContext& ctx) {
    auto node = make_simple_node(ctx, "ReduceSum");

    // Default keepdims = 1
    if (!node->has_attr("keepdims")) {
        node->set_attr("keepdims", int64_t{1});
    }

    return OpConversionResult::single(std::move(node));
}
REGISTER_ONNX_OP_CONVERTER(ReduceSum, convert_reduce_sum)

OpConversionResult convert_reduce_mean(const OpConversionContext& ctx) {
    auto node = make_simple_node(ctx, "ReduceMean");

    if (!node->has_attr("keepdims")) {
        node->set_attr("keepdims", int64_t{1});
    }

    return OpConversionResult::single(std::move(node));
}
REGISTER_ONNX_OP_CONVERTER(ReduceMean, convert_reduce_mean)

OpConversionResult convert_reduce_max(const OpConversionContext& ctx) {
    auto node = make_simple_node(ctx, "ReduceMax");

    if (!node->has_attr("keepdims")) {
        node->set_attr("keepdims", int64_t{1});
    }

    return OpConversionResult::single(std::move(node));
}
REGISTER_ONNX_OP_CONVERTER(ReduceMax, convert_reduce_max)

OpConversionResult convert_reduce_min(const OpConversionContext& ctx) {
    auto node = make_simple_node(ctx, "ReduceMin");

    if (!node->has_attr("keepdims")) {
        node->set_attr("keepdims", int64_t{1});
    }

    return OpConversionResult::single(std::move(node));
}
REGISTER_ONNX_OP_CONVERTER(ReduceMin, convert_reduce_min)

OpConversionResult convert_reduce_prod(const OpConversionContext& ctx) {
    auto node = make_simple_node(ctx, "ReduceProd");

    if (!node->has_attr("keepdims")) {
        node->set_attr("keepdims", int64_t{1});
    }

    return OpConversionResult::single(std::move(node));
}
REGISTER_ONNX_OP_CONVERTER(ReduceProd, convert_reduce_prod)

OpConversionResult convert_argmax(const OpConversionContext& ctx) {
    auto node = make_simple_node(ctx, "ArgMax");

    // Default axis = 0, keepdims = 1
    if (!node->has_attr("axis")) {
        node->set_attr("axis", int64_t{0});
    }
    if (!node->has_attr("keepdims")) {
        node->set_attr("keepdims", int64_t{1});
    }

    return OpConversionResult::single(std::move(node));
}
REGISTER_ONNX_OP_CONVERTER(ArgMax, convert_argmax)

OpConversionResult convert_argmin(const OpConversionContext& ctx) {
    auto node = make_simple_node(ctx, "ArgMin");

    if (!node->has_attr("axis")) {
        node->set_attr("axis", int64_t{0});
    }
    if (!node->has_attr("keepdims")) {
        node->set_attr("keepdims", int64_t{1});
    }

    return OpConversionResult::single(std::move(node));
}
REGISTER_ONNX_OP_CONVERTER(ArgMin, convert_argmin)

// ============================================================================
// Neural Network Layers
// ============================================================================

OpConversionResult convert_conv(const OpConversionContext& ctx) {
    auto node = make_simple_node(ctx, "Conv");

    // Set defaults for convolution
    if (!node->has_attr("group")) {
        node->set_attr("group", int64_t{1});
    }
    if (!node->has_attr("dilations")) {
        node->set_attr("dilations", std::vector<int64_t>{1, 1});
    }
    if (!node->has_attr("strides")) {
        node->set_attr("strides", std::vector<int64_t>{1, 1});
    }
    // pads default is auto-computed based on auto_pad if present
    if (!node->has_attr("pads") && !node->has_attr("auto_pad")) {
        node->set_attr("pads", std::vector<int64_t>{0, 0, 0, 0});
    }

    return OpConversionResult::single(std::move(node));
}
REGISTER_ONNX_OP_CONVERTER(Conv, convert_conv)

OpConversionResult convert_maxpool(const OpConversionContext& ctx) {
    auto node = make_simple_node(ctx, "MaxPool");

    // kernel_shape is required
    if (!node->has_attr("kernel_shape")) {
        return OpConversionResult::failure("MaxPool requires 'kernel_shape' attribute");
    }

    // Set defaults
    if (!node->has_attr("strides")) {
        node->set_attr("strides", std::vector<int64_t>{1, 1});
    }
    if (!node->has_attr("pads") && !node->has_attr("auto_pad")) {
        node->set_attr("pads", std::vector<int64_t>{0, 0, 0, 0});
    }

    return OpConversionResult::single(std::move(node));
}
REGISTER_ONNX_OP_CONVERTER(MaxPool, convert_maxpool)

OpConversionResult convert_averagepool(const OpConversionContext& ctx) {
    auto node = make_simple_node(ctx, "AveragePool");

    if (!node->has_attr("kernel_shape")) {
        return OpConversionResult::failure("AveragePool requires 'kernel_shape' attribute");
    }

    if (!node->has_attr("strides")) {
        node->set_attr("strides", std::vector<int64_t>{1, 1});
    }
    if (!node->has_attr("pads") && !node->has_attr("auto_pad")) {
        node->set_attr("pads", std::vector<int64_t>{0, 0, 0, 0});
    }
    if (!node->has_attr("count_include_pad")) {
        node->set_attr("count_include_pad", int64_t{0});
    }

    return OpConversionResult::single(std::move(node));
}
REGISTER_ONNX_OP_CONVERTER(AveragePool, convert_averagepool)

OpConversionResult convert_global_average_pool(const OpConversionContext& ctx) {
    return OpConversionResult::single(make_elementwise_node(ctx, "GlobalAveragePool"));
}
REGISTER_ONNX_OP_CONVERTER(GlobalAveragePool, convert_global_average_pool)

OpConversionResult convert_global_max_pool(const OpConversionContext& ctx) {
    return OpConversionResult::single(make_elementwise_node(ctx, "GlobalMaxPool"));
}
REGISTER_ONNX_OP_CONVERTER(GlobalMaxPool, convert_global_max_pool)

OpConversionResult convert_batch_normalization(const OpConversionContext& ctx) {
    auto node = make_simple_node(ctx, "BatchNorm");

    // Default epsilon
    if (!node->has_attr("epsilon")) {
        node->set_attr("epsilon", 1e-5f);
    }
    if (!node->has_attr("momentum")) {
        node->set_attr("momentum", 0.9f);
    }

    return OpConversionResult::single(std::move(node));
}
REGISTER_ONNX_OP_CONVERTER(BatchNormalization, convert_batch_normalization)

OpConversionResult convert_dropout(const OpConversionContext& ctx) {
    // In inference mode, Dropout is essentially Identity
    // But we keep it as Dropout for graph clarity
    auto node = make_simple_node(ctx, "Dropout");

    if (!node->has_attr("ratio")) {
        node->set_attr("ratio", 0.5f);
    }

    return OpConversionResult::single(std::move(node));
}
REGISTER_ONNX_OP_CONVERTER(Dropout, convert_dropout)

OpConversionResult convert_layer_normalization(const OpConversionContext& ctx) {
    auto node = make_simple_node(ctx, "LayerNorm");

    // Default epsilon and axis
    if (!node->has_attr("epsilon")) {
        node->set_attr("epsilon", 1e-5f);
    }
    if (!node->has_attr("axis")) {
        node->set_attr("axis", int64_t{-1});
    }

    return OpConversionResult::single(std::move(node));
}
REGISTER_ONNX_OP_CONVERTER(LayerNormalization, convert_layer_normalization)

// ============================================================================
// Utility Operators
// ============================================================================

OpConversionResult convert_constant(const OpConversionContext& ctx) {
    // Constant operator embeds a tensor value
    // We convert it to a Constant node
    return OpConversionResult::single(make_simple_node(ctx, "Constant"));
}
REGISTER_ONNX_OP_CONVERTER(Constant, convert_constant)

OpConversionResult convert_identity(const OpConversionContext& ctx) {
    return OpConversionResult::single(make_elementwise_node(ctx, "Identity"));
}
REGISTER_ONNX_OP_CONVERTER(Identity, convert_identity)

OpConversionResult convert_shape(const OpConversionContext& ctx) {
    // Shape operator returns the shape of input tensor
    return OpConversionResult::single(make_simple_node(ctx, "Shape"));
}
REGISTER_ONNX_OP_CONVERTER(Shape, convert_shape)

OpConversionResult convert_cast(const OpConversionContext& ctx) {
    // Cast changes the data type of a tensor
    return OpConversionResult::single(make_simple_node(ctx, "Cast"));
}
REGISTER_ONNX_OP_CONVERTER(Cast, convert_cast)

OpConversionResult convert_clip(const OpConversionContext& ctx) {
    // Clip limits values to [min, max] range
    return OpConversionResult::single(make_simple_node(ctx, "Clip"));
}
REGISTER_ONNX_OP_CONVERTER(Clip, convert_clip)

OpConversionResult convert_pad(const OpConversionContext& ctx) {
    // Pad adds padding to tensor
    return OpConversionResult::single(make_simple_node(ctx, "Pad"));
}
REGISTER_ONNX_OP_CONVERTER(Pad, convert_pad)

// ============================================================================
// Comparison Operators
// ============================================================================

OpConversionResult convert_equal(const OpConversionContext& ctx) {
    return OpConversionResult::single(make_elementwise_node(ctx, "Equal"));
}
REGISTER_ONNX_OP_CONVERTER(Equal, convert_equal)

OpConversionResult convert_greater(const OpConversionContext& ctx) {
    return OpConversionResult::single(make_elementwise_node(ctx, "Greater"));
}
REGISTER_ONNX_OP_CONVERTER(Greater, convert_greater)

OpConversionResult convert_less(const OpConversionContext& ctx) {
    return OpConversionResult::single(make_elementwise_node(ctx, "Less"));
}
REGISTER_ONNX_OP_CONVERTER(Less, convert_less)

OpConversionResult convert_not(const OpConversionContext& ctx) {
    return OpConversionResult::single(make_elementwise_node(ctx, "Not"));
}
REGISTER_ONNX_OP_CONVERTER(Not, convert_not)

OpConversionResult convert_and(const OpConversionContext& ctx) {
    return OpConversionResult::single(make_elementwise_node(ctx, "And"));
}
REGISTER_ONNX_OP_CONVERTER(And, convert_and)

OpConversionResult convert_or(const OpConversionContext& ctx) {
    return OpConversionResult::single(make_elementwise_node(ctx, "Or"));
}
REGISTER_ONNX_OP_CONVERTER(Or, convert_or)

OpConversionResult convert_where(const OpConversionContext& ctx) {
    // Where selects elements based on condition
    return OpConversionResult::single(make_elementwise_node(ctx, "Where"));
}
REGISTER_ONNX_OP_CONVERTER(Where, convert_where)

} // namespace import
} // namespace pyflame_rt
