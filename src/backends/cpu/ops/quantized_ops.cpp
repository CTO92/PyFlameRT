#include "pyflame_rt/registry.hpp"
#include "pyflame_rt/tensor.hpp"
#include "pyflame_rt/quantization/quant_params.hpp"
#include "pyflame_rt/quantization/quant_ops.hpp"
#include "pyflame_rt/quantization/half_types.hpp"
#include <stdexcept>
#include <cmath>
#include <algorithm>

namespace pyflame_rt {
namespace ops {

using namespace quantization;

namespace {

// ============================================================================
// Validation Helpers
// ============================================================================

inline void validate_input_count(const std::vector<const Tensor*>& inputs,
                                  size_t min_count, const char* op_name) {
    if (inputs.size() < min_count) {
        throw std::invalid_argument(
            std::string(op_name) + " requires at least " +
            std::to_string(min_count) + " inputs, got " +
            std::to_string(inputs.size()));
    }
}

// ============================================================================
// Quantize / Dequantize Operations
// ============================================================================

std::vector<Tensor> cpu_quantize(
    const std::vector<const Tensor*>& inputs,
    const OpContext& ctx)
{
    validate_input_count(inputs, 1, "Quantize");

    const Tensor& x = *inputs[0];
    float scale = ctx.node->get_attr<float>("scale", 1.0f);
    int64_t zero_point = ctx.node->get_attr<int64_t>("zero_point", 0);
    std::string dtype_str = ctx.node->get_attr<std::string>("dtype", "int8");

    DType target_dtype = DType::Int8;
    if (dtype_str == "uint8") {
        target_dtype = DType::UInt8;
    }

    QuantParams params = QuantParams::per_tensor(
        scale, static_cast<int32_t>(zero_point), target_dtype);

    return {quantize_tensor(x, params)};
}

std::vector<Tensor> cpu_dequantize(
    const std::vector<const Tensor*>& inputs,
    const OpContext& ctx)
{
    validate_input_count(inputs, 1, "Dequantize");

    const Tensor& x = *inputs[0];
    float scale = ctx.node->get_attr<float>("scale", 1.0f);
    int64_t zero_point = ctx.node->get_attr<int64_t>("zero_point", 0);

    QuantParams params = QuantParams::per_tensor(
        scale, static_cast<int32_t>(zero_point), x.dtype());

    return {dequantize_tensor(x, params)};
}

// ============================================================================
// Quantized Matrix Multiplication
// ============================================================================

std::vector<Tensor> cpu_quantized_matmul(
    const std::vector<const Tensor*>& inputs,
    const OpContext& ctx)
{
    validate_input_count(inputs, 2, "QuantizedMatMul");

    const Tensor& a = *inputs[0];
    const Tensor& b = *inputs[1];

    // Get quantization parameters from attributes
    float a_scale = ctx.node->get_attr<float>("a_scale", 1.0f);
    float b_scale = ctx.node->get_attr<float>("b_scale", 1.0f);
    int64_t a_zp = ctx.node->get_attr<int64_t>("a_zero_point", 0);
    int64_t b_zp = ctx.node->get_attr<int64_t>("b_zero_point", 0);

    // Validate inputs
    if (a.dtype() != DType::Int8 && a.dtype() != DType::UInt8) {
        throw std::invalid_argument(
            "QuantizedMatMul: input A must be Int8 or UInt8");
    }
    if (b.dtype() != DType::Int8 && b.dtype() != DType::UInt8) {
        throw std::invalid_argument(
            "QuantizedMatMul: input B must be Int8 or UInt8");
    }

    if (a.ndim() < 2 || b.ndim() < 2) {
        throw std::invalid_argument(
            "QuantizedMatMul: inputs must have at least 2 dimensions");
    }

    int64_t M = a.shape()[a.ndim() - 2];
    int64_t K_a = a.shape()[a.ndim() - 1];
    int64_t K_b = b.shape()[b.ndim() - 2];
    int64_t N = b.shape()[b.ndim() - 1];

    if (K_a != K_b) {
        throw std::invalid_argument(
            "QuantizedMatMul: inner dimensions must match (got " +
            std::to_string(K_a) + " and " + std::to_string(K_b) + ")");
    }
    int64_t K = K_a;

    // Build output shape
    std::vector<int64_t> out_shape;
    for (size_t i = 0; i < a.ndim() - 2; ++i) {
        out_shape.push_back(a.shape()[i]);
    }
    out_shape.push_back(M);
    out_shape.push_back(N);

    // Output is float32 (dequantized)
    Tensor result(out_shape, DType::Float32);
    result.zero();

    float* out_data = result.data_ptr<float>();

    // Combined output scale
    float output_scale = a_scale * b_scale;

    // Calculate batch size
    int64_t batch = 1;
    for (size_t i = 0; i < a.ndim() - 2; ++i) {
        batch *= a.shape()[i];
    }

    // Perform batched matrix multiplication with int32 accumulation
    const int8_t* a_data = a.data_ptr<int8_t>();
    const int8_t* b_data = b.data_ptr<int8_t>();

    for (int64_t n = 0; n < batch; ++n) {
        const int8_t* a_batch = a_data + n * M * K;
        const int8_t* b_batch = b_data + n * K * N;
        float* out_batch = out_data + n * M * N;

        for (int64_t i = 0; i < M; ++i) {
            for (int64_t j = 0; j < N; ++j) {
                int32_t acc = 0;
                for (int64_t k = 0; k < K; ++k) {
                    int32_t a_val = static_cast<int32_t>(a_batch[i * K + k])
                                    - static_cast<int32_t>(a_zp);
                    int32_t b_val = static_cast<int32_t>(b_batch[k * N + j])
                                    - static_cast<int32_t>(b_zp);
                    acc += a_val * b_val;
                }
                out_batch[i * N + j] = static_cast<float>(acc) * output_scale;
            }
        }
    }

    return {std::move(result)};
}

// ============================================================================
// Quantized Element-wise Add
// ============================================================================

std::vector<Tensor> cpu_quantized_add(
    const std::vector<const Tensor*>& inputs,
    const OpContext& ctx)
{
    validate_input_count(inputs, 2, "QuantizedAdd");

    const Tensor& a = *inputs[0];
    const Tensor& b = *inputs[1];

    float a_scale = ctx.node->get_attr<float>("a_scale", 1.0f);
    float b_scale = ctx.node->get_attr<float>("b_scale", 1.0f);
    int64_t a_zp = ctx.node->get_attr<int64_t>("a_zero_point", 0);
    int64_t b_zp = ctx.node->get_attr<int64_t>("b_zero_point", 0);

    // Dequantize, add, return float
    QuantParams a_params = QuantParams::per_tensor(a_scale,
        static_cast<int32_t>(a_zp), a.dtype());
    QuantParams b_params = QuantParams::per_tensor(b_scale,
        static_cast<int32_t>(b_zp), b.dtype());

    Tensor a_float = dequantize_tensor(a, a_params);
    Tensor b_float = dequantize_tensor(b, b_params);

    // Perform addition
    Tensor result(a_float.shape(), DType::Float32);
    const float* a_data = a_float.data_ptr<float>();
    const float* b_data = b_float.data_ptr<float>();
    float* out_data = result.data_ptr<float>();
    int64_t count = result.num_elements();

    for (int64_t i = 0; i < count; ++i) {
        out_data[i] = a_data[i] + b_data[i];
    }

    return {std::move(result)};
}

// ============================================================================
// Cast Operations (FP16 / BFloat16)
// ============================================================================

std::vector<Tensor> cpu_cast_to_fp16(
    const std::vector<const Tensor*>& inputs,
    const OpContext& /*ctx*/)
{
    validate_input_count(inputs, 1, "CastToFP16");
    return {cast_to_fp16(*inputs[0])};
}

std::vector<Tensor> cpu_cast_from_fp16(
    const std::vector<const Tensor*>& inputs,
    const OpContext& /*ctx*/)
{
    validate_input_count(inputs, 1, "CastFromFP16");
    return {cast_from_fp16(*inputs[0])};
}

std::vector<Tensor> cpu_cast_to_bf16(
    const std::vector<const Tensor*>& inputs,
    const OpContext& /*ctx*/)
{
    validate_input_count(inputs, 1, "CastToBF16");
    return {cast_to_bfloat16(*inputs[0])};
}

std::vector<Tensor> cpu_cast_from_bf16(
    const std::vector<const Tensor*>& inputs,
    const OpContext& /*ctx*/)
{
    validate_input_count(inputs, 1, "CastFromBF16");
    return {cast_from_bfloat16(*inputs[0])};
}

// ============================================================================
// DynamicQuantizeLinear (ONNX compatible)
// Computes quantization params and quantizes in one op
// ============================================================================

std::vector<Tensor> cpu_dynamic_quantize_linear(
    const std::vector<const Tensor*>& inputs,
    const OpContext& /*ctx*/)
{
    validate_input_count(inputs, 1, "DynamicQuantizeLinear");

    const Tensor& x = *inputs[0];

    if (x.dtype() != DType::Float32) {
        throw std::invalid_argument(
            "DynamicQuantizeLinear: input must be Float32");
    }

    // Compute dynamic params (uint8, asymmetric for ONNX compatibility)
    QuantParams params = compute_dynamic_params(x, DType::UInt8, false);

    // Quantize
    Tensor y = quantize_tensor(x, params);

    // Return: y (quantized), scale, zero_point
    Tensor scale_tensor({1}, DType::Float32);
    scale_tensor.data_ptr<float>()[0] = params.scales[0];

    Tensor zp_tensor({1}, DType::UInt8);
    zp_tensor.data_ptr<uint8_t>()[0] = static_cast<uint8_t>(params.zero_points[0]);

    return {std::move(y), std::move(scale_tensor), std::move(zp_tensor)};
}

// ============================================================================
// QuantizeLinear (ONNX compatible)
// ============================================================================

std::vector<Tensor> cpu_quantize_linear(
    const std::vector<const Tensor*>& inputs,
    const OpContext& ctx)
{
    validate_input_count(inputs, 2, "QuantizeLinear");

    const Tensor& x = *inputs[0];
    const Tensor& scale = *inputs[1];

    // Zero point is optional
    int32_t zp = 0;
    if (inputs.size() > 2 && inputs[2] != nullptr) {
        const Tensor& zp_tensor = *inputs[2];
        if (zp_tensor.dtype() == DType::Int8) {
            zp = static_cast<int32_t>(zp_tensor.data_ptr<int8_t>()[0]);
        } else if (zp_tensor.dtype() == DType::UInt8) {
            zp = static_cast<int32_t>(zp_tensor.data_ptr<uint8_t>()[0]);
        }
    }

    float scale_val = scale.data_ptr<float>()[0];

    // Determine output dtype
    DType out_dtype = DType::Int8;
    std::string dtype_str = ctx.node->get_attr<std::string>("output_dtype", "int8");
    if (dtype_str == "uint8") {
        out_dtype = DType::UInt8;
    }

    QuantParams params = QuantParams::per_tensor(scale_val, zp, out_dtype);
    return {quantize_tensor(x, params)};
}

// ============================================================================
// DequantizeLinear (ONNX compatible)
// ============================================================================

std::vector<Tensor> cpu_dequantize_linear(
    const std::vector<const Tensor*>& inputs,
    const OpContext& /*ctx*/)
{
    validate_input_count(inputs, 2, "DequantizeLinear");

    const Tensor& x = *inputs[0];
    const Tensor& scale = *inputs[1];

    // Zero point is optional
    int32_t zp = 0;
    if (inputs.size() > 2 && inputs[2] != nullptr) {
        const Tensor& zp_tensor = *inputs[2];
        if (zp_tensor.dtype() == DType::Int8) {
            zp = static_cast<int32_t>(zp_tensor.data_ptr<int8_t>()[0]);
        } else if (zp_tensor.dtype() == DType::UInt8) {
            zp = static_cast<int32_t>(zp_tensor.data_ptr<uint8_t>()[0]);
        }
    }

    float scale_val = scale.data_ptr<float>()[0];

    QuantParams params = QuantParams::per_tensor(scale_val, zp, x.dtype());
    return {dequantize_tensor(x, params)};
}

// ============================================================================
// Registration
// ============================================================================

struct QuantizedOpsRegistrar {
    QuantizedOpsRegistrar() {
        auto& reg = OperatorRegistry::instance();

        // Core quantization ops
        reg.register_op("Quantize", cpu_quantize);
        reg.register_op("Dequantize", cpu_dequantize);

        // ONNX-compatible ops
        reg.register_op("QuantizeLinear", cpu_quantize_linear);
        reg.register_op("DequantizeLinear", cpu_dequantize_linear);
        reg.register_op("DynamicQuantizeLinear", cpu_dynamic_quantize_linear);

        // Quantized compute ops
        reg.register_op("QuantizedMatMul", cpu_quantized_matmul);
        reg.register_op("QuantizedAdd", cpu_quantized_add);

        // Cast ops
        reg.register_op("CastToFP16", cpu_cast_to_fp16);
        reg.register_op("CastFromFP16", cpu_cast_from_fp16);
        reg.register_op("CastToBF16", cpu_cast_to_bf16);
        reg.register_op("CastFromBF16", cpu_cast_from_bf16);
    }
};

static QuantizedOpsRegistrar quantized_ops_registrar;

} // anonymous namespace
} // namespace ops
} // namespace pyflame_rt
