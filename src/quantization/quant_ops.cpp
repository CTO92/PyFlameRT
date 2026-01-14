#include "pyflame_rt/quantization/quant_ops.hpp"
#include "pyflame_rt/quantization/half_types.hpp"
#include <cmath>
#include <algorithm>
#include <stdexcept>
#include <limits>

namespace pyflame_rt {
namespace quantization {

// ============================================================================
// Core Quantization Operations
// ============================================================================

Tensor quantize_tensor(const Tensor& input, const QuantParams& params) {
    if (input.dtype() != DType::Float32) {
        throw std::invalid_argument(
            "quantize_tensor: input must be Float32, got " +
            dtype_name(input.dtype()));
    }

    if (!params.is_valid()) {
        throw std::invalid_argument(
            "quantize_tensor: invalid quantization parameters");
    }

    Tensor output(input.shape(), params.quantized_dtype);
    const float* src = input.data_ptr<float>();
    int64_t count = input.num_elements();

    auto [qmin, qmax] = get_quant_range(params.quantized_dtype);

    if (params.is_per_tensor()) {
        float scale = params.scales[0];
        int32_t zp = params.zero_points[0];

        // Security fix CRIT-Q02: Validate scale before division
        if (scale <= 0.0f || !std::isfinite(scale)) {
            throw std::invalid_argument(
                "quantize_tensor: invalid scale " + std::to_string(scale));
        }
        float inv_scale = 1.0f / scale;

        if (params.quantized_dtype == DType::Int8) {
            int8_t* dst = output.data_ptr<int8_t>();
            for (int64_t i = 0; i < count; ++i) {
                int32_t q = static_cast<int32_t>(std::round(src[i] * inv_scale)) + zp;
                dst[i] = static_cast<int8_t>(std::clamp(q, qmin, qmax));
            }
        } else {
            uint8_t* dst = output.data_ptr<uint8_t>();
            for (int64_t i = 0; i < count; ++i) {
                int32_t q = static_cast<int32_t>(std::round(src[i] * inv_scale)) + zp;
                dst[i] = static_cast<uint8_t>(std::clamp(q, qmin, qmax));
            }
        }
    } else {
        // Per-channel quantization
        const auto& shape = input.shape();
        int axis = params.channel_axis;

        int64_t outer_size = 1;
        for (int i = 0; i < axis; ++i) {
            outer_size *= shape[i];
        }
        int64_t channel_size = shape[axis];
        int64_t inner_size = 1;
        for (size_t i = axis + 1; i < shape.size(); ++i) {
            inner_size *= shape[i];
        }

        // Security fix HIGH-Q01/HIGH-Q04: Validate channel axis and scales
        if (axis < 0 || axis >= static_cast<int>(shape.size())) {
            throw std::invalid_argument(
                "quantize_tensor: invalid channel_axis " + std::to_string(axis));
        }
        if (params.scales.size() != static_cast<size_t>(channel_size)) {
            throw std::invalid_argument(
                "quantize_tensor: scale count mismatch");
        }

        // Pre-validate all scales
        for (size_t c = 0; c < params.scales.size(); ++c) {
            if (params.scales[c] <= 0.0f || !std::isfinite(params.scales[c])) {
                throw std::invalid_argument(
                    "quantize_tensor: invalid scale for channel " +
                    std::to_string(c));
            }
        }

        if (params.quantized_dtype == DType::Int8) {
            int8_t* dst = output.data_ptr<int8_t>();
            for (int64_t o = 0; o < outer_size; ++o) {
                for (int64_t c = 0; c < channel_size; ++c) {
                    size_t c_idx = static_cast<size_t>(c);
                    float inv_scale = 1.0f / params.scales[c_idx];
                    int32_t zp = params.zero_points[c_idx];
                    int64_t base = (o * channel_size + c) * inner_size;
                    for (int64_t i = 0; i < inner_size; ++i) {
                        int64_t idx = base + i;
                        int32_t q = static_cast<int32_t>(
                            std::round(src[idx] * inv_scale)) + zp;
                        dst[idx] = static_cast<int8_t>(std::clamp(q, qmin, qmax));
                    }
                }
            }
        } else {
            uint8_t* dst = output.data_ptr<uint8_t>();
            for (int64_t o = 0; o < outer_size; ++o) {
                for (int64_t c = 0; c < channel_size; ++c) {
                    size_t c_idx = static_cast<size_t>(c);
                    float inv_scale = 1.0f / params.scales[c_idx];
                    int32_t zp = params.zero_points[c_idx];
                    int64_t base = (o * channel_size + c) * inner_size;
                    for (int64_t i = 0; i < inner_size; ++i) {
                        int64_t idx = base + i;
                        int32_t q = static_cast<int32_t>(
                            std::round(src[idx] * inv_scale)) + zp;
                        dst[idx] = static_cast<uint8_t>(std::clamp(q, qmin, qmax));
                    }
                }
            }
        }
    }

    return output;
}

Tensor dequantize_tensor(const Tensor& input, const QuantParams& params) {
    if (input.dtype() != DType::Int8 && input.dtype() != DType::UInt8) {
        throw std::invalid_argument(
            "dequantize_tensor: input must be Int8 or UInt8, got " +
            dtype_name(input.dtype()));
    }

    Tensor output(input.shape(), DType::Float32);
    float* dst = output.data_ptr<float>();
    int64_t count = input.num_elements();

    if (params.is_per_tensor()) {
        float scale = params.scales[0];
        int32_t zp = params.zero_points[0];

        if (input.dtype() == DType::Int8) {
            const int8_t* src = input.data_ptr<int8_t>();
            for (int64_t i = 0; i < count; ++i) {
                dst[i] = (static_cast<float>(src[i]) - static_cast<float>(zp)) * scale;
            }
        } else {
            const uint8_t* src = input.data_ptr<uint8_t>();
            for (int64_t i = 0; i < count; ++i) {
                dst[i] = (static_cast<float>(src[i]) - static_cast<float>(zp)) * scale;
            }
        }
    } else {
        // Per-channel dequantization
        const auto& shape = input.shape();
        int axis = params.channel_axis;

        // Security fix HIGH-Q04: Validate axis
        if (axis < 0 || axis >= static_cast<int>(shape.size())) {
            throw std::invalid_argument(
                "dequantize_tensor: invalid channel_axis " + std::to_string(axis));
        }

        int64_t outer_size = 1;
        for (int i = 0; i < axis; ++i) {
            outer_size *= shape[i];
        }
        int64_t channel_size = shape[axis];
        int64_t inner_size = 1;
        for (size_t i = axis + 1; i < shape.size(); ++i) {
            inner_size *= shape[i];
        }

        // Security fix HIGH-Q01: Validate scales/zero_points size
        if (params.scales.size() != static_cast<size_t>(channel_size)) {
            throw std::invalid_argument(
                "dequantize_tensor: scale count mismatch");
        }

        if (input.dtype() == DType::Int8) {
            const int8_t* src = input.data_ptr<int8_t>();
            for (int64_t o = 0; o < outer_size; ++o) {
                for (int64_t c = 0; c < channel_size; ++c) {
                    size_t c_idx = static_cast<size_t>(c);
                    float scale = params.scales[c_idx];
                    float zp = static_cast<float>(params.zero_points[c_idx]);
                    int64_t base = (o * channel_size + c) * inner_size;
                    for (int64_t i = 0; i < inner_size; ++i) {
                        int64_t idx = base + i;
                        dst[idx] = (static_cast<float>(src[idx]) - zp) * scale;
                    }
                }
            }
        } else {
            const uint8_t* src = input.data_ptr<uint8_t>();
            for (int64_t o = 0; o < outer_size; ++o) {
                for (int64_t c = 0; c < channel_size; ++c) {
                    size_t c_idx = static_cast<size_t>(c);
                    float scale = params.scales[c_idx];
                    float zp = static_cast<float>(params.zero_points[c_idx]);
                    int64_t base = (o * channel_size + c) * inner_size;
                    for (int64_t i = 0; i < inner_size; ++i) {
                        int64_t idx = base + i;
                        dst[idx] = (static_cast<float>(src[idx]) - zp) * scale;
                    }
                }
            }
        }
    }

    return output;
}

// ============================================================================
// FP16 / BFloat16 Casting
// ============================================================================

Tensor cast_to_fp16(const Tensor& input) {
    if (input.dtype() != DType::Float32) {
        throw std::invalid_argument(
            "cast_to_fp16: input must be Float32, got " +
            dtype_name(input.dtype()));
    }

    Tensor output(input.shape(), DType::Float16);
    const float* src = input.data_ptr<float>();
    Float16* dst = reinterpret_cast<Float16*>(output.data());
    int64_t count = input.num_elements();

    float_to_fp16(src, dst, static_cast<size_t>(count));
    return output;
}

Tensor cast_from_fp16(const Tensor& input) {
    if (input.dtype() != DType::Float16) {
        throw std::invalid_argument(
            "cast_from_fp16: input must be Float16, got " +
            dtype_name(input.dtype()));
    }

    Tensor output(input.shape(), DType::Float32);
    const Float16* src = reinterpret_cast<const Float16*>(input.data());
    float* dst = output.data_ptr<float>();
    int64_t count = input.num_elements();

    fp16_to_float(src, dst, static_cast<size_t>(count));
    return output;
}

Tensor cast_to_bfloat16(const Tensor& input) {
    if (input.dtype() != DType::Float32) {
        throw std::invalid_argument(
            "cast_to_bfloat16: input must be Float32, got " +
            dtype_name(input.dtype()));
    }

    Tensor output(input.shape(), DType::BFloat16);
    const float* src = input.data_ptr<float>();
    BFloat16* dst = reinterpret_cast<BFloat16*>(output.data());
    int64_t count = input.num_elements();

    float_to_bf16(src, dst, static_cast<size_t>(count));
    return output;
}

Tensor cast_from_bfloat16(const Tensor& input) {
    if (input.dtype() != DType::BFloat16) {
        throw std::invalid_argument(
            "cast_from_bfloat16: input must be BFloat16, got " +
            dtype_name(input.dtype()));
    }

    Tensor output(input.shape(), DType::Float32);
    const BFloat16* src = reinterpret_cast<const BFloat16*>(input.data());
    float* dst = output.data_ptr<float>();
    int64_t count = input.num_elements();

    bf16_to_float(src, dst, static_cast<size_t>(count));
    return output;
}

// ============================================================================
// Dynamic Quantization Helpers
// ============================================================================

QuantParams compute_dynamic_params(const Tensor& tensor,
                                    DType target_dtype,
                                    bool symmetric) {
    if (tensor.dtype() != DType::Float32) {
        throw std::invalid_argument(
            "compute_dynamic_params: input must be Float32, got " +
            dtype_name(tensor.dtype()));
    }

    const float* data = tensor.data_ptr<float>();
    int64_t count = tensor.num_elements();

    if (count == 0) {
        return QuantParams::per_tensor(1.0f, 0, target_dtype);
    }

    // Find min/max
    float min_val = std::numeric_limits<float>::max();
    float max_val = std::numeric_limits<float>::lowest();

    for (int64_t i = 0; i < count; ++i) {
        float val = data[i];
        if (std::isfinite(val)) {
            min_val = std::min(min_val, val);
            max_val = std::max(max_val, val);
        }
    }

    // Handle all NaN/Inf case
    if (min_val > max_val) {
        return QuantParams::per_tensor(1.0f, 0, target_dtype);
    }

    return QuantParams::compute_from_minmax(min_val, max_val, target_dtype, symmetric);
}

QuantParams compute_dynamic_params_per_channel(const Tensor& tensor,
                                                int channel_axis,
                                                DType target_dtype,
                                                bool symmetric) {
    if (tensor.dtype() != DType::Float32) {
        throw std::invalid_argument(
            "compute_dynamic_params_per_channel: input must be Float32");
    }

    const auto& shape = tensor.shape();

    if (channel_axis < 0 || channel_axis >= static_cast<int>(shape.size())) {
        throw std::invalid_argument(
            "compute_dynamic_params_per_channel: invalid channel_axis " +
            std::to_string(channel_axis));
    }

    int64_t num_channels = shape[channel_axis];
    std::vector<float> min_vals(num_channels, std::numeric_limits<float>::max());
    std::vector<float> max_vals(num_channels, std::numeric_limits<float>::lowest());

    const float* data = tensor.data_ptr<float>();

    // Calculate strides
    int64_t outer_size = 1;
    for (int i = 0; i < channel_axis; ++i) {
        outer_size *= shape[i];
    }
    int64_t inner_size = 1;
    for (size_t i = channel_axis + 1; i < shape.size(); ++i) {
        inner_size *= shape[i];
    }

    // Find per-channel min/max
    for (int64_t o = 0; o < outer_size; ++o) {
        for (int64_t c = 0; c < num_channels; ++c) {
            int64_t base = (o * num_channels + c) * inner_size;
            for (int64_t i = 0; i < inner_size; ++i) {
                float val = data[base + i];
                if (std::isfinite(val)) {
                    min_vals[c] = std::min(min_vals[c], val);
                    max_vals[c] = std::max(max_vals[c], val);
                }
            }
        }
    }

    // Handle empty channels
    for (int64_t c = 0; c < num_channels; ++c) {
        if (min_vals[c] > max_vals[c]) {
            min_vals[c] = 0.0f;
            max_vals[c] = 0.0f;
        }
    }

    return QuantParams::compute_per_channel(min_vals, max_vals, channel_axis,
                                             target_dtype, symmetric);
}

// ============================================================================
// Quantized Arithmetic Operations
// ============================================================================

Tensor quantized_matmul(const Tensor& a, const QuantParams& a_params,
                        const Tensor& b, const QuantParams& b_params,
                        const QuantParams* output_params) {
    // Validate inputs
    if (a.dtype() != DType::Int8 && a.dtype() != DType::UInt8) {
        throw std::invalid_argument(
            "quantized_matmul: tensor A must be Int8 or UInt8");
    }
    if (b.dtype() != DType::Int8 && b.dtype() != DType::UInt8) {
        throw std::invalid_argument(
            "quantized_matmul: tensor B must be Int8 or UInt8");
    }

    if (a.ndim() < 2 || b.ndim() < 2) {
        throw std::invalid_argument(
            "quantized_matmul: inputs must have at least 2 dimensions");
    }

    int64_t M = a.shape()[a.ndim() - 2];
    int64_t K_a = a.shape()[a.ndim() - 1];
    int64_t K_b = b.shape()[b.ndim() - 2];
    int64_t N = b.shape()[b.ndim() - 1];

    if (K_a != K_b) {
        throw std::invalid_argument(
            "quantized_matmul: inner dimensions must match");
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

    // Combined scale for output
    float a_scale = a_params.scales[0];
    float b_scale = b_params.scales[0];
    int32_t a_zp = a_params.zero_points[0];
    int32_t b_zp = b_params.zero_points[0];
    float output_scale = a_scale * b_scale;

    // Calculate batch size
    int64_t batch = 1;
    for (size_t i = 0; i < a.ndim() - 2; ++i) {
        batch *= a.shape()[i];
    }

    // Security fix CRIT-Q03: Use int64_t accumulator to prevent overflow
    // With int8 values in [-128, 127], each product is max ~16384
    // For large K (e.g. K=100000), int32_t accumulator could overflow
    if (a.dtype() == DType::Int8 && b.dtype() == DType::Int8) {
        const int8_t* a_data = a.data_ptr<int8_t>();
        const int8_t* b_data = b.data_ptr<int8_t>();

        for (int64_t n = 0; n < batch; ++n) {
            const int8_t* a_batch = a_data + n * M * K;
            const int8_t* b_batch = b_data + n * K * N;
            float* out_batch = out_data + n * M * N;

            for (int64_t i = 0; i < M; ++i) {
                for (int64_t j = 0; j < N; ++j) {
                    // Security fix CRIT-Q03: int64_t accumulator
                    int64_t acc = 0;
                    for (int64_t k = 0; k < K; ++k) {
                        int32_t a_val = static_cast<int32_t>(a_batch[i * K + k]) - a_zp;
                        int32_t b_val = static_cast<int32_t>(b_batch[k * N + j]) - b_zp;
                        acc += static_cast<int64_t>(a_val) * static_cast<int64_t>(b_val);
                    }
                    out_batch[i * N + j] = static_cast<float>(acc) * output_scale;
                }
            }
        }
    } else {
        // Mixed dtype case - convert to int32 and compute
        for (int64_t n = 0; n < batch; ++n) {
            for (int64_t i = 0; i < M; ++i) {
                for (int64_t j = 0; j < N; ++j) {
                    // Security fix CRIT-Q03: int64_t accumulator
                    int64_t acc = 0;
                    for (int64_t k = 0; k < K; ++k) {
                        int32_t a_val, b_val;
                        int64_t a_idx = n * M * K + i * K + k;
                        int64_t b_idx = n * K * N + k * N + j;

                        if (a.dtype() == DType::Int8) {
                            a_val = static_cast<int32_t>(a.data_ptr<int8_t>()[a_idx]) - a_zp;
                        } else {
                            a_val = static_cast<int32_t>(a.data_ptr<uint8_t>()[a_idx]) - a_zp;
                        }

                        if (b.dtype() == DType::Int8) {
                            b_val = static_cast<int32_t>(b.data_ptr<int8_t>()[b_idx]) - b_zp;
                        } else {
                            b_val = static_cast<int32_t>(b.data_ptr<uint8_t>()[b_idx]) - b_zp;
                        }

                        acc += static_cast<int64_t>(a_val) * static_cast<int64_t>(b_val);
                    }
                    out_data[n * M * N + i * N + j] = static_cast<float>(acc) * output_scale;
                }
            }
        }
    }

    // If output params provided, requantize the output
    if (output_params) {
        return quantize_tensor(result, *output_params);
    }

    return result;
}

Tensor quantized_add(const Tensor& a, const QuantParams& a_params,
                     const Tensor& b, const QuantParams& b_params,
                     const QuantParams* output_params) {
    // For simplicity, dequantize, add, and optionally requantize
    Tensor a_float = dequantize_tensor(a, a_params);
    Tensor b_float = dequantize_tensor(b, b_params);

    // Element-wise add (assumes same shape)
    if (a_float.shape() != b_float.shape()) {
        throw std::invalid_argument(
            "quantized_add: tensors must have the same shape");
    }

    Tensor result(a_float.shape(), DType::Float32);
    const float* a_data = a_float.data_ptr<float>();
    const float* b_data = b_float.data_ptr<float>();
    float* out_data = result.data_ptr<float>();
    int64_t count = result.num_elements();

    for (int64_t i = 0; i < count; ++i) {
        out_data[i] = a_data[i] + b_data[i];
    }

    if (output_params) {
        return quantize_tensor(result, *output_params);
    }

    return result;
}

// ============================================================================
// Utility Functions
// ============================================================================

bool can_quantize(const Tensor& tensor) {
    return tensor.dtype() == DType::Float32 && tensor.is_valid();
}

QuantParams recommend_params(const Tensor& tensor,
                              DType target_dtype,
                              QuantGranularity granularity,
                              bool symmetric) {
    if (granularity == QuantGranularity::PerTensor) {
        return compute_dynamic_params(tensor, target_dtype, symmetric);
    } else {
        // Default to axis 0 for per-channel
        return compute_dynamic_params_per_channel(tensor, 0, target_dtype, symmetric);
    }
}

float compute_quant_error(const Tensor& original,
                          const Tensor& quantized,
                          const QuantParams& params) {
    // Security fix MED-Q02: Validate inputs
    if (!original.is_valid() || !quantized.is_valid()) {
        throw std::invalid_argument(
            "compute_quant_error: invalid tensor");
    }

    if (original.dtype() != DType::Float32) {
        throw std::invalid_argument(
            "compute_quant_error: original must be Float32");
    }

    // Security fix MED-Q05: Validate shapes match
    if (original.shape() != quantized.shape()) {
        throw std::invalid_argument(
            "compute_quant_error: tensor shapes must match");
    }

    Tensor dequantized = dequantize_tensor(quantized, params);

    const float* orig = original.data_ptr<float>();
    const float* deq = dequantized.data_ptr<float>();
    int64_t count = original.num_elements();

    // Security fix MED-Q02: Handle empty tensors gracefully
    if (count == 0) return 0.0f;

    // Security fix MED-Q01: Use double for accumulation to avoid precision loss
    double mse = 0.0;
    for (int64_t i = 0; i < count; ++i) {
        double diff = static_cast<double>(orig[i]) - static_cast<double>(deq[i]);
        mse += diff * diff;
    }

    return static_cast<float>(mse / static_cast<double>(count));
}

} // namespace quantization
} // namespace pyflame_rt
