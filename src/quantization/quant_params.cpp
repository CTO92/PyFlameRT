#include "pyflame_rt/quantization/quant_params.hpp"
#include <cmath>
#include <algorithm>
#include <limits>
#include <stdexcept>

namespace pyflame_rt {
namespace quantization {

// ============================================================================
// QuantParams Implementation
// ============================================================================

QuantParams QuantParams::compute_from_minmax(
    float min_val, float max_val, DType dtype, bool symmetric)
{
    auto [qmin, qmax] = get_quant_range(dtype);

    float scale;
    int32_t zero_point;

    // Handle edge case where min == max
    if (std::abs(max_val - min_val) < 1e-8f) {
        scale = 1.0f;
        zero_point = 0;
        return QuantParams::per_tensor(scale, zero_point, dtype);
    }

    if (symmetric) {
        // Symmetric quantization: zero_point = 0 for signed, 128 for unsigned
        float abs_max = std::max(std::abs(min_val), std::abs(max_val));

        if (dtype == DType::Int8) {
            // Range is [-127, 127] for symmetric (not using -128 for symmetry)
            scale = abs_max / 127.0f;
            zero_point = 0;
        } else {
            // UInt8: map [-abs_max, abs_max] to [0, 255], zero_point = 128
            scale = (2.0f * abs_max) / 255.0f;
            zero_point = 128;
        }
    } else {
        // Asymmetric quantization
        scale = (max_val - min_val) / static_cast<float>(qmax - qmin);
        zero_point = qmin - static_cast<int32_t>(std::round(min_val / scale));
        zero_point = std::clamp(zero_point, qmin, qmax);
    }

    // Ensure scale is not zero
    if (scale < 1e-10f) {
        scale = 1e-10f;
    }

    return QuantParams::per_tensor(scale, zero_point, dtype);
}

QuantParams QuantParams::compute_per_channel(
    const std::vector<float>& min_vals,
    const std::vector<float>& max_vals,
    int axis, DType dtype, bool symmetric)
{
    if (min_vals.size() != max_vals.size()) {
        throw std::invalid_argument(
            "min_vals and max_vals must have the same size");
    }

    size_t num_channels = min_vals.size();
    std::vector<float> scales(num_channels);
    std::vector<int32_t> zero_points(num_channels);

    auto [qmin, qmax] = get_quant_range(dtype);

    for (size_t i = 0; i < num_channels; ++i) {
        float min_val = min_vals[i];
        float max_val = max_vals[i];

        // Handle edge case
        if (std::abs(max_val - min_val) < 1e-8f) {
            scales[i] = 1.0f;
            zero_points[i] = 0;
            continue;
        }

        if (symmetric) {
            float abs_max = std::max(std::abs(min_val), std::abs(max_val));
            if (dtype == DType::Int8) {
                scales[i] = abs_max / 127.0f;
                zero_points[i] = 0;
            } else {
                scales[i] = (2.0f * abs_max) / 255.0f;
                zero_points[i] = 128;
            }
        } else {
            scales[i] = (max_val - min_val) / static_cast<float>(qmax - qmin);
            zero_points[i] = qmin - static_cast<int32_t>(std::round(min_val / scales[i]));
            zero_points[i] = std::clamp(zero_points[i], qmin, qmax);
        }

        // Ensure scale is not zero
        if (scales[i] < 1e-10f) {
            scales[i] = 1e-10f;
        }
    }

    return QuantParams::per_channel(scales, zero_points, axis, dtype);
}

// ============================================================================
// QuantizedTensor Implementation
// ============================================================================

QuantizedTensor::QuantizedTensor(const Tensor& float_tensor,
                                 const QuantParams& params)
    : params_(params)
{
    if (float_tensor.dtype() != DType::Float32) {
        throw std::invalid_argument(
            "QuantizedTensor: input must be Float32, got " +
            dtype_name(float_tensor.dtype()));
    }

    if (!params.is_valid()) {
        throw std::invalid_argument(
            "QuantizedTensor: invalid quantization parameters");
    }

    // Create quantized tensor with same shape
    data_ = Tensor(float_tensor.shape(), params.quantized_dtype);

    const float* src = float_tensor.data_ptr<float>();
    int64_t count = float_tensor.num_elements();

    auto [qmin, qmax] = get_quant_range(params.quantized_dtype);

    if (params.is_per_tensor()) {
        // Per-tensor quantization
        float scale = params.scales[0];
        int32_t zp = params.zero_points[0];

        // Security fix CRIT-Q02: Validate scale before division
        if (scale <= 0.0f || !std::isfinite(scale)) {
            throw std::invalid_argument(
                "QuantizedTensor: invalid scale value " + std::to_string(scale));
        }
        float inv_scale = 1.0f / scale;

        if (params.quantized_dtype == DType::Int8) {
            int8_t* dst = data_.data_ptr<int8_t>();
            for (int64_t i = 0; i < count; ++i) {
                int32_t q = static_cast<int32_t>(std::round(src[i] * inv_scale)) + zp;
                dst[i] = static_cast<int8_t>(std::clamp(q, qmin, qmax));
            }
        } else if (params.quantized_dtype == DType::UInt8) {
            uint8_t* dst = data_.data_ptr<uint8_t>();
            for (int64_t i = 0; i < count; ++i) {
                int32_t q = static_cast<int32_t>(std::round(src[i] * inv_scale)) + zp;
                dst[i] = static_cast<uint8_t>(std::clamp(q, qmin, qmax));
            }
        }
    } else {
        // Per-channel quantization
        const auto& shape = float_tensor.shape();
        int axis = params.channel_axis;

        // Validate axis
        if (axis < 0 || axis >= static_cast<int>(shape.size())) {
            throw std::invalid_argument(
                "QuantizedTensor: channel_axis " + std::to_string(axis) +
                " is out of range for tensor with " +
                std::to_string(shape.size()) + " dimensions");
        }

        // Validate number of scales matches channel dimension
        if (params.scales.size() != static_cast<size_t>(shape[axis])) {
            throw std::invalid_argument(
                "QuantizedTensor: number of scales (" +
                std::to_string(params.scales.size()) +
                ") does not match channel dimension (" +
                std::to_string(shape[axis]) + ")");
        }

        // Calculate strides for per-channel indexing
        int64_t outer_size = 1;
        for (int i = 0; i < axis; ++i) {
            outer_size *= shape[i];
        }
        int64_t channel_size = shape[axis];
        int64_t inner_size = 1;
        for (size_t i = axis + 1; i < shape.size(); ++i) {
            inner_size *= shape[i];
        }

        // Security fix HIGH-Q01: Pre-validate all per-channel scales
        for (size_t c = 0; c < params.scales.size(); ++c) {
            if (params.scales[c] <= 0.0f || !std::isfinite(params.scales[c])) {
                throw std::invalid_argument(
                    "QuantizedTensor: invalid scale for channel " +
                    std::to_string(c) + ": " + std::to_string(params.scales[c]));
            }
        }

        if (params.quantized_dtype == DType::Int8) {
            int8_t* dst = data_.data_ptr<int8_t>();
            for (int64_t o = 0; o < outer_size; ++o) {
                for (int64_t c = 0; c < channel_size; ++c) {
                    // Security fix CRIT-Q02/HIGH-Q01: bounds-checked index
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
        } else if (params.quantized_dtype == DType::UInt8) {
            uint8_t* dst = data_.data_ptr<uint8_t>();
            for (int64_t o = 0; o < outer_size; ++o) {
                for (int64_t c = 0; c < channel_size; ++c) {
                    // Security fix CRIT-Q02/HIGH-Q01: bounds-checked index
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
}

Tensor QuantizedTensor::dequantize() const {
    if (!is_valid()) {
        throw std::invalid_argument(
            "QuantizedTensor::dequantize: invalid quantized tensor");
    }

    Tensor result(data_.shape(), DType::Float32);
    float* dst = result.data_ptr<float>();
    int64_t count = data_.num_elements();

    if (params_.is_per_tensor()) {
        // Per-tensor dequantization
        float scale = params_.scales[0];
        int32_t zp = params_.zero_points[0];

        if (data_.dtype() == DType::Int8) {
            const int8_t* src = data_.data_ptr<int8_t>();
            for (int64_t i = 0; i < count; ++i) {
                dst[i] = (static_cast<float>(src[i]) - static_cast<float>(zp)) * scale;
            }
        } else if (data_.dtype() == DType::UInt8) {
            const uint8_t* src = data_.data_ptr<uint8_t>();
            for (int64_t i = 0; i < count; ++i) {
                dst[i] = (static_cast<float>(src[i]) - static_cast<float>(zp)) * scale;
            }
        }
    } else {
        // Per-channel dequantization
        const auto& shape = data_.shape();
        int axis = params_.channel_axis;

        int64_t outer_size = 1;
        for (int i = 0; i < axis; ++i) {
            outer_size *= shape[i];
        }
        int64_t channel_size = shape[axis];
        int64_t inner_size = 1;
        for (size_t i = axis + 1; i < shape.size(); ++i) {
            inner_size *= shape[i];
        }

        // Security fix HIGH-Q01: Validate channel axis bounds
        if (params_.channel_axis < 0 ||
            params_.channel_axis >= static_cast<int>(shape.size())) {
            throw std::invalid_argument(
                "QuantizedTensor::dequantize: invalid channel_axis");
        }

        // Security fix HIGH-Q01: Validate scales/zero_points size
        if (params_.scales.size() != static_cast<size_t>(channel_size) ||
            params_.zero_points.size() != static_cast<size_t>(channel_size)) {
            throw std::invalid_argument(
                "QuantizedTensor::dequantize: scale/zero_point count mismatch");
        }

        if (data_.dtype() == DType::Int8) {
            const int8_t* src = data_.data_ptr<int8_t>();
            for (int64_t o = 0; o < outer_size; ++o) {
                for (int64_t c = 0; c < channel_size; ++c) {
                    size_t c_idx = static_cast<size_t>(c);
                    float scale = params_.scales[c_idx];
                    float zp = static_cast<float>(params_.zero_points[c_idx]);
                    int64_t base = (o * channel_size + c) * inner_size;
                    for (int64_t i = 0; i < inner_size; ++i) {
                        int64_t idx = base + i;
                        dst[idx] = (static_cast<float>(src[idx]) - zp) * scale;
                    }
                }
            }
        } else if (data_.dtype() == DType::UInt8) {
            const uint8_t* src = data_.data_ptr<uint8_t>();
            for (int64_t o = 0; o < outer_size; ++o) {
                for (int64_t c = 0; c < channel_size; ++c) {
                    size_t c_idx = static_cast<size_t>(c);
                    float scale = params_.scales[c_idx];
                    float zp = static_cast<float>(params_.zero_points[c_idx]);
                    int64_t base = (o * channel_size + c) * inner_size;
                    for (int64_t i = 0; i < inner_size; ++i) {
                        int64_t idx = base + i;
                        dst[idx] = (static_cast<float>(src[idx]) - zp) * scale;
                    }
                }
            }
        }
    }

    return result;
}

} // namespace quantization
} // namespace pyflame_rt
