#pragma once

#include "pyflame_rt/types.hpp"
#include "pyflame_rt/tensor.hpp"
#include "pyflame_rt/quantization/quant_config.hpp"
#include <vector>
#include <unordered_map>
#include <memory>
#include <utility>
#include <cmath>
#include <stdexcept>
#include <string>

namespace pyflame_rt {
namespace quantization {

/// Quantization parameters for a single tensor
struct QuantParams {
    /// Scale factor(s) - single value for per-tensor, multiple for per-channel
    std::vector<float> scales;

    /// Zero point(s) - same size as scales
    std::vector<int32_t> zero_points;

    /// Quantized dtype
    DType quantized_dtype = DType::Int8;

    /// Channel axis for per-channel quantization (-1 for per-tensor)
    int channel_axis = -1;

    /// Whether this is symmetric quantization
    bool symmetric = true;

    // ========================================================================
    // Queries
    // ========================================================================

    /// Check if per-tensor quantization
    bool is_per_tensor() const { return channel_axis < 0; }

    /// Check if per-channel quantization
    bool is_per_channel() const { return channel_axis >= 0; }

    /// Get number of scale/zero-point pairs
    size_t num_params() const { return scales.size(); }

    /// Check if parameters are valid
    bool is_valid() const {
        return !scales.empty() &&
               scales.size() == zero_points.size() &&
               (quantized_dtype == DType::Int8 || quantized_dtype == DType::UInt8);
    }

    // ========================================================================
    // Factory Methods
    // ========================================================================

    /// Create per-tensor params
    /// Security fix HIGH-Q02: Added validation
    static QuantParams per_tensor(float scale, int32_t zero_point,
                                   DType dtype = DType::Int8) {
        // Security fix HIGH-Q02: Validate scale is positive and finite
        if (scale <= 0.0f || !std::isfinite(scale)) {
            throw std::invalid_argument(
                "QuantParams::per_tensor: scale must be positive and finite");
        }
        // Security fix HIGH-Q02: Validate dtype
        if (dtype != DType::Int8 && dtype != DType::UInt8) {
            throw std::invalid_argument(
                "QuantParams::per_tensor: dtype must be Int8 or UInt8");
        }

        QuantParams params;
        params.scales = {scale};
        params.zero_points = {zero_point};
        params.quantized_dtype = dtype;
        params.channel_axis = -1;
        params.symmetric = (zero_point == 0);
        return params;
    }

    /// Create per-channel params
    /// Security fix HIGH-Q02: Added validation
    static QuantParams per_channel(const std::vector<float>& scales,
                                    const std::vector<int32_t>& zero_points,
                                    int axis, DType dtype = DType::Int8) {
        // Security fix HIGH-Q02: Validate sizes match
        if (scales.empty()) {
            throw std::invalid_argument(
                "QuantParams::per_channel: scales cannot be empty");
        }
        if (scales.size() != zero_points.size()) {
            throw std::invalid_argument(
                "QuantParams::per_channel: scales and zero_points size mismatch");
        }
        // Security fix HIGH-Q02: Validate dtype
        if (dtype != DType::Int8 && dtype != DType::UInt8) {
            throw std::invalid_argument(
                "QuantParams::per_channel: dtype must be Int8 or UInt8");
        }
        // Security fix HIGH-Q02: Validate axis is non-negative
        if (axis < 0) {
            throw std::invalid_argument(
                "QuantParams::per_channel: axis must be non-negative");
        }
        // Security fix HIGH-Q02: Validate all scales are positive and finite
        for (size_t i = 0; i < scales.size(); ++i) {
            if (scales[i] <= 0.0f || !std::isfinite(scales[i])) {
                throw std::invalid_argument(
                    "QuantParams::per_channel: scale[" + std::to_string(i) +
                    "] must be positive and finite");
            }
        }

        QuantParams params;
        params.scales = scales;
        params.zero_points = zero_points;
        params.quantized_dtype = dtype;
        params.channel_axis = axis;

        // Check if all zero points are 0 for symmetric
        params.symmetric = true;
        for (int32_t zp : zero_points) {
            if (zp != 0) {
                params.symmetric = false;
                break;
            }
        }
        return params;
    }

    /// Compute params from min/max values
    static QuantParams compute_from_minmax(float min_val, float max_val,
                                            DType dtype, bool symmetric);

    /// Compute per-channel params from min/max arrays
    static QuantParams compute_per_channel(const std::vector<float>& min_vals,
                                            const std::vector<float>& max_vals,
                                            int axis, DType dtype, bool symmetric);
};

/// Get quantization range for a dtype
inline std::pair<int32_t, int32_t> get_quant_range(DType dtype) {
    switch (dtype) {
        case DType::Int8:
            return {-128, 127};
        case DType::UInt8:
            return {0, 255};
        default:
            throw std::invalid_argument(
                "Unsupported quantization dtype: " + dtype_name(dtype));
    }
}

/// Quantized tensor - holds both quantized data and parameters
class QuantizedTensor {
public:
    QuantizedTensor() = default;

    /// Create quantized tensor from float tensor
    QuantizedTensor(const Tensor& float_tensor, const QuantParams& params);

    /// Create from pre-quantized data
    QuantizedTensor(Tensor data, QuantParams params)
        : data_(std::move(data)), params_(std::move(params)) {}

    /// Get quantized data
    const Tensor& data() const { return data_; }
    Tensor& data() { return data_; }

    /// Get quantization parameters
    const QuantParams& params() const { return params_; }
    QuantParams& params() { return params_; }

    /// Dequantize to float tensor
    Tensor dequantize() const;

    /// Get shape
    const std::vector<int64_t>& shape() const { return data_.shape(); }

    /// Get number of elements
    int64_t num_elements() const { return data_.num_elements(); }

    /// Check if valid
    bool is_valid() const { return data_.is_valid() && params_.is_valid(); }

private:
    Tensor data_;       // Quantized int8/uint8 data
    QuantParams params_;
};

/// Graph-level quantization info - stores params for all tensors
struct GraphQuantInfo {
    /// Quantization parameters for each tensor (by name)
    std::unordered_map<std::string, QuantParams> tensor_params;

    /// Quantization mode used
    QuantMode mode = QuantMode::None;

    /// Whether weights are quantized
    bool weights_quantized = false;

    /// Whether activations are quantized
    bool activations_quantized = false;

    // ========================================================================
    // Accessors
    // ========================================================================

    /// Get params for a tensor (returns nullptr if not found)
    const QuantParams* get_params(const std::string& name) const {
        auto it = tensor_params.find(name);
        return (it != tensor_params.end()) ? &it->second : nullptr;
    }

    /// Get mutable params for a tensor (returns nullptr if not found)
    QuantParams* get_mutable_params(const std::string& name) {
        auto it = tensor_params.find(name);
        return (it != tensor_params.end()) ? &it->second : nullptr;
    }

    /// Set params for a tensor
    void set_params(const std::string& name, const QuantParams& params) {
        tensor_params[name] = params;
    }

    /// Check if tensor has quantization params
    bool has_params(const std::string& name) const {
        return tensor_params.find(name) != tensor_params.end();
    }

    /// Remove params for a tensor
    void remove_params(const std::string& name) {
        tensor_params.erase(name);
    }

    /// Clear all params
    void clear() {
        tensor_params.clear();
        mode = QuantMode::None;
        weights_quantized = false;
        activations_quantized = false;
    }

    /// Get number of quantized tensors
    size_t num_quantized() const { return tensor_params.size(); }
};

} // namespace quantization
} // namespace pyflame_rt
