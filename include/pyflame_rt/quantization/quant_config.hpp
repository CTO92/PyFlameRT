#pragma once

#include "pyflame_rt/types.hpp"
#include <string>
#include <vector>
#include <unordered_set>
#include <sstream>

namespace pyflame_rt {
namespace quantization {

/// Quantization mode
enum class QuantMode : uint8_t {
    None = 0,       // No quantization (FP32)
    FP16 = 1,       // Float16 precision
    BFloat16 = 2,   // BFloat16 precision
    DynamicInt8 = 3,// Dynamic INT8 quantization
    StaticInt8 = 4  // Static INT8 with calibration
};

/// Quantization granularity
enum class QuantGranularity : uint8_t {
    PerTensor = 0,  // Single scale/zero-point per tensor
    PerChannel = 1  // Scale/zero-point per output channel
};

/// Calibration algorithm for static quantization
enum class CalibrationMethod : uint8_t {
    MinMax = 0,     // Use min/max of observed values
    Entropy = 1,    // KL-divergence based calibration
    Percentile = 2  // Use percentile bounds (e.g., 99.99%)
};

/// Convert QuantMode to string
inline std::string quant_mode_name(QuantMode mode) {
    switch (mode) {
        case QuantMode::None: return "none";
        case QuantMode::FP16: return "fp16";
        case QuantMode::BFloat16: return "bfloat16";
        case QuantMode::DynamicInt8: return "dynamic_int8";
        case QuantMode::StaticInt8: return "static_int8";
        default: return "unknown";
    }
}

/// Convert QuantGranularity to string
inline std::string granularity_name(QuantGranularity granularity) {
    switch (granularity) {
        case QuantGranularity::PerTensor: return "per_tensor";
        case QuantGranularity::PerChannel: return "per_channel";
        default: return "unknown";
    }
}

/// Convert CalibrationMethod to string
inline std::string calibration_method_name(CalibrationMethod method) {
    switch (method) {
        case CalibrationMethod::MinMax: return "minmax";
        case CalibrationMethod::Entropy: return "entropy";
        case CalibrationMethod::Percentile: return "percentile";
        default: return "unknown";
    }
}

/// Quantization configuration
struct QuantConfig {
    /// Quantization mode
    QuantMode mode = QuantMode::None;

    /// Target dtype for quantized weights (Int8 or UInt8)
    DType weight_dtype = DType::Int8;

    /// Target dtype for quantized activations
    DType activation_dtype = DType::Int8;

    /// Quantization granularity
    QuantGranularity granularity = QuantGranularity::PerTensor;

    /// Calibration method (for static quantization)
    CalibrationMethod calibration_method = CalibrationMethod::MinMax;

    /// Number of calibration samples
    size_t calibration_samples = 100;

    /// Percentile for percentile calibration (e.g., 99.99)
    float calibration_percentile = 99.99f;

    /// Operators to exclude from quantization (by op_type)
    std::unordered_set<std::string> exclude_ops;

    /// Specific nodes to exclude (by node name)
    std::unordered_set<std::string> exclude_nodes;

    /// Whether to quantize input tensors
    bool quantize_inputs = false;

    /// Whether to quantize output tensors
    bool quantize_outputs = false;

    /// Enable symmetric quantization (zero_point = 0)
    bool symmetric = true;

    // ========================================================================
    // Convenience Constructors
    // ========================================================================

    /// Create FP16 quantization config
    static QuantConfig fp16() {
        QuantConfig config;
        config.mode = QuantMode::FP16;
        return config;
    }

    /// Create BFloat16 quantization config
    static QuantConfig bfloat16() {
        QuantConfig config;
        config.mode = QuantMode::BFloat16;
        return config;
    }

    /// Create dynamic INT8 quantization config
    static QuantConfig dynamic_int8() {
        QuantConfig config;
        config.mode = QuantMode::DynamicInt8;
        config.weight_dtype = DType::Int8;
        config.activation_dtype = DType::Int8;
        return config;
    }

    /// Create static INT8 quantization config
    static QuantConfig static_int8(size_t calib_samples = 100) {
        QuantConfig config;
        config.mode = QuantMode::StaticInt8;
        config.weight_dtype = DType::Int8;
        config.activation_dtype = DType::Int8;
        config.calibration_samples = calib_samples;
        return config;
    }

    // ========================================================================
    // Validation
    // ========================================================================

    /// Check if configuration is valid
    bool is_valid() const {
        return validation_error().empty();
    }

    /// Get validation error message (empty if valid)
    std::string validation_error() const {
        std::ostringstream errors;

        // Check weight dtype for INT8 modes
        if (mode == QuantMode::DynamicInt8 || mode == QuantMode::StaticInt8) {
            if (weight_dtype != DType::Int8 && weight_dtype != DType::UInt8) {
                errors << "Weight dtype must be Int8 or UInt8 for INT8 quantization. ";
            }
            if (activation_dtype != DType::Int8 && activation_dtype != DType::UInt8) {
                errors << "Activation dtype must be Int8 or UInt8 for INT8 quantization. ";
            }
        }

        // Check calibration percentile
        if (calibration_percentile <= 0.0f || calibration_percentile > 100.0f) {
            errors << "Calibration percentile must be in range (0, 100]. ";
        }

        // Check calibration samples
        if (mode == QuantMode::StaticInt8 && calibration_samples == 0) {
            errors << "Static INT8 quantization requires at least 1 calibration sample. ";
        }

        return errors.str();
    }

    // ========================================================================
    // Helpers
    // ========================================================================

    /// Check if an operator should be quantized
    bool should_quantize_op(const std::string& op_type) const {
        return exclude_ops.find(op_type) == exclude_ops.end();
    }

    /// Check if a node should be quantized
    bool should_quantize_node(const std::string& node_name) const {
        return exclude_nodes.find(node_name) == exclude_nodes.end();
    }

    /// Check if this is an INT8 quantization mode
    bool is_int8_mode() const {
        return mode == QuantMode::DynamicInt8 || mode == QuantMode::StaticInt8;
    }

    /// Check if this is a float reduction mode (FP16/BF16)
    bool is_float_mode() const {
        return mode == QuantMode::FP16 || mode == QuantMode::BFloat16;
    }

    /// Check if calibration is required
    bool requires_calibration() const {
        return mode == QuantMode::StaticInt8;
    }
};

} // namespace quantization
} // namespace pyflame_rt
