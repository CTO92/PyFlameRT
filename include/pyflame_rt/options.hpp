#pragma once

#include "pyflame_rt/types.hpp"
#include "pyflame_rt/quantization/quant_config.hpp"

#include <string>
#include <vector>
#include <unordered_map>
#include <optional>
#include <cstdint>
#include <functional>

namespace pyflame_rt {

/// Configuration options for InferenceSession
struct SessionOptions {
    // Device selection
    std::string device = "cpu";  // "cpu", "wse", "wse2", "wse3"

    // CPU backend options
    int num_threads = 0;  // 0 = auto-detect

    // Runtime behavior
    bool enable_profiling = false;
    std::optional<int> memory_limit_mb;
    std::string execution_mode = "sequential";  // "sequential" or "parallel"

    // Optimization
    OptLevel optimization_level = OptLevel::Extended;  // Default: Extended
    bool verbose_optimization = false;  // Log optimization passes
    int optimization_timeout_ms = 30000;  // Security fix MED-03: timeout for optimization (30 sec default)

    // Math safety (HIGH-04 security fix)
    // When true, division by zero throws an exception instead of producing inf/nan
    bool strict_math_mode = false;

    // Logging
    std::string log_level = "warning";  // "debug", "info", "warning", "error"

    // Quantization (Phase 4)
    std::optional<quantization::QuantConfig> quantization;

    // Calibration data provider for static INT8 quantization
    // Returns a map of tensor names to calibration data
    std::function<std::unordered_map<std::string, Tensor>()> calibration_data;

    // Number of calibration batches (only used if calibration_data is set)
    size_t calibration_batches = 100;

    /// Validate options, returns list of errors
    std::vector<std::string> validate() const;
};

/// Per-run configuration options
struct RunOptions {
    std::optional<std::string> log_level;
    std::optional<std::string> tag;  // For profiling identification
    std::optional<int> timeout_ms;
};

/// Model compilation options (for ahead-of-time compilation)
struct CompileOptions {
    std::optional<std::string> cache_dir;
    std::unordered_map<std::string, std::vector<int64_t>> input_shapes;
    bool dynamic_batch = false;
    int optimization_level = 2;  // OptLevel::Extended
};

} // namespace pyflame_rt
