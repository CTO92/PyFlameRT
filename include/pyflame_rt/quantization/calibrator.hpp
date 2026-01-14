#pragma once

#include "pyflame_rt/quantization/quant_config.hpp"
#include "pyflame_rt/quantization/quant_params.hpp"
#include "pyflame_rt/graph.hpp"
#include "pyflame_rt/tensor.hpp"
#include <functional>
#include <memory>
#include <vector>
#include <unordered_map>
#include <limits>

namespace pyflame_rt {
namespace quantization {

/// Statistics collected during calibration for a single tensor
struct CalibrationStats {
    float min_val = std::numeric_limits<float>::max();
    float max_val = std::numeric_limits<float>::lowest();
    double sum = 0.0;
    double sum_sq = 0.0;
    size_t num_samples = 0;

    /// Histogram for entropy calibration
    std::vector<size_t> histogram;
    float hist_min = 0.0f;
    float hist_max = 0.0f;
    static constexpr size_t NUM_BINS = 2048;

    // ========================================================================
    // Update Methods
    // ========================================================================

    /// Update stats with new values
    void update(const float* data, size_t count);

    /// Update stats with tensor
    void update(const Tensor& tensor);

    /// Merge stats from another CalibrationStats
    void merge(const CalibrationStats& other);

    // ========================================================================
    // Statistics Queries
    // ========================================================================

    /// Get mean of observed values
    float mean() const {
        return num_samples > 0 ? static_cast<float>(sum / num_samples) : 0.0f;
    }

    /// Get variance of observed values
    /// Security fix LOW-Q02: Use numerically stable variance computation
    float variance() const {
        if (num_samples < 2) return 0.0f;
        double m = sum / static_cast<double>(num_samples);
        double var = (sum_sq / static_cast<double>(num_samples)) - (m * m);
        // Security fix LOW-Q02: Clamp to non-negative to handle floating-point errors
        return var > 0.0 ? static_cast<float>(var) : 0.0f;
    }

    /// Get standard deviation
    float stddev() const {
        float var = variance();
        // Security fix LOW-Q02: Ensure non-negative before sqrt
        return var > 0.0f ? std::sqrt(var) : 0.0f;
    }

    // ========================================================================
    // Quantization Parameter Computation
    // ========================================================================

    /// Compute scale and zero-point using min-max
    QuantParams compute_minmax_params(DType target_dtype, bool symmetric) const;

    /// Compute scale and zero-point using percentile
    QuantParams compute_percentile_params(DType target_dtype, bool symmetric,
                                           float percentile) const;

    /// Compute scale and zero-point using entropy (KL-divergence)
    QuantParams compute_entropy_params(DType target_dtype, bool symmetric) const;

    /// Compute params based on calibration method
    QuantParams compute_params(CalibrationMethod method, DType target_dtype,
                                bool symmetric, float percentile = 99.99f) const;

private:
    /// Initialize histogram with current min/max
    void init_histogram();

    /// Update histogram with values
    void update_histogram(const float* data, size_t count);

    /// Find percentile value from histogram
    float find_percentile(float percentile) const;

    /// Compute KL divergence between original and quantized distributions
    float compute_kl_divergence(float threshold) const;
};

/// Per-channel calibration statistics
struct PerChannelCalibrationStats {
    std::vector<CalibrationStats> channel_stats;
    int channel_axis;

    PerChannelCalibrationStats() : channel_axis(0) {}
    PerChannelCalibrationStats(size_t num_channels, int axis)
        : channel_stats(num_channels), channel_axis(axis) {}

    /// Update with tensor data
    void update(const Tensor& tensor);

    /// Compute per-channel quantization params
    QuantParams compute_params(CalibrationMethod method, DType target_dtype,
                                bool symmetric, float percentile = 99.99f) const;
};

/// Calibration data provider callback type
/// Returns a map of input tensor names to tensor data
using CalibrationDataProvider = std::function<
    std::unordered_map<std::string, Tensor>()
>;

/// Calibrator for static quantization
class Calibrator {
public:
    /// Create calibrator for a graph with given config
    Calibrator(const Graph& graph, const QuantConfig& config);
    ~Calibrator();

    // ========================================================================
    // Calibration Methods
    // ========================================================================

    /// Run calibration with data provider
    /// @param data_provider Function that returns calibration inputs
    /// @param num_batches Number of calibration batches to process
    void calibrate(CalibrationDataProvider data_provider, size_t num_batches);

    /// Add calibration sample manually
    /// @param tensor_values Map of tensor names to tensor values
    void add_sample(const std::unordered_map<std::string, Tensor>& tensor_values);

    /// Compute final quantization parameters from collected stats
    void compute_quant_params();

    // ========================================================================
    // Accessors
    // ========================================================================

    /// Get computed quantization info
    const GraphQuantInfo& quant_info() const { return quant_info_; }
    GraphQuantInfo& quant_info() { return quant_info_; }

    /// Get statistics for a tensor
    const CalibrationStats* get_stats(const std::string& tensor_name) const;

    /// Get per-channel statistics for a tensor
    const PerChannelCalibrationStats* get_channel_stats(
        const std::string& tensor_name) const;

    /// Check if calibration is complete
    bool is_calibrated() const { return calibrated_; }

    /// Get number of samples processed
    size_t num_samples() const { return num_samples_; }

    /// Get configuration
    const QuantConfig& config() const { return config_; }

    // ========================================================================
    // Tensor Registration
    // ========================================================================

    /// Register a tensor for calibration (activation tensors)
    void register_tensor(const std::string& name,
                         const std::vector<int64_t>& shape);

    /// Register a tensor with per-channel quantization
    void register_tensor_per_channel(const std::string& name,
                                      const std::vector<int64_t>& shape,
                                      int channel_axis);

    /// Get list of registered tensors
    std::vector<std::string> get_registered_tensors() const;

private:
    const Graph& graph_;
    QuantConfig config_;
    GraphQuantInfo quant_info_;

    // Calibration statistics per tensor
    std::unordered_map<std::string, CalibrationStats> tensor_stats_;
    std::unordered_map<std::string, PerChannelCalibrationStats> channel_stats_;

    bool calibrated_ = false;
    size_t num_samples_ = 0;

    /// Collect stats from tensor values
    void collect_stats(const std::unordered_map<std::string, Tensor>& tensors);
};

} // namespace quantization
} // namespace pyflame_rt
