#include "pyflame_rt/quantization/calibrator.hpp"
#include <algorithm>
#include <numeric>
#include <cmath>
#include <stdexcept>

namespace pyflame_rt {
namespace quantization {

// ============================================================================
// CalibrationStats Implementation
// ============================================================================

void CalibrationStats::update(const float* data, size_t count) {
    // Security fix LOW-Q04: Track valid samples separately
    size_t valid_count = 0;

    for (size_t i = 0; i < count; ++i) {
        float val = data[i];
        // Security fix LOW-Q04: Skip NaN and Inf values
        if (!std::isfinite(val)) continue;

        min_val = std::min(min_val, val);
        max_val = std::max(max_val, val);
        sum += static_cast<double>(val);
        sum_sq += static_cast<double>(val) * static_cast<double>(val);
        valid_count++;
    }
    // Security fix LOW-Q04: Only count valid (finite) samples
    num_samples += valid_count;

    // Update histogram if initialized
    if (!histogram.empty()) {
        update_histogram(data, count);
    }
}

void CalibrationStats::update(const Tensor& tensor) {
    if (tensor.dtype() != DType::Float32) {
        throw std::invalid_argument(
            "CalibrationStats: tensor must be Float32");
    }
    update(tensor.data_ptr<float>(),
           static_cast<size_t>(tensor.num_elements()));
}

void CalibrationStats::merge(const CalibrationStats& other) {
    min_val = std::min(min_val, other.min_val);
    max_val = std::max(max_val, other.max_val);
    sum += other.sum;
    sum_sq += other.sum_sq;
    num_samples += other.num_samples;

    // Note: histogram merging is complex, skip for now
}

void CalibrationStats::init_histogram() {
    histogram.resize(NUM_BINS, 0);
    hist_min = min_val;
    hist_max = max_val;
}

void CalibrationStats::update_histogram(const float* data, size_t count) {
    if (histogram.empty()) return;

    float range = hist_max - hist_min;
    // Security fix CRIT-Q04: Ensure range is positive and non-negligible
    if (range <= 0.0f || !std::isfinite(range)) return;

    float bin_width = range / static_cast<float>(NUM_BINS);

    // Security fix CRIT-Q04: Validate bin_width to prevent division by zero
    if (bin_width <= 0.0f || !std::isfinite(bin_width)) return;

    for (size_t i = 0; i < count; ++i) {
        float val = data[i];
        if (!std::isfinite(val)) continue;

        // Clamp to histogram range
        val = std::clamp(val, hist_min, hist_max);

        // Security fix CRIT-Q04: Safe division with validated bin_width
        float bin_f = (val - hist_min) / bin_width;
        if (!std::isfinite(bin_f) || bin_f < 0.0f) {
            bin_f = 0.0f;
        }
        size_t bin = static_cast<size_t>(bin_f);
        bin = std::min(bin, NUM_BINS - 1);
        histogram[bin]++;
    }
}

float CalibrationStats::find_percentile(float percentile) const {
    // Security fix HIGH-Q06: Validate percentile range
    if (percentile < 0.0f || percentile > 100.0f) {
        percentile = std::clamp(percentile, 0.0f, 100.0f);
    }

    if (histogram.empty() || num_samples == 0) {
        return (percentile > 50.0f) ? max_val : min_val;
    }

    size_t total = 0;
    for (size_t count : histogram) {
        total += count;
    }

    if (total == 0) {
        return (percentile > 50.0f) ? max_val : min_val;
    }

    size_t target = static_cast<size_t>(
        static_cast<double>(total) * static_cast<double>(percentile) / 100.0);

    size_t cumulative = 0;
    float range = hist_max - hist_min;

    // Security fix CRIT-Q04: Validate range and bin_width
    if (range <= 0.0f || !std::isfinite(range)) {
        return (hist_min + hist_max) / 2.0f;
    }

    float bin_width = range / static_cast<float>(NUM_BINS);
    if (bin_width <= 0.0f || !std::isfinite(bin_width)) {
        return (hist_min + hist_max) / 2.0f;
    }

    for (size_t i = 0; i < NUM_BINS; ++i) {
        cumulative += histogram[i];
        if (cumulative >= target) {
            return hist_min + static_cast<float>(i) * bin_width + bin_width / 2.0f;
        }
    }

    return hist_max;
}

float CalibrationStats::compute_kl_divergence(float threshold) const {
    if (histogram.empty()) return std::numeric_limits<float>::max();

    // Simplified KL divergence computation
    // In practice, would need to create a quantized histogram and compare

    size_t total = 0;
    for (size_t count : histogram) {
        total += count;
    }
    if (total == 0) return 0.0f;

    // Security fix CRIT-Q04: Validate range and bin_width
    float range = hist_max - hist_min;
    if (range <= 0.0f || !std::isfinite(range)) {
        return std::numeric_limits<float>::max();
    }

    float bin_width = range / static_cast<float>(NUM_BINS);
    if (bin_width <= 0.0f || !std::isfinite(bin_width)) {
        return std::numeric_limits<float>::max();
    }

    // Find the bin corresponding to threshold
    float bin_f = (threshold - hist_min) / bin_width;
    if (!std::isfinite(bin_f)) {
        bin_f = 0.0f;
    }
    size_t threshold_bin = static_cast<size_t>(
        std::clamp(bin_f, 0.0f, static_cast<float>(NUM_BINS - 1)));

    // Security fix: Prevent division by zero in log computation
    size_t active_bins = NUM_BINS - threshold_bin;
    if (active_bins == 0) {
        return 0.0f;
    }

    // Compute divergence (simplified)
    float divergence = 0.0f;
    float q = 1.0f / static_cast<float>(active_bins);
    for (size_t i = threshold_bin; i < NUM_BINS; ++i) {
        if (histogram[i] > 0) {
            float p = static_cast<float>(histogram[i]) / static_cast<float>(total);
            // Security fix: Validate p before log
            if (p > 0.0f && q > 0.0f) {
                divergence += p * std::log(p / q);
            }
        }
    }

    return divergence;
}

QuantParams CalibrationStats::compute_minmax_params(
    DType target_dtype, bool symmetric) const
{
    return QuantParams::compute_from_minmax(min_val, max_val,
                                             target_dtype, symmetric);
}

QuantParams CalibrationStats::compute_percentile_params(
    DType target_dtype, bool symmetric, float percentile) const
{
    // Security fix LOW-Q03: Validate and clamp percentile
    if (percentile < 50.0f || percentile > 100.0f) {
        // Percentile should be in upper half (e.g., 99.9 means use 0.1 to 99.9)
        percentile = std::clamp(percentile, 50.0f, 100.0f);
    }

    float adj_min, adj_max;

    if (histogram.empty()) {
        // Fallback to approximate percentile
        float range = max_val - min_val;
        // Security fix: Validate range is positive
        if (range <= 0.0f || !std::isfinite(range)) {
            return QuantParams::compute_from_minmax(min_val, max_val,
                                                     target_dtype, symmetric);
        }
        float margin = (1.0f - percentile / 100.0f) * range / 2.0f;
        adj_min = min_val + margin;
        adj_max = max_val - margin;
    } else {
        float low_percentile = 100.0f - percentile;
        adj_min = find_percentile(low_percentile);
        adj_max = find_percentile(percentile);
    }

    // Security fix: Ensure adj_min <= adj_max
    if (adj_min > adj_max) {
        std::swap(adj_min, adj_max);
    }

    return QuantParams::compute_from_minmax(adj_min, adj_max,
                                             target_dtype, symmetric);
}

QuantParams CalibrationStats::compute_entropy_params(
    DType target_dtype, bool symmetric) const
{
    if (histogram.empty()) {
        // Fall back to minmax
        return compute_minmax_params(target_dtype, symmetric);
    }

    // Search for optimal threshold using KL divergence
    float best_threshold = max_val;
    float best_divergence = std::numeric_limits<float>::max();

    float range = max_val - min_val;
    int num_steps = 100;
    float step = range / num_steps;

    for (int i = num_steps / 2; i < num_steps; ++i) {
        float threshold = min_val + i * step;
        float divergence = compute_kl_divergence(threshold);
        if (divergence < best_divergence) {
            best_divergence = divergence;
            best_threshold = threshold;
        }
    }

    // Use symmetric threshold around zero if symmetric mode
    if (symmetric) {
        best_threshold = std::max(std::abs(min_val),
                                   std::abs(best_threshold));
        return QuantParams::compute_from_minmax(-best_threshold, best_threshold,
                                                 target_dtype, symmetric);
    }

    return QuantParams::compute_from_minmax(min_val, best_threshold,
                                             target_dtype, symmetric);
}

QuantParams CalibrationStats::compute_params(
    CalibrationMethod method, DType target_dtype,
    bool symmetric, float percentile) const
{
    switch (method) {
        case CalibrationMethod::MinMax:
            return compute_minmax_params(target_dtype, symmetric);
        case CalibrationMethod::Percentile:
            return compute_percentile_params(target_dtype, symmetric, percentile);
        case CalibrationMethod::Entropy:
            return compute_entropy_params(target_dtype, symmetric);
        default:
            return compute_minmax_params(target_dtype, symmetric);
    }
}

// ============================================================================
// PerChannelCalibrationStats Implementation
// ============================================================================

void PerChannelCalibrationStats::update(const Tensor& tensor) {
    if (tensor.dtype() != DType::Float32) {
        throw std::invalid_argument(
            "PerChannelCalibrationStats: tensor must be Float32");
    }

    const auto& shape = tensor.shape();
    if (channel_axis < 0 || channel_axis >= static_cast<int>(shape.size())) {
        throw std::invalid_argument(
            "PerChannelCalibrationStats: invalid channel_axis");
    }

    int64_t num_channels = shape[channel_axis];
    if (channel_stats.size() != static_cast<size_t>(num_channels)) {
        channel_stats.resize(num_channels);
    }

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

    // Update per-channel stats
    for (int64_t o = 0; o < outer_size; ++o) {
        for (int64_t c = 0; c < num_channels; ++c) {
            int64_t base = (o * num_channels + c) * inner_size;
            channel_stats[c].update(data + base, static_cast<size_t>(inner_size));
        }
    }
}

QuantParams PerChannelCalibrationStats::compute_params(
    CalibrationMethod method, DType target_dtype,
    bool symmetric, float percentile) const
{
    size_t num_channels = channel_stats.size();
    std::vector<float> scales(num_channels);
    std::vector<int32_t> zero_points(num_channels);

    for (size_t i = 0; i < num_channels; ++i) {
        QuantParams channel_params = channel_stats[i].compute_params(
            method, target_dtype, symmetric, percentile);
        scales[i] = channel_params.scales[0];
        zero_points[i] = channel_params.zero_points[0];
    }

    return QuantParams::per_channel(scales, zero_points, channel_axis, target_dtype);
}

// ============================================================================
// Calibrator Implementation
// ============================================================================

Calibrator::Calibrator(const Graph& graph, const QuantConfig& config)
    : graph_(graph), config_(config)
{
    // Initialize with graph structure - register tensors for calibration
    // In a complete implementation, would walk the graph and identify
    // tensors that need calibration
}

Calibrator::~Calibrator() = default;

void Calibrator::calibrate(CalibrationDataProvider data_provider,
                            size_t num_batches)
{
    for (size_t batch = 0; batch < num_batches; ++batch) {
        auto inputs = data_provider();
        collect_stats(inputs);
    }

    compute_quant_params();
    calibrated_ = true;
}

void Calibrator::add_sample(
    const std::unordered_map<std::string, Tensor>& tensor_values)
{
    collect_stats(tensor_values);
}

void Calibrator::collect_stats(
    const std::unordered_map<std::string, Tensor>& tensors)
{
    for (const auto& [name, tensor] : tensors) {
        if (tensor.dtype() != DType::Float32) continue;

        // Check if per-channel stats exist for this tensor
        auto ch_it = channel_stats_.find(name);
        if (ch_it != channel_stats_.end()) {
            ch_it->second.update(tensor);
        } else {
            // Per-tensor stats
            auto& stats = tensor_stats_[name];
            stats.update(tensor);
        }
    }
    num_samples_++;
}

void Calibrator::compute_quant_params() {
    // Compute params from per-tensor stats
    for (const auto& [name, stats] : tensor_stats_) {
        QuantParams params = stats.compute_params(
            config_.calibration_method,
            config_.activation_dtype,
            config_.symmetric,
            config_.calibration_percentile);

        quant_info_.set_params(name, params);
    }

    // Compute params from per-channel stats
    for (const auto& [name, stats] : channel_stats_) {
        QuantParams params = stats.compute_params(
            config_.calibration_method,
            config_.activation_dtype,
            config_.symmetric,
            config_.calibration_percentile);

        quant_info_.set_params(name, params);
    }

    quant_info_.mode = config_.mode;
    quant_info_.activations_quantized = true;
}

const CalibrationStats* Calibrator::get_stats(
    const std::string& tensor_name) const
{
    auto it = tensor_stats_.find(tensor_name);
    return (it != tensor_stats_.end()) ? &it->second : nullptr;
}

const PerChannelCalibrationStats* Calibrator::get_channel_stats(
    const std::string& tensor_name) const
{
    auto it = channel_stats_.find(tensor_name);
    return (it != channel_stats_.end()) ? &it->second : nullptr;
}

void Calibrator::register_tensor(const std::string& name,
                                  const std::vector<int64_t>& /*shape*/)
{
    // Ensure tensor has per-tensor stats
    if (tensor_stats_.find(name) == tensor_stats_.end()) {
        tensor_stats_[name] = CalibrationStats();
    }
}

void Calibrator::register_tensor_per_channel(
    const std::string& name,
    const std::vector<int64_t>& shape,
    int channel_axis)
{
    if (channel_axis < 0 || channel_axis >= static_cast<int>(shape.size())) {
        throw std::invalid_argument(
            "register_tensor_per_channel: invalid channel_axis");
    }

    size_t num_channels = static_cast<size_t>(shape[channel_axis]);
    channel_stats_[name] = PerChannelCalibrationStats(num_channels, channel_axis);
}

std::vector<std::string> Calibrator::get_registered_tensors() const {
    std::vector<std::string> result;
    result.reserve(tensor_stats_.size() + channel_stats_.size());

    for (const auto& [name, _] : tensor_stats_) {
        result.push_back(name);
    }
    for (const auto& [name, _] : channel_stats_) {
        result.push_back(name);
    }

    return result;
}

} // namespace quantization
} // namespace pyflame_rt
