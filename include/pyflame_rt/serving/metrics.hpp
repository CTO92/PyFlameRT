#pragma once

#include <string>
#include <unordered_map>
#include <atomic>
#include <mutex>
#include <vector>
#include <memory>

namespace pyflame_rt {
namespace serving {

/// Metric types
enum class MetricType {
    Counter,
    Gauge,
    Histogram,
    Summary
};

/// Histogram buckets for latency tracking
struct HistogramBuckets {
    std::vector<double> boundaries;
    std::vector<std::atomic<uint64_t>> counts;
    std::atomic<double> sum{0.0};
    std::atomic<uint64_t> count{0};

    HistogramBuckets() = default;
    HistogramBuckets(const HistogramBuckets& other);
    HistogramBuckets& operator=(const HistogramBuckets& other);

    /// Create default latency buckets (in seconds)
    static HistogramBuckets latency_buckets();

    /// Create default size buckets
    static HistogramBuckets size_buckets();

    /// Observe a value
    void observe(double value);

    /// Reset all counts
    void reset();
};

/// Metrics registry - singleton for collecting metrics
class MetricsRegistry {
public:
    static MetricsRegistry& instance();

    // Non-copyable
    MetricsRegistry(const MetricsRegistry&) = delete;
    MetricsRegistry& operator=(const MetricsRegistry&) = delete;

    // ========================================================================
    // Counter operations
    // ========================================================================

    /// Increment a counter
    void counter_inc(const std::string& name,
                     const std::unordered_map<std::string, std::string>& labels = {},
                     double value = 1.0);

    /// Get counter value
    double counter_get(const std::string& name,
                       const std::unordered_map<std::string, std::string>& labels = {}) const;

    // ========================================================================
    // Gauge operations
    // ========================================================================

    /// Set a gauge value
    void gauge_set(const std::string& name, double value,
                   const std::unordered_map<std::string, std::string>& labels = {});

    /// Increment a gauge
    void gauge_inc(const std::string& name,
                   const std::unordered_map<std::string, std::string>& labels = {},
                   double value = 1.0);

    /// Decrement a gauge
    void gauge_dec(const std::string& name,
                   const std::unordered_map<std::string, std::string>& labels = {},
                   double value = 1.0);

    /// Get gauge value
    double gauge_get(const std::string& name,
                     const std::unordered_map<std::string, std::string>& labels = {}) const;

    // ========================================================================
    // Histogram operations
    // ========================================================================

    /// Observe a histogram value
    void histogram_observe(const std::string& name, double value,
                          const std::unordered_map<std::string, std::string>& labels = {});

    /// Register custom histogram buckets
    void histogram_register(const std::string& name,
                           const std::vector<double>& bucket_boundaries);

    // ========================================================================
    // Export
    // ========================================================================

    /// Export metrics in Prometheus text format
    std::string export_prometheus() const;

    /// Export metrics as JSON
    std::string export_json() const;

    // ========================================================================
    // Management
    // ========================================================================

    /// Clear all metrics
    void clear();

    /// Reset all metric values (but keep registrations)
    void reset();

    /// Get list of metric names
    std::vector<std::string> list_metrics() const;

private:
    MetricsRegistry() = default;

    mutable std::mutex mutex_;

    struct MetricValue {
        MetricType type = MetricType::Counter;
        std::atomic<double> value{0.0};
        std::unique_ptr<HistogramBuckets> histogram;
        std::string help;
    };

    std::unordered_map<std::string, MetricValue> metrics_;

    std::string format_labels(const std::unordered_map<std::string, std::string>& labels) const;
    std::string make_key(const std::string& name,
                         const std::unordered_map<std::string, std::string>& labels) const;
};

// ============================================================================
// Standard serving metrics helper functions
// ============================================================================

namespace metrics {

/// Record a successful/failed request
inline void request_total(const std::string& model, const std::string& status) {
    MetricsRegistry::instance().counter_inc(
        "pyflame_request_total",
        {{"model", model}, {"status", status}}
    );
}

/// Record request latency in seconds
inline void request_latency(const std::string& model, double seconds) {
    MetricsRegistry::instance().histogram_observe(
        "pyflame_request_latency_seconds",
        seconds,
        {{"model", model}}
    );
}

/// Update active request count
inline void request_active_inc(const std::string& model) {
    MetricsRegistry::instance().gauge_inc(
        "pyflame_requests_active",
        {{"model", model}}
    );
}

inline void request_active_dec(const std::string& model) {
    MetricsRegistry::instance().gauge_dec(
        "pyflame_requests_active",
        {{"model", model}}
    );
}

/// Record batch size
inline void batch_size(const std::string& model, size_t size) {
    MetricsRegistry::instance().histogram_observe(
        "pyflame_batch_size",
        static_cast<double>(size),
        {{"model", model}}
    );
}

/// Record model loaded status
inline void model_loaded(const std::string& model, bool loaded) {
    MetricsRegistry::instance().gauge_set(
        "pyflame_model_loaded",
        loaded ? 1.0 : 0.0,
        {{"model", model}}
    );
}

/// Record model memory usage
inline void model_memory_bytes(const std::string& model, size_t bytes) {
    MetricsRegistry::instance().gauge_set(
        "pyflame_model_memory_bytes",
        static_cast<double>(bytes),
        {{"model", model}}
    );
}

/// Record server uptime
inline void server_uptime_seconds(double seconds) {
    MetricsRegistry::instance().gauge_set("pyflame_server_uptime_seconds", seconds);
}

/// Record server memory usage
inline void server_memory_bytes(size_t bytes) {
    MetricsRegistry::instance().gauge_set(
        "pyflame_server_memory_bytes",
        static_cast<double>(bytes)
    );
}

/// Record inference errors
inline void inference_error(const std::string& model, const std::string& error_type) {
    MetricsRegistry::instance().counter_inc(
        "pyflame_inference_errors_total",
        {{"model", model}, {"error_type", error_type}}
    );
}

} // namespace metrics

} // namespace serving
} // namespace pyflame_rt
