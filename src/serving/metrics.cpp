#include "pyflame_rt/serving/metrics.hpp"
#include <sstream>
#include <iomanip>
#include <algorithm>
#include <cmath>

namespace pyflame_rt {
namespace serving {

// ============================================================================
// HistogramBuckets Implementation
// ============================================================================

HistogramBuckets::HistogramBuckets(const HistogramBuckets& other) {
    boundaries = other.boundaries;
    counts.resize(other.counts.size());
    for (size_t i = 0; i < other.counts.size(); ++i) {
        counts[i].store(other.counts[i].load());
    }
    sum.store(other.sum.load());
    count.store(other.count.load());
}

HistogramBuckets& HistogramBuckets::operator=(const HistogramBuckets& other) {
    if (this != &other) {
        boundaries = other.boundaries;
        counts.resize(other.counts.size());
        for (size_t i = 0; i < other.counts.size(); ++i) {
            counts[i].store(other.counts[i].load());
        }
        sum.store(other.sum.load());
        count.store(other.count.load());
    }
    return *this;
}

HistogramBuckets HistogramBuckets::latency_buckets() {
    HistogramBuckets buckets;
    // Latency buckets in seconds: 1ms to 10s
    buckets.boundaries = {
        0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0
    };
    buckets.counts.resize(buckets.boundaries.size() + 1);  // +1 for +Inf
    for (auto& c : buckets.counts) {
        c.store(0);
    }
    return buckets;
}

HistogramBuckets HistogramBuckets::size_buckets() {
    HistogramBuckets buckets;
    // Size buckets: 1 to 1024
    buckets.boundaries = {
        1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024
    };
    buckets.counts.resize(buckets.boundaries.size() + 1);
    for (auto& c : buckets.counts) {
        c.store(0);
    }
    return buckets;
}

void HistogramBuckets::observe(double value) {
    // Update sum and count atomically
    double old_sum = sum.load();
    while (!sum.compare_exchange_weak(old_sum, old_sum + value)) {
        // Retry
    }
    count.fetch_add(1);

    // Find bucket and increment
    for (size_t i = 0; i < boundaries.size(); ++i) {
        if (value <= boundaries[i]) {
            counts[i].fetch_add(1);
            return;
        }
    }
    // +Inf bucket
    counts.back().fetch_add(1);
}

void HistogramBuckets::reset() {
    for (auto& c : counts) {
        c.store(0);
    }
    sum.store(0.0);
    count.store(0);
}

// ============================================================================
// MetricsRegistry Implementation
// ============================================================================

MetricsRegistry& MetricsRegistry::instance() {
    static MetricsRegistry registry;
    return registry;
}

std::string MetricsRegistry::format_labels(
    const std::unordered_map<std::string, std::string>& labels) const
{
    if (labels.empty()) return "";

    std::ostringstream oss;
    oss << "{";
    bool first = true;

    // Sort labels for consistent output
    std::vector<std::pair<std::string, std::string>> sorted_labels(
        labels.begin(), labels.end());
    std::sort(sorted_labels.begin(), sorted_labels.end());

    for (const auto& [key, value] : sorted_labels) {
        if (!first) oss << ",";
        // Escape special characters in value
        std::string escaped_value;
        for (char c : value) {
            if (c == '\\' || c == '"' || c == '\n') {
                escaped_value += '\\';
            }
            escaped_value += c;
        }
        oss << key << "=\"" << escaped_value << "\"";
        first = false;
    }
    oss << "}";
    return oss.str();
}

std::string MetricsRegistry::make_key(
    const std::string& name,
    const std::unordered_map<std::string, std::string>& labels) const
{
    return name + format_labels(labels);
}

void MetricsRegistry::counter_inc(
    const std::string& name,
    const std::unordered_map<std::string, std::string>& labels,
    double value)
{
    std::lock_guard<std::mutex> lock(mutex_);
    std::string key = make_key(name, labels);

    auto& metric = metrics_[key];
    metric.type = MetricType::Counter;

    double old_val = metric.value.load();
    while (!metric.value.compare_exchange_weak(old_val, old_val + value)) {
        // Retry
    }
}

double MetricsRegistry::counter_get(
    const std::string& name,
    const std::unordered_map<std::string, std::string>& labels) const
{
    std::lock_guard<std::mutex> lock(mutex_);
    std::string key = make_key(name, labels);

    auto it = metrics_.find(key);
    if (it == metrics_.end()) return 0.0;
    return it->second.value.load();
}

void MetricsRegistry::gauge_set(
    const std::string& name,
    double value,
    const std::unordered_map<std::string, std::string>& labels)
{
    std::lock_guard<std::mutex> lock(mutex_);
    std::string key = make_key(name, labels);

    auto& metric = metrics_[key];
    metric.type = MetricType::Gauge;
    metric.value.store(value);
}

void MetricsRegistry::gauge_inc(
    const std::string& name,
    const std::unordered_map<std::string, std::string>& labels,
    double value)
{
    std::lock_guard<std::mutex> lock(mutex_);
    std::string key = make_key(name, labels);

    auto& metric = metrics_[key];
    metric.type = MetricType::Gauge;

    double old_val = metric.value.load();
    while (!metric.value.compare_exchange_weak(old_val, old_val + value)) {
        // Retry
    }
}

void MetricsRegistry::gauge_dec(
    const std::string& name,
    const std::unordered_map<std::string, std::string>& labels,
    double value)
{
    gauge_inc(name, labels, -value);
}

double MetricsRegistry::gauge_get(
    const std::string& name,
    const std::unordered_map<std::string, std::string>& labels) const
{
    std::lock_guard<std::mutex> lock(mutex_);
    std::string key = make_key(name, labels);

    auto it = metrics_.find(key);
    if (it == metrics_.end()) return 0.0;
    return it->second.value.load();
}

void MetricsRegistry::histogram_observe(
    const std::string& name,
    double value,
    const std::unordered_map<std::string, std::string>& labels)
{
    std::lock_guard<std::mutex> lock(mutex_);
    std::string key = make_key(name, labels);

    auto& metric = metrics_[key];
    metric.type = MetricType::Histogram;

    if (!metric.histogram) {
        // Default to latency buckets
        metric.histogram = std::make_unique<HistogramBuckets>(
            HistogramBuckets::latency_buckets());
    }

    metric.histogram->observe(value);
}

void MetricsRegistry::histogram_register(
    const std::string& name,
    const std::vector<double>& bucket_boundaries)
{
    std::lock_guard<std::mutex> lock(mutex_);

    auto& metric = metrics_[name];
    metric.type = MetricType::Histogram;

    metric.histogram = std::make_unique<HistogramBuckets>();
    metric.histogram->boundaries = bucket_boundaries;
    metric.histogram->counts.resize(bucket_boundaries.size() + 1);
    for (auto& c : metric.histogram->counts) {
        c.store(0);
    }
}

std::string MetricsRegistry::export_prometheus() const {
    std::lock_guard<std::mutex> lock(mutex_);
    std::ostringstream oss;

    // Group metrics by base name for proper formatting
    std::unordered_map<std::string, std::vector<std::pair<std::string, const MetricValue*>>> grouped;

    for (const auto& [key, metric] : metrics_) {
        // Extract base name (before '{')
        size_t brace_pos = key.find('{');
        std::string base_name = (brace_pos != std::string::npos)
            ? key.substr(0, brace_pos)
            : key;
        grouped[base_name].push_back({key, &metric});
    }

    for (const auto& [base_name, entries] : grouped) {
        if (entries.empty()) continue;

        const MetricValue* first_metric = entries[0].second;

        // Write TYPE comment
        switch (first_metric->type) {
            case MetricType::Counter:
                oss << "# TYPE " << base_name << " counter\n";
                break;
            case MetricType::Gauge:
                oss << "# TYPE " << base_name << " gauge\n";
                break;
            case MetricType::Histogram:
                oss << "# TYPE " << base_name << " histogram\n";
                break;
            default:
                break;
        }

        // Write metric values
        for (const auto& [key, metric] : entries) {
            switch (metric->type) {
                case MetricType::Counter:
                case MetricType::Gauge:
                    oss << key << " " << std::fixed << std::setprecision(6)
                        << metric->value.load() << "\n";
                    break;

                case MetricType::Histogram:
                    if (metric->histogram) {
                        const auto& buckets = *metric->histogram;

                        // Extract labels part
                        size_t brace_pos = key.find('{');
                        std::string labels_part;
                        if (brace_pos != std::string::npos) {
                            labels_part = key.substr(brace_pos);
                            // Remove closing brace for adding le label
                            if (!labels_part.empty() && labels_part.back() == '}') {
                                labels_part.pop_back();
                            }
                        }

                        uint64_t cumulative = 0;
                        for (size_t i = 0; i < buckets.boundaries.size(); ++i) {
                            cumulative += buckets.counts[i].load();
                            oss << base_name << "_bucket";
                            if (labels_part.empty()) {
                                oss << "{le=\"" << buckets.boundaries[i] << "\"}";
                            } else {
                                oss << labels_part << ",le=\"" << buckets.boundaries[i] << "\"}";
                            }
                            oss << " " << cumulative << "\n";
                        }

                        // +Inf bucket
                        cumulative += buckets.counts.back().load();
                        oss << base_name << "_bucket";
                        if (labels_part.empty()) {
                            oss << "{le=\"+Inf\"}";
                        } else {
                            oss << labels_part << ",le=\"+Inf\"}";
                        }
                        oss << " " << cumulative << "\n";

                        // Sum and count
                        std::string suffix = labels_part.empty() ? "" : labels_part + "}";
                        oss << base_name << "_sum" << suffix << " "
                            << buckets.sum.load() << "\n";
                        oss << base_name << "_count" << suffix << " "
                            << buckets.count.load() << "\n";
                    }
                    break;

                default:
                    break;
            }
        }
        oss << "\n";
    }

    return oss.str();
}

std::string MetricsRegistry::export_json() const {
    std::lock_guard<std::mutex> lock(mutex_);
    std::ostringstream oss;

    oss << "{\n  \"metrics\": [\n";
    bool first = true;

    for (const auto& [key, metric] : metrics_) {
        if (!first) oss << ",\n";
        first = false;

        oss << "    {\"name\": \"" << key << "\", ";
        switch (metric.type) {
            case MetricType::Counter:
                oss << "\"type\": \"counter\", ";
                break;
            case MetricType::Gauge:
                oss << "\"type\": \"gauge\", ";
                break;
            case MetricType::Histogram:
                oss << "\"type\": \"histogram\", ";
                break;
            default:
                oss << "\"type\": \"unknown\", ";
        }

        if (metric.type == MetricType::Histogram && metric.histogram) {
            oss << "\"sum\": " << metric.histogram->sum.load() << ", ";
            oss << "\"count\": " << metric.histogram->count.load();
        } else {
            oss << "\"value\": " << metric.value.load();
        }
        oss << "}";
    }

    oss << "\n  ]\n}";
    return oss.str();
}

void MetricsRegistry::clear() {
    std::lock_guard<std::mutex> lock(mutex_);
    metrics_.clear();
}

void MetricsRegistry::reset() {
    std::lock_guard<std::mutex> lock(mutex_);
    for (auto& [key, metric] : metrics_) {
        metric.value.store(0.0);
        if (metric.histogram) {
            metric.histogram->reset();
        }
    }
}

std::vector<std::string> MetricsRegistry::list_metrics() const {
    std::lock_guard<std::mutex> lock(mutex_);
    std::vector<std::string> names;
    names.reserve(metrics_.size());
    for (const auto& [key, _] : metrics_) {
        names.push_back(key);
    }
    return names;
}

} // namespace serving
} // namespace pyflame_rt
