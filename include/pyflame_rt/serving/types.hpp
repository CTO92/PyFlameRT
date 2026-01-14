#pragma once

#include "pyflame_rt/tensor.hpp"
#include "pyflame_rt/types.hpp"
#include <string>
#include <vector>
#include <unordered_map>
#include <chrono>
#include <optional>

namespace pyflame_rt {
namespace serving {

/// Error codes for serving
enum class ServingErrorCode {
    OK = 0,
    InvalidRequest = 1,
    ModelNotFound = 2,
    ModelNotReady = 3,
    InputValidationFailed = 4,
    InferenceError = 5,
    Timeout = 6,
    ServerOverloaded = 7,
    InternalError = 8
};

/// Convert error code to string
inline std::string error_code_name(ServingErrorCode code) {
    switch (code) {
        case ServingErrorCode::OK: return "OK";
        case ServingErrorCode::InvalidRequest: return "INVALID_REQUEST";
        case ServingErrorCode::ModelNotFound: return "MODEL_NOT_FOUND";
        case ServingErrorCode::ModelNotReady: return "MODEL_NOT_READY";
        case ServingErrorCode::InputValidationFailed: return "INPUT_VALIDATION_FAILED";
        case ServingErrorCode::InferenceError: return "INFERENCE_ERROR";
        case ServingErrorCode::Timeout: return "TIMEOUT";
        case ServingErrorCode::ServerOverloaded: return "SERVER_OVERLOADED";
        case ServingErrorCode::InternalError: return "INTERNAL_ERROR";
        default: return "UNKNOWN";
    }
}

/// Inference request for serving
struct InferRequest {
    /// Unique request ID
    std::string request_id;

    /// Model name (empty = default)
    std::string model_name;

    /// Model version (empty = latest)
    std::string model_version;

    /// Input tensors
    std::unordered_map<std::string, Tensor> inputs;

    /// Requested output names (empty = all)
    std::vector<std::string> output_names;

    /// Request priority (higher = process first)
    int priority = 0;

    /// Request deadline (optional)
    std::optional<std::chrono::steady_clock::time_point> deadline;

    /// Custom metadata
    std::unordered_map<std::string, std::string> metadata;

    /// Request arrival time
    std::chrono::steady_clock::time_point arrival_time;

    /// Generate a unique request ID
    static std::string generate_id();
};

/// Inference response
struct InferResponse {
    /// Request ID this response corresponds to
    std::string request_id;

    /// Output tensors
    std::unordered_map<std::string, Tensor> outputs;

    /// Model name used
    std::string model_name;

    /// Model version used
    std::string model_version;

    /// Inference latency in microseconds
    int64_t latency_us = 0;

    /// Queue wait time in microseconds
    int64_t queue_time_us = 0;

    /// Success status
    bool success = true;

    /// Error message (if !success)
    std::string error_message;

    /// Error code (if !success)
    ServingErrorCode error_code = ServingErrorCode::OK;
};

/// Tensor data transfer format
enum class DataFormat {
    Raw,        // Raw bytes
    Base64,     // Base64 encoded
    JSON        // JSON array
};

/// Model input/output specification
struct IOSpec {
    std::string name;
    DType dtype;
    std::vector<int64_t> shape;  // -1 for dynamic dimensions
    std::optional<std::string> description;
};

/// Model metadata for serving
struct ServingModelMetadata {
    std::string name;
    std::string version;
    std::string platform = "pyflame_rt";
    std::vector<IOSpec> inputs;
    std::vector<IOSpec> outputs;
    std::unordered_map<std::string, std::string> custom_metadata;
};

/// Server health status
struct HealthStatus {
    bool live = false;
    bool ready = false;
    std::unordered_map<std::string, bool> model_status;
    std::string message;
};

/// Model instance statistics
struct ModelStats {
    uint64_t total_requests = 0;
    uint64_t successful_requests = 0;
    uint64_t failed_requests = 0;
    double avg_latency_ms = 0.0;
    double p50_latency_ms = 0.0;
    double p95_latency_ms = 0.0;
    double p99_latency_ms = 0.0;
    double throughput_rps = 0.0;
};

/// Server-wide metrics
struct ServerMetrics {
    uint64_t total_requests = 0;
    uint64_t active_requests = 0;
    double requests_per_second = 0.0;
    double avg_latency_ms = 0.0;
    size_t models_loaded = 0;
    size_t memory_used_bytes = 0;
    std::unordered_map<std::string, ModelStats> model_stats;
};

/// Model version info
struct ModelVersionInfo {
    std::string version;
    std::string path;
    std::chrono::system_clock::time_point created_at;
    std::chrono::system_clock::time_point last_accessed;
    size_t size_bytes = 0;
    bool is_loaded = false;
};

} // namespace serving
} // namespace pyflame_rt
