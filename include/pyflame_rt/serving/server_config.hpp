#pragma once

#include "pyflame_rt/options.hpp"
#include <string>
#include <vector>
#include <chrono>
#include <cstdint>
#include <unordered_map>
#include <optional>

namespace pyflame_rt {
namespace serving {

/// TLS configuration for secure connections
struct TLSConfig {
    /// Path to certificate file (PEM format)
    std::string cert_path;

    /// Path to private key file (PEM format)
    std::string key_path;

    /// Path to CA certificate for client verification (optional)
    std::string ca_cert_path;

    /// Require client certificate
    bool require_client_cert = false;

    /// Minimum TLS version (e.g., "1.2", "1.3")
    std::string min_version = "1.2";
};

/// Rate limiting configuration
struct RateLimitConfig {
    /// Enable rate limiting
    bool enabled = false;

    /// Maximum requests per second per client
    double requests_per_second = 100.0;

    /// Burst size (token bucket capacity)
    size_t burst_size = 200;

    /// Enable global rate limiting (across all clients)
    bool global_limit = false;

    /// Global requests per second
    double global_rps = 10000.0;
};

/// HTTP server configuration
struct HTTPServerConfig {
    /// Host to bind to
    std::string host = "0.0.0.0";

    /// Port to listen on
    uint16_t port = 8080;

    /// Number of worker threads (0 = auto)
    size_t num_workers = 0;

    /// Maximum request body size (bytes)
    size_t max_request_size = 100 * 1024 * 1024;  // 100MB

    /// Request timeout in milliseconds
    uint32_t request_timeout_ms = 30000;

    /// Keep-alive timeout in milliseconds
    uint32_t keepalive_timeout_ms = 5000;

    /// Maximum concurrent connections
    size_t max_connections = 1000;

    /// Enable CORS
    bool enable_cors = true;

    /// Allowed CORS origins (empty = all)
    std::vector<std::string> cors_origins;

    /// TLS configuration (nullopt = no TLS)
    std::optional<TLSConfig> tls;

    /// Rate limiting
    RateLimitConfig rate_limit;

    /// Base path for API (e.g., "/v1")
    std::string base_path = "/v1";

    /// Enable request logging
    bool enable_logging = true;
};

/// gRPC server configuration
struct GRPCServerConfig {
    /// Host to bind to
    std::string host = "0.0.0.0";

    /// Port to listen on
    uint16_t port = 50051;

    /// Number of completion queues (0 = auto)
    size_t num_cqs = 0;

    /// Maximum message size (bytes)
    size_t max_message_size = 100 * 1024 * 1024;  // 100MB

    /// Maximum concurrent streams per connection
    size_t max_concurrent_streams = 100;

    /// Keepalive time in milliseconds
    uint32_t keepalive_time_ms = 7200000;  // 2 hours

    /// Keepalive timeout in milliseconds
    uint32_t keepalive_timeout_ms = 20000;

    /// TLS configuration (nullopt = insecure)
    std::optional<TLSConfig> tls;

    /// Enable reflection (for grpcurl, etc.)
    bool enable_reflection = true;

    /// Enable health check service
    bool enable_health_check = true;
};

/// Model configuration for serving
struct ModelConfig {
    /// Model name (unique identifier)
    std::string name;

    /// Path to model file
    std::string model_path;

    /// Model version (for versioning support)
    std::string version = "1";

    /// Number of model instances (for parallelism)
    size_t num_instances = 1;

    /// Maximum batch size for this model
    size_t max_batch_size = 32;

    /// Batch timeout in microseconds
    uint32_t batch_timeout_us = 1000;

    /// Enable dynamic batching
    bool enable_batching = true;

    /// Preferred batch sizes
    std::vector<size_t> preferred_batch_sizes = {1, 2, 4, 8, 16, 32};

    /// Session options
    SessionOptions session_options;

    /// Warmup requests count
    size_t warmup_requests = 10;

    /// Enable model caching
    bool enable_cache = true;
};

/// Multi-model server configuration
struct ServerConfig {
    /// HTTP server configuration
    HTTPServerConfig http;

    /// gRPC server configuration
    GRPCServerConfig grpc;

    /// Models to serve
    std::vector<ModelConfig> models;

    /// Default model (used when no model specified)
    std::string default_model;

    /// Enable HTTP server
    bool enable_http = true;

    /// Enable gRPC server
    bool enable_grpc = true;

    /// Enable Prometheus metrics
    bool enable_metrics = true;

    /// Metrics port (separate from main server)
    uint16_t metrics_port = 9090;

    /// Enable health endpoints
    bool enable_health = true;

    /// Server name (for identification)
    std::string server_name = "pyflame-server";

    /// Maximum total memory for all models (0 = unlimited)
    size_t max_memory_bytes = 0;

    /// Enable model hot reload
    bool enable_hot_reload = true;

    /// Model directory for auto-discovery
    std::string model_dir;

    /// Log level (debug, info, warn, error)
    std::string log_level = "info";
};

} // namespace serving
} // namespace pyflame_rt
