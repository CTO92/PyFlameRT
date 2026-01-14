#pragma once

#include "pyflame_rt/serving/server_config.hpp"
#include "pyflame_rt/serving/types.hpp"
#include <functional>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace pyflame_rt {
namespace serving {

// Forward declarations
class ModelRegistry;

/// HTTP request structure
struct HTTPRequest {
    std::string method;
    std::string path;
    std::unordered_map<std::string, std::string> headers;
    std::unordered_map<std::string, std::string> query_params;
    std::unordered_map<std::string, std::string> path_params;
    std::string body;
    std::string client_ip;
    std::string request_id;
};

/// HTTP response structure
struct HTTPResponse {
    int status_code = 200;
    std::unordered_map<std::string, std::string> headers;
    std::string body;
    std::string content_type = "application/json";

    /// Helper to set JSON response
    void set_json(const std::string& json_body, int status = 200) {
        body = json_body;
        content_type = "application/json";
        status_code = status;
    }

    /// Helper to set error response
    void set_error(int status, const std::string& message) {
        status_code = status;
        content_type = "application/json";
        body = "{\"error\":\"" + message + "\"}";
    }
};

/// Route handler type
using RouteHandler = std::function<HTTPResponse(const HTTPRequest&)>;

/// HTTP server interface
class HTTPServer {
public:
    HTTPServer(const HTTPServerConfig& config);
    virtual ~HTTPServer();

    // Non-copyable
    HTTPServer(const HTTPServer&) = delete;
    HTTPServer& operator=(const HTTPServer&) = delete;

    /// Start the HTTP server
    virtual void start() = 0;

    /// Stop the HTTP server
    virtual void stop() = 0;

    /// Check if server is running
    virtual bool is_running() const = 0;

    /// Wait for server to stop
    virtual void wait() = 0;

    /// Register a route handler
    void route(const std::string& method, const std::string& path,
               RouteHandler handler);

    /// Set model registry for inference routes
    void set_registry(ModelRegistry* registry) { registry_ = registry; }

    /// Get configuration
    const HTTPServerConfig& config() const { return config_; }

protected:
    HTTPServerConfig config_;
    ModelRegistry* registry_ = nullptr;
    std::vector<std::tuple<std::string, std::string, RouteHandler>> routes_;

    /// Handle incoming request through registered routes
    HTTPResponse handle_request(const HTTPRequest& request);

    /// Setup default REST API routes
    void setup_inference_routes();

    /// Match path with pattern and extract parameters
    bool match_path(const std::string& pattern, const std::string& path,
                    std::unordered_map<std::string, std::string>& params) const;
};

/// Create platform-specific HTTP server implementation
std::unique_ptr<HTTPServer> create_http_server(const HTTPServerConfig& config);

// ============================================================================
// JSON Utilities for HTTP
// ============================================================================

/// Convert DType to string for JSON
std::string dtype_to_json_string(DType dtype);

/// Parse DType from JSON string
DType dtype_from_json_string(const std::string& str);

/// Build JSON error response
std::string build_error_json(ServingErrorCode code, const std::string& message);

/// Build JSON model list response
std::string build_model_list_json(const std::vector<ServingModelMetadata>& models);

/// Build JSON model metadata response
std::string build_model_metadata_json(const ServingModelMetadata& metadata);

/// Build JSON inference response
std::string build_infer_response_json(const InferResponse& response);

/// Parse inference request from JSON
InferRequest parse_infer_request_json(const std::string& json,
                                       const std::string& model_name,
                                       const std::string& model_version);

} // namespace serving
} // namespace pyflame_rt
