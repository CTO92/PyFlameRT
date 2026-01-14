#include "pyflame_rt/serving/model_server.hpp"
#include <sstream>
#include <iomanip>
#include <regex>

namespace pyflame_rt {
namespace serving {

// ============================================================================
// ModelServer Implementation
// ============================================================================

ModelServer::ModelServer(const ServerConfig& config)
    : config_(config)
    , registry_(config.max_memory)
{
    // Create HTTP server
    http_server_ = std::make_unique<SimpleHTTPServer>(config.http);
    http_server_->set_registry(&registry_);
}

ModelServer::~ModelServer() {
    stop();
}

void ModelServer::start() {
    if (running_.exchange(true)) {
        return;  // Already running
    }

    start_time_ = std::chrono::steady_clock::now();

    try {
        // Load models from directory if specified
        if (!config_.model_dir.empty()) {
            registry_.load_from_directory(config_.model_dir);
        }

        // Load configured models
        load_configured_models();

        // Setup HTTP routes
        setup_routes();

        // Start metrics endpoint if enabled
        if (config_.enable_metrics) {
            start_metrics_server();
        }

        // Start HTTP server
        http_server_->start();

        ready_ = true;

        if (ready_callback_) {
            ready_callback_();
        }

    } catch (const std::exception& e) {
        running_ = false;
        ready_ = false;
        if (error_callback_) {
            error_callback_(e.what());
        }
        throw;
    }
}

void ModelServer::stop() {
    if (!running_.exchange(false)) {
        return;  // Already stopped
    }

    ready_ = false;

    // Stop metrics server
    if (metrics_running_.exchange(false)) {
        if (metrics_thread_.joinable()) {
            metrics_thread_.join();
        }
    }

    // Stop HTTP server
    if (http_server_) {
        http_server_->stop();
    }
}

void ModelServer::wait() {
    while (running_.load()) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
}

uint16_t ModelServer::http_port() const {
    return config_.http.port;
}

void ModelServer::load_model(const ModelConfig& config) {
    registry_.register_model(config);
}

void ModelServer::unload_model(const std::string& name, const std::string& version) {
    registry_.unload(name, version);
}

ModelServer::ServerStats ModelServer::get_stats() const {
    ServerStats stats;

    auto reg_stats = registry_.get_stats();
    stats.total_models = reg_stats.total_models;
    stats.loaded_models = reg_stats.loaded_versions;
    stats.memory_used = reg_stats.memory_used;

    auto now = std::chrono::steady_clock::now();
    stats.uptime_seconds = std::chrono::duration<double>(
        now - start_time_).count();

    // Aggregate request counts from all models
    for (const auto& model_name : registry_.list_models()) {
        if (auto instance = registry_.get(model_name)) {
            auto model_stats = instance->get_stats();
            stats.total_requests += model_stats.total_requests;
        }
    }

    return stats;
}

void ModelServer::setup_routes() {
    // Inference endpoint
    http_server_->route("POST", "/v1/models/*/infer",
        [this](const HTTPRequest& req) {
            // Extract model name from path
            std::regex path_regex("/v1/models/([^/]+)/infer");
            std::smatch match;
            if (std::regex_match(req.path, match, path_regex)) {
                return handle_infer(req, match[1].str());
            }
            return HTTPResponse::error(404, "Model not found in path");
        });

    // Alternative inference endpoint (compatibility)
    http_server_->route("POST", "/v2/models/*/infer",
        [this](const HTTPRequest& req) {
            std::regex path_regex("/v2/models/([^/]+)/infer");
            std::smatch match;
            if (std::regex_match(req.path, match, path_regex)) {
                return handle_infer(req, match[1].str());
            }
            return HTTPResponse::error(404, "Model not found in path");
        });

    // List models
    http_server_->route("GET", "/v1/models",
        [this](const HTTPRequest& req) {
            return handle_list_models(req);
        });

    // Model metadata
    http_server_->route("GET", "/v1/models/*",
        [this](const HTTPRequest& req) {
            std::regex path_regex("/v1/models/([^/]+)$");
            std::smatch match;
            if (std::regex_match(req.path, match, path_regex)) {
                return handle_model_metadata(req, match[1].str());
            }
            return HTTPResponse::error(404, "Invalid path");
        });

    // Model statistics
    http_server_->route("GET", "/v1/models/*/stats",
        [this](const HTTPRequest& req) {
            std::regex path_regex("/v1/models/([^/]+)/stats");
            std::smatch match;
            if (std::regex_match(req.path, match, path_regex)) {
                return handle_model_stats(req, match[1].str());
            }
            return HTTPResponse::error(404, "Model not found");
        });

    // Health endpoints
    http_server_->route("GET", "/health/live",
        [this](const HTTPRequest& req) {
            return handle_health_live(req);
        });

    http_server_->route("GET", "/health/ready",
        [this](const HTTPRequest& req) {
            return handle_health_ready(req);
        });

    // Metrics endpoint
    http_server_->route("GET", "/metrics",
        [this](const HTTPRequest& req) {
            return handle_metrics(req);
        });

    // Root endpoint
    http_server_->route("GET", "/",
        [this](const HTTPRequest& req) {
            std::ostringstream json;
            json << "{\n";
            json << "  \"name\": \"PyFlameRT Model Server\",\n";
            json << "  \"version\": \"0.1.0\",\n";
            json << "  \"status\": \"" << (ready_.load() ? "ready" : "starting") << "\",\n";
            json << "  \"endpoints\": {\n";
            json << "    \"infer\": \"/v1/models/{model}/infer\",\n";
            json << "    \"models\": \"/v1/models\",\n";
            json << "    \"health\": \"/health/ready\",\n";
            json << "    \"metrics\": \"/metrics\"\n";
            json << "  }\n";
            json << "}";
            return HTTPResponse::json(json.str());
        });
}

void ModelServer::load_configured_models() {
    for (const auto& model_config : config_.models) {
        try {
            registry_.register_model(model_config);
        } catch (const std::exception& e) {
            if (error_callback_) {
                error_callback_("Failed to load model " + model_config.name + ": " + e.what());
            }
            // Continue loading other models
        }
    }
}

void ModelServer::start_metrics_server() {
    // Metrics are served on the same HTTP server at /metrics
    // This function could start a separate metrics server if needed
    metrics_running_ = true;
}

HTTPResponse ModelServer::handle_infer(const HTTPRequest& req,
                                        const std::string& model_name) {
    auto start_time = std::chrono::steady_clock::now();

    // Get model instance
    auto instance = registry_.get(model_name);
    if (!instance) {
        metrics::inference_error(model_name, "not_found");
        return HTTPResponse::error(404, "Model not found: " + model_name);
    }

    if (!instance->is_ready()) {
        metrics::inference_error(model_name, "not_ready");
        return HTTPResponse::error(503, "Model not ready: " + model_name);
    }

    try {
        // Parse request
        InferRequest infer_req = json::parse_infer_request(req.body);
        infer_req.model_name = model_name;
        infer_req.arrival_time = std::chrono::steady_clock::now();

        // Run inference
        InferResponse response = instance->infer(infer_req);

        // Build response
        std::string json_response = json::build_infer_response(response);
        return HTTPResponse::json(json_response);

    } catch (const std::exception& e) {
        metrics::inference_error(model_name, "parse_error");
        return HTTPResponse::error(400, std::string("Request parsing error: ") + e.what());
    }
}

HTTPResponse ModelServer::handle_list_models(const HTTPRequest& req) {
    auto models = registry_.list_models();

    std::ostringstream json;
    json << "{\n  \"models\": [\n";

    for (size_t i = 0; i < models.size(); ++i) {
        auto instance = registry_.get(models[i]);
        json << "    {\n";
        json << "      \"name\": \"" << models[i] << "\",\n";
        json << "      \"ready\": " << (instance && instance->is_ready() ? "true" : "false");

        if (instance) {
            auto versions = registry_.list_versions(models[i]);
            json << ",\n      \"versions\": [";
            for (size_t j = 0; j < versions.size(); ++j) {
                json << "\"" << versions[j].version << "\"";
                if (j + 1 < versions.size()) json << ", ";
            }
            json << "]";
        }

        json << "\n    }";
        if (i + 1 < models.size()) json << ",";
        json << "\n";
    }

    json << "  ]\n}";
    return HTTPResponse::json(json.str());
}

HTTPResponse ModelServer::handle_model_metadata(const HTTPRequest& req,
                                                  const std::string& model_name) {
    auto instance = registry_.get(model_name);
    if (!instance) {
        return HTTPResponse::error(404, "Model not found: " + model_name);
    }

    auto meta = instance->get_serving_metadata();

    std::ostringstream json;
    json << "{\n";
    json << "  \"name\": \"" << meta.name << "\",\n";
    json << "  \"version\": \"" << meta.version << "\",\n";
    json << "  \"platform\": \"" << meta.platform << "\",\n";
    json << "  \"ready\": " << (instance->is_ready() ? "true" : "false") << ",\n";

    // Inputs
    json << "  \"inputs\": [\n";
    for (size_t i = 0; i < meta.inputs.size(); ++i) {
        const auto& input = meta.inputs[i];
        json << "    {\n";
        json << "      \"name\": \"" << input.name << "\",\n";
        json << "      \"dtype\": \"" << json::dtype_to_string(input.dtype) << "\",\n";
        json << "      \"shape\": [";
        for (size_t j = 0; j < input.shape.size(); ++j) {
            json << input.shape[j];
            if (j + 1 < input.shape.size()) json << ", ";
        }
        json << "]\n    }";
        if (i + 1 < meta.inputs.size()) json << ",";
        json << "\n";
    }
    json << "  ],\n";

    // Outputs
    json << "  \"outputs\": [\n";
    for (size_t i = 0; i < meta.outputs.size(); ++i) {
        const auto& output = meta.outputs[i];
        json << "    {\n";
        json << "      \"name\": \"" << output.name << "\",\n";
        json << "      \"dtype\": \"" << json::dtype_to_string(output.dtype) << "\",\n";
        json << "      \"shape\": [";
        for (size_t j = 0; j < output.shape.size(); ++j) {
            json << output.shape[j];
            if (j + 1 < output.shape.size()) json << ", ";
        }
        json << "]\n    }";
        if (i + 1 < meta.outputs.size()) json << ",";
        json << "\n";
    }
    json << "  ]\n";

    json << "}";
    return HTTPResponse::json(json.str());
}

HTTPResponse ModelServer::handle_model_stats(const HTTPRequest& req,
                                              const std::string& model_name) {
    auto instance = registry_.get(model_name);
    if (!instance) {
        return HTTPResponse::error(404, "Model not found: " + model_name);
    }

    auto stats = instance->get_stats();

    std::ostringstream json;
    json << std::fixed << std::setprecision(3);
    json << "{\n";
    json << "  \"model\": \"" << model_name << "\",\n";
    json << "  \"total_requests\": " << stats.total_requests << ",\n";
    json << "  \"successful_requests\": " << stats.successful_requests << ",\n";
    json << "  \"failed_requests\": " << stats.failed_requests << ",\n";
    json << "  \"avg_latency_ms\": " << stats.avg_latency_ms << ",\n";
    json << "  \"p50_latency_ms\": " << stats.p50_latency_ms << ",\n";
    json << "  \"p95_latency_ms\": " << stats.p95_latency_ms << ",\n";
    json << "  \"p99_latency_ms\": " << stats.p99_latency_ms << "\n";
    json << "}";

    return HTTPResponse::json(json.str());
}

HTTPResponse ModelServer::handle_health_live(const HTTPRequest& req) {
    return HTTPResponse::json("{\"status\": \"alive\"}");
}

HTTPResponse ModelServer::handle_health_ready(const HTTPRequest& req) {
    if (!ready_.load()) {
        HTTPResponse response;
        response.status_code = 503;
        response.body = "{\"status\": \"not_ready\"}";
        response.headers["Content-Type"] = "application/json";
        return response;
    }

    // Check if at least one model is ready
    bool any_ready = false;
    for (const auto& model_name : registry_.list_models()) {
        if (auto instance = registry_.get(model_name)) {
            if (instance->is_ready()) {
                any_ready = true;
                break;
            }
        }
    }

    if (!any_ready && !registry_.list_models().empty()) {
        HTTPResponse response;
        response.status_code = 503;
        response.body = "{\"status\": \"no_models_ready\"}";
        response.headers["Content-Type"] = "application/json";
        return response;
    }

    return HTTPResponse::json("{\"status\": \"ready\"}");
}

HTTPResponse ModelServer::handle_metrics(const HTTPRequest& req) {
    std::string metrics = MetricsRegistry::instance().export_prometheus();

    // Add server-level metrics
    auto stats = get_stats();
    std::ostringstream extra;
    extra << "# HELP pyflame_server_uptime_seconds Server uptime in seconds\n";
    extra << "# TYPE pyflame_server_uptime_seconds gauge\n";
    extra << "pyflame_server_uptime_seconds " << stats.uptime_seconds << "\n";
    extra << "# HELP pyflame_server_models_total Total number of models\n";
    extra << "# TYPE pyflame_server_models_total gauge\n";
    extra << "pyflame_server_models_total " << stats.total_models << "\n";
    extra << "# HELP pyflame_server_models_loaded Number of loaded models\n";
    extra << "# TYPE pyflame_server_models_loaded gauge\n";
    extra << "pyflame_server_models_loaded " << stats.loaded_models << "\n";

    HTTPResponse response;
    response.status_code = 200;
    response.body = metrics + extra.str();
    response.headers["Content-Type"] = "text/plain; version=0.0.4; charset=utf-8";
    return response;
}

} // namespace serving
} // namespace pyflame_rt
