#pragma once

#include "pyflame_rt/serving/server_config.hpp"
#include "pyflame_rt/serving/model_registry.hpp"
#include "pyflame_rt/serving/http_server.hpp"
#include "pyflame_rt/serving/metrics.hpp"
#include <atomic>
#include <thread>
#include <functional>

namespace pyflame_rt {
namespace serving {

/**
 * @brief Main model serving server
 *
 * Combines HTTP server, model registry, and metrics into a single
 * easy-to-use server class for model inference serving.
 */
class ModelServer {
public:
    /**
     * @brief Construct a new Model Server
     * @param config Server configuration
     */
    explicit ModelServer(const ServerConfig& config);

    ~ModelServer();

    // Non-copyable, non-movable
    ModelServer(const ModelServer&) = delete;
    ModelServer& operator=(const ModelServer&) = delete;

    /**
     * @brief Start the server
     *
     * Loads all configured models, starts HTTP server, and begins
     * serving requests.
     */
    void start();

    /**
     * @brief Stop the server
     *
     * Gracefully shuts down all servers and unloads models.
     */
    void stop();

    /**
     * @brief Check if server is running
     * @return true if server is actively serving requests
     */
    bool is_running() const { return running_.load(); }

    /**
     * @brief Get the model registry
     * @return Reference to the model registry
     */
    ModelRegistry& registry() { return registry_; }
    const ModelRegistry& registry() const { return registry_; }

    /**
     * @brief Load a model into the server
     * @param config Model configuration
     */
    void load_model(const ModelConfig& config);

    /**
     * @brief Unload a model from the server
     * @param name Model name
     * @param version Optional version (empty = latest)
     */
    void unload_model(const std::string& name, const std::string& version = "");

    /**
     * @brief Get server statistics
     */
    struct ServerStats {
        size_t total_requests = 0;
        size_t active_requests = 0;
        size_t total_models = 0;
        size_t loaded_models = 0;
        double uptime_seconds = 0;
        size_t memory_used = 0;
    };
    ServerStats get_stats() const;

    /**
     * @brief Set callback for when server is ready
     * @param callback Function called when server is ready to accept requests
     */
    void on_ready(std::function<void()> callback) {
        ready_callback_ = std::move(callback);
    }

    /**
     * @brief Set callback for server errors
     * @param callback Function called when server error occurs
     */
    void on_error(std::function<void(const std::string&)> callback) {
        error_callback_ = std::move(callback);
    }

    /**
     * @brief Block until server stops
     */
    void wait();

    /**
     * @brief Get HTTP server port (useful when using port 0 for auto-assign)
     * @return The actual port the HTTP server is listening on
     */
    uint16_t http_port() const;

private:
    void setup_routes();
    void load_configured_models();
    void start_metrics_server();

    // Route handlers
    HTTPResponse handle_infer(const HTTPRequest& req, const std::string& model_name);
    HTTPResponse handle_list_models(const HTTPRequest& req);
    HTTPResponse handle_model_metadata(const HTTPRequest& req, const std::string& model_name);
    HTTPResponse handle_model_stats(const HTTPRequest& req, const std::string& model_name);
    HTTPResponse handle_health_live(const HTTPRequest& req);
    HTTPResponse handle_health_ready(const HTTPRequest& req);
    HTTPResponse handle_metrics(const HTTPRequest& req);

    ServerConfig config_;
    ModelRegistry registry_;
    std::unique_ptr<HTTPServer> http_server_;

    std::atomic<bool> running_{false};
    std::atomic<bool> ready_{false};

    std::chrono::steady_clock::time_point start_time_;

    std::function<void()> ready_callback_;
    std::function<void(const std::string&)> error_callback_;

    std::thread metrics_thread_;
    std::atomic<bool> metrics_running_{false};
};

/**
 * @brief Builder for creating ModelServer with fluent API
 */
class ModelServerBuilder {
public:
    ModelServerBuilder() = default;

    /**
     * @brief Set HTTP server host
     */
    ModelServerBuilder& host(const std::string& host) {
        config_.http.host = host;
        return *this;
    }

    /**
     * @brief Set HTTP server port
     */
    ModelServerBuilder& port(uint16_t port) {
        config_.http.port = port;
        return *this;
    }

    /**
     * @brief Set number of worker threads
     */
    ModelServerBuilder& workers(size_t num_workers) {
        config_.http.num_workers = num_workers;
        return *this;
    }

    /**
     * @brief Enable metrics endpoint
     */
    ModelServerBuilder& enable_metrics(bool enable = true) {
        config_.enable_metrics = enable;
        return *this;
    }

    /**
     * @brief Set metrics port
     */
    ModelServerBuilder& metrics_port(uint16_t port) {
        config_.metrics_port = port;
        return *this;
    }

    /**
     * @brief Add a model to be loaded on startup
     */
    ModelServerBuilder& add_model(const ModelConfig& model) {
        config_.models.push_back(model);
        return *this;
    }

    /**
     * @brief Add a model from path
     */
    ModelServerBuilder& add_model(const std::string& name,
                                   const std::string& path,
                                   const std::string& version = "1") {
        ModelConfig model;
        model.name = name;
        model.model_path = path;
        model.version = version;
        config_.models.push_back(model);
        return *this;
    }

    /**
     * @brief Set model directory for auto-discovery
     */
    ModelServerBuilder& model_dir(const std::string& dir) {
        config_.model_dir = dir;
        return *this;
    }

    /**
     * @brief Enable dynamic batching for all models
     */
    ModelServerBuilder& enable_batching(size_t max_batch_size = 32,
                                         size_t timeout_us = 5000) {
        for (auto& model : config_.models) {
            model.enable_batching = true;
            model.max_batch_size = max_batch_size;
            model.batch_timeout_us = timeout_us;
        }
        return *this;
    }

    /**
     * @brief Set request timeout
     */
    ModelServerBuilder& request_timeout(size_t timeout_ms) {
        config_.http.request_timeout_ms = timeout_ms;
        return *this;
    }

    /**
     * @brief Build the server
     */
    std::unique_ptr<ModelServer> build() {
        return std::make_unique<ModelServer>(config_);
    }

private:
    ServerConfig config_;
};

} // namespace serving
} // namespace pyflame_rt
