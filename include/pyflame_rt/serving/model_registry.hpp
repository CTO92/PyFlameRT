#pragma once

#include "pyflame_rt/serving/server_config.hpp"
#include "pyflame_rt/serving/types.hpp"
#include "pyflame_rt/session.hpp"
#include "pyflame_rt/batching/dynamic_batcher.hpp"
#include <memory>
#include <shared_mutex>
#include <unordered_map>
#include <filesystem>
#include <functional>
#include <atomic>

namespace pyflame_rt {
namespace serving {

/// Model instance - wraps session with batching support
class ModelInstance {
public:
    ModelInstance(const ModelConfig& config);
    ~ModelInstance();

    // Non-copyable
    ModelInstance(const ModelInstance&) = delete;
    ModelInstance& operator=(const ModelInstance&) = delete;

    /// Initialize the model (load session, setup batching)
    void initialize();

    /// Run synchronous inference
    InferResponse infer(const InferRequest& request);

    /// Run asynchronous inference
    std::future<InferResponse> infer_async(const InferRequest& request);

    /// Get model metadata
    const ModelMetadata& metadata() const;

    /// Get serving metadata
    ServingModelMetadata get_serving_metadata() const;

    /// Get model name
    const std::string& name() const { return config_.name; }

    /// Get model version
    const std::string& version() const { return config_.version; }

    /// Check if model is ready for inference
    bool is_ready() const { return ready_.load(); }

    /// Get input names
    std::vector<std::string> input_names() const;

    /// Get output names
    std::vector<std::string> output_names() const;

    /// Get statistics
    ModelStats get_stats() const;

    /// Reset statistics
    void reset_stats();

    /// Get configuration
    const ModelConfig& config() const { return config_; }

private:
    ModelConfig config_;
    std::shared_ptr<InferenceSession> session_;
    std::unique_ptr<batching::DynamicBatcher> batcher_;
    std::atomic<bool> ready_{false};

    // Statistics
    mutable std::mutex stats_mutex_;
    ModelStats stats_;
    std::vector<double> latency_samples_;
    static constexpr size_t MAX_LATENCY_SAMPLES = 1000;

    void update_stats(const InferResponse& response);
};

/// Model registry entry
struct ModelEntry {
    std::string name;
    std::unordered_map<std::string, std::shared_ptr<ModelInstance>> versions;
    std::string latest_version;
    ModelConfig base_config;
};

/// Model registry - manages model loading and versioning
class ModelRegistry {
public:
    ModelRegistry(size_t max_memory = 0);
    ~ModelRegistry();

    // Non-copyable
    ModelRegistry(const ModelRegistry&) = delete;
    ModelRegistry& operator=(const ModelRegistry&) = delete;

    // ========================================================================
    // Model Loading
    // ========================================================================

    /// Register a model from configuration
    void register_model(const ModelConfig& config);

    /// Load model from path with auto-versioning
    void load_from_path(const std::string& name,
                        const std::filesystem::path& path,
                        const std::string& version = "");

    /// Load all models from directory
    void load_from_directory(const std::filesystem::path& dir);

    /// Unload a model version
    void unload(const std::string& name, const std::string& version = "");

    /// Unload all versions of a model
    void unload_all(const std::string& name);

    // ========================================================================
    // Model Access
    // ========================================================================

    /// Get model instance
    std::shared_ptr<ModelInstance> get(const std::string& name,
                                        const std::string& version = "") const;

    /// Get latest version of a model
    std::shared_ptr<ModelInstance> get_latest(const std::string& name) const;

    /// Check if model exists
    bool has(const std::string& name, const std::string& version = "") const;

    /// List all registered model names
    std::vector<std::string> list_models() const;

    /// List versions of a model
    std::vector<ModelVersionInfo> list_versions(const std::string& name) const;

    /// Get default model name
    const std::string& default_model() const { return default_model_; }

    /// Set default model name
    void set_default_model(const std::string& name) { default_model_ = name; }

    // ========================================================================
    // Hot Reload
    // ========================================================================

    /// Enable file watching for hot reload
    void enable_hot_reload(bool enable = true);

    /// Reload a model
    void reload(const std::string& name, const std::string& version = "");

    /// Set reload callback
    void set_reload_callback(
        std::function<void(const std::string&, const std::string&)> callback);

    // ========================================================================
    // Memory Management
    // ========================================================================

    /// Get total memory used
    size_t memory_used() const;

    /// Get memory limit
    size_t memory_limit() const { return max_memory_; }

    /// Evict models to free memory
    void evict_if_needed(size_t required_bytes);

    // ========================================================================
    // Statistics
    // ========================================================================

    struct RegistryStats {
        size_t total_models = 0;
        size_t total_versions = 0;
        size_t loaded_versions = 0;
        size_t memory_used = 0;
    };
    RegistryStats get_stats() const;

private:
    mutable std::shared_mutex mutex_;
    std::unordered_map<std::string, ModelEntry> models_;
    size_t max_memory_;
    size_t current_memory_ = 0;
    bool hot_reload_enabled_ = false;
    std::string default_model_;
    std::function<void(const std::string&, const std::string&)> reload_callback_;

    std::string generate_version() const;
    void update_latest_version(ModelEntry& entry);
};

} // namespace serving
} // namespace pyflame_rt
