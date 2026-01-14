#include "pyflame_rt/serving/model_registry.hpp"
#include "pyflame_rt/serving/metrics.hpp"
#include <algorithm>
#include <chrono>
#include <sstream>
#include <iomanip>

namespace pyflame_rt {
namespace serving {

// ============================================================================
// ModelInstance Implementation
// ============================================================================

ModelInstance::ModelInstance(const ModelConfig& config)
    : config_(config)
{
}

ModelInstance::~ModelInstance() {
    if (batcher_) {
        batcher_->stop();
    }
}

void ModelInstance::initialize() {
    // Load model session
    session_ = std::make_shared<InferenceSession>(
        config_.model_path, config_.session_options);

    // Setup dynamic batching if enabled
    if (config_.enable_batching && config_.max_batch_size > 1) {
        batching::BatchConfig batch_config;
        batch_config.max_batch_size = config_.max_batch_size;
        batch_config.max_latency = std::chrono::microseconds(config_.batch_timeout_us);
        batch_config.preferred_sizes = config_.preferred_batch_sizes;
        batch_config.min_batch_size = 1;

        batcher_ = std::make_unique<batching::DynamicBatcher>(
            session_, batch_config);
        batcher_->start();
    }

    // Warmup inference
    if (config_.warmup_requests > 0) {
        // Create dummy inputs for warmup
        auto meta = session_->metadata();
        std::unordered_map<std::string, Tensor> warmup_inputs;

        for (const auto& input : meta.inputs) {
            // Create small tensor with shape (batch=1 for other dims use 1 or actual)
            std::vector<int64_t> shape;
            for (size_t i = 0; i < input.shape.size(); ++i) {
                if (input.shape[i].has_value()) {
                    shape.push_back(input.shape[i].value());
                } else {
                    shape.push_back(1);  // Dynamic dim = 1 for warmup
                }
            }
            warmup_inputs[input.name] = Tensor(shape, input.dtype);
        }

        // Run warmup inferences
        for (size_t i = 0; i < config_.warmup_requests; ++i) {
            try {
                session_->run({}, warmup_inputs);
            } catch (...) {
                // Ignore warmup errors
            }
        }
    }

    ready_ = true;
    metrics::model_loaded(config_.name, true);
}

InferResponse ModelInstance::infer(const InferRequest& request) {
    auto start_time = std::chrono::steady_clock::now();

    InferResponse response;
    response.request_id = request.request_id;
    response.model_name = config_.name;
    response.model_version = config_.version;

    if (!ready_.load()) {
        response.success = false;
        response.error_code = ServingErrorCode::ModelNotReady;
        response.error_message = "Model is not ready";
        return response;
    }

    metrics::request_active_inc(config_.name);

    try {
        std::unordered_map<std::string, Tensor> outputs;

        if (batcher_) {
            // Use dynamic batching
            batching::InferenceRequest batch_req;
            batch_req.inputs = request.inputs;
            batch_req.output_names = request.output_names;
            batch_req.priority = request.priority;

            auto batch_response = batcher_->infer(std::move(batch_req));

            if (!batch_response.success) {
                response.success = false;
                response.error_code = ServingErrorCode::InferenceError;
                response.error_message = batch_response.error_message;
            } else {
                outputs = std::move(batch_response.outputs);
                response.success = true;
            }
        } else {
            // Direct inference
            outputs = session_->run(request.output_names, request.inputs);
            response.success = true;
        }

        response.outputs = std::move(outputs);

    } catch (const std::exception& e) {
        response.success = false;
        response.error_code = ServingErrorCode::InferenceError;
        response.error_message = e.what();
        metrics::inference_error(config_.name, "exception");
    }

    auto end_time = std::chrono::steady_clock::now();
    response.latency_us = std::chrono::duration_cast<std::chrono::microseconds>(
        end_time - start_time).count();

    // Calculate queue time if arrival time is set
    if (request.arrival_time != std::chrono::steady_clock::time_point{}) {
        response.queue_time_us = std::chrono::duration_cast<std::chrono::microseconds>(
            start_time - request.arrival_time).count();
    }

    metrics::request_active_dec(config_.name);
    update_stats(response);

    // Record metrics
    metrics::request_total(config_.name, response.success ? "success" : "error");
    metrics::request_latency(config_.name, response.latency_us / 1e6);

    return response;
}

std::future<InferResponse> ModelInstance::infer_async(const InferRequest& request) {
    return std::async(std::launch::async, [this, request]() {
        return infer(request);
    });
}

const ModelMetadata& ModelInstance::metadata() const {
    return session_->metadata();
}

ServingModelMetadata ModelInstance::get_serving_metadata() const {
    ServingModelMetadata meta;
    meta.name = config_.name;
    meta.version = config_.version;
    meta.platform = "pyflame_rt";

    const auto& session_meta = session_->metadata();

    for (const auto& input : session_meta.inputs) {
        IOSpec spec;
        spec.name = input.name;
        spec.dtype = input.dtype;
        for (const auto& dim : input.shape) {
            spec.shape.push_back(dim.value_or(-1));
        }
        meta.inputs.push_back(spec);
    }

    for (const auto& output : session_meta.outputs) {
        IOSpec spec;
        spec.name = output.name;
        spec.dtype = output.dtype;
        for (const auto& dim : output.shape) {
            spec.shape.push_back(dim.value_or(-1));
        }
        meta.outputs.push_back(spec);
    }

    return meta;
}

std::vector<std::string> ModelInstance::input_names() const {
    return session_->input_names();
}

std::vector<std::string> ModelInstance::output_names() const {
    return session_->output_names();
}

ModelStats ModelInstance::get_stats() const {
    std::lock_guard<std::mutex> lock(stats_mutex_);
    ModelStats stats = stats_;

    // Calculate percentiles
    if (!latency_samples_.empty()) {
        std::vector<double> sorted = latency_samples_;
        std::sort(sorted.begin(), sorted.end());

        size_t n = sorted.size();
        stats.p50_latency_ms = sorted[n * 50 / 100];
        stats.p95_latency_ms = sorted[std::min(n * 95 / 100, n - 1)];
        stats.p99_latency_ms = sorted[std::min(n * 99 / 100, n - 1)];
    }

    return stats;
}

void ModelInstance::reset_stats() {
    std::lock_guard<std::mutex> lock(stats_mutex_);
    stats_ = ModelStats{};
    latency_samples_.clear();
}

void ModelInstance::update_stats(const InferResponse& response) {
    std::lock_guard<std::mutex> lock(stats_mutex_);

    stats_.total_requests++;
    if (response.success) {
        stats_.successful_requests++;
    } else {
        stats_.failed_requests++;
    }

    double latency_ms = response.latency_us / 1000.0;
    latency_samples_.push_back(latency_ms);

    // Keep only last MAX_LATENCY_SAMPLES
    if (latency_samples_.size() > MAX_LATENCY_SAMPLES) {
        latency_samples_.erase(latency_samples_.begin());
    }

    // Update running average
    if (stats_.total_requests == 1) {
        stats_.avg_latency_ms = latency_ms;
    } else {
        stats_.avg_latency_ms = stats_.avg_latency_ms * 0.99 + latency_ms * 0.01;
    }
}

// ============================================================================
// ModelRegistry Implementation
// ============================================================================

ModelRegistry::ModelRegistry(size_t max_memory)
    : max_memory_(max_memory)
{
}

ModelRegistry::~ModelRegistry() = default;

void ModelRegistry::register_model(const ModelConfig& config) {
    std::unique_lock<std::shared_mutex> lock(mutex_);

    // Create or get model entry
    auto& entry = models_[config.name];
    entry.name = config.name;
    entry.base_config = config;

    // Create and initialize instance
    auto instance = std::make_shared<ModelInstance>(config);
    instance->initialize();

    // Add to registry
    entry.versions[config.version] = instance;
    update_latest_version(entry);

    // Set as default if first model
    if (default_model_.empty()) {
        default_model_ = config.name;
    }
}

void ModelRegistry::load_from_path(const std::string& name,
                                    const std::filesystem::path& path,
                                    const std::string& version) {
    ModelConfig config;
    config.name = name;
    config.model_path = path.string();
    config.version = version.empty() ? generate_version() : version;

    register_model(config);
}

void ModelRegistry::load_from_directory(const std::filesystem::path& dir) {
    if (!std::filesystem::exists(dir)) return;

    for (const auto& entry : std::filesystem::directory_iterator(dir)) {
        if (entry.is_directory()) {
            // Subdirectory = model name
            std::string model_name = entry.path().filename().string();

            for (const auto& model_file : std::filesystem::directory_iterator(entry.path())) {
                auto ext = model_file.path().extension().string();
                if (ext == ".pfm" || ext == ".onnx") {
                    // Version from filename or directory
                    std::string version = model_file.path().stem().string();
                    load_from_path(model_name, model_file.path(), version);
                }
            }
        } else {
            // File directly in model_dir
            auto ext = entry.path().extension().string();
            if (ext == ".pfm" || ext == ".onnx") {
                std::string model_name = entry.path().stem().string();
                load_from_path(model_name, entry.path());
            }
        }
    }
}

void ModelRegistry::unload(const std::string& name, const std::string& version) {
    std::unique_lock<std::shared_mutex> lock(mutex_);

    auto it = models_.find(name);
    if (it == models_.end()) return;

    if (version.empty()) {
        // Unload latest version
        if (!it->second.latest_version.empty()) {
            it->second.versions.erase(it->second.latest_version);
        }
    } else {
        it->second.versions.erase(version);
    }

    update_latest_version(it->second);

    // Remove entry if no versions left
    if (it->second.versions.empty()) {
        models_.erase(it);
    }

    metrics::model_loaded(name, false);
}

void ModelRegistry::unload_all(const std::string& name) {
    std::unique_lock<std::shared_mutex> lock(mutex_);

    auto it = models_.find(name);
    if (it != models_.end()) {
        models_.erase(it);
    }

    metrics::model_loaded(name, false);
}

std::shared_ptr<ModelInstance> ModelRegistry::get(
    const std::string& name, const std::string& version) const
{
    std::shared_lock<std::shared_mutex> lock(mutex_);

    // Use default model if name is empty
    const std::string& model_name = name.empty() ? default_model_ : name;
    if (model_name.empty()) return nullptr;

    auto it = models_.find(model_name);
    if (it == models_.end()) return nullptr;

    const std::string& ver = version.empty() ? it->second.latest_version : version;
    if (ver.empty()) return nullptr;

    auto ver_it = it->second.versions.find(ver);
    if (ver_it == it->second.versions.end()) return nullptr;

    return ver_it->second;
}

std::shared_ptr<ModelInstance> ModelRegistry::get_latest(const std::string& name) const {
    return get(name, "");
}

bool ModelRegistry::has(const std::string& name, const std::string& version) const {
    std::shared_lock<std::shared_mutex> lock(mutex_);

    auto it = models_.find(name);
    if (it == models_.end()) return false;

    if (version.empty()) return true;
    return it->second.versions.count(version) > 0;
}

std::vector<std::string> ModelRegistry::list_models() const {
    std::shared_lock<std::shared_mutex> lock(mutex_);

    std::vector<std::string> names;
    names.reserve(models_.size());
    for (const auto& [name, entry] : models_) {
        names.push_back(name);
    }
    return names;
}

std::vector<ModelVersionInfo> ModelRegistry::list_versions(const std::string& name) const {
    std::shared_lock<std::shared_mutex> lock(mutex_);

    std::vector<ModelVersionInfo> versions;
    auto it = models_.find(name);
    if (it == models_.end()) return versions;

    for (const auto& [ver, instance] : it->second.versions) {
        ModelVersionInfo info;
        info.version = ver;
        info.path = instance->config().model_path;
        info.is_loaded = instance->is_ready();
        versions.push_back(info);
    }
    return versions;
}

void ModelRegistry::enable_hot_reload(bool enable) {
    hot_reload_enabled_ = enable;
    // File watcher would be implemented here
}

void ModelRegistry::reload(const std::string& name, const std::string& version) {
    std::unique_lock<std::shared_mutex> lock(mutex_);

    auto it = models_.find(name);
    if (it == models_.end()) return;

    const std::string& ver = version.empty() ? it->second.latest_version : version;
    auto ver_it = it->second.versions.find(ver);
    if (ver_it == it->second.versions.end()) return;

    // Get config from existing instance
    ModelConfig config = ver_it->second->config();

    // Create new instance
    auto new_instance = std::make_shared<ModelInstance>(config);
    new_instance->initialize();

    // Replace old instance
    ver_it->second = new_instance;

    if (reload_callback_) {
        lock.unlock();
        reload_callback_(name, ver);
    }
}

void ModelRegistry::set_reload_callback(
    std::function<void(const std::string&, const std::string&)> callback)
{
    reload_callback_ = std::move(callback);
}

size_t ModelRegistry::memory_used() const {
    return current_memory_;
}

void ModelRegistry::evict_if_needed(size_t required_bytes) {
    if (max_memory_ == 0) return;

    std::unique_lock<std::shared_mutex> lock(mutex_);

    while (current_memory_ + required_bytes > max_memory_ && !models_.empty()) {
        // Simple eviction: remove first model found
        // A real implementation would use LRU
        auto it = models_.begin();
        if (!it->second.versions.empty()) {
            auto ver_it = it->second.versions.begin();
            it->second.versions.erase(ver_it);
        }
        if (it->second.versions.empty()) {
            models_.erase(it);
        }
    }
}

ModelRegistry::RegistryStats ModelRegistry::get_stats() const {
    std::shared_lock<std::shared_mutex> lock(mutex_);

    RegistryStats stats;
    stats.total_models = models_.size();

    for (const auto& [name, entry] : models_) {
        stats.total_versions += entry.versions.size();
        for (const auto& [ver, instance] : entry.versions) {
            if (instance->is_ready()) {
                stats.loaded_versions++;
            }
        }
    }

    stats.memory_used = current_memory_;
    return stats;
}

std::string ModelRegistry::generate_version() const {
    auto now = std::chrono::system_clock::now();
    auto time = std::chrono::system_clock::to_time_t(now);
    std::ostringstream oss;
    oss << std::put_time(std::localtime(&time), "%Y%m%d%H%M%S");
    return oss.str();
}

void ModelRegistry::update_latest_version(ModelEntry& entry) {
    if (entry.versions.empty()) {
        entry.latest_version = "";
        return;
    }

    // Find highest version (lexicographically)
    std::string latest;
    for (const auto& [ver, _] : entry.versions) {
        if (latest.empty() || ver > latest) {
            latest = ver;
        }
    }
    entry.latest_version = latest;
}

} // namespace serving
} // namespace pyflame_rt
