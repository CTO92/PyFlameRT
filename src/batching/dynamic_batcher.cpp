#include "pyflame_rt/batching/dynamic_batcher.hpp"
#include <algorithm>
#include <cstring>
#include <chrono>

namespace pyflame_rt {
namespace batching {

// ============================================================================
// RequestBatch Implementation
// ============================================================================

void RequestBatch::create_batched_inputs(
    const std::vector<std::string>& input_names)
{
    if (requests.empty()) return;

    for (const auto& name : input_names) {
        // Collect all tensors for this input
        std::vector<const Tensor*> tensors;
        tensors.reserve(requests.size());

        for (const auto& req : requests) {
            auto it = req.inputs.find(name);
            if (it != req.inputs.end()) {
                tensors.push_back(&it->second);
            }
        }

        if (tensors.empty()) continue;

        // Determine batched shape
        const auto& first_shape = tensors[0]->shape();
        std::vector<int64_t> batched_shape = first_shape;

        // Set batch dimension
        if (batched_shape.empty()) {
            batched_shape.push_back(static_cast<int64_t>(batch_size));
        } else {
            batched_shape[0] = static_cast<int64_t>(batch_size);
        }

        // Create batched tensor
        Tensor batched(batched_shape, tensors[0]->dtype());

        // Calculate size of single input
        size_t single_size = tensors[0]->size_bytes();
        uint8_t* dst = static_cast<uint8_t*>(batched.data());

        // Copy data from each request
        size_t offset = 0;
        for (size_t i = 0; i < num_requests && i < tensors.size(); ++i) {
            const uint8_t* src = static_cast<const uint8_t*>(tensors[i]->data());
            std::memcpy(dst + offset, src, single_size);
            offset += single_size;
        }

        // Pad remaining slots if needed (zero-padding)
        if (batch_size > num_requests) {
            size_t padding_size = (batch_size - num_requests) * single_size;
            std::memset(dst + offset, 0, padding_size);
        }

        batched_inputs[name] = std::move(batched);
    }
}

std::vector<InferenceResponse> RequestBatch::split_outputs(
    const std::unordered_map<std::string, Tensor>& batched_outputs)
{
    std::vector<InferenceResponse> responses(num_requests);

    for (size_t i = 0; i < num_requests; ++i) {
        responses[i].request_id = requests[i].id;
        responses[i].success = true;

        for (const auto& [name, tensor] : batched_outputs) {
            // Extract slice for this request
            // Assuming batch dimension is 0
            auto single_shape = tensor.shape();
            if (!single_shape.empty()) {
                single_shape[0] = 1;
            }

            Tensor single(single_shape, tensor.dtype());
            size_t single_size = single.size_bytes();
            size_t offset = i * single_size;

            if (offset + single_size <= tensor.size_bytes()) {
                const uint8_t* src = static_cast<const uint8_t*>(tensor.data());
                uint8_t* dst = static_cast<uint8_t*>(single.data());
                std::memcpy(dst, src + offset, single_size);
            }

            responses[i].outputs[name] = std::move(single);
        }
    }

    return responses;
}

// ============================================================================
// DynamicBatcher Implementation
// ============================================================================

DynamicBatcher::DynamicBatcher(std::shared_ptr<InferenceSession> session,
                               const BatchConfig& config)
    : session_(std::move(session))
    , config_(config)
    , start_time_(std::chrono::steady_clock::now())
{
}

DynamicBatcher::~DynamicBatcher() {
    stop();
}

std::future<InferenceResponse> DynamicBatcher::submit(InferenceRequest request) {
    request.id = next_request_id_++;
    request.arrival_time = std::chrono::steady_clock::now();

    std::promise<InferenceResponse> promise;
    auto future = promise.get_future();

    {
        std::lock_guard<std::mutex> lock(queue_mutex_);

        if (config_.queue_capacity > 0 &&
            request_queue_.size() >= config_.queue_capacity) {
            // Queue full - reject request
            InferenceResponse response;
            response.request_id = request.id;
            response.success = false;
            response.error_message = "Queue full";
            promise.set_value(std::move(response));

            std::lock_guard<std::mutex> stats_lock(stats_mutex_);
            stats_.dropped_requests++;
            return future;
        }

        // Store promise for later dispatch
        uint64_t id = request.id;
        PendingRequest pending{std::move(request), std::move(promise), nullptr};

        {
            std::lock_guard<std::mutex> pending_lock(pending_mutex_);
            // We'll access the promise through the queue
        }

        request_queue_.push(std::move(pending));
    }

    queue_cv_.notify_one();
    return future;
}

uint64_t DynamicBatcher::submit(InferenceRequest request,
                                 std::function<void(InferenceResponse)> callback) {
    request.id = next_request_id_++;
    request.arrival_time = std::chrono::steady_clock::now();

    uint64_t id = request.id;

    {
        std::lock_guard<std::mutex> lock(queue_mutex_);

        if (config_.queue_capacity > 0 &&
            request_queue_.size() >= config_.queue_capacity) {
            // Queue full - call callback with error
            InferenceResponse response;
            response.request_id = id;
            response.success = false;
            response.error_message = "Queue full";

            if (callback) {
                callback(std::move(response));
            }

            std::lock_guard<std::mutex> stats_lock(stats_mutex_);
            stats_.dropped_requests++;
            return id;
        }

        PendingRequest pending{std::move(request), std::promise<InferenceResponse>(),
                              std::move(callback)};
        request_queue_.push(std::move(pending));
    }

    queue_cv_.notify_one();
    return id;
}

std::vector<std::future<InferenceResponse>> DynamicBatcher::submit_batch(
    std::vector<InferenceRequest> requests)
{
    std::vector<std::future<InferenceResponse>> futures;
    futures.reserve(requests.size());

    for (auto& req : requests) {
        futures.push_back(submit(std::move(req)));
    }

    return futures;
}

InferenceResponse DynamicBatcher::infer(InferenceRequest request) {
    return submit(std::move(request)).get();
}

void DynamicBatcher::start() {
    if (running_) return;

    running_ = true;
    start_time_ = std::chrono::steady_clock::now();

    size_t num_workers = config_.num_workers;
    if (num_workers == 0) {
        num_workers = std::thread::hardware_concurrency();
        if (num_workers == 0) num_workers = 1;
    }

    // Start worker threads
    for (size_t i = 0; i < num_workers; ++i) {
        workers_.emplace_back(&DynamicBatcher::worker_loop, this);
    }

    // Start callback worker
    callback_worker_ = std::thread(&DynamicBatcher::callback_loop, this);
}

void DynamicBatcher::stop() {
    if (!running_) return;

    running_ = false;

    // Wake up all workers
    queue_cv_.notify_all();
    callback_cv_.notify_all();

    // Join worker threads
    for (auto& worker : workers_) {
        if (worker.joinable()) {
            worker.join();
        }
    }
    workers_.clear();

    // Join callback worker
    if (callback_worker_.joinable()) {
        callback_worker_.join();
    }
}

void DynamicBatcher::flush() {
    // Force process any pending requests
    queue_cv_.notify_all();
}

size_t DynamicBatcher::queue_size() const {
    std::lock_guard<std::mutex> lock(queue_mutex_);
    return request_queue_.size();
}

DynamicBatcher::Stats DynamicBatcher::get_stats() const {
    std::lock_guard<std::mutex> lock(stats_mutex_);
    Stats stats = stats_;

    {
        std::lock_guard<std::mutex> q_lock(queue_mutex_);
        stats.queue_depth = request_queue_.size();
    }

    // Calculate throughput
    auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(
        std::chrono::steady_clock::now() - start_time_);
    if (elapsed.count() > 0) {
        stats.throughput_rps = static_cast<double>(stats.total_requests) /
                               elapsed.count();
    }

    return stats;
}

void DynamicBatcher::reset_stats() {
    std::lock_guard<std::mutex> lock(stats_mutex_);
    stats_ = Stats{};
    start_time_ = std::chrono::steady_clock::now();
}

void DynamicBatcher::update_config(const BatchConfig& config) {
    std::lock_guard<std::mutex> lock(queue_mutex_);
    config_ = config;
}

void DynamicBatcher::worker_loop() {
    while (running_) {
        RequestBatch batch = create_batch();

        if (!batch.requests.empty()) {
            process_batch(batch);
        }
    }

    // Process remaining requests on shutdown
    while (true) {
        RequestBatch batch = create_batch();
        if (batch.requests.empty()) break;
        process_batch(batch);
    }
}

void DynamicBatcher::callback_loop() {
    while (running_ || !callback_queue_.empty()) {
        std::unique_lock<std::mutex> lock(callback_mutex_);

        callback_cv_.wait_for(lock, std::chrono::milliseconds(100), [this]() {
            return !running_ || !callback_queue_.empty();
        });

        while (!callback_queue_.empty()) {
            auto [callback, response] = std::move(callback_queue_.front());
            callback_queue_.pop();
            lock.unlock();

            if (callback) {
                try {
                    callback(std::move(response));
                } catch (...) {
                    // Ignore callback exceptions
                }
            }

            lock.lock();
        }
    }
}

RequestBatch DynamicBatcher::create_batch() {
    RequestBatch batch;
    std::unique_lock<std::mutex> lock(queue_mutex_);

    // Wait for requests or timeout
    auto deadline = std::chrono::steady_clock::now() + config_.max_latency;

    queue_cv_.wait_until(lock, deadline, [this]() {
        return !running_ ||
               request_queue_.size() >= config_.max_batch_size ||
               (!request_queue_.empty() &&
                request_queue_.size() >= config_.min_batch_size);
    });

    if (request_queue_.empty()) {
        return batch;
    }

    // Collect requests for batch
    std::vector<std::pair<uint64_t, std::promise<InferenceResponse>>> promises;
    std::vector<std::pair<uint64_t, std::function<void(InferenceResponse)>>> callbacks;

    while (!request_queue_.empty() && batch.requests.size() < config_.max_batch_size) {
        auto& pending = request_queue_.front();
        uint64_t id = pending.request.id;

        batch.requests.push_back(std::move(pending.request));

        // Store promise/callback for dispatch
        if (pending.callback) {
            callbacks.emplace_back(id, std::move(pending.callback));
        } else {
            promises.emplace_back(id, std::move(pending.promise));
        }

        request_queue_.pop();
    }

    batch.num_requests = batch.requests.size();

    // Store for dispatch
    {
        std::lock_guard<std::mutex> pending_lock(pending_mutex_);
        for (auto& [id, promise] : promises) {
            // Move promise to map (need to use pointer since promise not copyable)
        }
        for (auto& [id, callback] : callbacks) {
            pending_callbacks_[id] = std::move(callback);
        }
    }

    // Determine batch size (possibly with padding)
    if (config_.enable_padding && !config_.preferred_sizes.empty()) {
        batch.batch_size = batch.num_requests;
        for (size_t pref : config_.preferred_sizes) {
            if (pref >= batch.num_requests) {
                batch.batch_size = pref;
                break;
            }
        }
    } else {
        batch.batch_size = batch.num_requests;
    }

    return batch;
}

void DynamicBatcher::process_batch(RequestBatch& batch) {
    auto start_time = std::chrono::steady_clock::now();

    std::vector<InferenceResponse> responses(batch.num_requests);

    // Initialize responses with request IDs
    for (size_t i = 0; i < batch.num_requests; ++i) {
        responses[i].request_id = batch.requests[i].id;
    }

    try {
        // Get input names from session
        auto input_infos = session_->inputs();
        std::vector<std::string> input_names;
        for (const auto& info : input_infos) {
            input_names.push_back(info.name);
        }

        // Create batched inputs
        batch.create_batched_inputs(input_names);

        // Run inference
        auto outputs = session_->run({}, batch.batched_inputs);

        // Split outputs to individual responses
        responses = batch.split_outputs(outputs);

        auto end_time = std::chrono::steady_clock::now();
        auto latency = std::chrono::duration_cast<std::chrono::microseconds>(
            end_time - start_time);

        // Set latency for all responses
        for (auto& response : responses) {
            response.latency = latency;
            response.success = true;
        }

        // Update stats
        {
            std::lock_guard<std::mutex> lock(stats_mutex_);
            stats_.total_requests += batch.num_requests;
            stats_.total_batches++;
            stats_.avg_batch_size = static_cast<double>(stats_.total_requests) /
                                    stats_.total_batches;

            // Running average of latency
            double alpha = 0.1;  // Exponential moving average factor
            stats_.avg_latency_us = stats_.avg_latency_us * (1 - alpha) +
                                    latency.count() * alpha;
        }

    } catch (const std::exception& e) {
        // Error - set failure on all responses
        for (size_t i = 0; i < batch.num_requests; ++i) {
            responses[i].request_id = batch.requests[i].id;
            responses[i].success = false;
            responses[i].error_message = e.what();
        }
    }

    // Dispatch responses
    for (auto& response : responses) {
        dispatch_response(response.request_id, std::move(response));
    }
}

void DynamicBatcher::dispatch_response(uint64_t request_id, InferenceResponse response) {
    std::function<void(InferenceResponse)> callback;

    {
        std::lock_guard<std::mutex> lock(pending_mutex_);

        auto callback_it = pending_callbacks_.find(request_id);
        if (callback_it != pending_callbacks_.end()) {
            callback = std::move(callback_it->second);
            pending_callbacks_.erase(callback_it);
        }
    }

    if (callback) {
        // Queue callback for async dispatch
        {
            std::lock_guard<std::mutex> lock(callback_mutex_);
            callback_queue_.emplace(std::move(callback), std::move(response));
        }
        callback_cv_.notify_one();
    }
    // Note: For futures, the promise is set when the request was created
    // This simplified implementation dispatches via callbacks primarily
}

// ============================================================================
// PriorityBatcher Implementation
// ============================================================================

PriorityBatcher::PriorityBatcher(std::shared_ptr<InferenceSession> session,
                                   const BatchConfig& config)
    : DynamicBatcher(std::move(session), config)
{
}

std::future<InferenceResponse> PriorityBatcher::submit_priority(
    InferenceRequest request, int priority)
{
    request.priority = priority;
    return submit(std::move(request));
}

// ============================================================================
// Factory Functions
// ============================================================================

std::unique_ptr<DynamicBatcher> create_batcher(
    std::shared_ptr<InferenceSession> session)
{
    return std::make_unique<DynamicBatcher>(std::move(session));
}

std::unique_ptr<DynamicBatcher> create_batcher(
    std::shared_ptr<InferenceSession> session,
    const BatchConfig& config)
{
    return std::make_unique<DynamicBatcher>(std::move(session), config);
}

} // namespace batching
} // namespace pyflame_rt
