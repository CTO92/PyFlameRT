#include "pyflame_rt/streaming/async_session.hpp"
#include <algorithm>
#include <stdexcept>

namespace pyflame_rt {
namespace streaming {

// ============================================================================
// StreamContext Implementation
// ============================================================================

StreamContext::StreamContext(size_t stream_id, InferenceSession* session)
    : stream_id_(stream_id)
    , session_(session)
{
}

StreamContext::~StreamContext() {
    synchronize();
}

void StreamContext::synchronize() {
    std::unique_lock<std::mutex> lock(mutex_);
    sync_cv_.wait(lock, [this]() { return pending_count_ == 0; });
}

void StreamContext::record_event(const std::string& name) {
    std::lock_guard<std::mutex> lock(mutex_);
    events_[name] = std::chrono::steady_clock::now();
}

void StreamContext::wait_event(const std::string& name) {
    std::lock_guard<std::mutex> lock(mutex_);
    // Events are already recorded, waiting is effectively a no-op in this
    // CPU implementation. In a GPU implementation, this would synchronize
    // with the actual event.
}

// ============================================================================
// AsyncSession Implementation
// ============================================================================

AsyncSession::AsyncSession(std::shared_ptr<InferenceSession> session,
                           const AsyncOptions& options)
    : session_(std::move(session))
    , options_(options)
{
    // Create stream contexts
    for (size_t i = 0; i < options_.num_streams; ++i) {
        streams_.push_back(std::make_unique<StreamContext>(i, session_.get()));
    }

    // Initialize per-stream queues
    stream_queues_.resize(options_.num_streams);
    stream_mutexes_ = std::vector<std::mutex>(options_.num_streams);
    stream_cvs_.resize(options_.num_streams);

    // Initialize stats
    stats_.stream_utilization.resize(options_.num_streams, 0);
}

AsyncSession::~AsyncSession() {
    stop();
}

AsyncSession::AsyncSession(AsyncSession&& other) noexcept
    : session_(std::move(other.session_))
    , options_(std::move(other.options_))
    , streams_(std::move(other.streams_))
    , running_(other.running_.load())
    , next_request_id_(other.next_request_id_.load())
    , workers_(std::move(other.workers_))
    , callback_thread_(std::move(other.callback_thread_))
    , stream_queues_(std::move(other.stream_queues_))
    , callback_queue_(std::move(other.callback_queue_))
    , stats_(std::move(other.stats_))
{
    other.running_ = false;
}

AsyncSession& AsyncSession::operator=(AsyncSession&& other) noexcept {
    if (this != &other) {
        stop();

        session_ = std::move(other.session_);
        options_ = std::move(other.options_);
        streams_ = std::move(other.streams_);
        running_ = other.running_.load();
        next_request_id_ = other.next_request_id_.load();
        workers_ = std::move(other.workers_);
        callback_thread_ = std::move(other.callback_thread_);
        stream_queues_ = std::move(other.stream_queues_);
        callback_queue_ = std::move(other.callback_queue_);
        stats_ = std::move(other.stats_);

        other.running_ = false;
    }
    return *this;
}

std::future<AsyncResult> AsyncSession::run_async(
    const std::vector<std::string>& output_names,
    const std::unordered_map<std::string, Tensor>& inputs)
{
    size_t stream_id = select_stream();
    return run_on_stream(stream_id, output_names, inputs);
}

uint64_t AsyncSession::run_async(
    const std::vector<std::string>& output_names,
    const std::unordered_map<std::string, Tensor>& inputs,
    CompletionCallback callback)
{
    uint64_t request_id = next_request_id_++;
    size_t stream_id = select_stream();

    PendingRequest request;
    request.id = request_id;
    request.output_names = output_names;
    request.inputs = inputs;
    request.callback = std::move(callback);
    request.submit_time = std::chrono::steady_clock::now();

    {
        std::lock_guard<std::mutex> lock(stream_mutexes_[stream_id]);
        stream_queues_[stream_id].push(std::move(request));
    }
    stream_cvs_[stream_id].notify_one();

    streams_[stream_id]->increment_pending();

    return request_id;
}

std::future<AsyncResult> AsyncSession::run_on_stream(
    size_t stream_id,
    const std::vector<std::string>& output_names,
    const std::unordered_map<std::string, Tensor>& inputs)
{
    if (stream_id >= streams_.size()) {
        throw std::out_of_range("Invalid stream ID");
    }

    uint64_t request_id = next_request_id_++;

    std::promise<AsyncResult> promise;
    auto future = promise.get_future();

    PendingRequest request;
    request.id = request_id;
    request.output_names = output_names;
    request.inputs = inputs;
    request.promise = std::move(promise);
    request.submit_time = std::chrono::steady_clock::now();

    {
        std::lock_guard<std::mutex> lock(stream_mutexes_[stream_id]);
        stream_queues_[stream_id].push(std::move(request));
    }
    stream_cvs_[stream_id].notify_one();

    streams_[stream_id]->increment_pending();

    return future;
}

StreamContext& AsyncSession::stream(size_t id) {
    if (id >= streams_.size()) {
        throw std::out_of_range("Invalid stream ID");
    }
    return *streams_[id];
}

const StreamContext& AsyncSession::stream(size_t id) const {
    if (id >= streams_.size()) {
        throw std::out_of_range("Invalid stream ID");
    }
    return *streams_[id];
}

size_t AsyncSession::select_stream() const {
    // Select least busy stream
    size_t min_pending = std::numeric_limits<size_t>::max();
    size_t selected = 0;

    for (size_t i = 0; i < streams_.size(); ++i) {
        size_t pending = streams_[i]->pending_count();
        if (pending < min_pending) {
            min_pending = pending;
            selected = i;
        }
    }

    return selected;
}

void AsyncSession::synchronize_all() {
    for (auto& stream : streams_) {
        stream->synchronize();
    }
}

void AsyncSession::synchronize(size_t stream_id) {
    if (stream_id >= streams_.size()) {
        throw std::out_of_range("Invalid stream ID");
    }
    streams_[stream_id]->synchronize();
}

void AsyncSession::start() {
    if (running_) return;

    running_ = true;

    // Start stream workers
    for (size_t i = 0; i < options_.num_streams; ++i) {
        workers_.emplace_back(&AsyncSession::stream_worker, this, i);
    }

    // Start callback worker
    if (options_.callback_threads > 0) {
        callback_thread_ = std::thread(&AsyncSession::callback_worker, this);
    }
}

void AsyncSession::stop() {
    if (!running_) return;

    running_ = false;

    // Wake up all stream workers
    for (size_t i = 0; i < options_.num_streams; ++i) {
        stream_cvs_[i].notify_all();
    }
    callback_cv_.notify_all();

    // Join workers
    for (auto& worker : workers_) {
        if (worker.joinable()) {
            worker.join();
        }
    }
    workers_.clear();

    if (callback_thread_.joinable()) {
        callback_thread_.join();
    }
}

bool AsyncSession::cancel(uint64_t request_id) {
    std::lock_guard<std::mutex> lock(cancelled_mutex_);
    cancelled_requests_.insert(request_id);

    {
        std::lock_guard<std::mutex> stats_lock(stats_mutex_);
        stats_.cancelled_requests++;
    }

    return true;
}

void AsyncSession::cancel_all() {
    // Mark all pending requests as cancelled
    for (size_t i = 0; i < options_.num_streams; ++i) {
        std::lock_guard<std::mutex> lock(stream_mutexes_[i]);
        while (!stream_queues_[i].empty()) {
            auto& req = stream_queues_[i].front();
            {
                std::lock_guard<std::mutex> cancel_lock(cancelled_mutex_);
                cancelled_requests_.insert(req.id);
            }
            stream_queues_[i].pop();
        }
    }
}

AsyncSession::Stats AsyncSession::get_stats() const {
    std::lock_guard<std::mutex> lock(stats_mutex_);
    Stats stats = stats_;

    // Update stream utilization
    stats.stream_utilization.resize(streams_.size());
    for (size_t i = 0; i < streams_.size(); ++i) {
        stats.stream_utilization[i] = streams_[i]->pending_count();
    }

    return stats;
}

void AsyncSession::reset_stats() {
    std::lock_guard<std::mutex> lock(stats_mutex_);
    stats_ = Stats{};
    stats_.stream_utilization.resize(streams_.size(), 0);
}

void AsyncSession::stream_worker(size_t stream_id) {
    while (running_) {
        PendingRequest request;
        bool have_request = false;

        {
            std::unique_lock<std::mutex> lock(stream_mutexes_[stream_id]);

            stream_cvs_[stream_id].wait_for(lock, std::chrono::milliseconds(100),
                [this, stream_id]() {
                    return !running_ || !stream_queues_[stream_id].empty();
                });

            if (!stream_queues_[stream_id].empty()) {
                request = std::move(stream_queues_[stream_id].front());
                stream_queues_[stream_id].pop();
                have_request = true;
            }
        }

        if (have_request) {
            process_request(stream_id, request);
        }
    }

    // Process remaining requests on shutdown
    while (true) {
        PendingRequest request;
        {
            std::lock_guard<std::mutex> lock(stream_mutexes_[stream_id]);
            if (stream_queues_[stream_id].empty()) break;
            request = std::move(stream_queues_[stream_id].front());
            stream_queues_[stream_id].pop();
        }
        process_request(stream_id, request);
    }
}

void AsyncSession::callback_worker() {
    while (running_ || !callback_queue_.empty()) {
        std::unique_lock<std::mutex> lock(callback_mutex_);

        callback_cv_.wait_for(lock, std::chrono::milliseconds(100), [this]() {
            return !running_ || !callback_queue_.empty();
        });

        while (!callback_queue_.empty()) {
            auto [callback, result] = std::move(callback_queue_.front());
            callback_queue_.pop();
            lock.unlock();

            if (callback) {
                try {
                    callback(std::move(result));
                } catch (...) {
                    // Ignore callback exceptions
                }
            }

            lock.lock();
        }
    }
}

void AsyncSession::process_request(size_t stream_id, PendingRequest& request) {
    auto start_time = std::chrono::steady_clock::now();
    auto queue_time = std::chrono::duration_cast<std::chrono::microseconds>(
        start_time - request.submit_time);

    AsyncResult result;
    result.request_id = request.id;
    result.stream_id = stream_id;

    // Check if cancelled
    if (is_cancelled(request.id)) {
        result.success = false;
        result.error_message = "Request cancelled";

        streams_[stream_id]->decrement_pending();

        if (request.callback) {
            std::lock_guard<std::mutex> lock(callback_mutex_);
            callback_queue_.emplace(std::move(request.callback), std::move(result));
            callback_cv_.notify_one();
        } else {
            try {
                request.promise.set_value(std::move(result));
            } catch (...) {}
        }
        return;
    }

    try {
        // Run inference
        auto outputs = session_->run(request.output_names, request.inputs);

        auto end_time = std::chrono::steady_clock::now();
        auto compute_time = std::chrono::duration_cast<std::chrono::microseconds>(
            end_time - start_time);

        result.outputs = std::move(outputs);
        result.latency = compute_time;
        result.success = true;

        if (options_.enable_profiling) {
            result.profiling = AsyncResult::ProfilingData{};
            result.profiling->queue_time = queue_time;
            result.profiling->compute_time = compute_time;
        }

        // Update stats
        {
            std::lock_guard<std::mutex> lock(stats_mutex_);
            stats_.total_requests++;
            stats_.completed_requests++;

            // Running average
            double alpha = 0.1;
            stats_.avg_latency_us = stats_.avg_latency_us * (1 - alpha) +
                                    compute_time.count() * alpha;
            stats_.avg_queue_time_us = stats_.avg_queue_time_us * (1 - alpha) +
                                       queue_time.count() * alpha;
        }

    } catch (const std::exception& e) {
        result.success = false;
        result.error_message = e.what();

        std::lock_guard<std::mutex> lock(stats_mutex_);
        stats_.total_requests++;
        stats_.failed_requests++;
    }

    streams_[stream_id]->decrement_pending();

    // Dispatch result
    if (request.callback) {
        std::lock_guard<std::mutex> lock(callback_mutex_);
        callback_queue_.emplace(std::move(request.callback), std::move(result));
        callback_cv_.notify_one();
    } else {
        try {
            request.promise.set_value(std::move(result));
        } catch (...) {
            // Promise already satisfied or broken
        }
    }
}

bool AsyncSession::is_cancelled(uint64_t request_id) const {
    std::lock_guard<std::mutex> lock(cancelled_mutex_);
    return cancelled_requests_.count(request_id) > 0;
}

// ============================================================================
// StreamingInference Implementation
// ============================================================================

StreamingInference::StreamingInference(std::shared_ptr<InferenceSession> session)
    : session_(std::move(session))
{
}

StreamingInference::~StreamingInference() {
    stop();
}

void StreamingInference::start(
    const std::unordered_map<std::string, Tensor>& initial_inputs)
{
    std::lock_guard<std::mutex> lock(mutex_);

    if (running_) {
        throw std::runtime_error("Streaming already started");
    }

    current_state_ = initial_inputs;
    accumulated_outputs_.clear();
    step_count_ = 0;
    running_ = true;
}

Tensor StreamingInference::step(const Tensor& input) {
    std::lock_guard<std::mutex> lock(mutex_);

    if (!running_) {
        throw std::runtime_error("Streaming not started");
    }

    // Update state with new input
    // Assuming single input for simplicity
    auto input_infos = session_->inputs();
    if (!input_infos.empty()) {
        current_state_[input_infos[0].name] = input;
    }

    // Run inference
    auto outputs = session_->run({}, current_state_);

    // Get first output
    Tensor result;
    auto output_infos = session_->outputs();
    if (!output_infos.empty() && outputs.count(output_infos[0].name)) {
        result = outputs[output_infos[0].name];
    }

    // Store output
    accumulated_outputs_.push_back(result);
    step_count_++;

    // Call callback
    if (step_callback_) {
        step_callback_(result);
    }

    return result;
}

std::vector<Tensor> StreamingInference::get_outputs() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return accumulated_outputs_;
}

void StreamingInference::stop() {
    std::lock_guard<std::mutex> lock(mutex_);
    running_ = false;
}

void StreamingInference::set_step_callback(
    std::function<void(const Tensor&)> callback)
{
    std::lock_guard<std::mutex> lock(mutex_);
    step_callback_ = std::move(callback);
}

// ============================================================================
// Factory Functions
// ============================================================================

std::unique_ptr<AsyncSession> create_async_session(
    std::shared_ptr<InferenceSession> session)
{
    return std::make_unique<AsyncSession>(std::move(session));
}

std::unique_ptr<AsyncSession> create_async_session(
    std::shared_ptr<InferenceSession> session,
    const AsyncOptions& options)
{
    return std::make_unique<AsyncSession>(std::move(session), options);
}

} // namespace streaming
} // namespace pyflame_rt
