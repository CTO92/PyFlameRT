#pragma once

#include "pyflame_rt/session.hpp"
#include "pyflame_rt/tensor.hpp"
#include <chrono>
#include <condition_variable>
#include <functional>
#include <future>
#include <memory>
#include <mutex>
#include <queue>
#include <thread>
#include <atomic>
#include <optional>

namespace pyflame_rt {
namespace streaming {

/// Async inference options
struct AsyncOptions {
    /// Number of concurrent inference streams
    size_t num_streams = 2;

    /// Enable pipelining (overlap host-device transfers with compute)
    bool enable_pipelining = true;

    /// Callback thread pool size
    size_t callback_threads = 1;

    /// Maximum pending requests per stream
    size_t max_pending = 100;

    /// Enable profiling for async operations
    bool enable_profiling = false;
};

/// Stream context for managing inference state
class StreamContext {
public:
    StreamContext(size_t stream_id, InferenceSession* session);
    ~StreamContext();

    /// Get stream ID
    size_t stream_id() const { return stream_id_; }

    /// Check if stream is busy
    bool is_busy() const { return pending_count_ > 0; }

    /// Get number of pending operations
    size_t pending_count() const { return pending_count_; }

    /// Wait for all pending operations
    void synchronize();

    /// Record event in stream
    void record_event(const std::string& name);

    /// Wait for event
    void wait_event(const std::string& name);

    /// Get session
    InferenceSession* session() { return session_; }

private:
    size_t stream_id_;
    InferenceSession* session_;
    std::atomic<size_t> pending_count_{0};
    std::unordered_map<std::string, std::chrono::steady_clock::time_point> events_;
    mutable std::mutex mutex_;
    std::condition_variable sync_cv_;

    friend class AsyncSession;
    void increment_pending() { ++pending_count_; }
    void decrement_pending() {
        --pending_count_;
        sync_cv_.notify_all();
    }
};

/// Async inference result
struct AsyncResult {
    /// Request ID
    uint64_t request_id = 0;

    /// Output tensors
    std::unordered_map<std::string, Tensor> outputs;

    /// Stream used for inference
    size_t stream_id = 0;

    /// Inference latency
    std::chrono::microseconds latency{0};

    /// Success status
    bool success = true;
    std::string error_message;

    /// Profiling data (if enabled)
    struct ProfilingData {
        std::chrono::microseconds queue_time{0};
        std::chrono::microseconds compute_time{0};
        std::chrono::microseconds callback_time{0};
    };
    std::optional<ProfilingData> profiling;
};

/// Completion callback type
using CompletionCallback = std::function<void(AsyncResult)>;

/// Asynchronous inference session
class AsyncSession {
public:
    AsyncSession(std::shared_ptr<InferenceSession> session,
                 const AsyncOptions& options = AsyncOptions());
    ~AsyncSession();

    // Non-copyable
    AsyncSession(const AsyncSession&) = delete;
    AsyncSession& operator=(const AsyncSession&) = delete;

    // Moveable
    AsyncSession(AsyncSession&&) noexcept;
    AsyncSession& operator=(AsyncSession&&) noexcept;

    // ========================================================================
    // Async Inference
    // ========================================================================

    /// Submit async inference request
    std::future<AsyncResult> run_async(
        const std::vector<std::string>& output_names,
        const std::unordered_map<std::string, Tensor>& inputs);

    /// Submit async inference with callback
    uint64_t run_async(
        const std::vector<std::string>& output_names,
        const std::unordered_map<std::string, Tensor>& inputs,
        CompletionCallback callback);

    /// Submit to specific stream
    std::future<AsyncResult> run_on_stream(
        size_t stream_id,
        const std::vector<std::string>& output_names,
        const std::unordered_map<std::string, Tensor>& inputs);

    // ========================================================================
    // Stream Management
    // ========================================================================

    /// Get number of streams
    size_t num_streams() const { return streams_.size(); }

    /// Get stream context
    StreamContext& stream(size_t id);
    const StreamContext& stream(size_t id) const;

    /// Get least busy stream
    size_t select_stream() const;

    /// Synchronize all streams
    void synchronize_all();

    /// Synchronize specific stream
    void synchronize(size_t stream_id);

    // ========================================================================
    // Control
    // ========================================================================

    /// Start async processing
    void start();

    /// Stop async processing (waits for pending)
    void stop();

    /// Check if running
    bool is_running() const { return running_; }

    /// Cancel pending request
    bool cancel(uint64_t request_id);

    /// Cancel all pending requests
    void cancel_all();

    // ========================================================================
    // Statistics
    // ========================================================================

    struct Stats {
        uint64_t total_requests = 0;
        uint64_t completed_requests = 0;
        uint64_t failed_requests = 0;
        uint64_t cancelled_requests = 0;
        double avg_latency_us = 0.0;
        double avg_queue_time_us = 0.0;
        std::vector<size_t> stream_utilization;
    };

    Stats get_stats() const;
    void reset_stats();

    // ========================================================================
    // Accessors
    // ========================================================================

    const AsyncOptions& options() const { return options_; }
    InferenceSession* session() { return session_.get(); }
    const InferenceSession* session() const { return session_.get(); }

private:
    std::shared_ptr<InferenceSession> session_;
    AsyncOptions options_;

    std::vector<std::unique_ptr<StreamContext>> streams_;
    std::atomic<bool> running_{false};
    std::atomic<uint64_t> next_request_id_{0};

    // Worker infrastructure
    std::vector<std::thread> workers_;
    std::thread callback_thread_;

    // Request queue per stream
    struct PendingRequest {
        uint64_t id;
        std::vector<std::string> output_names;
        std::unordered_map<std::string, Tensor> inputs;
        std::promise<AsyncResult> promise;
        CompletionCallback callback;
        std::chrono::steady_clock::time_point submit_time;
        bool cancelled = false;
    };
    std::vector<std::queue<PendingRequest>> stream_queues_;
    std::vector<std::mutex> stream_mutexes_;
    std::vector<std::condition_variable> stream_cvs_;

    // Cancelled request tracking
    std::unordered_set<uint64_t> cancelled_requests_;
    mutable std::mutex cancelled_mutex_;

    // Completed results for callbacks
    std::queue<std::pair<CompletionCallback, AsyncResult>> callback_queue_;
    std::mutex callback_mutex_;
    std::condition_variable callback_cv_;

    // Statistics
    mutable std::mutex stats_mutex_;
    Stats stats_;

    void stream_worker(size_t stream_id);
    void callback_worker();
    void process_request(size_t stream_id, PendingRequest& request);
    bool is_cancelled(uint64_t request_id) const;
};

/// Token-based streaming for sequence models
class StreamingInference {
public:
    StreamingInference(std::shared_ptr<InferenceSession> session);
    ~StreamingInference();

    /// Start streaming inference
    void start(const std::unordered_map<std::string, Tensor>& initial_inputs);

    /// Feed next input and get output
    Tensor step(const Tensor& input);

    /// Get accumulated outputs
    std::vector<Tensor> get_outputs() const;

    /// Check if streaming is complete
    bool is_complete() const { return !running_; }

    /// Stop streaming
    void stop();

    /// Set callback for each step
    void set_step_callback(std::function<void(const Tensor&)> callback);

    /// Get number of steps taken
    size_t step_count() const { return step_count_; }

private:
    std::shared_ptr<InferenceSession> session_;
    std::vector<Tensor> accumulated_outputs_;
    std::function<void(const Tensor&)> step_callback_;
    std::unordered_map<std::string, Tensor> current_state_;
    bool running_ = false;
    size_t step_count_ = 0;
    mutable std::mutex mutex_;
};

/// Create async session with default options
std::unique_ptr<AsyncSession> create_async_session(
    std::shared_ptr<InferenceSession> session);

/// Create async session with custom options
std::unique_ptr<AsyncSession> create_async_session(
    std::shared_ptr<InferenceSession> session,
    const AsyncOptions& options);

} // namespace streaming
} // namespace pyflame_rt
