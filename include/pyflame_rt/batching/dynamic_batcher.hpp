#pragma once

#include "pyflame_rt/tensor.hpp"
#include "pyflame_rt/session.hpp"
#include <chrono>
#include <condition_variable>
#include <functional>
#include <future>
#include <memory>
#include <mutex>
#include <queue>
#include <thread>
#include <vector>
#include <atomic>

namespace pyflame_rt {
namespace batching {

/// Batching configuration
struct BatchConfig {
    /// Maximum batch size
    size_t max_batch_size = 32;

    /// Maximum time to wait for batch to fill (microseconds)
    std::chrono::microseconds max_latency = std::chrono::microseconds(1000);

    /// Minimum batch size before executing (0 = execute immediately)
    size_t min_batch_size = 1;

    /// Preferred batch sizes (for padding optimization)
    std::vector<size_t> preferred_sizes = {1, 2, 4, 8, 16, 32};

    /// Enable padding to preferred batch sizes
    bool enable_padding = true;

    /// Number of worker threads (0 = auto-detect)
    size_t num_workers = 0;

    /// Queue capacity (0 = unlimited)
    size_t queue_capacity = 1000;

    /// Timeout for queue operations (0 = block forever)
    std::chrono::milliseconds queue_timeout = std::chrono::milliseconds(5000);
};

/// Single inference request
struct InferenceRequest {
    /// Unique request ID (assigned by batcher)
    uint64_t id = 0;

    /// Input tensors
    std::unordered_map<std::string, Tensor> inputs;

    /// Requested output names (empty = all)
    std::vector<std::string> output_names;

    /// Request arrival time
    std::chrono::steady_clock::time_point arrival_time;

    /// Priority (higher = process sooner)
    int priority = 0;
};

/// Inference response
struct InferenceResponse {
    /// Request ID this response corresponds to
    uint64_t request_id = 0;

    /// Output tensors
    std::unordered_map<std::string, Tensor> outputs;

    /// Request processing latency
    std::chrono::microseconds latency{0};

    /// Whether inference succeeded
    bool success = true;

    /// Error message (if !success)
    std::string error_message;
};

/// Batch of requests for processing
struct RequestBatch {
    /// Requests in this batch
    std::vector<InferenceRequest> requests;

    /// Batched input tensors
    std::unordered_map<std::string, Tensor> batched_inputs;

    /// Batch size (may include padding)
    size_t batch_size = 0;

    /// Actual number of requests (excluding padding)
    size_t num_requests = 0;

    /// Create batched tensors from individual requests
    void create_batched_inputs(const std::vector<std::string>& input_names);

    /// Split batched outputs back to individual responses
    std::vector<InferenceResponse> split_outputs(
        const std::unordered_map<std::string, Tensor>& batched_outputs);
};

/// Dynamic batcher for automatic request batching
class DynamicBatcher {
public:
    DynamicBatcher(std::shared_ptr<InferenceSession> session,
                   const BatchConfig& config = BatchConfig());
    ~DynamicBatcher();

    // Non-copyable
    DynamicBatcher(const DynamicBatcher&) = delete;
    DynamicBatcher& operator=(const DynamicBatcher&) = delete;

    // ========================================================================
    // Inference API
    // ========================================================================

    /// Submit inference request (async)
    std::future<InferenceResponse> submit(InferenceRequest request);

    /// Submit inference request with callback
    uint64_t submit(InferenceRequest request,
                    std::function<void(InferenceResponse)> callback);

    /// Submit batch of requests
    std::vector<std::future<InferenceResponse>> submit_batch(
        std::vector<InferenceRequest> requests);

    /// Blocking inference (waits for result)
    InferenceResponse infer(InferenceRequest request);

    // ========================================================================
    // Control
    // ========================================================================

    /// Start the batcher
    void start();

    /// Stop the batcher (waits for pending requests)
    void stop();

    /// Check if batcher is running
    bool is_running() const { return running_; }

    /// Flush pending requests (force execute)
    void flush();

    /// Get queue size
    size_t queue_size() const;

    // ========================================================================
    // Statistics
    // ========================================================================

    struct Stats {
        uint64_t total_requests = 0;
        uint64_t total_batches = 0;
        uint64_t dropped_requests = 0;
        double avg_batch_size = 0.0;
        double avg_latency_us = 0.0;
        double throughput_rps = 0.0;
        size_t queue_depth = 0;
    };

    Stats get_stats() const;
    void reset_stats();

    // ========================================================================
    // Configuration
    // ========================================================================

    const BatchConfig& config() const { return config_; }
    void update_config(const BatchConfig& config);

protected:
    std::shared_ptr<InferenceSession> session_;
    BatchConfig config_;

    std::atomic<bool> running_{false};
    std::atomic<uint64_t> next_request_id_{0};

    // Request queue
    struct PendingRequest {
        InferenceRequest request;
        std::promise<InferenceResponse> promise;
        std::function<void(InferenceResponse)> callback;
    };
    std::queue<PendingRequest> request_queue_;
    mutable std::mutex queue_mutex_;
    std::condition_variable queue_cv_;

    // Callback dispatch
    std::queue<std::pair<std::function<void(InferenceResponse)>, InferenceResponse>> callback_queue_;
    mutable std::mutex callback_mutex_;
    std::condition_variable callback_cv_;

    // Promise map for tracking pending promises
    std::unordered_map<uint64_t, std::promise<InferenceResponse>*> pending_promises_;
    std::unordered_map<uint64_t, std::function<void(InferenceResponse)>> pending_callbacks_;
    mutable std::mutex pending_mutex_;

    // Worker threads
    std::vector<std::thread> workers_;
    std::thread callback_worker_;

    // Statistics
    mutable std::mutex stats_mutex_;
    Stats stats_;
    std::chrono::steady_clock::time_point start_time_;

    void worker_loop();
    void callback_loop();
    RequestBatch create_batch();
    void process_batch(RequestBatch& batch);
    void dispatch_response(uint64_t request_id, InferenceResponse response);
};

/// Priority queue batcher (processes high-priority requests first)
class PriorityBatcher : public DynamicBatcher {
public:
    PriorityBatcher(std::shared_ptr<InferenceSession> session,
                    const BatchConfig& config = BatchConfig());

    /// Submit high-priority request
    std::future<InferenceResponse> submit_priority(
        InferenceRequest request, int priority);

private:
    // Priority queue comparator
    struct RequestComparator {
        bool operator()(const PendingRequest& a, const PendingRequest& b) const {
            return a.request.priority < b.request.priority;
        }
    };

    // Override with priority queue
    std::priority_queue<PendingRequest, std::vector<PendingRequest>, RequestComparator> priority_queue_;
};

/// Create a simple batcher with default config
std::unique_ptr<DynamicBatcher> create_batcher(
    std::shared_ptr<InferenceSession> session);

/// Create a batcher with custom config
std::unique_ptr<DynamicBatcher> create_batcher(
    std::shared_ptr<InferenceSession> session,
    const BatchConfig& config);

} // namespace batching
} // namespace pyflame_rt
