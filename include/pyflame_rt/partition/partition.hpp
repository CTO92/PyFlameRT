#pragma once

#include "pyflame_rt/graph.hpp"
#include "pyflame_rt/tensor.hpp"
#include <memory>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <future>

namespace pyflame_rt {
namespace partition {

/// Device type for partitioning
enum class DeviceType {
    CPU,
    WSE,         // Cerebras WSE
    GPU,         // Future
    Distributed  // Multi-node
};

/// Device specification
struct DeviceSpec {
    /// Device type
    DeviceType type = DeviceType::CPU;

    /// Device ID
    int device_id = 0;

    /// Available memory (bytes)
    size_t memory_bytes = 0;

    /// Compute capacity (FLOPS)
    double compute_flops = 0.0;

    /// Memory bandwidth (bytes/sec)
    double memory_bandwidth = 0.0;

    /// Device name
    std::string name;
};

/// Multi-chip configuration for WSE
struct WSEChipConfig {
    /// Number of chips
    size_t num_chips = 1;

    /// Chip topology (e.g., 2x2 grid)
    std::vector<size_t> topology = {1, 1};

    /// Inter-chip bandwidth (bytes/sec)
    double inter_chip_bandwidth = 0.0;

    /// Inter-chip latency (nanoseconds)
    double inter_chip_latency = 0.0;

    /// Chip memory (bytes each)
    size_t chip_memory_bytes = 0;

    /// Get total chips
    size_t total_chips() const {
        size_t total = 1;
        for (auto d : topology) total *= d;
        return total;
    }
};

/// Partitioning strategy
enum class PartitionStrategy {
    Manual,          // User-specified partitioning
    DataParallel,    // Replicate model, split data
    ModelParallel,   // Split model across devices
    PipelineParallel,// Pipeline stages across devices
    Hybrid,          // Combination of strategies
    Automatic        // Cost-model based automatic partitioning
};

/// Communication type
enum class CommType {
    PointToPoint,    // Direct send/recv
    AllReduce,       // Reduce across all devices
    AllGather,       // Gather from all devices
    Scatter,         // Scatter to devices
    Broadcast        // Broadcast from one device
};

/// Communication operation
struct CommOp {
    /// Communication type
    CommType type;

    /// Source device(s)
    std::vector<int> src_devices;

    /// Destination device(s)
    std::vector<int> dst_devices;

    /// Tensor name
    std::string tensor_name;

    /// Data size (bytes)
    size_t data_bytes = 0;

    /// Estimated latency (microseconds)
    double estimated_latency_us = 0.0;
};

/// Graph partition (subgraph assigned to a device)
struct GraphPartition {
    /// Partition ID
    int partition_id = 0;

    /// Assigned device
    DeviceSpec device;

    /// Nodes in this partition
    std::vector<std::shared_ptr<Node>> nodes;

    /// Input tensors (from other partitions or external)
    std::vector<std::string> input_tensors;

    /// Output tensors (to other partitions or external)
    std::vector<std::string> output_tensors;

    /// Subgraph for this partition
    std::shared_ptr<Graph> subgraph;

    /// Memory footprint
    size_t memory_bytes = 0;

    /// Estimated compute time (microseconds)
    double compute_time_us = 0.0;
};

/// Partition plan (complete partitioning of a graph)
struct PartitionPlan {
    /// Original graph
    std::shared_ptr<Graph> original_graph;

    /// Partitions
    std::vector<GraphPartition> partitions;

    /// Communication operations
    std::vector<CommOp> communications;

    /// Node to partition mapping
    std::unordered_map<std::string, int> node_to_partition;

    /// Total estimated latency
    double total_latency_us = 0.0;

    /// Total communication volume (bytes)
    size_t total_comm_bytes = 0;

    /// Load balance metric (0 = perfect, higher = worse)
    double load_imbalance = 0.0;
};

/// Cost model for partitioning decisions
class CostModel {
public:
    CostModel();
    ~CostModel();

    // ========================================================================
    // Node Costs
    // ========================================================================

    /// Estimate compute cost for a node (microseconds)
    double estimate_compute_cost(
        const Node& node,
        const DeviceSpec& device) const;

    /// Estimate memory cost for a node (bytes)
    size_t estimate_memory_cost(const Node& node) const;

    /// Estimate activation memory
    size_t estimate_activation_memory(const Node& node) const;

    // ========================================================================
    // Communication Costs
    // ========================================================================

    /// Estimate communication cost (microseconds)
    double estimate_comm_cost(
        size_t data_bytes,
        const DeviceSpec& src,
        const DeviceSpec& dst) const;

    /// Estimate all-reduce cost
    double estimate_allreduce_cost(
        size_t data_bytes,
        const std::vector<DeviceSpec>& devices) const;

    // ========================================================================
    // Configuration
    // ========================================================================

    /// Set inter-device bandwidth
    void set_inter_device_bandwidth(double bytes_per_sec);

    /// Set inter-device latency
    void set_inter_device_latency(double microseconds);

    /// Get inter-device bandwidth
    double inter_device_bandwidth() const { return inter_device_bandwidth_; }

    /// Get inter-device latency
    double inter_device_latency() const { return inter_device_latency_; }

private:
    double inter_device_bandwidth_ = 1e9;  // 1 GB/s default
    double inter_device_latency_ = 10.0;   // 10 us default
};

/// Partitioning configuration
struct PartitionConfig {
    /// Target devices
    std::vector<DeviceSpec> devices;

    /// WSE multi-chip configuration
    WSEChipConfig wse_config;

    /// Partitioning strategy
    PartitionStrategy strategy = PartitionStrategy::Automatic;

    /// Manual partition assignments (node name -> device ID)
    std::unordered_map<std::string, int> manual_assignments;

    /// Maximum memory per partition (0 = device limit)
    size_t max_memory_per_partition = 0;

    /// Balance compute across partitions
    bool balance_compute = true;

    /// Minimize communication
    bool minimize_communication = true;

    /// Allow tensor replication
    bool allow_replication = false;

    /// Maximum pipeline stages (for pipeline parallel)
    size_t max_pipeline_stages = 0;

    /// Micro-batch size (for pipeline parallel)
    size_t micro_batch_size = 1;
};

/// Graph partitioner
class GraphPartitioner {
public:
    explicit GraphPartitioner(const PartitionConfig& config);
    ~GraphPartitioner();

    /// Set cost model
    void set_cost_model(std::shared_ptr<CostModel> cost_model);

    // ========================================================================
    // Partitioning
    // ========================================================================

    /// Partition a graph
    PartitionPlan partition(const Graph& graph);

    /// Partition with manual assignments
    PartitionPlan partition_manual(
        const Graph& graph,
        const std::unordered_map<std::string, int>& assignments);

    /// Partition for data parallelism
    PartitionPlan partition_data_parallel(
        const Graph& graph,
        size_t num_replicas);

    /// Partition for model parallelism
    PartitionPlan partition_model_parallel(
        const Graph& graph,
        size_t num_partitions);

    /// Partition for pipeline parallelism
    PartitionPlan partition_pipeline(
        const Graph& graph,
        size_t num_stages);

    // ========================================================================
    // Analysis
    // ========================================================================

    /// Analyze graph for partitioning
    struct AnalysisResult {
        /// Recommended strategy
        PartitionStrategy recommended_strategy;

        /// Estimated speedup
        double estimated_speedup = 1.0;

        /// Memory per partition
        std::vector<size_t> memory_per_partition;

        /// Compute per partition
        std::vector<double> compute_per_partition;

        /// Communication volume
        size_t communication_bytes = 0;

        /// Bottleneck nodes
        std::vector<std::string> bottleneck_nodes;
    };

    AnalysisResult analyze(const Graph& graph) const;

    // ========================================================================
    // Utilities
    // ========================================================================

    /// Insert communication nodes into graph
    static Graph insert_communication(
        const Graph& graph,
        const PartitionPlan& plan);

    /// Validate partition plan
    static bool validate(const PartitionPlan& plan);

    /// Estimate total execution time
    static double estimate_execution_time(const PartitionPlan& plan);

    /// Get partition statistics
    struct PartitionStats {
        size_t num_partitions = 0;
        size_t total_nodes = 0;
        size_t total_edges = 0;
        size_t cut_edges = 0;  // Edges crossing partitions
        double edge_cut_ratio = 0.0;
        std::vector<size_t> nodes_per_partition;
        std::vector<size_t> memory_per_partition;
    };

    static PartitionStats get_stats(const PartitionPlan& plan);

    /// Get configuration
    const PartitionConfig& config() const { return config_; }

private:
    PartitionConfig config_;
    std::shared_ptr<CostModel> cost_model_;

    PartitionPlan partition_automatic(const Graph& graph);
    void assign_communications(PartitionPlan& plan);
    void optimize_plan(PartitionPlan& plan);
    void build_subgraphs(PartitionPlan& plan);
};

/// Partitioned executor for running partitioned graphs
class PartitionedExecutor {
public:
    explicit PartitionedExecutor(const PartitionPlan& plan);
    ~PartitionedExecutor();

    /// Execute partitioned graph
    std::unordered_map<std::string, Tensor> execute(
        const std::unordered_map<std::string, Tensor>& inputs);

    /// Execute asynchronously
    std::future<std::unordered_map<std::string, Tensor>> execute_async(
        const std::unordered_map<std::string, Tensor>& inputs);

    /// Get the partition plan
    const PartitionPlan& plan() const { return plan_; }

private:
    PartitionPlan plan_;
    std::vector<std::shared_ptr<InferenceSession>> sessions_;
};

/// WSE-specific partitioning utilities
namespace wse {

/// Partition for WSE multi-chip
PartitionPlan partition_for_wse(
    const Graph& graph,
    const WSEChipConfig& config);

/// Insert WSE-specific communication ops
Graph insert_wse_communication(
    const Graph& graph,
    const PartitionPlan& plan);

/// Optimize for WSE dataflow
Graph optimize_for_wse_dataflow(const Graph& graph);

} // namespace wse

/// Convenience functions
PartitionPlan auto_partition(const Graph& graph, size_t num_devices);
PartitionPlan partition_for_multi_chip(const Graph& graph, const WSEChipConfig& config);

} // namespace partition
} // namespace pyflame_rt
