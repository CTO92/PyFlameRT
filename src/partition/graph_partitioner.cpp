#include "pyflame_rt/partition/partition.hpp"
#include <algorithm>
#include <queue>
#include <stdexcept>
#include <numeric>

namespace pyflame_rt {
namespace partition {

// ============================================================================
// GraphPartitioner Implementation
// ============================================================================

GraphPartitioner::GraphPartitioner(const PartitionConfig& config)
    : config_(config)
    , cost_model_(std::make_shared<CostModel>()) {
}

GraphPartitioner::~GraphPartitioner() = default;

void GraphPartitioner::set_cost_model(std::shared_ptr<CostModel> cost_model) {
    cost_model_ = std::move(cost_model);
}

PartitionPlan GraphPartitioner::partition(const Graph& graph) {
    switch (config_.strategy) {
        case PartitionStrategy::Manual:
            return partition_manual(graph, config_.manual_assignments);

        case PartitionStrategy::DataParallel:
            return partition_data_parallel(graph, config_.devices.size());

        case PartitionStrategy::ModelParallel:
            return partition_model_parallel(graph, config_.devices.size());

        case PartitionStrategy::PipelineParallel:
            return partition_pipeline(graph, config_.max_pipeline_stages > 0 ?
                config_.max_pipeline_stages : config_.devices.size());

        case PartitionStrategy::Automatic:
        case PartitionStrategy::Hybrid:
        default:
            return partition_automatic(graph);
    }
}

PartitionPlan GraphPartitioner::partition_manual(
    const Graph& graph,
    const std::unordered_map<std::string, int>& assignments) {

    PartitionPlan plan;
    plan.original_graph = std::make_shared<Graph>(graph);

    // Create partitions for each device
    size_t num_partitions = config_.devices.size();
    // Security: validate num_partitions > 0 (CRIT-PT1 fix)
    if (num_partitions == 0) {
        throw std::invalid_argument("Cannot partition with zero devices");
    }
    plan.partitions.resize(num_partitions);

    for (size_t i = 0; i < num_partitions; ++i) {
        plan.partitions[i].partition_id = static_cast<int>(i);
        if (i < config_.devices.size()) {
            plan.partitions[i].device = config_.devices[i];
        }
    }

    // Assign nodes to partitions
    for (const auto& node : graph.nodes()) {
        const std::string& name = node->name();
        int partition_id = 0;

        auto it = assignments.find(name);
        if (it != assignments.end()) {
            partition_id = it->second;
        }

        if (partition_id < 0 || static_cast<size_t>(partition_id) >= num_partitions) {
            throw std::runtime_error("Invalid partition assignment for node: " + name);
        }

        plan.partitions[partition_id].nodes.push_back(node);
        plan.node_to_partition[name] = partition_id;

        // Update memory footprint
        plan.partitions[partition_id].memory_bytes += cost_model_->estimate_memory_cost(*node);
        plan.partitions[partition_id].compute_time_us += cost_model_->estimate_compute_cost(
            *node, plan.partitions[partition_id].device);
    }

    // Determine input/output tensors for each partition
    assign_communications(plan);

    // Build subgraphs
    build_subgraphs(plan);

    // Calculate statistics
    for (const auto& partition : plan.partitions) {
        plan.total_latency_us = std::max(plan.total_latency_us, partition.compute_time_us);
    }

    for (const auto& comm : plan.communications) {
        plan.total_comm_bytes += comm.data_bytes;
        plan.total_latency_us += comm.estimated_latency_us;
    }

    return plan;
}

PartitionPlan GraphPartitioner::partition_data_parallel(
    const Graph& graph,
    size_t num_replicas) {

    // Security: validate num_replicas > 0 (CRIT-PT2 fix)
    if (num_replicas == 0) {
        throw std::invalid_argument("Cannot partition with zero replicas");
    }

    PartitionPlan plan;
    plan.original_graph = std::make_shared<Graph>(graph);

    // In data parallelism, we replicate the entire graph to each device
    // Each replica processes a different batch of data
    plan.partitions.resize(num_replicas);

    for (size_t i = 0; i < num_replicas; ++i) {
        plan.partitions[i].partition_id = static_cast<int>(i);
        if (i < config_.devices.size()) {
            plan.partitions[i].device = config_.devices[i];
        }

        // Each partition gets all nodes (replicated)
        for (const auto& node : graph.nodes()) {
            plan.partitions[i].nodes.push_back(node);
            plan.partitions[i].memory_bytes += cost_model_->estimate_memory_cost(*node);
            plan.partitions[i].compute_time_us += cost_model_->estimate_compute_cost(
                *node, plan.partitions[i].device);
        }

        // Map all nodes to this partition
        for (const auto& node : graph.nodes()) {
            // For data parallel, each node belongs to all partitions
            // but we track the primary assignment
            if (i == 0) {
                plan.node_to_partition[node->name()] = 0;
            }
        }

        plan.partitions[i].subgraph = std::make_shared<Graph>(graph);
    }

    // Add all-reduce communication for gradient synchronization (if training)
    // For inference, just scatter input and gather output
    for (const auto& input : graph.inputs()) {
        CommOp scatter;
        scatter.type = CommType::Scatter;
        scatter.src_devices = {0};
        for (size_t i = 0; i < num_replicas; ++i) {
            scatter.dst_devices.push_back(static_cast<int>(i));
        }
        scatter.tensor_name = input.name;

        // Estimate data size
        size_t elements = 1;
        for (auto dim : input.shape) {
            if (dim.has_value() && dim.value() > 0) {
                elements *= static_cast<size_t>(dim.value());
            }
        }
        scatter.data_bytes = elements * dtype_size(input.dtype);
        scatter.estimated_latency_us = cost_model_->estimate_comm_cost(
            scatter.data_bytes, config_.devices[0],
            num_replicas > 1 ? config_.devices[1] : config_.devices[0]);

        plan.communications.push_back(scatter);
    }

    // Gather outputs
    for (const auto& output : graph.outputs()) {
        CommOp gather;
        gather.type = CommType::AllGather;
        for (size_t i = 0; i < num_replicas; ++i) {
            gather.src_devices.push_back(static_cast<int>(i));
        }
        gather.dst_devices = {0};
        gather.tensor_name = output.name;

        size_t elements = 1;
        for (auto dim : output.shape) {
            if (dim.has_value() && dim.value() > 0) {
                elements *= static_cast<size_t>(dim.value());
            }
        }
        gather.data_bytes = elements * dtype_size(output.dtype);
        gather.estimated_latency_us = cost_model_->estimate_comm_cost(
            gather.data_bytes, config_.devices[0],
            num_replicas > 1 ? config_.devices[1] : config_.devices[0]);

        plan.communications.push_back(gather);
    }

    // Calculate total latency (compute is parallel, comm is serial)
    if (!plan.partitions.empty()) {
        plan.total_latency_us = plan.partitions[0].compute_time_us;
    }
    for (const auto& comm : plan.communications) {
        plan.total_comm_bytes += comm.data_bytes;
        plan.total_latency_us += comm.estimated_latency_us;
    }

    return plan;
}

PartitionPlan GraphPartitioner::partition_model_parallel(
    const Graph& graph,
    size_t num_partitions) {

    // Security: validate num_partitions > 0 (CRIT-PT3 fix)
    if (num_partitions == 0) {
        throw std::invalid_argument("Cannot partition with zero partitions");
    }

    PartitionPlan plan;
    plan.original_graph = std::make_shared<Graph>(graph);
    plan.partitions.resize(num_partitions);

    for (size_t i = 0; i < num_partitions; ++i) {
        plan.partitions[i].partition_id = static_cast<int>(i);
        if (i < config_.devices.size()) {
            plan.partitions[i].device = config_.devices[i];
        }
    }

    // Get topological order
    auto sorted_nodes = graph.topological_sort();

    // Simple model parallel: divide nodes evenly
    size_t nodes_per_partition = (sorted_nodes.size() + num_partitions - 1) / num_partitions;

    for (size_t i = 0; i < sorted_nodes.size(); ++i) {
        size_t partition_id = std::min(i / nodes_per_partition, num_partitions - 1);
        const auto& node = sorted_nodes[i];

        plan.partitions[partition_id].nodes.push_back(node);
        plan.node_to_partition[node->name()] = static_cast<int>(partition_id);

        plan.partitions[partition_id].memory_bytes += cost_model_->estimate_memory_cost(*node);
        plan.partitions[partition_id].compute_time_us += cost_model_->estimate_compute_cost(
            *node, plan.partitions[partition_id].device);
    }

    // Assign communications
    assign_communications(plan);

    // Build subgraphs
    build_subgraphs(plan);

    // Calculate statistics
    for (const auto& partition : plan.partitions) {
        plan.total_latency_us += partition.compute_time_us;
    }

    for (const auto& comm : plan.communications) {
        plan.total_comm_bytes += comm.data_bytes;
        plan.total_latency_us += comm.estimated_latency_us;
    }

    return plan;
}

PartitionPlan GraphPartitioner::partition_pipeline(
    const Graph& graph,
    size_t num_stages) {

    // Security: validate num_stages > 0 (CRIT-PT4 fix)
    if (num_stages == 0) {
        throw std::invalid_argument("Cannot partition with zero stages");
    }

    PartitionPlan plan;
    plan.original_graph = std::make_shared<Graph>(graph);
    plan.partitions.resize(num_stages);

    for (size_t i = 0; i < num_stages; ++i) {
        plan.partitions[i].partition_id = static_cast<int>(i);
        if (i < config_.devices.size()) {
            plan.partitions[i].device = config_.devices[i];
        }
    }

    // Get topological order
    auto sorted_nodes = graph.topological_sort();

    // For pipeline parallelism, we want to balance compute time across stages
    std::vector<double> compute_times;
    double total_compute = 0.0;

    for (const auto& node : sorted_nodes) {
        DeviceSpec dummy_device;
        double time = cost_model_->estimate_compute_cost(*node, dummy_device);
        compute_times.push_back(time);
        total_compute += time;
    }

    double target_per_stage = total_compute / static_cast<double>(num_stages);

    // Assign nodes to stages based on compute time
    size_t current_stage = 0;
    double current_stage_compute = 0.0;

    for (size_t i = 0; i < sorted_nodes.size(); ++i) {
        const auto& node = sorted_nodes[i];

        // Move to next stage if current is full (but not for last stage)
        if (current_stage < num_stages - 1 &&
            current_stage_compute >= target_per_stage) {
            current_stage++;
            current_stage_compute = 0.0;
        }

        plan.partitions[current_stage].nodes.push_back(node);
        plan.node_to_partition[node->name()] = static_cast<int>(current_stage);

        plan.partitions[current_stage].memory_bytes += cost_model_->estimate_memory_cost(*node);
        plan.partitions[current_stage].compute_time_us += compute_times[i];
        current_stage_compute += compute_times[i];
    }

    // Assign communications
    assign_communications(plan);

    // Build subgraphs
    build_subgraphs(plan);

    // Pipeline latency calculation
    // Steady-state: max(stage_compute_times)
    // Startup/drain: (num_stages - 1) * micro_batch_time
    double max_stage_time = 0.0;
    for (const auto& partition : plan.partitions) {
        max_stage_time = std::max(max_stage_time, partition.compute_time_us);
    }

    // Calculate load imbalance
    double avg_stage_time = total_compute / static_cast<double>(num_stages);
    plan.load_imbalance = (max_stage_time - avg_stage_time) / avg_stage_time;

    // Estimate total latency for a single batch
    for (const auto& partition : plan.partitions) {
        plan.total_latency_us += partition.compute_time_us;
    }

    for (const auto& comm : plan.communications) {
        plan.total_comm_bytes += comm.data_bytes;
        plan.total_latency_us += comm.estimated_latency_us;
    }

    return plan;
}

PartitionPlan GraphPartitioner::partition_automatic(const Graph& graph) {
    // Analyze the graph to determine the best strategy
    auto analysis = analyze(graph);

    switch (analysis.recommended_strategy) {
        case PartitionStrategy::DataParallel:
            return partition_data_parallel(graph, config_.devices.size());

        case PartitionStrategy::ModelParallel:
            return partition_model_parallel(graph, config_.devices.size());

        case PartitionStrategy::PipelineParallel:
            return partition_pipeline(graph, config_.devices.size());

        default:
            // Default to model parallel
            return partition_model_parallel(graph, config_.devices.size());
    }
}

void GraphPartitioner::assign_communications(PartitionPlan& plan) {
    // Build a map of which partition produces each tensor
    std::unordered_map<std::string, int> tensor_producer;

    // Graph inputs are produced externally (partition -1)
    for (const auto& input : plan.original_graph->inputs()) {
        tensor_producer[input.name] = -1;
    }

    // Initializers are available to all partitions
    for (const auto& init : plan.original_graph->initializers()) {
        tensor_producer[init.first] = -2;  // Special: available everywhere
    }

    // Each node's outputs are produced by its partition
    for (const auto& partition : plan.partitions) {
        for (const auto& node : partition.nodes) {
            for (const auto& output : node->outputs()) {
                tensor_producer[output.name] = partition.partition_id;
            }
        }
    }

    // For each partition, determine inputs and outputs
    for (auto& partition : plan.partitions) {
        std::unordered_set<std::string> internal_tensors;
        std::unordered_set<std::string> needed_tensors;
        std::unordered_set<std::string> produced_tensors;

        // Collect all tensors produced in this partition
        for (const auto& node : partition.nodes) {
            for (const auto& output : node->outputs()) {
                internal_tensors.insert(output.name);
                produced_tensors.insert(output.name);
            }
        }

        // Collect all tensors needed by this partition
        for (const auto& node : partition.nodes) {
            for (const auto& input : node->inputs()) {
                if (internal_tensors.find(input.name) == internal_tensors.end()) {
                    needed_tensors.insert(input.name);
                }
            }
        }

        // Input tensors: needed but not produced internally
        for (const auto& tensor : needed_tensors) {
            partition.input_tensors.push_back(tensor);

            // Add communication if from another partition
            auto it = tensor_producer.find(tensor);
            if (it != tensor_producer.end() && it->second >= 0 &&
                it->second != partition.partition_id) {

                CommOp comm;
                comm.type = CommType::PointToPoint;
                comm.src_devices = {it->second};
                comm.dst_devices = {partition.partition_id};
                comm.tensor_name = tensor;

                // Estimate data size (we don't have exact shape info here)
                comm.data_bytes = 1024 * 1024;  // Default 1MB estimate
                comm.estimated_latency_us = cost_model_->estimate_comm_cost(
                    comm.data_bytes,
                    plan.partitions[it->second].device,
                    partition.device);

                plan.communications.push_back(comm);
            }
        }

        // Output tensors: check if any other partition needs them
        for (const auto& tensor : produced_tensors) {
            bool is_graph_output = false;
            for (const auto& graph_output : plan.original_graph->outputs()) {
                if (graph_output.name == tensor) {
                    is_graph_output = true;
                    break;
                }
            }

            if (is_graph_output) {
                partition.output_tensors.push_back(tensor);
            }
        }
    }
}

void GraphPartitioner::optimize_plan(PartitionPlan& plan) {
    // Optimization 1: Merge small communications
    // Optimization 2: Reorder communications to overlap with compute
    // Optimization 3: Balance load by moving boundary nodes

    // For now, just sort communications by source partition for better scheduling
    std::sort(plan.communications.begin(), plan.communications.end(),
        [](const CommOp& a, const CommOp& b) {
            if (a.src_devices.empty() || b.src_devices.empty()) return false;
            return a.src_devices[0] < b.src_devices[0];
        });
}

void GraphPartitioner::build_subgraphs(PartitionPlan& plan) {
    for (auto& partition : plan.partitions) {
        auto subgraph = std::make_shared<Graph>();

        // Add nodes to subgraph
        for (const auto& node : partition.nodes) {
            subgraph->add_node(node);
        }

        // Add inputs
        for (const auto& input_name : partition.input_tensors) {
            // Find the tensor info
            for (const auto& graph_input : plan.original_graph->inputs()) {
                if (graph_input.name == input_name) {
                    subgraph->add_input(graph_input);
                    break;
                }
            }
        }

        // Add outputs
        for (const auto& output_name : partition.output_tensors) {
            for (const auto& graph_output : plan.original_graph->outputs()) {
                if (graph_output.name == output_name) {
                    subgraph->add_output(graph_output);
                    break;
                }
            }
        }

        // Copy relevant initializers
        for (const auto& node : partition.nodes) {
            for (const auto& input : node->inputs()) {
                auto init = plan.original_graph->get_initializer(input.name);
                if (init) {
                    subgraph->add_initializer(input.name, *init);
                }
            }
        }

        partition.subgraph = subgraph;
    }
}

GraphPartitioner::AnalysisResult GraphPartitioner::analyze(const Graph& graph) const {
    AnalysisResult result;

    auto sorted_nodes = graph.topological_sort();

    // Calculate total memory and compute
    size_t total_memory = 0;
    double total_compute = 0.0;
    size_t num_devices = config_.devices.size();

    DeviceSpec default_device;
    if (!config_.devices.empty()) {
        default_device = config_.devices[0];
    }

    for (const auto& node : sorted_nodes) {
        total_memory += cost_model_->estimate_memory_cost(*node);
        total_compute += cost_model_->estimate_compute_cost(*node, default_device);
    }

    // Estimate communication if we partition
    size_t estimated_comm = 0;
    if (num_devices > 1) {
        // Rough estimate: each partition boundary requires data transfer
        size_t num_boundaries = num_devices - 1;
        estimated_comm = total_memory / num_devices * num_boundaries;
    }

    result.communication_bytes = estimated_comm;

    // Determine recommended strategy
    // Heuristics:
    // - If model fits in one device, prefer data parallel for throughput
    // - If model doesn't fit, use model/pipeline parallel
    // - If compute-heavy with small activations, prefer model parallel
    // - If many sequential dependencies, prefer pipeline parallel

    bool model_fits = true;
    if (!config_.devices.empty()) {
        size_t device_memory = config_.devices[0].memory_bytes;
        if (device_memory > 0) {
            model_fits = total_memory < device_memory;
        }
    }

    if (model_fits && num_devices > 1) {
        result.recommended_strategy = PartitionStrategy::DataParallel;
        result.estimated_speedup = static_cast<double>(num_devices) * 0.8;  // 80% efficiency
    } else if (!model_fits) {
        // Need to split the model
        // Check if pipeline is better (sequential dependencies)
        size_t sequential_depth = sorted_nodes.size();  // Simplified
        if (sequential_depth > num_devices * 2) {
            result.recommended_strategy = PartitionStrategy::PipelineParallel;
        } else {
            result.recommended_strategy = PartitionStrategy::ModelParallel;
        }
        result.estimated_speedup = static_cast<double>(num_devices) * 0.6;  // 60% efficiency for model parallel
    } else {
        result.recommended_strategy = PartitionStrategy::DataParallel;
        result.estimated_speedup = 1.0;
    }

    // Calculate per-partition estimates
    // Security: only calculate if num_devices > 0 (CRIT-PT5 fix)
    if (num_devices > 0) {
        result.memory_per_partition.resize(num_devices);
        result.compute_per_partition.resize(num_devices);

        for (size_t i = 0; i < num_devices; ++i) {
            result.memory_per_partition[i] = total_memory / num_devices;
            result.compute_per_partition[i] = total_compute / num_devices;
        }
    }

    // Find bottleneck nodes (top compute consumers)
    std::vector<std::pair<std::string, double>> node_costs;
    for (const auto& node : sorted_nodes) {
        double cost = cost_model_->estimate_compute_cost(*node, default_device);
        node_costs.emplace_back(node->name(), cost);
    }

    std::sort(node_costs.begin(), node_costs.end(),
        [](const auto& a, const auto& b) { return a.second > b.second; });

    for (size_t i = 0; i < std::min(size_t(5), node_costs.size()); ++i) {
        result.bottleneck_nodes.push_back(node_costs[i].first);
    }

    return result;
}

Graph GraphPartitioner::insert_communication(
    const Graph& graph,
    const PartitionPlan& plan) {

    Graph result = graph;

    // Insert Send/Recv nodes at partition boundaries
    for (const auto& comm : plan.communications) {
        if (comm.type == CommType::PointToPoint) {
            // Create Send node
            auto send_node = std::make_shared<Node>("Send", "send_" + comm.tensor_name);
            // Set attributes for device routing
            send_node->set_attribute("src_device", comm.src_devices[0]);
            send_node->set_attribute("dst_device", comm.dst_devices[0]);

            // Create Recv node
            auto recv_node = std::make_shared<Node>("Recv", "recv_" + comm.tensor_name);
            recv_node->set_attribute("src_device", comm.src_devices[0]);
            recv_node->set_attribute("dst_device", comm.dst_devices[0]);
        }
    }

    return result;
}

bool GraphPartitioner::validate(const PartitionPlan& plan) {
    if (!plan.original_graph) {
        return false;
    }

    // Check all nodes are assigned
    std::unordered_set<std::string> assigned_nodes;
    for (const auto& partition : plan.partitions) {
        for (const auto& node : partition.nodes) {
            if (assigned_nodes.count(node->name())) {
                return false;  // Duplicate assignment
            }
            assigned_nodes.insert(node->name());
        }
    }

    // Check all original nodes are assigned
    for (const auto& node : plan.original_graph->nodes()) {
        if (assigned_nodes.find(node->name()) == assigned_nodes.end()) {
            return false;  // Missing node
        }
    }

    return true;
}

double GraphPartitioner::estimate_execution_time(const PartitionPlan& plan) {
    return plan.total_latency_us;
}

GraphPartitioner::PartitionStats GraphPartitioner::get_stats(const PartitionPlan& plan) {
    PartitionStats stats;

    stats.num_partitions = plan.partitions.size();

    for (const auto& partition : plan.partitions) {
        stats.total_nodes += partition.nodes.size();
        stats.nodes_per_partition.push_back(partition.nodes.size());
        stats.memory_per_partition.push_back(partition.memory_bytes);
    }

    // Count edges and cut edges
    std::unordered_map<std::string, int> tensor_partition;
    for (const auto& partition : plan.partitions) {
        for (const auto& node : partition.nodes) {
            for (const auto& output : node->outputs()) {
                tensor_partition[output.name] = partition.partition_id;
            }
        }
    }

    for (const auto& partition : plan.partitions) {
        for (const auto& node : partition.nodes) {
            for (const auto& input : node->inputs()) {
                stats.total_edges++;
                auto it = tensor_partition.find(input.name);
                if (it != tensor_partition.end() && it->second != partition.partition_id) {
                    stats.cut_edges++;
                }
            }
        }
    }

    if (stats.total_edges > 0) {
        stats.edge_cut_ratio = static_cast<double>(stats.cut_edges) /
                               static_cast<double>(stats.total_edges);
    }

    return stats;
}

// ============================================================================
// Convenience Functions
// ============================================================================

PartitionPlan auto_partition(const Graph& graph, size_t num_devices) {
    PartitionConfig config;
    config.strategy = PartitionStrategy::Automatic;

    for (size_t i = 0; i < num_devices; ++i) {
        DeviceSpec device;
        device.device_id = static_cast<int>(i);
        device.type = DeviceType::CPU;
        config.devices.push_back(device);
    }

    GraphPartitioner partitioner(config);
    return partitioner.partition(graph);
}

PartitionPlan partition_for_multi_chip(const Graph& graph, const WSEChipConfig& config) {
    return wse::partition_for_wse(graph, config);
}

} // namespace partition
} // namespace pyflame_rt
