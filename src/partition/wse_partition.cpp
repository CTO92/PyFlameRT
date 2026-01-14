#include "pyflame_rt/partition/partition.hpp"
#include <algorithm>
#include <cmath>

namespace pyflame_rt {
namespace partition {
namespace wse {

// ============================================================================
// WSE-Specific Partitioning
// ============================================================================

PartitionPlan partition_for_wse(
    const Graph& graph,
    const WSEChipConfig& config) {

    PartitionConfig partition_config;
    partition_config.strategy = PartitionStrategy::ModelParallel;

    // Create device specs for each WSE chip
    size_t num_chips = config.total_chips();
    for (size_t i = 0; i < num_chips; ++i) {
        DeviceSpec device;
        device.type = DeviceType::WSE;
        device.device_id = static_cast<int>(i);
        device.memory_bytes = config.chip_memory_bytes;
        device.compute_flops = 1e15;  // ~1 PFLOP/s per chip
        device.memory_bandwidth = config.inter_chip_bandwidth;
        device.name = "WSE-Chip-" + std::to_string(i);

        partition_config.devices.push_back(device);
    }

    partition_config.wse_config = config;

    // Create cost model with WSE-specific parameters
    auto cost_model = std::make_shared<CostModel>();
    cost_model->set_inter_device_bandwidth(config.inter_chip_bandwidth);
    cost_model->set_inter_device_latency(config.inter_chip_latency / 1000.0);  // ns to us

    GraphPartitioner partitioner(partition_config);
    partitioner.set_cost_model(cost_model);

    // Use pipeline parallel for WSE (exploits its dataflow architecture)
    auto plan = partitioner.partition_pipeline(graph, num_chips);

    // Optimize for WSE topology
    // If 2D grid topology, assign partitions to minimize communication hops
    if (config.topology.size() == 2) {
        size_t rows = config.topology[0];
        size_t cols = config.topology[1];

        // Reorder partitions for 2D mesh locality
        // Snake pattern: row 0 left-to-right, row 1 right-to-left, etc.
        std::vector<int> chip_order;
        for (size_t r = 0; r < rows; ++r) {
            if (r % 2 == 0) {
                for (size_t c = 0; c < cols; ++c) {
                    chip_order.push_back(static_cast<int>(r * cols + c));
                }
            } else {
                for (size_t c = cols; c > 0; --c) {
                    chip_order.push_back(static_cast<int>(r * cols + (c - 1)));
                }
            }
        }

        // Reassign partitions to follow the snake pattern
        std::vector<GraphPartition> reordered_partitions = plan.partitions;
        for (size_t i = 0; i < std::min(plan.partitions.size(), chip_order.size()); ++i) {
            reordered_partitions[i].device.device_id = chip_order[i];
        }
        plan.partitions = reordered_partitions;
    }

    return plan;
}

Graph insert_wse_communication(
    const Graph& graph,
    const PartitionPlan& plan) {

    Graph result = graph;

    // WSE uses fabric transfers for inter-chip communication
    // Insert WSEFabricSend/WSEFabricRecv nodes

    for (const auto& comm : plan.communications) {
        if (comm.type == CommType::PointToPoint) {
            // Create WSE fabric send node
            auto send_node = std::make_shared<Node>("WSEFabricSend", "wse_send_" + comm.tensor_name);
            send_node->set_attribute("src_chip", comm.src_devices[0]);
            send_node->set_attribute("dst_chip", comm.dst_devices[0]);
            send_node->set_attribute("tensor_name", comm.tensor_name);

            // Create WSE fabric receive node
            auto recv_node = std::make_shared<Node>("WSEFabricRecv", "wse_recv_" + comm.tensor_name);
            recv_node->set_attribute("src_chip", comm.src_devices[0]);
            recv_node->set_attribute("dst_chip", comm.dst_devices[0]);
            recv_node->set_attribute("tensor_name", comm.tensor_name);

            // Note: In a full implementation, we'd insert these nodes
            // at the appropriate places in the graph
        } else if (comm.type == CommType::AllReduce) {
            // WSE all-reduce using ring topology
            auto allreduce_node = std::make_shared<Node>("WSEAllReduce", "wse_allreduce_" + comm.tensor_name);
            allreduce_node->set_attribute("tensor_name", comm.tensor_name);
            allreduce_node->set_attribute("num_chips", static_cast<int64_t>(comm.src_devices.size()));
        }
    }

    return result;
}

Graph optimize_for_wse_dataflow(const Graph& graph) {
    Graph optimized = graph;

    // WSE optimizations:
    // 1. Fuse small operations to reduce memory accesses
    // 2. Reorder operations for better dataflow
    // 3. Insert streaming boundaries

    auto sorted_nodes = graph.topological_sort();

    // Look for fusible patterns
    std::vector<std::shared_ptr<Node>> fused_nodes;
    std::unordered_set<std::string> fused_node_names;

    for (size_t i = 0; i < sorted_nodes.size(); ++i) {
        const auto& node = sorted_nodes[i];

        if (fused_node_names.count(node->name())) {
            continue;
        }

        // Pattern: MatMul followed by Add (bias) followed by activation
        if (node->op_type() == "MatMul" && i + 2 < sorted_nodes.size()) {
            const auto& next1 = sorted_nodes[i + 1];
            const auto& next2 = sorted_nodes[i + 2];

            bool can_fuse = false;

            // Check if MatMul -> Add -> Relu pattern
            if (next1->op_type() == "Add" &&
                (next2->op_type() == "Relu" || next2->op_type() == "Gelu")) {

                // Check if Add uses MatMul output
                for (const auto& input : next1->inputs()) {
                    for (const auto& output : node->outputs()) {
                        if (input.name == output.name) {
                            can_fuse = true;
                            break;
                        }
                    }
                }

                if (can_fuse) {
                    // Check if activation uses Add output
                    can_fuse = false;
                    for (const auto& input : next2->inputs()) {
                        for (const auto& output : next1->outputs()) {
                            if (input.name == output.name) {
                                can_fuse = true;
                                break;
                            }
                        }
                    }
                }
            }

            if (can_fuse) {
                // Create fused node
                auto fused = std::make_shared<Node>("WSEFusedMatMulBiasActivation",
                    "fused_" + node->name());

                // Copy inputs from MatMul
                for (const auto& input : node->inputs()) {
                    fused->add_input(input);
                }

                // Add bias input from Add
                for (const auto& input : next1->inputs()) {
                    bool is_matmul_output = false;
                    for (const auto& output : node->outputs()) {
                        if (input.name == output.name) {
                            is_matmul_output = true;
                            break;
                        }
                    }
                    if (!is_matmul_output) {
                        fused->add_input(input);
                    }
                }

                // Use activation output
                for (const auto& output : next2->outputs()) {
                    fused->add_output(output);
                }

                fused->set_attribute("activation", next2->op_type());

                fused_nodes.push_back(fused);
                fused_node_names.insert(node->name());
                fused_node_names.insert(next1->name());
                fused_node_names.insert(next2->name());
            }
        }

        if (fused_node_names.find(node->name()) == fused_node_names.end()) {
            fused_nodes.push_back(node);
        }
    }

    // Rebuild graph with fused nodes
    Graph result;
    for (const auto& input : graph.inputs()) {
        result.add_input(input);
    }
    for (const auto& output : graph.outputs()) {
        result.add_output(output);
    }
    for (const auto& [name, tensor] : graph.initializers()) {
        result.add_initializer(name, tensor);
    }
    for (const auto& node : fused_nodes) {
        result.add_node(node);
    }

    return result;
}

} // namespace wse
} // namespace partition
} // namespace pyflame_rt
