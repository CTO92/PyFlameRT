#include "pyflame_rt/partition/partition.hpp"
#include "pyflame_rt/session.hpp"
#include <thread>
#include <algorithm>

namespace pyflame_rt {
namespace partition {

// ============================================================================
// PartitionedExecutor Implementation
// ============================================================================

PartitionedExecutor::PartitionedExecutor(const PartitionPlan& plan)
    : plan_(plan) {

    // Create an InferenceSession for each partition
    sessions_.reserve(plan_.partitions.size());

    for (const auto& partition : plan_.partitions) {
        if (partition.subgraph) {
            SessionOptions options;

            // Configure based on device type
            switch (partition.device.type) {
                case DeviceType::CPU:
                    options.execution_provider = "CPU";
                    break;
                case DeviceType::WSE:
                    options.execution_provider = "WSE";
                    break;
                case DeviceType::GPU:
                    options.execution_provider = "CUDA";
                    break;
                case DeviceType::Distributed:
                    options.execution_provider = "Distributed";
                    break;
            }

            // Create session from the subgraph
            auto session = std::make_shared<InferenceSession>(*partition.subgraph, options);
            sessions_.push_back(session);
        } else {
            sessions_.push_back(nullptr);
        }
    }
}

PartitionedExecutor::~PartitionedExecutor() = default;

std::unordered_map<std::string, Tensor> PartitionedExecutor::execute(
    const std::unordered_map<std::string, Tensor>& inputs) {

    std::unordered_map<std::string, Tensor> all_tensors = inputs;
    std::unordered_map<std::string, Tensor> outputs;

    // Execute partitions in topological order
    // For pipeline parallel, we'd want to overlap execution
    // For now, execute sequentially

    for (size_t i = 0; i < plan_.partitions.size(); ++i) {
        const auto& partition = plan_.partitions[i];
        auto& session = sessions_[i];

        if (!session) continue;

        // Gather inputs for this partition
        std::unordered_map<std::string, Tensor> partition_inputs;
        for (const auto& input_name : partition.input_tensors) {
            auto it = all_tensors.find(input_name);
            if (it != all_tensors.end()) {
                partition_inputs[input_name] = it->second;
            }
        }

        // Execute partition
        auto partition_outputs = session->run({}, partition_inputs);

        // Store outputs for next partitions
        for (const auto& [name, tensor] : partition_outputs) {
            all_tensors[name] = tensor;
        }

        // Collect final outputs
        for (const auto& output_name : partition.output_tensors) {
            auto it = partition_outputs.find(output_name);
            if (it != partition_outputs.end()) {
                outputs[output_name] = it->second;
            }
        }
    }

    // If no explicit outputs collected, return all tensors from last partition
    if (outputs.empty() && !plan_.partitions.empty()) {
        const auto& last_partition = plan_.partitions.back();
        auto& last_session = sessions_.back();
        // Security: check if original_graph is valid (CRIT-PT6 fix)
        if (last_session && plan_.original_graph) {
            // The last partition outputs are the graph outputs
            for (const auto& [name, tensor] : all_tensors) {
                // Check if this is a graph output
                for (const auto& graph_output : plan_.original_graph->outputs()) {
                    if (graph_output.name == name) {
                        outputs[name] = tensor;
                        break;
                    }
                }
            }
        }
    }

    return outputs;
}

std::future<std::unordered_map<std::string, Tensor>> PartitionedExecutor::execute_async(
    const std::unordered_map<std::string, Tensor>& inputs) {

    // Copy inputs for the async task
    auto inputs_copy = inputs;

    return std::async(std::launch::async, [this, inputs_copy]() {
        return this->execute(inputs_copy);
    });
}

} // namespace partition
} // namespace pyflame_rt
