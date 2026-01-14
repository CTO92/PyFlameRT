#include "pyflame_rt/graph.hpp"
#include "pyflame_rt/errors.hpp"
#include <queue>
#include <unordered_set>

namespace pyflame_rt {

Graph::Graph(std::string name, GraphLimits limits)
    : name_(std::move(name))
    , limits_(limits)
{
}

void Graph::add_node(std::shared_ptr<Node> node) {
    if (node_map_.count(node->name()) > 0) {
        throw std::invalid_argument(
            "Node '" + node->name() + "' already exists in graph");
    }

    // Security fix MED-02: Enforce graph size limits
    if (nodes_.size() >= limits_.max_nodes) {
        throw std::runtime_error(
            "Graph node limit exceeded: max " + std::to_string(limits_.max_nodes) + " nodes allowed");
    }

    nodes_.push_back(node);
    node_map_[node->name()] = node;

    // Track tensor producers
    for (const auto& output : node->outputs()) {
        tensor_producers_[output] = node->name();
    }
}

Node* Graph::get_node(const std::string& name) {
    auto it = node_map_.find(name);
    return it != node_map_.end() ? it->second.get() : nullptr;
}

const Node* Graph::get_node(const std::string& name) const {
    auto it = node_map_.find(name);
    return it != node_map_.end() ? it->second.get() : nullptr;
}

void Graph::add_input(const TensorInfo& info) {
    inputs_.push_back(info);
}

void Graph::add_output(const TensorInfo& info) {
    outputs_.push_back(info);
}

void Graph::add_initializer(const std::string& name, Tensor tensor) {
    // Security fix MED-02: Enforce initializer count limits
    if (initializers_.size() >= limits_.max_initializers) {
        throw std::runtime_error(
            "Graph initializer limit exceeded: max " +
            std::to_string(limits_.max_initializers) + " initializers allowed");
    }

    // Security fix MED-02: Enforce total initializer size limits
    size_t tensor_bytes = tensor.size_bytes();
    if (total_initializer_bytes_ + tensor_bytes > limits_.max_initializer_bytes) {
        throw std::runtime_error(
            "Graph initializer memory limit exceeded: max " +
            std::to_string(limits_.max_initializer_bytes / (1024 * 1024)) + " MB allowed");
    }

    // Update or add initializer
    auto it = initializers_.find(name);
    if (it != initializers_.end()) {
        // Replacing existing - subtract old size first
        total_initializer_bytes_ -= it->second.size_bytes();
    }

    total_initializer_bytes_ += tensor_bytes;
    initializers_[name] = std::move(tensor);
}

// Security fix HIGH-01: Safe optional reference accessors
std::optional<std::reference_wrapper<Tensor>> Graph::get_initializer(const std::string& name) {
    auto it = initializers_.find(name);
    if (it != initializers_.end()) {
        return std::ref(it->second);
    }
    return std::nullopt;
}

std::optional<std::reference_wrapper<const Tensor>> Graph::get_initializer(const std::string& name) const {
    auto it = initializers_.find(name);
    if (it != initializers_.end()) {
        return std::cref(it->second);
    }
    return std::nullopt;
}

// Deprecated legacy raw pointer accessors
Tensor* Graph::get_initializer_unsafe(const std::string& name) {
    auto it = initializers_.find(name);
    return it != initializers_.end() ? &it->second : nullptr;
}

const Tensor* Graph::get_initializer_unsafe(const std::string& name) const {
    auto it = initializers_.find(name);
    return it != initializers_.end() ? &it->second : nullptr;
}

const std::string* Graph::get_producer(const std::string& tensor_name) const {
    auto it = tensor_producers_.find(tensor_name);
    return it != tensor_producers_.end() ? &it->second : nullptr;
}

std::vector<std::string> Graph::get_consumers(const std::string& tensor_name) const {
    std::vector<std::string> consumers;
    for (const auto& node : nodes_) {
        for (const auto& input : node->inputs()) {
            if (input == tensor_name) {
                consumers.push_back(node->name());
                break;
            }
        }
    }
    return consumers;
}

std::vector<std::shared_ptr<Node>> Graph::topological_sort() const {
    // Build in-degree map
    std::unordered_map<std::string, int> in_degree;
    std::unordered_map<std::string, std::vector<std::string>> dependents;

    for (const auto& node : nodes_) {
        in_degree[node->name()] = 0;
        dependents[node->name()] = {};
    }

    for (const auto& node : nodes_) {
        for (const auto& input : node->inputs()) {
            auto producer = get_producer(input);
            if (producer && in_degree.count(*producer)) {
                in_degree[node->name()]++;
                dependents[*producer].push_back(node->name());
            }
        }
    }

    // Kahn's algorithm
    std::queue<std::string> queue;
    for (const auto& [name, degree] : in_degree) {
        if (degree == 0) {
            queue.push(name);
        }
    }

    std::vector<std::shared_ptr<Node>> result;
    result.reserve(nodes_.size());

    while (!queue.empty()) {
        std::string name = queue.front();
        queue.pop();
        result.push_back(node_map_.at(name));

        for (const auto& dep : dependents[name]) {
            in_degree[dep]--;
            if (in_degree[dep] == 0) {
                queue.push(dep);
            }
        }
    }

    if (result.size() != nodes_.size()) {
        throw ValidationError({"Graph contains a cycle"});
    }

    return result;
}

std::vector<std::string> Graph::validate() const {
    std::vector<std::string> errors;
    std::unordered_set<std::string> available;

    // Graph inputs and initializers are available
    for (const auto& input : inputs_) {
        available.insert(input.name);
    }
    for (const auto& [name, _] : initializers_) {
        available.insert(name);
    }

    // Check nodes in topological order
    std::vector<std::shared_ptr<Node>> sorted;
    try {
        sorted = topological_sort();
    } catch (const ValidationError& e) {
        return e.errors();
    }

    for (const auto& node : sorted) {
        // Check inputs exist
        for (const auto& input : node->inputs()) {
            if (available.find(input) == available.end()) {
                errors.push_back(
                    "Node '" + node->name() + "' input '" + input + "' not found");
            }
        }
        // Mark outputs as available
        for (const auto& output : node->outputs()) {
            available.insert(output);
        }
    }

    // Check outputs exist
    for (const auto& output : outputs_) {
        if (available.find(output.name) == available.end()) {
            errors.push_back("Graph output '" + output.name + "' not produced");
        }
    }

    return errors;
}

} // namespace pyflame_rt
