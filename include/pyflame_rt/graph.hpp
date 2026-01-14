#pragma once

#include "pyflame_rt/node.hpp"
#include "pyflame_rt/tensor.hpp"
#include "pyflame_rt/types.hpp"
#include <memory>
#include <vector>
#include <unordered_map>
#include <string>
#include <optional>
#include <functional>

namespace pyflame_rt {

/// Graph size limits for security (prevents resource exhaustion)
/// MED-02 fix: Reduced default limits to prevent DoS attacks
struct GraphLimits {
    size_t max_nodes = 100000;          // 100k nodes max (was 10M)
    size_t max_initializers = 50000;    // 50k initializers max (was 1M)
    size_t max_initializer_bytes = 4ULL * 1024 * 1024 * 1024;  // 4 GB max (was 16 GB)

    /// Create limits for large models (use with caution)
    static GraphLimits large_model() {
        return {
            .max_nodes = 1000000,          // 1M nodes
            .max_initializers = 500000,    // 500k initializers
            .max_initializer_bytes = 16ULL * 1024 * 1024 * 1024  // 16 GB
        };
    }
};

/// Represents a computational graph of operations
class Graph {
public:
    explicit Graph(std::string name = "", GraphLimits limits = {});

    // Graph info
    const std::string& name() const { return name_; }
    size_t num_nodes() const { return nodes_.size(); }

    // Node management
    void add_node(std::shared_ptr<Node> node);
    Node* get_node(const std::string& name);
    const Node* get_node(const std::string& name) const;
    const std::vector<std::shared_ptr<Node>>& nodes() const { return nodes_; }

    // Input/output management
    void add_input(const TensorInfo& info);
    void add_output(const TensorInfo& info);
    const std::vector<TensorInfo>& inputs() const { return inputs_; }
    const std::vector<TensorInfo>& outputs() const { return outputs_; }

    // Initializers (weights, biases)
    void add_initializer(const std::string& name, Tensor tensor);

    /// Get initializer by name (returns nullopt if not found)
    /// Security: Returns optional reference instead of raw pointer
    /// to prevent dangling pointer issues after map modifications
    std::optional<std::reference_wrapper<Tensor>> get_initializer(const std::string& name);
    std::optional<std::reference_wrapper<const Tensor>> get_initializer(const std::string& name) const;

    const std::unordered_map<std::string, Tensor>& initializers() const {
        return initializers_;
    }
    bool has_initializer(const std::string& name) const {
        return initializers_.count(name) > 0;
    }

    /// Get current total size of all initializers in bytes
    size_t total_initializer_bytes() const { return total_initializer_bytes_; }

    // Tensor producer tracking
    const std::string* get_producer(const std::string& tensor_name) const;
    std::vector<std::string> get_consumers(const std::string& tensor_name) const;

    // Graph operations
    std::vector<std::shared_ptr<Node>> topological_sort() const;
    std::vector<std::string> validate() const;

    // Iteration
    auto begin() { return nodes_.begin(); }
    auto end() { return nodes_.end(); }
    auto begin() const { return nodes_.begin(); }
    auto end() const { return nodes_.end(); }

private:
    std::string name_;
    std::vector<std::shared_ptr<Node>> nodes_;
    std::unordered_map<std::string, std::shared_ptr<Node>> node_map_;
    std::vector<TensorInfo> inputs_;
    std::vector<TensorInfo> outputs_;
    std::unordered_map<std::string, Tensor> initializers_;
    std::unordered_map<std::string, std::string> tensor_producers_;
    GraphLimits limits_;
    size_t total_initializer_bytes_ = 0;
};

} // namespace pyflame_rt
