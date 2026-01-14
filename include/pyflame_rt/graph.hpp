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
struct GraphLimits {
    size_t max_nodes = 10000000;        // 10 million nodes max
    size_t max_initializers = 1000000;  // 1 million initializers max
    size_t max_initializer_bytes = 16ULL * 1024 * 1024 * 1024;  // 16 GB max total
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
    /// Security fix HIGH-01: Returns optional reference instead of raw pointer
    /// to prevent dangling pointer issues after map modifications
    std::optional<std::reference_wrapper<Tensor>> get_initializer(const std::string& name);
    std::optional<std::reference_wrapper<const Tensor>> get_initializer(const std::string& name) const;

    /// Legacy raw pointer API (deprecated - use optional version)
    /// WARNING: Pointer may be invalidated if initializers are added/removed
    [[deprecated("Use get_initializer() returning optional<reference_wrapper> instead")]]
    Tensor* get_initializer_unsafe(const std::string& name);
    [[deprecated("Use get_initializer() returning optional<reference_wrapper> instead")]]
    const Tensor* get_initializer_unsafe(const std::string& name) const;

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
