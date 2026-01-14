#pragma once

#include "pyflame_rt/graph.hpp"
#include "pyflame_rt/tensor.hpp"
#include "pyflame_rt/errors.hpp"
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace pyflame_rt {

/// Abstract base class for execution backends
class Backend {
public:
    virtual ~Backend() = default;

    /// Backend identifier
    virtual const std::string& name() const = 0;

    /// Check if backend supports an operation type
    virtual bool supports_op(const std::string& op_type) const = 0;

    /// Get list of supported operation types
    virtual std::vector<std::string> get_supported_ops() const = 0;

    /// Execute graph with given inputs
    /// @param graph Computational graph to execute
    /// @param input_feed Mapping of input names to tensors
    /// @param output_names Optional list of outputs to return (empty = all)
    /// @return List of output tensors in order of output_names
    virtual std::vector<Tensor> execute(
        const Graph& graph,
        const std::unordered_map<std::string, Tensor>& input_feed,
        const std::vector<std::string>& output_names = {}
    ) = 0;

    /// Validate inputs against graph requirements
    void validate_inputs(
        const Graph& graph,
        const std::unordered_map<std::string, Tensor>& input_feed
    ) const;
};

/// Factory for creating backends by name
std::unique_ptr<Backend> create_backend(const std::string& name, int num_threads = 0);

} // namespace pyflame_rt
