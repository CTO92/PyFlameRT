#include "executor.hpp"
#include "pyflame_rt/errors.hpp"
#include <unordered_set>

namespace pyflame_rt {

CPUExecutor::CPUExecutor(int num_threads,
                         std::optional<size_t> memory_limit_bytes,
                         bool strict_math_mode)
    : num_threads_(num_threads)
    , memory_limit_bytes_(memory_limit_bytes)
    , strict_math_mode_(strict_math_mode)
{
}

void CPUExecutor::check_memory_limit(size_t current_usage, size_t additional_bytes) const {
    if (!memory_limit_bytes_.has_value() || memory_limit_bytes_.value() == 0) {
        return;  // No limit
    }

    size_t limit = memory_limit_bytes_.value();

    // Check for overflow in addition
    if (current_usage > SIZE_MAX - additional_bytes) {
        throw BackendError("Memory usage calculation overflow");
    }

    if (current_usage + additional_bytes > limit) {
        throw BackendError(
            "Memory limit exceeded: would use " +
            std::to_string((current_usage + additional_bytes) / (1024 * 1024)) +
            " MB, limit is " + std::to_string(limit / (1024 * 1024)) + " MB"
        );
    }
}

bool CPUExecutor::supports_op(const std::string& op_type) const {
    return OperatorRegistry::instance().has(op_type);
}

std::vector<std::string> CPUExecutor::get_supported_ops() const {
    return OperatorRegistry::instance().list_ops();
}

std::vector<Tensor> CPUExecutor::execute(
    const Graph& graph,
    const std::unordered_map<std::string, Tensor>& input_feed,
    const std::vector<std::string>& output_names)
{
    // Validate inputs
    validate_inputs(graph, input_feed);

    // Initialize tensor storage and memory tracking (LOW-01 fix)
    std::unordered_map<std::string, Tensor> tensors;
    size_t current_memory_usage = 0;

    // Helper to track memory when adding tensors
    auto track_tensor = [&](const Tensor& t) {
        size_t bytes = t.size_bytes();
        check_memory_limit(current_memory_usage, bytes);
        current_memory_usage += bytes;
    };

    // Add inputs (make copies to avoid modifying originals)
    for (const auto& [name, tensor] : input_feed) {
        track_tensor(tensor);
        tensors[name] = tensor.clone();
    }

    // Add initializers
    for (const auto& [name, tensor] : graph.initializers()) {
        track_tensor(tensor);
        tensors[name] = tensor.clone();
    }

    // Execute nodes in topological order
    auto sorted_nodes = graph.topological_sort();
    auto& registry = OperatorRegistry::instance();

    for (const auto& node : sorted_nodes) {
        const OpFunc* op_impl = registry.get(node->op_type());

        if (!op_impl) {
            throw UnsupportedOperatorError(node->op_type(), name_);
        }

        // Gather input tensors
        std::vector<const Tensor*> inputs;
        inputs.reserve(node->inputs().size());
        for (const auto& input_name : node->inputs()) {
            auto it = tensors.find(input_name);
            if (it == tensors.end()) {
                throw BackendError(
                    "Input tensor '" + input_name + "' not found",
                    node->name()
                );
            }
            inputs.push_back(&it->second);
        }

        // Create execution context with runtime options (HIGH-04 fix)
        OpContext ctx{node.get(), strict_math_mode_};

        // Execute operation
        std::vector<Tensor> outputs;
        try {
            outputs = (*op_impl)(inputs, ctx);
        } catch (const std::exception& e) {
            throw BackendError(e.what(), node->name());
        }

        // Store outputs
        if (outputs.size() != node->outputs().size()) {
            throw BackendError(
                "Operator returned " + std::to_string(outputs.size()) +
                " outputs, expected " + std::to_string(node->outputs().size()),
                node->name()
            );
        }

        for (size_t i = 0; i < outputs.size(); ++i) {
            // Track memory for new output tensors (LOW-01 fix)
            track_tensor(outputs[i]);
            tensors[node->outputs()[i]] = std::move(outputs[i]);
        }
    }

    // Gather requested outputs
    std::vector<std::string> result_names = output_names;
    if (result_names.empty()) {
        for (const auto& output : graph.outputs()) {
            result_names.push_back(output.name);
        }
    }

    std::vector<Tensor> results;
    results.reserve(result_names.size());
    for (const auto& name : result_names) {
        auto it = tensors.find(name);
        if (it == tensors.end()) {
            throw BackendError("Output tensor '" + name + "' not found");
        }
        results.push_back(std::move(it->second));
    }

    return results;
}

// Backend validation implementation
void Backend::validate_inputs(
    const Graph& graph,
    const std::unordered_map<std::string, Tensor>& input_feed) const
{
    std::unordered_set<std::string> expected_names;
    for (const auto& inp : graph.inputs()) {
        expected_names.insert(inp.name);
    }

    std::unordered_set<std::string> provided_names;
    for (const auto& [name, _] : input_feed) {
        provided_names.insert(name);
    }

    // Check for missing inputs
    for (const auto& name : expected_names) {
        if (provided_names.find(name) == provided_names.end()) {
            throw InputError("Missing input: " + name);
        }
    }

    // Check for extra inputs
    for (const auto& name : provided_names) {
        if (expected_names.find(name) == expected_names.end()) {
            throw InputError("Unexpected input: " + name);
        }
    }

    // Validate shapes and dtypes
    for (const auto& inp_info : graph.inputs()) {
        const Tensor& arr = input_feed.at(inp_info.name);

        // Shape check (allowing dynamic dims)
        const auto& expected_shape = inp_info.shape;
        const auto& actual_shape = arr.shape();

        if (expected_shape.size() != actual_shape.size()) {
            throw ShapeMismatchError(
                inp_info.name,
                shape_to_string(expected_shape),
                "[" + std::to_string(actual_shape.size()) + " dims]"
            );
        }

        for (size_t i = 0; i < expected_shape.size(); ++i) {
            if (expected_shape[i].has_value() &&
                expected_shape[i].value() != actual_shape[i]) {
                std::string actual_str = "[";
                for (size_t j = 0; j < actual_shape.size(); ++j) {
                    if (j > 0) actual_str += ", ";
                    actual_str += std::to_string(actual_shape[j]);
                }
                actual_str += "]";

                throw ShapeMismatchError(
                    inp_info.name,
                    shape_to_string(expected_shape),
                    actual_str
                );
            }
        }
    }
}

// Backend factory implementation
std::unique_ptr<Backend> create_backend(const std::string& name, int num_threads) {
    if (name == "cpu") {
        return std::make_unique<CPUExecutor>(num_threads);
    }
    throw UnsupportedFormatError(name, {"cpu"});
}

} // namespace pyflame_rt
