#pragma once

#include "pyflame_rt/types.hpp"
#include <string>
#include <vector>
#include <unordered_map>
#include <variant>
#include <optional>

namespace pyflame_rt {

/// Attribute value type (supports common types)
using AttributeValue = std::variant<
    int64_t,
    float,
    std::string,
    std::vector<int64_t>,
    std::vector<float>,
    std::vector<std::string>
>;

/// Represents a single operation in the computational graph
class Node {
public:
    Node() = default;

    Node(std::string name,
         std::string op_type,
         std::vector<std::string> inputs,
         std::vector<std::string> outputs);

    // Accessors
    const std::string& name() const { return name_; }
    const std::string& op_type() const { return op_type_; }
    const std::vector<std::string>& inputs() const { return inputs_; }
    const std::vector<std::string>& outputs() const { return outputs_; }
    const std::unordered_map<std::string, AttributeValue>& attributes() const {
        return attributes_;
    }

    // Attribute access with default value
    template<typename T>
    T get_attr(const std::string& name, const T& default_value) const;

    // Attribute access returning optional
    template<typename T>
    std::optional<T> get_attr(const std::string& name) const;

    // Set attribute
    void set_attr(const std::string& name, AttributeValue value);
    bool has_attr(const std::string& name) const;

    // Shape info (set during inference)
    void set_input_shapes(std::vector<std::vector<int64_t>> shapes);
    void set_output_shapes(std::vector<std::vector<int64_t>> shapes);

    const std::vector<std::vector<int64_t>>& input_shapes() const {
        return input_shapes_;
    }
    const std::vector<std::vector<int64_t>>& output_shapes() const {
        return output_shapes_;
    }

private:
    std::string name_;
    std::string op_type_;
    std::vector<std::string> inputs_;
    std::vector<std::string> outputs_;
    std::unordered_map<std::string, AttributeValue> attributes_;
    std::vector<std::vector<int64_t>> input_shapes_;
    std::vector<std::vector<int64_t>> output_shapes_;
};

// Template implementations
template<typename T>
T Node::get_attr(const std::string& name, const T& default_value) const {
    auto it = attributes_.find(name);
    if (it == attributes_.end()) {
        return default_value;
    }
    if (auto* val = std::get_if<T>(&it->second)) {
        return *val;
    }
    return default_value;
}

template<typename T>
std::optional<T> Node::get_attr(const std::string& name) const {
    auto it = attributes_.find(name);
    if (it == attributes_.end()) {
        return std::nullopt;
    }
    if (auto* val = std::get_if<T>(&it->second)) {
        return *val;
    }
    return std::nullopt;
}

} // namespace pyflame_rt
