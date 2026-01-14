#include "pyflame_rt/node.hpp"

namespace pyflame_rt {

Node::Node(std::string name,
           std::string op_type,
           std::vector<std::string> inputs,
           std::vector<std::string> outputs)
    : name_(std::move(name))
    , op_type_(std::move(op_type))
    , inputs_(std::move(inputs))
    , outputs_(std::move(outputs))
{
}

void Node::set_attr(const std::string& name, AttributeValue value) {
    attributes_[name] = std::move(value);
}

bool Node::has_attr(const std::string& name) const {
    return attributes_.count(name) > 0;
}

void Node::set_input_shapes(std::vector<std::vector<int64_t>> shapes) {
    input_shapes_ = std::move(shapes);
}

void Node::set_output_shapes(std::vector<std::vector<int64_t>> shapes) {
    output_shapes_ = std::move(shapes);
}

} // namespace pyflame_rt
