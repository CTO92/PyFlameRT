#include "pyflame_rt/registry.hpp"
#include <algorithm>

namespace pyflame_rt {

OperatorRegistry& OperatorRegistry::instance() {
    static OperatorRegistry registry;
    return registry;
}

void OperatorRegistry::register_op(const std::string& op_type, OpFunc impl) {
    ops_[op_type] = std::move(impl);
}

const OpFunc* OperatorRegistry::get(const std::string& op_type) const {
    auto it = ops_.find(op_type);
    return it != ops_.end() ? &it->second : nullptr;
}

bool OperatorRegistry::has(const std::string& op_type) const {
    return ops_.count(op_type) > 0;
}

std::vector<std::string> OperatorRegistry::list_ops() const {
    std::vector<std::string> result;
    result.reserve(ops_.size());
    for (const auto& [name, _] : ops_) {
        result.push_back(name);
    }
    std::sort(result.begin(), result.end());
    return result;
}

bool OperatorRegistry::unregister_op(const std::string& op_type) {
    return ops_.erase(op_type) > 0;
}

void OperatorRegistry::clear() {
    ops_.clear();
}

} // namespace pyflame_rt
