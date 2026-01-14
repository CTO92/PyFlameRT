#include "pyflame_rt/import/op_converter.hpp"

namespace pyflame_rt {
namespace import {

// ============================================================================
// OpConverterRegistry Implementation
// ============================================================================

OpConverterRegistry& OpConverterRegistry::instance() {
    static OpConverterRegistry instance;
    return instance;
}

void OpConverterRegistry::register_converter(
    const std::string& framework,
    const std::string& op_name,
    OpConverter converter
) {
    converters_[framework][op_name] = std::move(converter);
}

OpConverter OpConverterRegistry::get(
    const std::string& framework,
    const std::string& op_name
) const {
    auto fw_it = converters_.find(framework);
    if (fw_it == converters_.end()) {
        return nullptr;
    }

    auto op_it = fw_it->second.find(op_name);
    if (op_it == fw_it->second.end()) {
        return nullptr;
    }

    return op_it->second;
}

bool OpConverterRegistry::has(
    const std::string& framework,
    const std::string& op_name
) const {
    auto fw_it = converters_.find(framework);
    if (fw_it == converters_.end()) {
        return false;
    }
    return fw_it->second.find(op_name) != fw_it->second.end();
}

std::vector<std::string> OpConverterRegistry::supported_ops(
    const std::string& framework
) const {
    std::vector<std::string> ops;

    auto fw_it = converters_.find(framework);
    if (fw_it == converters_.end()) {
        return ops;
    }

    ops.reserve(fw_it->second.size());
    for (const auto& [op_name, _] : fw_it->second) {
        ops.push_back(op_name);
    }

    return ops;
}

std::vector<std::string> OpConverterRegistry::supported_frameworks() const {
    std::vector<std::string> frameworks;
    frameworks.reserve(converters_.size());

    for (const auto& [framework, _] : converters_) {
        frameworks.push_back(framework);
    }

    return frameworks;
}

} // namespace import
} // namespace pyflame_rt
