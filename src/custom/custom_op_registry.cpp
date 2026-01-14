#include "pyflame_rt/custom/custom_op.hpp"
#include <stdexcept>
#include <algorithm>

namespace pyflame_rt {
namespace custom {

// ============================================================================
// CustomOpRegistry Implementation
// ============================================================================

CustomOpRegistry& CustomOpRegistry::instance() {
    static CustomOpRegistry registry;
    return registry;
}

CustomOp& CustomOpRegistry::register_op(const OpSchema& schema) {
    std::lock_guard<std::mutex> lock(mutex_);

    std::string key;
    if (schema.domain.empty() || schema.domain == "custom") {
        key = schema.name;
    } else {
        key = schema.domain + "::" + schema.name;
    }

    if (ops_.count(key)) {
        throw std::runtime_error("Custom op already registered: " + key);
    }

    auto op = std::make_unique<CustomOp>(schema);
    CustomOp& ref = *op;
    ops_[key] = std::move(op);

    return ref;
}

CustomOp* CustomOpRegistry::get(const std::string& name) {
    std::lock_guard<std::mutex> lock(mutex_);

    // Try exact name first
    auto it = ops_.find(name);
    if (it != ops_.end()) {
        return it->second.get();
    }

    // Try with "custom::" prefix
    it = ops_.find("custom::" + name);
    if (it != ops_.end()) {
        return it->second.get();
    }

    return nullptr;
}

CustomOp* CustomOpRegistry::get(const std::string& domain, const std::string& name) {
    std::lock_guard<std::mutex> lock(mutex_);

    std::string key;
    if (domain.empty() || domain == "custom") {
        key = name;
    } else {
        key = domain + "::" + name;
    }

    auto it = ops_.find(key);
    if (it != ops_.end()) {
        return it->second.get();
    }

    return nullptr;
}

bool CustomOpRegistry::has(const std::string& name) const {
    std::lock_guard<std::mutex> lock(mutex_);
    return ops_.count(name) > 0 || ops_.count("custom::" + name) > 0;
}

bool CustomOpRegistry::has(const std::string& domain, const std::string& name) const {
    std::lock_guard<std::mutex> lock(mutex_);

    std::string key;
    if (domain.empty() || domain == "custom") {
        key = name;
    } else {
        key = domain + "::" + name;
    }

    return ops_.count(key) > 0;
}

std::vector<std::string> CustomOpRegistry::list() const {
    std::lock_guard<std::mutex> lock(mutex_);

    std::vector<std::string> names;
    names.reserve(ops_.size());
    for (const auto& [key, _] : ops_) {
        names.push_back(key);
    }

    std::sort(names.begin(), names.end());
    return names;
}

std::vector<std::string> CustomOpRegistry::list(const std::string& domain) const {
    std::lock_guard<std::mutex> lock(mutex_);

    std::vector<std::string> names;
    std::string prefix = domain + "::";

    for (const auto& [key, _] : ops_) {
        if (domain.empty() || domain == "custom") {
            // List ops without domain prefix
            if (key.find("::") == std::string::npos) {
                names.push_back(key);
            }
        } else {
            // List ops with specific domain
            // Security: bounds check before substr (MED-C1 fix)
            if (key.length() >= prefix.length() &&
                key.substr(0, prefix.length()) == prefix) {
                names.push_back(key.substr(prefix.length()));
            }
        }
    }

    std::sort(names.begin(), names.end());
    return names;
}

void CustomOpRegistry::unregister(const std::string& name) {
    std::lock_guard<std::mutex> lock(mutex_);

    if (ops_.erase(name) == 0) {
        ops_.erase("custom::" + name);
    }

    // Also unregister from the global operator registry
    OperatorRegistry::instance().unregister_op(name);
}

void CustomOpRegistry::unregister(const std::string& domain, const std::string& name) {
    std::lock_guard<std::mutex> lock(mutex_);

    std::string key;
    if (domain.empty() || domain == "custom") {
        key = name;
    } else {
        key = domain + "::" + name;
    }

    ops_.erase(key);

    // Also unregister from the global operator registry
    OperatorRegistry::instance().unregister_op(key);
}

void CustomOpRegistry::clear() {
    std::lock_guard<std::mutex> lock(mutex_);

    // Unregister all from global registry
    for (const auto& [key, _] : ops_) {
        OperatorRegistry::instance().unregister_op(key);
    }

    ops_.clear();
}

size_t CustomOpRegistry::size() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return ops_.size();
}

} // namespace custom
} // namespace pyflame_rt
