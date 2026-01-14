#pragma once

#include "pyflame_rt/tensor.hpp"
#include "pyflame_rt/node.hpp"
#include <functional>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace pyflame_rt {

/// Operator execution context
struct OpContext {
    const Node* node;

    // Math safety options (HIGH-04 security fix)
    // When true, division by zero throws instead of producing inf/nan
    bool strict_math_mode = false;

    // Future: execution options, profiling, etc.
};

/// Operator function signature
/// Takes input tensors, attributes from context, returns output tensors
using OpFunc = std::function<std::vector<Tensor>(
    const std::vector<const Tensor*>& inputs,
    const OpContext& ctx
)>;

/// Registry mapping operator types to implementations
class OperatorRegistry {
public:
    /// Get singleton instance
    static OperatorRegistry& instance();

    /// Register an operator implementation
    void register_op(const std::string& op_type, OpFunc impl);

    /// Get operator implementation
    const OpFunc* get(const std::string& op_type) const;

    /// Check if operator is registered
    bool has(const std::string& op_type) const;

    /// List all registered operators
    std::vector<std::string> list_ops() const;

    /// Unregister operator
    bool unregister_op(const std::string& op_type);

    /// Clear all registrations (for testing)
    void clear();

private:
    OperatorRegistry() = default;
    std::unordered_map<std::string, OpFunc> ops_;
};

/// Helper class for static operator registration
class OpRegistrar {
public:
    OpRegistrar(const std::string& op_type, OpFunc impl) {
        OperatorRegistry::instance().register_op(op_type, std::move(impl));
    }
};

/// Helper macro for operator registration
#define REGISTER_OP(op_type, impl) \
    static ::pyflame_rt::OpRegistrar _reg_##op_type(#op_type, impl)

} // namespace pyflame_rt
