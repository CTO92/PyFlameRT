#pragma once

#include "pyflame_rt/backend.hpp"
#include "pyflame_rt/registry.hpp"
#include <optional>

namespace pyflame_rt {

/// CPU reference backend using standard C++ operations
class CPUExecutor : public Backend {
public:
    explicit CPUExecutor(int num_threads = 0,
                         std::optional<size_t> memory_limit_bytes = std::nullopt,
                         bool strict_math_mode = false);

    const std::string& name() const override { return name_; }
    bool supports_op(const std::string& op_type) const override;
    std::vector<std::string> get_supported_ops() const override;

    std::vector<Tensor> execute(
        const Graph& graph,
        const std::unordered_map<std::string, Tensor>& input_feed,
        const std::vector<std::string>& output_names = {}
    ) override;

    /// Set memory limit in bytes (0 or nullopt = unlimited)
    void set_memory_limit(std::optional<size_t> limit_bytes) {
        memory_limit_bytes_ = limit_bytes;
    }

    /// Set strict math mode (HIGH-04 fix)
    /// When enabled, division by zero throws instead of producing inf/nan
    void set_strict_math_mode(bool enabled) {
        strict_math_mode_ = enabled;
    }

private:
    std::string name_ = "cpu";
    int num_threads_;
    std::optional<size_t> memory_limit_bytes_;
    bool strict_math_mode_ = false;  // HIGH-04 fix

    /// Check if allocating additional bytes would exceed memory limit
    /// @throws BackendError if limit would be exceeded
    void check_memory_limit(size_t current_usage, size_t additional_bytes) const;
};

} // namespace pyflame_rt
