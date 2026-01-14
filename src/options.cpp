#include "pyflame_rt/options.hpp"
#include <unordered_set>

namespace pyflame_rt {

// Security constants for options validation
static constexpr int MAX_THREADS = 256;  // LOW-02 fix: upper limit on threads

std::vector<std::string> SessionOptions::validate() const {
    std::vector<std::string> errors;

    static const std::unordered_set<std::string> valid_devices = {
        "cpu", "wse", "wse2", "wse3"
    };
    if (valid_devices.find(device) == valid_devices.end()) {
        errors.push_back("Invalid device: " + device);
    }

    static const std::unordered_set<std::string> valid_modes = {
        "sequential", "parallel"
    };
    if (valid_modes.find(execution_mode) == valid_modes.end()) {
        errors.push_back("Invalid execution_mode: " + execution_mode);
    }

    static const std::unordered_set<std::string> valid_levels = {
        "debug", "info", "warning", "error"
    };
    if (valid_levels.find(log_level) == valid_levels.end()) {
        errors.push_back("Invalid log_level: " + log_level);
    }

    // Security: validate num_threads range (LOW-02 fix)
    if (num_threads < 0) {
        errors.push_back("num_threads must be >= 0");
    } else if (num_threads > MAX_THREADS) {
        errors.push_back("num_threads must be <= " + std::to_string(MAX_THREADS) +
                        " (got " + std::to_string(num_threads) + ")");
    }

    // Note: memory_limit validation could be added here when enforcement is implemented (MED-04)
    // For now, any non-negative value is accepted

    return errors;
}

} // namespace pyflame_rt
