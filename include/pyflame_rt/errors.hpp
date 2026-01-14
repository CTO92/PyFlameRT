#pragma once

#include <exception>
#include <string>
#include <vector>
#include <optional>

namespace pyflame_rt {

// ============================================================================
// Path Sanitization for Error Messages (LOW-01 fix)
// ============================================================================

/// Global flag to control path sanitization in error messages.
/// When true (default for release builds), full paths are stripped to basename only.
/// When false, full paths are included (useful for debugging).
#ifndef PYFLAME_RT_SANITIZE_ERROR_PATHS
#ifdef NDEBUG
#define PYFLAME_RT_SANITIZE_ERROR_PATHS 1
#else
#define PYFLAME_RT_SANITIZE_ERROR_PATHS 0
#endif
#endif

/// Extract filename from a path for safer error messages (LOW-01 fix)
/// This prevents leaking directory structure information in production.
inline std::string sanitize_path_for_error(const std::string& path) {
#if PYFLAME_RT_SANITIZE_ERROR_PATHS
    // Find the last path separator (handle both Unix and Windows)
    size_t last_sep = path.find_last_of("/\\");
    if (last_sep != std::string::npos) {
        return path.substr(last_sep + 1);
    }
#endif
    return path;
}

/// Base exception for all PyFlameRT errors
class PyFlameRTError : public std::exception {
public:
    explicit PyFlameRTError(std::string message)
        : message_(std::move(message)) {}

    const char* what() const noexcept override {
        return message_.c_str();
    }

protected:
    std::string message_;
};

/// Raised when a model file is invalid or corrupted
class InvalidModelError : public PyFlameRTError {
public:
    explicit InvalidModelError(const std::string& message,
                               std::optional<std::string> path = std::nullopt)
        : PyFlameRTError(path.has_value()
            ? message + " (file: " + sanitize_path_for_error(path.value()) + ")"
            : message)
        , path_(path) {}

    /// Get the original path (may be empty for security in release builds)
    /// In debug builds, returns full path. In release builds, consider this sensitive.
    const std::optional<std::string>& path() const { return path_; }

private:
    std::optional<std::string> path_;
};

/// Raised when a model format is not supported
class UnsupportedFormatError : public PyFlameRTError {
public:
    UnsupportedFormatError(const std::string& format_name,
                           const std::vector<std::string>& supported)
        : PyFlameRTError(build_message(format_name, supported))
        , format_name_(format_name)
        , supported_(supported) {}

    const std::string& format_name() const { return format_name_; }
    const std::vector<std::string>& supported() const { return supported_; }

private:
    static std::string build_message(const std::string& format_name,
                                     const std::vector<std::string>& supported) {
        std::string msg = "Unsupported format: '" + format_name + "'. Supported: ";
        for (size_t i = 0; i < supported.size(); ++i) {
            if (i > 0) msg += ", ";
            msg += supported[i];
        }
        return msg;
    }

    std::string format_name_;
    std::vector<std::string> supported_;
};

/// Raised when an operator is not supported by the backend
class UnsupportedOperatorError : public PyFlameRTError {
public:
    UnsupportedOperatorError(const std::string& op_type,
                             const std::string& backend)
        : PyFlameRTError("Operator '" + op_type +
                         "' is not supported by the '" + backend + "' backend")
        , op_type_(op_type)
        , backend_(backend) {}

    const std::string& op_type() const { return op_type_; }
    const std::string& backend() const { return backend_; }

private:
    std::string op_type_;
    std::string backend_;
};

/// Raised when input shape doesn't match expected shape
class ShapeMismatchError : public PyFlameRTError {
public:
    ShapeMismatchError(const std::string& name,
                       const std::string& expected,
                       const std::string& got)
        : PyFlameRTError("Shape mismatch for '" + name +
                         "': expected " + expected + ", got " + got)
        , name_(name) {}

    const std::string& tensor_name() const { return name_; }

private:
    std::string name_;
};

/// Raised when input dtype doesn't match expected dtype
class DTypeMismatchError : public PyFlameRTError {
public:
    DTypeMismatchError(const std::string& name,
                       const std::string& expected,
                       const std::string& got)
        : PyFlameRTError("DType mismatch for '" + name +
                         "': expected " + expected + ", got " + got)
        , name_(name) {}

    const std::string& tensor_name() const { return name_; }

private:
    std::string name_;
};

/// Raised when graph validation fails
class ValidationError : public PyFlameRTError {
public:
    explicit ValidationError(const std::vector<std::string>& errors)
        : PyFlameRTError(build_message(errors))
        , errors_(errors) {}

    const std::vector<std::string>& errors() const { return errors_; }

private:
    static std::string build_message(const std::vector<std::string>& errors) {
        std::string msg = "Graph validation failed:\n";
        for (const auto& e : errors) {
            msg += "  - " + e + "\n";
        }
        return msg;
    }

    std::vector<std::string> errors_;
};

/// Raised when backend execution fails
class BackendError : public PyFlameRTError {
public:
    explicit BackendError(const std::string& message,
                          std::optional<std::string> node_name = std::nullopt)
        : PyFlameRTError(node_name.has_value()
            ? "[" + node_name.value() + "] " + message
            : message)
        , node_name_(node_name) {}

    const std::optional<std::string>& node_name() const { return node_name_; }

private:
    std::optional<std::string> node_name_;
};

/// Raised for invalid input to inference session
class InputError : public PyFlameRTError {
public:
    explicit InputError(const std::string& message)
        : PyFlameRTError(message) {}
};

} // namespace pyflame_rt
