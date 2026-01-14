#include "loader.hpp"
#include "pyflame_format.hpp"
#include "pyflame_rt/io/model_io.hpp"
#include "pyflame_rt/errors.hpp"
#include <algorithm>
#include <sstream>
#include <fstream>

namespace pyflame_rt {

// ============================================================================
// Path Security Validation (Path Traversal Prevention)
// ============================================================================

/// Validate path for security issues (path traversal, null bytes, etc.)
/// Throws InvalidModelError if path is suspicious
inline void validate_path_security(const std::string& path) {
    // Check for null bytes (can truncate path in some systems)
    if (path.find('\0') != std::string::npos) {
        throw InvalidModelError("Path contains null byte - potential path injection");
    }

    // Check for empty path
    if (path.empty()) {
        throw InvalidModelError("Empty path provided");
    }

    // Check for path traversal sequences
    // Look for ".." components that could escape directories
    size_t pos = 0;
    while ((pos = path.find("..", pos)) != std::string::npos) {
        // Check if this is actually a path traversal attempt
        // ".." is suspicious if preceded by start/separator and followed by separator/end
        bool preceded_by_sep = (pos == 0) ||
                               (path[pos - 1] == '/') ||
                               (path[pos - 1] == '\\');
        bool followed_by_sep = (pos + 2 >= path.size()) ||
                               (path[pos + 2] == '/') ||
                               (path[pos + 2] == '\\');

        if (preceded_by_sep && followed_by_sep) {
            throw InvalidModelError(
                "Path contains directory traversal sequence '..' - "
                "potential path traversal attack");
        }
        pos += 2;
    }

    // Check for suspicious control characters (ASCII 0-31 except tab)
    for (char c : path) {
        if (c >= 0 && c < 32 && c != '\t') {
            throw InvalidModelError(
                "Path contains control character (ASCII " +
                std::to_string(static_cast<int>(c)) + ") - potential injection");
        }
    }

    // Platform-specific checks for Windows
#ifdef _WIN32
    // Check for reserved device names on Windows (CON, PRN, AUX, NUL, COM1-9, LPT1-9)
    std::string base = path;
    auto last_sep = base.find_last_of("/\\");
    if (last_sep != std::string::npos) {
        base = base.substr(last_sep + 1);
    }
    // Remove extension for comparison
    auto dot_pos = base.find('.');
    if (dot_pos != std::string::npos) {
        base = base.substr(0, dot_pos);
    }
    // Convert to uppercase for comparison
    std::transform(base.begin(), base.end(), base.begin(), ::toupper);

    static const char* reserved_names[] = {
        "CON", "PRN", "AUX", "NUL",
        "COM1", "COM2", "COM3", "COM4", "COM5", "COM6", "COM7", "COM8", "COM9",
        "LPT1", "LPT2", "LPT3", "LPT4", "LPT5", "LPT6", "LPT7", "LPT8", "LPT9"
    };
    for (const char* reserved : reserved_names) {
        if (base == reserved) {
            throw InvalidModelError(
                "Path contains Windows reserved device name '" +
                std::string(reserved) + "'");
        }
    }
#endif
}

/// Check if file exists and is readable
inline bool file_exists(const std::string& path) {
    std::ifstream f(path);
    return f.good();
}

std::string get_extension(const std::string& path) {
    auto pos = path.rfind('.');
    if (pos == std::string::npos) {
        return "";
    }
    std::string ext = path.substr(pos);
    std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
    return ext;
}

std::unique_ptr<Graph> load_model(const std::string& path) {
    // Security: validate path before any file operations
    validate_path_security(path);

    std::string ext = get_extension(path);

    if (ext == ".pfm") {
        return PyFlameFormat::load(path);
    }

    throw UnsupportedFormatError(ext, {".pfm"});
}

void save_model(const Graph& graph, const std::string& path) {
    // Security: validate path before any file operations
    validate_path_security(path);

    std::string ext = get_extension(path);

    if (ext == ".pfm") {
        PyFlameFormat::save(graph, path);
        return;
    }

    throw UnsupportedFormatError(ext, {".pfm"});
}

} // namespace pyflame_rt

// ============================================================================
// Public io namespace functions
// ============================================================================

namespace pyflame_rt {
namespace io {

std::unique_ptr<Graph> load_model(const std::string& path) {
    return pyflame_rt::load_model(path);
}

std::unique_ptr<Graph> load_model_from_buffer(const void* data, size_t size) {
    // Check minimum size for header
    if (size < 4) {
        throw InvalidModelError("Buffer too small to contain model header");
    }

    const char* header = static_cast<const char*>(data);

    // Check for PFM format magic number
    if (header[0] == 'P' && header[1] == 'F' && header[2] == 'M') {
        // For now, throw to indicate file-based loading should be used
        // A full implementation would add stream-based loading to PyFlameFormat
        throw InvalidModelError("Direct buffer loading not yet implemented - use file path");
    }

    throw UnsupportedFormatError("unknown", {".pfm"});
}

bool save_model(const std::string& path, const Graph& graph) {
    try {
        pyflame_rt::save_model(graph, path);
        return true;
    } catch (...) {
        return false;
    }
}

std::vector<uint8_t> save_model_to_buffer(const Graph& /*graph*/) {
    // Placeholder - full implementation would serialize to buffer
    return {};
}

std::string get_extension(const std::string& path) {
    return pyflame_rt::get_extension(path);
}

bool is_supported_format(const std::string& path) {
    std::string ext = get_extension(path);
    return ext == ".pfm";
}

} // namespace io
} // namespace pyflame_rt
