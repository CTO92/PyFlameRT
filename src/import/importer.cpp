#include "pyflame_rt/import/importer.hpp"
#include <algorithm>
#include <cctype>

namespace pyflame_rt {
namespace import {

// ============================================================================
// ModelImporter Implementation
// ============================================================================

bool ModelImporter::can_import(const std::string& path) const {
    std::string ext = get_extension(path);
    for (const auto& supported_ext : extensions()) {
        if (ext == supported_ext) {
            return true;
        }
    }
    return false;
}

bool ModelImporter::has_extension(const std::string& path, const std::string& ext) const {
    return get_extension(path) == ext;
}

std::string ModelImporter::get_extension(const std::string& path) const {
    size_t dot_pos = path.rfind('.');
    if (dot_pos == std::string::npos) {
        return "";
    }

    std::string ext = path.substr(dot_pos);

    // Convert to lowercase for case-insensitive comparison
    std::transform(ext.begin(), ext.end(), ext.begin(),
                   [](unsigned char c) { return std::tolower(c); });

    return ext;
}

// ============================================================================
// ImporterRegistry Implementation
// ============================================================================

ImporterRegistry& ImporterRegistry::instance() {
    static ImporterRegistry instance;
    return instance;
}

void ImporterRegistry::register_importer(std::unique_ptr<ModelImporter> importer) {
    if (importer) {
        importers_.push_back(std::move(importer));
    }
}

ModelImporter* ImporterRegistry::get(const std::string& path) const {
    for (const auto& importer : importers_) {
        if (importer->can_import(path)) {
            return importer.get();
        }
    }
    return nullptr;
}

ModelImporter* ImporterRegistry::get_by_name(const std::string& name) const {
    for (const auto& importer : importers_) {
        if (importer->name() == name) {
            return importer.get();
        }
    }
    return nullptr;
}

std::vector<std::string> ImporterRegistry::list() const {
    std::vector<std::string> names;
    names.reserve(importers_.size());
    for (const auto& importer : importers_) {
        names.push_back(importer->name());
    }
    return names;
}

std::vector<std::string> ImporterRegistry::supported_extensions() const {
    std::vector<std::string> extensions;
    for (const auto& importer : importers_) {
        auto exts = importer->extensions();
        extensions.insert(extensions.end(), exts.begin(), exts.end());
    }
    return extensions;
}

bool ImporterRegistry::can_import(const std::string& path) const {
    return get(path) != nullptr;
}

// ============================================================================
// Convenience Functions
// ============================================================================

ImportResult import_model(const std::string& path, const ImportOptions& options) {
    auto& registry = ImporterRegistry::instance();

    ModelImporter* importer = registry.get(path);
    if (!importer) {
        // Build list of supported formats for error message
        auto extensions = registry.supported_extensions();

        ImportResult result;
        result.warnings.push_back(
            "No importer found for file: " + path +
            ". Supported formats: " +
            (extensions.empty() ? "none" : [&]() {
                std::string s;
                for (size_t i = 0; i < extensions.size(); ++i) {
                    if (i > 0) s += ", ";
                    s += extensions[i];
                }
                return s;
            }())
        );
        return result;
    }

    return importer->import_file(path, options);
}

} // namespace import
} // namespace pyflame_rt
