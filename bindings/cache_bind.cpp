#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/chrono.h>
#include <pybind11/functional.h>
#include "pyflame_rt/cache/binary_cache.hpp"
#include "pyflame_rt/cache/mmap_loader.hpp"

namespace py = pybind11;
using namespace pyflame_rt;
using namespace pyflame_rt::cache;

void bind_cache(py::module_& m) {
    auto cache_mod = m.def_submodule("cache", "Binary caching and memory mapping support");

    // ========================================================================
    // CacheConfig
    // ========================================================================
    py::class_<CacheConfig>(cache_mod, "CacheConfig",
        "Configuration for binary cache.")
        .def(py::init<>())
        .def_readwrite("cache_dir", &CacheConfig::cache_dir,
            "Directory to store cached files")
        .def_readwrite("max_size_bytes", &CacheConfig::max_size_bytes,
            "Maximum total cache size in bytes (0 = unlimited)")
        .def_readwrite("max_entries", &CacheConfig::max_entries,
            "Maximum number of cache entries (0 = unlimited)")
        .def_readwrite("use_mmap", &CacheConfig::use_mmap,
            "Use memory-mapped files for loading")
        .def_readwrite("compress", &CacheConfig::compress,
            "Enable compression for cached files")
        .def_readwrite("version", &CacheConfig::version,
            "Cache version string")
        .def_readwrite("thread_safe", &CacheConfig::thread_safe,
            "Enable thread-safe operations");

    // ========================================================================
    // CacheEntryInfo
    // ========================================================================
    py::class_<CacheEntryInfo>(cache_mod, "CacheEntryInfo",
        "Information about a cache entry.")
        .def_readonly("key", &CacheEntryInfo::key,
            "Cache key")
        .def_readonly("model_path", &CacheEntryInfo::model_path,
            "Original model path")
        .def_readonly("cache_path", &CacheEntryInfo::cache_path,
            "Path to cached file")
        .def_readonly("size_bytes", &CacheEntryInfo::size_bytes,
            "Size of cached file in bytes")
        .def_readonly("is_valid", &CacheEntryInfo::is_valid,
            "Whether the cache entry is valid")
        .def_readonly("version", &CacheEntryInfo::version,
            "Cache entry version");

    // ========================================================================
    // BinaryCache::Stats
    // ========================================================================
    py::class_<BinaryCache::Stats>(cache_mod, "CacheStats",
        "Cache statistics.")
        .def_readonly("hits", &BinaryCache::Stats::hits,
            "Number of cache hits")
        .def_readonly("misses", &BinaryCache::Stats::misses,
            "Number of cache misses")
        .def_readonly("stores", &BinaryCache::Stats::stores,
            "Number of cache stores")
        .def_readonly("evictions", &BinaryCache::Stats::evictions,
            "Number of evictions")
        .def("hit_rate", [](const BinaryCache::Stats& self) {
            size_t total = self.hits + self.misses;
            return total > 0 ? static_cast<double>(self.hits) / total : 0.0;
        }, "Calculate cache hit rate");

    // ========================================================================
    // BinaryCache
    // ========================================================================
    py::class_<BinaryCache>(cache_mod, "BinaryCache",
        "Binary cache for compiled model artifacts.\n"
        "Stores compiled graphs to speed up subsequent loads.")
        .def(py::init<const CacheConfig&>(),
             py::arg("config") = CacheConfig(),
             "Create a binary cache with the given configuration.")
        .def("has", &BinaryCache::has,
             py::arg("key"),
             "Check if a key exists in the cache.")
        .def("remove", &BinaryCache::remove,
             py::arg("key"),
             "Remove an entry from the cache.")
        .def("clear", &BinaryCache::clear,
             "Clear all entries from the cache.")
        .def("get_info", &BinaryCache::get_info,
             py::arg("key"),
             "Get information about a cache entry.")
        .def("list_entries", &BinaryCache::list_entries,
             "List all cache entry keys.")
        .def("total_size", &BinaryCache::total_size,
             "Get total size of all cached entries in bytes.")
        .def("entry_count", &BinaryCache::entry_count,
             "Get number of cache entries.")
        .def("validate", &BinaryCache::validate,
             "Validate all cache entries.")
        .def("warmup", &BinaryCache::warmup,
             py::arg("model_paths"),
             py::arg("options"),
             "Pre-load models into the cache.")
        .def("set_cache_dir", &BinaryCache::set_cache_dir,
             py::arg("dir"),
             "Set the cache directory.")
        .def("set_enabled", &BinaryCache::set_enabled,
             py::arg("enabled"),
             "Enable or disable caching.")
        .def("is_enabled", &BinaryCache::is_enabled,
             "Check if caching is enabled.")
        .def("get_stats", &BinaryCache::get_stats,
             "Get cache statistics.")
        .def("reset_stats", &BinaryCache::reset_stats,
             "Reset cache statistics.")
        .def_property_readonly("config", &BinaryCache::config,
             "Get cache configuration.");

    // ========================================================================
    // MappedFile::Advice
    // ========================================================================
    py::enum_<MappedFile::Advice>(cache_mod, "MMapAdvice",
        "Memory mapping advice hints.")
        .value("Normal", MappedFile::Advice::Normal,
               "No special access pattern")
        .value("Sequential", MappedFile::Advice::Sequential,
               "Sequential read access expected")
        .value("Random", MappedFile::Advice::Random,
               "Random access expected")
        .value("WillNeed", MappedFile::Advice::WillNeed,
               "Data will be needed soon")
        .value("DontNeed", MappedFile::Advice::DontNeed,
               "Data will not be needed soon");

    // ========================================================================
    // MappedFile
    // ========================================================================
    py::class_<MappedFile>(cache_mod, "MappedFile",
        "Memory-mapped file for efficient I/O.")
        .def_static("open", &MappedFile::open,
             py::arg("path"),
             "Open a file for memory mapping.")
        .def("is_valid", &MappedFile::is_valid,
             "Check if the mapping is valid.")
        .def("size", &MappedFile::size,
             "Get the size of the mapped file.")
        .def("advise", py::overload_cast<MappedFile::Advice>(&MappedFile::advise),
             py::arg("advice"),
             "Provide access pattern hint.")
        .def("lock", &MappedFile::lock,
             "Lock the mapped memory in RAM.")
        .def("unlock", &MappedFile::unlock,
             "Unlock the mapped memory.");

    // ========================================================================
    // MMapLoader
    // ========================================================================
    py::class_<MMapLoader>(cache_mod, "MMapLoader",
        "Memory-mapped model loader.")
        .def_static("can_mmap", &MMapLoader::can_mmap,
             py::arg("path"),
             "Check if a file can be memory-mapped.")
        .def_static("page_alignment", &MMapLoader::page_alignment,
             "Get platform page alignment.");

    // ========================================================================
    // Global functions
    // ========================================================================
    cache_mod.def("global_cache", &global_cache,
                  py::return_value_policy::reference,
                  "Get the global cache instance.");
    cache_mod.def("set_global_cache_config", &set_global_cache_config,
                  py::arg("config"),
                  "Set the global cache configuration.");
}
