#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "pyflame_rt/memory/memory_pool.hpp"

namespace py = pybind11;
using namespace pyflame_rt;
using namespace pyflame_rt::memory;

void bind_memory(py::module_& m) {
    auto mem_mod = m.def_submodule("memory", "Memory pool management");

    // ========================================================================
    // AllocationStats
    // ========================================================================
    py::class_<AllocationStats>(mem_mod, "AllocationStats",
        "Memory allocation statistics.")
        .def_readonly("total_allocated", &AllocationStats::total_allocated,
            "Total bytes allocated")
        .def_readonly("total_freed", &AllocationStats::total_freed,
            "Total bytes freed")
        .def_readonly("current_usage", &AllocationStats::current_usage,
            "Current memory usage in bytes")
        .def_readonly("peak_usage", &AllocationStats::peak_usage,
            "Peak memory usage in bytes")
        .def_readonly("num_allocations", &AllocationStats::num_allocations,
            "Number of allocation operations")
        .def_readonly("num_frees", &AllocationStats::num_frees,
            "Number of free operations")
        .def_readonly("pool_hits", &AllocationStats::pool_hits,
            "Number of pool hits (allocations from pool)")
        .def_readonly("pool_misses", &AllocationStats::pool_misses,
            "Number of pool misses (direct allocations)")
        .def("hit_rate", &AllocationStats::hit_rate,
            "Calculate pool hit rate");

    // ========================================================================
    // PoolConfig
    // ========================================================================
    py::class_<PoolConfig>(mem_mod, "PoolConfig",
        "Memory pool configuration.")
        .def(py::init<>())
        .def_readwrite("size_classes", &PoolConfig::size_classes,
            "Size classes for pooled allocations")
        .def_readwrite("max_blocks_per_class", &PoolConfig::max_blocks_per_class,
            "Maximum blocks per size class")
        .def_readwrite("memory_limit", &PoolConfig::memory_limit,
            "Total memory limit (0 = unlimited)")
        .def_readwrite("alignment", &PoolConfig::alignment,
            "Alignment boundary")
        .def_readwrite("thread_safe", &PoolConfig::thread_safe,
            "Enable thread safety")
        .def_readwrite("track_stats", &PoolConfig::track_stats,
            "Enable statistics tracking")
        .def_readwrite("auto_grow", &PoolConfig::auto_grow,
            "Automatically grow pool when exhausted");

    // ========================================================================
    // BlockPool
    // ========================================================================
    py::class_<BlockPool>(mem_mod, "BlockPool",
        "Fixed-size block pool allocator.")
        .def(py::init<size_t, size_t, size_t>(),
             py::arg("block_size"),
             py::arg("max_blocks"),
             py::arg("alignment") = 64,
             "Create a block pool.")
        .def("block_size", &BlockPool::block_size,
             "Get block size")
        .def("free_count", &BlockPool::free_count,
             "Get number of free blocks")
        .def("used_count", &BlockPool::used_count,
             "Get number of used blocks")
        .def("empty", &BlockPool::empty,
             "Check if pool is empty")
        .def("reset", &BlockPool::reset,
             "Reset pool (free all blocks)");

    // ========================================================================
    // MemoryPool
    // ========================================================================
    py::class_<MemoryPool>(mem_mod, "MemoryPool",
        "Multi-size memory pool for efficient allocation.\n"
        "Routes allocations to appropriate size classes.")
        .def(py::init<const PoolConfig&>(),
             py::arg("config") = PoolConfig(),
             "Create a memory pool with configuration.")
        .def("reset", &MemoryPool::reset,
             "Reset pool (free all allocations)")
        .def("trim", &MemoryPool::trim,
             "Trim unused memory")
        .def("reserve", &MemoryPool::reserve,
             py::arg("size"),
             "Pre-allocate memory")
        .def("get_stats", &MemoryPool::get_stats,
             "Get allocation statistics")
        .def("reset_stats", &MemoryPool::reset_stats,
             "Reset statistics")
        .def("current_usage", &MemoryPool::current_usage,
             "Get current memory usage")
        .def("peak_usage", &MemoryPool::peak_usage,
             "Get peak memory usage")
        .def_property_readonly("config", &MemoryPool::config,
             "Get pool configuration");

    // ========================================================================
    // Arena
    // ========================================================================
    py::class_<Arena>(mem_mod, "Arena",
        "Arena allocator for temporary allocations.\n"
        "Fast linear allocation with bulk reset.")
        .def(py::init<size_t>(),
             py::arg("initial_size") = 1024 * 1024,
             "Create an arena with initial size.")
        .def("reset", &Arena::reset,
             "Reset arena (invalidates all allocations)")
        .def("used", &Arena::used,
             "Get current usage")
        .def("capacity", &Arena::capacity,
             "Get capacity")
        .def("remaining", &Arena::remaining,
             "Get remaining space");

    // ========================================================================
    // BufferCache::CacheStats
    // ========================================================================
    py::class_<BufferCache::CacheStats>(mem_mod, "BufferCacheStats",
        "Buffer cache statistics.")
        .def_readonly("hits", &BufferCache::CacheStats::hits,
            "Number of cache hits")
        .def_readonly("misses", &BufferCache::CacheStats::misses,
            "Number of cache misses")
        .def_readonly("cached_buffers", &BufferCache::CacheStats::cached_buffers,
            "Number of cached buffers")
        .def_readonly("cached_bytes", &BufferCache::CacheStats::cached_bytes,
            "Total cached bytes");

    // ========================================================================
    // BufferCache
    // ========================================================================
    py::class_<BufferCache>(mem_mod, "BufferCache",
        "Buffer cache for tensor allocations.\n"
        "Reuses allocated buffers to reduce allocation overhead.")
        .def(py::init<size_t, size_t>(),
             py::arg("max_cached_buffers") = 32,
             py::arg("max_cached_size") = 256 * 1024 * 1024,
             "Create a buffer cache.")
        .def("clear", &BufferCache::clear,
             "Clear all cached buffers")
        .def("get_stats", &BufferCache::get_stats,
             "Get cache statistics");

    // ========================================================================
    // Global functions
    // ========================================================================
    mem_mod.def("global_pool", &global_pool,
                py::return_value_policy::reference,
                "Get the global memory pool.");
    mem_mod.def("global_buffer_cache", &global_buffer_cache,
                py::return_value_policy::reference,
                "Get the global buffer cache.");
    mem_mod.def("set_global_pool_config", &set_global_pool_config,
                py::arg("config"),
                "Set the global pool configuration.");
}
