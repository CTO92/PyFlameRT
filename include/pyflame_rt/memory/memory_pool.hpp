#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <vector>
#include <map>
#include <unordered_map>
#include <unordered_set>
#include <atomic>

namespace pyflame_rt {
namespace memory {

/// Memory allocation statistics
struct AllocationStats {
    size_t total_allocated = 0;
    size_t total_freed = 0;
    size_t current_usage = 0;
    size_t peak_usage = 0;
    size_t num_allocations = 0;
    size_t num_frees = 0;
    size_t pool_hits = 0;
    size_t pool_misses = 0;

    double hit_rate() const {
        size_t total = pool_hits + pool_misses;
        return total > 0 ? static_cast<double>(pool_hits) / total : 0.0;
    }
};

/// Memory pool configuration
struct PoolConfig {
    /// Size classes for pooled allocations (bytes)
    std::vector<size_t> size_classes = {
        64, 128, 256, 512, 1024,
        2048, 4096, 8192, 16384, 32768,
        65536, 131072, 262144, 524288,
        1048576, 2097152, 4194304
    };

    /// Maximum blocks per size class
    size_t max_blocks_per_class = 64;

    /// Total memory limit (0 = unlimited)
    size_t memory_limit = 0;

    /// Initial pool size (pre-allocate)
    size_t initial_pool_size = 0;

    /// Alignment boundary (must be power of 2)
    size_t alignment = 64;

    /// Enable thread safety
    bool thread_safe = true;

    /// Enable statistics tracking
    bool track_stats = true;

    /// Grow pool automatically when exhausted
    bool auto_grow = true;
};

/// Fixed-size block pool for efficient allocation of same-size objects
class BlockPool {
public:
    BlockPool(size_t block_size, size_t max_blocks, size_t alignment = 64);
    ~BlockPool();

    // Non-copyable
    BlockPool(const BlockPool&) = delete;
    BlockPool& operator=(const BlockPool&) = delete;

    /// Allocate a block from the pool
    void* allocate();

    /// Return a block to the pool
    void deallocate(void* ptr);

    /// Get the size of blocks in this pool
    size_t block_size() const { return block_size_; }

    /// Get number of free blocks
    size_t free_count() const;

    /// Get number of allocated blocks
    size_t used_count() const;

    /// Check if pool is exhausted
    bool empty() const;

    /// Reset pool (returns all blocks to free list)
    void reset();

    /// Get total capacity
    size_t capacity() const { return max_blocks_; }

private:
    size_t block_size_;
    size_t max_blocks_;
    size_t alignment_;
    std::vector<void*> free_list_;
    std::vector<uint8_t> memory_;
    mutable std::mutex mutex_;

#ifndef NDEBUG
    // Security fix MED-02: Track allocated blocks for double-free detection (debug only)
    std::unordered_set<void*> allocated_blocks_;
#endif
};

/// Multi-size memory pool with size class routing
class MemoryPool {
public:
    explicit MemoryPool(const PoolConfig& config = PoolConfig());
    ~MemoryPool();

    // Non-copyable
    MemoryPool(const MemoryPool&) = delete;
    MemoryPool& operator=(const MemoryPool&) = delete;

    // ========================================================================
    // Allocation
    // ========================================================================

    /// Allocate memory of given size
    void* allocate(size_t size);

    /// Allocate aligned memory
    void* allocate_aligned(size_t size, size_t alignment);

    /// Deallocate memory
    void deallocate(void* ptr);

    /// Reallocate memory to new size
    void* reallocate(void* ptr, size_t new_size);

    // ========================================================================
    // Pool Management
    // ========================================================================

    /// Reset pool (invalidates all allocations)
    void reset();

    /// Trim unused memory back to system
    void trim();

    /// Pre-allocate memory for expected usage
    void reserve(size_t size);

    // ========================================================================
    // Statistics
    // ========================================================================

    /// Get allocation statistics
    AllocationStats get_stats() const;

    /// Reset statistics counters
    void reset_stats();

    /// Get current memory usage
    size_t current_usage() const;

    /// Get peak memory usage
    size_t peak_usage() const;

    // ========================================================================
    // Configuration
    // ========================================================================

    /// Get pool configuration
    const PoolConfig& config() const { return config_; }

private:
    PoolConfig config_;
    std::vector<std::unique_ptr<BlockPool>> pools_;
    std::map<size_t, size_t> size_to_pool_;
    mutable AllocationStats stats_;
    mutable std::mutex mutex_;

    // Security fix HIGH-04/MED-04: Track allocations for proper deallocation and reallocate
    std::unordered_map<void*, size_t> allocation_sizes_;

    size_t find_size_class(size_t size) const;
    void* allocate_from_pool(size_t size_class);
    void* allocate_direct(size_t size);
    void update_stats_alloc(size_t size, bool from_pool);
    void update_stats_free(size_t size);
};

/// Arena allocator for fast, temporary allocations
/// All memory is freed at once when arena is reset or destroyed
class Arena {
public:
    explicit Arena(size_t initial_size = 1024 * 1024);
    ~Arena();

    // Non-copyable
    Arena(const Arena&) = delete;
    Arena& operator=(const Arena&) = delete;

    /// Allocate memory from arena
    void* allocate(size_t size);

    /// Allocate aligned memory from arena
    void* allocate_aligned(size_t size, size_t alignment);

    /// Reset arena (invalidates all allocations)
    void reset();

    /// Get current usage in bytes
    size_t used() const { return offset_; }

    /// Get total capacity
    size_t capacity() const { return capacity_; }

    /// Get remaining space
    size_t remaining() const { return capacity_ - offset_; }

private:
    std::vector<uint8_t> buffer_;
    size_t offset_ = 0;
    size_t capacity_;
    std::vector<std::vector<uint8_t>> overflow_blocks_;
};

/// Scoped arena - saves position on construction, restores on destruction
class ScopedArena {
public:
    explicit ScopedArena(Arena& arena);
    ~ScopedArena();

    // Non-copyable
    ScopedArena(const ScopedArena&) = delete;
    ScopedArena& operator=(const ScopedArena&) = delete;

    /// Allocate from the scoped arena
    void* allocate(size_t size) { return arena_.allocate(size); }

    /// Allocate aligned from the scoped arena
    void* allocate_aligned(size_t size, size_t alignment) {
        return arena_.allocate_aligned(size, alignment);
    }

private:
    Arena& arena_;
    size_t saved_offset_;
};

/// Buffer cache for reusing tensor allocations
class BufferCache {
public:
    BufferCache(size_t max_cached_buffers = 32,
                size_t max_cached_size = 256 * 1024 * 1024);
    ~BufferCache();

    // Non-copyable
    BufferCache(const BufferCache&) = delete;
    BufferCache& operator=(const BufferCache&) = delete;

    /// Get a buffer of at least the given size
    void* get_buffer(size_t size);

    /// Return a buffer to the cache
    void return_buffer(void* ptr, size_t size);

    /// Clear all cached buffers
    void clear();

    /// Get cache statistics
    struct CacheStats {
        size_t hits = 0;
        size_t misses = 0;
        size_t cached_buffers = 0;
        size_t cached_bytes = 0;
    };
    CacheStats get_stats() const;

private:
    size_t max_cached_buffers_;
    size_t max_cached_size_;
    std::multimap<size_t, void*> free_buffers_;
    size_t cached_bytes_ = 0;
    mutable CacheStats stats_;
    mutable std::mutex mutex_;
};

// ============================================================================
// Global Instances
// ============================================================================

/// Get the global memory pool instance
MemoryPool& global_pool();

/// Get the global buffer cache instance
BufferCache& global_buffer_cache();

/// Set the global memory pool configuration
void set_global_pool_config(const PoolConfig& config);

} // namespace memory
} // namespace pyflame_rt
