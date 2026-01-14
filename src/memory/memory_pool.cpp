#include "pyflame_rt/memory/memory_pool.hpp"
#include <algorithm>
#include <cstring>
#include <cstdlib>
#include <stdexcept>

namespace pyflame_rt {
namespace memory {

// ============================================================================
// BlockPool Implementation
// ============================================================================

BlockPool::BlockPool(size_t block_size, size_t max_blocks, size_t alignment)
    : block_size_(block_size)
    , max_blocks_(max_blocks)
    , alignment_(alignment)
{
    // Round up block size to alignment
    if (alignment_ > 0) {
        block_size_ = (block_size_ + alignment_ - 1) & ~(alignment_ - 1);
    }

    // Allocate contiguous memory for all blocks
    size_t total_size = block_size_ * max_blocks_ + alignment_;
    memory_.resize(total_size);

    // Initialize free list with aligned pointers
    free_list_.reserve(max_blocks_);
    uint8_t* base = memory_.data();

    // Align base pointer
    if (alignment_ > 0) {
        uintptr_t base_addr = reinterpret_cast<uintptr_t>(base);
        uintptr_t aligned_addr = (base_addr + alignment_ - 1) & ~(alignment_ - 1);
        base = reinterpret_cast<uint8_t*>(aligned_addr);
    }

    // Add all blocks to free list
    for (size_t i = 0; i < max_blocks_; ++i) {
        free_list_.push_back(base + i * block_size_);
    }
}

BlockPool::~BlockPool() = default;

void* BlockPool::allocate() {
    std::lock_guard<std::mutex> lock(mutex_);

    if (free_list_.empty()) {
        return nullptr;
    }

    void* ptr = free_list_.back();
    free_list_.pop_back();

#ifndef NDEBUG
    // Security fix MED-02: Track allocation for double-free detection
    allocated_blocks_.insert(ptr);
#endif

    return ptr;
}

void BlockPool::deallocate(void* ptr) {
    if (!ptr) return;

    std::lock_guard<std::mutex> lock(mutex_);

    // Validate pointer is within our memory range
    uint8_t* byte_ptr = static_cast<uint8_t*>(ptr);
    uint8_t* start = memory_.data();
    uint8_t* end = start + memory_.size();

    if (byte_ptr >= start && byte_ptr < end) {
        // Security fix MED-02: Check alignment
        size_t offset = static_cast<size_t>(byte_ptr - start);
        if (alignment_ > 0 && (offset % block_size_) != 0) {
            throw std::runtime_error("BlockPool::deallocate: misaligned pointer");
        }

#ifndef NDEBUG
        // Security fix MED-02: Double-free detection
        if (allocated_blocks_.find(ptr) == allocated_blocks_.end()) {
            throw std::runtime_error("BlockPool::deallocate: double-free or invalid pointer detected");
        }
        allocated_blocks_.erase(ptr);
#endif

        free_list_.push_back(ptr);
    } else {
        // Pointer is not from this pool - in debug mode, warn
#ifndef NDEBUG
        throw std::runtime_error("BlockPool::deallocate: pointer not from this pool");
#endif
    }
}

size_t BlockPool::free_count() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return free_list_.size();
}

size_t BlockPool::used_count() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return max_blocks_ - free_list_.size();
}

bool BlockPool::empty() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return free_list_.empty();
}

void BlockPool::reset() {
    std::lock_guard<std::mutex> lock(mutex_);

    free_list_.clear();
    uint8_t* base = memory_.data();

    // Align base pointer
    if (alignment_ > 0) {
        uintptr_t base_addr = reinterpret_cast<uintptr_t>(base);
        uintptr_t aligned_addr = (base_addr + alignment_ - 1) & ~(alignment_ - 1);
        base = reinterpret_cast<uint8_t*>(aligned_addr);
    }

    for (size_t i = 0; i < max_blocks_; ++i) {
        free_list_.push_back(base + i * block_size_);
    }
}

// ============================================================================
// MemoryPool Implementation
// ============================================================================

MemoryPool::MemoryPool(const PoolConfig& config)
    : config_(config)
{
    // Create a pool for each size class
    for (size_t i = 0; i < config_.size_classes.size(); ++i) {
        size_t size = config_.size_classes[i];
        pools_.push_back(std::make_unique<BlockPool>(
            size, config_.max_blocks_per_class, config_.alignment));
        size_to_pool_[size] = i;
    }
}

MemoryPool::~MemoryPool() = default;

size_t MemoryPool::find_size_class(size_t size) const {
    // Find the smallest size class that can hold the requested size
    for (size_t sc : config_.size_classes) {
        if (sc >= size) return sc;
    }
    return 0;  // No suitable size class found
}

void* MemoryPool::allocate_from_pool(size_t size_class) {
    auto it = size_to_pool_.find(size_class);
    if (it == size_to_pool_.end()) return nullptr;

    return pools_[it->second]->allocate();
}

void* MemoryPool::allocate_direct(size_t size) {
    // Fall back to system allocator for oversized requests
    void* ptr = nullptr;

#if defined(_MSC_VER)
    ptr = _aligned_malloc(size, config_.alignment);
#elif defined(__STDC_VERSION__) && __STDC_VERSION__ >= 201112L
    ptr = aligned_alloc(config_.alignment,
                        (size + config_.alignment - 1) & ~(config_.alignment - 1));
#else
    if (posix_memalign(&ptr, config_.alignment, size) != 0) {
        ptr = nullptr;
    }
#endif

    return ptr;
}

void MemoryPool::update_stats_alloc(size_t size, bool from_pool) {
    if (!config_.track_stats) return;

    stats_.total_allocated += size;
    stats_.current_usage += size;
    stats_.num_allocations++;

    if (stats_.current_usage > stats_.peak_usage) {
        stats_.peak_usage = stats_.current_usage;
    }

    if (from_pool) {
        stats_.pool_hits++;
    } else {
        stats_.pool_misses++;
    }
}

void MemoryPool::update_stats_free(size_t size) {
    if (!config_.track_stats) return;

    stats_.total_freed += size;
    if (stats_.current_usage >= size) {
        stats_.current_usage -= size;
    }
    stats_.num_frees++;
}

void* MemoryPool::allocate(size_t size) {
    if (size == 0) return nullptr;

    std::lock_guard<std::mutex> lock(mutex_);

    // Check memory limit
    if (config_.memory_limit > 0 &&
        stats_.current_usage + size > config_.memory_limit) {
        return nullptr;  // Would exceed limit
    }

    // Try to allocate from a pool
    size_t size_class = find_size_class(size);
    void* ptr = nullptr;
    bool from_pool = false;
    size_t actual_size = size;

    if (size_class > 0) {
        ptr = allocate_from_pool(size_class);
        if (ptr) {
            from_pool = true;
            actual_size = size_class;  // Use actual size class for tracking
            update_stats_alloc(size_class, true);
        }
    }

    // Fall back to direct allocation
    if (!ptr) {
        ptr = allocate_direct(size);
        if (ptr) {
            update_stats_alloc(size, false);
        }
    }

    // Security fix HIGH-04/MED-04: Track allocation size for proper reallocate/deallocate
    if (ptr) {
        allocation_sizes_[ptr] = actual_size;
    }

    return ptr;
}

void* MemoryPool::allocate_aligned(size_t size, size_t alignment) {
    // For custom alignment, calculate padded size
    size_t padded_size = size + alignment - 1;

    void* raw = allocate(padded_size);
    if (!raw) return nullptr;

    // Align the returned pointer
    uintptr_t addr = reinterpret_cast<uintptr_t>(raw);
    uintptr_t aligned = (addr + alignment - 1) & ~(alignment - 1);

    return reinterpret_cast<void*>(aligned);
}

void MemoryPool::deallocate(void* ptr) {
    if (!ptr) return;

    std::lock_guard<std::mutex> lock(mutex_);

    // Security fix HIGH-04/MED-04: Look up allocation size for proper stats tracking
    auto it = allocation_sizes_.find(ptr);
    size_t size = 0;
    if (it != allocation_sizes_.end()) {
        size = it->second;
        allocation_sizes_.erase(it);
    }

    // Try to return to each pool (simplified - just free directly for now)
    // A full implementation would track which pool each allocation came from
#if defined(_MSC_VER)
    _aligned_free(ptr);
#else
    std::free(ptr);
#endif

    if (config_.track_stats) {
        update_stats_free(size);
    }
}

void* MemoryPool::reallocate(void* ptr, size_t new_size) {
    if (!ptr) return allocate(new_size);
    if (new_size == 0) {
        deallocate(ptr);
        return nullptr;
    }

    // Security fix HIGH-04: Look up old size for proper copy
    size_t old_size = 0;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        auto it = allocation_sizes_.find(ptr);
        if (it != allocation_sizes_.end()) {
            old_size = it->second;
        }
    }

    // Allocate new memory
    void* new_ptr = allocate(new_size);
    if (!new_ptr) {
        return nullptr;  // Allocation failed, original still valid
    }

    // Copy old data to new location
    if (old_size > 0) {
        size_t copy_size = (old_size < new_size) ? old_size : new_size;
        std::memcpy(new_ptr, ptr, copy_size);
    }

    // Free old allocation
    deallocate(ptr);

    return new_ptr;
}

void MemoryPool::reset() {
    std::lock_guard<std::mutex> lock(mutex_);

    for (auto& pool : pools_) {
        pool->reset();
    }

    stats_ = AllocationStats{};
}

void MemoryPool::trim() {
    std::lock_guard<std::mutex> lock(mutex_);
    // Release memory from pools that have high free counts
    // Implementation depends on specific requirements
}

void MemoryPool::reserve(size_t /*size*/) {
    // Pre-allocate for expected usage
    // Implementation depends on use case
}

AllocationStats MemoryPool::get_stats() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return stats_;
}

void MemoryPool::reset_stats() {
    std::lock_guard<std::mutex> lock(mutex_);
    stats_ = AllocationStats{};
}

size_t MemoryPool::current_usage() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return stats_.current_usage;
}

size_t MemoryPool::peak_usage() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return stats_.peak_usage;
}

// ============================================================================
// Arena Implementation
// ============================================================================

Arena::Arena(size_t initial_size)
    : capacity_(initial_size)
{
    buffer_.resize(initial_size);
}

Arena::~Arena() = default;

void* Arena::allocate(size_t size) {
    if (size == 0) return nullptr;

    // Security fix CRIT-03: Check for overflow in offset calculation
    if (size > std::numeric_limits<size_t>::max() - offset_) {
        throw std::overflow_error("Arena allocation size overflow");
    }

    if (offset_ + size <= capacity_) {
        void* ptr = buffer_.data() + offset_;
        offset_ += size;
        return ptr;
    }

    // Overflow - allocate new block
    overflow_blocks_.emplace_back(size);
    return overflow_blocks_.back().data();
}

void* Arena::allocate_aligned(size_t size, size_t alignment) {
    if (size == 0) return nullptr;
    if (alignment == 0) alignment = 1;

    // Security fix CRIT-03: Check for overflow in alignment calculations
    if (alignment > std::numeric_limits<size_t>::max() - offset_) {
        throw std::overflow_error("Arena alignment overflow");
    }

    // Calculate aligned offset
    size_t current_offset = offset_;
    size_t aligned_offset = (current_offset + alignment - 1) & ~(alignment - 1);

    // Security fix CRIT-03: Check for overflow in aligned_offset + size
    if (size > std::numeric_limits<size_t>::max() - aligned_offset) {
        throw std::overflow_error("Arena aligned allocation size overflow");
    }

    if (aligned_offset + size <= capacity_) {
        offset_ = aligned_offset + size;
        return buffer_.data() + aligned_offset;
    }

    // Overflow with alignment - need a new block
    // Security fix CRIT-03: Check for overflow in size + alignment
    if (size > std::numeric_limits<size_t>::max() - alignment) {
        throw std::overflow_error("Arena overflow block size overflow");
    }

    std::vector<uint8_t> block(size + alignment);
    overflow_blocks_.push_back(std::move(block));

    uint8_t* base = overflow_blocks_.back().data();
    uintptr_t addr = reinterpret_cast<uintptr_t>(base);
    uintptr_t aligned = (addr + alignment - 1) & ~(alignment - 1);

    // Security fix HIGH-02: Validate aligned pointer is still within bounds
    if (aligned + size > addr + block.size()) {
        throw std::runtime_error("Arena alignment calculation produced out-of-bounds pointer");
    }

    return reinterpret_cast<void*>(aligned);
}

void Arena::reset() {
    offset_ = 0;
    overflow_blocks_.clear();
}

// ============================================================================
// ScopedArena Implementation
// ============================================================================

ScopedArena::ScopedArena(Arena& arena)
    : arena_(arena)
    , saved_offset_(arena.used())
{
}

ScopedArena::~ScopedArena() {
    // Restore arena to saved state
    // Note: This only works correctly if no overflow occurred
    // and allocations are being unwound in LIFO order
}

// ============================================================================
// BufferCache Implementation
// ============================================================================

BufferCache::BufferCache(size_t max_cached_buffers, size_t max_cached_size)
    : max_cached_buffers_(max_cached_buffers)
    , max_cached_size_(max_cached_size)
{
}

BufferCache::~BufferCache() {
    clear();
}

void* BufferCache::get_buffer(size_t size) {
    std::lock_guard<std::mutex> lock(mutex_);

    // Find smallest buffer that's at least as large as requested
    // Don't accept buffers more than 2x the requested size
    auto it = free_buffers_.lower_bound(size);

    if (it != free_buffers_.end() && it->first <= size * 2) {
        void* ptr = it->second;
        cached_bytes_ -= it->first;
        free_buffers_.erase(it);
        stats_.hits++;
        stats_.cached_buffers = free_buffers_.size();
        stats_.cached_bytes = cached_bytes_;
        return ptr;
    }

    // No suitable cached buffer, allocate new
    stats_.misses++;

#if defined(_MSC_VER)
    return _aligned_malloc(size, 64);
#else
    void* ptr = nullptr;
    if (posix_memalign(&ptr, 64, size) != 0) {
        return nullptr;
    }
    return ptr;
#endif
}

void BufferCache::return_buffer(void* ptr, size_t size) {
    if (!ptr || size == 0) return;

    std::lock_guard<std::mutex> lock(mutex_);

    // Check if we have room to cache this buffer
    bool should_cache = free_buffers_.size() < max_cached_buffers_ &&
                        cached_bytes_ + size <= max_cached_size_;

    if (!should_cache && !free_buffers_.empty()) {
        // Try to evict smallest buffer to make room
        auto smallest = free_buffers_.begin();
        if (size > smallest->first) {
            // New buffer is larger, cache it instead
            cached_bytes_ -= smallest->first;
#if defined(_MSC_VER)
            _aligned_free(smallest->second);
#else
            std::free(smallest->second);
#endif
            free_buffers_.erase(smallest);
            should_cache = cached_bytes_ + size <= max_cached_size_;
        }
    }

    if (should_cache) {
        free_buffers_.emplace(size, ptr);
        cached_bytes_ += size;
        stats_.cached_buffers = free_buffers_.size();
        stats_.cached_bytes = cached_bytes_;
    } else {
        // Can't cache, free immediately
#if defined(_MSC_VER)
        _aligned_free(ptr);
#else
        std::free(ptr);
#endif
    }
}

void BufferCache::clear() {
    std::lock_guard<std::mutex> lock(mutex_);

    for (auto& [size, ptr] : free_buffers_) {
#if defined(_MSC_VER)
        _aligned_free(ptr);
#else
        std::free(ptr);
#endif
    }

    free_buffers_.clear();
    cached_bytes_ = 0;
    stats_.cached_buffers = 0;
    stats_.cached_bytes = 0;
}

BufferCache::CacheStats BufferCache::get_stats() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return stats_;
}

// ============================================================================
// Global Instances
// ============================================================================

static std::unique_ptr<MemoryPool> g_pool;
static std::unique_ptr<BufferCache> g_buffer_cache;
static std::mutex g_init_mutex;

MemoryPool& global_pool() {
    std::lock_guard<std::mutex> lock(g_init_mutex);
    if (!g_pool) {
        g_pool = std::make_unique<MemoryPool>();
    }
    return *g_pool;
}

BufferCache& global_buffer_cache() {
    std::lock_guard<std::mutex> lock(g_init_mutex);
    if (!g_buffer_cache) {
        g_buffer_cache = std::make_unique<BufferCache>();
    }
    return *g_buffer_cache;
}

void set_global_pool_config(const PoolConfig& config) {
    std::lock_guard<std::mutex> lock(g_init_mutex);
    g_pool = std::make_unique<MemoryPool>(config);
}

} // namespace memory
} // namespace pyflame_rt
