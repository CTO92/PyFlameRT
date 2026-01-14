#pragma once

#include "pyflame_rt/graph.hpp"
#include "pyflame_rt/options.hpp"
#include <filesystem>
#include <memory>
#include <string>
#include <optional>
#include <chrono>
#include <mutex>
#include <functional>
#include <vector>

namespace pyflame_rt {
namespace cache {

/// Cache entry metadata
struct CacheEntryInfo {
    std::string key;
    std::string model_path;
    std::string cache_path;
    std::chrono::system_clock::time_point created_at;
    std::chrono::system_clock::time_point last_accessed;
    size_t size_bytes = 0;
    std::string version;
    std::string options_hash;
    bool is_valid = false;
};

/// Cache key computation utilities
struct CacheKey {
    /// Compute cache key from model file path and session options
    static std::string compute(const std::string& model_path,
                               const SessionOptions& options);

    /// Compute cache key from model data in memory
    static std::string compute(const void* model_data, size_t model_size,
                               const SessionOptions& options);

    /// Get hash of session options that affect compilation
    static std::string hash_options(const SessionOptions& options);

    /// Compute a hash from raw bytes
    static std::string hash_bytes(const void* data, size_t size);
};

/// Cache storage configuration
struct CacheConfig {
    /// Directory to store cached artifacts
    std::filesystem::path cache_dir = ".pyflame_cache";

    /// Maximum total cache size in bytes (0 = unlimited)
    size_t max_size_bytes = 0;

    /// Maximum number of cached entries (0 = unlimited)
    size_t max_entries = 0;

    /// Time-to-live for cache entries in seconds (0 = never expire)
    std::chrono::seconds ttl = std::chrono::seconds(0);

    /// Enable memory-mapped loading of cached models
    bool use_mmap = true;

    /// Enable compression for cached artifacts
    bool compress = false;

    /// Cache version string (changing this invalidates old entries)
    std::string version = "1.0";

    /// Enable thread-safe concurrent access
    bool thread_safe = true;
};

/// Binary cache manager for compiled model artifacts
class BinaryCache {
public:
    explicit BinaryCache(const CacheConfig& config = CacheConfig());
    ~BinaryCache();

    // Non-copyable
    BinaryCache(const BinaryCache&) = delete;
    BinaryCache& operator=(const BinaryCache&) = delete;

    // ========================================================================
    // Cache Operations
    // ========================================================================

    /// Check if a cached entry exists for the given key
    bool has(const std::string& key) const;

    /// Get cached compiled graph (returns nullptr if not found or invalid)
    std::unique_ptr<Graph> get(const std::string& key);

    /// Store compiled graph in cache
    /// @return true if stored successfully
    bool put(const std::string& key, const Graph& graph,
             const CacheEntryInfo& info = CacheEntryInfo());

    /// Remove entry from cache
    /// @return true if entry was removed
    bool remove(const std::string& key);

    /// Clear all cached entries
    void clear();

    // ========================================================================
    // Cache Management
    // ========================================================================

    /// Get information about a cached entry
    std::optional<CacheEntryInfo> get_info(const std::string& key) const;

    /// Get all cached entry keys
    std::vector<std::string> list_entries() const;

    /// Get total cache size in bytes
    size_t total_size() const;

    /// Get number of cached entries
    size_t entry_count() const;

    /// Manually trigger eviction to meet size/count constraints
    void evict_if_needed();

    /// Validate cache integrity
    bool validate() const;

    // ========================================================================
    // High-Level API
    // ========================================================================

    /// Get graph from cache, or compile and cache it
    /// @param model_path Path to the model file
    /// @param options Session options used for compilation
    /// @param compile_fn Function to compile the model if not cached
    /// @return Compiled graph (from cache or freshly compiled)
    std::unique_ptr<Graph> get_or_compile(
        const std::string& model_path,
        const SessionOptions& options,
        std::function<std::unique_ptr<Graph>()> compile_fn);

    /// Warm up cache by pre-compiling multiple models
    void warmup(const std::vector<std::string>& model_paths,
                const SessionOptions& options);

    // ========================================================================
    // Configuration
    // ========================================================================

    /// Get current cache configuration
    const CacheConfig& config() const { return config_; }

    /// Set the cache directory
    void set_cache_dir(const std::filesystem::path& dir);

    /// Enable or disable caching
    void set_enabled(bool enabled) { enabled_ = enabled; }

    /// Check if caching is enabled
    bool is_enabled() const { return enabled_; }

    // ========================================================================
    // Statistics
    // ========================================================================

    struct Stats {
        size_t hits = 0;
        size_t misses = 0;
        size_t stores = 0;
        size_t evictions = 0;

        double hit_rate() const {
            size_t total = hits + misses;
            return total > 0 ? static_cast<double>(hits) / total : 0.0;
        }
    };

    Stats get_stats() const;
    void reset_stats();

private:
    CacheConfig config_;
    bool enabled_ = true;
    mutable std::mutex mutex_;
    mutable Stats stats_;

    std::filesystem::path get_entry_path(const std::string& key) const;
    bool write_entry(const std::string& key, const Graph& graph);
    std::unique_ptr<Graph> read_entry(const std::string& key);
    void update_access_time(const std::string& key);
    void evict_lru();
    bool is_entry_expired(const std::string& key) const;
};

// ============================================================================
// Global Cache
// ============================================================================

/// Get the global binary cache instance
BinaryCache& global_cache();

/// Set the global cache configuration
void set_global_cache_config(const CacheConfig& config);

} // namespace cache
} // namespace pyflame_rt
