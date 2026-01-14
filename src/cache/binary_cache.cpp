#include "pyflame_rt/cache/binary_cache.hpp"
#include "pyflame_rt/io/model_io.hpp"
#include <fstream>
#include <sstream>
#include <iomanip>
#include <algorithm>

namespace fs = std::filesystem;

namespace pyflame_rt {
namespace cache {

namespace {

// FNV-1a hash for cache key computation
uint64_t fnv1a_hash(const uint8_t* data, size_t size) {
    uint64_t hash = 14695981039346656037ULL;
    for (size_t i = 0; i < size; ++i) {
        hash ^= data[i];
        hash *= 1099511628211ULL;
    }
    return hash;
}

std::string to_hex_string(uint64_t value) {
    std::ostringstream oss;
    oss << std::hex << std::setfill('0') << std::setw(16) << value;
    return oss.str();
}

} // anonymous namespace

// ============================================================================
// CacheKey Implementation
// ============================================================================

std::string CacheKey::hash_bytes(const void* data, size_t size) {
    uint64_t hash = fnv1a_hash(static_cast<const uint8_t*>(data), size);
    return to_hex_string(hash);
}

std::string CacheKey::compute(const std::string& model_path,
                               const SessionOptions& options) {
    // Read model file for hashing
    std::ifstream file(model_path, std::ios::binary | std::ios::ate);
    if (!file) {
        throw std::runtime_error("Cannot open model file: " + model_path);
    }

    size_t size = static_cast<size_t>(file.tellg());
    file.seekg(0, std::ios::beg);

    std::vector<uint8_t> data(size);
    if (!file.read(reinterpret_cast<char*>(data.data()), size)) {
        throw std::runtime_error("Failed to read model file: " + model_path);
    }

    return compute(data.data(), data.size(), options);
}

std::string CacheKey::compute(const void* model_data, size_t model_size,
                               const SessionOptions& options) {
    // Hash model content
    std::string model_hash = hash_bytes(model_data, model_size);

    // Hash relevant options
    std::string options_hash = hash_options(options);

    return model_hash + "_" + options_hash;
}

std::string CacheKey::hash_options(const SessionOptions& options) {
    // Create a deterministic string representation of options
    std::ostringstream oss;
    oss << "threads:" << options.num_threads << ";";
    oss << "optlevel:" << static_cast<int>(options.optimization_level) << ";";
    oss << "profile:" << options.enable_profiling << ";";
    oss << "memlimit:" << options.memory_limit << ";";

    std::string opts_str = oss.str();
    return hash_bytes(opts_str.data(), opts_str.size());
}

// ============================================================================
// BinaryCache Implementation
// ============================================================================

BinaryCache::BinaryCache(const CacheConfig& config)
    : config_(config)
{
    // Create cache directory if it doesn't exist
    if (!fs::exists(config_.cache_dir)) {
        std::error_code ec;
        fs::create_directories(config_.cache_dir, ec);
        if (ec) {
            // Failed to create directory, disable caching
            enabled_ = false;
        }
    }
}

BinaryCache::~BinaryCache() = default;

fs::path BinaryCache::get_entry_path(const std::string& key) const {
    return config_.cache_dir / (key + ".pfcache");
}

bool BinaryCache::has(const std::string& key) const {
    if (!enabled_) return false;

    std::lock_guard<std::mutex> lock(mutex_);
    fs::path entry_path = get_entry_path(key);

    if (!fs::exists(entry_path)) {
        return false;
    }

    // Check if entry has expired
    if (is_entry_expired(key)) {
        return false;
    }

    return true;
}

std::unique_ptr<Graph> BinaryCache::get(const std::string& key) {
    if (!enabled_) {
        stats_.misses++;
        return nullptr;
    }

    std::lock_guard<std::mutex> lock(mutex_);

    fs::path entry_path = get_entry_path(key);

    if (!fs::exists(entry_path)) {
        stats_.misses++;
        return nullptr;
    }

    // Check expiration
    if (is_entry_expired(key)) {
        stats_.misses++;
        return nullptr;
    }

    try {
        auto graph = read_entry(key);
        if (graph) {
            update_access_time(key);
            stats_.hits++;
            return graph;
        }
    } catch (const std::exception&) {
        // Invalid cache entry, remove it
        fs::remove(entry_path);
    }

    stats_.misses++;
    return nullptr;
}

bool BinaryCache::put(const std::string& key, const Graph& graph,
                       const CacheEntryInfo& /*info*/) {
    if (!enabled_) return false;

    std::lock_guard<std::mutex> lock(mutex_);

    // Evict entries if needed to make room
    evict_if_needed();

    try {
        bool success = write_entry(key, graph);
        if (success) {
            stats_.stores++;
        }
        return success;
    } catch (const std::exception&) {
        return false;
    }
}

bool BinaryCache::remove(const std::string& key) {
    std::lock_guard<std::mutex> lock(mutex_);

    fs::path entry_path = get_entry_path(key);

    if (fs::exists(entry_path)) {
        std::error_code ec;
        return fs::remove(entry_path, ec);
    }

    return false;
}

void BinaryCache::clear() {
    std::lock_guard<std::mutex> lock(mutex_);

    std::error_code ec;
    for (const auto& entry : fs::directory_iterator(config_.cache_dir, ec)) {
        if (entry.path().extension() == ".pfcache") {
            fs::remove(entry.path(), ec);
        }
    }
}

std::optional<CacheEntryInfo> BinaryCache::get_info(const std::string& key) const {
    std::lock_guard<std::mutex> lock(mutex_);

    fs::path entry_path = get_entry_path(key);

    if (!fs::exists(entry_path)) {
        return std::nullopt;
    }

    CacheEntryInfo info;
    info.key = key;
    info.cache_path = entry_path.string();

    std::error_code ec;
    info.size_bytes = fs::file_size(entry_path, ec);
    if (ec) {
        info.size_bytes = 0;
    }

    info.is_valid = !is_entry_expired(key);
    info.version = config_.version;

    return info;
}

std::vector<std::string> BinaryCache::list_entries() const {
    std::lock_guard<std::mutex> lock(mutex_);

    std::vector<std::string> entries;
    std::error_code ec;

    for (const auto& entry : fs::directory_iterator(config_.cache_dir, ec)) {
        if (entry.path().extension() == ".pfcache") {
            entries.push_back(entry.path().stem().string());
        }
    }

    return entries;
}

size_t BinaryCache::total_size() const {
    std::lock_guard<std::mutex> lock(mutex_);

    size_t total = 0;
    std::error_code ec;

    for (const auto& entry : fs::directory_iterator(config_.cache_dir, ec)) {
        if (entry.path().extension() == ".pfcache") {
            total += fs::file_size(entry.path(), ec);
        }
    }

    return total;
}

size_t BinaryCache::entry_count() const {
    std::lock_guard<std::mutex> lock(mutex_);

    size_t count = 0;
    std::error_code ec;

    for (const auto& entry : fs::directory_iterator(config_.cache_dir, ec)) {
        if (entry.path().extension() == ".pfcache") {
            count++;
        }
    }

    return count;
}

void BinaryCache::evict_if_needed() {
    // Check max entries constraint
    if (config_.max_entries > 0) {
        while (entry_count() >= config_.max_entries) {
            evict_lru();
            stats_.evictions++;
        }
    }

    // Check max size constraint
    if (config_.max_size_bytes > 0) {
        while (total_size() >= config_.max_size_bytes) {
            evict_lru();
            stats_.evictions++;
        }
    }
}

void BinaryCache::evict_lru() {
    // Find the oldest (least recently accessed) entry
    fs::file_time_type oldest_time = fs::file_time_type::max();
    fs::path oldest_path;

    std::error_code ec;
    for (const auto& entry : fs::directory_iterator(config_.cache_dir, ec)) {
        if (entry.path().extension() == ".pfcache") {
            auto ftime = fs::last_write_time(entry.path(), ec);
            if (!ec && ftime < oldest_time) {
                oldest_time = ftime;
                oldest_path = entry.path();
            }
        }
    }

    if (!oldest_path.empty()) {
        fs::remove(oldest_path, ec);
    }
}

bool BinaryCache::is_entry_expired(const std::string& key) const {
    if (config_.ttl == std::chrono::seconds(0)) {
        return false;  // No expiration
    }

    fs::path entry_path = get_entry_path(key);

    std::error_code ec;
    auto ftime = fs::last_write_time(entry_path, ec);
    if (ec) {
        return true;  // Can't read time, consider expired
    }

    // Convert file_time_type to system_clock for comparison
    auto now = std::chrono::system_clock::now();
    auto file_time_sys = std::chrono::time_point_cast<std::chrono::system_clock::duration>(
        ftime - fs::file_time_type::clock::now() + now);

    auto age = now - file_time_sys;
    return age > config_.ttl;
}

bool BinaryCache::validate() const {
    std::lock_guard<std::mutex> lock(mutex_);

    std::error_code ec;
    for (const auto& entry : fs::directory_iterator(config_.cache_dir, ec)) {
        if (entry.path().extension() == ".pfcache") {
            // Try to read the file header to validate
            std::ifstream file(entry.path(), std::ios::binary);
            if (!file) {
                return false;
            }

            // Check for magic bytes
            char magic[8];
            if (!file.read(magic, 8)) {
                return false;
            }

            // Validate magic matches expected format
            if (std::string(magic, 8) != "PFMODEL\0" &&
                std::string(magic, 7) != "PFCACHE") {
                return false;
            }
        }
    }

    return true;
}

std::unique_ptr<Graph> BinaryCache::get_or_compile(
    const std::string& model_path,
    const SessionOptions& options,
    std::function<std::unique_ptr<Graph>()> compile_fn)
{
    std::string key = CacheKey::compute(model_path, options);

    // Try cache first
    auto cached = get(key);
    if (cached) {
        return cached;
    }

    // Compile the model
    auto graph = compile_fn();

    // Cache the result
    if (graph) {
        put(key, *graph);
    }

    return graph;
}

void BinaryCache::warmup(const std::vector<std::string>& model_paths,
                          const SessionOptions& options) {
    for (const auto& path : model_paths) {
        try {
            std::string key = CacheKey::compute(path, options);

            if (!has(key)) {
                // Load and cache the model
                auto graph = io::load_model(path);
                if (graph) {
                    put(key, *graph);
                }
            }
        } catch (const std::exception&) {
            // Continue with other models
        }
    }
}

void BinaryCache::set_cache_dir(const fs::path& dir) {
    std::lock_guard<std::mutex> lock(mutex_);

    config_.cache_dir = dir;

    if (!fs::exists(config_.cache_dir)) {
        std::error_code ec;
        fs::create_directories(config_.cache_dir, ec);
        if (ec) {
            enabled_ = false;
        }
    }
}

BinaryCache::Stats BinaryCache::get_stats() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return stats_;
}

void BinaryCache::reset_stats() {
    std::lock_guard<std::mutex> lock(mutex_);
    stats_ = Stats{};
}

bool BinaryCache::write_entry(const std::string& key, const Graph& graph) {
    fs::path entry_path = get_entry_path(key);
    return io::save_model(entry_path.string(), graph);
}

std::unique_ptr<Graph> BinaryCache::read_entry(const std::string& key) {
    fs::path entry_path = get_entry_path(key);
    return io::load_model(entry_path.string());
}

void BinaryCache::update_access_time(const std::string& key) {
    fs::path entry_path = get_entry_path(key);

    std::error_code ec;
    fs::last_write_time(entry_path, fs::file_time_type::clock::now(), ec);
}

// ============================================================================
// Global Cache
// ============================================================================

static std::unique_ptr<BinaryCache> g_cache;
static std::mutex g_cache_mutex;

BinaryCache& global_cache() {
    std::lock_guard<std::mutex> lock(g_cache_mutex);
    if (!g_cache) {
        g_cache = std::make_unique<BinaryCache>();
    }
    return *g_cache;
}

void set_global_cache_config(const CacheConfig& config) {
    std::lock_guard<std::mutex> lock(g_cache_mutex);
    g_cache = std::make_unique<BinaryCache>(config);
}

} // namespace cache
} // namespace pyflame_rt
