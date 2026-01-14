#pragma once

#include "pyflame_rt/tensor.hpp"
#include "pyflame_rt/graph.hpp"
#include <filesystem>
#include <memory>
#include <string>
#include <optional>
#include <vector>

namespace pyflame_rt {
namespace cache {

/// Memory-mapped file handle with RAII semantics
class MappedFile {
public:
    MappedFile() = default;
    ~MappedFile();

    // Move-only semantics
    MappedFile(MappedFile&& other) noexcept;
    MappedFile& operator=(MappedFile&& other) noexcept;
    MappedFile(const MappedFile&) = delete;
    MappedFile& operator=(const MappedFile&) = delete;

    /// Open and map a file into memory
    /// @param path Path to the file to map
    /// @return Unique pointer to mapped file, or nullptr on failure
    static std::unique_ptr<MappedFile> open(const std::filesystem::path& path);

    /// Get pointer to the mapped data
    const void* data() const { return data_; }
    void* data() { return data_; }

    /// Get size of the mapped region
    size_t size() const { return size_; }

    /// Check if the mapping is valid
    bool is_valid() const { return data_ != nullptr; }

    /// Memory advice hints for the kernel
    enum class Advice {
        Normal,      ///< No special treatment
        Sequential,  ///< Will be accessed sequentially
        Random,      ///< Will be accessed randomly
        WillNeed,    ///< Will be needed soon (prefetch)
        DontNeed     ///< Won't be needed soon (can be paged out)
    };

    /// Advise kernel about expected access pattern for entire file
    void advise(Advice advice);

    /// Advise kernel about expected access pattern for a region
    void advise(Advice advice, size_t offset, size_t length);

    /// Lock pages in memory (prevent paging to disk)
    /// @return true if successfully locked
    bool lock();

    /// Unlock pages (allow paging to disk)
    void unlock();

private:
    void* data_ = nullptr;
    size_t size_ = 0;

#ifdef _WIN32
    void* file_handle_ = nullptr;
    void* mapping_handle_ = nullptr;
#else
    int fd_ = -1;
#endif

    void close_handles();
};

/// Memory-mapped model loader for fast startup
class MMapLoader {
public:
    /// Load a model using memory mapping
    /// @param path Path to the model file
    /// @return Loaded graph, or nullptr on failure
    static std::unique_ptr<Graph> load(const std::filesystem::path& path);

    /// Load a tensor directly from a memory-mapped file (zero-copy)
    /// @param file The mapped file containing the tensor data
    /// @param offset Byte offset within the file
    /// @param shape Shape of the tensor
    /// @param dtype Data type of the tensor
    /// @return Tensor referencing the mapped memory
    static Tensor load_tensor(const MappedFile& file,
                              size_t offset,
                              const std::vector<int64_t>& shape,
                              DType dtype);

    /// Check if a file can be memory-mapped
    static bool can_mmap(const std::filesystem::path& path);

    /// Get the system's page alignment requirement
    static size_t page_alignment();
};

/// Lazy tensor that loads data on first access
class LazyTensor {
public:
    LazyTensor() = default;

    /// Create a lazy tensor backed by a memory-mapped file
    /// @param file Shared pointer to the mapped file (keeps file alive)
    /// @param offset Byte offset of tensor data within the file
    /// @param shape Shape of the tensor
    /// @param dtype Data type of the tensor
    LazyTensor(std::shared_ptr<MappedFile> file,
               size_t offset,
               std::vector<int64_t> shape,
               DType dtype);

    /// Materialize the tensor (load data into memory if needed)
    /// @return The tensor data
    Tensor materialize() const;

    /// Check if tensor has been materialized
    bool is_materialized() const { return materialized_.has_value(); }

    /// Get tensor shape without loading data
    const std::vector<int64_t>& shape() const { return shape_; }

    /// Get tensor dtype without loading data
    DType dtype() const { return dtype_; }

    /// Get number of elements without loading data
    int64_t num_elements() const;

    /// Get size in bytes without loading data
    size_t size_bytes() const;

    /// Check if valid (has backing file)
    bool is_valid() const { return file_ != nullptr; }

private:
    std::shared_ptr<MappedFile> file_;
    size_t offset_ = 0;
    std::vector<int64_t> shape_;
    DType dtype_ = DType::Float32;
    mutable std::optional<Tensor> materialized_;
};

/// Options for memory-mapped loading
struct MMapOptions {
    /// Use memory mapping if available
    bool enabled = true;

    /// Minimum file size to use mmap (smaller files use regular I/O)
    size_t min_file_size = 4096;

    /// Lock pages in memory after mapping
    bool lock_pages = false;

    /// Prefetch entire file after mapping
    bool prefetch = true;

    /// Use lazy loading for tensors
    bool lazy_tensors = false;
};

} // namespace cache
} // namespace pyflame_rt
