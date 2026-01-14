#include "pyflame_rt/cache/mmap_loader.hpp"
#include "pyflame_rt/io/model_io.hpp"
#include "pyflame_rt/errors.hpp"
#include <stdexcept>
#include <cstring>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#else
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#endif

namespace fs = std::filesystem;

namespace pyflame_rt {
namespace cache {

// ============================================================================
// MappedFile Implementation
// ============================================================================

void MappedFile::close_handles() {
#ifdef _WIN32
    if (data_) {
        UnmapViewOfFile(data_);
        data_ = nullptr;
    }
    if (mapping_handle_) {
        CloseHandle(mapping_handle_);
        mapping_handle_ = nullptr;
    }
    if (file_handle_ && file_handle_ != INVALID_HANDLE_VALUE) {
        CloseHandle(file_handle_);
        file_handle_ = nullptr;
    }
#else
    if (data_ && data_ != MAP_FAILED) {
        munmap(data_, size_);
        data_ = nullptr;
    }
    if (fd_ >= 0) {
        ::close(fd_);
        fd_ = -1;
    }
#endif
    size_ = 0;
}

MappedFile::~MappedFile() {
    close_handles();
}

MappedFile::MappedFile(MappedFile&& other) noexcept
    : data_(other.data_)
    , size_(other.size_)
#ifdef _WIN32
    , file_handle_(other.file_handle_)
    , mapping_handle_(other.mapping_handle_)
#else
    , fd_(other.fd_)
#endif
{
    other.data_ = nullptr;
    other.size_ = 0;
#ifdef _WIN32
    other.file_handle_ = nullptr;
    other.mapping_handle_ = nullptr;
#else
    other.fd_ = -1;
#endif
}

MappedFile& MappedFile::operator=(MappedFile&& other) noexcept {
    if (this != &other) {
        close_handles();

        data_ = other.data_;
        size_ = other.size_;
#ifdef _WIN32
        file_handle_ = other.file_handle_;
        mapping_handle_ = other.mapping_handle_;
        other.file_handle_ = nullptr;
        other.mapping_handle_ = nullptr;
#else
        fd_ = other.fd_;
        other.fd_ = -1;
#endif
        other.data_ = nullptr;
        other.size_ = 0;
    }
    return *this;
}

std::unique_ptr<MappedFile> MappedFile::open(const fs::path& path) {
    auto file = std::make_unique<MappedFile>();

#ifdef _WIN32
    // Windows implementation
    file->file_handle_ = CreateFileW(
        path.wstring().c_str(),
        GENERIC_READ,
        FILE_SHARE_READ,
        nullptr,
        OPEN_EXISTING,
        FILE_ATTRIBUTE_NORMAL | FILE_FLAG_SEQUENTIAL_SCAN,
        nullptr);

    if (file->file_handle_ == INVALID_HANDLE_VALUE) {
        return nullptr;
    }

    LARGE_INTEGER file_size;
    if (!GetFileSizeEx(file->file_handle_, &file_size)) {
        CloseHandle(file->file_handle_);
        return nullptr;
    }
    file->size_ = static_cast<size_t>(file_size.QuadPart);

    if (file->size_ == 0) {
        CloseHandle(file->file_handle_);
        return nullptr;
    }

    file->mapping_handle_ = CreateFileMappingW(
        file->file_handle_,
        nullptr,
        PAGE_READONLY,
        0, 0,
        nullptr);

    if (!file->mapping_handle_) {
        CloseHandle(file->file_handle_);
        return nullptr;
    }

    file->data_ = MapViewOfFile(
        file->mapping_handle_,
        FILE_MAP_READ,
        0, 0, 0);

    if (!file->data_) {
        CloseHandle(file->mapping_handle_);
        CloseHandle(file->file_handle_);
        return nullptr;
    }

#else
    // Unix/POSIX implementation
    file->fd_ = ::open(path.c_str(), O_RDONLY);
    if (file->fd_ < 0) {
        return nullptr;
    }

    struct stat st;
    if (fstat(file->fd_, &st) < 0) {
        ::close(file->fd_);
        return nullptr;
    }
    file->size_ = static_cast<size_t>(st.st_size);

    if (file->size_ == 0) {
        ::close(file->fd_);
        return nullptr;
    }

    file->data_ = mmap(nullptr, file->size_,
                       PROT_READ, MAP_PRIVATE,
                       file->fd_, 0);

    if (file->data_ == MAP_FAILED) {
        ::close(file->fd_);
        file->data_ = nullptr;
        return nullptr;
    }
#endif

    return file;
}

void MappedFile::advise(Advice advice) {
    advise(advice, 0, size_);
}

void MappedFile::advise(Advice advice, size_t offset, size_t length) {
    if (!data_ || offset + length > size_) return;

#ifdef _WIN32
    // Windows: Use PrefetchVirtualMemory for WillNeed
    if (advice == Advice::WillNeed) {
        WIN32_MEMORY_RANGE_ENTRY range;
        range.VirtualAddress = static_cast<uint8_t*>(data_) + offset;
        range.NumberOfBytes = length;
        PrefetchVirtualMemory(GetCurrentProcess(), 1, &range, 0);
    }
    // Other advice types don't have direct Windows equivalents
#else
    int adv = MADV_NORMAL;
    switch (advice) {
        case Advice::Normal:     adv = MADV_NORMAL; break;
        case Advice::Sequential: adv = MADV_SEQUENTIAL; break;
        case Advice::Random:     adv = MADV_RANDOM; break;
        case Advice::WillNeed:   adv = MADV_WILLNEED; break;
        case Advice::DontNeed:   adv = MADV_DONTNEED; break;
    }
    madvise(static_cast<uint8_t*>(data_) + offset, length, adv);
#endif
}

bool MappedFile::lock() {
    if (!data_) return false;

#ifdef _WIN32
    return VirtualLock(data_, size_) != 0;
#else
    return mlock(data_, size_) == 0;
#endif
}

void MappedFile::unlock() {
    if (!data_) return;

#ifdef _WIN32
    VirtualUnlock(data_, size_);
#else
    munlock(data_, size_);
#endif
}

// ============================================================================
// MMapLoader Implementation
// ============================================================================

std::unique_ptr<Graph> MMapLoader::load(const fs::path& path) {
    auto file = MappedFile::open(path);
    if (!file || !file->is_valid()) {
        // Fall back to regular file loading
        return io::load_model(path.string());
    }

    // Advise kernel about sequential access pattern
    file->advise(MappedFile::Advice::Sequential);

    // Try to parse model from mapped memory
    try {
        return io::load_model_from_buffer(file->data(), file->size());
    } catch (const InvalidModelError&) {
        // Buffer loading not implemented, fall back to file loading
        return io::load_model(path.string());
    }
}

Tensor MMapLoader::load_tensor(const MappedFile& file,
                                size_t offset,
                                const std::vector<int64_t>& shape,
                                DType dtype) {
    if (!file.is_valid()) {
        throw std::invalid_argument("MMapLoader::load_tensor: invalid mapped file");
    }

    if (offset >= file.size()) {
        throw std::out_of_range("MMapLoader::load_tensor: offset beyond file size");
    }

    // Calculate tensor size
    int64_t num_elements = 1;
    for (int64_t dim : shape) {
        if (dim <= 0) {
            throw std::invalid_argument("MMapLoader::load_tensor: invalid shape dimension");
        }
        num_elements *= dim;
    }

    size_t tensor_bytes = static_cast<size_t>(num_elements) * dtype_size(dtype);

    if (offset + tensor_bytes > file.size()) {
        throw std::out_of_range("MMapLoader::load_tensor: tensor extends beyond file");
    }

    // Create tensor that references the mapped memory
    const void* data_ptr = static_cast<const uint8_t*>(file.data()) + offset;

    // Create a borrowed tensor (doesn't own the memory)
    return Tensor::from_external(data_ptr, shape, dtype);
}

bool MMapLoader::can_mmap(const fs::path& path) {
    // Check if file exists and is readable
    std::error_code ec;
    if (!fs::exists(path, ec) || ec) {
        return false;
    }

    // Check file size (very small files might not benefit from mmap)
    auto size = fs::file_size(path, ec);
    if (ec || size < 4096) {
        return false;
    }

    // Try to actually open the file with mmap
    auto file = MappedFile::open(path);
    return file && file->is_valid();
}

size_t MMapLoader::page_alignment() {
#ifdef _WIN32
    SYSTEM_INFO si;
    GetSystemInfo(&si);
    return static_cast<size_t>(si.dwPageSize);
#else
    return static_cast<size_t>(sysconf(_SC_PAGESIZE));
#endif
}

// ============================================================================
// LazyTensor Implementation
// ============================================================================

LazyTensor::LazyTensor(std::shared_ptr<MappedFile> file,
                       size_t offset,
                       std::vector<int64_t> shape,
                       DType dtype)
    : file_(std::move(file))
    , offset_(offset)
    , shape_(std::move(shape))
    , dtype_(dtype)
{
}

Tensor LazyTensor::materialize() const {
    if (materialized_) {
        return *materialized_;
    }

    if (!file_ || !file_->is_valid()) {
        throw std::runtime_error("LazyTensor::materialize: no backing file");
    }

    materialized_ = MMapLoader::load_tensor(*file_, offset_, shape_, dtype_);
    return *materialized_;
}

int64_t LazyTensor::num_elements() const {
    int64_t count = 1;
    for (int64_t dim : shape_) {
        count *= dim;
    }
    return count;
}

size_t LazyTensor::size_bytes() const {
    return static_cast<size_t>(num_elements()) * dtype_size(dtype_);
}

} // namespace cache
} // namespace pyflame_rt
