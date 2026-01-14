#include "pyflame_rt/serving/types.hpp"
#include <random>
#include <sstream>
#include <iomanip>
#include <chrono>

namespace pyflame_rt {
namespace serving {

std::string InferRequest::generate_id() {
    static std::random_device rd;
    static std::mt19937_64 gen(rd());
    static std::uniform_int_distribution<uint64_t> dis;

    // Combine timestamp with random value for uniqueness
    auto now = std::chrono::high_resolution_clock::now();
    auto timestamp = std::chrono::duration_cast<std::chrono::microseconds>(
        now.time_since_epoch()).count();

    uint64_t random_part = dis(gen);

    std::ostringstream oss;
    oss << std::hex << std::setfill('0')
        << std::setw(12) << (timestamp & 0xFFFFFFFFFFFF)
        << std::setw(4) << (random_part & 0xFFFF);
    return oss.str();
}

} // namespace serving
} // namespace pyflame_rt
