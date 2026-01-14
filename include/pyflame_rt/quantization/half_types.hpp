#pragma once

#include <cstdint>
#include <cmath>
#include <limits>

namespace pyflame_rt {
namespace quantization {

/// IEEE 754 half-precision float (binary16)
/// Sign: 1 bit, Exponent: 5 bits, Mantissa: 10 bits
struct Float16 {
    uint16_t bits;

    Float16() : bits(0) {}
    explicit Float16(uint16_t raw) : bits(raw) {}

    /// Convert from float32
    static Float16 from_float(float value);

    /// Convert to float32
    float to_float() const;

    /// Check for special values
    bool is_nan() const {
        return ((bits & 0x7C00) == 0x7C00) && ((bits & 0x03FF) != 0);
    }

    bool is_inf() const {
        return ((bits & 0x7FFF) == 0x7C00);
    }

    bool is_zero() const {
        return (bits & 0x7FFF) == 0;
    }

    bool is_negative() const {
        return (bits & 0x8000) != 0;
    }

    // Arithmetic operators (convert to float, compute, convert back)
    Float16 operator+(Float16 other) const {
        return Float16::from_float(to_float() + other.to_float());
    }

    Float16 operator-(Float16 other) const {
        return Float16::from_float(to_float() - other.to_float());
    }

    Float16 operator*(Float16 other) const {
        return Float16::from_float(to_float() * other.to_float());
    }

    Float16 operator/(Float16 other) const {
        return Float16::from_float(to_float() / other.to_float());
    }

    Float16 operator-() const {
        Float16 result;
        result.bits = bits ^ 0x8000;  // Flip sign bit
        return result;
    }

    // Comparison
    bool operator==(Float16 other) const {
        // Handle NaN
        if (is_nan() || other.is_nan()) return false;
        // Handle +0 == -0
        if (is_zero() && other.is_zero()) return true;
        return bits == other.bits;
    }

    bool operator!=(Float16 other) const {
        return !(*this == other);
    }

    bool operator<(Float16 other) const {
        if (is_nan() || other.is_nan()) return false;
        // Handle zeros
        if (is_zero() && other.is_zero()) return false;
        return to_float() < other.to_float();
    }

    bool operator>(Float16 other) const {
        return other < *this;
    }

    bool operator<=(Float16 other) const {
        return !(other < *this);
    }

    bool operator>=(Float16 other) const {
        return !(*this < other);
    }
};

/// Google Brain bfloat16 format
/// Sign: 1 bit, Exponent: 8 bits, Mantissa: 7 bits
/// Same exponent range as float32, reduced precision
struct BFloat16 {
    uint16_t bits;

    BFloat16() : bits(0) {}
    explicit BFloat16(uint16_t raw) : bits(raw) {}

    /// Convert from float32 (truncate mantissa with rounding)
    static BFloat16 from_float(float value);

    /// Convert to float32 (exact, just add zero bits)
    float to_float() const;

    /// Check for special values
    bool is_nan() const {
        return ((bits & 0x7F80) == 0x7F80) && ((bits & 0x007F) != 0);
    }

    bool is_inf() const {
        return ((bits & 0x7FFF) == 0x7F80);
    }

    bool is_zero() const {
        return (bits & 0x7FFF) == 0;
    }

    bool is_negative() const {
        return (bits & 0x8000) != 0;
    }

    // Arithmetic operators
    BFloat16 operator+(BFloat16 other) const {
        return BFloat16::from_float(to_float() + other.to_float());
    }

    BFloat16 operator-(BFloat16 other) const {
        return BFloat16::from_float(to_float() - other.to_float());
    }

    BFloat16 operator*(BFloat16 other) const {
        return BFloat16::from_float(to_float() * other.to_float());
    }

    BFloat16 operator/(BFloat16 other) const {
        return BFloat16::from_float(to_float() / other.to_float());
    }

    BFloat16 operator-() const {
        BFloat16 result;
        result.bits = bits ^ 0x8000;  // Flip sign bit
        return result;
    }

    // Comparison
    bool operator==(BFloat16 other) const {
        if (is_nan() || other.is_nan()) return false;
        if (is_zero() && other.is_zero()) return true;
        return bits == other.bits;
    }

    bool operator!=(BFloat16 other) const {
        return !(*this == other);
    }

    bool operator<(BFloat16 other) const {
        if (is_nan() || other.is_nan()) return false;
        if (is_zero() && other.is_zero()) return false;
        return to_float() < other.to_float();
    }

    bool operator>(BFloat16 other) const {
        return other < *this;
    }

    bool operator<=(BFloat16 other) const {
        return !(other < *this);
    }

    bool operator>=(BFloat16 other) const {
        return !(*this < other);
    }
};

// ============================================================================
// Bulk Conversion Functions
// ============================================================================

/// Convert array of float32 to Float16
void float_to_fp16(const float* input, Float16* output, size_t count);

/// Convert array of Float16 to float32
void fp16_to_float(const Float16* input, float* output, size_t count);

/// Convert array of float32 to BFloat16
void float_to_bf16(const float* input, BFloat16* output, size_t count);

/// Convert array of BFloat16 to float32
void bf16_to_float(const BFloat16* input, float* output, size_t count);

// ============================================================================
// Constants
// ============================================================================

namespace fp16_limits {
    constexpr float max_value = 65504.0f;
    constexpr float min_positive = 6.103515625e-05f;  // 2^-14
    constexpr float epsilon = 0.0009765625f;  // 2^-10
}

namespace bf16_limits {
    // BFloat16 has same range as float32
    constexpr float max_value = 3.38953139e+38f;
    constexpr float min_positive = 1.17549435e-38f;
    constexpr float epsilon = 0.0078125f;  // 2^-7
}

} // namespace quantization
} // namespace pyflame_rt
