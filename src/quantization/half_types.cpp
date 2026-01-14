#include "pyflame_rt/quantization/half_types.hpp"
#include <cstring>
#include <cmath>

namespace pyflame_rt {
namespace quantization {

// ============================================================================
// Float16 Implementation
// ============================================================================

Float16 Float16::from_float(float value) {
    uint32_t f32;
    std::memcpy(&f32, &value, sizeof(f32));

    // Extract components
    uint32_t sign = (f32 >> 31) & 0x1;
    int32_t exp = static_cast<int32_t>((f32 >> 23) & 0xFF) - 127;
    uint32_t mant = f32 & 0x7FFFFF;

    uint16_t result;

    // Handle special cases
    if (exp == 128) {
        // Infinity or NaN
        if (mant == 0) {
            // Infinity
            result = static_cast<uint16_t>((sign << 15) | 0x7C00);
        } else {
            // NaN - preserve some mantissa bits
            result = static_cast<uint16_t>((sign << 15) | 0x7C00 | (mant >> 13));
            if ((result & 0x03FF) == 0) {
                result |= 1;  // Ensure it's still NaN
            }
        }
    } else if (exp > 15) {
        // Overflow to infinity
        result = static_cast<uint16_t>((sign << 15) | 0x7C00);
    } else if (exp < -24) {
        // Underflow to zero
        result = static_cast<uint16_t>(sign << 15);
    } else if (exp < -14) {
        // Denormalized number
        mant |= 0x800000;  // Add implicit 1
        int32_t shift = -14 - exp;

        // Security fix CRIT-Q01: Validate shift is in safe range [1, 9]
        // exp is in range [-23, -15], so shift is in range [1, 9]
        // But add explicit bounds check for safety
        if (shift < 1 || shift > 23) {
            // Out of valid range, flush to zero
            result = static_cast<uint16_t>(sign << 15);
        } else {
            // Round to nearest even
            // shift + 12 is in range [13, 21], safe for uint32_t shifts
            uint32_t round_bit = 1U << (shift + 12);
            uint32_t sticky_bits = round_bit - 1;

            if ((mant & round_bit) && ((mant & sticky_bits) || (mant & (round_bit << 1)))) {
                mant += round_bit;
            }

            mant >>= (shift + 13);
            result = static_cast<uint16_t>((sign << 15) | mant);
        }
    } else {
        // Normalized number
        int32_t new_exp = exp + 15;

        // Round to nearest even
        uint32_t round_bit = 1U << 12;
        uint32_t sticky_bits = (1U << 12) - 1;

        if ((mant & round_bit) && ((mant & sticky_bits) || (mant & (round_bit << 1)))) {
            mant += round_bit;
            if (mant >= 0x800000) {
                mant = 0;
                new_exp++;
            }
        }

        if (new_exp >= 31) {
            // Overflow after rounding
            result = static_cast<uint16_t>((sign << 15) | 0x7C00);
        } else {
            mant >>= 13;
            result = static_cast<uint16_t>((sign << 15) | (new_exp << 10) | (mant & 0x3FF));
        }
    }

    return Float16(result);
}

float Float16::to_float() const {
    uint32_t sign = (bits >> 15) & 0x1;
    uint32_t exp = (bits >> 10) & 0x1F;
    uint32_t mant = bits & 0x3FF;

    uint32_t f32;

    if (exp == 0) {
        if (mant == 0) {
            // Zero
            f32 = sign << 31;
        } else {
            // Denormalized - convert to normalized float32
            exp = 1;
            while ((mant & 0x400) == 0) {
                mant <<= 1;
                exp--;
            }
            mant &= 0x3FF;
            f32 = (sign << 31) | ((exp + 127 - 15) << 23) | (mant << 13);
        }
    } else if (exp == 31) {
        // Infinity or NaN
        f32 = (sign << 31) | 0x7F800000 | (mant << 13);
    } else {
        // Normalized number
        f32 = (sign << 31) | ((exp + 127 - 15) << 23) | (mant << 13);
    }

    float result;
    std::memcpy(&result, &f32, sizeof(result));
    return result;
}

// ============================================================================
// BFloat16 Implementation
// ============================================================================

BFloat16 BFloat16::from_float(float value) {
    uint32_t f32;
    std::memcpy(&f32, &value, sizeof(f32));

    // Check for NaN - preserve NaN
    if ((f32 & 0x7F800000) == 0x7F800000 && (f32 & 0x007FFFFF) != 0) {
        // NaN - truncate mantissa but ensure it stays NaN
        uint16_t result = static_cast<uint16_t>(f32 >> 16);
        if ((result & 0x007F) == 0) {
            result |= 0x0040;  // Set a mantissa bit to keep it NaN
        }
        return BFloat16(result);
    }

    // Security fix HIGH-Q03: Check for infinity before rounding to prevent overflow
    // If already infinity, just truncate without rounding
    if ((f32 & 0x7F800000) == 0x7F800000) {
        // Infinity - return directly (mantissa is 0)
        return BFloat16(static_cast<uint16_t>(f32 >> 16));
    }

    // Security fix HIGH-Q03: Check for values that would overflow with rounding
    // Max finite bfloat16 is 0x7F7F0000 before rounding
    // If adding rounding bias would overflow the exponent, clamp to infinity
    uint32_t rounding_bias = 0x7FFF + ((f32 >> 16) & 1);

    // Check if rounding would cause overflow
    uint32_t upper = f32 >> 16;
    if (upper >= 0x7F7F) {
        // Close to infinity - check if rounding causes overflow
        uint32_t lower = f32 & 0xFFFF;
        if (lower + rounding_bias > 0xFFFF) {
            // Would overflow - clamp to infinity
            uint32_t sign = f32 & 0x80000000;
            return BFloat16(static_cast<uint16_t>((sign >> 16) | 0x7F80));
        }
    }

    // Safe to round
    f32 += rounding_bias;

    return BFloat16(static_cast<uint16_t>(f32 >> 16));
}

float BFloat16::to_float() const {
    uint32_t f32 = static_cast<uint32_t>(bits) << 16;
    float result;
    std::memcpy(&result, &f32, sizeof(result));
    return result;
}

// ============================================================================
// Bulk Conversion Functions
// ============================================================================

void float_to_fp16(const float* input, Float16* output, size_t count) {
    for (size_t i = 0; i < count; ++i) {
        output[i] = Float16::from_float(input[i]);
    }
}

void fp16_to_float(const Float16* input, float* output, size_t count) {
    for (size_t i = 0; i < count; ++i) {
        output[i] = input[i].to_float();
    }
}

void float_to_bf16(const float* input, BFloat16* output, size_t count) {
    for (size_t i = 0; i < count; ++i) {
        output[i] = BFloat16::from_float(input[i]);
    }
}

void bf16_to_float(const BFloat16* input, float* output, size_t count) {
    for (size_t i = 0; i < count; ++i) {
        output[i] = input[i].to_float();
    }
}

} // namespace quantization
} // namespace pyflame_rt
