#include "fixed_point.h"

#include <algorithm>
#include <cmath>
#include <limits>

fixed16_t float_to_fixed(float x) {
    int32_t scaled = static_cast<int32_t>(std::round(x * FIXED_SCALE));  // Round to the nearest integer as a float then cast it to a 32 bit integer. 32 bits because multiplication can result in a number greater than 16 bits.

    scaled = std::clamp(
        scaled,
        static_cast<int32_t>(std::numeric_limits<fixed16_t>::min()),  // Clamping: if the scaled 32 bit integer is less than -32768, then make scaled = -32768.
        static_cast<int32_t>(std::numeric_limits<fixed16_t>::max())   // Clamping: if the sclaed 32 bit integer is greater than 32767, then scaled = 32767.
    );

    return static_cast<fixed16_t>(scaled);  // Finally, return it as a 16 bit integer (since it is now safely in the [-32768, 32767] range).
}

float fixed_to_float(fixed16_t x) {
    return static_cast<float>(x) / FIXED_SCALE;  // Divide your 16 bit integer by the scale (in this case, 2^8 because we are using Q8.8) to get the original float number.
}

accum32_t mac(accum32_t acc, fixed16_t a, fixed16_t b) {
    return acc + static_cast<accum32_t>(a) * static_cast<accum32_t>(b);
}

fixed16_t accum_to_fixed(accum32_t acc) {
    // acc currently has scale 256 * 256 because it came from a*b.
    // Shift right by 8 to return to Q8.8 scale.
    accum32_t shifted = acc >> FIXED_FRAC_BITS;

    shifted = std::clamp(
        shifted,
        static_cast<accum32_t>(std::numeric_limits<fixed16_t>::min()),  // Again we clamp.
        static_cast<accum32_t>(std::numeric_limits<fixed16_t>::max())
    );

    return static_cast<fixed16_t>(shifted);
}