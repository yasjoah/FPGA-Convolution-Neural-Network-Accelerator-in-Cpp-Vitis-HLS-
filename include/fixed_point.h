#pragma once

#include "config.h"
#include <cstdint>

using fixed16_t = int16_t;
using accum32_t = int32_t;

constexpr int FIXED_FRAC_BITS = FRAC_BITS;
constexpr int FIXED_SCALE = 1 << FIXED_FRAC_BITS;

fixed16_t float_to_fixed(float x);
float fixed_to_float(fixed16_t x);
accum32_t mac(accum32_t acc, fixed16_t a, fixed16_t b);
fixed16_t accum_to_fixed(accum32_t acc);
