#include "../include/config.h"
#include "../include/types.h"
#include "../include/layers.h"

void pooling_fixed(const ReluTensorFixed& input, OutputTensorFixed& output) {
    for (int ox = 0; ox < OUT_W; ++ox) {
        for (int oy = 0; oy < OUT_H; ++oy) {
            for (int oc = 0; oc < OUT_C; ++oc) {
                HLS_PIPELINE
                fixed16_t max_val = 0;
                for (int ky = 0; ky < POOL_SIZE; ++ky) {
                    HLS_UNROLL
                    for (int kx = 0; kx < POOL_SIZE; ++kx) {
                        HLS_UNROLL
                        const fixed16_t v = input[ox * POOL_STRIDE + kx][oy * POOL_STRIDE + ky][oc];
                        if (v > max_val) max_val = v;
                    }
                }
                output[ox][oy][oc] = max_val;
            }
        }
    }
}

void pooling_unit_fixed(const ReluTensorFixed& in, OutputTensorFixed& out) {
    HLS_INLINE_OFF
    pooling_fixed(in, out);
}
