#include "../include/config.h"
#include "../include/types.h"
#include "../include/layers.h"

void relu_fixed(const ConvTensorFixed& input, ReluTensorFixed& output) {
    for (int x = 0; x < CONV_W; ++x) {
        for (int y = 0; y < CONV_H; ++y) {
            for (int oc = 0; oc < OUT_C; ++oc) {
                HLS_PIPELINE
                const fixed16_t v = input[x][y][oc];
                output[x][y][oc] = (v > 0) ? v : static_cast<fixed16_t>(0);
            }
        }
    }
}

void relu_unit_fixed(const ConvTensorFixed& in, ReluTensorFixed& out) {
    HLS_INLINE_OFF
    relu_fixed(in, out);
}
