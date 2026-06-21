#include "../include/layers.h"
#include "../include/fixed_point.h"

// Sequential reference for HLS baseline comparison only.
void conv_fixed_baseline(const InputTensorFixed& input,
                         const WeightTensorFixed& weights,
                         ConvTensorFixed& output) {
#pragma HLS INLINE off
#pragma HLS PIPELINE off

    for (int oc = 0; oc < OUT_C; ++oc) {
        for (int oy = 0; oy < CONV_H; ++oy) {
            for (int ox = 0; ox < CONV_W; ++ox) {
                accum32_t acc = 0;
                for (int ky = 0; ky < K_SIZE; ++ky) {
                    for (int kx = 0; kx < K_SIZE; ++kx) {
                        for (int ic = 0; ic < IN_C; ++ic) {
                            acc = mac(acc,
                                      input[ox + kx][oy + ky][ic],
                                      weights[oc][ky][kx][ic]);
                        }
                    }
                }
                output[ox][oy][oc] = accum_to_fixed(acc);
            }
        }
    }
}

void cnn_top_fixed_baseline(const InputTensorFixed& input,
                            const WeightTensorFixed& weights,
                            OutputTensorFixed& output) {
#pragma HLS INLINE off
#pragma HLS PIPELINE off

    ConvTensorFixed conv_out {};
    ReluTensorFixed relu_out {};

    conv_fixed_baseline(input, weights, conv_out);
    relu_fixed(conv_out, relu_out);
    pooling_fixed(relu_out, output);
}
