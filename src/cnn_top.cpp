#include "../include/config.h"
#include "../include/types.h"
#include "../include/layers.h"

// ---------------------------------------------------------------------------
// Stage 3 float blocks
// ---------------------------------------------------------------------------

void load_input_tile(const InputTensor& src, InputTensor& dst) {
    for (int x = 0; x < IN_W; ++x) {
        for (int y = 0; y < IN_H; ++y) {
            for (int c = 0; c < IN_C; ++c) {
                dst[x][y][c] = src[x][y][c];
            }
        }
    }
}

void load_weight_tile(const WeightTensor& src, WeightTensor& dst) {
    for (int oc = 0; oc < OUT_C; ++oc) {
        for (int ky = 0; ky < K_SIZE; ++ky) {
            for (int kx = 0; kx < K_SIZE; ++kx) {
                for (int ic = 0; ic < IN_C; ++ic) {
                    dst[oc][ky][kx][ic] = src[oc][ky][kx][ic];
                }
            }
        }
    }
}

void cnn_top_float(const InputTensor& input,
                   const WeightTensor& weights,
                   OutputTensor& output) {
    InputTensor  in_buf;
    WeightTensor w_buf;
    ConvTensor   conv_buf;
    ReluTensor   relu_buf;

    load_input_tile(input, in_buf);
    load_weight_tile(weights, w_buf);
    conv_engine(in_buf, w_buf, conv_buf);
    relu_unit(conv_buf, relu_buf);
    pooling_unit(relu_buf, output);
}

// ---------------------------------------------------------------------------
// Stage 7: fixed accelerator top (DATAFLOW between stages)
// ---------------------------------------------------------------------------

void cnn_top_fixed(const InputTensorFixed& input,
                   const WeightTensorFixed& weights,
                   OutputTensorFixed& output) {
    HLS_INLINE_OFF

    ConvTensorFixed conv_out;
    ReluTensorFixed relu_out;

#pragma HLS ARRAY_PARTITION variable=conv_out cyclic factor=2 dim=1
#pragma HLS ARRAY_PARTITION variable=relu_out cyclic factor=2 dim=1

    HLS_DATAFLOW
    conv_engine_fixed(input, weights, conv_out);
    relu_unit_fixed(conv_out, relu_out);
    pooling_unit_fixed(relu_out, output);
}
