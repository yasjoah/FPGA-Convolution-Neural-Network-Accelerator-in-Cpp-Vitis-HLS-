#include "../include/config.h"
#include "../include/types.h"
#include "../include/layers.h"
#include "../include/fixed_point.h"

// ---------------------------------------------------------------------------
// Untiled reference — used by CPU tests to verify tiled conv_engine_fixed
// ---------------------------------------------------------------------------
static void conv_fixed_untiled(const InputTensorFixed& in,
                               const WeightTensorFixed& w,
                               ConvTensorFixed& out) {
    for (int oc = 0; oc < OUT_C; ++oc) {
        for (int oy = 0; oy < CONV_H; ++oy) {
            for (int ox = 0; ox < CONV_W; ++ox) {
                accum32_t acc = 0;
                for (int ky = 0; ky < K_SIZE; ++ky) {
                    for (int kx = 0; kx < K_SIZE; ++kx) {
                        for (int ic = 0; ic < IN_C; ++ic) {
                            acc = mac(acc, in[ox + kx][oy + ky][ic], w[oc][ky][kx][ic]);
                        }
                    }
                }
                out[ox][oy][oc] = accum_to_fixed(acc);
            }
        }
    }
}

void conv_fixed(const InputTensorFixed& in,
                const WeightTensorFixed& w,
                ConvTensorFixed& out) {
    conv_fixed_untiled(in, w, out);
}

// ---------------------------------------------------------------------------
// Stage 4–7: tiled conv blocks (models on-chip BRAM + MAC array)
// ---------------------------------------------------------------------------

static void load_input_tile(const InputTensorFixed& input,
                            fixed16_t input_tile[TILE_W][TILE_H][IN_C],
                            int tile_x,
                            int tile_y) {
    HLS_INLINE_OFF

    for (int ly = 0; ly < TILE_H; ++ly) {
        for (int lx = 0; lx < TILE_W; ++lx) {
            for (int ic = 0; ic < IN_C; ++ic) {
                HLS_PIPELINE
                const int gx = tile_x + lx;
                const int gy = tile_y + ly;
                if (gx < IN_W && gy < IN_H) {
                    input_tile[lx][ly][ic] = input[gx][gy][ic];
                } else {
                    input_tile[lx][ly][ic] = 0;
                }
            }
        }
    }
}

static void conv_engine_tile(
    const fixed16_t input_tile[TILE_W][TILE_H][IN_C],
    const WeightTensorFixed& weights,
    fixed16_t output_tile[TILE_W][TILE_H][OUT_C],
    int valid_w,
    int valid_h) {
    HLS_INLINE_OFF

#pragma HLS ARRAY_PARTITION variable=input_tile complete dim=1
#pragma HLS ARRAY_PARTITION variable=input_tile complete dim=2

    for (int oc = 0; oc < OUT_C; ++oc) {
        for (int oy = 0; oy < valid_h; ++oy) {
            for (int ox = 0; ox < valid_w; ++ox) {
                HLS_PIPELINE

                accum32_t partial[K_SIZE][K_SIZE];
#pragma HLS ARRAY_PARTITION variable=partial complete dim=0

                for (int ky = 0; ky < K_SIZE; ++ky) {
                    HLS_UNROLL
                    for (int kx = 0; kx < K_SIZE; ++kx) {
                        HLS_UNROLL
                        partial[ky][kx] =
                            static_cast<accum32_t>(input_tile[ox + kx][oy + ky][0]) *
                            static_cast<accum32_t>(weights[oc][ky][kx][0]);
                    }
                }

                accum32_t acc =
                    partial[0][0] + partial[0][1] + partial[0][2] +
                    partial[1][0] + partial[1][1] + partial[1][2] +
                    partial[2][0] + partial[2][1] + partial[2][2];

                output_tile[ox][oy][oc] = accum_to_fixed(acc);
            }
        }
    }
}

static void store_output_tile(ConvTensorFixed& output,
                              const fixed16_t output_tile[TILE_W][TILE_H][OUT_C],
                              int tile_x,
                              int tile_y,
                              int valid_w,
                              int valid_h) {
    HLS_INLINE_OFF

    for (int oy = 0; oy < valid_h; ++oy) {
        for (int ox = 0; ox < valid_w; ++ox) {
            for (int oc = 0; oc < OUT_C; ++oc) {
                HLS_PIPELINE
                output[tile_x + ox][tile_y + oy][oc] = output_tile[ox][oy][oc];
            }
        }
    }
}

void conv_engine_fixed(const InputTensorFixed& input,
                       const WeightTensorFixed& weights,
                       ConvTensorFixed& output) {
    HLS_INLINE_OFF

    for (int tile_y = 0; tile_y < CONV_H; tile_y += TILE_H) {
        for (int tile_x = 0; tile_x < CONV_W; tile_x += TILE_W) {

            fixed16_t input_tile[TILE_W][TILE_H][IN_C];
            fixed16_t output_tile[TILE_W][TILE_H][OUT_C];

#pragma HLS ARRAY_PARTITION variable=input_tile complete dim=1
#pragma HLS ARRAY_PARTITION variable=output_tile complete dim=3

            int valid_w = TILE_W;
            int valid_h = TILE_H;
            if (tile_x + valid_w > CONV_W) valid_w = CONV_W - tile_x;
            if (tile_y + valid_h > CONV_H) valid_h = CONV_H - tile_y;

            HLS_DATAFLOW
            load_input_tile(input, input_tile, tile_x, tile_y);
            conv_engine_tile(input_tile, weights, output_tile, valid_w, valid_h);
            store_output_tile(output, output_tile, tile_x, tile_y, valid_w, valid_h);
        }
    }
}
