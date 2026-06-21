#include "../include/config.h"
#include "../include/types.h"
#include "../include/layers.h"

void relu(const ConvTensor& input, ReluTensor& output) {
    for (int x = 0; x < CONV_W; ++x) {
        for (int y = 0; y < CONV_H; ++y) {
            for (int oc = 0; oc < OUT_C; ++oc) {
                const float v = input[x][y][oc];
                output[x][y][oc] = (v > 0.0f) ? v : 0.0f;
            }
        }
    }
}

void relu_unit(const ConvTensor& in, ReluTensor& out) {
    relu(in, out);
}
