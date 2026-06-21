#include "../include/config.h"
#include "../include/layers.h"
#include "../include/types.h" 
// HARDWARE: Each (oh, ow, oc) iteration = one output pixel from one filter.
//           Inner (ky, kx, ic) loops = MAC ops into a single accumulator register.
//           In FPGA: inner loop → DSP48 MAC; outer loops → address generators.
void conv_naive(const InputTensor& in, const WeightTensor& w, ConvTensor& out) {
    for (int oc = 0; oc < OUT_C; ++oc) {
        for (int oh = 0; oh < CONV_H; ++oh) {
            for (int ow = 0; ow < CONV_W; ++ow) {
                float acc = 0.0f;
                for (int ky = 0; ky < K_SIZE; ++ky) {
                    for (int kx = 0; kx < K_SIZE; ++kx) {
                        for (int ic = 0; ic < IN_C; ++ic) {

                            acc += in[ow + kx][oh + ky][ic] * w[oc][ky][kx][ic];

                        }
                    }
                }
                out[ow][oh][oc] = acc;  // Assign the value of the MAC to the corresponding output tensor.
            }
        }
    }
}

void conv_engine(const InputTensor& in,
                 const WeightTensor& w,
                 ConvTensor& out) {
    conv_naive(in, w, out);
}