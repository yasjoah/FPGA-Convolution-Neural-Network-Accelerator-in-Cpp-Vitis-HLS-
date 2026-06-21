#include "../include/config.h"
#include "../include/types.h" 
// HARDWARE: 3 compare operations form a comparator tree (4 inputs → 1 max).


void pooling(const ReluTensor& out, OutputTensor& out1 ){
    for (int oc = 0; oc < OUT_C; ++oc) {
        for (int oh = 0; oh < OUT_H; ++oh){
            for (int ow = 0; ow < OUT_W; ow++){
                float max_val {};
                for (int ky = 0; ky < POOL_SIZE; ++ky){
                    for (int kx = 0; kx < POOL_SIZE; ++kx){
                        int actual_xpos = kx + ow*POOL_STRIDE;  // Here I take the stride into consideration
                        int actual_ypos = ky + oh*POOL_STRIDE;
                        max_val = (out[actual_xpos][actual_ypos][oc] > max_val) ? out[actual_xpos][actual_ypos][oc] : max_val; // Calculate the max value out of the filter.
                    }
                
                }
                out1[ow][oh][oc] = max_val;
            }

        }
    }  
}

void pooling_unit(const ReluTensor& in,
                  OutputTensor& out) {
    pooling(in, out);
}