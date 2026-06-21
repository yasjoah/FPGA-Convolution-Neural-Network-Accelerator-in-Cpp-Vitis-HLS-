#pragma once
#include "config.h"
#include <cstdint>

using InputTensor  = float[IN_W][IN_H][IN_C];  // Input tensor.
using WeightTensor = float[OUT_C][K_SIZE][K_SIZE][IN_C];  // Filter tensor with dimensions OC x K x K x IC.
using ConvTensor   = float[CONV_W][CONV_H][OUT_C];  // Resulting tensor after the convolution.
using ReluTensor   = float[CONV_W][CONV_H][OUT_C];  // Tensor result after applying Relu algorithm x = max(0, x).
using OutputTensor = float[OUT_W][OUT_H][OUT_C];  //Resulting tensor after pooling.


// Same thing just with fixed-point numbers now.


using InputTensorFixed  = int16_t[IN_W][IN_H][IN_C];
using WeightTensorFixed = int16_t[OUT_C][K_SIZE][K_SIZE][IN_C];
using ConvTensorFixed   = int16_t[CONV_W][CONV_H][OUT_C];
using ReluTensorFixed   = int16_t[CONV_W][CONV_H][OUT_C];
using OutputTensorFixed = int16_t[OUT_W][OUT_H][OUT_C];
