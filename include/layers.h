#pragma once

#include "types.h"
#include "fixed_point.h"

// Float reference path
void conv_naive(const InputTensor& input, const WeightTensor& weights, ConvTensor& output);
void conv_engine(const InputTensor& in, const WeightTensor& w, ConvTensor& out);
void relu(const ConvTensor& input, ReluTensor& output);
void pooling(const ReluTensor& input, OutputTensor& output);
void cnn_top_float(const InputTensor& input, const WeightTensor& weights, OutputTensor& output);

void load_input_tile(const InputTensor& src, InputTensor& dst);
void load_weight_tile(const WeightTensor& src, WeightTensor& dst);
void relu_unit(const ConvTensor& in, ReluTensor& out);
void pooling_unit(const ReluTensor& in, OutputTensor& out);

// Fixed-point paths
void conv_fixed(const InputTensorFixed& input, const WeightTensorFixed& weights, ConvTensorFixed& output);
void conv_engine_fixed(const InputTensorFixed& in, const WeightTensorFixed& w, ConvTensorFixed& out);
void relu_fixed(const ConvTensorFixed& input, ReluTensorFixed& output);
void relu_unit_fixed(const ConvTensorFixed& in, ReluTensorFixed& out);
void pooling_fixed(const ReluTensorFixed& input, OutputTensorFixed& output);
void pooling_unit_fixed(const ReluTensorFixed& in, OutputTensorFixed& out);
void cnn_top_fixed(const InputTensorFixed& input, const WeightTensorFixed& weights, OutputTensorFixed& output);

// HLS baseline only (sequential, no tiling/unroll)
void conv_fixed_baseline(const InputTensorFixed& input, const WeightTensorFixed& weights, ConvTensorFixed& output);
void cnn_top_fixed_baseline(const InputTensorFixed& input, const WeightTensorFixed& weights, OutputTensorFixed& output);

void run_benchmarks();
