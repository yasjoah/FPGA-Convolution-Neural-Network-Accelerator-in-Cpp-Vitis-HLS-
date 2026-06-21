#pragma once

// Network geometry
constexpr int IN_H  = 8;
constexpr int IN_W  = 8;
constexpr int IN_C  = 1;
constexpr int OUT_C = 4;

constexpr int K_SIZE      = 3;
constexpr int CONV_STRIDE = 1;
constexpr int CONV_PAD    = 0;

constexpr int CONV_H = IN_H - K_SIZE + 1;   // 6
constexpr int CONV_W = IN_W - K_SIZE + 1;   // 6

constexpr int POOL_SIZE   = 2;
constexpr int POOL_STRIDE = 2;
constexpr int OUT_H = CONV_H / POOL_STRIDE; // 3
constexpr int OUT_W = CONV_W / POOL_STRIDE; // 3

// One tile covers the full conv map for this tiny network (best HLS latency).
// For larger networks, use smaller TILE_W/TILE_H and more tile iterations.
constexpr int TILE_W = CONV_W;
constexpr int TILE_H = CONV_H;

constexpr int TILE_IN_W = TILE_W + K_SIZE - 1;
constexpr int TILE_IN_H = TILE_H + K_SIZE - 1;

constexpr int UNROLL_FACTOR = K_SIZE * K_SIZE * IN_C;  // full 3x3 MAC unroll
constexpr int FRAC_BITS     = 8;

// HLS pragma macros — active under Vitis HLS (__SYNTHESIS__), comments for g++
#if defined(__SYNTHESIS__)
  #define HLS_INLINE_OFF  _Pragma("HLS INLINE off")
  #define HLS_PIPELINE    _Pragma("HLS PIPELINE II=1")
  #define HLS_UNROLL      _Pragma("HLS UNROLL")
  #define HLS_DATAFLOW    _Pragma("HLS DATAFLOW")
  #define HLS_ARRAY_PART(var, dim) _Pragma(HLS ARRAY_PARTITION variable=var cyclic factor=2 dim=dim)
#else
  #define HLS_INLINE_OFF
  #define HLS_PIPELINE
  #define HLS_UNROLL
  #define HLS_DATAFLOW
  #define HLS_ARRAY_PART(var, dim)
#endif
