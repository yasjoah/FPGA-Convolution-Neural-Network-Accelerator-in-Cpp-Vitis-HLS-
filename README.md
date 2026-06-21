# CNN FPGA Accelerator

A small C++/HLS CNN inference pipeline built to demonstrate how convolutional neural networks map to FPGA-style hardware: fixed-point arithmetic, tiled on-chip buffers, pipelined MAC arrays, and dataflow between stages.

This is **not** a general ML framework. It is an educational accelerator codebase that runs on CPU for verification and synthesizes to RTL with AMD Vitis HLS.

## Pipeline

```
Input (8×8×1)
  → 3×3 convolution (4 filters)
  → ReLU
  → 2×2 max pooling (stride 2)
  → Output (3×3×4)
```

| Stage        | Tensor shape |
|--------------|--------------|
| Input        | 8×8×1        |
| After conv   | 6×6×4        |
| After ReLU   | 6×6×4        |
| After pool   | 3×3×4        |

**MACs per inference:** 1,296 (6×6×4×3×3×1)

---

## Why FPGA/ASIC-oriented?

| Software ML library | This project |
|---------------------|--------------|
| Float tensors, dynamic shapes | Fixed `constexpr` dimensions in `config.h` |
| Heap allocation (`vector`, etc.) | Fixed-size C arrays on stack / BRAM |
| One CPU thread, sequential | Pipelined MAC arrays, DATAFLOW stages |
| Train + infer | Inference-only, hand-verifiable test vectors |

The code is written so each function corresponds to a **hardware block** you could instantiate on an FPGA: load tile → conv engine → activation → pool → store.

---

## Architecture

```
[DRAM] --DMA--> [Input BRAM / tile buffer]
                      |
                      v
               [Weight BRAM] --> [MAC array + accumulator]
                      |
                      v
                  [ReLU unit]
                      |
                      v
               [Max-pool comparators]
                      |
                      v
              [Output BRAM] --> [DRAM]
```

### Hardware blocks

| Function | File | Hardware meaning |
|----------|------|------------------|
| `load_input_tile()` | `cnn_top.cpp` / `conv_fixed.cpp` | DMA burst read into on-chip BRAM |
| `load_weight_tile()` | `cnn_top.cpp` | Weight buffer fill |
| `conv_engine()` / `conv_engine_fixed()` | `conv_naive.cpp` / `conv_fixed.cpp` | MAC array + accumulator |
| `relu_unit()` / `relu_unit_fixed()` | `relu.cpp` / `relu_fixed.cpp` | Combinational clip (no multipliers) |
| `pooling_unit()` / `pooling_unit_fixed()` | `pooling.cpp` / `pooling_fixed.cpp` | 2×2 comparator tree |
| `cnn_top_float()` / `cnn_top_fixed()` | `cnn_top.cpp` | Top-level wiring / DATAFLOW region |

---

## Two designs: baseline vs accelerator

### Baseline (`cnn_top_fixed_baseline`)

- Plain nested loops, sequential conv → ReLU → pool
- Reference for correctness and HLS comparison
- **`#pragma HLS PIPELINE off`** on conv

### Accelerator (`cnn_top_fixed`)

- **Tiled convolution** with on-chip tile buffers (models BRAM)
- **Full 3×3 MAC unroll** — nine parallel multipliers + adder tree
- **`#pragma HLS DATAFLOW`** inside conv (load → compute → store) and at top (conv → ReLU → pool)
- **Q8.8 fixed-point** (`int16_t` activations/weights, `int32_t` accumulators)

### Vitis HLS results (xczu3eg, 10 ns clock)

| Design | Top latency | Conv latency | Pipeline type |
|--------|---------------|--------------|---------------|
| Baseline | **1062 cycles** (~10.6 µs) | 725 cycles | Sequential |
| Accelerator | **716 cycles** (~7.2 µs) | 530 cycles | DATAFLOW |

The accelerator is ~**32% faster** end-to-end. Most of the gain is in convolution (725 → 530 cycles) from parallel MACs, pipelining, and overlapped load/compute/store.

---

## Fixed-point quantization

Format: **Q8.8** in `int16_t` (8 fractional bits, scale = 256).

```cpp
fixed16_t x = float_to_fixed(1.25f);  // raw value 320
float y = fixed_to_float(x);          // back to 1.25
```

MAC operations widen to `int32_t` to avoid overflow, then `accum_to_fixed()` shifts back to Q8.8:

```cpp
acc = mac(acc, a, b);           // int32 accumulate
out = accum_to_fixed(acc);      // shift >> 8, clamp to int16
```

The float path (`cnn_top_float`) is the golden reference; the fixed path is compared with a tolerance (~0.1) in tests.

---

## Tiling and data reuse

A 3×3 kernel needs a **halo** of neighboring pixels around each tile. Tile buffers in `conv_fixed.cpp` stand in for on-chip BRAM:

```cpp
fixed16_t input_tile[TILE_IN_W][TILE_IN_H][IN_C];   // tile + halo
fixed16_t output_tile[TILE_W][TILE_H][OUT_C];
```

For this tiny network, `TILE_W` and `TILE_H` equal the full conv map (6×6), so one tile covers everything. On larger networks, smaller tiles reuse overlapping input rows/columns across iterations, reducing external memory bandwidth.

Configure in `include/config.h`:

```cpp
constexpr int TILE_W = CONV_W;
constexpr int TILE_H = CONV_H;
```

---

## Pipelining, unrolling, and dataflow

| Pragma | Where | Effect |
|--------|-------|--------|
| `HLS PIPELINE II=1` | Inner conv / ReLU / pool loops | Start one iteration per clock |
| `HLS UNROLL` | 3×3 MAC loop in `conv_engine_tile` | Nine parallel DSP multipliers |
| `HLS ARRAY_PARTITION` | Tile buffers | Multiple BRAM read ports |
| `HLS DATAFLOW` | `cnn_top_fixed`, per-tile conv | Stages run concurrently on different data |

Macros in `config.h` expand to real pragmas under Vitis HLS (`__SYNTHESIS__`) and to no-ops for normal `g++` builds.

---

## FPGA vs ASIC mapping

| Software concept | FPGA (Xilinx) | ASIC |
|------------------|---------------|------|
| MAC loop | DSP48 slice | Hard multiplier macro |
| Tile buffer | BRAM / URAM | SRAM bank |
| `ARRAY_PARTITION` | Multiple BRAM ports | Multi-port register file |
| `DATAFLOW` | FIFO / ping-pong BRAM | Queue between IP blocks |
| ReLU | LUT comparators | Combinational clip |
| Max pool | Comparator tree | Comparator tree |

---

## Repository structure

```
cnn-fpga-accelerator/
├── include/
│   ├── config.h          # Dimensions, tile size, HLS macros
│   ├── types.h           # Fixed-size tensor typedefs
│   ├── fixed_point.h     # Q8.8 conversion + MAC helpers
│   └── layers.h          # Function declarations
├── src/
│   ├── conv_naive.cpp    # Float reference conv
│   ├── conv_fixed.cpp    # Tiled fixed-point conv (accelerator)
│   ├── conv_baseline.cpp # Sequential conv (HLS baseline only)
│   ├── relu.cpp / relu_fixed.cpp
│   ├── pooling.cpp / pooling_fixed.cpp
│   ├── cnn_top.cpp       # Float + fixed tops
│   ├── fixed_point.cpp
│   └── benchmark.cpp
├── testbench/
│   └── main.cpp          # PASS/FAIL tests + benchmarks
├── script.tcl            # HLS: synthesize accelerator
├── script_baseline.tcl   # HLS: synthesize baseline
└── CMakeLists.txt
```

---

## Build and test (CPU)

Requirements: C++17 compiler, CMake 3.16+

```bash
mkdir build && cd build
cmake ..
cmake --build .
./cnn_test        # Linux/macOS
cnn_test.exe      # Windows
```

Tests cover convolution, ReLU, pooling, fixed-point conversion, float vs fixed top-level, and tiled vs untiled conv equality. On success, a benchmark table prints automatically.

---

## Vitis HLS synthesis

Requirements: AMD Vitis HLS 2024.x / 2025.x

```bash
# Accelerator (cnn_top_fixed)
vitis_hls -f script.tcl

# Baseline comparison (cnn_top_fixed_baseline)
vitis_hls -f script_baseline.tcl
```

Reports:

- `cnn_hls_project/solution1/syn/report/cnn_top_fixed_csynth.rpt`
- `cnn_hls_baseline/solution1/syn/report/cnn_top_fixed_baseline_csynth.rpt`

Target device: **xczu3eg-sbva484-1-e** (Zynq UltraScale+), 100 MHz default clock.

---

## Future work

- [ ] AXI4 master/slave interfaces for DRAM access
- [ ] Line buffers for streaming conv (one row at a time)
- [ ] Systolic array conv engine
- [ ] Cosimulation (C/RTL) with Vitis HLS
- [ ] Larger input sizes with multi-tile scheduling
- [ ] Resource/latency tradeoff sweep over `TILE_W`, `UNROLL_FACTOR`

---

## License

Educational / portfolio project. Add a license file if you publish publicly.
