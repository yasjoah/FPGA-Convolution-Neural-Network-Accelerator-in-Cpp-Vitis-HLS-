#include "../include/layers.h"
#include "../include/fixed_point.h"

#include <chrono>
#include <iostream>
#include <iomanip>

struct BenchResult {
    const char* name;
    double ms;
    long long macs;
};

static constexpr int BENCH_ITERS = 10000;

static long long conv_mac_count() {
    return static_cast<long long>(CONV_W) *
           static_cast<long long>(CONV_H) *
           static_cast<long long>(OUT_C) *
           static_cast<long long>(K_SIZE) *
           static_cast<long long>(K_SIZE) *
           static_cast<long long>(IN_C);
}

static void fill_bench_input(InputTensor& input) {
    for (int x = 0; x < IN_W; ++x) {
        for (int y = 0; y < IN_H; ++y) {
            for (int c = 0; c < IN_C; ++c) {
                input[x][y][c] = 0.1f * static_cast<float>(x + y + c + 1);
            }
        }
    }
}

static void fill_bench_weights(WeightTensor& weights) {
    for (int oc = 0; oc < OUT_C; ++oc) {
        for (int ky = 0; ky < K_SIZE; ++ky) {
            for (int kx = 0; kx < K_SIZE; ++kx) {
                for (int ic = 0; ic < IN_C; ++ic) {
                    weights[oc][ky][kx][ic] =
                        0.07f * static_cast<float>(oc + ky + kx + ic + 1);
                }
            }
        }
    }
}

static void convert_input_to_fixed_bench(const InputTensor& src,
                                         InputTensorFixed& dst) {
    for (int x = 0; x < IN_W; ++x) {
        for (int y = 0; y < IN_H; ++y) {
            for (int c = 0; c < IN_C; ++c) {
                dst[x][y][c] = float_to_fixed(src[x][y][c]);
            }
        }
    }
}

static void convert_weights_to_fixed_bench(const WeightTensor& src,
                                           WeightTensorFixed& dst) {
    for (int oc = 0; oc < OUT_C; ++oc) {
        for (int ky = 0; ky < K_SIZE; ++ky) {
            for (int kx = 0; kx < K_SIZE; ++kx) {
                for (int ic = 0; ic < IN_C; ++ic) {
                    dst[oc][ky][kx][ic] = float_to_fixed(src[oc][ky][kx][ic]);
                }
            }
        }
    }
}

static BenchResult benchmark_float_top(const InputTensor& input,
                                       const WeightTensor& weights) {
    OutputTensor output {};

    auto start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < BENCH_ITERS; ++i) {
        cnn_top_float(input, weights, output);
    }

    auto end = std::chrono::high_resolution_clock::now();

    double ms = std::chrono::duration<double, std::milli>(end - start).count();

    return {
        "naive float",
        ms,
        conv_mac_count() * BENCH_ITERS
    };
}

static BenchResult benchmark_fixed_top(const InputTensorFixed& input,
                                       const WeightTensorFixed& weights) {
    OutputTensorFixed output {};

    auto start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < BENCH_ITERS; ++i) {
        cnn_top_fixed(input, weights, output);
    }

    auto end = std::chrono::high_resolution_clock::now();

    double ms = std::chrono::duration<double, std::milli>(end - start).count();

    return {
        "fixed tiled",
        ms,
        conv_mac_count() * BENCH_ITERS
    };
}

static void print_benchmark_table(const BenchResult results[], int n) {
    std::cout << "\nBenchmark results over "
              << BENCH_ITERS
              << " iterations\n\n";

    std::cout << "| Version      | Time (ms) | MACs       | MACs/ms     |\n";
    std::cout << "|--------------|-----------|------------|-------------|\n";

    for (int i = 0; i < n; ++i) {
        double macs_per_ms = results[i].macs / results[i].ms;

        std::cout << "| "
                  << std::left << std::setw(12) << results[i].name
                  << " | "
                  << std::right << std::setw(9) << std::fixed << std::setprecision(3) << results[i].ms
                  << " | "
                  << std::setw(10) << results[i].macs
                  << " | "
                  << std::setw(11) << std::fixed << std::setprecision(1) << macs_per_ms
                  << " |\n";
    }

    std::cout << "\nConv MACs per inference = "
              << conv_mac_count()
              << "\n";
}

void run_benchmarks() {
    InputTensor input_float {};
    WeightTensor weights_float {};

    InputTensorFixed input_fixed {};
    WeightTensorFixed weights_fixed {};

    fill_bench_input(input_float);
    fill_bench_weights(weights_float);

    convert_input_to_fixed_bench(input_float, input_fixed);
    convert_weights_to_fixed_bench(weights_float, weights_fixed);

    BenchResult results[2];

    results[0] = benchmark_float_top(input_float, weights_float);
    results[1] = benchmark_fixed_top(input_fixed, weights_fixed);

    print_benchmark_table(results, 2);
}