#include <iostream>
#include <cmath>

#include "../include/config.h"
#include "../include/types.h"
#include "../include/layers.h"
#include "../include/fixed_point.h"

bool assert_near(float a, float b, float tol);

void convert_input_to_fixed(const InputTensor& input_float,
                            InputTensorFixed& input_fixed) {
    for (int x = 0; x < IN_W; ++x) {
        for (int y = 0; y < IN_H; ++y) {
            for (int c = 0; c < IN_C; ++c) {
                input_fixed[x][y][c] = float_to_fixed(input_float[x][y][c]);
            }
        }
    }
}

void convert_weights_to_fixed(const WeightTensor& weights_float,
                              WeightTensorFixed& weights_fixed) {
    for (int oc = 0; oc < OUT_C; ++oc) {
        for (int ky = 0; ky < K_SIZE; ++ky) {
            for (int kx = 0; kx < K_SIZE; ++kx) {
                for (int ic = 0; ic < IN_C; ++ic) {
                    weights_fixed[oc][ky][kx][ic] =
                        float_to_fixed(weights_float[oc][ky][kx][ic]);
                }
            }
        }
    }
}


bool assert_near(float a, float b, float tol = 1e-4f) {
    if (std::fabs(a - b) > tol) {
        std::cout << "ASSERT FAILED: got " << a << ", expected " << b << "\n";
        return false;
    }
    return true;
}

void fill_test_input(InputTensor& input) {
    for (int x = 0; x < IN_W; ++x) {
        for (int y = 0; y < IN_H; ++y) {
            for (int c = 0; c < IN_C; ++c) {
                input[x][y][c] = 0.1f * static_cast<float>(x + y + c + 1);
            }
        }
    }
}

void fill_test_weights(WeightTensor& weights) {
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

void print_tensor_slice(const ConvTensor& t, int oc) {
    std::cout << "ConvTensor slice oc=" << oc << "\n";
    for (int y = 0; y < CONV_H; ++y) {
        for (int x = 0; x < CONV_W; ++x) {
            std::cout << t[x][y][oc] << " ";
        }
        std::cout << "\n";
    }
}

bool test_conv() {
    std::cout << "Running test_conv...\n";

    InputTensor input {};
    WeightTensor weights {};
    ConvTensor out {};

    // Local ramp input just for this hand-check test.
    for (int x = 0; x < IN_W; ++x) {
        for (int y = 0; y < IN_H; ++y) {
            for (int c = 0; c < IN_C; ++c) {
                input[x][y][c] = 0.0f;
            }

            input[x][y][0] = static_cast<float>(x + y);
        }
    }

    // Zero all weights.
    for (int oc = 0; oc < OUT_C; ++oc) {
        for (int ky = 0; ky < K_SIZE; ++ky) {
            for (int kx = 0; kx < K_SIZE; ++kx) {
                for (int ic = 0; ic < IN_C; ++ic) {
                    weights[oc][ky][kx][ic] = 0.0f;
                }
            }
        }
    }

    // Sobel-x filter for output channel 0.
    weights[0][0][0][0] = -1.0f;
    weights[0][0][1][0] =  0.0f;
    weights[0][0][2][0] =  1.0f;

    weights[0][1][0][0] = -2.0f;
    weights[0][1][1][0] =  0.0f;
    weights[0][1][2][0] =  2.0f;

    weights[0][2][0][0] = -1.0f;
    weights[0][2][1][0] =  0.0f;
    weights[0][2][2][0] =  1.0f;

    conv_naive(input, weights, out);

    float expected = 8.0f;

    std::cout << "Golden conv out[0][0][0] = " << expected << "\n";
    std::cout << "Actual conv out[0][0][0] = " << out[0][0][0] << "\n";

    return assert_near(out[0][0][0], expected);
}

bool test_relu() {
    std::cout << "Running test_relu...\n";

    ConvTensor input {};
    ReluTensor out {};

    input[0][0][0] = -3.0f;
    input[0][1][0] =  2.5f;
    input[1][0][0] =  0.0f;

    relu(input, out);

    bool ok = true;
    ok &= assert_near(out[0][0][0], 0.0f);
    ok &= assert_near(out[0][1][0], 2.5f);
    ok &= assert_near(out[1][0][0], 0.0f);

    return ok;
}

bool test_pooling() {
    std::cout << "Running test_pooling...\n";

    ReluTensor input {};
    OutputTensor out {};

    // Assuming 2x2 max pooling.
    //
    // Top-left pooling window:
    // 1 5
    // 3 4
    //
    // Expected max = 5.
    input[0][0][0] = 1.0f;
    input[0][1][0] = 5.0f;
    input[1][0][0] = 3.0f;
    input[1][1][0] = 4.0f;

    pooling(input, out);

    float expected = 5.0f;

    std::cout << "Golden pooling out[0][0][0] = " << expected << "\n";
    std::cout << "Actual pooling out[0][0][0] = " << out[0][0][0] << "\n";

    return assert_near(out[0][0][0], expected);
}

bool test_cnn_top_float() {
    std::cout << "Running test_cnn_top_float...\n";

    InputTensor input {};
    WeightTensor weights {};

    OutputTensor top_out {};
    OutputTensor ref_out {};

    ConvTensor conv_out {};
    ReluTensor relu_out {};

    fill_test_input(input);
    fill_test_weights(weights);

    // Run top-level function.
    cnn_top_float(input, weights, top_out);

    // Run reference pipeline manually.
    conv_naive(input, weights, conv_out);
    relu(conv_out, relu_out);
    pooling(relu_out, ref_out);

    bool ok = true;

    for (int y = 0; y < OUT_H; ++y) {
        for (int x = 0; x < OUT_W; ++x) {
            for (int oc = 0; oc < OUT_C; ++oc) {
                ok &= assert_near(top_out[x][y][oc], ref_out[x][y][oc]);
            }
        }
    }

    return ok;
}

bool test_fixed_point_mac() {
    bool ok = true;

    fixed16_t a = float_to_fixed(1.5f); // raw 384
    fixed16_t b = float_to_fixed(2.0f); // raw 512

    accum32_t acc = 0;
    acc = mac(acc, a, b);

    fixed16_t result_fixed = accum_to_fixed(acc);
    float result_float = fixed_to_float(result_fixed);

    ok &= assert_near(result_float, 3.0f);

    if (ok) {
        std::cout << "test_fixed_point_mac PASSED\n";
    } else {
        std::cout << "test_fixed_point_mac FAILED\n";
    }

    return ok;
}

bool test_fixed_point_conversion() {
    std::cout << "Running test_fixed_point_conversion...\n";

    bool ok = true;

    ok &= (float_to_fixed(1.0f) == 256);
    ok &= (float_to_fixed(0.5f) == 128);
    ok &= (float_to_fixed(1.25f) == 320);
    ok &= (float_to_fixed(-1.0f) == -256);

    ok &= assert_near(fixed_to_float(float_to_fixed(1.0f)), 1.0f);
    ok &= assert_near(fixed_to_float(float_to_fixed(0.5f)), 0.5f);
    ok &= assert_near(fixed_to_float(float_to_fixed(1.25f)), 1.25f);
    ok &= assert_near(fixed_to_float(float_to_fixed(-1.0f)), -1.0f);

    return ok;
}

bool test_cnn_top_fixed() {
    std::cout << "Running test_cnn_top_fixed...\n";

    InputTensor input_float {};
    WeightTensor weights_float {};

    InputTensorFixed input_fixed {};
    WeightTensorFixed weights_fixed {};

    OutputTensor output_float {};
    OutputTensorFixed output_fixed {};

    fill_test_input(input_float);
    fill_test_weights(weights_float);

    convert_input_to_fixed(input_float, input_fixed);
    convert_weights_to_fixed(weights_float, weights_fixed);

    cnn_top_float(input_float, weights_float, output_float);
    cnn_top_fixed(input_fixed, weights_fixed, output_fixed);

bool ok = true;
constexpr float TOL = 0.10f;

float max_error = 0.0f;
int max_x = 0;
int max_y = 0;
int max_oc = 0;

for (int x = 0; x < OUT_W; ++x) {
    for (int y = 0; y < OUT_H; ++y) {
        for (int oc = 0; oc < OUT_C; ++oc) {
            float fixed_as_float = fixed_to_float(output_fixed[x][y][oc]);
            float expected = output_float[x][y][oc];

            float error = std::fabs(fixed_as_float - expected);

            if (error > max_error) {
                max_error = error;
                max_x = x;
                max_y = y;
                max_oc = oc;
            }

            if (error > TOL) {
                std::cout << "Mismatch at ["
                          << x << "]["
                          << y << "]["
                          << oc << "]: fixed="
                          << fixed_as_float
                          << ", float="
                          << expected
                          << ", error="
                          << error
                          << "\n";
                ok = false;
            }
        }
    }
}

std::cout << "Max fixed-point error = " << max_error
          << " at ["
          << max_x << "]["
          << max_y << "]["
          << max_oc << "]\n";

return ok;
}

bool test_tiled_conv_matches_untiled() {
    std::cout << "Running test_tiled_conv_matches_untiled...\n";

    InputTensor input_float {};
    WeightTensor weights_float {};

    InputTensorFixed input_fixed {};
    WeightTensorFixed weights_fixed {};

    ConvTensorFixed untiled_out {};
    ConvTensorFixed tiled_out {};

    fill_test_input(input_float);
    fill_test_weights(weights_float);

    convert_input_to_fixed(input_float, input_fixed);
    convert_weights_to_fixed(weights_float, weights_fixed);

    conv_fixed(input_fixed, weights_fixed, untiled_out);
    conv_engine_fixed(input_fixed, weights_fixed, tiled_out);

    bool ok = true;

    for (int x = 0; x < CONV_W; ++x) {
        for (int y = 0; y < CONV_H; ++y) {
            for (int oc = 0; oc < OUT_C; ++oc) {
                if (untiled_out[x][y][oc] != tiled_out[x][y][oc]) {
                    std::cout << "Tiled mismatch at ["
                              << x << "]["
                              << y << "]["
                              << oc << "]: untiled="
                              << fixed_to_float(untiled_out[x][y][oc])
                              << ", tiled="
                              << fixed_to_float(tiled_out[x][y][oc])
                              << "\n";
                    ok = false;
                }
            }
        }
    }

    if (ok) {
        std::cout << "Tiled convolution matches untiled convolution exactly.\n";
    }

    return ok;
}

int main() {
    bool ok = true;

    ok &= test_conv();
    ok &= test_relu();
    ok &= test_pooling();
    ok &= test_cnn_top_float();
    ok &= test_fixed_point_mac();
    ok &= test_fixed_point_conversion();
    ok &= test_cnn_top_fixed();
    ok &= test_tiled_conv_matches_untiled();

    if (ok) {
        std::cout << "\nAll tests passed.\n";
        run_benchmarks();
        return 0;
    }

    std::cout << "\nSome tests failed.\n";
    return 1;
}