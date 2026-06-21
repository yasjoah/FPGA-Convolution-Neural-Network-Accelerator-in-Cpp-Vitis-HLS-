cmake_minimum_required(VERSION 3.16)
project(cnn_fpga_accelerator CXX)
set(CMAKE_CXX_STANDARD 17)
add_executable(cnn_test
    testbench/main.cpp
    src/conv_naive.cpp
    src/conv_fixed.cpp
    src/fixed_point.cpp
    src/relu.cpp
    src/relu_fixed.cpp
    src/pooling.cpp
    src/pooling_fixed.cpp
    src/cnn_top.cpp
    src/benchmark.cpp)
target_include_directories(cnn_test PRIVATE include)
