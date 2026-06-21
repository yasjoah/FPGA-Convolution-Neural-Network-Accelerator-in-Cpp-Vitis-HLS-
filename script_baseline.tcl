open_project cnn_hls_baseline
set_top cnn_top_fixed_baseline

set CFLAGS "-I./include -std=c++17"

add_files src/cnn_top.cpp        -cflags $CFLAGS
add_files src/conv_baseline.cpp  -cflags $CFLAGS
add_files src/relu_fixed.cpp     -cflags $CFLAGS
add_files src/pooling_fixed.cpp  -cflags $CFLAGS
add_files src/fixed_point.cpp    -cflags $CFLAGS

open_solution "solution1"
set_part {xczu3eg-sbva484-1-e}
create_clock -period 10 -name default

csynth_design

exit