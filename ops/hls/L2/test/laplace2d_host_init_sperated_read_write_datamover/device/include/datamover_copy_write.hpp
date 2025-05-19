
#pragma once

#include "common_config.hpp"
#include <ops_hls_datamover.hpp>

extern "C" void datamover_copy_write(
    const unsigned short range_start_x,
    const unsigned short range_end_x,
    const unsigned short range_start_y,
    const unsigned short range_end_y,
    const unsigned short arg1_gridSize_x,
    const unsigned short arg1_gridSize_y,
    ap_uint<data_width>* arg1_out,
    hls::stream<ap_axiu<axis_data_width,0,0,0>>& arg1_stream_in
);

