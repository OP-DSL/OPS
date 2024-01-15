
#pragma once

#include "common_config.hpp"
#include <ops_hls_datamover.hpp>

extern "C" void datamover_outerloop_1(
    const unsigned short range_start_x,
    const unsigned short range_end_x,
    const unsigned short range_start_y,
    const unsigned short range_end_y,
    const unsigned short gridSize_x,
    const unsigned short gridSize_y,
    const unsigned int outer_itr,
	ap_uint<mem_data_width>* arg0,
    ap_uint<mem_data_width>* arg1,
	hls::stream<ap_axiu<axis_data_width,0,0,0>>& arg0_stream_out,
    hls::stream<ap_axiu<axis_data_width,0,0,0>>& arg1_stream_in
);
