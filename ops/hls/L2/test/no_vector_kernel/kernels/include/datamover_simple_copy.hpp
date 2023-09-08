#pragma once

#include "common_config.hpp"
#include <ops_hls_datamover.hpp>

extern "C" void kernel_datamover_simple_copy(
    const unsigned short range_start_x,
	const unsigned short range_end_x,
	const unsigned short range_start_y,
	const unsigned short range_end_y,
	const unsigned short gridSize_x,
	const unsigned short gridSize_y,
    ap_uint<data_width>* arg0_out,
    hls::stream<ap_axiu<axis_data_width,0,0,0>>& stream0_in
);
