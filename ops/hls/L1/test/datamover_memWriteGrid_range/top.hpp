
#pragma once

#include "../../include/ops_hls_defs.hpp"
#include "../../include/ops_hls_datamover.hpp"

#define AXI_M_WIDTH 512
#define AXIS_WIDTH 256
#define DATA_WIDTH 32

//template <unsigned int MEM_DATA_WIDTH, unsigned int STREAM_DATA_WIDTH, unsigned int BURST_SIZE=32>
void dut(ap_uint<DATA_WIDTH>* mem_out,
		hls::stream<ap_axiu<AXIS_WIDTH,0,0,0>>& strm_in,
		ops::hls::SizeType gridSize,
		ops::hls::AccessRange& range);

