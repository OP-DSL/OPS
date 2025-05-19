
#pragma once

#include "../../include/ops_hls_defs.hpp"
#include "../../include/ops_hls_datamover.hpp"
#define AXI_M_WIDTH 512
#define STREAM_WIDTH 256
#define AXIS_WIDTH 256
#define DATA_WIDTH 32

//template <unsigned int MEM_DATA_WIDTH, unsigned int STREAM_DATA_WIDTH, unsigned int BURST_SIZE=32>
void dut(hls::stream<ap_axiu<AXIS_WIDTH,0,0,0>>& strm_in,
		hls::stream<ap_uint<STREAM_WIDTH>>& data_out,
		hls::stream<ap_uint<STREAM_WIDTH/8>>& mask_out,
		unsigned int size,
		unsigned short shiftBytes = 0);

