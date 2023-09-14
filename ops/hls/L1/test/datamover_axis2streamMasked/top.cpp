
#include "top.hpp"

//template <unsigned int MEM_DATA_WIDTH, unsigned int STREAM_DATA_WIDTH, unsigned int BURST_SIZE>
void dut(hls::stream<ap_axiu<AXIS_WIDTH,0,0,0>>& strm_in,
		hls::stream<ap_uint<STREAM_WIDTH>>& data_out,
		hls::stream<ap_uint<STREAM_WIDTH/8>>& mask_out,
		unsigned int size,
		unsigned short shiftBytes)
{
	#pragma HLS TOP
	#pragma HLS INTERFACE axis port=strm_in register
	#pragma HLS INTERFACE mode=s_axilite port=size bundle=control
	#pragma HLS INTERFACE mode=s_axilite port=shiftBytes bundle=control
	#pragma HLS INTERFACE mode=s_axilite port=return bundle=control
	#pragma HLS INTERFACE ap_ctrl_chain port=return

	ops::hls::axis2streamMasked<AXIS_WIDTH, STREAM_WIDTH>(strm_in, data_out, mask_out, size, shiftBytes);
}
