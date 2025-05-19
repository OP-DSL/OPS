
#include "top.hpp"

//template <unsigned int MEM_DATA_WIDTH, unsigned int STREAM_DATA_WIDTH, unsigned int BURST_SIZE>
void dut(ap_uint<DATA_WIDTH>* mem_out,
		hls::stream<ap_axiu<AXIS_WIDTH,0,0,0>>& strm_in,
		ops::hls::SizeType gridSize,
		ops::hls::AccessRange& range)
{
	#pragma HLS TOP
	#pragma HLS INTERFACE mode=m_axi depth=8192 max_read_burst_length=64 max_write_burst_length=64 num_read_outstanding=4 num_write_outstanding=4 port=mem_out offset=slave
	#pragma HLS INTERFACE mode=s_axilite port=mem_out bundle=control

	#pragma HLS INTERFACE axis port=strm_in register

	#pragma HLS INTERFACE mode=s_axilite port=gridSize bundle=control

	#pragma HLS DISAGGREGATE variable=range
	#pragma HLS INTERFACE mode=s_axilite port=range.start bundle=control
	#pragma HLS INTERFACE mode=s_axilite port=range.end bundle=control
	#pragma HLS INTERFACE mode=s_axilite port=range.dim bundle=control

	#pragma HLS INTERFACE mode=s_axilite port=return bundle=control
//	#pragma HLS INTERFACE ap_ctrl_chain port=return

	ops::hls::memWriteGrid<AXI_M_WIDTH, AXIS_WIDTH, DATA_WIDTH>(mem_out, strm_in, gridSize, range);
}
