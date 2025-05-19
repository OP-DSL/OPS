
#include "top.hpp"

//template <unsigned int MEM_DATA_WIDTH, unsigned int STREAM_DATA_WIDTH, unsigned int BURST_SIZE>
void dut(ap_uint<AXI_M_WIDTH>* mem_out0,
		ap_uint<AXI_M_WIDTH>* mem_out1,
		hls::stream<ap_axiu<AXIS_WIDTH,0,0,0>>& strm_in,
		unsigned int size,
		unsigned int selector)
{
#pragma HLS TOP
	#pragma HLS INTERFACE mode=m_axi bundle=gmem0 depth=4096 max_read_burst_length=64 max_write_burst_length=64 \
			num_read_outstanding=4 num_write_outstanding=4 port=mem_out0 offset=slave
	#pragma HLS INTERFACE mode=m_axi bundle=gmem0 depth=4096 max_read_burst_length=64 max_write_burst_length=64 \
			num_read_outstanding=4 num_write_outstanding=4 port=mem_out1 offset=slave
	#pragma HLS INTERFACE axis port=strm_in register
	#pragma HLS INTERFACE ap_ctrl_chain port=return

	ops::hls::axis2mem<AXI_M_WIDTH, AXIS_WIDTH>(mem_out0, mem_out1, strm_in, size, selector);
}
