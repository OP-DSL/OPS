
#include "top.hpp"

//template <unsigned int MEM_DATA_WIDTH, unsigned int STREAM_DATA_WIDTH, unsigned int BURST_SIZE>
void dut(ap_uint<AXI_M_WIDTH>* mem_in,
		ap_uint<AXI_M_WIDTH>* mem_out,
		hls::stream<ap_axiu<AXIS_WIDTH,0,0,0>>& strm_loopback,
		unsigned int size)
{
#pragma HLS TOP
	#pragma HLS INTERFACE mode=m_axi bundle=gmem0 depth=4096 max_read_burst_length=64 max_write_burst_length=64 \
			num_read_outstanding=4 num_write_outstanding=4 port=mem_in offset=slave
	#pragma HLS INTERFACE mode=m_axi bundle=gmem0 depth=4096 max_read_burst_length=64 max_write_burst_length=64 \
			num_read_outstanding=4 num_write_outstanding=4 port=mem_out offset=slave
	#pragma HLS INTERFACE axis port=strm_loopback register

	ops::hls::mem2axis<AXI_M_WIDTH, AXIS_WIDTH>(mem_in, strm_loopback, size);
	ops::hls::axis2mem<AXI_M_WIDTH, AXIS_WIDTH>(mem_out, strm_loopback, size);
}