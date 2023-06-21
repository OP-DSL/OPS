
#include "top.hpp"

//template <unsigned int MEM_DATA_WIDTH, unsigned int STREAM_DATA_WIDTH, unsigned int BURST_SIZE>
void dut(ap_uint<AXI_M_WIDTH>* mem_in0,
		ap_uint<AXI_M_WIDTH>* mem_in1,
		hls::stream<ap_axiu<AXIS_WIDTH,0,0,0>>& strm_loopback,
		unsigned int size)
{
#pragma HLS TOP
	#pragma HLS INTERFACE mode=m_axi bundle=gmem0 depth=4096 max_read_burst_length=64 max_write_burst_length=64 \
			num_read_outstanding=4 num_write_outstanding=4 port=mem_in0 offset=slave
	#pragma HLS INTERFACE mode=m_axi bundle=gmem0 depth=4096 max_read_burst_length=64 max_write_burst_length=64 \
			num_read_outstanding=4 num_write_outstanding=4 port=mem_in1 offset=slave
	#pragma HLS INTERFACE axis port=strm_loopback register

	ops::hls::mem2stream<AXI_M_WIDTH, AXIS_WIDTH>(mem_in0, mem_in1, strm_loopback, size, 0);
	ops::hls::stream2mem<AXI_M_WIDTH, AXIS_WIDTH>(mem_in0, mem_in1, strm_loopback, size, 1);
}
