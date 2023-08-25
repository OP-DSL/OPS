
#include "top.hpp"

//template <unsigned int MEM_DATA_WIDTH, unsigned int STREAM_DATA_WIDTH, unsigned int BURST_SIZE>
void dut(ap_uint<AXI_M_WIDTH>* mem_in,
		ap_uint<AXI_M_WIDTH>* mem_out,
		unsigned int size)
{
#pragma HLS TOP
//	#pragma HLS INTERFACE mode=m_axi bundle=gmem0 depth=4096 max_read_burst_length=64 max_write_burst_length=64 \
//			num_read_outstanding=4 num_write_outstanding=4 port=mem_in offset=slave
//	#pragma HLS INTERFACE mode=m_axi bundle=gmem0 depth=4096 max_read_burst_length=64 max_write_burst_length=64 \
//			num_read_outstanding=4 num_write_outstanding=4 port=mem_out offset=slave

#pragma HLS DATAFLOW
	hls::stream<ap_axiu<AXIS_WIDTH,0,0,0>> axis_out;
	hls::stream<ap_axiu<AXIS_WIDTH,0,0,0>> axis_in;
	hls::stream<ap_uint<HLS_STREAM_WIDTH>> hls_stream_loopback;
	
	ops::hls::mem2axis<AXI_M_WIDTH, AXIS_WIDTH>(mem_in, axis_out, size);
	ops::hls::axis2stream<AXIS_WIDTH, HLS_STREAM_WIDTH>(axis_out, hls_stream_loopback, size);
	ops::hls::stream2axis<AXIS_WIDTH, HLS_STREAM_WIDTH>(axis_in, hls_stream_loopback, size);
	ops::hls::axis2mem<AXI_M_WIDTH, AXIS_WIDTH>(mem_out, axis_in, size);
}
