
#include "top.hpp"

//template <unsigned int MEM_DATA_WIDTH, unsigned int STREAM_DATA_WIDTH, unsigned int BURST_SIZE>
void dut(hls::stream<ap_axiu<AXIS_WIDTH,0,0,0>>& axis_in,
		ap_uint<DATA_WIDTH>* mem_out,
		unsigned int size)
{
#pragma HLS TOP
//	#pragma HLS INTERFACE mode=m_axi bundle=gmem0 depth=4096 max_read_burst_length=64 max_write_burst_length=64 \
//			num_read_outstanding=4 num_write_outstanding=4 port=mem_in offset=slave
//	#pragma HLS INTERFACE mode=m_axi bundle=gmem0 depth=4096 max_read_burst_length=64 max_write_burst_length=64 \
//			num_read_outstanding=4 num_write_outstanding=4 port=mem_out offset=slave

	unsigned int num_of_bytes_in_widen_axi = AXI_M_WIDTH/8;
	unsigned int widen_size = (size / num_of_bytes_in_widen_axi) * num_of_bytes_in_widen_axi;
	unsigned int data_size = DATA_WIDTH/8;
	unsigned int precise_offset_index = widen_size/data_size;
	unsigned int precise_size = size - widen_size;

	widen_w(axis_in, (ap_uint<AXI_M_WIDTH>*)mem_out, widen_size);

	precise_w(axis_in, (ap_uint<DATA_WIDTH>*)mem_out + precise_offset_index, precise_size);

}


void widen_w(hls::stream<ap_axiu<AXIS_WIDTH,0,0,0>>& axis_in,
		ap_uint<AXI_M_WIDTH>* mem_out,
		unsigned int size)
{
	ops::hls::axis2mem<AXI_M_WIDTH, AXIS_WIDTH>(mem_out, axis_in, size);
}

void precise_w(hls::stream<ap_axiu<AXIS_WIDTH,0,0,0>>& axis_in,
		ap_uint<DATA_WIDTH>* mem_out,
		unsigned int size)
{
	ops::hls::axis2memMasked<DATA_WIDTH, AXIS_WIDTH>(mem_out, axis_in, size);
}
