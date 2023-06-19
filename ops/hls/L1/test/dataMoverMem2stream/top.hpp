#include "../../include/dataMover.hpp"

#define AXI_M_WIDTH 512
#define AXIS_WIDTH 256

//template <unsigned int MEM_DATA_WIDTH, unsigned int STREAM_DATA_WIDTH, unsigned int BURST_SIZE=32>
void dut(ap_uint<AXI_M_WIDTH>* mem_in0,
		ap_uint<AXI_M_WIDTH>* mem_in1,
		hls::stream<ap_axiu<AXIS_WIDTH,0,0,0>>& strm_out,
		unsigned int size,
		unsigned int selector);

