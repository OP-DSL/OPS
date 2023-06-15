#include "../../include/dataMover.hpp"

//template <unsigned int MEM_DATA_WIDTH, unsigned int STREAM_DATA_WIDTH, unsigned int BURST_SIZE=32>
void dut(ap_uint<512>* mem_in0,
		ap_uint<512>* mem_in1,
		hls::stream<ap_axiu<256,0,0,0>>& strm_out,
		unsigned int size,
		unsigned int selector);

