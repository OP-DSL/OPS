
#include "../../include/ops_hls_datamover.hpp"

#define AXI_M_WIDTH 512
#define AXIS_WIDTH 256
#define HLS_STREAM_WIDTH 128

//template <unsigned int MEM_DATA_WIDTH, unsigned int STREAM_DATA_WIDTH, unsigned int BURST_SIZE=32>
void dut(ap_uint<AXI_M_WIDTH>* mem_in,
		ap_uint<AXI_M_WIDTH>* mem_out,
		unsigned int size);

