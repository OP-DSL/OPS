
#include "../../include/ops_hls_datamover.hpp"

#define AXI_M_WIDTH 512
#define DATA_WIDTH 32
#define AXIS_WIDTH 256
#define HLS_STREAM_WIDTH 128
constexpr unsigned int max_size_x = 30;
constexpr unsigned int max_depth = max_size_x * max_size_x * max_size_x;

//template <unsigned int MEM_DATA_WIDTH, unsigned int STREAM_DATA_WIDTH, unsigned int BURST_SIZE=32>
void dut(hls::stream<ap_axiu<AXIS_WIDTH,0,0,0>>& axis_in,
		ap_uint<DATA_WIDTH>* mem_out,
		unsigned int size);

void widen_w(hls::stream<ap_axiu<AXIS_WIDTH,0,0,0>>& axis_in,
		ap_uint<AXI_M_WIDTH>* mem_out,
		unsigned int size);


void precise_w(hls::stream<ap_axiu<AXIS_WIDTH,0,0,0>>& axis_in,
		ap_uint<DATA_WIDTH>* mem_out,
		unsigned int size);
