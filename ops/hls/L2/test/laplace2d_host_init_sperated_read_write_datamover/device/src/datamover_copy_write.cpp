
#include <ops_hls_datamover.hpp>
#include "../include/datamover_copy_write.hpp"

static void datamover_copy_write_dataflow_region(ops::hls::AccessRange& range,
		ops::hls::SizeType& arg1_gridSize,
	    ap_uint<data_width>* arg1_out,
	    hls::stream<ap_axiu<axis_data_width,0,0,0>>& arg1_stream_in)
{
#pragma HLS DATAFLOW
    ops::hls::memWriteGridSimpleV2<mem_data_width, axis_data_width, data_width>(arg1_out, arg1_stream_in, arg1_gridSize, range);
}


extern "C" void datamover_copy_write(
    const unsigned short range_start_x,
    const unsigned short range_end_x,
    const unsigned short range_start_y,
    const unsigned short range_end_y,
    const unsigned short arg1_gridSize_x,
    const unsigned short arg1_gridSize_y,
    ap_uint<data_width>* arg1_out,
    hls::stream<ap_axiu<axis_data_width,0,0,0>>& arg1_stream_in
)
{
    #pragma HLS INTERFACE s_axilite port = range_start_x bundle = control
	#pragma HLS INTERFACE s_axilite port = range_end_x bundle = control
	#pragma HLS INTERFACE s_axilite port = range_start_y bundle = control
	#pragma HLS INTERFACE s_axilite port = range_end_y bundle = control

    #pragma HLS INTERFACE s_axilite port = arg1_gridSize_x bundle = control
	#pragma HLS INTERFACE s_axilite port = arg1_gridSize_y bundle = control

    #pragma HLS INTERFACE mode=m_axi bundle=gmem1 depth=4096 max_read_burst_length=64 max_write_burst_length=64 \
        num_read_outstanding=4 num_write_outstanding=4 port=arg1_out offset=slave
    #pragma HLS INTERFACE s_axilite port = arg1_out bundle = control

    #pragma HLS INTERFACE mode=axis port=arg1_stream_in register

	#pragma HLS INTERFACE ap_ctrl_chain port = return
	#pragma HLS INTERFACE s_axilite port = return bundle = control

    ops::hls::AccessRange range;
    range.start[0] = range_start_x;
    range.start[1] = range_start_y;
    range.end[0] = range_end_x;
    range.end[1] = range_end_y;
    range.dim = 2;

    ops::hls::SizeType arg1_gridSize = {arg1_gridSize_x, arg1_gridSize_y, 1};

    datamover_copy_write_dataflow_region(range,  arg1_gridSize, arg1_out, arg1_stream_in);
}
