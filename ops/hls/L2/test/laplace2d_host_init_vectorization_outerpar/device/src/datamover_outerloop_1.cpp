
#include <ops_hls_datamover.hpp>
#include "../include/datamover_outerloop_1.hpp"

static void datamover_outerloop_1_dataflow_region(ops::hls::AccessRange& range,
		ops::hls::SizeType& gridSize,
		ap_uint<mem_data_width>* arg0_1,
	    ap_uint<mem_data_width>* arg0_2,
		hls::stream<ap_axiu<axis_data_width,0,0,0>>& arg0_stream_out,
	    hls::stream<ap_axiu<axis_data_width,0,0,0>>& arg1_stream_in)
{
#pragma HLS DATAFLOW
    ops::hls::memReadGridV2<mem_data_width, axis_data_width, data_width>(arg0_1, arg0_stream_out, gridSize, range);
//    ops::hls::memWriteGrid<mem_data_width, axis_data_width, data_width>(arg1_out, arg1_stream_in, arg1_gridSize, write_range);
    ops::hls::memWriteGridSimpleV2<mem_data_width, axis_data_width, data_width>(arg0_2, arg1_stream_in, gridSize, range);
}

extern "C" void datamover_outerloop_1(
    const unsigned short range_start_x,
    const unsigned short range_end_x,
    const unsigned short range_start_y,
    const unsigned short range_end_y,
    const unsigned short gridSize_x,
    const unsigned short gridSize_y,
    const unsigned int outer_itr,
	ap_uint<mem_data_width>* arg0,
    ap_uint<mem_data_width>* arg1,
	hls::stream<ap_axiu<axis_data_width,0,0,0>>& arg0_stream_out,
    hls::stream<ap_axiu<axis_data_width,0,0,0>>& arg1_stream_in)
{
    #pragma HLS INTERFACE s_axilite port = range_start_x bundle = control
	#pragma HLS INTERFACE s_axilite port = range_end_x bundle = control
	#pragma HLS INTERFACE s_axilite port = range_start_y bundle = control
	#pragma HLS INTERFACE s_axilite port = range_end_y bundle = control
	#pragma HLS INTERFACE s_axilite port = gridSize_x bundle = control
	#pragma HLS INTERFACE s_axilite port = gridSize_y bundle = control
	#pragma HLS INTERFACE s_axilite port = outer_itr bundle = control

    #pragma HLS INTERFACE mode=m_axi bundle=gmem0 depth=4096 max_read_burst_length=64 max_write_burst_length=64 \
        num_read_outstanding=4 num_write_outstanding=4 port=arg0 offset=slave
    #pragma HLS INTERFACE mode=m_axi bundle=gmem0 depth=4096 max_read_burst_length=64 max_write_burst_length=64 \
        num_read_outstanding=4 num_write_outstanding=4 port=arg1 offset=slave
	#pragma HLS INTERFACE s_axilite port = arg0 bundle = control
    #pragma HLS INTERFACE s_axilite port = arg1 bundle = control

    #pragma HLS INTERFACE mode=axis port=arg0_stream_out register
    #pragma HLS INTERFACE mode=axis port=arg1_stream_in register

	#pragma HLS INTERFACE ap_ctrl_chain port = return
	#pragma HLS INTERFACE s_axilite port = return bundle = control



    ops::hls::AccessRange range;
    range.start[0] = range_start_x;
    range.start[1] = range_start_y;
    range.end[0] = range_end_x;
    range.end[1] = range_end_y;
    range.dim = 2;

    ops::hls::SizeType gridSize = {gridSize_x, gridSize_y, 1};

#ifdef DEBUG_LOG
    printf("[KERNEL_DEBUG]|%s| starting datamover TOP range:(%d,%d,%d) ---> (%d,%d,%d)\n", __func__,
            range.start[0], range.start[1], range.start[2], range.end[0], range.end[1], range.end[2]);
    printf("[KERNEL_DEBUG]|%s| gridSize: (%d, %d, %d), \n", __func__,
            gridSize[0], gridSize[1], gridSize[2]);
#endif

    for (int i = 0; i < outer_itr; i++)
    {
        if (i % 2 == 0)
        	datamover_outerloop_1_dataflow_region(range, gridSize, arg0, arg1, arg0_stream_out, arg1_stream_in);
        else
        	datamover_outerloop_1_dataflow_region(range, gridSize, arg1, arg0, arg0_stream_out, arg1_stream_in);
    }

#ifdef DEBUG_LOG
    printf("[KERNEL_DEBUG]|%s| Ending datamover TOP\n", __func__);
#endif
}



