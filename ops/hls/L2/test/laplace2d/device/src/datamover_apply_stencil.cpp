
#include <ops_hls_datamover.hpp>
#include "../include/datamover_apply_stencil.hpp"

static void datamover_apply_stencil_dataflow_region(ops::hls::AccessRange& read_range,
        ops::hls::AccessRange& write_range,
		ops::hls::SizeType& arg0_gridSize,
		ops::hls::SizeType& arg1_gridSize,
	    ap_uint<mem_data_width>* arg0_in,
	    ap_uint<mem_data_width>* arg1_out,
	    hls::stream<ap_axiu<axis_data_width,0,0,0>>& arg0_stream_out,
	    hls::stream<ap_axiu<axis_data_width,0,0,0>>& arg1_stream_in)
{
#pragma HLS DATAFLOW
    ops::hls::memReadGrid<mem_data_width, axis_data_width, data_width>(arg0_in, arg0_stream_out, arg0_gridSize, read_range);
//    ops::hls::memWriteGrid<mem_data_width, axis_data_width, data_width>(arg1_out, arg1_stream_in, arg1_gridSize, write_range);
    ops::hls::memWriteGridSimple<mem_data_width, axis_data_width, data_width>(arg1_out, arg1_stream_in, arg1_gridSize, write_range);
}

extern "C" void datamover_apply_stencil(
    const unsigned short read_range_start_x,
    const unsigned short read_range_end_x,
    const unsigned short read_range_start_y,
    const unsigned short read_range_end_y,
    const unsigned short write_range_start_x,
    const unsigned short write_range_end_x,
    const unsigned short write_range_start_y,
    const unsigned short write_range_end_y,
    const unsigned short arg0_gridSize_x,
    const unsigned short arg0_gridSize_y,
    const unsigned short arg1_gridSize_x,
    const unsigned short arg1_gridSize_y,
    ap_uint<mem_data_width>* arg0_in,
    ap_uint<mem_data_width>* arg1_out,
    hls::stream<ap_axiu<axis_data_width,0,0,0>>& arg0_stream_out,
    hls::stream<ap_axiu<axis_data_width,0,0,0>>& arg1_stream_in
)
{
    #pragma HLS INTERFACE s_axilite port = read_range_start_x bundle = control
	#pragma HLS INTERFACE s_axilite port = read_range_end_x bundle = control
	#pragma HLS INTERFACE s_axilite port = read_range_start_y bundle = control
	#pragma HLS INTERFACE s_axilite port = read_range_end_y bundle = control

    #pragma HLS INTERFACE s_axilite port = write_range_start_x bundle = control
	#pragma HLS INTERFACE s_axilite port = write_range_end_x bundle = control
	#pragma HLS INTERFACE s_axilite port = write_range_start_y bundle = control
	#pragma HLS INTERFACE s_axilite port = write_range_end_y bundle = control

	#pragma HLS INTERFACE s_axilite port = arg0_gridSize_x bundle = control
	#pragma HLS INTERFACE s_axilite port = arg0_gridSize_y bundle = control

    #pragma HLS INTERFACE s_axilite port = arg1_gridSize_x bundle = control
	#pragma HLS INTERFACE s_axilite port = arg1_gridSize_y bundle = control

    #pragma HLS INTERFACE mode=m_axi bundle=gmem0 depth=4096 max_read_burst_length=64 max_write_burst_length=64 \
        num_read_outstanding=4 num_write_outstanding=4 port=arg0_in offset=slave
    #pragma HLS INTERFACE mode=m_axi bundle=gmem1 depth=4096 max_read_burst_length=64 max_write_burst_length=64 \
        num_read_outstanding=4 num_write_outstanding=4 port=arg1_out offset=slave
	#pragma HLS INTERFACE s_axilite port = arg0_in bundle = control
    #pragma HLS INTERFACE s_axilite port = arg1_out bundle = control

    #pragma HLS INTERFACE mode=axis port=arg0_stream_out register
    #pragma HLS INTERFACE mode=axis port=arg1_stream_in register

	#pragma HLS INTERFACE ap_ctrl_chain port = return bundle = control
	#pragma HLS INTERFACE s_axilite port = return bundle = control



    ops::hls::AccessRange read_range;
    read_range.start[0] = read_range_start_x;
    read_range.start[1] = read_range_start_y;
    read_range.end[0] = read_range_end_x;
    read_range.end[1] = read_range_end_y;
    read_range.dim = 2;

    ops::hls::AccessRange write_range;
    write_range.start[0] = write_range_start_x;
    write_range.start[1] = write_range_start_y;
    write_range.end[0] = write_range_end_x;
    write_range.end[1] = write_range_end_y;
    write_range.dim = 2;

    ops::hls::SizeType arg0_gridSize = {arg0_gridSize_x, arg0_gridSize_y, 1};
    ops::hls::SizeType arg1_gridSize = {arg1_gridSize_x, arg1_gridSize_y, 1};

#ifdef DEBUG_LOG
            printf("[KERNEL_DEBUG]|%s| starting datamover TOP read_range:(%d,%d,%d) ---> (%d,%d,%d), write_range: (%d,%d,%d) ---> (%d,%d,%d)\n", __func__,
                    read_range.start[0],read_range.start[1], read_range.start[2], read_range.end[0], read_range.end[1], read_range.end[2],
                    write_range.start[0], write_range.start[1], write_range.start[2], write_range.end[0], write_range.end[1], write_range.end[2]);
            printf("[KERNEL_DEBUG]|%s| arg0_gridSize: (%d, %d, %d), arg1_gridSize(%d, %d, %d)\n", __func__,
                    arg0_gridSize[0], arg0_gridSize[1], arg0_gridSize[1],
                    arg1_gridSize[0], arg1_gridSize[1], arg1_gridSize[2]);
#endif

    datamover_apply_stencil_dataflow_region(read_range, write_range, arg0_gridSize, arg1_gridSize, arg0_in, arg1_out, arg0_stream_out, arg1_stream_in);

#ifdef DEBUG_LOG
			printf("[KERNEL_DEBUG]|%s| Ending datamover TOP\n", __func__);
#endif
}



