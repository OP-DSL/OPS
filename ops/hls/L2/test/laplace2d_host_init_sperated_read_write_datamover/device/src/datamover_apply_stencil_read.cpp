
#include <ops_hls_datamover.hpp>
#include "../include/datamover_apply_stencil_read.hpp"

static void datamover_apply_stencil_read_dataflow_region(ops::hls::AccessRange& read_range,
		ops::hls::SizeType& arg0_gridSize,
	    ap_uint<data_width>* arg0_in,
	    hls::stream<ap_axiu<axis_data_width,0,0,0>>& arg0_stream_out)
{
#pragma HLS DATAFLOW
    ops::hls::memReadGrid<mem_data_width, axis_data_width, data_width>(arg0_in, arg0_stream_out, arg0_gridSize, read_range);
}

extern "C" void datamover_apply_stencil_read(
        const unsigned short range_start_x,
        const unsigned short range_end_x,
        const unsigned short range_start_y,
        const unsigned short range_end_y,
        const unsigned short arg0_gridSize_x,
        const unsigned short arg0_gridSize_y,
        ap_uint<data_width>* arg0_in,
        hls::stream<ap_axiu<axis_data_width,0,0,0>>& arg0_stream_out
)
{
    #pragma HLS INTERFACE s_axilite port = range_start_x bundle = control
	#pragma HLS INTERFACE s_axilite port = range_end_x bundle = control
	#pragma HLS INTERFACE s_axilite port = range_start_y bundle = control
	#pragma HLS INTERFACE s_axilite port = range_end_y bundle = control

	#pragma HLS INTERFACE s_axilite port = arg0_gridSize_x bundle = control
	#pragma HLS INTERFACE s_axilite port = arg0_gridSize_y bundle = control

    #pragma HLS INTERFACE mode=m_axi bundle=gmem0 depth=4096 max_read_burst_length=64 max_write_burst_length=64 \
        num_read_outstanding=4 num_write_outstanding=4 port=arg0_in offset=slave
	#pragma HLS INTERFACE s_axilite port = arg0_in bundle = control

    #pragma HLS INTERFACE mode=axis port=arg0_stream_out register

	#pragma HLS INTERFACE ap_ctrl_chain port = return
	#pragma HLS INTERFACE s_axilite port = return bundle = control


    ops::hls::AccessRange read_range;
    read_range.start[0] = range_start_x;
    read_range.start[1] = range_start_y;
    read_range.end[0] = range_end_x;
    read_range.end[1] = range_end_y;
    read_range.dim = 2;

    ops::hls::SizeType arg0_gridSize = {arg0_gridSize_x, arg0_gridSize_y, 1};

#ifdef DEBUG_LOG
            printf("[KERNEL_DEBUG]|%s| starting datamover TOP read_range:(%d,%d,%d) ---> (%d,%d,%d)\n", __func__,
                    read_range.start[0],read_range.start[1], read_range.start[2], read_range.end[0], read_range.end[1], read_range.end[2]);
            printf("[KERNEL_DEBUG]|%s| arg0_gridSize: (%d, %d, %d)\n", __func__,
                    arg0_gridSize[0], arg0_gridSize[1], arg0_gridSize[1]);
#endif

    datamover_apply_stencil_read_dataflow_region(read_range, arg0_gridSize, arg0_in, arg0_stream_out);

#ifdef DEBUG_LOG
			printf("[KERNEL_DEBUG]|%s| Ending datamover TOP\n", __func__);
#endif
}
