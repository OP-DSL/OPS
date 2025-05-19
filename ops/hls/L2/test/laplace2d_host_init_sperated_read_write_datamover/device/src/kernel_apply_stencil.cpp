
#include <ops_hls_datamover.hpp>
#include "../include/kernel_apply_stencil.hpp"

static void kernel_apply_stencil_dataflow_region(ops::hls::GridPropertyCore& read_gridProp,
		ops::hls::GridPropertyCore& write_gridProp,
		const unsigned int& total_bytes_read,
		const unsigned int& total_bytes_write,
		hls::stream <ap_axiu<axis_data_width,0,0,0>>& arg0_axis_in,
        hls::stream <ap_axiu<axis_data_width,0,0,0>>& arg1_axis_out,
	    s2d_1pt::widen_stream_dt& arg1_input_stream,
	    s2d_1pt::mask_stream_dt& arg1_inmask_stream,
	    s2d_1pt::widen_stream_dt& arg0_output_stream)
{
#pragma HLS_DATAFLOW
    ops::hls::axis2stream<axis_data_width, axis_data_width>(arg0_axis_in, arg0_output_stream, total_bytes_read);
    kernel_apply_stencil_PE(read_gridProp, write_gridProp, arg0_output_stream, arg1_input_stream, arg1_inmask_stream);
    ops::hls::stream2axisMasked<axis_data_width, axis_data_width>(arg1_axis_out, arg1_input_stream, arg1_inmask_stream, total_bytes_write);
}

extern "C" void kernel_apply_stencil(
        const unsigned short read_gridProp_size_x,
        const unsigned short read_gridProp_size_y,
        const unsigned short read_gridProp_actual_size_x,
        const unsigned short read_gridProp_actual_size_y,
        const unsigned short read_gridProp_grid_size_x,
        const unsigned short read_gridProp_grid_size_y,
        const unsigned short read_gridProp_dim,
        const unsigned short read_gridProp_xblocks,
        const unsigned int read_gridProp_total_itr,
        const unsigned int read_gridProp_outer_loop_limit,
        const unsigned short write_gridProp_size_x,
        const unsigned short write_gridProp_size_y,
        const unsigned short write_gridProp_actual_size_x,
        const unsigned short write_gridProp_actual_size_y,
        const unsigned short write_gridProp_grid_size_x,
        const unsigned short write_gridProp_grid_size_y,
        const unsigned short write_gridProp_dim,
        const unsigned short write_gridProp_xblocks,
        const unsigned int write_gridProp_total_itr,
        const unsigned int write_gridProp_outer_loop_limit,
        hls::stream <ap_axiu<axis_data_width,0,0,0>>& arg0_axis_in,
        hls::stream <ap_axiu<axis_data_width,0,0,0>>& arg1_axis_out)
{
    #pragma HLS INTERFACE s_axilite port = read_gridProp_size_x bundle = control
    #pragma HLS INTERFACE s_axilite port = read_gridProp_size_y bundle = control
    #pragma HLS INTERFACE s_axilite port = read_gridProp_actual_size_x bundle = control
    #pragma HLS INTERFACE s_axilite port = read_gridProp_actual_size_y bundle = control
    #pragma HLS INTERFACE s_axilite port = read_gridProp_grid_size_x bundle = control
    #pragma HLS INTERFACE s_axilite port = read_gridProp_grid_size_y bundle = control
    #pragma HLS INTERFACE s_axilite port = read_gridProp_dim bundle = control
    #pragma HLS INTERFACE s_axilite port = read_gridProp_xblocks bundle = control
    #pragma HLS INTERFACE s_axilite port = read_gridProp_total_itr bundle = control
    #pragma HLS INTERFACE s_axilite port = read_gridProp_outer_loop_limit bundle = control

	#pragma HLS INTERFACE s_axilite port = write_gridProp_size_x bundle = control
	#pragma HLS INTERFACE s_axilite port = write_gridProp_size_y bundle = control
	#pragma HLS INTERFACE s_axilite port = write_gridProp_actual_size_x bundle = control
	#pragma HLS INTERFACE s_axilite port = write_gridProp_actual_size_y bundle = control
	#pragma HLS INTERFACE s_axilite port = write_gridProp_grid_size_x bundle = control
	#pragma HLS INTERFACE s_axilite port = write_gridProp_grid_size_y bundle = control
	#pragma HLS INTERFACE s_axilite port = write_gridProp_dim bundle = control
	#pragma HLS INTERFACE s_axilite port = write_gridProp_xblocks bundle = control
	#pragma HLS INTERFACE s_axilite port = write_gridProp_total_itr bundle = control
	#pragma HLS INTERFACE s_axilite port = write_gridProp_outer_loop_limit bundle = control

    #pragma HLS INTERFACE axis port = arg0_axis_in register
    #pragma HLS INTERFACE axis port = arg1_axis_out register

    #pragma HLS INTERFACE ap_ctrl_chain port = return bundle = control
    #pragma HLS INTERFACE s_axilite port = return bundle = control

    static s2d_1pt::widen_stream_dt arg1_input_stream;
    static s2d_1pt::mask_stream_dt arg1_inmask_stream;
    static s2d_5pt::widen_stream_dt arg0_output_stream;

    #pragma HLS STREAM variable = arg1_input_stream depth = max_depth_v8
    #pragma HLS STREAM variable = arg1_inmask_stream depth = max_depth_v8
    #pragma HLS STREAM variable = arg0_output_stream depth = max_depth_v8

#ifdef DEBUG_LOG
			printf("[KERNEL_DEBUG]|%s| Starting stencil kernel TOP \n", __func__);
#endif
    ops::hls::GridPropertyCore read_gridProp, write_gridProp;
    read_gridProp.dim = read_gridProp_dim;
    read_gridProp.size[0] = read_gridProp_size_x;
    read_gridProp.size[1] = read_gridProp_size_y;
    read_gridProp.actual_size[0] = read_gridProp_actual_size_x;
    read_gridProp.actual_size[1] = read_gridProp_actual_size_y;
    read_gridProp.grid_size[0] = read_gridProp_grid_size_x;
    read_gridProp.grid_size[1] = read_gridProp_grid_size_y;
    read_gridProp.xblocks = read_gridProp_xblocks;
    read_gridProp.total_itr = read_gridProp_total_itr;
    read_gridProp.outer_loop_limit = read_gridProp_outer_loop_limit;

    write_gridProp.dim = write_gridProp_dim;
    write_gridProp.size[0] = write_gridProp_size_x;
    write_gridProp.size[1] = write_gridProp_size_y;
    write_gridProp.actual_size[0] = write_gridProp_actual_size_x;
    write_gridProp.actual_size[1] = write_gridProp_actual_size_y;
    write_gridProp.grid_size[0] = write_gridProp_grid_size_x;
    write_gridProp.grid_size[1] = write_gridProp_grid_size_y;
    write_gridProp.xblocks = write_gridProp_xblocks;
    write_gridProp.total_itr = write_gridProp_total_itr;
    write_gridProp.outer_loop_limit = write_gridProp_outer_loop_limit;


    unsigned int total_bytes_read = read_gridProp_grid_size_x * read_gridProp_grid_size_y * sizeof(float);
    unsigned int total_bytes_write  = write_gridProp_grid_size_x * write_gridProp_grid_size_y * sizeof(float);
    kernel_apply_stencil_dataflow_region(read_gridProp, write_gridProp, total_bytes_read, total_bytes_write, arg0_axis_in, arg1_axis_out, arg1_input_stream, arg1_inmask_stream, arg0_output_stream);

#ifdef DEBUG_LOG
			printf("[KERNEL_DEBUG]|%s| Ending stencil kernel TOP \n", __func__);
#endif
}
