
#include <ops_hls_datamover.hpp>
#include "../include/kernel_apply_stencil.hpp"

static void kernel_apply_stencil_dataflow_region(ops::hls::StencilConfigCore& read_stencilConfig,
		ops::hls::StencilConfigCore& write_stencilConfig,
		const unsigned int& total_bytes,
		hls::stream <ap_axiu<axis_data_width,0,0,0>>& arg0_axis_in,
        hls::stream <ap_axiu<axis_data_width,0,0,0>>& arg1_axis_out,
	    s2d_1pt::widen_stream_dt& arg1_input_stream,
	    s2d_1pt::mask_stream_dt& arg1_inmask_stream,
	    s2d_1pt::widen_stream_dt& arg0_output_stream)
{
#pragma HLS_DATAFLOW
    ops::hls::axis2stream<axis_data_width, axis_data_width>(arg0_axis_in, arg0_output_stream, total_bytes);
    kernel_apply_stencil_PE(read_stencilConfig, write_stencilConfig, arg0_output_stream, arg1_input_stream, arg1_inmask_stream);
    ops::hls::stream2axisMasked<axis_data_width, axis_data_width>(arg1_axis_out, arg1_input_stream, arg1_inmask_stream, total_bytes);
}

extern "C" void kernel_apply_stencil(
        const unsigned short read_stencilConfig_grid_size_x,
        const unsigned short read_stencilConfig_grid_size_y,
        const unsigned short read_stencilConfig_lower_limit_x,
        const unsigned short read_stencilConfig_lower_limit_y,
        const unsigned short read_stencilConfig_upper_limit_x,
        const unsigned short read_stencilConfig_upper_limit_y,
        const unsigned short read_stencilConfig_dim,
        const unsigned short read_stencilConfig_outer_loop_limit,
        const unsigned int read_stencilConfig_total_itr,
        const unsigned short write_stencilConfig_grid_size_x,
        const unsigned short write_stencilConfig_grid_size_y,
        const unsigned short write_stencilConfig_lower_limit_x,
        const unsigned short write_stencilConfig_lower_limit_y,
        const unsigned short write_stencilConfig_upper_limit_x,
        const unsigned short write_stencilConfig_upper_limit_y,
        const unsigned short write_stencilConfig_dim,
        const unsigned short write_stencilConfig_outer_loop_limit,
        const unsigned int write_stencilConfig_total_itr,
        hls::stream <ap_axiu<axis_data_width,0,0,0>>& arg0_axis_in,
        hls::stream <ap_axiu<axis_data_width,0,0,0>>& arg1_axis_out)
{
    #pragma HLS INTERFACE s_axilite port = read_stencilConfig_grid_size_x bundle = control
    #pragma HLS INTERFACE s_axilite port = read_stencilConfig_grid_size_y bundle = control
    #pragma HLS INTERFACE s_axilite port = read_stencilConfig_lower_limit_x bundle = control
    #pragma HLS INTERFACE s_axilite port = read_stencilConfig_lower_limit_y bundle = control
    #pragma HLS INTERFACE s_axilite port = read_stencilConfig_upper_limit_x bundle = control
    #pragma HLS INTERFACE s_axilite port = read_stencilConfig_upper_limit_y bundle = control
    #pragma HLS INTERFACE s_axilite port = read_stencilConfig_dim bundle = control
    #pragma HLS INTERFACE s_axilite port = read_stencilConfig_total_itr bundle = control
    #pragma HLS INTERFACE s_axilite port = read_stencilConfig_outer_loop_limit bundle = control

    #pragma HLS INTERFACE s_axilite port = write_stencilConfig_grid_size_x bundle = control
    #pragma HLS INTERFACE s_axilite port = write_stencilConfig_grid_size_y bundle = control
    #pragma HLS INTERFACE s_axilite port = write_stencilConfig_lower_limit_x bundle = control
    #pragma HLS INTERFACE s_axilite port = write_stencilConfig_lower_limit_y bundle = control
    #pragma HLS INTERFACE s_axilite port = write_stencilConfig_upper_limit_x bundle = control
    #pragma HLS INTERFACE s_axilite port = write_stencilConfig_upper_limit_y bundle = control
    #pragma HLS INTERFACE s_axilite port = write_stencilConfig_dim bundle = control
    #pragma HLS INTERFACE s_axilite port = write_stencilConfig_total_itr bundle = control
    #pragma HLS INTERFACE s_axilite port = write_stencilConfig_outer_loop_limit bundle = control

    #pragma HLS INTERFACE axis port = arg0_axis_in register
    #pragma HLS INTERFACE axis port = arg1_axis_out register

    #pragma HLS INTERFACE ap_ctrl_chain port = return
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
    ops::hls::StencilConfigCore read_stencilConfig, write_stencilConfig;
    read_stencilConfig.dim = read_stencilConfig_dim;
    read_stencilConfig.grid_size[0] = read_stencilConfig_grid_size_x;
    read_stencilConfig.grid_size[1] = read_stencilConfig_grid_size_y;
    read_stencilConfig.lower_limit[0] = read_stencilConfig_lower_limit_x;
    read_stencilConfig.lower_limit[1] = read_stencilConfig_lower_limit_y;
    read_stencilConfig.upper_limit[0] = read_stencilConfig_upper_limit_x;
    read_stencilConfig.upper_limit[1] = read_stencilConfig_upper_limit_y;
    read_stencilConfig.total_itr = read_stencilConfig_total_itr;
    read_stencilConfig.outer_loop_limit = read_stencilConfig_outer_loop_limit;

    write_stencilConfig.dim = write_stencilConfig_dim;
    write_stencilConfig.grid_size[0] = write_stencilConfig_grid_size_x;
    write_stencilConfig.grid_size[1] = write_stencilConfig_grid_size_y;
    write_stencilConfig.lower_limit[0] = write_stencilConfig_lower_limit_x;
    write_stencilConfig.lower_limit[1] = write_stencilConfig_lower_limit_y;
    write_stencilConfig.upper_limit[0] = write_stencilConfig_upper_limit_x;
    write_stencilConfig.upper_limit[1] = write_stencilConfig_upper_limit_y;
    write_stencilConfig.total_itr = write_stencilConfig_total_itr;
    write_stencilConfig.outer_loop_limit = write_stencilConfig_outer_loop_limit;

    unsigned int total_bytes = read_stencilConfig_grid_size_x * vector_factor * read_stencilConfig_grid_size_y * sizeof(float);
    kernel_apply_stencil_dataflow_region(read_stencilConfig, write_stencilConfig, total_bytes, arg0_axis_in, arg1_axis_out, arg1_input_stream, arg1_inmask_stream, arg0_output_stream);

#ifdef DEBUG_LOG
			printf("[KERNEL_DEBUG]|%s| Ending stencil kernel TOP \n", __func__);
#endif
}
