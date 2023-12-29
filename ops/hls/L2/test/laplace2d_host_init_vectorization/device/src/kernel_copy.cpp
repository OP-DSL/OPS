
#include <ops_hls_datamover.hpp>
#include "../include/kernel_copy.hpp"

static void kernel_copy_dataflow_region(ops::hls::StencilConfigCore& stencilConfig,
		const unsigned int& total_bytes,
		hls::stream <ap_axiu<axis_data_width,0,0,0>>& arg0_axis_in,
        hls::stream <ap_axiu<axis_data_width,0,0,0>>& arg1_axis_out,
	    s2d_1pt::widen_stream_dt& arg1_input_stream,
	    s2d_1pt::mask_stream_dt& arg1_inmask_stream,
	    s2d_1pt::widen_stream_dt& arg0_output_stream)
{
#pragma HLS DATAFLOW
    ops::hls::axis2stream<axis_data_width, axis_data_width>(arg0_axis_in, arg0_output_stream, total_bytes);
    kernel_copy_PE(stencilConfig, arg0_output_stream, arg1_input_stream, arg1_inmask_stream);
    ops::hls::stream2axisMasked<axis_data_width, axis_data_width>(arg1_axis_out, arg1_input_stream, arg1_inmask_stream, total_bytes);
}

extern "C" void kernel_copy(
        const unsigned short stencilConfig_grid_size_x,
        const unsigned short stencilConfig_grid_size_y,
        const unsigned short stencilConfig_lower_limit_x,
        const unsigned short stencilConfig_lower_limit_y,
        const unsigned short stencilConfig_upper_limit_x,
        const unsigned short stencilConfig_upper_limit_y,
        const unsigned short stencilConfig_dim,
        const unsigned short stencilConfig_outer_loop_limit,
        const unsigned int stencilConfig_total_itr,
        hls::stream <ap_axiu<axis_data_width,0,0,0>>& arg0_axis_in,
        hls::stream <ap_axiu<axis_data_width,0,0,0>>& arg1_axis_out)
{
    #pragma HLS INTERFACE s_axilite port = stencilConfig_grid_size_x bundle = control
    #pragma HLS INTERFACE s_axilite port = stencilConfig_grid_size_y bundle = control
    #pragma HLS INTERFACE s_axilite port = stencilConfig_dim bundle = control
    #pragma HLS INTERFACE s_axilite port = stencilConfig_lower_limit_x bundle = control
    #pragma HLS INTERFACE s_axilite port = stencilConfig_lower_limit_y bundle = control
    #pragma HLS INTERFACE s_axilite port = stencilConfig_upper_limit_x bundle = control
    #pragma HLS INTERFACE s_axilite port = stencilConfig_upper_limit_y bundle = control
    #pragma HLS INTERFACE s_axilite port = stencilConfig_total_itr bundle = control
    #pragma HLS INTERFACE s_axilite port = stencilConfig_outer_loop_limit bundle = control

    #pragma HLS INTERFACE axis port = arg0_axis_in register
    #pragma HLS INTERFACE axis port = arg1_axis_out register

    #pragma HLS INTERFACE ap_ctrl_chain port = return bundle = control
    #pragma HLS INTERFACE s_axilite port = return bundle = control

    static s2d_1pt::widen_stream_dt arg1_input_stream;
    static s2d_1pt::mask_stream_dt arg1_inmask_stream;
    static s2d_1pt::widen_stream_dt arg0_output_stream;

    #pragma HLS STREAM variable = arg1_input_stream depth = max_depth_v8
    #pragma HLS STREAM variable = arg1_inmask_stream depth = max_depth_v8
    #pragma HLS STREAM variable = arg0_output_stream depth = max_depth_v8

    ops::hls::StencilConfigCore stencilConfig;
    stencilConfig.dim = stencilConfig_dim;
    stencilConfig.grid_size[0] = stencilConfig_grid_size_x;
    stencilConfig.grid_size[1] = stencilConfig_grid_size_y;
    stencilConfig.lower_limit[0] = stencilConfig_lower_limit_x;
    stencilConfig.lower_limit[1] = stencilConfig_lower_limit_y;
    stencilConfig.upper_limit[0] = stencilConfig_upper_limit_x;
    stencilConfig.upper_limit[1] = stencilConfig_upper_limit_y;
    stencilConfig.total_itr = stencilConfig_total_itr;
    stencilConfig.outer_loop_limit = stencilConfig_outer_loop_limit;

    unsigned int total_bytes = stencilConfig_grid_size_x * vector_factor * stencilConfig_grid_size_y * sizeof(float);
    kernel_copy_dataflow_region(stencilConfig, total_bytes, arg0_axis_in, arg1_axis_out, arg1_input_stream, arg1_inmask_stream, arg0_output_stream);
}
