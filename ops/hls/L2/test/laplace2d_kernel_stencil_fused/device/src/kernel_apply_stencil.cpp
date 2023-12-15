
#include <ops_hls_datamover.hpp>
#include "../include/kernel_apply_stencil.hpp"

static void kernel_apply_stencil_dataflow_region(ops::hls::GridPropertyCore& gridProp,
		const unsigned int& total_bytes,
		hls::stream <ap_axiu<axis_data_width,0,0,0>>& arg0_axis_in,
        hls::stream <ap_axiu<axis_data_width,0,0,0>>& arg1_axis_out,
		kernel_apply_stencil_stencil::widen_stream_dt& arg1_input_stream,
		kernel_apply_stencil_stencil::widen_stream_dt& arg0_output_stream)
{
#pragma HLS DATAFLOW
    ops::hls::axis2stream<axis_data_width, axis_data_width>(arg0_axis_in, arg0_output_stream, total_bytes);
    kernel_apply_stencil_PE(gridProp, arg0_output_stream, arg1_input_stream);
    ops::hls::stream2axis<axis_data_width, axis_data_width>(arg1_axis_out, arg1_input_stream, total_bytes);
}

extern "C" void kernel_apply_stencil(
        const unsigned short gridProp_size_x,
        const unsigned short gridProp_size_y,
        const unsigned short gridProp_actual_size_x,
        const unsigned short gridProp_actual_size_y,
        const unsigned short gridProp_grid_size_x,
        const unsigned short gridProp_grid_size_y,
        const unsigned short gridProp_dim,
        const unsigned short gridProp_xblocks,
        const unsigned int gridProp_total_itr,
        const unsigned int gridProp_outer_loop_limit,
        hls::stream <ap_axiu<axis_data_width,0,0,0>>& arg0_axis_in,
        hls::stream <ap_axiu<axis_data_width,0,0,0>>& arg1_axis_out)
{
    #pragma HLS INTERFACE s_axilite port = gridProp_size_x bundle = control
    #pragma HLS INTERFACE s_axilite port = gridProp_size_y bundle = control
    #pragma HLS INTERFACE s_axilite port = gridProp_actual_size_x bundle = control
    #pragma HLS INTERFACE s_axilite port = gridProp_actual_size_y bundle = control
    #pragma HLS INTERFACE s_axilite port = gridProp_grid_size_x bundle = control
    #pragma HLS INTERFACE s_axilite port = gridProp_grid_size_y bundle = control
    #pragma HLS INTERFACE s_axilite port = gridProp_dim bundle = control
    #pragma HLS INTERFACE s_axilite port = gridProp_xblocks bundle = control
    #pragma HLS INTERFACE s_axilite port = gridProp_total_itr bundle = control
    #pragma HLS INTERFACE s_axilite port = gridProp_outer_loop_limit bundle = control

    #pragma HLS INTERFACE axis port = arg0_axis_in register
    #pragma HLS INTERFACE axis port = arg1_axis_out register

    #pragma HLS INTERFACE ap_ctrl_chain port = return
    #pragma HLS INTERFACE s_axilite port = return bundle = control

    static kernel_apply_stencil_stencil::widen_stream_dt arg1_input_stream;
    static kernel_apply_stencil_stencil::widen_stream_dt arg0_output_stream;

    #pragma HLS STREAM variable = arg1_input_stream depth = max_depth_v8
    #pragma HLS STREAM variable = arg0_output_stream depth = max_depth_v8

#ifdef DEBUG_LOG
			printf("[KERNEL_DEBUG]|%s| Starting stencil kernel TOP \n", __func__);
#endif
    ops::hls::GridPropertyCore gridProp;
    gridProp.dim = gridProp_dim;
    gridProp.size[0] = gridProp_size_x;
    gridProp.size[1] = gridProp_size_y;
    gridProp.actual_size[0] = gridProp_actual_size_x;
    gridProp.actual_size[1] = gridProp_actual_size_y;
    gridProp.grid_size[0] = gridProp_grid_size_x;
    gridProp.grid_size[1] = gridProp_grid_size_y;
    gridProp.xblocks = gridProp_xblocks;
    gridProp.total_itr = gridProp_total_itr;
    gridProp.outer_loop_limit = gridProp_outer_loop_limit;


    unsigned int total_bytes = gridProp_grid_size_x * gridProp_grid_size_y * sizeof(float);
    kernel_apply_stencil_dataflow_region(gridProp, total_bytes, arg0_axis_in, arg1_axis_out, arg1_input_stream, arg0_output_stream);

#ifdef DEBUG_LOG
			printf("[KERNEL_DEBUG]|%s| Ending stencil kernel TOP \n", __func__);
#endif
}
