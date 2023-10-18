
#include <ops_hls_datamover.hpp>
#include "../include/kernel_apply_stencil.hpp"


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

    #pragma HLS INTERFACE ap_hls_chain port = return bundle = control
    #pragma HLS INTERFACE s_axilite port = return bundle = control

#pragma HLS DATAFLOW
    static s2d_1pt::widen_stream_dt arg1_input_stream;
    static s2d_1pt::mask_stream_dt arg1_inmask_stream;
    static s2d_5pt::widen_stream_dt arg0_output_stream;

    #pragma HLS STREAM variable = arg1_input_stream depth = max_depth_v8;
    #pragma HLS STREAM variable = arg1_inmask_stream depth = max_depth_v8;
    #pragma HLS STREAM variable = arg0_output_stream depth = max_depth_v8;

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

    unsigned int total_bytes = gridProp.grid_size[0] * gridProp.grid_size[1] * sizeof(stencil_type);

    ops::hls::axis2stream<axis_data_width, axis_data_width>(arg0_axis_in, arg0_output_stream, total_bytes);
    kernel_apply_stencil_PE(gridProp, arg0_output_stream, arg1_input_stream, arg1_inmask_stream);
    ops::hls::stream2axisMasked<axis_data_width, axis_data_width>(arg1_axis_out, arg1_input_stream, arg1_inmask_stream, total_bytes);
}
