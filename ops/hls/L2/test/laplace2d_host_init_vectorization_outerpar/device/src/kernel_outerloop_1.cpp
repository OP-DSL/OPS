#include <ops_hls_datamover.hpp>
#include "kernel_outerloop_1.hpp"


static void kernel_outerloop_1_dataflow_region(ops::hls::StencilConfigCore& stencilConfig, const unsigned int total_bytes,
        hls::stream <ap_axiu<axis_data_width,0,0,0>>& arg0_axis_in, hls::stream <ap_axiu<axis_data_width,0,0,0>>& arg1_axis_out)
{


    ::hls::stream<ap_uint<axis_data_width>> streams[iter_par_factor + 1];
#pragma HLS STREAM variable = streams depth = 10
//    static ::hls::stream<ap_uint<axis_data_width>> arg0_input_stream;
//    static ::hls::stream<ap_uint<axis_data_width>> arg1_output_stream;
//    static ::hls::stream<ap_uint<axis_data_width>> arg1_input_stream;

#pragma HLS DATAFLOW

    ops::hls::axis2stream<axis_data_width, axis_data_width> (arg0_axis_in, streams[0], total_bytes);

    for (int i = 0; i < iter_par_factor; i++)
    {
#pragma HLS UNROLL factor=iter_par_factor
    	kernel_apply_stencil_PE(stencilConfig, streams[i], streams[i+1]);
    }
//    kernel_copy_PE(s2d_1pt_stencilConfig, arg1_input_stream, arg1_output_stream);
    ops::hls::stream2axis<axis_data_width, axis_data_width> (arg1_axis_out, streams[iter_par_factor], total_bytes);

}

extern "C" void kernel_outerloop_1(
        const unsigned int outer_itr,
        const unsigned short stencilConfig_grid_size_x,
        const unsigned short stencilConfig_grid_size_y,
        const unsigned short stencilConfig_dim,
        const unsigned int stencilConfig_total_itr,
        const unsigned short stencilConfig_lower_limit_x,
        const unsigned short stencilConfig_lower_limit_y,
        const unsigned short stencilConfig_upper_limit_x,
        const unsigned short stencilConfig_upper_limit_y,
        const unsigned short stencilConfig_outer_loop_limit,
        hls::stream <ap_axiu<axis_data_width,0,0,0>>& arg0_axis_in,
        hls::stream <ap_axiu<axis_data_width,0,0,0>>& arg1_axis_out)
{
    #pragma HLS INTERFACE s_axilite port = outer_itr bundle = control
    #pragma HLS INTERFACE s_axilite port = stencilConfig_grid_size_x bundle = control
    #pragma HLS INTERFACE s_axilite port = stencilConfig_grid_size_y bundle = control
    #pragma HLS INTERFACE s_axilite port = stencilConfig_dim bundle = control
    #pragma HLS INTERFACE s_axilite port = stencilConfig_total_itr bundle = control
    #pragma HLS INTERFACE s_axilite port = stencilConfig_lower_limit_x bundle = control
    #pragma HLS INTERFACE s_axilite port = stencilConfig_lower_limit_y bundle = control
    #pragma HLS INTERFACE s_axilite port = stencilConfig_upper_limit_x bundle = control
    #pragma HLS INTERFACE s_axilite port = stencilConfig_upper_limit_y bundle = control
    #pragma HLS INTERFACE s_axilite port = stencilConfig_outer_loop_limit bundle = control

    #pragma HLS INTERFACE axis port = arg0_axis_in register
    #pragma HLS INTERFACE axis port = arg1_axis_out register

    #pragma HLS INTERFACE ap_ctrl_chain port = return
    #pragma HLS INTERFACE s_axilite port = return bundle = control


//#ifdef DEBUG_LOG
    printf("[KERNEL_DEBUG]|%s| Starting outerloop_1 kernel TOP \n", __func__);
//#endif

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

    unsigned int total_bytes = stencilConfig_total_itr * vector_factor * sizeof(stencil_type);

    for (unsigned int i = 0; i < outer_itr; i++)
        kernel_outerloop_1_dataflow_region(stencilConfig, total_bytes, arg0_axis_in, arg1_axis_out);
//#ifdef DEBUG_LOG
    printf("[KERNEL_DEBUG]|%s| Ending outerloop_1 kernel TOP \n", __func__);
//#endif
}
