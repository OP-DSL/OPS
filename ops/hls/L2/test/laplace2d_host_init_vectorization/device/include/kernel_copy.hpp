#pragma once

#include "common_config.hpp"
#include "stencil_s2d_1pt.hpp"

void kernel_copy_core(const unsigned int num_itr,
        ::hls::stream<stencil_type> arg0_input_bus_0[vector_factor],
        ::hls::stream<stencil_type> arg1_output_bus_0[vector_factor])
{
    for (unsigned int itr = 0; itr < num_itr; itr++)
    {
#pragma HLS PIPELINE II=1
        for (unsigned short k = 0; k < vector_factor; k++)
        {
#pragma HLS UNROLL factor=vector_factor
            stencil_type rd_val = arg0_input_bus_0[k].read();
#ifdef DEBUG_LOG
        	printf("[KERNEL_DEBUG]|%s| writing to bus: %d, itr: %d, val: %f\n", __func__, k, itr, rd_val);
#endif
            arg1_output_bus_0[k].write(rd_val);
        }
    }
}

void kernel_copy_PE_datflow_region(s2d_1pt::widen_stream_dt& arg0_input_stream,
        s2d_1pt::widen_stream_dt& arg1_output_stream,
        s2d_1pt::mask_stream_dt& arg1_outmask_stream,
		s2d_1pt& arg0_read_stencil,
		s2d_1pt& arg1_write_stencil,
		::hls::stream<stencil_type> arg0_input_bus_0[vector_factor],
		::hls::stream<stencil_type> arg1_output_bus_0[vector_factor],
		unsigned int& kernel_iterations)
{
#pragma HLS DATAFLOW
    arg0_read_stencil.stencilRead(arg0_input_stream, arg0_input_bus_0);
    kernel_copy_core(kernel_iterations, arg0_input_bus_0, arg1_output_bus_0);
    arg1_write_stencil.stencilWrite(arg1_output_stream, arg1_outmask_stream, arg1_output_bus_0);
}

void kernel_copy_PE(ops::hls::StencilConfigCore& stencilConfig,
        s2d_1pt::widen_stream_dt& arg0_input_stream,
        s2d_1pt::widen_stream_dt& arg1_output_stream,
        s2d_1pt::mask_stream_dt& arg1_outmask_stream)
{
    s2d_1pt arg0_read_stencil;
    s2d_1pt arg1_write_stencil;

    arg0_read_stencil.setConfig(stencilConfig);
    arg1_write_stencil.setConfig(stencilConfig);

    static ::hls::stream<stencil_type> arg0_input_bus_0[vector_factor];
    static ::hls::stream<stencil_type> arg1_output_bus_0[vector_factor];
    #pragma HLS STREAM variable = arg0_input_bus_0 depth = max_depth_v8
    #pragma HLS STREAM variable = arg1_output_bus_0 depth = max_depth_v8

    unsigned int kernel_iterations = stencilConfig.total_itr;

    kernel_copy_PE_datflow_region(arg0_input_stream, arg1_output_stream, arg1_outmask_stream, arg0_read_stencil, arg1_write_stencil, arg0_input_bus_0, arg1_output_bus_0, kernel_iterations);
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
        hls::stream <ap_axiu<axis_data_width,0,0,0>>& arg1_axis_out);
