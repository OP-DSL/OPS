#pragma once

#include "common_config.hpp"
#include "stencil_s2d_1pt.hpp"

void kernel_set_zero_core(const unsigned int num_itr,
        ::hls::stream<stencil_type> arg0_output_bus_0[vector_factor])
{
    for (unsigned int itr = 0; itr < num_itr; itr++)
    {
#pragma HLS PIPELINE II=1
        for (unsigned int k = 0; k < vector_factor; k++)
        {
#pragma HLS UNROLL complete

            arg0_output_bus_0[k].write(0.0);
        }
    }
}

void kernel_set_zero_PE(ops::hls::GridPropertyCore& gridProp,
        s2d_1pt::widen_stream_dt& arg0_output_stream,
        s2d_1pt::mask_stream_dt& arg0_outmask_stream)
{
#pragma HLS DATAFLOW
    s2d_1pt arg0_write_stencil;

    arg0_write_stencil.setGridProp(gridProp);

    static ::hls::stream<stencil_type> arg0_output_bus_0[vector_factor];
    #pragma HLS STREAM variable = arg0_output_bus_0 depth = max_depth_v8

    unsigned int kernel_iterations = gridProp.outer_loop_limit * gridProp.xblocks;

    kernel_set_zero_core(kernel_iterations, arg0_output_bus_0);
    arg0_write_stencil.stencilWrite(arg0_output_stream, arg0_outmask_stream, arg0_output_bus_0);
} 

extern "C" void kernel_set_zero(
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
    hls::stream <ap_axiu<axis_data_width,0,0,0>>& arg0_axis_out);
