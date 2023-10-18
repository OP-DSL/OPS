#pragma once

#include "common_config.hpp"
#include "stencil_s2d_1pt.hpp"

void kernel_right_bndcon_core(const unsigned int num_itr,
        ::hls::stream<stencil_type> arg0_output_bus_0[vector_factor],
        s2d_1pt::index_stream_dt idx_bus[vector_factor],
        const float const_pi,
        const int const_jmax)
{
    for (unsigned int itr = 0; itr < num_itr; itr++)
    {
#pragma HLS PIPELINE II=1
        for (unsigned int k = 0; k < vector_factor; k++)
        {
#pragma HLS UNROLL complete
            s2d_1pt::index_dt idx = idx_bus[k].read();
            ops::hls::IndexConv indexConv;
            indexConv.flatten = idx;
            float r1 = const_pi * (indexConv.index[1] + 1);
            float r2 = (const_jmax + 1);
            float r3 = r1 / r2;
            float r4 = sin(r3);
            float r5 = exp(-const_pi);
            float r = r4 * r5;
#ifdef DEBUG_LOG
        	printf("[KERNEL_DEBUG]|%s| writing to bus: %d, itr: %d, val: %f\n", __func__, k, itr, r);
#endif
            arg0_output_bus_0[k].write(r);
        }
    }
}

void kernel_right_bndcon_PE(ops::hls::GridPropertyCore& gridProp,
        s2d_1pt::widen_stream_dt& arg0_output_stream,
        s2d_1pt::mask_stream_dt& arg0_outmask_stream,
        const float const_pi,
        const int const_jmax)
{
#pragma HLS DATAFLOW
    s2d_1pt arg0_write_stencil;
    s2d_1pt index_read_stencil;
    arg0_write_stencil.setGridProp(gridProp);
    index_read_stencil.setGridProp(gridProp);

    static ::hls::stream<stencil_type> arg0_output_bus_0[vector_factor];
    #pragma HLS STREAM variable = arg0_output_bus_0 depth = max_depth_v8
    static s2d_1pt::index_stream_dt idx_bus[vector_factor];
    #pragma HLS STREAM variable = idx_bus depth = max_depth_v8

    unsigned int kernel_iterations = gridProp.outer_loop_limit * gridProp.xblocks;

    index_read_stencil.idxRead(idx_bus);
    kernel_right_bndcon_core(kernel_iterations, arg0_output_bus_0, idx_bus, const_pi, const_jmax);
    arg0_write_stencil.stencilWrite(arg0_output_stream, arg0_outmask_stream, arg0_output_bus_0);
} 

extern "C" void kernel_right_bndcon(
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
    const float const_pi,
    const int const_jmax,
    hls::stream <ap_axiu<axis_data_width,0,0,0>>& arg0_axis_out);
