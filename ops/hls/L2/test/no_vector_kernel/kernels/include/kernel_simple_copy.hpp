#pragma once

#include "common_config.hpp"
#include <ops_hls_stencil_core.hpp>
#include "stencil_s2d_1pt_no_vect.hpp"
#include <stdio.h>

void kernel_simple_copy_core(const unsigned int num_itr,
        const float& const_val,
        ::hls::stream<stencil_type> output_u_bus_0[vector_factor])
{
    stencil_type r = const_val;

    for (unsigned int itr = 0; itr < num_itr; itr++)
    {
#pragma HLS PIPELINE II=1
        for (unsigned int k = 0; k < vector_factor; k++)
        {
#pragma HLS UNROLL complete
#ifdef DEBUG_LOG
        	printf("[KERNEL_DEBUG]|%s| writing to bus: %d, itr: %d, val: %f\n", __func__, k, itr, r);
#endif
            output_u_bus_0[k].write(r);
        }
    }
#ifdef DEBUG_LOG
    printf("[KERNEL_DEBUG]|%s| exiting.", __func__);
#endif
}

void kernel_simple_copy_PE(ops::hls::GridPropertyCore& gridProp,
        const float& const_val,
        s2d_1pt_no_vect::widen_stream_dt& output_stream_u,
		s2d_1pt_no_vect::mask_stream_dt& mask_stream_u)
{
    s2d_1pt_no_vect write_stencil_u;
#ifdef DEBUG_LOG
    printf("[KERNEL_DEBUG]|%s| setting grid property\n", __func__);
#endif
    write_stencil_u.setGridProp(gridProp);

    static ::hls::stream<stencil_type> output_u_bus_0[vector_factor];
	#pragma HLS STREAM variable = output_u_bus_0 depth = max_depth_v8

    unsigned int kernel_iterations = gridProp.outer_loop_limit * gridProp.xblocks;
    
#ifdef DEBUG_LOG
    printf("[KERNEL_DEBUG]|%s| launching kernel_simple_copy_core with iterations: %d\n", __func__, kernel_iterations);
#endif
    kernel_simple_copy_core(kernel_iterations, const_val, output_u_bus_0);

#ifdef DEBUG_LOG
    printf("[KERNEL_DEBUG]|%s| launching stencil write\n", __func__);
#endif
    write_stencil_u.stencilWrite(output_stream_u, mask_stream_u, output_u_bus_0);
#ifdef DEBUG_LOG
    printf("[KERNEL_DEBUG]|%s| exiting.", __func__);
#endif
}

extern "C" void kernel_simple_copy(
    const float const_val,
    const unsigned short gridProp_size_x,
    const unsigned short gridProp_size_y,
    const unsigned short gridProp_actual_size_x,
    const unsigned short gridProp_actual_size_y,
    const unsigned short gridProp_grid_size_x,
    const unsigned short gridProp_grid_size_y,
    const unsigned short gridProp_dim,
    const unsigned short gridProb_xblocks,
    const unsigned int gridProp_total_itr,
    const unsigned int gridProp_outer_loop_limit,
	hls::stream <ap_axiu<axis_data_width,0,0,0>> &axis_out_u
);
