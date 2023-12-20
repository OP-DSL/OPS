#pragma once

#include "common_config.hpp"
#include "stencil_s2d_5pt.hpp"
#include "stencil_s2d_1pt.hpp"

void kernel_apply_stencil_core(const unsigned int num_itr,
        ::hls::stream<stencil_type> arg0_input_bus_0[vector_factor],
        ::hls::stream<stencil_type> arg0_input_bus_1[vector_factor],
        ::hls::stream<stencil_type> arg0_input_bus_2[vector_factor],
        ::hls::stream<stencil_type> arg0_input_bus_3[vector_factor],
        ::hls::stream<stencil_type> arg0_input_bus_4[vector_factor],
	::hls::stream<bool> neg_cond_bus[vector_factor],
        ::hls::stream<stencil_type> arg1_output_bus_0[vector_factor])
{
	#ifdef DEBUG_LOG
	    printf("[KERNEL_INTERNAL]|%s| starting. iteration: %d\n",__func__, num_itr);
	#endif
    for (unsigned int itr = 0; itr < num_itr; itr++)
    {
#pragma HLS PIPELINE II=1
        for (unsigned int k = 0; k < vector_factor; k++)
        {
#pragma HLS UNROLL complete
            stencil_type r0 = arg0_input_bus_0[k].read();
            stencil_type r1 = arg0_input_bus_1[k].read();
            stencil_type r2 = arg0_input_bus_2[k].read();
            stencil_type r3 = arg0_input_bus_3[k].read();
            stencil_type r4 = arg0_input_bus_4[k].read();
            bool cond = neg_cond_bus[k].read();

            stencil_type r5 = r0 + r1;
            stencil_type r6 = r3 + r4;
            stencil_type r7 = r5 + r6;
            stencil_type r;
            if (not cond)
            	r = 0.25f * r7;
            else
            	r = r2;

#ifdef DEBUG_LOG
			printf("[KERNEL_INTERNAL]|%s| bus: %d, itr: %d, read_val: (%f, %f, %f, %f, %f) neg_cond: %d, write_val: (%f)\n",
					 __func__, k, itr, r0, r1, r2, r3, r4, cond, r);
#endif
            arg1_output_bus_0[k].write(r);
        }
    }

#ifdef DEBUG_LOG
    printf("[KERNEL_INTERNAL]|%s| exiting.\n",__func__);
#endif
}

static void kernel_apply_stencil_PE_dataflow_region(s2d_5pt& arg0_read_stencil, s2d_1pt& arg1_write_stencil, unsigned int& kernel_iterations,         s2d_5pt::widen_stream_dt& arg0_input_stream,
        s2d_1pt::widen_stream_dt& arg1_output_stream,
        s2d_1pt::mask_stream_dt& arg1_outmask_stream)
{
#pragma HLS DATAFLOW
    static ::hls::stream<stencil_type> arg0_input_bus_0[vector_factor];
    static ::hls::stream<stencil_type> arg0_input_bus_1[vector_factor];
    static ::hls::stream<stencil_type> arg0_input_bus_2[vector_factor];
    static ::hls::stream<stencil_type> arg0_input_bus_3[vector_factor];
    static ::hls::stream<stencil_type> arg0_input_bus_4[vector_factor];
    static ::hls::stream<bool> arg0_neg_cond_bus[vector_factor];

    static ::hls::stream<stencil_type> arg1_output_bus_0[vector_factor];

    #pragma HLS STREAM variable = arg0_input_bus_0 depth = max_depth_v8
    #pragma HLS STREAM variable = arg0_input_bus_1 depth = max_depth_v8
    #pragma HLS STREAM variable = arg0_input_bus_2 depth = max_depth_v8
    #pragma HLS STREAM variable = arg0_input_bus_3 depth = max_depth_v8
    #pragma HLS STREAM variable = arg0_input_bus_4 depth = max_depth_v8
    #pragma HLS STREAM variable = arg0_neg_cond_bus depth = max_depth_v8

    #pragma HLS STREAM variable = arg1_output_bus_0 depth = max_depth_v8

    arg0_read_stencil.stencilRead(arg0_input_stream, 
            arg0_input_bus_0,
            arg0_input_bus_1,
            arg0_input_bus_2,
            arg0_input_bus_3,
            arg0_input_bus_4,
			arg0_neg_cond_bus);
    kernel_apply_stencil_core(kernel_iterations,
            arg0_input_bus_0,
            arg0_input_bus_1,
            arg0_input_bus_2,
            arg0_input_bus_3,
            arg0_input_bus_4,
			arg0_neg_cond_bus,
            arg1_output_bus_0);
    arg1_write_stencil.stencilWrite(arg1_output_stream, 
            arg1_outmask_stream, 
            arg1_output_bus_0);
}

void kernel_apply_stencil_PE(ops::hls::GridPropertyCore& read_gridProp,
		ops::hls::GridPropertyCore& write_gridProp,
        s2d_5pt::widen_stream_dt& arg0_input_stream,
        s2d_1pt::widen_stream_dt& arg1_output_stream,
        s2d_1pt::mask_stream_dt& arg1_outmask_stream)
{
    s2d_5pt arg0_read_stencil;
    s2d_1pt arg1_write_stencil;

    arg0_read_stencil.setGridProp(read_gridProp);
    arg1_write_stencil.setGridProp(write_gridProp);

    unsigned int kernel_iterations = read_gridProp.total_itr;

#ifdef DEBUG_LOG
    printf("[KERNEL_DEBUG]|%s| starting stencil kernel PE\n", __func__);
#endif

    kernel_apply_stencil_PE_dataflow_region(arg0_read_stencil, arg1_write_stencil, kernel_iterations, arg0_input_stream, arg1_output_stream, arg1_outmask_stream);

#ifdef DEBUG_LOG
    printf("[KERNEL_DEBUG]|%s| Ending stencil kernel PE\n", __func__);
#endif
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
        hls::stream <ap_axiu<axis_data_width,0,0,0>>& arg1_axis_out);
