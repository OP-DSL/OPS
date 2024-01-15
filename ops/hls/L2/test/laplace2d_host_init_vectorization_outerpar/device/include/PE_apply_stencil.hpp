#pragma once

#include "common_config.hpp"
#include "stencil_apply_stencil.hpp"


void kernel_apply_stencil_PE(ops::hls::StencilConfigCore& stencilConfig,
		::hls::stream<ap_uint<axis_data_width>>& arg0_input_stream,
		::hls::stream<ap_uint<axis_data_width>>& arg1_output_stream)
{
    Stencil_apply_stencil stencil;

#ifdef DEBUG_LOG
    printf("[KERNEL_DEBUG]|%s| stencil config gridSize: %d (xblocks), %d, %d\n", __func__, read_stencilConfig.grid_size[0], read_stencilConfig.grid_size[1], read_stencilConfig.grid_size[2]);
#endif
    stencil.setConfig(stencilConfig);

#ifdef DEBUG_LOG
    printf("[KERNEL_DEBUG]|%s| starting stencil kernel PE\n", __func__);
#endif

    stencil.stencilRun(arg0_input_stream,
    		arg1_output_stream);

#ifdef DEBUG_LOG
    printf("[KERNEL_DEBUG]|%s| Ending stencil kernel PE\n", __func__);
#endif
} 

// extern "C" void kernel_apply_stencil(
//         const unsigned short read_stencilConfig_grid_size_x,
//         const unsigned short read_stencilConfig_grid_size_y,
//         const unsigned short read_stencilConfig_lower_limit_x,
//         const unsigned short read_stencilConfig_lower_limit_y,
//         const unsigned short read_stencilConfig_upper_limit_x,
//         const unsigned short read_stencilConfig_upper_limit_y,
//         const unsigned short read_stencilConfig_dim,
//         const unsigned short read_stencilConfig_outer_loop_limit,
//         const unsigned int read_stencilConfig_total_itr,
//         const unsigned short write_stencilConfig_grid_size_x,
//         const unsigned short write_stencilConfig_grid_size_y,
//         const unsigned short write_stencilConfig_lower_limit_x,
//         const unsigned short write_stencilConfig_lower_limit_y,
//         const unsigned short write_stencilConfig_upper_limit_x,
//         const unsigned short write_stencilConfig_upper_limit_y,
//         const unsigned short write_stencilConfig_dim,
//         const unsigned short write_stencilConfig_outer_loop_limit,
//         const unsigned int write_stencilConfig_total_itr,
//         hls::stream <ap_axiu<axis_data_width,0,0,0>>& arg0_axis_in,
//         hls::stream <ap_axiu<axis_data_width,0,0,0>>& arg1_axis_out);
