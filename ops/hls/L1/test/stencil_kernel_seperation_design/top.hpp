#pragma once

#include "stencil2d.hpp"
#include <iostream>

void dut(const unsigned short gridProp_size_x,
	    const unsigned short gridProp_size_y,
	    const unsigned short gridProp_actual_size_x,
	    const unsigned short gridProp_actual_size_y,
	    const unsigned short gridProp_grid_size_x,
	    const unsigned short gridProp_grid_size_y,
	    const unsigned short gridProp_dim,
	    const unsigned short gridProb_xblocks,
	    const unsigned int gridProp_total_itr,
	    const unsigned int gridProp_outer_loop_limit,
		stencil_type* data_in,
		stencil_type* data_out);

void kernel(stencil_type * coef, 
        ::hls::stream<stencil_type> input_bus_0[vector_factor],
		 ::hls::stream<stencil_type> input_bus_1[vector_factor],
		  ::hls::stream<stencil_type> input_bus_2[vector_factor],
		   ::hls::stream<stencil_type> input_bus_3[vector_factor],
			::hls::stream<stencil_type> input_bus_4[vector_factor],
			 ::hls::stream<stencil_type> output_bus[vector_factor],
			  ::hls::stream<stencil_type> alt_bus[vector_factor],
			   const unsigned int num_iter);
