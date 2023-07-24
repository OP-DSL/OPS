#pragma once

#include "stencil2d.hpp"
#include <iostream>

void dut(ops::hls::GridPropertyCore& gridProp, Stencil2D & cross_stencil, stencil_type* data_in, stencil_type* data_out);
void kernel(stencil_type * coef, 
        ::hls::stream<stencil_type> input_bus_0[vector_factor],
		 ::hls::stream<stencil_type> input_bus_1[vector_factor],
		  ::hls::stream<stencil_type> input_bus_2[vector_factor],
		   ::hls::stream<stencil_type> input_bus_3[vector_factor],
			::hls::stream<stencil_type> input_bus_4[vector_factor],
			 ::hls::stream<stencil_type> output_bus[vector_factor],
			  ::hls::stream<stencil_type> alt_bus[vector_factor],
			   const unsigned int num_iter);
