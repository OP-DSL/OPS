#pragma once

#include "common_config.hpp"
#include <ops_hls_stencil_core.hpp>
#include "stencil_s2d_1pt.hpp"

void poisson_kernel_initialguess(const unsigned int num_itr,
        ::hls::stream<stencil_type> output_u_bus_0[vector_factor],
        ::hls::stream<stencil_type> altout_u_bus_0[vector_factor])
{
    stencil_type r = 0.0;
    for (unsigned int itr = 0; itr < num_itr; itr++)
    {
        for (unsigned int k = 0; k < vector_factor; k++)
        {
            output_u_bus_0[k].write(r);
            altout_u_bus_0[k].write(r);
        }
    }
}