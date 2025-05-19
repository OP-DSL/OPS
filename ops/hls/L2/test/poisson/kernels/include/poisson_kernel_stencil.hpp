#pragma once

#include "common_config.hpp"
#include <ops_hls_stencil_core.hpp>
#include "stencil_s2d_1pt.hpp"
#include "stencil_s2d_5pt.hpp"

void poisson_kernel_stencil_core(const unsigned int num_itr,
        ::hls::stream<stencil_type> input_u_bus_0[vector_factor],
        ::hls::stream<stencil_type> input_u_bus_1[vector_factor],
        ::hls::stream<stencil_type> input_u_bus_2[vector_factor],
        ::hls::stream<stencil_type> input_u_bus_3[vector_factor],
        ::hls::stream<stencil_type> input_u_bus_4[vector_factor],
        ::hls::stream<stencil_type> input_f_bus_0[vector_factor],
        ::hls::stream<stencil_type> output_u2_bus_0[vector_factor],
        ::hls::stream<stencil_type> altout_u2_bus_0[vector_factor])
{
    for (unsigned int itr = 0; itr < num_itr; itr++)
    {
        for (unsigned int k = 0; k < vector_factor; k++)
        {
            stencil_type r1 = input_u_bus_0[k].read();
            stencil_type r2 = input_u_bus_1[k].read();
            stencil_type r3 = input_u_bus_2[k].read();
            stencil_type r4 = input_u_bus_3[k].read();
            stencil_type r5 = input_u_bus_4[k].read();
            stencil_type r6 = input_f_bus_0[k].read();

            stencil_type r7 = r2 + r4;
            stencil_type r8_1 = r7 * dy;
            stencil_type r8 = r8_1 * dy;
            stencil_type r9 = r1 + r5;
            stencil_type r10_1 = r9 * dx;
            stencil_type r10 = r10_1 * dx;
            stencil_type r11_1 = r6 * dx;
            stencil_type r11_2 = r11_1 * dx;
            stencil_type r11_3 = r11_2 * dy;
            stencil_type r11 = r11_3 * dy;
            stencil_type r12 = r8 + r10;
            stencil_type r13 = r12 - r11;
            stencil_type r14_1 = dx * dx;
            stencil_type r14_2 = dy * dy;
            stencil_type r14 = r14_1 + r14_2;
            stencil_type r15 = r14 * 2.0;
            
            stencil_type r = r3 / r15;

            output_u2_bus_0[k].write(r);
            altout_u2_bus_0[k].write(r3);
        }
    }
}

void poission_kernel_stencil_PE(ops::hls::GridPropertyCore& gridProp,
        s2d_5pt::stream_dt& input_hls_stream_u,
        s2d_1pt::stream_dt& input_hls_stream_f,
        s2d_1pt::stream_dt& output_hls_stream_u2)
{
    s2d_5pt read_stencil_u;
    s2d_1pt read_stencil_f;
    s2d_1pt write_stencil_u2;

    read_stencil_u.setGridProp(gridProp);
    read_stencil_f.setGridProp(gridProp);
    write_stencil_u2.setGridProp(gridProp);

    static ::hls::stream<stencil_type> input_u_bus_0[vector_factor];
    static ::hls::stream<stencil_type> input_u_bus_1[vector_factor];
    static ::hls::stream<stencil_type> input_u_bus_2[vector_factor];
    static ::hls::stream<stencil_type> input_u_bus_3[vector_factor];
    static ::hls::stream<stencil_type> input_u_bus_4[vector_factor];
    static ::hls::stream<stencil_type> input_f_bus_0[vector_factor];
    static ::hls::stream<stencil_type> output_u2_bus_0[vector_factor];
    static ::hls::stream<stencil_type> altout_u2_bus_0[vector_factor];

    unsigned int kernel_iterations = gridProp.total_itr;

    #pragma HLS DATAFLOW
    read_stencil_u.stencilRead(input_hls_stream_u,
        input_u_bus_0,
        input_u_bus_1,
        input_u_bus_2,
        input_u_bus_3,
        input_u_bus_4);
    
    read_stencil_f.stencilRead(input_hls_stream_f,
        input_f_bus_0);

    poisson_kernel_stencil_core(kernel_iterations,
        input_u_bus_0,
        input_u_bus_1,
        input_u_bus_2,
        input_u_bus_3,
        input_u_bus_4,
        input_f_bus_0,
        output_u2_bus_0,
        altout_u2_bus_0);
    
    write_stencil_u2.stencilWrite(output_hls_stream_u2,
        output_u2_bus_0,
        altout_u2_bus_0);
}