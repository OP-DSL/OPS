/*
* Open source copyright declaration based on BSD open source template:
* http://www.opensource.org/licenses/bsd-license.php
*
* This file is part of the OPS distribution.
*
* Copyright (c) 2013, Mike Giles and others. Please see the AUTHORS file in
* the main source directory for a full list of copyright holders.
* All rights reserved.
*
* Redistribution and use in source and binary forms, with or without
* modification, are permitted provided that the following conditions are met:
* * Redistributions of source code must retain the above copyright
* notice, this list of conditions and the following disclaimer.
* * Redistributions in binary form must reproduce the above copyright
* notice, this list of conditions and the following disclaimer in the
* documentation and/or other materials provided with the distribution.
* * The name of Mike Giles may not be used to endorse or promote products
* derived from this software without specific prior written permission.
*
* THIS SOFTWARE IS PROVIDED BY Mike Giles ''AS IS'' AND ANY
* EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
* WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
* DISCLAIMED. IN NO EVENT SHALL Mike Giles BE LIABLE FOR ANY
* DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
* (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
* LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
* ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
* (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
* SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

/** @brief Test Application on intended for FPGA with multi kernel 
 *      dataflow with buffer propagation optimization.
 * @author Beniel Thileepan
  */

// standard headers
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

int grid_size_x, grid_size_y, logical_size_x, logical_size_y;
float k1, k2, k3, k4, k5, k6;
extern const unsigned short mem_vector_factor;

// OPS header file
#define OPS_2D
// #define VERIFICATION
// #define OPS_CPP_API
#define OPS_HLS_V2
// #define OPS_FPGA

#include <ops_seq_v2.h>
#include "stencil_kernel.h"
#include "stencil_cpu_verification.hpp"


/******************************************************************************
* Main program
*******************************************************************************/
int main(int argc, const char **argv)
{
  /**-------------------------- Initialisation --------------------------**/

    // OPS initialisation
    ops_init(argc,argv,1);


    //Mesh
    int imax = 20;
    int jmax = 20;
    unsigned int iter_max = 135;
    unsigned int batches = 1;

    const char* pch;
    for ( int n = 1; n < argc; n++ ) 
    {
        pch = strstr(argv[n], "-sizex=");

        if(pch != NULL) {
            imax = atoi ( argv[n] + 7 ); continue;
        }
        pch = strstr(argv[n], "-sizey=");

        if(pch != NULL) {
            jmax = atoi ( argv[n] + 7 ); continue;
        }
        pch = strstr(argv[n], "-iters=");

        if(pch != NULL) {
            iter_max = atoi ( argv[n] + 7 ); continue;
        }
        pch = strstr(argv[n], "-batch=");

        if(pch != NULL) {
            batches = atoi ( argv[n] + 7 ); continue;
        }
    }

    //The 2D block
    ops_block block;

    block = ops_decl_block(2, "block");

    //declare stencils
    int s2D_00[] = {0,0};
    ops_stencil S2D_00 = ops_decl_stencil(2, 1, s2D_00, "00");
    int s2D_00_P10_M10_0P1_0M1[] = {0,0, 1,0, -1,0, 0,1, 0,-1};
    ops_stencil S2D_5PT = ops_decl_stencil(2, 5, s2D_00_P10_M10_0P1_0M1, "00:10:-10:01:0-1");

    //declare datasets
    int size[] = {imax, jmax};
    int base[] = {0,0};
    int d_m[] = {-1,-1};
    int d_p[] = {1,1};
    float* temp = NULL;

    int grid_size_y = size[1] - d_m[1] + d_p[1];
#ifdef OPS_FPGA
    int grid_size_x = ((size[0] - d_m[0] + d_p[0] + mem_vector_factor - 1) / mem_vector_factor) * mem_vector_factor;
#else
    int grid_size_x = size[0] - d_m[0] + d_p[0];
#endif

    logical_size_x = size[0] - d_m[0] + d_p[0];
    logical_size_y = size[1] - d_m[1] + d_p[1];
    ops_decl_const("size_x", 1, "int", &logical_size_x);
    ops_decl_const("size_y", 1, "int", &logical_size_y);

    ops_dat d0;
    ops_dat d1;
    ops_dat d2;
    ops_dat d3;
    ops_dat d4;
    ops_dat d5;
    // ops_dat a;
    // ops_dat b;

#ifdef VERIFICATION
    float* d0_cpu;
    float* d1_cpu;
    float* d2_cpu;
    float* a_cpu;
    float* b_cpu;
#endif

    // Allocation
    std::string name = std::string("d0");
    d0 = ops_decl_dat(block, 1, size, base, d_m, d_p, temp, "float", name.c_str());
    name = std::string("d1");
    d1 = ops_decl_dat(block, 1, size, base, d_m, d_p, temp, "float", name.c_str());
    name = std::string("d2");
    d2 = ops_decl_dat(block, 1, size, base, d_m, d_p, temp, "float", name.c_str());
    name = std::string("d3");
    d3 = ops_decl_dat(block, 1, size, base, d_m, d_p, temp, "float", name.c_str());
    name = std::string("d4");
    d4 = ops_decl_dat(block, 1, size, base, d_m, d_p, temp, "float", name.c_str());
    name = std::string("d5");
    d5 = ops_decl_dat(block, 1, size, base, d_m, d_p, temp, "float", name.c_str());
    // name = std::string("a");
    // a = ops_decl_dat(block, 1, size, base, d_m, d_p, temp, "float", name.c_str());
    // name = std::string("b");
    // b = ops_decl_dat(block, 1, size, base, d_m, d_p, temp, "float", name.c_str());

#ifdef VERIFICATION
    d0_cpu = (float*)malloc(sizeof(float) * grid_size_x * grid_size_y);
    d1_cpu = (float*)malloc(sizeof(float) * grid_size_x * grid_size_y);
    d2_cpu = (float*)malloc(sizeof(float) * grid_size_x * grid_size_y);
    a_cpu = (float*)malloc(sizeof(float) * grid_size_x * grid_size_y);
    b_cpu = (float*)malloc(sizeof(float) * grid_size_x * grid_size_y);
#endif
    ops_partition("");

    int full_range[] = {d_m[0], size[0] + d_p[0], d_m[1], size[1] + d_p[1]};
    int internal_range[] = {0, size[0], 0, size[1]};

    // init 
    // ops_par_loop(kernel_init_zero, "init_d0", block, 2, full_range,
    //         ops_arg_dat(d0, 1, S2D_00, "float", OPS_WRITE));
    
    // float init_const = 1;
    // ops_par_loop(kernel_const_init, "init_d0_step2", block, 2, internal_range,
    //         ops_arg_gbl(&init_const, 1, "float", OPS_READ),
    //         ops_arg_dat(d0, 1, S2D_00, "float", OPS_WRITE));
    // ops_par_loop(kernel_idx_init, "init_d0_step2", block, 2, full_range,
    //         ops_arg_idx(),
    //         ops_arg_dat(d0, 1, S2D_00, "float", OPS_WRITE));

    // ops_par_loop(copy, "copy_1", block, 2, full_range, 
    //         ops_arg_dat(d0, 1, S2D_00, "float", OPS_READ),
    //         ops_arg_dat(d2, 1, S2D_00, "float", OPS_WRITE));

    // float const_a = 1;
    // float const_b = 1.1;
    // // init a and b
    // ops_par_loop(kernel_const_init, "init_a", block, 2, full_range,
    //         ops_arg_gbl(&const_a, 1, "float", OPS_READ),
    //         ops_arg_dat(a, 1, S2D_00, "float", OPS_WRITE));
    // ops_par_loop(kernel_const_init, "init_b", block, 2, full_range,
    //         ops_arg_gbl(&const_b, 1, "float", OPS_READ),
    //         ops_arg_dat(b, 1, S2D_00, "float", OPS_WRITE));

#ifdef VERIFICATION
    init_a_b_cpu(a_cpu, b_cpu, const_a, const_b, size, d_m, d_p, full_range);
    init_zero_cpu(d0_cpu, size, d_m, d_p, full_range);
    // init_const_cpu(init_const, d0_cpu, size, d_m, d_p, internal_range);
    init_index_cpu(d0_cpu, size, d_m, d_p, full_range);
    copy_cpu(d0_cpu, d2_cpu, size, d_m, d_p, full_range);

    auto d0_raw = (float*)ops_dat_get_raw_pointer(d0, 0, S2D_00, OPS_HOST);
    auto a_raw = (float*)ops_dat_get_raw_pointer(a, 0, S2D_00, OPS_HOST);
    auto b_raw = (float*)ops_dat_get_raw_pointer(b, 0, S2D_00, OPS_HOST);

    // printGrid2D(d0_raw, d0.originalProperty, "d0_before_calc");
    // printGrid2D(d0_cpu, d0.originalProperty, "d0_cpu_before_calc");
    // printGrid2D(a_raw, a.originalProperty, "a_before_calc");
    // printGrid2D(b_raw, b.originalProperty, "b_before_calc");

#endif
    // iterative stencil loop
    ops_printf("Launching stencil calculation: %d x %d mesh\n", size[0], size[1]);

#ifdef OPS_FPGA
    #pragma ISL "isl0" iter_max
#endif
    for (int iter = 0; iter < iter_max; iter++)
    {
        ops_par_loop(kernel_1, "kernel_1", block, 2, internal_range,
                ops_arg_dat(d0, 1, S2D_5PT, "float", OPS_READ),
                ops_arg_dat(d1, 1, S2D_5PT, "float", OPS_READ),
                ops_arg_dat(d2, 1, S2D_00, "float", OPS_WRITE),
                ops_arg_dat(d3, 1, S2D_00, "float", OPS_WRITE));
        
        ops_par_loop(kernel_2, "kernel_2", block, 2, internal_range,
                ops_arg_dat(d2, 1, S2D_5PT, "float", OPS_READ),
                ops_arg_dat(d3, 1, S2D_5PT, "float", OPS_READ),
                ops_arg_dat(d4, 1, S2D_00, "float", OPS_WRITE),
                ops_arg_dat(d5, 1, S2D_00, "float", OPS_WRITE));

        ops_par_loop(copy, "copy_2", block, 2, internal_range,
                ops_arg_dat(d4, 1, S2D_00, "float", OPS_READ),
                ops_arg_dat(d0, 1, S2D_00, "float", OPS_WRITE));
        
        ops_par_loop(copy, "copy_3", block, 2, internal_range,
                ops_arg_dat(d5, 1, S2D_00, "float", OPS_READ),
                ops_arg_dat(d1, 1, S2D_00, "float", OPS_WRITE));
    }

    //Final Verification after calc
#ifdef VERIFICATION

        auto d0_raw = (float*)ops_dat_get_raw_pointer(d0, 0, S2D_00, OPS_HOST);
        auto d2_raw = (float*)ops_dat_get_raw_pointer(d2, 0, S2D_00, OPS_HOST);

        for (int iter = 0; iter < iter_max; iter++)
        {
            kernel_1_cpu(a_cpu, d0_cpu, d1_cpu, size, d_m, d_p, internal_range);
            kernel_2_cpu(b_cpu, d0_cpu, d1_cpu, d2_cpu, size, d_m, d_p, internal_range);
            copy_cpu(d2_cpu, d0_cpu, size, d_m, d_p, internal_range);
        }

        // printGrid2D(d0_2_raw, d0_2.originalProperty, "d0_after_calc");
        // printGrid2D(d0_2_cpu, d0_2.originalProperty, "d0_cpu_after_calc");
        // printGrid2D(d1_2_raw, d1_2.originalProperty, "d1_after_calc");
        // printGrid2D(d1_2_cpu, d1_2.originalProperty, "d1_cpu_after_calc");

        if(verify(d0_raw, d0_cpu, size, d_m, d_p, internal_range))
            std::cout << "verification of d0 after calculation" << "[PASSED]" << std::endl;
        else
            std::cout << "verification of d0 after calculation" << "[FAILED]" << std::endl;

                if(verify(d2_raw, d2_cpu, size, d_m, d_p, internal_range))
            std::cout << "verification of d1 after calculation" << "[PASSED]" << std::endl;
        else
            std::cout << "verification of d1 after calculation" << "[FAILED]" << std::endl;
#endif

    //cleaning
    ops_free_dat(d0);
    ops_free_dat(d1);
    ops_free_dat(d2);
    ops_free_dat(d3);
    ops_free_dat(d4);
    ops_free_dat(d5);

#ifdef VERIFICATION
    free(d0_cpu);
    free(d1_cpu);
    free(d2_cpu);
#endif

    ops_exit();

    std::cout << "Exit properly" << std::endl;
    return 0;
}
