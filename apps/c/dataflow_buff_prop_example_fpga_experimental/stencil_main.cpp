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

/** @Test application for fpga batched temporal blocked stencil 2D
  * @author Gihan Mudalige, Istvan Reguly, Beniel Thileepan
  */

// standard headers
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

int grid_size_x, grid_size_y, logical_size_x, logical_size_y;
extern const unsigned short mem_vector_factor;

// OPS header file
#define OPS_2D
#define VERIFICATION
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
    ops_stencil S2D_00_P10_M10_0P1_0M1 = ops_decl_stencil(2, 5, s2D_00_P10_M10_0P1_0M1, "00:10:-10:01:0-1");

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

    ops_dat dat0;
    ops_dat dat1;
    ops_dat dat2;
    ops_dat dat3;
    ops_dat dat0_2;
    ops_dat dat1_2;
    ops_dat dat_a;
    ops_dat dat_b;

#ifdef VERIFICATION
    float* dat0_cpu;
    float* dat1_cpu;
    float* dat2_cpu;
    float* dat3_cpu;
    float* dat0_2_cpu;
    float* dat1_2_cpu;
    float* a_cpu;
    float* b_cpu;
#endif

    // Allocation
    std::string name = std::string("dat0");
    dat0 = ops_decl_dat(block, 1, size, base, d_m, d_p, temp, "float", name.c_str());
    name = std::string("dat1");
    dat1 = ops_decl_dat(block, 1, size, base, d_m, d_p, temp, "float", name.c_str());
    name = std::string("dat2");
    dat2 = ops_decl_dat(block, 1, size, base, d_m, d_p, temp, "float", name.c_str());
    name = std::string("dat3");
    dat3 = ops_decl_dat(block, 1, size, base, d_m, d_p, temp, "float", name.c_str());
    name = std::string("dat0_2");
    dat0_2 = ops_decl_dat(block, 1, size, base, d_m, d_p, temp, "float", name.c_str());
    name = std::string("dat1_2");
    dat1_2 = ops_decl_dat(block, 1, size, base, d_m, d_p, temp, "float", name.c_str());
    name = std::string("a");
    dat_a = ops_decl_dat(block, 1, size, base, d_m, d_p, temp, "float", name.c_str());
    name = std::string("b");
    dat_b = ops_decl_dat(block, 1, size, base, d_m, d_p, temp, "float", name.c_str());

#ifdef VERIFICATION
    dat0_cpu = (float*)malloc(sizeof(float) * grid_size_x * grid_size_y);
    dat1_cpu = (float*)malloc(sizeof(float) * grid_size_x * grid_size_y);
    dat2_cpu = (float*)malloc(sizeof(float) * grid_size_x * grid_size_y);
    dat3_cpu = (float*)malloc(sizeof(float) * grid_size_x * grid_size_y);
    dat0_2_cpu = (float*)malloc(sizeof(float) * grid_size_x * grid_size_y);
    dat1_2_cpu = (float*)malloc(sizeof(float) * grid_size_x * grid_size_y);
    a_cpu = (float*)malloc(sizeof(float) * grid_size_x * grid_size_y);
    b_cpu = (float*)malloc(sizeof(float) * grid_size_x * grid_size_y);
#endif
    ops_partition("");

    int full_range[] = {d_m[0], size[0] + d_p[0], d_m[1], size[1] + d_p[1]};
    int internal_range[] = {0, size[0], 0, size[1]};

    // init 
    ops_par_loop(kernel_init_zero, "init_dat0", block, 2, full_range,
            ops_arg_dat(dat0, 1, S2D_00, "float", OPS_WRITE));
    
    // float init_const = 1;
    // ops_par_loop(kernel_const_init, "init_dat0_step2", block, 2, internal_range,
    //         ops_arg_gbl(&init_const, 1, "float", OPS_READ),
    //         ops_arg_dat(dat0, 1, S2D_00, "float", OPS_WRITE));
    ops_par_loop(kernel_idx_init, "init_dat0_step2", block, 2, full_range,
            ops_arg_idx(),
            ops_arg_dat(dat0, 1, S2D_00, "float", OPS_WRITE));

    ops_par_loop(copy, "copy_1", block, 2, full_range, 
            ops_arg_dat(dat0, 1, S2D_00, "float", OPS_READ),
            ops_arg_dat(dat1, 1, S2D_00, "float", OPS_WRITE));

    float const_a = 1;
    float const_b = 1.1;
    // init a and b
    ops_par_loop(kernel_const_init, "init_a", block, 2, full_range,
            ops_arg_gbl(&const_a, 1, "float", OPS_READ),
            ops_arg_dat(dat_a, 1, S2D_00, "float", OPS_WRITE));
    ops_par_loop(kernel_const_init, "init_b", block, 2, full_range,
            ops_arg_gbl(&const_b, 1, "float", OPS_READ),
            ops_arg_dat(dat_b, 1, S2D_00, "float", OPS_WRITE));

#ifdef VERIFICATION
    init_a_b_cpu(a_cpu, b_cpu, const_a, const_b, size, d_m, d_p, full_range);
    init_zero_cpu(dat0_cpu, size, d_m, d_p, full_range);
    // init_const_cpu(init_const, dat0_cpu, size, d_m, d_p, internal_range);
    init_index_cpu(dat0_cpu, size, d_m, d_p, full_range);
    copy_cpu(dat0_cpu, dat1_cpu, size, d_m, d_p, full_range);

    auto dat0_raw = (float*)ops_dat_get_raw_pointer(dat0, 0, S2D_00, OPS_HOST);
    auto dat1_raw = (float*)ops_dat_get_raw_pointer(dat1, 0, S2D_00, OPS_HOST);
    auto a_raw = (float*)ops_dat_get_raw_pointer(dat_a, 0, S2D_00, OPS_HOST);
    auto b_raw = (float*)ops_dat_get_raw_pointer(dat_b, 0, S2D_00, OPS_HOST);

    // printGrid2D(dat0_raw, dat0.originalProperty, "dat0_before_calc");
    // printGrid2D(dat0_cpu, dat0.originalProperty, "dat0_cpu_before_calc");
    // printGrid2D(dat1_raw, dat1.originalProperty, "dat1_before_calc");
    // printGrid2D(dat1_cpu, dat1.originalProperty, "dat1_cpu_before_calc");
    // printGrid2D(a_raw, dat_a.originalProperty, "a_before_calc");
    // printGrid2D(b_raw, dat_b.originalProperty, "b_before_calc");

#endif
    // iterative stencil loop
    ops_printf("Launching stencil calculation: %d x %d mesh\n", size[0], size[1]);

#ifndef OPS_FPGA
    for (int iter = 0; iter < iter_max; iter++)
    {
        ops_par_loop(kernel_1_5pt, "stencil5pt_k1", block, 2, internal_range,
                ops_arg_dat(dat_a, 1, S2D_00, "float", OPS_READ),
                ops_arg_dat(dat0, 1, S2D_00_P10_M10_0P1_0M1, "float", OPS_READ),
                ops_arg_dat(dat1, 1, S2D_00_P10_M10_0P1_0M1, "float", OPS_READ),
                ops_arg_dat(dat2, 1, S2D_00, "float", OPS_WRITE),
                ops_arg_dat(dat3, 1, S2D_00, "float", OPS_WRITE));
        
        ops_par_loop(kernel_2_1pt, "stencil1pt_k2", block, 2, internal_range,
                ops_arg_dat(dat_b, 1, S2D_00, "float", OPS_READ),
                ops_arg_dat(dat0, 1, S2D_00_P10_M10_0P1_0M1, "float", OPS_READ),
                ops_arg_dat(dat2, 1, S2D_00, "float", OPS_READ),
                ops_arg_dat(dat3, 1, S2D_00, "float", OPS_READ),
                ops_arg_dat(dat0_2, 1, S2D_00, "float", OPS_WRITE),
                ops_arg_dat(dat1_2, 1, S2D_00, "float", OPS_WRITE));

        ops_par_loop(copy, "copy_2", block, 2, internal_range,
                ops_arg_dat(dat0_2, 1, S2D_00, "float", OPS_READ),
                ops_arg_dat(dat0, 1, S2D_00, "float", OPS_WRITE));

        ops_par_loop(copy, "copy_3", block, 2, internal_range,
                ops_arg_dat(dat1_2, 1, S2D_00, "float", OPS_READ),
                ops_arg_dat(dat1, 1, S2D_00, "float", OPS_WRITE));
    }
#else

    ops_iter_par_loop("ops_iter_par_loop_0", iter_max,
        ops_par_loop(kernel_1_5pt, "stencil5pt_k1", block, 2, internal_range,
                ops_arg_dat(dat_a, 1, S2D_00, "float", OPS_READ),
                ops_arg_dat(dat0, 1, S2D_00_P10_M10_0P1_0M1, "float", OPS_READ),
                ops_arg_dat(dat1, 1, S2D_00_P10_M10_0P1_0M1, "float", OPS_READ),
                ops_arg_dat(dat2, 1, S2D_00, "float", OPS_WRITE),
                ops_arg_dat(dat3, 1, S2D_00, "float", OPS_WRITE)),
        
        ops_par_loop(kernel_2_1pt, "stencil1pt_k2", block, 2, internal_range,
                ops_arg_dat(dat_b, 1, S2D_00, "float", OPS_READ),
                ops_arg_dat(dat0, 1, S2D_00_P10_M10_0P1_0M1, "float", OPS_READ),
                ops_arg_dat(dat2, 1, S2D_00, "float", OPS_READ),
                ops_arg_dat(dat3, 1, S2D_00, "float", OPS_READ),
                ops_arg_dat(dat0_2, 1, S2D_00, "float", OPS_WRITE),
                ops_arg_dat(dat1_2, 1, S2D_00, "float", OPS_WRITE)),

        ops_par_loop(copy, "copy_2", block, 2, internal_range,
                ops_arg_dat(dat0_2, 1, S2D_00, "float", OPS_READ),
                ops_arg_dat(dat0, 1, S2D_00, "float", OPS_WRITE)),

        ops_par_loop(copy, "copy_3", block, 2, internal_range,
                ops_arg_dat(dat1_2, 1, S2D_00, "float", OPS_READ),
                ops_arg_dat(dat1, 1, S2D_00, "float", OPS_WRITE)));
#endif

    //Final Verification after calc
#ifdef VERIFICATION

        auto dat0_2_raw = (float*)ops_dat_get_raw_pointer(dat0_2, 0, S2D_00, OPS_HOST);
        auto dat1_2_raw = (float*)ops_dat_get_raw_pointer(dat1_2, 0, S2D_00, OPS_HOST);

        for (int iter = 0; iter < iter_max; iter++)
        {
            kernel_1_5pt_cpu(a_cpu, dat0_cpu, dat1_cpu, dat2_cpu, dat3_cpu, size, d_m, d_p, internal_range);
            kernel_2_1pt_cpu(b_cpu, dat0_cpu, dat2_cpu, dat3_cpu, dat0_2_cpu, dat1_2_cpu, size, d_m, d_p, internal_range);
            copy_cpu(dat0_2_cpu, dat0_cpu, size, d_m, d_p, internal_range);
            copy_cpu(dat1_2_cpu, dat1_cpu, size, d_m, d_p, internal_range);
        }

        // printGrid2D(dat0_2_raw, dat0_2.originalProperty, "dat0_after_calc");
        // printGrid2D(dat0_2_cpu, dat0_2.originalProperty, "dat0_cpu_after_calc");
        // printGrid2D(dat1_2_raw, dat1_2.originalProperty, "dat1_after_calc");
        // printGrid2D(dat1_2_cpu, dat1_2.originalProperty, "dat1_cpu_after_calc");

        if(verify(dat0_2_raw, dat0_2_cpu, size, d_m, d_p, internal_range))
            std::cout << "verification of dat0 after calculation" << "[PASSED]" << std::endl;
        else
            std::cout << "verification of dat0 after calculation" << "[FAILED]" << std::endl;

                if(verify(dat1_2_raw, dat1_2_cpu, size, d_m, d_p, internal_range))
            std::cout << "verification of dat1 after calculation" << "[PASSED]" << std::endl;
        else
            std::cout << "verification of dat1 after calculation" << "[FAILED]" << std::endl;
#endif

    //cleaning
    ops_free_dat(dat0);
    ops_free_dat(dat1);
    ops_free_dat(dat2);
    ops_free_dat(dat3);
    ops_free_dat(dat0_2);
    ops_free_dat(dat1_2);
    ops_free_dat(dat_a);
    ops_free_dat(dat_b);

#ifdef VERIFICATION
    free(dat0_cpu);
    free(dat1_cpu);
    free(dat2_cpu);
    free(dat3_cpu);
    free(dat0_2_cpu);
    free(dat1_2_cpu);
#endif

    ops_exit();

    std::cout << "Exit properly" << std::endl;
    return 0;
}
