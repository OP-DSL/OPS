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

/** @Test application for fpga batched temporal blocked poisson 2D
  * @author Gihan Mudalige, Istvan Reguly, Beniel Thileepan
  */

// standard headers
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

float dx,dy,dx_2,dy_2,dx_2_dy_2,dx_2_plus_dy_2_mult_2;
extern const unsigned short mem_vector_factor;

// OPS header file
#define OPS_2D
#define VERIFICATION
// #define OPS_CPP_API
#define OPS_HLS_V2
// #define OPS_FPGA
#define PROFILE
#include "user_types.h"
#include <ops_seq_v2.h>
#include "poisson_kernel.h"
#include "poisson_cpu_verification.hpp"

#ifdef PROFILE
    #include <chrono>
#endif

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

#ifdef PROFILE
	double init_runtime[batches];
	double main_loop_runtime[batches];
#endif

    //declare consts
    dx = 0.01;
    dy = 0.01;
    dy_2 = dy*dy;
    dx_2 = dx*dx;
    dx_2_plus_dy_2_mult_2 = (dy_2 + dx_2) * 2.0;
    dx_2_dy_2 = dy_2 * dx_2;

    ops_decl_const("dx", 1, "float", &dx);
    ops_decl_const("dy", 1, "float", &dy);
    ops_decl_const("dy_2", 1, "float", &dy_2);
    ops_decl_const("dx_2", 1, "float", &dx_2);
    ops_decl_const("dx_2_plus_dy_2_mult_2",1, "float", &dx_2_plus_dy_2_mult_2);
    ops_decl_const("dx_2_dy_2",1, "float", &dx_2_dy_2);


    //The 2D block
    ops_block blocks[batches];

    for (unsigned int bat = 0; bat < batches; bat++)
    {
        std::string name = std::string("batch_") + std::to_string(bat);
        blocks[bat] = ops_decl_block(2, name.c_str());
    }


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

    ops_dat u[batches];
    ops_dat u2[batches];
    ops_dat f[batches];
    ops_dat ref[batches];
#ifdef VERIFICATION
    float* u_cpu[batches];
    float* u2_cpu[batches];
    float* f_cpu[batches];
    float* ref_cpu[batches];

    int grid_size_y = size[1] - d_m[1] + d_p[1];
    #ifdef OPS_FPGA
    int grid_size_x = ((size[0] - d_m[0] + d_p[0] + mem_vector_factor - 1) / mem_vector_factor) * mem_vector_factor;
    #else
    int grid_size_x = size[0] - d_m[0] + d_p[0];
    #endif
#endif

    // Allocation
    for (unsigned int bat = 0; bat < batches; bat++)
    {
        std::string name = std::string("u_") + std::to_string(bat);
        u[bat] = ops_decl_dat(blocks[bat], 1, size, base, d_m, d_p, temp, "float", name.c_str());
        name = std::string("u2_") + std::to_string(bat);
        u2[bat] = ops_decl_dat(blocks[bat], 1, size, base, d_m, d_p, temp, "float", name.c_str());
        name = std::string("f_") + std::to_string(bat);
        f[bat] = ops_decl_dat(blocks[bat], 1, size, base, d_m, d_p, temp, "float", name.c_str());
        name = std::string("ref_") + std::to_string(bat);
        ref[bat] = ops_decl_dat(blocks[bat], 1, size, base, d_m, d_p, temp, "float", name.c_str());
#ifdef VERIFICATION
        u_cpu[bat] = (float*)malloc(sizeof(float) * grid_size_x * grid_size_y);
        u2_cpu[bat] = (float*)malloc(sizeof(float) * grid_size_x * grid_size_y);
        f_cpu[bat] = (float*)malloc(sizeof(float) * grid_size_x * grid_size_y);
        ref_cpu[bat] = (float*)malloc(sizeof(float) * grid_size_x * grid_size_y);
#endif
    }

    ops_partition("");

    int full_range[] = {d_m[0], size[0] + d_p[0], d_m[1], size[1] + d_p[1]};
    int internal_range[] = {0, size[0], 0, size[1]};
    //Producer
    for (unsigned int bat = 0; bat < batches; bat++)
    {
#ifdef PROFILE
        auto init_start_clk_point =  std::chrono::high_resolution_clock::now();
#endif
        ops_par_loop(poisson_kernel_populate, "poisson_kernel_populate", blocks[bat], 2, full_range, ops_arg_idx(),
                ops_arg_dat(u[bat], 1, S2D_00, "float", OPS_WRITE),
                ops_arg_dat(f[bat], 1, S2D_00, "float", OPS_WRITE),
                ops_arg_dat(ref[bat], 1, S2D_00, "float", OPS_WRITE));

        ops_par_loop(poisson_kernel_update, "poisson_kernel_update", blocks[bat], 2, full_range, 
                ops_arg_dat(u[bat], 1, S2D_00, "float", OPS_READ),
                ops_arg_dat(u2[bat], 1, S2D_00, "float", OPS_WRITE));

    //initial guess 0
        ops_par_loop(poisson_kernel_initialguess, "poisson_kernel_initialguess", blocks[bat], 2, internal_range,
                ops_arg_dat(u[bat], 1, S2D_00, "float", OPS_WRITE));
#ifdef PROFILE
        auto init_end_clk_point = std::chrono::high_resolution_clock::now();
        init_runtime[bat] = std::chrono::duration<double, std::micro> (init_end_clk_point - init_start_clk_point).count();
#endif

#ifdef VERIFICATION
        auto u_raw = (float*)ops_dat_get_raw_pointer(u[bat], 0, S2D_00, OPS_HOST);
        auto u2_raw = (float*)ops_dat_get_raw_pointer(u2[bat], 0, S2D_00, OPS_HOST);
        auto f_raw = (float*)ops_dat_get_raw_pointer(f[bat], 0, S2D_00, OPS_HOST);
        auto ref_raw = (float*)ops_dat_get_raw_pointer(ref[bat], 0, S2D_00, OPS_HOST);

        poisson_kernel_populate_cpu(u_cpu[bat], f_cpu[bat], ref_cpu[bat], size, d_m, d_p, full_range);
        poisson_kernel_update_cpu(u2_cpu[bat], u_cpu[bat], size, d_m, d_p, full_range);

        poisson_kernel_initialguess_cpu(u_cpu[bat], size, d_m, d_p, internal_range);

        if(verify(u_raw, u_cpu[bat], size, d_m, d_p, full_range))
            std::cout << "[BATCH - " << bat << "] verification of u after initiation" << "[PASSED]" << std::endl;
        else
            std::cout << "[BATCH - " << bat << "] verification of u after initiation" << "[FAILED]" << std::endl;

        if(verify(u2_raw, u2_cpu[bat], size, d_m, d_p, full_range))
            std::cout << "[BATCH - " << bat << "] verification of u2 after initiation" << "[PASSED]" << std::endl;
        else
            std::cout << "[BATCH - " << bat << "] verification of u2 after initiation" << "[FAILED]" << std::endl;

        if(verify(f_raw, f_cpu[bat], size, d_m, d_p, full_range))
            std::cout << "[BATCH - " << bat << "] verification of f after initiation" << "[PASSED]" << std::endl;
        else
            std::cout << "[BATCH - " << bat << "] verification of f after initiation" << "[FAILED]" << std::endl;
        
        if(verify(ref_raw, ref_cpu[bat], size, d_m, d_p, full_range))
            std::cout << "[BATCH - " << bat << "] verification of ref after initiation" << "[PASSED]" << std::endl;
        else
            std::cout << "[BATCH - " << bat << "] verification of ref after initiation" << "[FAILED]" << std::endl;
#endif
    }


    //iterative stencil loop
    for (unsigned int bat = 0; bat < batches; bat++)
    {
        ops_printf("Launching poisson calculation: %d x %d mesh\n", size[0], size[1]);
#ifdef PROFILE
        auto main_loop_start_clk_point = std::chrono::high_resolution_clock::now();
#endif
#ifndef OPS_FPGA
        for (int iter = 0; iter < iter_max; iter++)
        {
            ops_par_loop(poisson_kernel_stencil, "poisson_kernel_stencil", blocks[bat], 2, internal_range,
                    ops_arg_dat(u[bat], 1, S2D_00_P10_M10_0P1_0M1, "float", OPS_READ),
                    ops_arg_dat(u2[bat], 1, S2D_00, "float", OPS_WRITE));
            
            ops_par_loop(poisson_kernel_update, "poisson_kernel_update", blocks[bat], 2, internal_range,
                    ops_arg_dat(u2[bat], 1, S2D_00, "float", OPS_READ),
                    ops_arg_dat(u[bat], 1, S2D_00, "float", OPS_WRITE));
        }
#else
        ops_iter_par_loop("ops_iter_par_loop_0", iter_max,
            ops_par_loop(poisson_kernel_stencil, "poisson_kernel_stencil", blocks[bat], 2, internal_range,
                    ops_arg_dat(u[bat], 1, S2D_00_P10_M10_0P1_0M1, "float", OPS_READ),
                    ops_arg_dat(u2[bat], 1, S2D_00, "float", OPS_WRITE)),
            ops_par_copy<float>(u[bat], u2[bat]));
#endif
#ifdef PROFILE
        auto main_loop_end_clk_point = std::chrono::high_resolution_clock::now();
    #ifndef OPS_FPGA
        main_loop_runtime[bat] = std::chrono::duration<double, std::micro>(main_loop_end_clk_point - main_loop_start_clk_point).count();
    #else
        main_loop_runtime[bat] = ops_hls_get_execution_runtime<std::chrono::microseconds>(std::string("ops_iter_par_loop_0"));
    #endif
#endif
    }

    //Final Verification after calc
#ifdef VERIFICATION
    for (unsigned int bat = 0; bat < batches; bat++)
    {
        auto u_raw = (float*)ops_dat_get_raw_pointer(u[bat], 0, S2D_00, OPS_HOST);
        auto u2_raw = (float*)ops_dat_get_raw_pointer(u2[bat], 0, S2D_00, OPS_HOST);

        for (int iter = 0; iter < iter_max; iter++)
        {
            poisson_kernel_stencil_cpu(u_cpu[bat], f_cpu[bat], u2_cpu[bat], size, d_m, d_p, internal_range);
            poisson_kernel_update_cpu(u_cpu[bat], u2_cpu[bat], size, d_m, d_p, internal_range);
        }

		// printGrid2D<float>(u_raw, u[bat].originalProperty, "u after computation");
		// printGrid2D<float>(u_cpu[bat], u[bat].originalProperty, "u_Acpu after computation");

        // if(verify(u_raw, u_cpu[bat], size, d_m, d_p, full_range))
        //     std::cout << "[BATCH - " << bat << "] verification of u after calculation" << "[PASSED]" << std::endl;
        // else
        //     std::cout << "[BATCH - " << bat << "] verification of u after calculation" << "[FAILED]" << std::endl;

        if(verify(u2_raw, u2_cpu[bat], size, d_m, d_p, full_range))
            std::cout << "[BATCH - " << bat << "] verification of u2 after calculation" << "[PASSED]" << std::endl;
        else
            std::cout << "[BATCH - " << bat << "] verification of u2 after calculation" << "[FAILED]" << std::endl;

    }
#endif

    //cleaning
    for (unsigned int bat = 0; bat < batches; bat++)
    {
        ops_free_dat(u[bat]);
        ops_free_dat(u2[bat]);
        ops_free_dat(f[bat]);
        ops_free_dat(ref[bat]);
#ifdef VERIFICATION
        free(u_cpu[bat]);
        free(u2_cpu[bat]);
        free(f_cpu[bat]);
        free(ref_cpu[bat]);
#endif
    }

#ifdef PROFILE
	std::cout << std::endl;
	std::cout << "******************************************************" << std::endl;
	std::cout << "**                runtime summary                   **" << std::endl;
	std::cout << "******************************************************" << std::endl;

	double avg_main_loop_runtime = 0;
	double max_main_loop_runtime = 0;
	double min_main_loop_runtime = 0;
	double avg_init_runtime = 0;
	double max_init_runtime = 0;
	double min_init_runtime = 0;
	double main_loop_std = 0;
	double init_std = 0;
	double total_std = 0;

	for (unsigned int bat = 0; bat < batches; bat++)
	{
		std::cout << "run: "<< bat << "| total runtime: " << main_loop_runtime[bat] + init_runtime[bat] << "(us)" << std::endl;
		std::cout << "     |--> init runtime: " << init_runtime[bat] << "(us)" << std::endl;
		std::cout << "     |--> main loop runtime: " << main_loop_runtime[bat] << "(us)" << std::endl;
		avg_init_runtime += init_runtime[bat];
		avg_main_loop_runtime += main_loop_runtime[bat];

		if (bat == 0)
		{
			max_main_loop_runtime = main_loop_runtime[bat];
			min_main_loop_runtime = main_loop_runtime[bat];
			max_init_runtime = init_runtime[bat];
			min_init_runtime = init_runtime[bat];
		}
		else
		{
			max_main_loop_runtime = std::max(max_main_loop_runtime, main_loop_runtime[bat]);
			min_main_loop_runtime = std::min(min_main_loop_runtime, main_loop_runtime[bat]);
			max_init_runtime = std::max(max_init_runtime, init_runtime[bat]);
			min_init_runtime = std::min(min_init_runtime, init_runtime[bat]);
		}
	}

	avg_init_runtime /= batches;
	avg_main_loop_runtime /= batches;

	for (unsigned int bat = 0; bat < batches; bat++)
	{
		main_loop_std += std::pow(main_loop_runtime[bat] - avg_main_loop_runtime, 2);
		init_std += std::pow(init_runtime[bat] - avg_init_runtime, 2);
		total_std += std::pow(main_loop_runtime[bat] + init_runtime[bat] - avg_init_runtime - avg_main_loop_runtime, 2);
	}

	main_loop_std = std::sqrt(main_loop_std / batches);
	init_std = std::sqrt(init_std / batches);
	total_std = std::sqrt(total_std / batches);

	std::cout << "Total runtime (AVG): " << avg_main_loop_runtime + avg_init_runtime << "(us)" << std::endl;
	std::cout << "     |--> init runtime: " << avg_init_runtime << "(us)" << std::endl;
	std::cout << "     |--> main loop runtime: " << avg_main_loop_runtime << "(us)" << std::endl;
	std::cout << "Total runtime (MIN): " << min_main_loop_runtime + min_init_runtime << "(us)" << std::endl;
	std::cout << "     |--> init runtime: " << min_init_runtime << "(us)" << std::endl;
	std::cout << "     |--> main loop runtime: " << min_main_loop_runtime << "(us)" << std::endl;
	std::cout << "Total runtime (MAX): " << max_main_loop_runtime + max_init_runtime << "(us)" << std::endl;
	std::cout << "     |--> init runtime: " << max_init_runtime << "(us)" << std::endl;
	std::cout << "     |--> main loop runtime: " << max_main_loop_runtime << "(us)" << std::endl;
	std::cout << "Standard Deviation init: " << init_std << std::endl;
	std::cout << "Standard Deviation main loop: " << main_loop_std << std::endl;
	std::cout << "Standard Deviation total: " << total_std << std::endl;
	std::cout << "======================================================" << std::endl;
#endif

    ops_exit();

    std::cout << "Exit properly" << std::endl;
    return 0;
}
