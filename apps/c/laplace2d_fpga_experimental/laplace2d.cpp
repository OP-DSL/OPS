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

/** @Test application for fpga batched temporal blocked laplace2d
  * @author Beniel Thileepan
  */


#include <math.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>

int imax, jmax;
float pi  = 2.0 * asin(1.0);

//Including main OPS header file, and setting 2D
#define OPS_2D
#define OPS_CPP_API
#define OPS_HLS_V2
// #define OPS_FPGA
#define PROFILE
#define VERIFICATION
#include <ops_seq_v2.h>
//Including applicaiton-specific "user kernels"
#include "laplace_kernels.h"
#include "laplace2d_cpu_verification.hpp"

#ifdef PROFILE
    #include <chrono>
#endif

int main(int argc, const char** argv)
{
    //Initialise the OPS library, passing runtime args, and setting diagnostics level to low (1)
    ops_init(argc, argv,1);

    unsigned int batches = 1;

#ifdef PROFILE
	double init_runtime[batches];
	double main_loop_runtime[batches];
#endif

    for (unsigned int bat = 0; bat < batches; bat++)
    {
        //Size along y
        jmax = 100;
        //Size along x
        imax = 100;
        unsigned int iter_max = 100;

        float *A=NULL;
        float *Anew=NULL;
#ifdef VERIFICATION
        float *Acpu=NULL;
        float *AnewCpu=NULL;
#endif
        //
        //Declare & define key data structures
        //
        
        //The 2D block
        ops_block block = ops_decl_block(2, "my_grid");
        //The two datasets
        int size[] = {imax, jmax};
        int base[] = {0,0};
        int d_m[] = {-1,-1};
        int d_p[] = {1,1};
        ops_dat d_A    = ops_decl_dat(block, 1, size, base,
                                    d_m, d_p, A,    "float", "A");
        ops_dat d_Anew = ops_decl_dat(block, 1, size, base,
                                    d_m, d_p, Anew, "float", "Anew");

#ifdef VERIFICATION
        int grid_size_y = size[1] - d_m[1] + d_p[1];
#ifdef OPS_FPGA
        int grid_size_x = ((size[0] - d_m[0] + d_p[0] + 16 - 1) / 16) * 16;
#else
        int grid_size_x = size[0] - d_m[0] + d_p[0];
#endif
        Acpu = (float*) malloc(sizeof(float) * grid_size_x * grid_size_y);
        AnewCpu = (float*) malloc(sizeof(float) * grid_size_x * grid_size_y);
#endif

        //Two stencils, a 1-point, and a 5-point
        int s2d_00[] = {0,0};
        ops_stencil S2D_00 = ops_decl_stencil(2,1,s2d_00,"0,0");
        int s2d_5pt[] = {0,0, 1,0, -1,0, 0,1, 0,-1};
        ops_stencil S2D_5pt = ops_decl_stencil(2,5,s2d_5pt,"5pt");

        //Reduction handle
        //   ops_reduction h_err = ops_decl_reduction_handle(sizeof(float), "float", "error");

        //declare and define global constants
        ops_decl_const("imax",1,"int",&imax);
        ops_decl_const("jmax",1,"int",&jmax);
        ops_decl_const("pi",1,"float",&pi);

        ops_partition("");

#ifdef PROFILE
		auto init_start_clk_point = std::chrono::high_resolution_clock::now();
#endif

        // set boundary conditions
        int bottom_range[] = {-1, imax+1, -1, 0};
        ops_par_loop(set_zero, "set_zero", block, 2, bottom_range,
            ops_arg_dat(d_A, 1, S2D_00, "float", OPS_WRITE));

        int top_range[] = {-1, imax+1, jmax, jmax+1};
        ops_par_loop(set_zero, "set_zero", block, 2, top_range,
            ops_arg_dat(d_A, 1, S2D_00, "float", OPS_WRITE));

        int left_range[] = {-1, 0, -1, jmax+1};
        ops_par_loop(left_bndcon, "left_bndcon", block, 2, left_range,
            ops_arg_dat(d_A, 1, S2D_00, "float", OPS_WRITE),
            ops_arg_idx());

        int right_range[] = {imax, imax+1, -1, jmax+1};
        ops_par_loop(right_bndcon, "right_bndcon", block, 2, right_range,
            ops_arg_dat(d_A, 1, S2D_00, "float", OPS_WRITE),
            ops_arg_idx());

#ifdef PROFILE
		auto init_end_clk_point = std::chrono::high_resolution_clock::now();
		init_runtime[bat] = std::chrono::duration<double, std::micro> (init_end_clk_point - init_start_clk_point).count();
#endif

        ops_printf("Jacobi relaxation Calculation: %d x %d mesh\n", imax+2, jmax+2);

#ifdef PROFILE
		init_start_clk_point = std::chrono::high_resolution_clock::now();
#endif

        ops_par_loop(set_zero, "set_zero", block, 2, bottom_range,
            ops_arg_dat(d_Anew, 1, S2D_00, "float", OPS_WRITE));

        ops_par_loop(set_zero, "set_zero", block, 2, top_range,
            ops_arg_dat(d_Anew, 1, S2D_00, "float", OPS_WRITE));

        ops_par_loop(left_bndcon, "left_bndcon", block, 2, left_range,
            ops_arg_dat(d_Anew, 1, S2D_00, "float", OPS_WRITE),
            ops_arg_idx());

        ops_par_loop(right_bndcon, "right_bndcon", block, 2, right_range,
            ops_arg_dat(d_Anew, 1, S2D_00, "float", OPS_WRITE),
            ops_arg_idx());

#ifdef PROFILE
		init_end_clk_point = std::chrono::high_resolution_clock::now();
		init_runtime[bat] += std::chrono::duration<double, std::micro> (init_end_clk_point - init_start_clk_point).count();
#endif

#ifdef VERIFICATION
        A = (float*)ops_dat_get_raw_pointer(d_A, 0, S2D_00, OPS_HOST);
        Anew = (float*)ops_dat_get_raw_pointer(d_Anew, 0, S2D_00, OPS_HOST);

        if(verify(A, Anew, size, d_m, d_p))
            std::cout << "verification of d_A and d_Anew" << "[PASSED]" << std::endl;
        else
            std::cerr << "verification of d_A and d_Anew" << "[FAILED]" << std::endl;

        initilizeGrid(Acpu, size, d_m, d_p, pi, jmax);
        copyGrid(AnewCpu, Acpu, size, d_m, d_p);

        if (verify(Acpu, A, size, d_m, d_p))
            std::cout << "verification of Acpu and A" << "[PASSED]" << std::endl;
        else
            std::cerr << "verification of Acpu and A" << "[FAILED]" << std::endl;

        if (verify(AnewCpu, Anew, size, d_m, d_p))
            std::cout << "verification of AnewCpu and Anew" << "[PASSED]" << std::endl;
        else
            std::cerr << "verification of AnewCpu and Anew" << "[FAILED]" << std::endl;

        // printGrid2D<float>(A, d_A.originalProperty, "d_A after computation");
        // printGrid2D<float>(Acpu, d_A.originalProperty, "d_Acpu after computation");

        // printGrid2D<float>(Anew, d_Anew.originalProperty, "d_Anew after computation");
        // printGrid2D<float>(AnewCpu, d_Anew.originalProperty, "d_AnewCpu after computation");
#endif

        int interior_range[] = {0,imax,0,jmax};
        // ops_par_loop(test_init, "test_init", block, 2, interior_range, 
        //     ops_arg_dat(d_A, 2, S2D_00, "float", OPS_WRITE),
        //     ops_arg_idx());

#ifdef PROFILE
		auto main_loop_start_clk_point = std::chrono::high_resolution_clock::now();
#endif

#ifndef OPS_FPGA
        for (unsigned int i = 0; i < iter_max; i++)
        {
            
            ops_par_loop(apply_stencil, "apply_stencil", block, 2, interior_range,
                ops_arg_dat(d_A,    1, S2D_5pt, "float", OPS_READ),
                ops_arg_dat(d_Anew, 1, S2D_00, "float", OPS_WRITE));

            // ops_dat_deep_copy(d_A, d_Anew);
            ops_par_loop(copy, "copy", block, 2, interior_range,
                ops_arg_dat(d_A,    1, S2D_00, "float", OPS_WRITE),
                ops_arg_dat(d_Anew, 1, S2D_00, "float", OPS_READ));

            // if(iter % 10 == 0) ops_printf("%5d, %0.6f\n", iter, error);        
            // iter++;
        }
#else
        ops_iter_par_loop("ops_iter_par_loop_0", iter_max, 
            ops_par_loop(apply_stencil, "apply_stencil", block, 2, interior_range,
                ops_arg_dat(d_A,    1, S2D_5pt, "float", OPS_READ),
                ops_arg_dat(d_Anew, 1, S2D_00, "float", OPS_WRITE)), 
        ops_par_copy<float>(d_A, d_Anew));
#endif

#ifdef PROFILE
		auto main_loop_end_clk_point = std::chrono::high_resolution_clock::now();
    #ifndef OPS_FPGA
		main_loop_runtime[bat] = std::chrono::duration<double, std::micro>(main_loop_end_clk_point - main_loop_start_clk_point).count();
    #else
        main_loop_runtime[bat] = ops_hls_get_execution_runtime<std::chrono::microseconds>(std::string("ops_iter_par_loop_0"));
    #endif
#endif

#ifdef VERIFICATION
        A = (float*)d_A->get_raw_pointer(0, S2D_00, OPS_HOST);
        Anew = (float*)d_Anew->get_raw_pointer(0, S2D_00, OPS_HOST);

		for (int iter = 0; iter < iter_max; iter++)
		{
			calcGrid(Acpu, AnewCpu, size, d_m, d_p);
			copyGrid(Acpu, AnewCpu, size, d_m, d_p);
		}

		// if (verify(A, Acpu, size, d_m, d_p))
		// 	std::cout << "verification of A and Acpu after calc" << "[PASSED]" << std::endl;
		// else
		// 	std::cerr << "verification of A and Acpu after calc" << "[FAILED]" << std::endl;

		if (verify(Anew, AnewCpu, size, d_m, d_p))
			std::cout << "verification of Anew and AnewCpu after calc" << "[PASSED]" << std::endl;
		else
			std::cerr << "verification of Anew and AnewCpu after calc" << "[FAILED]" << std::endl;

		// printGrid2D<float>(A, d_A.originalProperty, "d_A after computation");
		// printGrid2D<float>(Acpu, d_A.originalProperty, "d_Acpu after computation");

        // printGrid2D<float>(Anew, d_Anew.originalProperty, "d_Anew after computation");
		// printGrid2D<float>(AnewCpu, d_Anew.originalProperty, "d_AnewCpu after computation");

		free(Acpu);
		free(AnewCpu);
    #endif
        //   ops_printf("%5d, %0.6f\n", iter, error);        

        //   ops_timing_output(std::cout);

        //   float err_diff = fabs((100.0*(error/2.421354960840227e-03))-100.0);
        //   printf("Total error is within %3.15E %% of the expected error\n",err_diff);
        //   if(err_diff < 0.001)
        //     printf("This run is considered PASSED\n");
        //   else
        //     printf("This test is considered FAILED\n");

        //Finalising the OPS library
		ops_free_dat(d_A);
		ops_free_dat(d_Anew);
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

