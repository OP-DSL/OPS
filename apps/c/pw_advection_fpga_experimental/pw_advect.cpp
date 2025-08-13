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
* DISCLAIMED. IN NO EVENT SHALL Mike Giles BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
* (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
* LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
* ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
* (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
* SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

/** @Test application for multi-kernel functionality: Piacsek and Williams Advection
  * @author Beniel Thileepan
  * 
  */

// standard headers
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

// OPS header file
// #define OPS_SOA
#define OPS_3D
#define OPS_HLS_V2
// #define OPS_FPGA
#define PROFILE
// #define VERIFICATION
// #define DEBUG
#define OPT_KERNEL
#include <ops_seq_v2.h>

int grid_size_x, grid_size_y, grid_size_z;
extern const unsigned short mem_vector_factor;

#include "pw_advect_kernel.h"

#ifdef PROFILE 
    #include <chrono>
#endif

/******************************************************************************
* Main program
*******************************************************************************/
int main(int argc, const char** argv)
{
  /**-------------------------- Initialisation --------------------------**/

  // OPS initialisation  
  ops_init(argc,argv,1);
//   OPS_instance::getOPSInstance()->OPS_soa = 1;

  //Mesh
  int logical_size_x = 16;
  int logical_size_y = 16;
  int logical_size_z = 16;
  int ngrid_x = 1;
  int ngrid_y = 1;
  int ngrid_z = 1;
  int n_iter = 100;
//   int itertile = n_iter;
//   int non_copy = 0;

  int batches = 1;

  const char* pch;
  printf(" argc = %d\n",argc);
  for ( int n = 1; n < argc; n++ ) {
    pch = strstr(argv[n], "-sizex=");
    if(pch != NULL) {
      logical_size_x = atoi ( argv[n] + 7 ); continue;
    }
    pch = strstr(argv[n], "-sizey=");
    if(pch != NULL) {
      logical_size_y = atoi ( argv[n] + 7 ); continue;
    }
    pch = strstr(argv[n], "-sizez=");
    if(pch != NULL) {
      logical_size_z = atoi ( argv[n] + 7 ); continue;
    }
    pch = strstr(argv[n], "-iters=");
    if(pch != NULL) {
      n_iter = atoi ( argv[n] + 7 ); continue;
    }
    // pch = strstr(argv[n], "-itert=");
    // if(pch != NULL) {
    //   itertile = atoi ( argv[n] + 7 ); continue;
    // }
    // pch = strstr(argv[n], "-non-copy");
    // if(pch != NULL) {
    //   non_copy = 1; continue;
    // }
    pch = strstr(argv[n], "-batch=");
    if(pch != NULL) {
      batches = atoi ( argv[n] + 7 ); continue;
    }
  }

#ifdef PROFILE
	double init_runtime[batches];
	double main_loop_runtime[batches];

    std::string profile_filename = "perf_profile.csv";

    std::ofstream fstream;
    fstream.open(profile_filename, std::ios::out | std::ios::trunc);

    if (!fstream.is_open()) {
        std::cerr << "Error: Could not open the file " << profile_filename << std::endl;
        return 1; // Indicate an error occurred
    }
#endif

  printf("Grid: %dx%dx%d , %d iterations\n",logical_size_x,logical_size_y,logical_size_z,n_iter);
  
  //declare blocks
  ops_block blocks[batches];

  for (unsigned int bat=0; bat < batches; bat++)
  {
     std::string name = std::string("batch_") + std::to_string(bat);
     blocks[bat] = ops_decl_block(3, name.c_str());
  }
  printf(" HERE \n");
  
  //declare stencils
  int s3D_000[]         = {0,0,0};
  ops_stencil S3D_000 = ops_decl_stencil( 3, 1, s3D_000, "000");
  int s3D_27pt_sten[] = {-1,-1,-1, 0,-1,-1, 1,-1,-1, -1,0,-1, 0,0,-1, 1,0,-1, -1,1,-1, 0,1,-1, 1,1,-1,
                         -1,-1,0, 0,-1,0, 1,-1,0, -1,0,0, 0,0,0, 1,0,0, -1,1,0, 0,1,0, 1,1,0,
                         -1,-1,1, 0,-1,1, 1,-1,1, -1,0,1, 0,0,1, 1,0,1, -1,1,1, 0,1,1, 1,1,1};

  ops_stencil S3D_27PT_STEN = ops_decl_stencil( 3, 27, s3D_27pt_sten, "s3d_27pt_sten");

  printf(" HERE2 \n");
  

  //declare datasets
    int size[] = {logical_size_x, logical_size_y, logical_size_z};
    int d_p[] = {11,1,1}; //max halo depths for the dat in the possitive direction
    int d_m[] = {-1,-1,-1}; //max halo depths for the dat in the negative direction
    int base[] = {0,0,0};
//   int uniform_size[3] = {(logical_size_x-1)/ngrid_x+1,(logical_size_y-1)/ngrid_y+1,(logical_size_z-1)/ngrid_z+1};
    float* temp = NULL;
    int disps[] = {0,0,0};

    grid_size_z = size[2] - d_m[2] + d_p[2];
    grid_size_y = size[1] - d_m[1] + d_p[1];
    #ifdef OPS_FPGA
    grid_size_x = ((size[0] - d_m[0] + d_p[0] + mem_vector_factor - 1) / mem_vector_factor) * mem_vector_factor;
    #else
    grid_size_x = size[0] - d_m[0] + d_p[0];
    #endif

    ops_decl_const("grid_size_x",1, "int", &grid_size_x);
    ops_decl_const("grid_size_y",1, "int", &grid_size_y);
    ops_decl_const("grid_size_z",1, "int", &grid_size_z);
    
  printf(" HERE 3\n");
  

//   if (size[0]>logical_size_x) size[0] = logical_size_x;
//   if (size[1]>logical_size_y) size[1] = logical_size_y;
//   if (size[2]>logical_size_z) size[2] = logical_size_z;

    ops_dat u[batches];
    ops_dat v[batches];
    ops_dat w[batches];
    ops_dat u2[batches];
    ops_dat v2[batches];
    ops_dat w2[batches];
    ops_dat tzc1[batches];
    ops_dat tzc2[batches];
    ops_dat tzd1[batches];
    ops_dat tzd2[batches];
    
    //Allocations

    for (unsigned int bat=0; bat < batches; bat++)
    {
        std::string name = std::string("u_bat_") + std::to_string(bat);
        u[bat] = ops_decl_dat(blocks[bat], 1, size, base, d_m, d_p, temp, "float", name.c_str());
        name = std::string("v_bat_") + std::to_string(bat);
        v[bat] = ops_decl_dat(blocks[bat], 1, size, base, d_m, d_p, temp, "float", name.c_str());
        name = std::string("w_bat_") + std::to_string(bat);
        w[bat] = ops_decl_dat(blocks[bat], 1, size, base, d_m, d_p, temp, "float", name.c_str());
        name = std::string("u2_bat_") + std::to_string(bat);
        u2[bat] = ops_decl_dat(blocks[bat], 1, size, base, d_m, d_p, temp, "float", name.c_str());
        name = std::string("v2_bat_") + std::to_string(bat);
        v2[bat] = ops_decl_dat(blocks[bat], 1, size, base, d_m, d_p, temp, "float", name.c_str());
        name = std::string("w2_bat_") + std::to_string(bat);
        w2[bat] = ops_decl_dat(blocks[bat], 1, size, base, d_m, d_p, temp, "float", name.c_str());
        name = std::string("tzc1_bat_") + std::to_string(bat);
        name = std::string("tzc1_bat_") + std::to_string(bat);
        tzc1[bat] = ops_decl_dat(blocks[bat], 1, size, base, d_m, d_p, temp, "float", name.c_str());
        name = std::string("tzc2_bat_") + std::to_string(bat);
        tzc2[bat] = ops_decl_dat(blocks[bat], 1, size, base, d_m, d_p, temp, "float", name.c_str());
        name = std::string("tzd1_bat_") + std::to_string(bat);
        tzd1[bat] = ops_decl_dat(blocks[bat], 1, size, base, d_m, d_p, temp, "float", name.c_str());
        name = std::string("tzd2_bat_") + std::to_string(bat);
        tzd2[bat] = ops_decl_dat(blocks[bat], 1, size, base, d_m, d_p, temp, "float", name.c_str());
    }

  printf(" HERE 4\n");

  
  ops_partition("");
//   ops_checkpointing_init("check.h5", 5.0, 0);
//   ops_diagnostic_output();
  /**-------------------------- Computations --------------------------**/


//   double ct0, ct1, et0, et1;
//   ops_timers(&ct0, &et0);

//   ops_par_loop_blocks_all(batches);
	
    printf(" HERE 5\n");
  
  //populate density, bulk modulus, velx, vely, velz, and boundary conditions
	int full_range[] = {d_m[0], size[0] + d_p[0], d_m[1], size[1] + d_p[1], d_m[2], size[2] + d_p[2]};
    int internal_range[] = {0,size[0],0,size[1],0,size[2]};
	
    //Intialisation
    for (unsigned int bat = 0; bat < batches; bat++)
    {
#ifdef PROFILE
        auto init_start_clk_point =  std::chrono::high_resolution_clock::now();
#endif    
        ops_par_loop(pw_initialize, "pw_initialize", blocks[bat],  3 ,  full_range, 
                ops_arg_dat(u[bat], 1, S3D_000, "float", OPS_WRITE),
                ops_arg_dat(v[bat], 1, S3D_000, "float", OPS_WRITE),
                ops_arg_dat(w[bat], 1, S3D_000, "float", OPS_WRITE),
                ops_arg_dat(tzc1[bat], 1, S3D_000, "float", OPS_WRITE),
                ops_arg_dat(tzc2[bat], 1, S3D_000, "float", OPS_WRITE),
                ops_arg_dat(tzd1[bat], 1, S3D_000, "float", OPS_WRITE),
                ops_arg_dat(tzd2[bat], 1, S3D_000, "float", OPS_WRITE));
        ops_par_loop(pw_copy_all, "copy_all_1", blocks[bat],  3 ,  full_range,
                ops_arg_dat(u[bat], 1, S3D_000, "float", OPS_READ),
                ops_arg_dat(v[bat], 1, S3D_000, "float", OPS_READ),
                ops_arg_dat(w[bat], 1, S3D_000, "float", OPS_READ),
                ops_arg_dat(u2[bat], 1, S3D_000, "float", OPS_WRITE),
                ops_arg_dat(v2[bat], 1, S3D_000, "float", OPS_WRITE),
                ops_arg_dat(w2[bat], 1, S3D_000, "float", OPS_WRITE));

#ifdef VERIFICATION
    #ifdef DEBUG
        auto u_raw = (float*)ops_dat_get_raw_pointer(u[bat], 0, S3D_000, OPS_HOST);
        auto v_raw = (float*)ops_dat_get_raw_pointer(u[bat], 0, S3D_000, OPS_HOST);
        auto w_raw = (float*)ops_dat_get_raw_pointer(u[bat], 0, S3D_000, OPS_HOST);

        printGrid3D(u_raw, u[bat].originalProperty, "u_raw_after_init");
        printGrid3D(v_raw, v[bat].originalProperty, "v_raw_after_init");
        printGrid3D(w_raw, w[bat].originalProperty, "w_raw_after_init");
    #endif
#endif

  
#ifdef PROFILE
        auto init_end_clk_point = std::chrono::high_resolution_clock::now();
        init_runtime[bat] = std::chrono::duration<double, std::micro> (init_end_clk_point - init_start_clk_point).count();
#endif
    }
    printf(" DONE populate\n");

    //Iterative stencil loop
    for (unsigned int bat =0; bat < batches; bat++)
    {
        printf("Launching pw advection calculation: %d x %d x %d mesh\n", size[0], size[1], size[2]);
#ifdef PROFILE
        auto main_loop_start_clk_point = std::chrono::high_resolution_clock::now();
#endif
#ifdef OPS_FPGA
        #pragma ISL "isl_0" n_iter 
#endif
        for (int iter = 0; iter < n_iter; iter++) {
#ifndef OPT_KERNEL
            ops_par_loop(pw_advection_kernel, "fd3d_pml_kernel1", blocks[bat], 3, internal_range,
#else
            ops_par_loop(pw_advection_opt_kernel, "fd3d_pml_kernel1", blocks[bat], 3, internal_range,
#endif   
                ops_arg_dat(u[bat], 1, S3D_27PT_STEN, "float", OPS_READ),
                ops_arg_dat(v[bat], 1, S3D_27PT_STEN, "float", OPS_READ),
                ops_arg_dat(w[bat], 1, S3D_27PT_STEN, "float", OPS_READ),
                ops_arg_dat(u2[bat], 1, S3D_000, "float", OPS_WRITE),
                ops_arg_dat(v2[bat], 1, S3D_000, "float", OPS_WRITE),
                ops_arg_dat(w2[bat], 1, S3D_000, "float", OPS_WRITE),
                ops_arg_dat(tzc1[bat], 1, S3D_000, "float", OPS_READ),
                ops_arg_dat(tzc2[bat], 1, S3D_000, "float", OPS_READ),
                ops_arg_dat(tzd1[bat], 1, S3D_000, "float", OPS_READ),
                ops_arg_dat(tzd2[bat], 1, S3D_000, "float", OPS_READ));

            ops_par_loop(pw_copy_all, "copy_all_2", blocks[bat],  3 ,  internal_range,
                    ops_arg_dat(u2[bat], 1, S3D_000, "float", OPS_READ),
                    ops_arg_dat(v2[bat], 1, S3D_000, "float", OPS_READ),
                    ops_arg_dat(w2[bat], 1, S3D_000, "float", OPS_READ),
                    ops_arg_dat(u[bat], 1, S3D_000, "float", OPS_WRITE),
                    ops_arg_dat(v[bat], 1, S3D_000, "float", OPS_WRITE),
                    ops_arg_dat(w[bat], 1, S3D_000, "float", OPS_WRITE));
        } 
#ifdef PROFILE
        auto main_loop_end_clk_point = std::chrono::high_resolution_clock::now();
    #ifndef OPS_FPGA
        main_loop_runtime[bat] = std::chrono::duration<double, std::micro>(main_loop_end_clk_point - main_loop_start_clk_point).count();
    #else
        main_loop_runtime[bat] = ops_hls_get_execution_runtime<std::chrono::microseconds>(std::string("isl_0"));
    #endif
#endif
    }

    //Cleaning
    for (unsigned int bat = 0; bat < batches; bat++)
    {
        ops_free_dat(u[bat]);
        ops_free_dat(v[bat]);
        ops_free_dat(w[bat]);
        ops_free_dat(u2[bat]);
        ops_free_dat(v2[bat]);
        ops_free_dat(w2[bat]);
        ops_free_dat(tzc1[bat]);
        ops_free_dat(tzc2[bat]);
        ops_free_dat(tzd1[bat]);
        ops_free_dat(tzd2[bat]);
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

    fstream << "grid_x," << "grid_y," << "grid_z," << "iters," << "batch_size," << "batch_id," << "init_time," << "main_time," << "total_time" << std::endl; 

	for (unsigned int bat = 0; bat < batches; bat++)
	{
        fstream << logical_size_x << "," << logical_size_y << "," << logical_size_z << "," << n_iter << "," << 1 << "," << bat << "," << init_runtime[bat] \
                << "," << main_loop_runtime[bat] << "," << main_loop_runtime[bat] + init_runtime[bat] << std::endl;

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

    fstream.close();

    if (fstream.good()) { // Check if operations were successful after closing
        std::cout << "Successfully wrote data to " << profile_filename << std::endl;
    } else {
            std::cerr << "Error occurred during writing to " << profile_filename << std::endl;
            return 1; // Indicate an error occurred
    }
#endif

	ops_exit();


    std::cout << "Exit properly" << std::endl;
    return 0;
}