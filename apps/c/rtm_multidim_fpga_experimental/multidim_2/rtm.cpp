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

/** @Test application for multi-block functionality
  * @author Gihan Mudalige, Istvan Reguly, Beniel Thileepan
  */

// standard headers
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

float dx,dy,dz, invdx, invdy, invdz;
int pml_width;
int half_order, order, nx, ny, nz, grid_size_x, grid_size_y, grid_size_z;
#include "coeffs8.h"
// OPS header file
// #define OPS_SOA
#define OPS_3D
#define OPS_HLS_V2
// #define OPS_FPGA
// #define PROFILE
// #define VERIFICATION
#include <ops_seq_v2.h>
#include "rtm_kernel.h"

#ifdef PROFILE 
    #include <chrono>
#endif
#ifdef POWER_PROFILE
    unsigned int power_iter = 1;
    #ifdef PROFILE
    std::cerr << "POWER_PROFILE cannot be enabled with PROFILE" << std::endl;
    exit(-1);
    #endif  
#endif
// void derivs1(int ngrid_x, int ngrid_y, int ngrid_z, int* sizes, int disps[], ops_block blocks, ops_stencil S3D_000, ops_stencil S3D_big_sten,  float* dt,  float* scale1, float* scale2, ops_dat rho, ops_dat mu, ops_dat yy, ops_dat dyy, ops_dat sum);
// void derivs2(int ngrid_x, int ngrid_y, int ngrid_z, int* sizes, int disps[], ops_block blocks, ops_stencil S3D_000, ops_stencil S3D_big_sten,  float* dt,  float* scale1, float* scale2, ops_dat rho, ops_dat mu, ops_dat yy, ops_dat dyyIn,  ops_dat dyyOut, ops_dat sum);

// void derivs3(int ngrid_x, int ngrid_y, int ngrid_z, int* sizes, int disps[], ops_block blocks, ops_stencil S3D_000, ops_stencil S3D_big_sten,  float* dt,  float* scale1, float* scale2, ops_dat rho, ops_dat mu, ops_dat yy, ops_dat dyyIn, ops_dat sum);
	      
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
  nx = logical_size_x;
  ny = logical_size_y;
  nz = logical_size_z;
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
#ifdef POWER_PROFILE
    pch = strstr(argv[n], "-piter=");
    if(pch != NULL) {
        power_iter = atoi ( argv[n] + 7 ); continue;
    }
    batches = 1;
#endif
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
  
  dx = 0.005;
  dy = 0.005;
  dz = 0.005;
  invdx = 1/dx;
  invdy = 1/dy;
  invdz = 1/dz;
  pml_width = 10;
  ops_decl_const("dx",1,"float",&dx);
  ops_decl_const("dy",1,"float",&dy);
  ops_decl_const("dz",1,"float",&dz);
  ops_decl_const("invdx",1,"float",&invdx);
  ops_decl_const("invdy",1,"float",&invdy);
  ops_decl_const("invdz",1,"float",&invdz);
  ops_decl_const("nx",1,"int",&nx);
  ops_decl_const("ny",1,"int",&ny);
  ops_decl_const("nz",1,"int",&nz);
  ops_decl_const("pml_width",1,"int",&pml_width);
  //int ncoeffs = (ORDER+1)*(ORDER+1);
  //ops_decl_const("coeffs",ncoeffs,"double",&coeffs[0][0]);
  half_order = HALF_ORDER;
  ops_decl_const("half_order",1,"int",&half_order);
  order = ORDER;
  ops_decl_const("order",1,"int",&order);

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
  int s3D_big_sten[] = {-4,0,0, -3,0,0, -2,0,0, -1,0,0, 0,0,0, 1,0,0, 2,0,0, 3,0,0, 4,0,0, 
                        0,-4,0, 0,-3,0, 0,-2,0, 0,-1,0,        0,1,0, 0,2,0, 0,3,0, 0,4,0,
                        0,0,-4, 0,0,-3, 0,0,-2, 0,0,-1,        0,0,1, 0,0,2, 0,0,3, 0,0,4};

  ops_stencil S3D_big_sten = ops_decl_stencil( 3, 3*ORDER+1, s3D_big_sten, "big_sten");

  printf(" HERE2 \n");
  

  //declare datasets
    int size[] = {logical_size_x, logical_size_y, logical_size_z};
    int d_p[] = {HALF_ORDER,HALF_ORDER,HALF_ORDER}; //max halo depths for the dat in the possitive direction
    int d_m[] = {-HALF_ORDER,-HALF_ORDER,-HALF_ORDER}; //max halo depths for the dat in the negative direction
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

    ops_dat coordx[batches];
    ops_dat coordy[batches];
    ops_dat coordz[batches];
    ops_dat rho_mu[batches];
    // ops_dat rho[batches];
    // ops_dat mu[batches];
    ops_dat yy_0_1[batches];
    ops_dat yy_2_3[batches];
    ops_dat yy_4_5[batches];
    ops_dat yy_sum_0_1[batches];
    ops_dat yy_sum_2_3[batches];
    ops_dat yy_sum_4_5[batches];
    ops_dat ytemp1_0_1[batches];
    ops_dat ytemp1_2_3[batches];
    ops_dat ytemp1_4_5[batches];
    ops_dat ytemp2_0_1[batches];
    ops_dat ytemp2_2_3[batches];
    ops_dat ytemp2_4_5[batches];

    //Allocationß
    for (unsigned int bat=0; bat < batches; bat++)
    {
        std::string name = std::string("coordx_bat_") + std::to_string(bat);
        coordx[bat] = ops_hls_decl_dat(blocks[bat], 1, size, base, d_m, d_p, temp, "float", name.c_str());
        name = std::string("coordy_bat_") + std::to_string(bat);
        coordy[bat] = ops_hls_decl_dat(blocks[bat], 1, size, base, d_m, d_p, temp, "float", name.c_str());
        name = std::string("coordz_bat_") + std::to_string(bat);
        coordz[bat] = ops_hls_decl_dat(blocks[bat], 1, size, base, d_m, d_p, temp, "float", name.c_str());
        // name = std::string("rho_bat_") + std::to_string(bat);
        // rho[bat] = ops_hls_decl_dat(blocks[bat], 1, size, base, d_m, d_p, temp, "float", name.c_str());
        // name = std::string("mu_bat_") + std::to_string(bat);
        // mu[bat] = ops_hls_decl_dat(blocks[bat], 1, size, base, d_m, d_p, temp, "float", name.c_str());
        name = std::string("rho_mu_bat_") + std::to_string(bat);
        rho_mu[bat] = ops_hls_decl_dat(blocks[bat], 1, size, base, d_m, d_p, temp, "float", name.c_str());

        name = std::string("yy_0_1_bat_") + std::to_string(bat);
        yy_0_1[bat] = ops_hls_decl_dat(blocks[bat], 2, size, base, d_m, d_p, temp, "float", name.c_str());
        name = std::string("yy_2_3_bat_") + std::to_string(bat);
        yy_2_3[bat] = ops_hls_decl_dat(blocks[bat], 2, size, base, d_m, d_p, temp, "float", name.c_str());
        name = std::string("yy_4_5_bat_") + std::to_string(bat);
        yy_4_5[bat] = ops_hls_decl_dat(blocks[bat], 2, size, base, d_m, d_p, temp, "float", name.c_str());

        name = std::string("yy_sum_0_1_bat_") + std::to_string(bat);
        yy_sum_0_1[bat] = ops_hls_decl_dat(blocks[bat], 2, size, base, d_m, d_p, temp, "float", name.c_str());
        name = std::string("yy_sum_2_3_bat_") + std::to_string(bat);
        yy_sum_2_3[bat] = ops_hls_decl_dat(blocks[bat], 2, size, base, d_m, d_p, temp, "float", name.c_str());
        name = std::string("yy_sum_4_5_bat_") + std::to_string(bat);
        yy_sum_4_5[bat] = ops_hls_decl_dat(blocks[bat], 2, size, base, d_m, d_p, temp, "float", name.c_str());

        name = std::string("ytemp1_0_1_bat_") + std::to_string(bat);
        ytemp1_0_1[bat] = ops_hls_decl_dat(blocks[bat], 2, size, base, d_m, d_p, temp, "float", name.c_str());
        name = std::string("ytemp1_2_3_bat_") + std::to_string(bat);
        ytemp1_2_3[bat] = ops_hls_decl_dat(blocks[bat], 2, size, base, d_m, d_p, temp, "float", name.c_str());
        name = std::string("ytemp1_4_5_bat_") + std::to_string(bat);
        ytemp1_4_5[bat] = ops_hls_decl_dat(blocks[bat], 2, size, base, d_m, d_p, temp, "float", name.c_str());

        name = std::string("ytemp2_0_1_bat_") + std::to_string(bat);
        ytemp2_0_1[bat] = ops_hls_decl_dat(blocks[bat], 2, size, base, d_m, d_p, temp, "float", name.c_str());
        name = std::string("ytemp2_2_3_bat_") + std::to_string(bat);
        ytemp2_2_3[bat] = ops_hls_decl_dat(blocks[bat], 2, size, base, d_m, d_p, temp, "float", name.c_str());
        name = std::string("ytemp2_4_5_bat_") + std::to_string(bat);
        ytemp2_4_5[bat] = ops_hls_decl_dat(blocks[bat], 2, size, base, d_m, d_p, temp, "float", name.c_str());
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
        ops_par_loop(rtm_kernel_populate, "kernel_populate", blocks[bat],  3 ,  full_range, 
                ops_arg_gbl(&disps[0], 1, "int", OPS_READ),
                ops_arg_gbl(&disps[1], 1, "int", OPS_READ),
                ops_arg_gbl(&disps[2], 1, "int", OPS_READ),
                ops_arg_idx(),
                // ops_arg_dat(rho[bat], 1, S3D_000, "float", OPS_WRITE),
                // ops_arg_dat(mu[bat], 1, S3D_000, "float", OPS_WRITE),
                ops_arg_dat(rho_mu[bat], 2, S3D_000, "float", OPS_WRITE),
                ops_arg_dat(yy_0_1[bat], 2, S3D_000, "float", OPS_WRITE));
        ops_par_loop(kernel_copy_d2, "copyd2_0", blocks[bat], 3, full_range, 
                ops_arg_dat(yy_0_1[bat], 2, S3D_000, "float", OPS_READ),
                ops_arg_dat(yy_2_3[bat], 2, S3D_000, "float", OPS_WRITE));
        ops_par_loop(kernel_copy_d2, "copyd2_0", blocks[bat], 3, full_range, 
                ops_arg_dat(yy_0_1[bat], 2, S3D_000, "float", OPS_READ),
                ops_arg_dat(yy_4_5[bat], 2, S3D_000, "float", OPS_WRITE));
        // ops_par_loop(rtm_kernel_populate, "copy_yy_sum_1", blocks[bat],  3 ,  full_range,
        //         ops_arg_dat(yy_sum_0[bat], 1, S3D_000, "float", OPS_READ), 
        //         ops_arg_dat(yy_sum_1[bat], 1, S3D_big_sten, "float", OPS_WRITE));
        // ops_par_loop(rtm_kernel_populate, "copy_yy_sum_2", blocks[bat],  3 ,  full_range,
        //         ops_arg_dat(yy_sum_0[bat], 1, S3D_000, "float", OPS_READ), 
        //         ops_arg_dat(yy_sum_2[bat], 1, S3D_big_sten, "float", OPS_WRITE));
        // ops_par_loop(rtm_kernel_populate, "copy_yy_sum_3", blocks[bat],  3 ,  full_range,
        //         ops_arg_dat(yy_sum_0[bat], 1, S3D_000, "float", OPS_READ), 
        //         ops_arg_dat(yy_sum_3[bat], 1, S3D_big_sten, "float", OPS_WRITE));
        // ops_par_loop(rtm_kernel_populate, "copy_yy_sum_4", blocks[bat],  3 ,  full_range,
        //         ops_arg_dat(yy_sum_0[bat], 1, S3D_000, "float", OPS_READ), 
        //         ops_arg_dat(yy_sum_4[bat], 1, S3D_big_sten, "float", OPS_WRITE));
        // ops_par_loop(rtm_kernel_populate, "copy_yy_sum_5", blocks[bat],  3 ,  full_range,
        //         ops_arg_dat(yy_sum_0[bat], 1, S3D_000, "float", OPS_READ), 
        //         ops_arg_dat(yy_sum_5[bat], 1, S3D_big_sten, "float", OPS_WRITE));
#ifdef PROFILE
        auto init_end_clk_point = std::chrono::high_resolution_clock::now();
        init_runtime[bat] = std::chrono::duration<double, std::micro> (init_end_clk_point - init_start_clk_point).count();
#endif
    }
    printf(" DONE populate\n");
  
      /* The following is 4th order Runga-Kutta */
    float dt = 0.1f; //=sqrt(mu/rho);
    float scale1_der1 = 0.5f;
    float scale2_der1 = 1/6.0f;
    float scale1_der2_1 = 0.5f;
    float scale2_der2_1 = 1/3.0f;
    float scale1_der2_2 = 1.0f;
    float scale2_der2_2 = 1/3.0f;
    float scale1_der3 = 1.0f;
    float scale2_der3 = 1/6.0f;

#ifdef POWER_PROFILE
    for (unsigned int p = 0; p < power_iter; p++)
    {
#endif
    //Iterative stencil loop
    for (unsigned int bat =0; bat < batches; bat++)
    {
        printf("Launching rtm calculation: %d x %d x %d mesh\n", size[0], size[1], size[2]);
#ifdef PROFILE
        auto main_loop_start_clk_point = std::chrono::high_resolution_clock::now();
#endif
#ifdef OPS_FPGA
        #pragma ISL "isl0" n_iter
#endif
        for (int iter = 0; iter < n_iter; iter++) {
            ops_par_loop(fd3d_pml_kernel1, "fd3d_pml_kernel1", blocks[bat], 3, internal_range,
                ops_arg_gbl(&disps[0], 1, "int", OPS_READ),
                ops_arg_gbl(&disps[1], 1, "int", OPS_READ),
                ops_arg_gbl(&disps[2], 1, "int", OPS_READ),
                ops_arg_idx(),
                ops_arg_gbl(&dt, 1, "float", OPS_READ),
                ops_arg_gbl(&scale1_der1, 1, "float", OPS_READ),
                ops_arg_gbl(&scale2_der1, 1, "float", OPS_READ),
                // ops_arg_dat(rho[bat], 1, S3D_000, "float", OPS_READ),
                // ops_arg_dat(mu[bat], 1, S3D_000, "float", OPS_READ),
                ops_arg_dat(rho_mu[bat], 2, S3D_000, "float", OPS_READ),
                ops_arg_dat(yy_0_1[bat], 2, S3D_big_sten, "float", OPS_READ),
                ops_arg_dat(yy_2_3[bat], 2, S3D_big_sten, "float", OPS_READ),
                ops_arg_dat(yy_4_5[bat], 2, S3D_big_sten, "float", OPS_READ),
                ops_arg_dat(ytemp1_0_1[bat], 2, S3D_000, "float", OPS_WRITE),
                ops_arg_dat(ytemp1_2_3[bat], 2, S3D_000, "float", OPS_WRITE),
                ops_arg_dat(ytemp1_4_5[bat], 2, S3D_000, "float", OPS_WRITE),
                ops_arg_dat(yy_sum_0_1[bat], 2, S3D_000, "float", OPS_WRITE),
                ops_arg_dat(yy_sum_2_3[bat], 2, S3D_000, "float", OPS_WRITE),
                ops_arg_dat(yy_sum_4_5[bat], 2, S3D_000, "float", OPS_WRITE));
            

            ops_par_loop(fd3d_pml_kernel2, "fd3d_pml_kernel2", blocks[bat], 3, internal_range,
                ops_arg_gbl(&disps[0], 1, "int", OPS_READ),
                ops_arg_gbl(&disps[1], 1, "int", OPS_READ),
                ops_arg_gbl(&disps[2], 1, "int", OPS_READ),
                ops_arg_idx(),
                ops_arg_gbl(&dt, 1, "float", OPS_READ),
                ops_arg_gbl(&scale1_der2_1, 1, "float", OPS_READ),
                ops_arg_gbl(&scale2_der2_1, 1, "float", OPS_READ),
                // ops_arg_dat(rho[bat], 1, S3D_000, "float", OPS_READ),
                // ops_arg_dat(mu[bat], 1, S3D_000, "float", OPS_READ),
                ops_arg_dat(rho_mu[bat], 2, S3D_000, "float", OPS_READ),
                ops_arg_dat(yy_0_1[bat], 2, S3D_big_sten, "float", OPS_READ),
                ops_arg_dat(yy_2_3[bat], 2, S3D_big_sten, "float", OPS_READ),
                ops_arg_dat(yy_4_5[bat], 2, S3D_big_sten, "float", OPS_READ),
                ops_arg_dat(ytemp1_0_1[bat], 2, S3D_big_sten, "float", OPS_READ),
                ops_arg_dat(ytemp1_2_3[bat], 2, S3D_big_sten, "float", OPS_READ),
                ops_arg_dat(ytemp1_4_5[bat], 2, S3D_big_sten, "float", OPS_READ),
                ops_arg_dat(ytemp2_0_1[bat], 2, S3D_000, "float", OPS_WRITE),
                ops_arg_dat(ytemp2_2_3[bat], 2, S3D_000, "float", OPS_WRITE),
                ops_arg_dat(ytemp2_4_5[bat], 2, S3D_000, "float", OPS_WRITE),
                ops_arg_dat(yy_sum_0_1[bat], 2, S3D_000, "float", OPS_RW),
                ops_arg_dat(yy_sum_2_3[bat], 2, S3D_000, "float", OPS_RW),
                ops_arg_dat(yy_sum_4_5[bat], 2, S3D_000, "float", OPS_RW));

            ops_par_loop(fd3d_pml_kernel2, "fd3d_pml_kernel2", blocks[bat], 3, internal_range,
                ops_arg_gbl(&disps[0], 1, "int", OPS_READ),
                ops_arg_gbl(&disps[1], 1, "int", OPS_READ),
                ops_arg_gbl(&disps[2], 1, "int", OPS_READ),
                ops_arg_idx(),
                ops_arg_gbl(&dt, 1, "float", OPS_READ),
                ops_arg_gbl(&scale1_der2_2, 1, "float", OPS_READ),
                ops_arg_gbl(&scale2_der2_2, 1, "float", OPS_READ),
                // ops_arg_dat(rho[bat], 1, S3D_000, "float", OPS_READ),
                // ops_arg_dat(mu[bat], 1, S3D_000, "float", OPS_READ),
                ops_arg_dat(rho_mu[bat], 2, S3D_000, "float", OPS_READ),
                ops_arg_dat(yy_0_1[bat], 2, S3D_000, "float", OPS_READ),
                ops_arg_dat(yy_2_3[bat], 2, S3D_000, "float", OPS_READ),
                ops_arg_dat(yy_4_5[bat], 2, S3D_000, "float", OPS_READ),
                ops_arg_dat(ytemp2_0_1[bat], 2, S3D_big_sten, "float", OPS_READ),
                ops_arg_dat(ytemp2_2_3[bat], 2, S3D_big_sten, "float", OPS_READ),
                ops_arg_dat(ytemp2_4_5[bat], 2, S3D_big_sten, "float", OPS_READ),
                ops_arg_dat(ytemp1_0_1[bat], 2, S3D_000, "float", OPS_WRITE),
                ops_arg_dat(ytemp1_2_3[bat], 2, S3D_000, "float", OPS_WRITE),
                ops_arg_dat(ytemp1_4_5[bat], 2, S3D_000, "float", OPS_WRITE),
                ops_arg_dat(yy_sum_0_1[bat], 2, S3D_000, "float", OPS_RW),
                ops_arg_dat(yy_sum_2_3[bat], 2, S3D_000, "float", OPS_RW),
                ops_arg_dat(yy_sum_4_5[bat], 2, S3D_000, "float", OPS_RW));

            ops_par_loop(fd3d_pml_kernel3, "fd3d_pml_kernel3", blocks[bat], 3, internal_range,
                ops_arg_gbl(&disps[0], 1, "int", OPS_READ),
                ops_arg_gbl(&disps[1], 1, "int", OPS_READ),
                ops_arg_gbl(&disps[2], 1, "int", OPS_READ),
                ops_arg_idx(),
                ops_arg_gbl(&dt, 1, "float", OPS_READ),
                ops_arg_gbl(&scale1_der3, 1, "float", OPS_READ),
                ops_arg_gbl(&scale2_der3, 1, "float", OPS_READ),
                // ops_arg_dat(rho[bat], 1, S3D_000, "float", OPS_READ),
                // ops_arg_dat(mu[bat], 1, S3D_000, "float", OPS_READ),
                ops_arg_dat(rho_mu[bat], 2, S3D_000, "float", OPS_READ),
                ops_arg_dat(yy_0_1[bat], 2, S3D_000, "float", OPS_READ),
                ops_arg_dat(yy_2_3[bat], 2, S3D_000, "float", OPS_READ),
                ops_arg_dat(yy_4_5[bat], 2, S3D_000, "float", OPS_READ),
                ops_arg_dat(ytemp1_0_1[bat], 2, S3D_big_sten, "float", OPS_READ),
                ops_arg_dat(ytemp1_2_3[bat], 2, S3D_big_sten, "float", OPS_READ),
                ops_arg_dat(ytemp1_4_5[bat], 2, S3D_big_sten, "float", OPS_READ),
                ops_arg_dat(ytemp2_0_1[bat], 2, S3D_000, "float", OPS_WRITE),
                ops_arg_dat(ytemp2_2_3[bat], 2, S3D_000, "float", OPS_WRITE),
                ops_arg_dat(ytemp2_4_5[bat], 2, S3D_000, "float", OPS_WRITE),
                ops_arg_dat(yy_sum_0_1[bat], 2, S3D_000, "float", OPS_READ),
                ops_arg_dat(yy_sum_2_3[bat], 2, S3D_000, "float", OPS_READ),
                ops_arg_dat(yy_sum_4_5[bat], 2, S3D_000, "float", OPS_READ));

            ops_par_loop(simple_forward_k1, "simple_forward_k1", blocks[bat], 3, internal_range,
                ops_arg_dat(ytemp2_0_1[bat], 2, S3D_000, "float", OPS_READ),
                ops_arg_dat(ytemp2_2_3[bat], 2, S3D_000, "float", OPS_READ),
                ops_arg_dat(ytemp2_4_5[bat], 2, S3D_000, "float", OPS_READ),
                ops_arg_dat(yy_0_1[bat], 2, S3D_000, "float", OPS_WRITE),
                ops_arg_dat(yy_2_3[bat], 2, S3D_000, "float", OPS_WRITE),
                ops_arg_dat(yy_4_5[bat], 2, S3D_000, "float", OPS_WRITE));
        } 
#ifdef PROFILE
        auto main_loop_end_clk_point = std::chrono::high_resolution_clock::now();
    #ifndef OPS_FPGA
        main_loop_runtime[bat] = std::chrono::duration<double, std::micro>(main_loop_end_clk_point - main_loop_start_clk_point).count();
    #else
        main_loop_runtime[bat] = ops_hls_get_execution_runtime<std::chrono::microseconds>(std::string("isl0"));
    #endif
#endif
    }
#ifdef POWER_PROFILE
    }
#endif

    //Cleaning
    for (unsigned int bat = 0; bat < batches; bat++)
    {
        ops_free_dat(coordx[bat]);
        ops_free_dat(coordy[bat]);
        ops_free_dat(coordz[bat]);
        ops_free_dat(rho_mu[bat]);
        ops_free_dat(yy_0_1[bat]);
        ops_free_dat(yy_2_3[bat]);
        ops_free_dat(yy_4_5[bat]);
        ops_free_dat(ytemp1_0_1[bat]);
        ops_free_dat(ytemp1_2_3[bat]);
        ops_free_dat(ytemp1_4_5[bat]);
        ops_free_dat(ytemp2_0_1[bat]);
        ops_free_dat(ytemp2_2_3[bat]);
        ops_free_dat(ytemp2_4_5[bat]);
        ops_free_dat(yy_sum_0_1[bat]);
        ops_free_dat(yy_sum_2_3[bat]);
        ops_free_dat(yy_sum_4_5[bat]);
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

// void derivs1(int ngrid_x, int ngrid_y, int ngrid_z, int* sizes, int disps[], ops_block blocks, ops_stencil S3D_000, ops_stencil S3D_big_sten, float* dt,  float* scale1, float* scale2, ops_dat rho, ops_dat mu, ops_dat yy, ops_dat dyy, ops_dat sum) {
  
  
//   ops_par_loop(fd3d_pml_kernel1, "fd3d_pml_kernel1", blocks, 3, internal_range,
// 	       ops_arg_gbl(&disps[0], 1, "int", OPS_READ),
// 	       ops_arg_gbl(&disps[1], 1, "int", OPS_READ),
// 	       ops_arg_gbl(&disps[2], 1, "int", OPS_READ),
// 	       ops_arg_idx(),
// 	       ops_arg_gbl(dt, 1, "float", OPS_READ),
// 	       ops_arg_gbl(scale1, 1, "float", OPS_READ),
// 	       ops_arg_gbl(scale2, 1, "float", OPS_READ),
// 	       ops_arg_dat(rho, 1, S3D_000, "float", OPS_READ),
// 	       ops_arg_dat(mu, 1, S3D_000, "float", OPS_READ),
// 	       ops_arg_dat(yy, 6, S3D_big_sten, "float", OPS_READ),
// 	       ops_arg_dat(dyy, 6, S3D_000, "float", OPS_WRITE),
// 	       ops_arg_dat(sum, 6, S3D_000, "float", OPS_WRITE)
// 	       );
// }

// void derivs2(int ngrid_x, int ngrid_y, int ngrid_z, int* sizes, int disps[], ops_block blocks, ops_stencil S3D_000, ops_stencil S3D_big_sten, float* dt,  float* scale1, float* scale2, ops_dat rho, ops_dat mu, ops_dat yy, ops_dat dyyIn,  ops_dat dyyOut, ops_dat sum) {
  
//   int internal_range[] = {0,sizes[0],0,sizes[1],0,sizes[2]};
//   ops_par_loop(fd3d_pml_kernel2, "fd3d_pml_kernel2", blocks, 3, internal_range,
// 	       ops_arg_gbl(&disps[0], 1, "int", OPS_READ),
// 	       ops_arg_gbl(&disps[1], 1, "int", OPS_READ),
// 	       ops_arg_gbl(&disps[2], 1, "int", OPS_READ),
// 	       ops_arg_idx(),
// 	       ops_arg_gbl(dt, 1, "float", OPS_READ),
// 	       ops_arg_gbl(scale1, 1, "float", OPS_READ),
// 	       ops_arg_gbl(scale2, 1, "float", OPS_READ),
// 	       ops_arg_dat(rho, 1, S3D_000, "float", OPS_READ),
// 	       ops_arg_dat(mu, 1, S3D_000, "float", OPS_READ),
// 	       ops_arg_dat(yy, 6, S3D_000, "float", OPS_READ),
// 	       ops_arg_dat(dyyIn, 6, S3D_big_sten, "float", OPS_READ),
// 	       ops_arg_dat(dyyOut, 6, S3D_000, "float", OPS_WRITE),
// 	       ops_arg_dat(sum, 6, S3D_000, "float", OPS_RW)
// 	       );
// }

// void derivs3(int ngrid_x, int ngrid_y, int ngrid_z, int* sizes, int disps[], ops_block blocks, ops_stencil S3D_000, ops_stencil S3D_big_sten, float* dt,  float* scale1, float* scale2, ops_dat rho, ops_dat mu, ops_dat yy,  ops_dat dyyIn, ops_dat sum) {
  
//   int internal_range[] = {0,sizes[0],0,sizes[1],0,sizes[2]};
//   ops_par_loop(fd3d_pml_kernel3, "fd3d_pml_kernel3", blocks, 3, internal_range,
// 	       ops_arg_gbl(&disps[0], 1, "int", OPS_READ),
// 	       ops_arg_gbl(&disps[1], 1, "int", OPS_READ),
// 	       ops_arg_gbl(&disps[2], 1, "int", OPS_READ),
// 	       ops_arg_idx(),
// 	       ops_arg_gbl(dt, 1, "float", OPS_READ),
// 	       ops_arg_gbl(scale1, 1, "float", OPS_READ),
// 	       ops_arg_gbl(scale2, 1, "float", OPS_READ),
// 	       ops_arg_dat(rho, 1, S3D_000, "float", OPS_READ),
// 	       ops_arg_dat(mu, 1, S3D_000, "float", OPS_READ),
// 	       ops_arg_dat(yy, 6, S3D_000, "float", OPS_RW),
// 	       ops_arg_dat(dyyIn, 6, S3D_big_sten, "float", OPS_READ),
// 	       ops_arg_dat(sum, 6, S3D_000, "float", OPS_READ)
// 	       );
// }