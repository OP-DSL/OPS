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

/** @Test application for multi-block functionality
  * @author Gihan Mudalige, Istvan Reguly
  */

// standard headers
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

double dx,dy,dz;


// OPS header file
#define OPS_3D
#include "ops_seq_v2.h"
#include "poisson_kernel.h"

/******************************************************************************
* Main program
*******************************************************************************/
int main(int argc, char **argv)
{
  /**-------------------------- Initialisation --------------------------**/

  // OPS initialisation
  ops_init(argc,argv,1);


  //Mesh
  int logical_size_x = 100;
  int logical_size_y = 100;
  int logical_size_z = 100;
  int n_iter = 10;
  int itertile = n_iter;
  int non_copy = 0;

  const char* pch;
  for ( int n = 1; n < argc; n++ ) {
    pch = strstr(argv[n], "-sizex=");
    if(pch != NULL) {
      logical_size_x = atoi ( argv[n] + 7 ); continue;
    }
    pch = strstr(argv[n], "-sizey=");
    if(pch != NULL) {
      logical_size_y = atoi ( argv[n] + 7 ); continue;
    }
    pch = strstr(argv[n], "-iters=");
    if(pch != NULL) {
      n_iter = atoi ( argv[n] + 7 ); continue;
    }
    pch = strstr(argv[n], "-itert=");
    if(pch != NULL) {
      itertile = atoi ( argv[n] + 7 ); continue;
    }
    pch = strstr(argv[n], "-non-copy");
    if(pch != NULL) {
      non_copy = 1; continue;
    }
  }

  ops_printf("Grid: %dx%dx%d, %d iterations, %d tile height\n",logical_size_x,logical_size_y,logical_size_z,n_iter,itertile);
  dx = 0.01;
  dy = 0.01;
  dz = 0.01;
  ops_decl_const("dx",1,"double",&dx);
  ops_decl_const("dy",1,"double",&dy);
  ops_decl_const("dz",1,"double",&dz);

  //declare blocks
  ops_block block = ops_decl_block(3,"block");
  

  //declare stencils
  int s3D_000[]         = {0,0,0};
  ops_stencil S3D_000 = ops_decl_stencil( 3, 1, s3D_000, "000");
  int s3D_000_P100_M100_0P10_0M10_00P1_00M1[] =
      {0,0,0, 1,0,0, -1,0,0, 0,1,0, 0,-1,0, 0,0,1, 0,0,-1};
  ops_stencil S3D_000_P100_M100_0P10_0M10_00P1_00M1 = 
      ops_decl_stencil( 3, 7, s3D_000_P100_M100_0P10_0M10_00P1_00M1, "000:100:-100:010:0-10:001:00-1");

  ops_reduction red_err = ops_decl_reduction_handle(sizeof(double), "double", "err");

  //declare datasets
  int d_p[3] = {1,1,1}; //max halo depths for the dat in the possitive direction
  int d_m[3] = {-1,-1,-1}; //max halo depths for the dat in the negative direction
  int base[3] = {0,0,0};
  int size[3] = {logical_size_x,logical_size_y,logical_size_z};
  double* temp = NULL;
  ops_dat u = ops_decl_dat(block, 1, size, base, d_m, d_p, temp, "double", "u");
  ops_dat u2 = ops_decl_dat(block, 1, size, base, d_m, d_p, temp, "double", "u2");
  ops_dat f = ops_decl_dat(block, 1, size, base, d_m, d_p, temp, "double", "f");
  ops_dat ref = ops_decl_dat(block, 1, size, base, d_m, d_p, temp, "double", "ref");
  

  ops_partition("");
	ops_diagnostic_output();
  /**-------------------------- Computations --------------------------**/


  double ct0, ct1, et0, et1;
  ops_timers(&ct0, &et0);
  int iter_range_full[] = {-1,logical_size_x+1,-1,logical_size_y+1,-1,logical_size_z+1};
  int iter_range[] = {0,logical_size_x,0,logical_size_y,0,logical_size_z};

  ops_par_loop(poisson_kernel_populate, "poisson_kernel_populate", block, 3, iter_range_full,
               ops_arg_idx(),
               ops_arg_dat(u, 1, S3D_000, "double", OPS_WRITE),
               ops_arg_dat(f, 1, S3D_000, "double", OPS_WRITE),
               ops_arg_dat(ref, 1, S3D_000, "double", OPS_WRITE));

	ops_par_loop(poisson_kernel_update, "poisson_kernel_update", block, 3, iter_range_full,
							ops_arg_dat(u, 1, S3D_000, "double", OPS_READ),
							ops_arg_dat(u2 , 1, S3D_000, "double", OPS_WRITE));


  //initial guess 0
  ops_par_loop(poisson_kernel_initialguess, "poisson_kernel_initialguess", block, 3, iter_range,
               ops_arg_dat(u, 1, S3D_000, "double", OPS_WRITE));

  double it0, it1;
  ops_timers(&ct0, &it0);

  for (int iter = 0; iter < n_iter; iter++) {
    if (iter%itertile == 0) ops_execute(block->instance);


    ops_par_loop(poisson_kernel_stencil, "poisson_kernel_stencil", block, 3, iter_range,
             ops_arg_dat(u, 1, S3D_000_P100_M100_0P10_0M10_00P1_00M1, "double", OPS_READ),
             ops_arg_dat(u2, 1, S3D_000, "double", OPS_WRITE));

		if (non_copy) {
			
			ops_par_loop(poisson_kernel_stencil, "poisson_kernel_stencil", block, 3, iter_range,
							ops_arg_dat(u2, 1, S3D_000_P100_M100_0P10_0M10_00P1_00M1, "double", OPS_READ),
							ops_arg_dat(u, 1, S3D_000, "double", OPS_WRITE));
			
		} else {
			ops_par_loop(poisson_kernel_update, "poisson_kernel_update", block, 3, iter_range,
							ops_arg_dat(u2, 1, S3D_000, "double", OPS_READ),
							ops_arg_dat(u , 1, S3D_000, "double", OPS_WRITE));
			}
  }
	ops_execute(block->instance);
  ops_timers(&ct0, &it1);

  //ops_print_dat_to_txtfile(u[0], "poisson.dat");
  //ops_print_dat_to_txtfile(ref[0], "poisson.dat");
  //exit(0);

  double err = 0.0;
  
  ops_par_loop(poisson_kernel_error, "poisson_kernel_error", block, 3, iter_range,
               ops_arg_dat(u, 1, S3D_000, "double", OPS_READ),
               ops_arg_dat(ref , 1, S3D_000, "double", OPS_READ),
               ops_arg_reduce(red_err, 1, "double", OPS_INC));
  
  ops_reduction_result(red_err,&err);

  ops_timers(&ct1, &et1);
  ops_timing_output(std::cout);
  ops_printf("\nTotal Wall time %lf\n",et1-et0);
  double err_diff=fabs((100.0*(err/228976.30141736))-100.0);
  ops_printf("Total error: %3.15g\n",err);
  ops_printf("Total error is within %3.15E %% of the expected error\n",err_diff);

  if(err_diff < 0.001) {
    ops_printf("This run is considered PASSED\n");
  }
  else {
    ops_printf("This test is considered FAILED\n");
  }

  ops_exit();
  return 0;
}
