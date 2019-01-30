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

double dx,dy;


// OPS header file
#define OPS_2D
#include "ops_seq_v2.h"
#include "user_types.h"
#include "poisson_kernel.h"

/******************************************************************************
* Main program
*******************************************************************************/
int main(int argc, const char **argv)
{
  /**-------------------------- Initialisation --------------------------**/

  // OPS initialisation
  ops_init(argc,argv,1);


  //Mesh
  int logical_size_x = 100;
  int logical_size_y = 100;
  int n_iter = 100;
  int itertile = n_iter;
  int non_copy = 0;
  int num_systems = 100;

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
    pch = strstr(argv[n], "-batch=");
    if(pch != NULL) {
      num_systems = atoi ( argv[n] + 7 ); continue;
    }
  }

  ops_printf("Grid: %dx%d, %d batch size %d iterations, %d tile height\n",logical_size_x,logical_size_y,num_systems, n_iter,itertile);
  dx = 0.01;
  dy = 0.01;
  ops_decl_const("dx",1,"double",&dx);
  ops_decl_const("dy",1,"double",&dy);

  //declare block
  ops_block block = ops_decl_block_batch(2,"block",num_systems,OPS_BATCHED);

  //declare stencils
  int s2D_00[]         = {0,0};
  ops_stencil S2D_00 = ops_decl_stencil( 2, 1, s2D_00, "00");
  int s2D_00_P10_M10_0P1_0M1[]         = {0,0, 1,0, -1,0, 0,1, 0,-1};
  ops_stencil S2D_00_P10_M10_0P1_0M1 = ops_decl_stencil( 2, 5, s2D_00_P10_M10_0P1_0M1, "00:10:-10:01:0-1");

  ops_reduction red_err = ops_decl_reduction_handle_batch(sizeof(double), "double", "err", num_systems);

  //declare datasets
  int d_p[2] = {1,1}; //max halo depths for the dat in the possitive direction
  int d_m[2] = {-1,-1}; //max halo depths for the dat in the negative direction
  int base[2] = {0,0};
  int size[2] = {logical_size_x,logical_size_y};
  double* temp = NULL;
  ops_dat coordx = ops_decl_dat(block, 1, size, base, d_m, d_p, temp,"double", "coordx");
  ops_dat coordy = ops_decl_dat(block, 1, size, base, d_m, d_p, temp,"double", "coordy");
  ops_dat u = ops_decl_dat(block, 1, size, base, d_m, d_p, temp,"double", "u");
  ops_dat u2 = ops_decl_dat(block, 1, size, base, d_m, d_p, temp,"double", "u2");
  ops_dat f = ops_decl_dat(block, 1, size, base, d_m, d_p, temp,"double", "f");
  ops_dat ref = ops_decl_dat(block, 1, size, base, d_m, d_p, temp,"double", "ref");

  ops_partition("");
	ops_diagnostic_output();
  /**-------------------------- Computations --------------------------**/


  double ct0, ct1, et0, et1;
  ops_timers(&ct0, &et0);

  ops_par_loop_blocks_all(num_systems);

  //populate forcing, reference solution and boundary conditions
  int iter_range_full[] = {-1,size[0]+1,-1,size[1]+1};

  ops_par_loop(poisson_kernel_populate, "poisson_kernel_populate", block, 2, iter_range_full,
      ops_arg_idx(),
      ops_arg_dat(u, 1, S2D_00, "double", OPS_WRITE),
      ops_arg_dat(f, 1, S2D_00, "double", OPS_WRITE),
      ops_arg_dat(ref, 1, S2D_00, "double", OPS_WRITE));

  ops_par_loop(poisson_kernel_update, "poisson_kernel_update", block, 2, iter_range_full,
      ops_arg_dat(u, 1, S2D_00, "double", OPS_READ),
      ops_arg_dat(u2 , 1, S2D_00, "double", OPS_WRITE));



  //initial guess 0
  int iter_range[] = {0,size[0],0,size[1]};

  ops_par_loop(poisson_kernel_initialguess, "poisson_kernel_initialguess", block, 2, iter_range,
      ops_arg_dat(u, 1, S2D_00, "double", OPS_WRITE));

  double it0, it1;
  ops_timers(&ct0, &it0);

  for (int iter = 0; iter < n_iter; iter++) {
    if (iter%itertile == 0) ops_execute();

    ops_par_loop(poisson_kernel_stencil, "poisson_kernel_stencil", block, 2, iter_range,
        ops_arg_dat(u, 1, S2D_00_P10_M10_0P1_0M1, "double", OPS_READ),
        ops_arg_dat(u2, 1, S2D_00, "double", OPS_WRITE));

    if (non_copy) {
      ops_par_loop(poisson_kernel_stencil, "poisson_kernel_stencil", block, 2, iter_range,
          ops_arg_dat(u2, 1, S2D_00_P10_M10_0P1_0M1, "double", OPS_READ),
          ops_arg_dat(u, 1, S2D_00, "double", OPS_WRITE));
    } else {
      ops_par_loop(poisson_kernel_update, "poisson_kernel_update", block, 2, iter_range,
          ops_arg_dat(u2, 1, S2D_00, "double", OPS_READ),
          ops_arg_dat(u , 1, S2D_00, "double", OPS_WRITE));
    }
  }
  ops_execute();
  ops_timers(&ct0, &it1);

  ops_par_loop(poisson_kernel_error, "poisson_kernel_error", block, 2, iter_range,
      ops_arg_dat(u, 1, S2D_00, "double", OPS_READ),
      ops_arg_dat(ref , 1, S2D_00, "double", OPS_READ),
      ops_arg_reduce(red_err, 1, "double", OPS_INC));

  ops_par_loop_blocks_end();

  double *err = new double[num_systems];
  memset(err, 0, num_systems*sizeof(double));
  ops_reduction_result(red_err,err);

  ops_timers(&ct1, &et1);
  ops_timing_output(stdout);
  ops_printf("\nTotal Wall time %lf\n",et1-et0);
  double err_diff = 0.0;
  int i = 0;
  for (i = 0; i < num_systems; i++) {
    err_diff=fabs((100.0*(err[i]/2008.72990634426))-100.0);
    if (err_diff > 0.001) {
      break;
    }
  }
  ops_printf("Total error: %3.15g\n",err[MIN(num_systems-1,i)]);
  ops_printf("Total error is within %3.15E %% of the expected error\n",err_diff);
  delete[] err;

  if(err_diff < 0.001) {
    ops_printf("This run is considered PASSED\n");
  }
  else {
    ops_printf("This test is considered FAILED\n");
  }

  ops_printf("%lf\n",it1-it0);

  ops_exit();
  return 0;
}
