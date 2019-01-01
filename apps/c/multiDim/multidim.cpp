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

/** @Test application for multi-dimensional ops_dats functionality
  * @author Gihan Mudalige, Istvan Reguly
  */

// standard headers
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

// OPS header file
#define OPS_2D
#define OPS_SOA
#include "ops_seq_variadic.h"

#include "multidim_kernel.h"
#include "multidim_print_kernel.h"
#include "multidim_copy_kernel.h"
#include "multidim_reduce_kernel.h"

int main(int argc, const char **argv)
{
  //initialize sizes using global values
  int x_cells = 4;
  int y_cells = 4;

  /**-------------------------- Initialisation --------------------------**/

  // OPS initialisation
  ops_init(argc,argv,1);
  OPS_soa = 1;

  /**----------------------------OPS Declarations----------------------------**/

  //declare block
  ops_block grid2D = ops_decl_block(2, "grid2D");

  //declare stencils
  int s2D_00[]         = {0,0};
  int s2D_00_P10_M10[]         = {0,0,1,0,-1,0};
  int s2D_00_P10_P20_M10_M20[]         = {0,0,1,0,2,0,-1,0,-2,0};
  ops_stencil S2D_00 = ops_decl_stencil( 2, 1, s2D_00, "00");
  ops_stencil S2D_00_P10_M10 = ops_decl_stencil( 2, 3, s2D_00_P10_M10, "00:10:-10");
  ops_stencil S2D_00_P10_P20_M10_M20 = ops_decl_stencil( 2, 5, s2D_00_P10_P20_M10_M20, "00:10:20:-10:-20");


  //declare data on blocks
  int d_p[2] = {2,1}; //max halo depths for the dat in the possitive direction
  int d_m[2] = {-2,-1}; //max halo depths for the dat in the negative direction
  int size[2] = {x_cells, y_cells}; //size of the dat -- should be identical to the block on which its define on
  int base[2] = {0,0};
  double* temp = NULL;

  //declare ops_dat with dim = 2
  ops_dat dat0    = ops_decl_dat(grid2D, 2, size, base, d_m, d_p, temp, "double", "dat0");
  ops_dat dat1    = ops_decl_dat(grid2D, 2, size, base, d_m, d_p, temp, "double", "dat1");

  ops_halo_group halos0;
  {
    int halo_iter[] = {1,4};
    int base_from[] = {3,0};
    int base_to[] = {-1,0};
    int dir[] = {1,2};
    ops_halo h0 = ops_decl_halo(dat0, dat0, halo_iter, base_from, base_to, dir, dir);
    base_from[0] = 0; base_to[0] = 4;
    ops_halo h1 = ops_decl_halo(dat0, dat0, halo_iter, base_from, base_to, dir, dir);
    ops_halo grp[] = {h0,h1};
    halos0 = ops_decl_halo_group(2,grp);
  }


  //declare reduction handles
  double reduct_result[2] = {0.0, 0.0};
  ops_reduction reduct_dat1 = ops_decl_reduction_handle(2*sizeof(double), "double", "reduct_dat1");

  //decompose the block
  ops_partition("2D_BLOCK_DECOMPSE");

  double ct0, ct1, et0, et1;
  ops_timers(&ct0, &et0);

  int iter_range[] = {0,4,0,4};
  ops_par_loop(multidim_kernel, "multidim_kernel", grid2D, 2, iter_range,
               ops_arg_dat(dat0, 2, S2D_00, "double", OPS_WRITE),
               ops_arg_idx());
  ops_par_loop(multidim_copy_kernel,"multidim_copy_kernel", grid2D, 2, iter_range,
               ops_arg_dat(dat0, 2, S2D_00_P10_P20_M10_M20, "double", OPS_READ),
               ops_arg_dat(dat1, 2, S2D_00, "double", OPS_WRITE));
  ops_halo_transfer(halos0);

  //ops_printf("\n\n");
  //ops_par_loop(multidim_print_kernel,"multidim_print_kernel", grid2D, 2, iter_range,
  //             ops_arg_dat(dat1, 2, S2D_00, "double", OPS_READ));

  ops_par_loop(multidim_reduce_kernel,"multidim_reduce_kernel", grid2D, 2, iter_range,
               ops_arg_dat(dat1, 2, S2D_00, "double", OPS_READ),
               ops_arg_reduce(reduct_dat1, 2, "double", OPS_INC));

  ops_reduction_result(reduct_dat1, reduct_result);

  ops_timers(&ct1, &et1);
  ops_print_dat_to_txtfile(dat0, "multidim.dat");

  //ops_fetch_block_hdf5_file(grid2D, "multidim.h5");
//  ops_fetch_dat_hdf5_file(dat0, "multidim.h5");

  ops_printf("\nTotal Wall time %lf\n",et1-et0);
  double result_diff=fabs((100.0*((reduct_result[0]+reduct_result[0])/(2*24.000000)))-100.0);
  ops_printf("Reduction result = %lf, %lf\n", reduct_result[0],reduct_result[1]);
  ops_printf("Result is within %3.15E %% of the expected result\n",result_diff);

  if(result_diff < 0.0000000000001) {
    ops_printf("This run is considered PASSED\n");
  }
  else {
    ops_printf("This test is considered FAILED\n");
  }


  ops_exit();
}
