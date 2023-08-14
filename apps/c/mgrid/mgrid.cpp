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

/** @Test application for multi-grid functionality
  * @author Gihan Mudalige, Istvan Reguly
  */

// standard headers
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

// OPS header file
#define OPS_2D
#include "ops_seq.h"

//#include "mgrid_populate_kernel.h"
//#include "mgrid_restrict_kernel.h"
//#include "mgrid_prolong_kernel.h"
#include "mgrid_kernels.h"

/******************************************************************************
* Main program
/******************************************************************************/
int main(int argc, const char **argv)
{
  /**-------------------------- Initialisation --------------------------**/

  // OPS initialisation
  ops_init(argc,argv,2);

  //declare blocks
  ops_block grid0 = ops_decl_block(2, "grid0");


  //declare stencils
  int s2D_00[]         = {0,0};
  int s2D_00_M10_P10[]         = {0,0,-1,0,1,0};
  int s2D_5pt[]         = {0,0,-1,0,1,0,0,-1,0,1};
  ops_stencil S2D_00 = ops_decl_stencil( 2, 1, s2D_00, "00");
  ops_stencil S2D_5pt = ops_decl_stencil( 2, 5, s2D_5pt, "5pt");

  int fac = 1;

  //declare datasets
  int d_p[2] = {2,2};
  //int d_p[2] = {0,0};
  int d_m[2] = {-2,-2};
  //int d_m[2] = {0,0};
  int size4[2] = {24*fac, 24*fac};
  int size0[2] = {12*fac, 12*fac};
  int size1[2] = {6*fac, 6*fac};
  int size2[2] = {4*fac, 4*fac};

  int size3[2] = {6*fac, 6*fac};

  int stride0[2] = {1, 1};
  int stride1[2] = {2, 2};
  int stride2[2] = {3, 3};
  int stride3[2] = {4, 4};
  //declare restrict and prolong stencils
  //ops_stencil S2D_RESTRICT_00 = ops_decl_restrict_stencil( 2, 1, s2D_00, stride1, "RESTRICT_00");
  ops_stencil S2D_RESTRICT_00 = ops_decl_restrict_stencil( 2, 1, s2D_00, stride1, "RESTRICT_00");
  //ops_stencil S2D_PROLONG_00 = ops_decl_prolong_stencil( 2, 1, s2D_00, stride1, "PROLONG_00");
  ops_stencil S2D_PROLONG_00 = ops_decl_prolong_stencil( 2, 1, s2D_00, stride1, "PROLONG_00");
  ops_stencil S2D_PROLONG_00_M10_P10 = ops_decl_prolong_stencil( 2, 3, s2D_00_M10_P10, stride1, "PROLONG_00_M10_P10");
  ops_stencil S2D_RESTRICT_00_M10_P10 = ops_decl_restrict_stencil( 2, 3, s2D_00_M10_P10, stride1, "RESTRICT_00_M10_P10");
#define ZEROBASE
#ifdef ZEROBASE
  int base[2] = {0,0};
#else
  int base[2] = {-1,-1};
#endif
  double* temp = NULL;

  ops_dat data0 = ops_decl_dat(grid0, 1, size0, base, d_m, d_p, stride1 , temp, "double", "data0");
  ops_dat data1 = ops_decl_dat(grid0, 1, size1, base, d_m, d_p, stride3 , temp, "double", "data1");
  //ops_dat data2 = ops_decl_dat(grid0, 1, size2, base, d_m, d_p, stride2 , temp, "double", "data2");
  ops_dat data5 = ops_decl_dat(grid0, 1, size4, base, d_m, d_p, stride0, temp, "double", "data5");
  ops_dat data6 = ops_decl_dat(grid0, 1, size0, base, d_m, d_p, stride1 , temp, "double", "data6");

  ops_dat data3 = ops_decl_dat(grid0, 1, size1, base, d_m, d_p, stride3 , temp, "double", "data3");
  //ops_dat data4 = ops_decl_dat(grid0, 1, size3, base, d_m, d_p, stride2 , temp, "double", "data4");

  ops_reduction reduct_err = ops_decl_reduction_handle(sizeof(int), "int", "reduct_err");

  ops_halo_group halos[4];
  {
    int halo_iter[] = {2, size4[1]+4};
    int from_base[] = {0,-2};
    int to_base[] = {size4[0],-2};
    int dir[] = {1,2};
    ops_halo halo1 = ops_decl_halo(data5, data5, halo_iter, from_base, to_base, dir, dir);
    from_base[0] = size4[0]-2;
    to_base[0] = -2;
    ops_halo halo2 = ops_decl_halo(data5, data5, halo_iter, from_base, to_base, dir, dir);
    ops_halo halog1[] = {halo1,halo2};
    halos[0] = ops_decl_halo_group(2,halog1);

    int halo_iter2[] = {size4[0]+4,2};
    int from_base2[] = {-2,0};
    int to_base2[] = {-2,size4[1]};
    ops_halo halo1_2 = ops_decl_halo(data5, data5, halo_iter2, from_base2, to_base2, dir, dir);
    from_base2[1] = size4[1]-2;
    to_base2[1] = -2;
    ops_halo halo2_2 = ops_decl_halo(data5, data5, halo_iter2, from_base2, to_base2, dir, dir);
    ops_halo halog1_2[] = {halo1_2,halo2_2};
    halos[1] = ops_decl_halo_group(2,halog1_2);

    halo_iter[1] = size0[1]+4;
    from_base[0] = 0;
    to_base[0] = size0[1];
    ops_halo halo3 = ops_decl_halo(data0, data0, halo_iter, from_base, to_base, dir, dir);
    from_base[0] = size0[0]-2;
    to_base[0] = -2;
    ops_halo halo4 = ops_decl_halo(data0, data0, halo_iter, from_base, to_base, dir, dir);
    ops_halo halog2[] = {halo3,halo4};
    halos[2] = ops_decl_halo_group(2,halog2);

    halo_iter[1] = size1[1]+4;
    from_base[0] = 0;
    to_base[0] = size1[1];
    ops_halo halo5 = ops_decl_halo(data1, data1, halo_iter, from_base, to_base, dir, dir);
    from_base[0] = size1[0]-2;
    to_base[0] = -2;
    ops_halo halo6 = ops_decl_halo(data1, data1, halo_iter, from_base, to_base, dir, dir);
    ops_halo halog3[] = {halo5,halo6};
    halos[3] = ops_decl_halo_group(2,halog3);
  }
  ops_partition("");


  /**-------------------------- Computations --------------------------**/

  double ct0, ct1, et0, et1;
  ops_timers_core(&ct0, &et0);
#ifdef ZEROBASE
  int iter_range[] = {0,12,0,12};
  int iter_range_large[] = {0,24,0,24};
  int iter_range_small[] = {0,6,0,6};
  int iter_range_tiny[] = {0,4,0,4};
#else
  int iter_range[] = {-1,11,-1,11};
  int iter_range_large[] = {-1,23,-1,23};
  int iter_range_small[] = {-1,5,-1,5};
  int iter_range_tiny[] = {-1,3,-1,3};
#endif

  ops_par_loop(mgrid_populate_kernel_1, "mgrid_populate_kernel_1", grid0, 2, iter_range_small,
               ops_arg_dat(data1, 1, S2D_00, "double", OPS_WRITE),
               ops_arg_idx());
  ops_halo_transfer(halos[3]);
  //ops_print_dat_to_txtfile(data1, "data.txt");
  /*ops_par_loop(mgrid_populate_kernel_2, "mgrid_populate_kernel_2", grid0, 2, iter_range_tiny,
               ops_arg_dat(data2, 1, S2D_00, "double", OPS_WRITE),
               ops_arg_idx());*/

  ops_par_loop(mgrid_prolong_kernel, "mgrid_prolong_kernel", grid0, 2, iter_range,
               //ops_arg_dat(data2, 1, S2D_PROLONG_00, "double", OPS_READ),
               ops_arg_dat(data1, 1, S2D_PROLONG_00_M10_P10, "double", OPS_READ),
               ops_arg_dat(data0, 1, S2D_00, "double", OPS_WRITE),
               ops_arg_idx());
  ops_halo_transfer(halos[2]);

  ops_par_loop(mgrid_prolong_kernel, "mgrid_prolong_kernel", grid0, 2, iter_range_large,
               //ops_arg_dat(data2, 1, S2D_PROLONG_00, "double", OPS_READ),
               ops_arg_dat(data0, 1, S2D_PROLONG_00_M10_P10, "double", OPS_READ),
               ops_arg_dat(data5, 1, S2D_00, "double", OPS_WRITE),
               ops_arg_idx());
  ops_halo_transfer(halos[0]);
  ops_halo_transfer(halos[1]);

  int size4_0 = size4[0];
  int size4_1 = size4[1];
  ops_par_loop(prolong_check, "prolong_check", grid0, 2, iter_range_large,
               ops_arg_dat(data5, 1, S2D_5pt, "double", OPS_READ),
               ops_arg_idx(),
               ops_arg_reduce(reduct_err, 1, "int", OPS_MAX),
               ops_arg_gbl(&size4_0, 1, "int", OPS_READ),
               ops_arg_gbl(&size4_1, 1, "int", OPS_READ));

  int err_prolong = 0;
  ops_reduction_result(reduct_err, &err_prolong);

  ops_fetch_block_hdf5_file(grid0, "data.h5");
  ops_fetch_dat_hdf5_file(data5, "data.h5");

  //ops_print_dat_to_txtfile(data2, "data.txt");
  //ops_print_dat_to_txtfile(data0, "data.txt");
  //ops_print_dat_to_txtfile(data5, "data.txt");

  ops_par_loop(mgrid_populate_kernel_3, "mgrid_populate_kernel_3", grid0, 2, iter_range_large,
               ops_arg_dat(data5, 1, S2D_00, "double", OPS_WRITE),
               ops_arg_idx());

  ops_par_loop(mgrid_restrict_kernel, "mgrid_restrict_kernel", grid0, 2, iter_range,
               ops_arg_dat(data5, 1, S2D_RESTRICT_00_M10_P10, "double", OPS_READ),
               ops_arg_dat(data6, 1, S2D_00, "double", OPS_WRITE),
               ops_arg_idx());
  ops_par_loop(mgrid_restrict_kernel, "mgrid_restrict_kernel", grid0, 2, iter_range_small,
               ops_arg_dat(data6, 1, S2D_RESTRICT_00_M10_P10, "double", OPS_READ),
               ops_arg_dat(data3, 1, S2D_00, "double", OPS_WRITE),
               ops_arg_idx());

  /*ops_par_loop(mgrid_restrict_kernel, "mgrid_restrict_kernel", grid0, 2, iter_range_tiny,
               ops_arg_dat(data0, 1, S2D_RESTRICT_00, "double", OPS_READ),
               ops_arg_dat(data4, 1, S2D_00, "double", OPS_WRITE),
               ops_arg_idx());*/

  /*ops_print_dat_to_txtfile(data5, "data.txt");
  ops_print_dat_to_txtfile(data6, "data.txt");
  ops_print_dat_to_txtfile(data3, "data.txt");*/

  ops_par_loop(restrict_check, "restrict_check", grid0, 2, iter_range_small,
               ops_arg_dat(data3, 1, S2D_00, "double", OPS_READ),
               ops_arg_idx(),
               ops_arg_reduce(reduct_err, 1, "int", OPS_MAX),
               ops_arg_gbl(&size4_0, 1, "int", OPS_READ));

  int err_restrict = 0;
  ops_reduction_result(reduct_err, &err_restrict);


  ops_timers_core(&ct1, &et1);
  ops_timing_output(std::cout);

  ops_printf("\nTotal Wall time %lf\n",et1-et0);

  ops_fetch_dat_hdf5_file(data6, "data.h5");
  ops_fetch_dat_hdf5_file(data3, "data.h5");

  if (err_prolong==0 && err_restrict ==0) ops_printf("\nPASSED\n");
  else ops_printf("\nFAILED\n");

  ops_exit();
}
