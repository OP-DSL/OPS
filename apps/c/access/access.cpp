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

// OPS header file
#define OPS_2D
#include "ops_seq.h"

#include "access_populate_kernel.h"

void fetch_test(ops_dat data);

/******************************************************************************
* Main program
*******************************************************************************/
int main(int argc, char **argv)
{
  /**-------------------------- Initialisation --------------------------**/

  // OPS initialisation
  ops_init(argc,argv,2);

  //declare blocks
  ops_block grid0 = ops_decl_block(2, "grid0");

  //declare stencils
  int s2D_00[]         = {0,0};
  int s2D_5pt[]          = {0,0, 1,0, -1,0, 0,1, 0,-1};
  ops_stencil S2D_00 = ops_decl_stencil( 2, 1, s2D_00, "00");
  ops_stencil S2D_5pt = ops_decl_stencil( 2, 5, s2D_5pt, "5pt");

  //declare datasets
  int d_p[2] = {1,1}; //max halo depths for the dat in the possitive direction
  int d_m[2] = {-1,-1}; //max halo depths for the dat in the negative direction
  int size[2] = {20, 20}; //size of the dat
  int base[2] = {0,0};
  double* temp = NULL;

  ops_dat data0 = ops_decl_dat(grid0, 1, size, base, d_m, d_p, temp, "double", "data0");

  ops_partition("");

  /**-------------------------- Computations --------------------------**/


  double ct0, ct1, et0, et1;
  ops_timers(&ct0, &et0);

  //populate
  int iter_range[] = {0,20,0,20};
  ops_par_loop(access_populate_kernel, "access_populate_kernel", grid0, 2, iter_range,
               ops_arg_dat(data0, 1, S2D_00, "double", OPS_WRITE),
               ops_arg_idx());

  int sizes[2], disp[2], strides[2];
  int test_slab_range[] = {6, 8, 6, 8};
  size_t bytes = ops_dat_get_slab_extents(data0, 0, disp, sizes, test_slab_range);
  printf("hyperslab local displacement %d,%d, size %d,%d, bytes: %ld\n",disp[0], disp[1], sizes[0], sizes[1], bytes);

  ops_memspace memspace = OPS_HOST;
  ops_dat_get_raw_metadata(data0, 0, disp, sizes, strides, NULL, NULL);

  double *raw_data = (double*)ops_dat_get_raw_pointer(data0, 0, S2D_5pt, &memspace);

  //Insert some value at coordinates 10,10, reading neighbours
  if (10 >= disp[0] && 10 < disp[0]+sizes[0] &&
      10 >= disp[1] && 10 < disp[1]+sizes[1]) {
    raw_data[(10-disp[0]) + (10-disp[1])*strides[0]] = 
        raw_data[(10-disp[0]    ) + (10-disp[1]    )*strides[0]] +
        raw_data[(10-disp[0] + 1) + (10-disp[1]    )*strides[0]] +
        raw_data[(10-disp[0] - 1) + (10-disp[1]    )*strides[0]] +
        raw_data[(10-disp[0]    ) + (10-disp[1] + 1)*strides[0]] +
        raw_data[(10-disp[0]    ) + (10-disp[1] - 1)*strides[0]];
  }
  ops_dat_release_raw_data(data0, 0, OPS_RW);

  ops_print_dat_to_txtfile(data0, "data0.txt");

  fetch_test(data0);

  double *slab = (double*)malloc(4*sizeof(double));
  int slab_range[] = {10,12,10,12};
  ops_dat_fetch_data_slab_memspace(data0, 0, (char*)slab, slab_range, OPS_HOST);
  ops_printf("2D slab extracted on HOST:\n%g %g\n%g %g\n", slab[0], slab[1], slab[2], slab[3]);
  free(slab);

  ops_timers(&ct1, &et1);
  ops_timing_output(std::cout);

  ops_printf("\nTotal Wall time %lf\n",et1-et0);

  ops_exit();
}
