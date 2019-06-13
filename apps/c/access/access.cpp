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
  ops_dat_get_extents(data0, 0, disp, sizes);
  double *raw_data = (double*)ops_dat_get_raw_pointer(data0, 0, S2D_5pt, strides);
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

  ops_timers(&ct1, &et1);
  ops_timing_output(stdout);

  ops_printf("\nTotal Wall time %lf\n",et1-et0);

  ops_exit();
}
