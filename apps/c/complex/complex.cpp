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
#include <complex>

// OPS header file
#define OPS_2D
#include "ops_seq.h"

#include "complex_populate_kernel.h"

/******************************************************************************
* Main program
/******************************************************************************/
int main(int argc, char **argv)
{
  /**-------------------------- Initialisation --------------------------**/

  // OPS initialisation
  ops_init(argc,argv,2);

  //declare blocks
  ops_block grid0 = ops_decl_block(2, "grid0");


  //declare stencils
  int s2D_00[]         = {0,0};
  ops_stencil S2D_00 = ops_decl_stencil( 2, 1, s2D_00, "00");

  //declare datasets
  int d_p[2] = {2,2};
  int d_m[2] = {-2,-2};
  int size1[2] = {6, 6};

  int stride0[2] = {1, 1};
  int base[2] = {0,0};
  complexd* temp = NULL;

  ops_dat data1 = ops_decl_dat(grid0, 1, size1, base, d_m, d_p, stride0 , temp, "complexd", "data1");

  ops_reduction red = ops_decl_reduction_handle(sizeof(complexd), "complexd", "reduction");

  ops_partition("");


  /**-------------------------- Computations --------------------------**/

  double ct0, ct1, et0, et1;
  ops_timers_core(&ct0, &et0);
  int iter_range_small[] = {0,6,0,6};

  ops_par_loop(complex_populate_kernel, "complex_populate_kernel", grid0, 2, iter_range_small,
               ops_arg_dat(data1, 1, S2D_00, "complexd", OPS_WRITE),
               ops_arg_reduce(red, 1, "complexd", OPS_INC),
               ops_arg_idx());
  ops_print_dat_to_txtfile(data1, "data.txt");

  complexd redval;
  ops_reduction_result(red, &redval);

  ops_printf("reduction result: %g+%gi\n",std::real(redval), std::imag(redval));
  ops_timers_core(&ct1, &et1);
  ops_timing_output(std::cout);

  ops_printf("\nTotal Wall time %lf\n",et1-et0);

  if (std::real(redval) == 90.0 &&
      std::imag(redval) == 90.0)
    ops_printf("PASSED\n");
  else
    ops_printf("FAILED\n");

  ops_exit();
}
