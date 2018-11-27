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

/** @Application for testing muti-dimension ops_dats being written to HDF5 files
  * @author Piotr Zacharzewski and Gihan Mudalige
  */

#define OPS_3D
#include "ops_seq.h"

#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "write_kernel.h"

int main(int argc, char **argv) {

  //*******************************************************************
  // INITIALISE OPS
  //---------------------------------------
  ops_init(argc, argv, 5);
  ops_printf("Initialize OPS\n\n");
  //*******************************************************************

  ops_block grid0 = ops_decl_block(3, "grid0");

  int d_p[3] = {1, 1, 0};
  int d_m[3] = {-1, -1, 0};
  int size[3] = {4, 5, 1}; // size of the dat
  int base[3] = {0, 0, 0};

  double *temp = NULL;
  int *tempi = NULL;

  ops_dat single =
      ops_decl_dat(grid0, 1, size, base, d_m, d_p, temp, "double", "single");
  ops_dat multi =
      ops_decl_dat(grid0, 2, size, base, d_m, d_p, temp, "double", "multi");
  ops_dat integ =
      ops_decl_dat(grid0, 1, size, base, d_m, d_p, tempi, "int", "integ");

  int range_full[6];
  range_full[0] = 0;
  range_full[1] = 4;
  range_full[2] = 0;
  range_full[3] = 5;
  range_full[4] = 0;
  range_full[5] = 1;

  int s3D_000[] = {0, 0, 0};
  ops_stencil S3D_000 = ops_decl_stencil(3, 1, s3D_000, "0,0,0");

  ops_partition("empty_string_that_does_nothing_yet");
  ops_diagnostic_output();

  ops_par_loop(write_kernel, "write_kernel", grid0, 3, range_full,
               ops_arg_dat(multi, 2, S3D_000, "double", OPS_WRITE),
               ops_arg_dat(single, 1, S3D_000, "double", OPS_WRITE),
               ops_arg_dat(integ, 1, S3D_000, "int", OPS_WRITE), ops_arg_idx());

  ops_fetch_block_hdf5_file(grid0, "write_data.h5");

  ops_fetch_dat_hdf5_file(multi, "write_data.h5");
  ops_fetch_dat_hdf5_file(single, "write_data.h5");
  ops_fetch_dat_hdf5_file(integ, "write_data.h5");

  int my_const = 42;
  ops_write_const_hdf5("my_const", 1, "int", (char*)&my_const, "write_data.h5");

  ops_print_dat_to_txtfile(integ, "integers.txt");

  //*******************************************************************
  // EXIT OPS AND PRINT TIMING INFO
  //---------------------------------------
  ops_timing_output(stdout);
  ops_printf("\nSucessful Exit from OPS!\n");
  ops_exit();
  // return 0;
  //*******************************************************************
}
