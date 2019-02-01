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
#define OPS_4D
#include "ops_seq_v2.h"

#include "mblock_populate_kernel.h"

/******************************************************************************
* Main program
*******************************************************************************/
int main(int argc, char **argv)
{
  /**-------------------------- Initialisation --------------------------**/

  // OPS initialisation
  ops_init(argc,argv,2);

  //declare blocks
  ops_block grid0 = ops_decl_block(4, "grid0");
  ops_block grid1 = ops_decl_block(4, "grid1");

  //declare stencils
  int s4D_0000[]         = {0,0,0,0};
  ops_stencil S4D_0000 = ops_decl_stencil( 4, 1, s4D_0000, "0000");

  //declare datasets
  int d_p[] = {1,1,1,1}; //max halo depths for the dat in the possitive direction
  int d_m[] = {-1,-1,-1,-1}; //max halo depths for the dat in the negative direction
  int size[] = {4,4,4,4}; //size of the dat
  int base[] = {0,0,0,0};
  double* temp = NULL;

  ops_dat data0 = ops_decl_dat(grid0, 1, size, base, d_m, d_p, temp, "double", "data0");
  ops_dat data1 = ops_decl_dat(grid1, 1, size, base, d_m, d_p, temp, "double", "data1");


  //straightforward matching orientation halos data0 - data1 in x
  //last two x lines of data0 and first two of data1
  ops_halo_group halos0;
  {
    int halo_iter[] = {1,4,4,4};
    int base_from[] = {3,0,0,0};
    int base_to[] = {-1,0,0,0};
    int dir[] = {1,2,3,4};
    ops_halo h0 = ops_decl_halo(data0, data1, halo_iter, base_from, base_to, dir, dir);
    //ops_halo h0 = ops_decl_halo_hdf5(data0, data1,"halo_file.h5");
    base_from[0] = 0; base_to[0] = 4;
    ops_halo h1 = ops_decl_halo(data1, data0, halo_iter, base_from, base_to, dir, dir);
    ops_halo grp[] = {h0,h1};
    halos0 = ops_decl_halo_group(2,grp);
    //ops_fetch_halo_hdf5_file(h0,"halo_file.h5");
  }


  ops_partition("");

  /**-------------------------- Computations --------------------------**/


  double ct0, ct1, et0, et1;
  ops_timers(&ct0, &et0);

  //populate
  int iter_range[] = {0,4,0,4,0,4,0,4};
  ops_par_loop(mblock_populate_kernel, "mblock_populate_kernel", grid0, 4, iter_range,
               ops_arg_dat(data0, 1, S4D_0000, "double", OPS_WRITE),
               ops_arg_idx());
  ops_par_loop(mblock_populate_kernel, "mblock_populate_kernel", grid1, 4, iter_range,
               ops_arg_dat(data1, 1, S4D_0000, "double", OPS_WRITE),
               ops_arg_idx());

  ops_halo_transfer(halos0);
  ops_print_dat_to_txtfile(data0, "data0.txt");
  ops_print_dat_to_txtfile(data1, "data1.txt");

  ops_timers(&ct1, &et1);
  ops_timing_output(stdout);

  ops_printf("\nTotal Wall time %lf\n",et1-et0);

  ops_exit();
}
