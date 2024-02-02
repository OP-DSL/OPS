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
  ops_block grid0 = ops_decl_block(2, "grid0");
  ops_block grid1 = ops_decl_block(2, "grid1");

  //declare stencils
  int s2D_00[]         = {0,0};
  ops_stencil S2D_00 = ops_decl_stencil( 2, 1, s2D_00, "00");

  //declare datasets
  int d_p[2] = {2,2}; //max halo depths for the dat in the possitive direction
  int d_m[2] = {-2,-2}; //max halo depths for the dat in the negative direction
  int size[2] = {20, 20}; //size of the dat
  int base[2] = {0,0};
  double* temp = NULL;

  ops_dat data0 = ops_decl_dat(grid0, 1, size, base, d_m, d_p, temp, "double", "data0");
  ops_dat data1 = ops_decl_dat(grid1, 1, size, base, d_m, d_p, temp, "double", "data1");


  //straightforward matching orientation halos data0 - data1 in x
  //last two x lines of data0 and first two of data1
  ops_halo_group halos0;
  {
    int halo_iter[] = {2,20};
    int base_from[] = {18,0};
    int base_to[] = {-2,0};
    int dir[] = {1,2};
    ops_halo h0 = ops_decl_halo(data0, data1, halo_iter, base_from, base_to, dir, dir);
    //ops_halo h0 = ops_decl_halo_hdf5(data0, data1,"halo_file.h5");
    base_from[0] = 0; base_to[0] = 20;
    ops_halo h1 = ops_decl_halo(data1, data0, halo_iter, base_from, base_to, dir, dir);
    ops_halo grp[] = {h0,h1};
    halos0 = ops_decl_halo_group(2,grp);
    //ops_fetch_halo_hdf5_file(h0,"halo_file.h5");
  }



  //straightforward matching orientation halos data0 - data1 in y
  //last two y lines of data0 and first two of data1
  ops_halo_group halos1;
  {
    int halo_iter[] = {20,2};
    int base_from[] = {0,18};
    int base_to[] = {0,-2};
    int dir[] = {1,2};
    ops_halo h0 = ops_decl_halo(data0, data1, halo_iter, base_from, base_to, dir, dir);
    base_from[1] = 0; base_to[1] = 20;
    ops_halo h1 = ops_decl_halo(data1, data0, halo_iter, base_from, base_to, dir, dir);
    ops_halo grp[] = {h0,h1};
    halos1 = ops_decl_halo_group(2,grp);
  }

  //reverse data1 - data0 in x
  //last two x lines of data0 and first two of data1, but data1 is flipped in y
  ops_halo_group halos2;
  {
    int halo_iter[] = {2,20};
    int base_from[] = {0,0};
    int base_to[] = {20,0};
    int dir[] = {1,2};
    int dir_to[] = {1,-2};
    ops_halo h0 = ops_decl_halo(data0, data1, halo_iter, base_from, base_to, dir, dir_to);
    base_from[0] = 18; base_to[0] = -2;
    ops_halo h1 = ops_decl_halo(data1, data0, halo_iter, base_from, base_to, dir_to, dir);
    ops_halo grp[] = {h0,h1};
    halos2 = ops_decl_halo_group(2,grp);
  }

  //reverse data1 - data0 in y
  //last two y lines of data0 and first two of data1, but data1 is flipped in x
  ops_halo_group halos3;
  {
    int halo_iter[] = {20,2};
    int base_from[] = {0,0};
    int base_to[] = {0,20};
    int dir[] = {1,2};
    int dir_to[] = {-1,2};
    ops_halo h0 = ops_decl_halo(data0, data1, halo_iter, base_from, base_to, dir, dir_to);
    base_from[1] = 18; base_to[1] = -2;
    ops_halo h1 = ops_decl_halo(data1, data0, halo_iter, base_from, base_to, dir_to, dir);
    ops_halo grp[] = {h0,h1};
    halos3 = ops_decl_halo_group(2,grp);
  }

  //rotated data0-data1 x<->y
  //last two x lines of data0 to first two y lines of data1 (and back)
  ops_halo_group halos4;
  {
    int halo_iter[] = {2,20};
    int base_from[] = {18,0};
    int base_to[] = {0,-2};
    int dir[] = {1,2};
    int dir_to[] = {2,1};
    ops_halo h0 = ops_decl_halo(data0, data1, halo_iter, base_from, base_to, dir, dir_to);
    base_from[0] = 0; base_to[0] = 20; base_to[1] = 0;
    ops_halo h1 = ops_decl_halo(data1, data0, halo_iter, base_from, base_to, dir_to, dir);
    ops_halo grp[] = {h0,h1};
    halos4 = ops_decl_halo_group(2,grp);
  }

  ops_partition("");

  /**-------------------------- Computations --------------------------**/


  double ct0, ct1, et0, et1;
  ops_timers(&ct0, &et0);

  //populate
  int iter_range[] = {0,20,0,20};
  ops_par_loop(mblock_populate_kernel, "mblock_populate_kernel", grid0, 2, iter_range,
               ops_arg_dat(data0, 1, S2D_00, "double", OPS_WRITE),
               ops_arg_idx());
  ops_par_loop(mblock_populate_kernel, "mblock_populate_kernel", grid1, 2, iter_range,
               ops_arg_dat(data1, 1, S2D_00, "double", OPS_WRITE),
               ops_arg_idx());

  // test writing multiple blocks to same file
  ops_fetch_block_hdf5_file(grid0, "mblocktest.h5");
  ops_fetch_block_hdf5_file(grid1, "mblocktest.h5");

  // test writing dats on multiple blocks to same file
  ops_fetch_dat_hdf5_file(data0, "mblocktest.h5");
  ops_fetch_dat_hdf5_file(data1, "mblocktest.h5");

  // test writing multiple blocks to different files
  ops_fetch_block_hdf5_file(grid0, "mblocktest0.h5");
  ops_fetch_block_hdf5_file(grid1, "mblocktest1.h5");

  // test writing dats on multiple blocks to different files
  ops_fetch_dat_hdf5_file(data0, "mblocktest0.h5");
  ops_fetch_dat_hdf5_file(data1, "mblocktest1.h5");

  ops_halo_transfer(halos0);
  ops_halo_transfer(halos1);
  ops_halo_transfer(halos2);
  ops_halo_transfer(halos3);
  ops_halo_transfer(halos4);
  ops_print_dat_to_txtfile(data0, "data0.txt");
  ops_print_dat_to_txtfile(data1, "data1.txt");

  ops_printf("This test is considered PASSED\n");

  ops_timers(&ct1, &et1);
  ops_timing_output(std::cout);

  ops_printf("\nTotal Wall time %lf\n",et1-et0);

  ops_exit();
}
