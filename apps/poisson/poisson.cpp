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
#include "ops_seq.h"

#include "poisson_kernel.h"

/******************************************************************************
* Main program
/******************************************************************************/
int main(int argc, char **argv)
{
  /**-------------------------- Initialisation --------------------------**/

  // OPS initialisation
  ops_init(argc,argv,2);

  int logical_size_x = 50;
  int logical_size_y = 50;
  int ngrid_x = 2;
  int ngrid_y = 1;
  int n_iter = 1000;
  dx = 0.01;
  dy = 0.01;
  ops_decl_const("dx",1,"double",&dx);
  ops_decl_const("dy",1,"double",&dy);

  //declare blocks
  ops_block *blocks = (ops_block *)malloc(ngrid_x*ngrid_y*sizeof(ops_block*));
  char buf[50];
  for (int j = 0; j < ngrid_y; j++) {
    for (int i = 0; i < ngrid_x; i++) {
      sprintf(buf,"block %d,%d",i,j);
      blocks[i+ngrid_x*j] = ops_decl_block(2,buf);
    }
  }

  //declare stencils
  int s2D_00[]         = {0,0};
  ops_stencil S2D_00 = ops_decl_stencil( 2, 1, s2D_00, "00");
  int s2D_00_P10_M10_0P1_0M1[]         = {0,0, 1,0, -1,0, 0,1, 0,-1};
  ops_stencil S2D_00_P10_M10_0P1_0M1 = ops_decl_stencil( 2, 5, s2D_00_P10_M10_0P1_0M1, "00:10:-10:01:0-1");

  ops_reduction red_err = ops_decl_reduction_handle(sizeof(double), "double", "err");

  //declare datasets
  int d_p[2] = {-1,-1}; //max halo depths for the dat in the possitive direction
  int d_m[2] = {-1,-1}; //max halo depths for the dat in the negative direction
  int base[2] = {0,0};
  int uniform_size[2] = {(logical_size_x-1)/ngrid_x+1,(logical_size_y-1)/ngrid_y+1};
  double* temp = NULL;
  ops_dat *coordx = (ops_dat *)malloc(ngrid_x*ngrid_y*sizeof(ops_dat*));
  ops_dat *coordy = (ops_dat *)malloc(ngrid_x*ngrid_y*sizeof(ops_dat*));
  ops_dat *u = (ops_dat *)malloc(ngrid_x*ngrid_y*sizeof(ops_dat*));
  ops_dat *u2 = (ops_dat *)malloc(ngrid_x*ngrid_y*sizeof(ops_dat*));
  ops_dat *f = (ops_dat *)malloc(ngrid_x*ngrid_y*sizeof(ops_dat*));
  ops_dat *ref = (ops_dat *)malloc(ngrid_x*ngrid_y*sizeof(ops_dat*));
  int *sizes = (int*)malloc(2*ngrid_x*ngrid_y*sizeof(int));
  int *disps = (int*)malloc(2*ngrid_x*ngrid_y*sizeof(int));

  for (int j = 0; j < ngrid_y; j++) {
    for (int i = 0; i < ngrid_x; i++) {
      int size[2] = {uniform_size[0], uniform_size[1]};
      if ((i+1)*size[0]>logical_size_x) size[0] = logical_size_x - i*size[0];
      if ((j+1)*size[1]>logical_size_y) size[1] = logical_size_y - j*size[1];
      sprintf(buf,"coordx %d,%d",i,j);
      coordx[i+ngrid_x*j] = ops_decl_dat(blocks[i+ngrid_x*j], 1, size, base, d_m, d_p, temp, "double", buf);
      sprintf(buf,"coordy %d,%d",i,j);
      coordy[i+ngrid_x*j] = ops_decl_dat(blocks[i+ngrid_x*j], 1, size, base, d_m, d_p, temp, "double", buf);
      sprintf(buf,"u %d,%d",i,j);
      u[i+ngrid_x*j] = ops_decl_dat(blocks[i+ngrid_x*j], 1, size, base, d_m, d_p, temp, "double", buf);
      sprintf(buf,"u2 %d,%d",i,j);
      u2[i+ngrid_x*j] = ops_decl_dat(blocks[i+ngrid_x*j], 1, size, base, d_m, d_p, temp, "double", buf);
      sprintf(buf,"f %d,%d",i,j);
      f[i+ngrid_x*j] = ops_decl_dat(blocks[i+ngrid_x*j], 1, size, base, d_m, d_p, temp, "double", buf);
      sprintf(buf,"ref %d,%d",i,j);
      ref[i+ngrid_x*j] = ops_decl_dat(blocks[i+ngrid_x*j], 1, size, base, d_m, d_p, temp, "double", buf);

      sizes[2*(i+ngrid_x*j)]   = size[0];
      sizes[2*(i+ngrid_x*j)+1] = size[1];
      disps[2*(i+ngrid_x*j)]   = i*uniform_size[0];
      disps[2*(i+ngrid_x*j)+1] = j*uniform_size[1];
    }
  }

  ops_halo *halos = (ops_halo *)malloc(2*(ngrid_x*(ngrid_y-1)+(ngrid_x-1)*ngrid_y)*sizeof(ops_halo *));
  int off = 0;
  for (int j = 0; j < ngrid_y; j++) {
    for (int i = 0; i < ngrid_x; i++) {
      if (i > 0) {
        int halo_iter[] = {1,sizes[2*(i+ngrid_x*j)+1]};
        int base_from[] = {sizes[2*(i-1+ngrid_x*j)]-1,0};
        int base_to[] = {-1,0};
        int dir[] = {1,2};
        halos[off++] = ops_decl_halo(u[i-1+ngrid_x*j], u[i+ngrid_x*j], halo_iter, base_from, base_to, dir, dir);
        base_from[0] = 0; base_to[0] = sizes[2*(i+ngrid_x*j)];
        halos[off++] = ops_decl_halo(u[i+ngrid_x*j], u[i-1+ngrid_x*j], halo_iter, base_from, base_to, dir, dir);
      }
      if (j > 0) {
        int halo_iter[] = {sizes[2*(i+ngrid_x*j)],1};
        int base_from[] = {0,sizes[2*(i+ngrid_x*(j-1))+1]-1};
        int base_to[] = {0,-1};
        int dir[] = {1,2};
        halos[off++] = ops_decl_halo(u[i+ngrid_x*(j-1)], u[i+ngrid_x*j], halo_iter, base_from, base_to, dir, dir);
        base_from[1] = 0; base_to[1] = sizes[2*(i+ngrid_x*j)+1];
        halos[off++] = ops_decl_halo(u[i+ngrid_x*j], u[i+ngrid_x*(j-1)], halo_iter, base_from, base_to, dir, dir);
      }
    }
  }
  if (off != 2*(ngrid_x*(ngrid_y-1)+(ngrid_x-1)*ngrid_y)) printf("Something is not right\n");
  ops_halo_group u_halos = ops_decl_halo_group(off,halos);

  ops_partition("");

  /**-------------------------- Computations --------------------------**/


  double ct0, ct1, et0, et1;
  ops_timers_core(&ct0, &et0);

  //populate forcing, reference solution and boundary conditions
  for (int j = 0; j < ngrid_y; j++) {
    for (int i = 0; i < ngrid_x; i++) {
      int iter_range[] = {-1,sizes[2*(i+ngrid_x*j)]+1,-1,sizes[2*(i+ngrid_x*j)+1]+1};
      ops_par_loop(poisson_kernel_populate, "poisson_kernel_populate", blocks[i+ngrid_x*j], 2, iter_range,
               ops_arg_gbl(&disps[2*(i+ngrid_x*j)], 1, "int", OPS_READ),
               ops_arg_gbl(&disps[2*(i+ngrid_x*j)+1], 1, "int", OPS_READ),
               ops_arg_idx(),
               ops_arg_dat(u[i+ngrid_x*j], S2D_00, "double", OPS_WRITE),
               ops_arg_dat(f[i+ngrid_x*j], S2D_00, "double", OPS_WRITE),
               ops_arg_dat(ref[i+ngrid_x*j], S2D_00, "double", OPS_WRITE));
    }
  }

  //initial guess 0
  for (int j = 0; j < ngrid_y; j++) {
    for (int i = 0; i < ngrid_x; i++) {
      int iter_range[] = {0,sizes[2*(i+ngrid_x*j)],0,sizes[2*(i+ngrid_x*j)+1]};
      ops_par_loop(poisson_kernel_initialguess, "poisson_kernel_initialguess", blocks[i+ngrid_x*j], 2, iter_range,
               ops_arg_dat(u[i+ngrid_x*j], S2D_00, "double", OPS_WRITE));
    }
  }

  for (int iter = 0; iter < n_iter; iter++) {
    ops_halo_transfer(u_halos);
    for (int j = 0; j < ngrid_y; j++) {
      for (int i = 0; i < ngrid_x; i++) {
        int iter_range[] = {0,sizes[2*(i+ngrid_x*j)],0,sizes[2*(i+ngrid_x*j)+1]};
        ops_par_loop(poisson_kernel_stencil, "poisson_kernel_stencil", blocks[i+ngrid_x*j], 2, iter_range,
                 ops_arg_dat(u[i+ngrid_x*j], S2D_00_P10_M10_0P1_0M1, "double", OPS_READ),
                 ops_arg_dat(f[i+ngrid_x*j], S2D_00, "double", OPS_READ),
                 ops_arg_dat(u2[i+ngrid_x*j], S2D_00, "double", OPS_WRITE));
      }
    }
    for (int j = 0; j < ngrid_y; j++) {
      for (int i = 0; i < ngrid_x; i++) {
        int iter_range[] = {0,sizes[2*(i+ngrid_x*j)],0,sizes[2*(i+ngrid_x*j)+1]};
        ops_par_loop(poisson_kernel_update, "poisson_kernel_update", blocks[i+ngrid_x*j], 2, iter_range,
                 ops_arg_dat(u2[i+ngrid_x*j], S2D_00, "double", OPS_READ),
                 ops_arg_dat(u[i+ngrid_x*j] , S2D_00, "double", OPS_WRITE));
      }
    }
  }

  double err = 0.0;
  for (int j = 0; j < ngrid_y; j++) {
    for (int i = 0; i < ngrid_x; i++) {
      int iter_range[] = {0,sizes[2*(i+ngrid_x*j)],0,sizes[2*(i+ngrid_x*j)+1]};
      ops_par_loop(poisson_kernel_error, "poisson_kernel_error", blocks[i+ngrid_x*j], 2, iter_range,
               ops_arg_dat(u[i+ngrid_x*j],    S2D_00, "double", OPS_READ),
               ops_arg_dat(ref[i+ngrid_x*j] , S2D_00, "double", OPS_READ),
               ops_arg_reduce(red_err, 1, "double", OPS_INC));
    }
  }

  ops_reduction_result(red_err,&err);
  ops_printf("Total error: %g\n",err);

  ops_timers_core(&ct1, &et1);
  ops_timing_output();

  ops_printf("\nTotal Wall time %lf\n",et1-et0);

  ops_exit();
}
