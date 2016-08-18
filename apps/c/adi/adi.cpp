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

/** @brief 3D heat diffusion PDE solved using ADI
 *  @author Endre Lazlo, converted to OPS by Gihan Mudalige
 *  @details PDE is solved with the ADI (Alternating Direction Implicit) method
 *	uses the Scalar tridiagonal solver for CPU and GPU written by Endre. Lazslo
**/

// standard headers
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

// OPS header file
#define OPS_3D
#include "ops_seq.h"


#include "data.h"

#include "init_kernel.h"
#include "preproc_kernel.h"
#include "rms_kernel.h"



// declare defaults options
int nx;
int ny;
int nz;
int iter;
int opts[3], synch;

//declare constants
double lambda;

int main(int argc, char **argv)
{
  // Set defaults options
  nx = 256;
  ny = 256;
  nz = 256;
  iter = 10;
  opts[0] = 0;
  opts[1] = 0;
  opts[2] = 0;
  synch = 1;

  //constants
  lambda=1.0f;

  /**------------------------------ Initialisation ------------------------------**/

  // OPS initialisation
  ops_init(argc,argv,1);

  /**------------------------------OPS Declarations------------------------------**/

  //declare block
  ops_block heat3D = ops_decl_block(3, "Heat3D");

  //declare data on blocks
  int d_p[3] = {1,1,1}; //max halo depths for the dat in the possitive direction
  int d_m[3] = {-1,-1,-1}; //max halo depths for the dat in the negative direction
  int size[3] = {nx, ny, nz}; //size of the dat -- should be identical to the block on which its define on
  int base[3] = {0,0,0};
  double* temp = NULL;

  ops_dat h_u     = ops_decl_dat(heat3D, 1, size, base, d_m, d_p, temp, "double", "h_u");

  d_p[0] = 0; d_p[1] = 0; d_p[2] = 0; //max halo depths for the dat in the possitive direction
  d_m[0] = 0; d_m[1] = 0; d_m[2] = 0; //max halo depths for the dat in the negative direction

  ops_dat h_temp  = ops_decl_dat(heat3D, 1, size, base, d_m, d_p, temp, "double", "h_tmp");
  ops_dat h_du    = ops_decl_dat(heat3D, 1, size, base, d_m, d_p, temp, "double", "h_du");
  ops_dat h_ax    = ops_decl_dat(heat3D, 1, size, base, d_m, d_p, temp, "double", "h_ax");
  ops_dat h_bx    = ops_decl_dat(heat3D, 1, size, base, d_m, d_p, temp, "double", "h_bx");
  ops_dat h_cx    = ops_decl_dat(heat3D, 1, size, base, d_m, d_p, temp, "double", "h_cx");
  ops_dat h_ay    = ops_decl_dat(heat3D, 1, size, base, d_m, d_p, temp, "double", "h_ay");
  ops_dat h_by    = ops_decl_dat(heat3D, 1, size, base, d_m, d_p, temp, "double", "h_by");
  ops_dat h_cy    = ops_decl_dat(heat3D, 1, size, base, d_m, d_p, temp, "double", "h_cy");
  ops_dat h_az    = ops_decl_dat(heat3D, 1, size, base, d_m, d_p, temp, "double", "h_az");
  ops_dat h_bz    = ops_decl_dat(heat3D, 1, size, base, d_m, d_p, temp, "double", "h_bz");
  ops_dat h_cz    = ops_decl_dat(heat3D, 1, size, base, d_m, d_p, temp, "double", "h_cz");

  //declare stencils
  int s3D_000[]         = {0,0,0};
  ops_stencil S3D_000 = ops_decl_stencil( 3, 1, s3D_000, "000");

  int s3D_7pt[] = { 0,0,0, -1,0,0, 1,0,0, 0,-1,0, 0,1,0, 0,0,-1, 0,0,1 };
  ops_stencil S3D_7PT = ops_decl_stencil( 3, 7, s3D_7pt, "3d7Point");

  //declare constants
  ops_decl_const( "nx", 1, "int", &nx );
  ops_decl_const( "ny", 1, "int", &ny );
  ops_decl_const( "nz", 1, "int", &nz );
  ops_decl_const( "lambda", 1, "double", &lambda );

  //reduction handle to check output
  ops_reduction rms = ops_decl_reduction_handle(sizeof(double), "double", "rms");

  //decompose the block
  ops_partition("3D_BLOCK_DECOMPSE");

  double ct0, ct1, et0, et1, ct2, et2, ct3, et3;

  printf("\nGrid dimensions: %d x %d x %d\n", nx, ny, nz);
  ops_diagnostic_output();

  //initialize Tridiagonal Library
  ops_initTridMultiDimBatchSolve(3 /*dimension*/, size /*size in each dimension*/);

  /**--------------------------------- Initialize -------------------------------**/
  int iter_range[] = {0,nx, 0,ny, 0,nz};
  ops_par_loop(init_kernel, "init_kernel", heat3D, 3, iter_range,
               ops_arg_dat(h_u, 1, S3D_000, "double", OPS_WRITE),
               ops_arg_idx());

  ops_timers(&ct0, &et0);

  for(int it = 0; it<iter; it++) { // Start main iteration loop

  /**-------------- calculate r.h.s. and set tri-diagonal coefficients-----------**/
  int iter_range[] = {0,nx, 0,ny, 0,nz};

  ops_timers(&ct2, &et2);
  ops_par_loop(preproc_kernel, "preproc_kernel", heat3D, 3, iter_range,
               ops_arg_dat(h_u, 1, S3D_7PT, "double", OPS_READ),
               ops_arg_dat(h_du, 1, S3D_000, "double", OPS_WRITE),
               ops_arg_dat(h_ax, 1, S3D_000, "double", OPS_WRITE),
               ops_arg_dat(h_bx, 1, S3D_000, "double", OPS_WRITE),
               ops_arg_dat(h_cx, 1, S3D_000, "double", OPS_WRITE),
               ops_arg_dat(h_ay, 1, S3D_000, "double", OPS_WRITE),
               ops_arg_dat(h_by, 1, S3D_000, "double", OPS_WRITE),
               ops_arg_dat(h_cy, 1, S3D_000, "double", OPS_WRITE),
               ops_arg_dat(h_az, 1, S3D_000, "double", OPS_WRITE),
               ops_arg_dat(h_bz, 1, S3D_000, "double", OPS_WRITE),
               ops_arg_dat(h_cz, 1, S3D_000, "double", OPS_WRITE),
               ops_arg_idx());
  ops_timers(&ct3, &et3);
  ops_printf("Elapsed preproc (sec): %lf (s)\n",et3-et2);

  double rms_value = 0.0;
  ops_par_loop(rms_kernel, "rms_kernel", heat3D, 3, iter_range,
               ops_arg_dat(h_ax, 1, S3D_000, "double", OPS_READ),
               ops_arg_reduce(rms, 1, "double", OPS_INC));
  ops_reduction_result(rms, &rms_value);
  ops_printf("h_ax Value %lg\n", rms_value);

  rms_value = 0.0;
  ops_par_loop(rms_kernel, "rms_kernel", heat3D, 3, iter_range,
               ops_arg_dat(h_bx, 1, S3D_000, "double", OPS_READ),
               ops_arg_reduce(rms, 1, "double", OPS_INC));
  ops_reduction_result(rms,&rms_value);
  ops_printf("h_bx Value %lg\n", rms_value);

  rms_value = 0.0;
  ops_par_loop(rms_kernel, "rms_kernel", heat3D, 3, iter_range,
               ops_arg_dat(h_cx, 1, S3D_000, "double", OPS_READ),
               ops_arg_reduce(rms, 1, "double", OPS_INC));
  ops_reduction_result(rms, &rms_value);
  ops_printf("h_cx Value %lg\n", rms_value);

  rms_value = 0.0;
  ops_par_loop(rms_kernel, "rms_kernel", heat3D, 3, iter_range,
               ops_arg_dat(h_du, 1, S3D_000, "double", OPS_READ),
               ops_arg_reduce(rms, 1, "double", OPS_INC));
  ops_reduction_result(rms, &rms_value);
  ops_printf("h_du Value %lg\n", rms_value);

  rms_value = 0.0;
  ops_par_loop(rms_kernel, "rms_kernel", heat3D, 3, iter_range,
               ops_arg_dat(h_u, 1, S3D_000, "double", OPS_READ),
               ops_arg_reduce(rms, 1, "double", OPS_INC));
  ops_reduction_result(rms, &rms_value);
  ops_printf("h_u Value %lg\n", rms_value);
  exit(-2);

  /**----------------- perform tri-diagonal solves in x-direction
   * ---------------**/
  ops_timers(&ct2, &et2);
  ops_tridMultiDimBatch(3, 0, size, h_ax, h_bx, h_cx, h_du, h_u, opts);
  ops_timers(&ct3, &et3);
  ops_printf("Elapsed trid_x (sec): %lf (s)\n", et3 - et2);

  /**----------------- perform tri-diagonal solves in y-direction ---------------**/
  ops_timers(&ct2, &et2);
  ops_tridMultiDimBatch( 3, 1 , size, h_ay, h_by, h_cy, h_du, h_u, opts );
  ops_timers(&ct3, &et3);
  ops_printf("Elapsed trid_y (sec): %lf (s)\n",et3-et2);

  /**----------------- perform tri-diagonal solves in z-direction ---------------**/
  ops_timers(&ct2, &et2);
  ops_tridMultiDimBatch_Inc( 3, 2 , size, h_az, h_bz, h_cz, h_du, h_u, opts );
  ops_timers(&ct3, &et3);
  ops_printf("Elapsed trid_z (sec): %lf (s)\n",et3-et2);

  } // End main iteration loop

  ops_timers(&ct1, &et1);

  /**---------------------------- Print solution with OPS------------------------**/

  ops_print_dat_to_txtfile(h_u, "h_u.dat");

  /**-------------------------- Check solution without OPS-----------------------**/

  // print out left corner of array
  for(int k=0; k<2; k++) {
    printf("k = %i\n",k);
    for(int j=0; j<MIN(ny,17); j++) {
      printf(" %d   ", j);
      for(int i=0; i<MIN(nx,17); i++) {
        int ind = i + j*nx + k*nx*ny;
        printf(" %5.5g ", ((double *)h_u->data)[ind]);
      }
      printf("\n");
    }
    printf("\n");
  }

  // print out right corner of array
  for(int k=0; k<2; k++) {
    printf("k = %i\n",k);
    for(int j=0; j<MIN(ny,17); j++) {
      printf(" %d   ", j);
      for(int i=MAX(0,nx-17); i<nx; i++) {
        int ind = i + j*nx + k*nx*ny;
        printf(" %5.5g ", ((double *)h_u->data)[ind]);
      }
      printf("\n");
    }
    printf("\n");
  }

  ops_printf("\nTotal Wall time %lf\n",et1-et0);
  ops_exit();
}