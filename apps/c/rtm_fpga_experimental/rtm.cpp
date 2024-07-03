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
* DISCLAIMED. IN NO EVENT SHALL Mike Giles BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
* (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
* LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
* ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
* (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
* SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

/** @Test application for multi-block functionality
  * @author Gihan Mudalige, Istvan Reguly, Beniel Thileepan
  */

// standard headers
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

float dx,dy,dz;
int pml_width;
int half, order, nx, ny, nz;
#include "coeffs8.h"
// OPS header file
#define OPS_SOA
#define OPS_3D
#include "ops_seq_v2.h"

#include "rtm_kernel.h"


void derivs1(int ngrid_x, int ngrid_y, int ngrid_z, int* sizes, int disps[], ops_block blocks, ops_stencil S3D_000, ops_stencil S3D_big_sten,  float* dt,  float* scale1, float* scale2, ops_dat rho, ops_dat mu, ops_dat yy, ops_dat dyy, ops_dat sum);
void derivs2(int ngrid_x, int ngrid_y, int ngrid_z, int* sizes, int disps[], ops_block blocks, ops_stencil S3D_000, ops_stencil S3D_big_sten,  float* dt,  float* scale1, float* scale2, ops_dat rho, ops_dat mu, ops_dat yy, ops_dat dyyIn,  ops_dat dyyOut, ops_dat sum);

void derivs3(int ngrid_x, int ngrid_y, int ngrid_z, int* sizes, int disps[], ops_block blocks, ops_stencil S3D_000, ops_stencil S3D_big_sten,  float* dt,  float* scale1, float* scale2, ops_dat rho, ops_dat mu, ops_dat yy, ops_dat dyyIn, ops_dat sum);
	      
/******************************************************************************
* Main program
*******************************************************************************/
int main(int argc,  char **argv)
{
  /**-------------------------- Initialisation --------------------------**/

  // OPS initialisation  
  ops_init(argc,argv,1);
  OPS_instance::getOPSInstance()->OPS_soa = 1;

  //Mesh
  int logical_size_x = 16;
  int logical_size_y = 16;
  int logical_size_z = 16;
  nx = logical_size_x;
  ny = logical_size_y;
  nz = logical_size_z;
  int ngrid_x = 1;
  int ngrid_y = 1;
  int ngrid_z = 1;
  int n_iter = 100;
  int itertile = n_iter;
  int non_copy = 0;

  int num_systems = 100;

  const char* pch;
  printf(" argc = %d\n",argc);
  for ( int n = 1; n < argc; n++ ) {
    pch = strstr(argv[n], "-sizex=");
    if(pch != NULL) {
      logical_size_x = atoi ( argv[n] + 7 ); continue;
    }
    pch = strstr(argv[n], "-sizey=");
    if(pch != NULL) {
      logical_size_y = atoi ( argv[n] + 7 ); continue;
    }
    pch = strstr(argv[n], "-sizez=");
    if(pch != NULL) {
      logical_size_z = atoi ( argv[n] + 7 ); continue;
    }
    pch = strstr(argv[n], "-iters=");
    if(pch != NULL) {
      n_iter = atoi ( argv[n] + 7 ); continue;
    }
    pch = strstr(argv[n], "-itert=");
    if(pch != NULL) {
      itertile = atoi ( argv[n] + 7 ); continue;
    }
    pch = strstr(argv[n], "-non-copy");
    if(pch != NULL) {
      non_copy = 1; continue;
    }
    pch = strstr(argv[n], "-batch=");
    if(pch != NULL) {
      num_systems = atoi ( argv[n] + 7 ); continue;
    }
  }

  // logical_size_x = 16;
  // logical_size_y = 16;
  // logical_size_z = 16;
  nx = logical_size_x;
  ny = logical_size_y;
  nz = logical_size_z;
  ngrid_x = 1;
  ngrid_y = 1;
  ngrid_z = 1;
  // n_iter = 1;
  itertile = n_iter;
  non_copy = 0;

  ops_printf("Grid: %dx%dx%d , %d iterations, %d tile height\n",logical_size_x,logical_size_y,logical_size_z,n_iter,itertile);
  dx = 0.005;
  dy = 0.005;
  dz = 0.005;
  pml_width = 10;
  ops_decl_const("dx",1,"float",&dx);
  ops_decl_const("dy",1,"float",&dy);
  ops_decl_const("dz",1,"float",&dz);
  ops_decl_const("nx",1,"int",&nx);
  ops_decl_const("ny",1,"int",&ny);
  ops_decl_const("nz",1,"int",&nz);
  ops_decl_const("pml_width",1,"int",&pml_width);
  //int ncoeffs = (ORDER+1)*(ORDER+1);
  //ops_decl_const("coeffs",ncoeffs,"double",&coeffs[0][0]);
  half = HALF;
  ops_decl_const("half",1,"int",&half);
  order = ORDER;
  ops_decl_const("order",1,"int",&order);

  //declare block
  char buf[50];
  sprintf(buf,"block");
  ops_block blocks = ops_decl_block(3, buf);
  printf(" HERE \n");
  
  //declare stencils
  int s3D_000[]         = {0,0,0};
  ops_stencil S3D_000 = ops_decl_stencil( 3, 1, s3D_000, "000");
  int s3D_big_sten[3*3*(2*ORDER+1)];
  int is = 0;
  for (int ix=-HALF;ix<=HALF;ix++) {
    printf("ix = %d\n",ix);
    s3D_big_sten[is] = ix;
    is = is + 1;
    s3D_big_sten[is] = 0;
    is = is + 1;
    s3D_big_sten[is] = 0;
    is = is + 1;
  }
  for (int ix=-HALF;ix<=HALF;ix++) {
    s3D_big_sten[is] = 0;
    is = is + 1;
    s3D_big_sten[is] = ix;
    is = is + 1;
    s3D_big_sten[is] = 0;
    is = is + 1;
  }
  for (int ix=-HALF;ix<=HALF;ix++) {
    s3D_big_sten[is] = 0;
    is = is + 1;
    s3D_big_sten[is] = 0;
    is = is + 1;
    s3D_big_sten[is] = ix;
    is = is + 1;
  }
  ops_stencil S3D_big_sten = ops_decl_stencil( 3, 3*(2*ORDER+1), s3D_big_sten, "big_sten");

  printf(" HERE2 \n");
  

  //declare datasets
  int d_p[3] = {HALF,HALF,HALF}; //max halo depths for the dat in the possitive direction
  int d_m[3] = {-HALF,-HALF,-HALF}; //max halo depths for the dat in the negative direction
  int base[3] = {0,0,0};
  int uniform_size[3] = {(logical_size_x-1)/ngrid_x+1,(logical_size_y-1)/ngrid_y+1,(logical_size_z-1)/ngrid_z+1};
  float* temp = NULL;
  int *sizes = (int*)malloc(3*sizeof(int));
  int *disps = (int*)malloc(3*sizeof(int));

  printf(" HERE 3\n");
  
  int size[3] = {uniform_size[0], uniform_size[1], uniform_size[2]};
  if (size[0]>logical_size_x) size[0] = logical_size_x;
  if (size[1]>logical_size_y) size[1] = logical_size_y;
  if (size[2]>logical_size_z) size[2] = logical_size_z;
	
  sprintf(buf,"coordx");
  ops_dat coordx = ops_decl_dat(blocks, 1, size, base, d_m, d_p, temp, "float", buf);

  sprintf(buf,"coordy");
  ops_dat coordy = ops_decl_dat(blocks, 1, size, base, d_m, d_p, temp, "float", buf);

  sprintf(buf,"coordz");
  ops_dat coordz= ops_decl_dat(blocks, 1, size, base, d_m, d_p, temp, "float", buf);

  sprintf(buf,"rho");
  ops_dat rho = ops_decl_dat(blocks, 1, size, base, d_m, d_p, temp, "float", buf);

  sprintf(buf,"mu");
  ops_dat mu = ops_decl_dat(blocks, 1, size, base, d_m, d_p, temp, "float", buf);

  sprintf(buf,"yy");
  ops_dat yy = ops_decl_dat(blocks, 6, size, base, d_m, d_p, temp, "float", buf);

  sprintf(buf,"yy_sum");
  ops_dat yy_sum = ops_decl_dat(blocks, 6, size, base, d_m, d_p, temp, "float", buf);

  sprintf(buf,"ytemp1");
  ops_dat ytemp1 = ops_decl_dat(blocks, 6, size, base, d_m, d_p, temp, "float", buf);

  sprintf(buf,"ytemp2");
  ops_dat ytemp2 = ops_decl_dat(blocks, 6, size, base, d_m, d_p, temp, "float", buf);


  
  sizes[0] = size[0];
  sizes[1] = size[1];
  sizes[2] = size[2];
  disps[0] = 0;
  disps[1] = 0;
  disps[2] = 0;

  printf(" HERE 4\n");

  
  ops_partition("");
  ops_checkpointing_init("check.h5", 5.0, 0);
  ops_diagnostic_output();
  /**-------------------------- Computations --------------------------**/


  double ct0, ct1, et0, et1;
  ops_timers(&ct0, &et0);

  ops_par_loop_blocks_all(num_systems);
	
  printf(" HERE 5\n");
  
  //populate density, bulk modulus, velx, vely, velz, and boundary conditions
	
  int iter_range[] = {0,sizes[0],0,sizes[1],0,sizes[2]};
	
  ops_par_loop(rtm_kernel_populate, "rtm_kernel_populate", blocks, 3, iter_range,
	       ops_arg_gbl(&disps[0], 1, "int", OPS_READ),
	       ops_arg_gbl(&disps[1], 1, "int", OPS_READ),
	       ops_arg_gbl(&disps[2], 1, "int", OPS_READ),
	       ops_arg_idx(),
	       ops_arg_dat(rho, 1, S3D_000, "float", OPS_WRITE),
	       ops_arg_dat(mu, 1, S3D_000, "float", OPS_WRITE),
	       ops_arg_dat(yy, 6, S3D_000, "float", OPS_WRITE),
	       ops_arg_dat(yy_sum, 6, S3D_000, "float", OPS_WRITE),
	       );
		
  printf(" DONE populate\n");
  

  double it0, it1;
  ops_timers(&ct0, &it0);
//#ifdef NOT_NOW  
  for (int iter = 0; iter < n_iter; iter++) {
    //if (ngrid_x>1 || ngrid_y>1 || ngrid_z>1) ops_halo_transfer(u_halos);
    // if (iter%itertile == 0) ops_execute();

    /* The following is 4th order Runga-Kutta */
    float dt = 0.1f; //=sqrt(mu/rho);
    float scale1 = 0.5f;
    float scale2 = 1/6.0f;

    derivs1(ngrid_x, ngrid_y, ngrid_z, sizes, disps, blocks, S3D_000, S3D_big_sten,
	   &dt, &scale1, &scale2, rho, mu, yy, ytemp1, yy_sum);


    scale1 = 0.5f;
    scale2 = 1/3.0f;
    derivs2(ngrid_x, ngrid_y, ngrid_z, sizes, disps, blocks, S3D_000, S3D_big_sten,
	   &dt, &scale1, &scale2, rho, mu, yy, ytemp1, ytemp2, yy_sum);

    scale1 = 1.0f;
    scale2 = 1/3.0f;
    derivs2(ngrid_x, ngrid_y, ngrid_z, sizes, disps, blocks, S3D_000, S3D_big_sten,
	   &dt, &scale1, &scale2, rho, mu, yy, ytemp2, ytemp1, yy_sum);


    scale1 = 1.0f;
    scale2 = 1/6.0f;
    derivs3(ngrid_x, ngrid_y, ngrid_z, sizes, disps, blocks, S3D_000, S3D_big_sten,
	   &dt, &scale1, &scale2, rho, mu, yy, ytemp1, yy_sum);

	     
  }
//#endif
  // ops_execute();
  ops_timers(&ct0, &it1);
   // ops_print_dat_to_txtfile(yy, "yy.txt");

  //ops_print_dat_to_txtfile(u[0], "poisson.dat");
  //ops_print_dat_to_txtfile(ref[0], "poisson.dat");
  //exit(0);


  // free(coordx);
  // free(coordy);
  // free(coordz);
  // free(rho);
  // free(mu);
  // free(yy);
  // free(ytemp);
  // free(k1);
  // free(k2);
  // free(k3);
  // free(k4);


  ops_timers(&ct1, &et1);
  ops_timing_output(stdout);
  ops_printf("\nTotal Wall time %lf\n",et1-et0);

  ops_printf("%lf\n",it1-it0);
  
  ops_printf(" DONE !!!\n");

  ops_exit();
}

void derivs1(int ngrid_x, int ngrid_y, int ngrid_z, int* sizes, int disps[], ops_block blocks, ops_stencil S3D_000, ops_stencil S3D_big_sten, float* dt,  float* scale1, float* scale2, ops_dat rho, ops_dat mu, ops_dat yy, ops_dat dyy, ops_dat sum) {
  
  int iter_range[] = {0,sizes[0],0,sizes[1],0,sizes[2]};
  ops_par_loop(fd3d_pml_kernel1, "fd3d_pml_kernel1", blocks, 3, iter_range,
	       ops_arg_gbl(&disps[0], 1, "int", OPS_READ),
	       ops_arg_gbl(&disps[1], 1, "int", OPS_READ),
	       ops_arg_gbl(&disps[2], 1, "int", OPS_READ),
	       ops_arg_idx(),
	       ops_arg_gbl(dt, 1, "float", OPS_READ),
	       ops_arg_gbl(scale1, 1, "float", OPS_READ),
	       ops_arg_gbl(scale2, 1, "float", OPS_READ),
	       ops_arg_dat(rho, 1, S3D_000, "float", OPS_READ),
	       ops_arg_dat(mu, 1, S3D_000, "float", OPS_READ),
	       ops_arg_dat(yy, 6, S3D_big_sten, "float", OPS_READ),
	       ops_arg_dat(dyy, 6, S3D_000, "float", OPS_WRITE),
	       ops_arg_dat(sum, 6, S3D_000, "float", OPS_WRITE)
	       );
}

void derivs2(int ngrid_x, int ngrid_y, int ngrid_z, int* sizes, int disps[], ops_block blocks, ops_stencil S3D_000, ops_stencil S3D_big_sten, float* dt,  float* scale1, float* scale2, ops_dat rho, ops_dat mu, ops_dat yy, ops_dat dyyIn,  ops_dat dyyOut, ops_dat sum) {
  
  int iter_range[] = {0,sizes[0],0,sizes[1],0,sizes[2]};
  ops_par_loop(fd3d_pml_kernel2, "fd3d_pml_kernel2", blocks, 3, iter_range,
	       ops_arg_gbl(&disps[0], 1, "int", OPS_READ),
	       ops_arg_gbl(&disps[1], 1, "int", OPS_READ),
	       ops_arg_gbl(&disps[2], 1, "int", OPS_READ),
	       ops_arg_idx(),
	       ops_arg_gbl(dt, 1, "float", OPS_READ),
	       ops_arg_gbl(scale1, 1, "float", OPS_READ),
	       ops_arg_gbl(scale2, 1, "float", OPS_READ),
	       ops_arg_dat(rho, 1, S3D_000, "float", OPS_READ),
	       ops_arg_dat(mu, 1, S3D_000, "float", OPS_READ),
	       ops_arg_dat(yy, 6, S3D_000, "float", OPS_READ),
	       ops_arg_dat(dyyIn, 6, S3D_big_sten, "float", OPS_READ),
	       ops_arg_dat(dyyOut, 6, S3D_000, "float", OPS_WRITE),
	       ops_arg_dat(sum, 6, S3D_000, "float", OPS_RW)
	       );
}

void derivs3(int ngrid_x, int ngrid_y, int ngrid_z, int* sizes, int disps[], ops_block blocks, ops_stencil S3D_000, ops_stencil S3D_big_sten, float* dt,  float* scale1, float* scale2, ops_dat rho, ops_dat mu, ops_dat yy,  ops_dat dyyIn, ops_dat sum) {
  
  int iter_range[] = {0,sizes[0],0,sizes[1],0,sizes[2]};
  ops_par_loop(fd3d_pml_kernel3, "fd3d_pml_kernel3", blocks, 3, iter_range,
	       ops_arg_gbl(&disps[0], 1, "int", OPS_READ),
	       ops_arg_gbl(&disps[1], 1, "int", OPS_READ),
	       ops_arg_gbl(&disps[2], 1, "int", OPS_READ),
	       ops_arg_idx(),
	       ops_arg_gbl(dt, 1, "float", OPS_READ),
	       ops_arg_gbl(scale1, 1, "float", OPS_READ),
	       ops_arg_gbl(scale2, 1, "float", OPS_READ),
	       ops_arg_dat(rho, 1, S3D_000, "float", OPS_READ),
	       ops_arg_dat(mu, 1, S3D_000, "float", OPS_READ),
	       ops_arg_dat(yy, 6, S3D_000, "float", OPS_RW),
	       ops_arg_dat(dyyIn, 6, S3D_big_sten, "float", OPS_READ),
	       ops_arg_dat(sum, 6, S3D_000, "float", OPS_READ)
	       );
}