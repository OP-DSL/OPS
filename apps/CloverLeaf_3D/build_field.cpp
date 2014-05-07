/* Crown Copyright 2012 AWE.

 This file is part of CloverLeaf.

 CloverLeaf is free software: you can redistribute it and/or modify it under
 the terms of the GNU General Public License as published by the
 Free Software Foundation, either version 3 of the License, or (at your option)
 any later version.

 CloverLeaf is distributed in the hope that it will be useful, but
 WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
 FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more
 details.

 You should have received a copy of the GNU General Public License along with
 CloverLeaf. If not, see http://www.gnu.org/licenses/. */

/** @brief Allocates the data for each mesh chunk
 *  @author Wayne Gaudin, converted to OPS by Gihan Mudalige
 *  @details The data fields for the mesh chunk are allocated based on the mesh size
**/

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

// OPS header file
#include "ops_seq.h"


#include "data.h"
#include "definitions.h"


void test_kernel(double *volume, double *xarea, double *yarea) {

  printf("%lf ",volume[OPS_ACC0(0,0,0)]);
  printf("%lf ",xarea[OPS_ACC1(0,0,0)]);
  printf("%lf ",yarea[OPS_ACC2(0,0,0)]);

}


void build_field()
{
  //initialize sizes using global values
  int x_cells = grid.x_cells;
  int y_cells = grid.y_cells;
  int z_cells = grid.z_cells;
  int x_min = field.x_min;
  int x_max = field.x_max;
  int y_min = field.y_min;
  int y_max = field.y_max;
  int z_min = field.z_min;
  int z_max = field.z_max;

  ops_printf("Global x_min = %d, y_min = %d, z_min = %d\n",x_min,y_min,z_min);
  ops_printf("Global x_max = %d, y_max = %d, z_max = %d\n",x_max,y_max,z_max);

  /**----------------------------OPS Declarations----------------------------**/

  int dims[3] = {x_cells+5, y_cells+5, z_cells+5};  //cloverleaf 2D block dimensions: +5 because we allocate the largest ops_dat's size
  clover_grid = ops_decl_block(3, dims, "clover grid");

  //decompose the block
  ops_partition(3, dims, "3D_BLOCK_DECOMPSE");

  //
  //declare data on blocks
  //
  int d_p[3] = {-2,-2,-2}; //max halo depths for the dat in the possitive direction
  int d_m[3] = {-2,-2,-2}; //max halo depths for the dat in the negative direction
  int size[3] = {x_cells+5, y_cells+5, z_cells+5}; //size of the dat -- should be identical to the block on which its define on
  double* temp = NULL;

  density0    = ops_decl_dat(clover_grid, 1, size, d_m, d_p, temp, "double", "density0");
  density1    = ops_decl_dat(clover_grid, 1, size, d_m, d_p, temp, "double", "density1");
  energy0     = ops_decl_dat(clover_grid, 1, size, d_m, d_p, temp, "double", "energy0");
  energy1     = ops_decl_dat(clover_grid, 1, size, d_m, d_p, temp, "double", "energy1");
  pressure    = ops_decl_dat(clover_grid, 1, size, d_m, d_p, temp, "double", "pressure");
  viscosity   = ops_decl_dat(clover_grid, 1, size, d_m, d_p, temp, "double", "viscosity");
  soundspeed  = ops_decl_dat(clover_grid, 1, size, d_m, d_p, temp, "double", "soundspeed");
  volume      = ops_decl_dat(clover_grid, 1, size, d_m, d_p, temp, "double", "volume");

  d_p[0]=-3;d_p[1]=-3;
  xvel0    = ops_decl_dat(clover_grid, 1, size, d_m, d_p, temp, "double", "xvel0");
  xvel1    = ops_decl_dat(clover_grid, 1, size, d_m, d_p, temp, "double", "xvel1");
  yvel0    = ops_decl_dat(clover_grid, 1, size, d_m, d_p, temp, "double", "yvel0");
  yvel1    = ops_decl_dat(clover_grid, 1, size, d_m, d_p, temp, "double", "yvel1");
  zvel0    = ops_decl_dat(clover_grid, 1, size, d_m, d_p, temp, "double", "zvel0");
  zvel1    = ops_decl_dat(clover_grid, 1, size, d_m, d_p, temp, "double", "zvel1");

  d_p[0]=-3;d_p[1]=-2;d_p[2]=-2;
  printf("%d %d %d\n",size[0],size[1],size[2]);
  vol_flux_x  = ops_decl_dat(clover_grid, 1, size, d_m, d_p, temp, "double", "vol_flux_x");
  mass_flux_x = ops_decl_dat(clover_grid, 1, size, d_m, d_p, temp, "double", "mass_flux_x");
  xarea       = ops_decl_dat(clover_grid, 1, size, d_m, d_p, temp, "double", "xarea");

  d_p[0]=-2;d_p[1]=-3;d_p[2]=-2;
  printf("%d %d %d\n",size[0],size[1],size[2]);
  vol_flux_y  = ops_decl_dat(clover_grid, 1, size, d_m, d_p, temp, "double", "vol_flux_y");
  mass_flux_y = ops_decl_dat(clover_grid, 1, size, d_m, d_p, temp, "double", "mass_flux_y");
  yarea       = ops_decl_dat(clover_grid, 1, size, d_m, d_p, temp, "double", "yarea");

  d_p[0]=-2;d_p[1]=-2;d_p[2]=-3;
  printf("%d %d %d\n",size[0],size[1],size[2]);
  vol_flux_z  = ops_decl_dat(clover_grid, 1, size, d_m, d_p, temp, "double", "vol_flux_z");
  mass_flux_z = ops_decl_dat(clover_grid, 1, size, d_m, d_p, temp, "double", "mass_flux_z");
  zarea       = ops_decl_dat(clover_grid, 1, size, d_m, d_p, temp, "double", "zarea");

  d_p[0]=-3;d_p[1]=-3;
  work_array1    = ops_decl_dat(clover_grid, 1, size, d_m, d_p, temp, "double", "work_array1");
  work_array2    = ops_decl_dat(clover_grid, 1, size, d_m, d_p, temp, "double", "work_array2");
  work_array3    = ops_decl_dat(clover_grid, 1, size, d_m, d_p, temp, "double", "work_array3");
  work_array4    = ops_decl_dat(clover_grid, 1, size, d_m, d_p, temp, "double", "work_array4");
  work_array5    = ops_decl_dat(clover_grid, 1, size, d_m, d_p, temp, "double", "work_array5");
  work_array6    = ops_decl_dat(clover_grid, 1, size, d_m, d_p, temp, "double", "work_array6");
  work_array7    = ops_decl_dat(clover_grid, 1, size, d_m, d_p, temp, "double", "work_array7");

  int size2[3] = {x_cells+5,1,1};
  d_m[0]=-2;d_m[1]=0;d_m[2]=0;d_p[0]=-2;d_p[1]=0;d_p[2]=0;
  cellx    = ops_decl_dat(clover_grid, 1, size2, d_m, d_p, temp, "double", "cellx");
  celldx   = ops_decl_dat(clover_grid, 1, size2, d_m, d_p, temp, "double", "celldx");

  int size3[3] = {1,y_cells+5,1};
  d_m[0]=0;d_m[1]=-2;d_m[2]=0;d_p[0]=0;d_p[1]=-2;d_p[2]=0;
  celly    = ops_decl_dat(clover_grid, 1, size3, d_m, d_p, temp, "double", "celly");
  celldy   = ops_decl_dat(clover_grid, 1, size3, d_m, d_p, temp, "double", "celldy");
  
  int size4[3] = {1,1,z_cells+5};
  d_m[0]=0;d_m[1]=0;d_m[2]=-2;d_p[0]=0;d_p[1]=0;d_p[2]=-2;
  cellz    = ops_decl_dat(clover_grid, 1, size4, d_m, d_p, temp, "double", "cellz");
  celldz   = ops_decl_dat(clover_grid, 1, size4, d_m, d_p, temp, "double", "celldz");

  d_m[0]=-2;d_m[1]=0;d_m[2]=0;d_p[0]=-3;d_p[1]=0;d_p[2]=0;
  vertexx  = ops_decl_dat(clover_grid, 1, size2, d_m, d_p, temp, "double", "vertexx");
  vertexdx = ops_decl_dat(clover_grid, 1, size2, d_m, d_p, temp, "double", "vertexdx");

  d_m[0]=0;d_m[1]=-2;d_m[2]=0;d_p[0]=0;d_p[1]=-3;d_p[2]=0;
  vertexy  = ops_decl_dat(clover_grid, 1, size3, d_m, d_p, temp, "double", "vertexy");
  vertexdy = ops_decl_dat(clover_grid, 1, size3, d_m, d_p, temp, "double", "vertexdy");

  d_m[0]=0;d_m[1]=0;d_m[2]=-2;d_p[0]=0;d_p[1]=0;d_p[2]=-3;
  vertexz  = ops_decl_dat(clover_grid, 1, size4, d_m, d_p, temp, "double", "vertexz");
  vertexdz = ops_decl_dat(clover_grid, 1, size4, d_m, d_p, temp, "double", "vertexdz");

  //contains x indicies from 0 to xmax+3 -- needed for initialization
  sub_block_list sb = OPS_sub_block_list[0];

  int* temp2 = NULL;
  d_m[0]=-2;d_m[1]=0;d_m[2]=0;d_p[0]=-3;d_p[1]=0;d_p[2]=0;
  xx  = ops_decl_dat(clover_grid, 1, size2, d_m, d_p, temp2, "int", "xx");
  for(int i=sb->istart[0]-2; i<sb->iend[0]+3+1; i++)
    ((int *)(xx->data))[i-d_m[0]-sb->istart[0]] = i - x_min;
  xx->dirty_hd=1;
  d_m[0]=0;d_m[1]=-2;d_m[2]=0;d_p[0]=0;d_p[1]=-3;d_p[2]=0;
  yy  = ops_decl_dat(clover_grid, 1, size3, d_m, d_p, temp2, "int", "yy");
  for(int i=sb->istart[1]-2; i<sb->iend[1]+3+1; i++)
    ((int *)(yy->data))[i-d_m[1]-sb->istart[1]] = i - y_min;
  yy->dirty_hd=1;
  d_m[0]=0;d_m[1]=0;d_m[2]=-2;d_p[0]=0;d_p[1]=0;d_p[2]=-3;
  zz  = ops_decl_dat(clover_grid, 1, size4, d_m, d_p, temp2, "int", "zz");
  for(int i=sb->istart[2]-2; i<sb->iend[2]+3+1; i++)
    ((int *)(zz->data))[i-d_m[1]-sb->istart[1]] = i - z_min;
  zz->dirty_hd=1;

  //
  //Declare commonly used stencils
  //
  int s3D_000[]         = {0,0,0};
  
  int s3D_000_P100[]      = {0,0,0, 1,0,0};
  int s3D_000_0P10[]      = {0,0,0, 0,1,0};
  int s3D_000_00P1[]      = {0,0,0, 0,0,1};
  
  int s3D_000_M100[]      = {0,0,0, -1,0,0};
  int s3D_000_0M10[]      = {0,0,0, 0,-1,0};
  int s3D_000_00M1[]      = {0,0,0, 0,0,-1};
  
  int s3D_000_P100_M100[]      = {0,0,0, 1,0,0, -1,0,0};
  int s3D_000_0P10_0M10[]      = {0,0,0, 0,1,0, 0,-1,0};
  int s3D_000_00P1_00M1[]      = {0,0,0, 0,0,1, 0,0,-1};
  
  int s3D_000_f0M1M1[]     = {0,0,0, 0,-1,0, 0,0,-1, 0,-1,-1};
  int s3D_000_fM10M1[]     = {0,0,0, -1,0,0, 0,0,-1, -1,0,-1};
  int s3D_000_fM1M10[]     = {0,0,0, 0,-1,0, -1,0,0, -1,-1,0};

  int s3D_000_f0P1P1[]     = {0,0,0, 0,1,0, 0,0,1, 0,1,1};
  int s3D_000_fP10P1[]     = {0,0,0, 1,0,0, 0,0,1, 1,0,1};
  int s3D_000_fP1P10[]     = {0,0,0, 0,1,0, 1,0,0, 1,1,0};
  
  int s3D_000_fP1P1P1[]    = {0,0,0, 1,0,0, 1,1,0, 1,1,1, 0,1,0, 0,1,1, 0,0,1, 1,0,1};
 
  int s3D_000_fP1M1M1[]    = {0,0,0, 1,0,0, 1,-1,0, 1,-1,-1, 0,-1,0, 0,-1,-1, 0,0,-1, 1,0,-1};
  int s3D_000_fM1P1M1[]    = {0,0,0, -1,0,0, -1,1,0, -1,1,-1, 0,1,0, 0,1,-1, 0,0,-1, -1,0,-1};
  int s3D_000_fM1M1P1[]    = {0,0,0, -1,0,0, -1,-1,0, -1,-1,1, 0,-1,0, 0,-1,1, 0,0,1, -1,0,1};

  int s3D_000_fM1P1P1[]    = {0,0,0, -1,0,0, -1,1,0, -1,1,1, 0,1,0,  0,1,1,  0,0,1,  -1,0,1};
  int s3D_000_fP1M1P1[]    = {0,0,0, 1,0,0, 1,-1,0,  1,-1,1, 0,-1,0, 0,-1,1, 0,0,1,  1,0,1};
  int s3D_000_fP1P1M1[]    = {0,0,0, 1,0,0, 1,1,0,   1,1,-1, 0,1,0,  0,1,-1, 0,0,-1, 1,0,-1};
 
  int s3D_000_fM1M1M1[]    = {0,0,0, -1,0,0, -1,-1,0, -1,-1,-1, 0,-1,0, 0,-1,-1, 0,0,-1, -1,0,-1};
  
  int s3D_000_P100_P200_M100[] = {0,0,0, 1,0,0, 2,0,0, -1,0,0};
  int s3D_000_0P10_0P20_0M10[] = {0,0,0, 0,1,0, 0,2,0, 0,-1,0};
  int s3D_000_00P1_00P2_00M1[] = {0,0,0, 0,0,1, 0,0,2, 0,0,-1};
  int s3D_000_P100_M100_M200[] = {0,0,0, 1,0,0, -1,0,0, -2,0,0};
  int s3D_000_0P10_0M10_0M20[] = {0,0,0, 0,1,0, 0,-1,0, 0,-2,0};
  int s3D_000_00P1_00M1_00M2[] = {0,0,0, 0,0,1, 0,0,-1, 0,0,-2};
  
  int s3D_P100_M100_0P10_0M10_00P1_00M1[] = {1,0,0, -1,0,0, 0,1,0, 0,-1,0, 0,0,1, 0,0,-1};

  int s3D_000_P200[]     = {0,0,0, 2,0,0};
  int s3D_000_0P20[]     = {0,0,0, 0,2,0};
  int s3D_000_00P2[]     = {0,0,0, 0,0,2};
  int s3D_000_M200[]     = {0,0,0, -2,0,0};
  int s3D_000_0M20[]     = {0,0,0, 0,-2,0};
  int s3D_000_00M2[]     = {0,0,0, 0,0,-2};
  int s3D_000_P300[]     = {0,0,0, 3,0,0};
  int s3D_000_0P30[]     = {0,0,0, 0,3,0};
  int s3D_000_00P3[]     = {0,0,0, 0,0,3};
  int s3D_000_M300[]     = {0,0,0, -3,0,0};
  int s3D_000_0M30[]     = {0,0,0, 0,-3,0};
  int s3D_000_00M3[]     = {0,0,0, 0,0,-3};
  int s3D_000_P400[]     = {0,0,0, 4,0,0};
  int s3D_000_0P40[]     = {0,0,0, 0,4,0};
  int s3D_000_00P4[]     = {0,0,0, 0,0,4};
  int s3D_000_M400[]     = {0,0,0, -4,0,0};
  int s3D_000_0M40[]     = {0,0,0, 0,-4,0};
  int s3D_000_00M4[]     = {0,0,0, 0,0,-4};

  int stride3D_x[] = {1,0,0};
  int stride3D_y[] = {0,1,0};
  int stride3D_z[] = {0,0,1};


  S3D_000         = ops_decl_stencil( 3, 1, s3D_000, "0,0,0");
  
  S3D_000_P100    = ops_decl_stencil( 3, 1, s3D_000_P100, "0,0,0:1,0,0");
  S3D_000_0P10    = ops_decl_stencil( 3, 1, s3D_000_0P10, "0,0,0:0,1,0");
  S3D_000_00P1    = ops_decl_stencil( 3, 1, s3D_000_00P1, "0,0,0:0,0,1");
  
  S3D_000_M100    = ops_decl_stencil( 3, 1, s3D_000_M100, "0,0,0:-1,0,0");
  S3D_000_0M10    = ops_decl_stencil( 3, 1, s3D_000_0M10, "0,0,0:0,-1,0");
  S3D_000_00M1    = ops_decl_stencil( 3, 1, s3D_000_00M1, "0,0,0:0,0,-1");
  
  S3D_000_f0M1M1 = ops_decl_stencil( 3, 4, s3D_000_f0M1M1, "f0,0,0:0,-1,-1");
  S3D_000_fM10M1 = ops_decl_stencil( 3, 4, s3D_000_fM10M1, "f0,0,0:-1,0,-1");
  S3D_000_fM1M10 = ops_decl_stencil( 3, 4, s3D_000_fM1M10, "f0,0,0:-1,-1,0");
  
  S3D_000_f0P1P1 = ops_decl_stencil( 3, 4, s3D_000_f0P1P1, "f0,0,0:0,1,1");
  S3D_000_fP10P1 = ops_decl_stencil( 3, 4, s3D_000_fP10P1, "f0,0,0:1,0,1");
  S3D_000_fP1P10 = ops_decl_stencil( 3, 4, s3D_000_fP1P10, "f0,0,0:1,1,0");
  
  
  S3D_000_fP1P1P1 = ops_decl_stencil( 3, 8, s3D_000_fP1P1P1, "f0,0,0:1,1,1");

  S3D_000_fP1M1M1 = ops_decl_stencil( 3, 8, s3D_000_fP1M1M1, "f0,0,0:1,-1,-1");
  S3D_000_fM1P1M1 = ops_decl_stencil( 3, 8, s3D_000_fM1P1M1, "f0,0,0:-1,1,-1");
  S3D_000_fM1M1P1 = ops_decl_stencil( 3, 8, s3D_000_fM1M1P1, "f0,0,0:-1,-1,1");

  S3D_000_fM1P1P1 = ops_decl_stencil( 3, 8, s3D_000_fM1P1P1, "f0,0,0:-1,1,1");
  S3D_000_fP1M1P1 = ops_decl_stencil( 3, 8, s3D_000_fP1M1P1, "f0,0,0:1,-1,1");
  S3D_000_fP1P1M1 = ops_decl_stencil( 3, 8, s3D_000_fP1P1M1, "f0,0,0:1,1,-1");

  S3D_000_fM1M1M1 = ops_decl_stencil( 3, 8, s3D_000_fM1M1M1, "f0,0,0:-1,-1,-1");
  
  S3D_000_P200     = ops_decl_stencil( 3, 2, s3D_000_P200, "0,0,0:2,0,0");
  S3D_000_0P20     = ops_decl_stencil( 3, 2, s3D_000_0P20, "0,0,0:0,2,0");
  S3D_000_00P2     = ops_decl_stencil( 3, 2, s3D_000_00P2, "0,0,0:0,0,2");
  
  S3D_000_M200     = ops_decl_stencil( 3, 2, s3D_000_M200, "0,0,0:-2,0,0");
  S3D_000_0M20     = ops_decl_stencil( 3, 2, s3D_000_0M20, "0,0,0:0,-2,0");
  S3D_000_00M2     = ops_decl_stencil( 3, 2, s3D_000_00M2, "0,0,0:0,0,-2");

  S3D_000_P300     = ops_decl_stencil( 3, 2, s3D_000_P300, "0,0,0:3,0,0");
  S3D_000_0P30     = ops_decl_stencil( 3, 2, s3D_000_0P30, "0,0,0:0,3,0");
  S3D_000_00P3     = ops_decl_stencil( 3, 2, s3D_000_00P3, "0,0,0:0,0,3");

  S3D_000_M300     = ops_decl_stencil( 3, 2, s3D_000_M300, "0,0,0:-3,0,0");
  S3D_000_0M30     = ops_decl_stencil( 3, 2, s3D_000_0M30, "0,0,0:0,-3,0");
  S3D_000_00M3     = ops_decl_stencil( 3, 2, s3D_000_00M3, "0,0,0:0,0,-3");

  S3D_000_P400     = ops_decl_stencil( 3, 2, s3D_000_P400, "0,0,0:4,0,0");
  S3D_000_0P40     = ops_decl_stencil( 3, 2, s3D_000_0P40, "0,0,0:0,4,0");
  S3D_000_00P4     = ops_decl_stencil( 3, 2, s3D_000_00P4, "0,0,0:0,0,4");

  S3D_000_M400     = ops_decl_stencil( 3, 2, s3D_000_M400, "0,0,0:-4,0,0");
  S3D_000_0M40     = ops_decl_stencil( 3, 2, s3D_000_0M40, "0,0,0:0,-4,0");
  S3D_000_00M4     = ops_decl_stencil( 3, 2, s3D_000_00M4, "0,0,0:0,0,-4");

  S3D_000_P100_P200_M100 = ops_decl_stencil( 3, 4, s3D_000_P100_P200_M100, "0,0,0:1,0,0:2,0,0:-1,0,0");
  S3D_000_0P10_0P20_0M10 = ops_decl_stencil( 3, 4, s3D_000_0P10_0P20_0M10, "0,0,0:1,0,0:0,2,0:0,-1,0");
  S3D_000_00P1_00P2_00M1 = ops_decl_stencil( 3, 4, s3D_000_00P1_00P2_00M1, "0,0,0:1,0,0:0,0,2:0,0,-1");
  S3D_000_P100_M100_M200 = ops_decl_stencil( 3, 4, s3D_000_P100_P200_M100, "0,0,0:1,0,0:-1,0,0:-2,0,0");
  S3D_000_0P10_0M10_0M20 = ops_decl_stencil( 3, 4, s3D_000_0P10_0P20_0M10, "0,0,0:1,0,0:0,-1,0:0,-2,0");
  S3D_000_00P1_00M1_00M2 = ops_decl_stencil( 3, 4, s3D_000_00P1_00P2_00M1, "0,0,0:1,0,0:0,0,-1:0,0,-2");
  S3D_P100_M100_0P10_0M10_00P1_00M1 = ops_decl_stencil( 3, 6, s3D_P100_M100_0P10_0M10_00P1_00M1, "1,0,0:-1,0,0:0,1,0:0,0,-1:0,0,1:0,0,-1");

  S3D_000_STRID3D_X = ops_decl_strided_stencil( 3, 1, s3D_000, stride3D_x, "s2D_000_stride3D_x");
  S3D_000_STRID3D_Y = ops_decl_strided_stencil( 3, 1, s3D_000, stride3D_y, "s2D_000_stride3D_y");
  S3D_000_STRID3D_Z = ops_decl_strided_stencil( 3, 1, s3D_000, stride3D_z, "s2D_000_stride3D_z");

  S3D_000_P100_STRID3D_X = ops_decl_strided_stencil( 3, 2, s3D_000_P100, stride3D_x, "s3D_000_P100_stride3D_x");
  S3D_000_0P10_STRID3D_Y = ops_decl_strided_stencil( 3, 2, s3D_000_0P10, stride3D_y, "s3D_000_0P10_stride3D_y");
  S3D_000_00P1_STRID3D_Z = ops_decl_strided_stencil( 3, 2, s3D_000_00P1, stride3D_z, "s3D_000_00P1_stride3D_z");
  
  S3D_000_P100_M100_STRID3D_X = ops_decl_strided_stencil( 3, 3, s3D_000_P100_M100, stride3D_x, "s3D_000_P100_M100_stride3D_x");
  S3D_000_0P10_0M10_STRID3D_Y = ops_decl_strided_stencil( 3, 3, s3D_000_0P10_0M10, stride3D_y, "s3D_000_0P10_0M10_stride3D_y");
  S3D_000_00P1_00M1_STRID3D_Z = ops_decl_strided_stencil( 3, 3, s3D_000_00P1_00M1, stride3D_z, "s3D_000_00P1_00M1_stride3D_z");

  S3D_000_P100_M100_M200_STRID3D_X = ops_decl_strided_stencil( 3, 4, s3D_000_P100_P200_M100, stride3D_x, "0,0,0:1,0,0:-1,0,0:-2,0,0");
  S3D_000_0P10_0M10_0M20_STRID3D_Y = ops_decl_strided_stencil( 3, 4, s3D_000_0P10_0P20_0M10, stride3D_y, "0,0,0:1,0,0:0,-1,0:0,-2,0");
  S3D_000_00P1_00M1_00M2_STRID3D_Z = ops_decl_strided_stencil( 3, 4, s3D_000_00P1_00P2_00M1, stride3D_z, "0,0,0:1,0,0:0,0,-1:0,0,-2");
  
  //print ops blocks and dats details
  ops_diagnostic_output();

}
