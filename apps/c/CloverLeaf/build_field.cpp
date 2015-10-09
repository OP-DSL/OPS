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
#define OPS_2D
#include "ops_seq.h"


#include "data.h"
#include "definitions.h"


void test_kernel(double *volume, double *xarea, double *yarea) {

  printf("%lf ",volume[OPS_ACC0(0,0)]);
  printf("%lf ",xarea[OPS_ACC1(0,0)]);
  printf("%lf ",yarea[OPS_ACC2(0,0)]);

}


void build_field()
{
  //initialize sizes using global values
  int x_cells = grid.x_cells;
  int y_cells = grid.y_cells;
  int x_min = field.x_min;
  int x_max = field.x_max;
  int y_min = field.y_min;
  int y_max = field.y_max;

  ops_printf("Global x_min = %d, y_min = %d\n",x_min,y_min);
  ops_printf("Global x_max = %d, y_max = %d\n",x_max,y_max);

  /**----------------------------OPS Declarations----------------------------**/

  clover_grid = ops_decl_block(2, "clover grid");

  //
  //declare data on blocks
  //
  int d_p[2] = {2,2}; //max halo depths for the dat in the possitive direction
  int d_m[2] = {-2,-2}; //max halo depths for the dat in the negative direction
  int size[2] = {x_cells+5, y_cells+5}; //size of the dat
  int base[2] = {0,0};
  double* temp = NULL;

  density0    = ops_decl_dat(clover_grid, 1, size, base, d_m, d_p, temp, "double", "density0");
  density1    = ops_decl_dat(clover_grid, 1, size, base, d_m, d_p, temp, "double", "density1");
  energy0     = ops_decl_dat(clover_grid, 1, size, base, d_m, d_p, temp, "double", "energy0");
  energy1     = ops_decl_dat(clover_grid, 1, size, base, d_m, d_p, temp, "double", "energy1");
  pressure    = ops_decl_dat(clover_grid, 1, size, base, d_m, d_p, temp, "double", "pressure");
  viscosity   = ops_decl_dat(clover_grid, 1, size, base, d_m, d_p, temp, "double", "viscosity");
  soundspeed  = ops_decl_dat(clover_grid, 1, size, base, d_m, d_p, temp, "double", "soundspeed");
  volume      = ops_decl_dat(clover_grid, 1, size, base, d_m, d_p, temp, "double", "volume");

  size[0]++; size[1]++;
  xvel0    = ops_decl_dat(clover_grid, 1, size, base, d_m, d_p, temp, "double", "xvel0");
  xvel1    = ops_decl_dat(clover_grid, 1, size, base, d_m, d_p, temp, "double", "xvel1");

  yvel0    = ops_decl_dat(clover_grid, 1, size, base, d_m, d_p, temp, "double", "yvel0");
  yvel1    = ops_decl_dat(clover_grid, 1, size, base, d_m, d_p, temp, "double", "yvel1");

  work_array1    = ops_decl_dat(clover_grid, 1, size, base, d_m, d_p, temp, "double", "work_array1");
  work_array2    = ops_decl_dat(clover_grid, 1, size, base, d_m, d_p, temp, "double", "work_array2");
  work_array3    = ops_decl_dat(clover_grid, 1, size, base, d_m, d_p, temp, "double", "work_array3");
  work_array4    = ops_decl_dat(clover_grid, 1, size, base, d_m, d_p, temp, "double", "work_array4");
  work_array5    = ops_decl_dat(clover_grid, 1, size, base, d_m, d_p, temp, "double", "work_array5");
  work_array6    = ops_decl_dat(clover_grid, 1, size, base, d_m, d_p, temp, "double", "work_array6");
  work_array7    = ops_decl_dat(clover_grid, 1, size, base, d_m, d_p, temp, "double", "work_array7");

  size[0] = x_cells+6; size[1] = y_cells+5;
  vol_flux_x  = ops_decl_dat(clover_grid, 1, size, base, d_m, d_p, temp, "double", "vol_flux_x");
  mass_flux_x = ops_decl_dat(clover_grid, 1, size, base, d_m, d_p, temp, "double", "mass_flux_x");
  xarea       = ops_decl_dat(clover_grid, 1, size, base, d_m, d_p, temp, "double", "xarea");

  size[0] = x_cells+5; size[1] = y_cells+6;
  vol_flux_y  = ops_decl_dat(clover_grid, 1, size, base, d_m, d_p, temp, "double", "vol_flux_y");
  mass_flux_y = ops_decl_dat(clover_grid, 1, size, base, d_m, d_p, temp, "double", "mass_flux_y");
  yarea       = ops_decl_dat(clover_grid, 1, size, base, d_m, d_p, temp, "double", "yarea");

  int size2[2] = {x_cells+5,1};
  d_m[0]=-2;d_m[1]=0;d_p[0]=2;d_p[1]=0;
  cellx    = ops_decl_dat(clover_grid, 1, size2, base, d_m, d_p, temp, "double", "cellx");
  celldx   = ops_decl_dat(clover_grid, 1, size2, base, d_m, d_p, temp, "double", "celldx");

  int size3[2] = {1,y_cells+5};
  d_m[0]=0;d_m[1]=-2;d_p[0]=0;d_p[1]=2;
  celly    = ops_decl_dat(clover_grid, 1, size3, base, d_m, d_p, temp, "double", "celly");
  celldy   = ops_decl_dat(clover_grid, 1, size3, base, d_m, d_p, temp, "double", "celldy");

  int size4[2] = {x_cells+6,1};
  d_m[0]=-2;d_m[1]=0;d_p[0]=2;d_p[1]=0;
  vertexx  = ops_decl_dat(clover_grid, 1, size4, base, d_m, d_p, temp, "double", "vertexx");
  vertexdx = ops_decl_dat(clover_grid, 1, size4, base, d_m, d_p, temp, "double", "vertexdx");

  int size5[2] = {1,y_cells+6};
  d_m[0]=0;d_m[1]=-2;d_p[0]=0;d_p[1]=2;
  vertexy  = ops_decl_dat(clover_grid, 1, size5, base, d_m, d_p, temp, "double", "vertexy");
  vertexdy = ops_decl_dat(clover_grid, 1, size5, base, d_m, d_p, temp, "double", "vertexdy");

  //contains x indicies from 0 to xmax+3 -- needed for initialization

  int* temp2 = NULL;
  d_m[0]=-2;d_m[1]=0;d_p[0]=2;d_p[1]=0;
  xx  = ops_decl_dat(clover_grid, 1, size4, base, d_m, d_p, temp2, "int", "xx");

  d_m[0]=0;d_m[1]=-2;d_p[0]=0;d_p[1]=2;
  yy  = ops_decl_dat(clover_grid, 1, size5, base, d_m, d_p, temp2, "int", "yy");

  //
  //Declare commonly used stencils
  //
  int s2D_00[]         = {0,0};
  int s2D_00_P10[]     = {0,0, 1,0};
  int s2D_00_0P1[]     = {0,0, 0,1};
  int s2D_00_M10[]     = {0,0, -1,0};
  int s2D_00_0M1[]     = {0,0, 0,-1};
  int s2D_00_P10_M10[] = {0,0, 1,0, -1,0};
  int s2D_00_0P1_0M1[] = {0,0, 0,1, 0,-1};
  int s2D_00_M10_M20[] = {0,0, -1,0, -2,0};
  int s2D_00_0M1_0M2[] = {0,0, 0,-1, 0,-2};
  int s2D_00_P20[]     = {0,0, 2,0};
  int s2D_00_0P2[]     = {0,0, 0,2};
  int s2D_00_M20[]     = {0,0, -2,0};
  int s2D_00_0M2[]     = {0,0, 0,-2};
  int s2D_00_P30[]     = {0,0, 3,0};
  int s2D_00_0P3[]     = {0,0, 0,3};
  int s2D_00_M30[]     = {0,0, -3,0};
  int s2D_00_0M3[]     = {0,0, 0,-3};
  int s2D_00_P40[]     = {0,0, 4,0};
  int s2D_00_0P4[]     = {0,0, 0,4};
  int s2D_00_M40[]     = {0,0, -4,0};
  int s2D_00_0M4[]     = {0,0, 0,-4};

  int s2D_00_P10_0P1_P1P1[]  = {0,0, 1,0, 0,1, 1,1};
  int s2D_00_M10_0M1_M1M1[]  = {0,0, -1,0, 0,-1, -1,-1};
  int s2D_00_P10_0M1_P1M1[] = {0,0, 1,0, 0,-1, 1,-1};
  int s2D_00_0P1_M10_M1P1[] = {0,0, 0,1, -1,0, -1,1};

  int s2D_10_M10_01_0M1[]  = {1,0, -1,0, 0,1, 0,-1};

  int s2D_00_P10_M10_M20[] = {0,0, 1,0, -1,0, -2,0};
  int s2D_00_0P1_0M1_0M2[] = {0,0, 0,1, 0,-1, 0,-2};
  int s2D_00_P10_P20_M10[] = {0,0, 1,0, 2,0, -1,0};
  int s2D_00_0P1_0P2_0M1[] = {0,0, 0,1, 0,2, 0,-1};


  int stride2D_x[] = {1,0};
  int stride2D_y[] = {0,1};

  S2D_00         = ops_decl_stencil( 2, 1, s2D_00, "00");
  S2D_00_P10     = ops_decl_stencil( 2, 2, s2D_00_P10, "0,0:1,0");
  S2D_00_0P1     = ops_decl_stencil( 2, 2, s2D_00_0P1, "0,0:0,1");
  S2D_00_M10     = ops_decl_stencil( 2, 2, s2D_00_M10, "0,0:-1,0");
  S2D_00_0M1     = ops_decl_stencil( 2, 2, s2D_00_0M1, "0,0:0,-1");

  S2D_00_P10_M10 = ops_decl_stencil( 2, 3, s2D_00_P10_M10, "0,0:1,0:1,0");
  S2D_00_0P1_0M1 = ops_decl_stencil( 2, 3, s2D_00_0P1_0M1, "0,0:0,1:0,-1");

  S2D_00_M10_M20 = ops_decl_stencil( 2, 3, s2D_00_M10_M20, "0,0:-1,0:-2,0");
  S2D_00_0M1_0M2 = ops_decl_stencil( 2, 3, s2D_00_0M1_0M2, "0,0:0,-1:0,-2");

  S2D_00_P20     = ops_decl_stencil( 2, 2, s2D_00_P20, "0,0:2,0");
  S2D_00_0P2     = ops_decl_stencil( 2, 2, s2D_00_0P2, "0,0:0,2");

  S2D_00_M20     = ops_decl_stencil( 2, 2, s2D_00_M20, "0,0:-2,0");
  S2D_00_0M2     = ops_decl_stencil( 2, 2, s2D_00_0M2, "0,0:0,-2");

  S2D_00_P30     = ops_decl_stencil( 2, 2, s2D_00_P30, "0,0:3,0");
  S2D_00_0P3     = ops_decl_stencil( 2, 2, s2D_00_0P3, "0,0:0,3");

  S2D_00_M30     = ops_decl_stencil( 2, 2, s2D_00_M30, "0,0:-3,0");
  S2D_00_0M3     = ops_decl_stencil( 2, 2, s2D_00_0M3, "0,0:0,-3");

  S2D_00_P40     = ops_decl_stencil( 2, 2, s2D_00_P40, "0,0:4,0");
  S2D_00_0P4     = ops_decl_stencil( 2, 2, s2D_00_0P4, "0,0,0,-4");

  S2D_00_M40     = ops_decl_stencil( 2, 2, s2D_00_M40, "0,0:-4,0");
  S2D_00_0M4     = ops_decl_stencil( 2, 2, s2D_00_0M4, "0,0:0,-4");

  S2D_00_P10_0P1_P1P1 = ops_decl_stencil( 2, 4, s2D_00_P10_0P1_P1P1, "0,0:1,0:0,1:1,1");
  S2D_00_M10_0M1_M1M1 = ops_decl_stencil( 2, 4, s2D_00_M10_0M1_M1M1, "0,0:-1,0:0,-1:-1,-1");
  S2D_00_P10_0M1_P1M1= ops_decl_stencil( 2, 4, s2D_00_P10_0M1_P1M1, "0,0:1,0:0,-1:1,-1");
  S2D_00_0P1_M10_M1P1= ops_decl_stencil( 2, 4, s2D_00_0P1_M10_M1P1, "0,0:0,1:-1,0:-1,1");

  S2D_10_M10_01_0M1 = ops_decl_stencil( 2, 4, s2D_10_M10_01_0M1, "1,0:-1,0:0,1:0,-1");

  S2D_00_P10_M10_M20 = ops_decl_stencil( 2, 4, s2D_00_P10_M10_M20, "0,0:1,0:-1,0:-2,0");
  S2D_00_P10_M10_STRID2D_X = ops_decl_strided_stencil( 2, 3, s2D_00_P10_M10, stride2D_x, "self_stride2D_x");

  S2D_00_0P1_0M1_0M2 = ops_decl_stencil( 2, 4, s2D_00_0P1_0M1_0M2, "0,0:0,1:0-1:0,-2");
  S2D_00_0P1_0M1_STRID2D_Y = ops_decl_strided_stencil( 2, 3, s2D_00_0P1_0M1, stride2D_y, "self_stride2D_y");

  S2D_00_P10_P20_M10 = ops_decl_stencil( 2, 4, s2D_00_P10_P20_M10, "0,0:1,0:2,0:-1,0}");
  S2D_00_0P1_0P2_0M1 = ops_decl_stencil( 2, 4, s2D_00_0P1_0P2_0M1, "0,0,:0,1,:0,2,:0,-1");


  S2D_00_STRID2D_X = ops_decl_strided_stencil( 2, 1, s2D_00, stride2D_x, "s2D_00_stride2D_x");
  S2D_00_STRID2D_Y = ops_decl_strided_stencil( 2, 1, s2D_00, stride2D_y, "s2D_00_stride2D_y");

  S2D_00_P10_STRID2D_X = ops_decl_strided_stencil( 2, 2, s2D_00_P10, stride2D_x, "s2D_00_P10_stride2D_x");
  S2D_00_0P1_STRID2D_Y = ops_decl_strided_stencil( 2, 2, s2D_00_0P1, stride2D_y, "s2D_00_0P1_stride2D_y");

  S2D_00_M10_STRID2D_X = ops_decl_strided_stencil( 2, 2, s2D_00_M10, stride2D_x, "s2D_00_M10_stride2D_x");
  S2D_00_0M1_STRID2D_Y = ops_decl_strided_stencil( 2, 2, s2D_00_0M1, stride2D_y, "s2D_00_0M1_stride2D_y");

  red_local_dt = ops_decl_reduction_handle(sizeof(double), "double", "local_dt");
  red_xl_pos = ops_decl_reduction_handle(sizeof(double), "double", "xl_pos");
  red_yl_pos = ops_decl_reduction_handle(sizeof(double), "double", "yl_pos");
  red_vol = ops_decl_reduction_handle(sizeof(double), "double", "vol");
  red_mass = ops_decl_reduction_handle(sizeof(double), "double", "mass");
  red_ie = ops_decl_reduction_handle(sizeof(double), "double", "ie");
  red_ke = ops_decl_reduction_handle(sizeof(double), "double", "ke");
  red_press = ops_decl_reduction_handle(sizeof(double), "double", "press");
  red_output = ops_decl_reduction_handle(12*sizeof(double), "double", "output");

  //decompose the block
  ops_partition("2D_BLOCK_DECOMPSE");
  ops_checkpointing_init("check.h5", 15.0);
  //print ops blocks and dats details
  ops_diagnostic_output();

}
