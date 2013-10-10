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

void build_field()
{
  //initialize sizes using global values
  int x_cells = grid->x_cells;
  int y_cells = grid->y_cells;
  int x_min = field->x_min;
  int x_max = field->x_max;
  int y_min = field->y_min;
  int y_max = field->y_max;

  /**----------------------------OPS Declarations----------------------------**/

  int dims[2] = {x_cells, y_cells};  //cloverleaf 2D block dimensions
  ops_block clover_grid = ops_decl_block(2, dims, "grid");

  //declare edges of block
  dims[0] = x_cells; dims[1] = 1;
  clover_xedge = ops_decl_block(2, dims, "xedge");

  dims[0] = 1; dims[1] = y_cells;
  clover_yedge = ops_decl_block(2, dims, "yedge");

  //
  //declare data on blocks
  //
  int offset[2] = {-2,-2};
  int size[2] = {(x_max+2)-(x_min-2), (y_max+2)-(y_min-2)};
  double* temp = NULL;

  density0    = ops_decl_dat(clover_grid, 1, size, offset, temp, "double", "density0");
  density1    = ops_decl_dat(clover_grid, 1, size, offset, temp, "double", "density1");
  energy0     = ops_decl_dat(clover_grid, 1, size, offset, temp, "double", "energy0");
  energy1     = ops_decl_dat(clover_grid, 1, size, offset, temp, "double", "energy1");
  pressure    = ops_decl_dat(clover_grid, 1, size, offset, temp, "double", "pressure");
  viscosity   = ops_decl_dat(clover_grid, 1, size, offset, temp, "double", "viscosity");
  soundspeed  = ops_decl_dat(clover_grid, 1, size, offset, temp, "double", "soundspeed");
  volume      = ops_decl_dat(clover_grid, 1, size, offset, temp, "double", "volume");

  size[0] = (x_max+3)-(x_min-2); size[1] = (y_max+3)-(y_min-2);
  xvel0    = ops_decl_dat(clover_grid, 1, size, offset, temp, "double", "xvel0");
  xvel1    = ops_decl_dat(clover_grid, 1, size, offset, temp, "double", "xvel1");
  yvel0    = ops_decl_dat(clover_grid, 1, size, offset, temp, "double", "yvel0");
  yvel1    = ops_decl_dat(clover_grid, 1, size, offset, temp, "double", "yvel1");

  size[0] = (x_max+3)-(x_min-2); size[1] = (y_max+2)-(y_min-2);
  vol_flux_x  = ops_decl_dat(clover_grid, 1, size, offset, temp, "double", "vol_flux_x");
  mass_flux_x = ops_decl_dat(clover_grid, 1, size, offset, temp, "double", "mass_flux_x");
  xarea       = ops_decl_dat(clover_grid, 1, size, offset, temp, "double", "xarea");

  size[0] = (x_max+2)-(x_min-2); size[1] = (y_max+3)-(y_min-2);
  vol_flux_y  = ops_decl_dat(clover_grid, 1, size, offset, temp, "double", "vol_flux_y");
  mass_flux_y = ops_decl_dat(clover_grid, 1, size, offset, temp, "double", "mass_flux_y");
  yarea       = ops_decl_dat(clover_grid, 1, size, offset, temp, "double", "yarea");

  size[0] = (x_max+3)-(x_min-2); size[1] = (y_max+3)-(y_min-2);
  work_array1    = ops_decl_dat(clover_grid, 1, size, offset, temp, "double", "work_array1");
  work_array2    = ops_decl_dat(clover_grid, 1, size, offset, temp, "double", "work_array2");
  work_array3    = ops_decl_dat(clover_grid, 1, size, offset, temp, "double", "work_array3");
  work_array4    = ops_decl_dat(clover_grid, 1, size, offset, temp, "double", "work_array4");
  work_array5    = ops_decl_dat(clover_grid, 1, size, offset, temp, "double", "work_array5");
  work_array6    = ops_decl_dat(clover_grid, 1, size, offset, temp, "double", "work_array6");
  work_array7    = ops_decl_dat(clover_grid, 1, size, offset, temp, "double", "work_array7");

  int size2[2] = {(x_max+2)-(x_min-2),1};
  int size3[2] = {1, (y_max+2)-(y_min-2)};
  int size4[2] = {(x_max+3)-(x_min-2),1};
  int size5[2] = {1,(y_max+3)-(y_min-2)};
  int offsetx[2] = {-2,0};
  int offsety[2] = {0,-2};
  cellx    = ops_decl_dat(clover_xedge, 1, size2, offsetx, temp, "double", "cellx");
  celly    = ops_decl_dat(clover_yedge, 1, size3, offsety, temp, "double", "celly");
  vertexx  = ops_decl_dat(clover_xedge, 1, size4, offsetx, temp, "double", "vertexx");
  vertexy  = ops_decl_dat(clover_yedge, 1, size5, offsety, temp, "double", "vertexy");
  celldx   = ops_decl_dat(clover_xedge, 1, size2, offsetx, temp, "double", "celldx");
  celldy   = ops_decl_dat(clover_yedge, 1, size3, offsety, temp, "double", "celldy");
  vertexdx = ops_decl_dat(clover_xedge, 1, size4, offsetx, temp, "double", "vertexdx");
  vertexdy = ops_decl_dat(clover_yedge, 1, size5, offsety, temp, "double", "vertexdy");

  //contains x indicies from 0 to xmax+3 -- needed for initialization
  int* xindex = (int *)xmalloc(sizeof(int)*size4[0]);
  for(int i=x_min-2; i<x_max+3; i++) xindex[i-offsetx[0]] = i - x_min;
  xx  = ops_decl_dat(clover_xedge, 1, size4, offsetx, xindex, "int", "xx");

  //contains y indicies from 0 to ymax+3 -- needed for initialization
  int* yindex = (int *)xmalloc(sizeof(int)*size5[1]);
  for(int i=y_min-2; i<y_max+3; i++) yindex[i-offsety[1]] = i - y_min;
  yy  = ops_decl_dat(clover_yedge, 1, size5, offsety, yindex, "int", "yy");

  //
  //Declare commonly used stencils
  //
  int s2D[]            = {0,0};
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

  S2D_00         = ops_decl_stencil( 2, 1, s2D, "00");

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

  S2D_00_P30 = ops_decl_stencil( 2, 2, s2D_00_P30, "0,0:3,0");
  S2D_00_0P3 = ops_decl_stencil( 2, 2, s2D_00_0P3, "0,0:0,3");

  S2D_00_M30 = ops_decl_stencil( 2, 2, s2D_00_M30, "0,0:-3,0");
  S2D_00_0M3 = ops_decl_stencil( 2, 2, s2D_00_0M3, "0,0:0,-3");

  S2D_00_P40 = ops_decl_stencil( 2, 2, s2D_00_P40, "0,0:4,0");
  S2D_00_0P4 = ops_decl_stencil( 2, 2, s2D_00_0P4, "0,0,0,-4");

  S2D_00_M40 = ops_decl_stencil( 2, 2, s2D_00_M40, "0,0:=4,0");
  S2D_00_0M4 = ops_decl_stencil( 2, 2, s2D_00_0M4, "0,0:0,-4");


  int s2D_00_P10_0P1_P1P1[]  = {0,0, 1,0, 0,1, 1,1};
  S2D_00_P10_0P1_P1P1 = ops_decl_stencil( 2, 4, s2D_00_P10_0P1_P1P1, "0,0:1,0:0,1:1,1");

  int self2D_minus1xy[]  = {0,0, -1,0, 0,-1, -1,-1};
  sten_self2D_minus1xy = ops_decl_stencil( 2, 4, self2D_minus1xy, "self2D_minus1xy");

  int self2D_plus1x_minus1y[] = {0,0, 1,0, 0,-1, 1,-1};
  int self2D_plus1y_minus1x[] = {0,0, 0,1, -1,0, -1,1};
  sten_self2D_plus1x_minus1y= ops_decl_stencil( 2, 4, self2D_plus1x_minus1y, "self2D_plus1x_minus1y");
  sten_self2D_plus1y_minus1x= ops_decl_stencil( 2, 4, self2D_plus1y_minus1x, "self2D_plus1y_minus1x");


  int self2D_4point1xy[]  = {1,0, -1,0, 0,1, 0,-1};
  sten_self2D_4point1xy = ops_decl_stencil( 2, 4, self2D_4point1xy, "self2D_4point1xy");


  int self2D_plus_1_minus1_2_x[] = {0,0, 1,0, -1,0, -2,0};
  int self2D_plus_1_minus1_2_y[] = {0,0, 0,1, 0,-1, 0,-2};

  int self2D_plus_1_2_minus1_x[] = {0,0, 1,0, 2,0, -1,0};
  int self2D_plus_1_2_minus1_y[] = {0,0, 0,1, 0,2, 0,-1};

  int stride2D_x[] = {1,0};
  int stride2D_y[] = {0,1};
  int stride2D_null[] = {0,0};

  int xmax2D[] = {x_max+2,0};
  int ymax2D[] = {0,y_max+2};


  sten_self_stride2D_x = ops_decl_strided_stencil( 2, 1, s2D, stride2D_x, "self_stride2D_x");
  sten_self_stride2D_y = ops_decl_strided_stencil( 2, 1, s2D, stride2D_y, "self_stride2D_y");

  sten_self_plus1_stride2D_x = ops_decl_strided_stencil( 2, 2, s2D_00_P10, stride2D_x, "self_stride2D_x");
  sten_self_plus1_stride2D_y = ops_decl_strided_stencil( 2, 2, s2D_00_0P1, stride2D_y, "self_stride2D_y");

  sten_self_minus1_stride2D_x = ops_decl_strided_stencil( 2, 2, s2D_00_M10, stride2D_x, "self_stride2D_x");
  sten_self_minus1_stride2D_y = ops_decl_strided_stencil( 2, 2, s2D_00_0M1, stride2D_y, "self_stride2D_y");

  sten_self2D_plus_1_minus1_2_x = ops_decl_stencil( 2, 4, self2D_plus_1_minus1_2_x, "self2D_plus_1_minus1_2_x");
  sten_self_plus_1_minus1_2_x_stride2D_x = ops_decl_strided_stencil( 2, 4, self2D_plus_1_minus1_2_x, stride2D_x, "self_stride2D_x");
  sten_self2D_plus_1_minus1_2_y = ops_decl_stencil( 2, 4, self2D_plus_1_minus1_2_y, "self2D_plus_1_minus1_2_y");
  sten_self_plus_1_minus1_2_y_stride2D_y = ops_decl_strided_stencil( 2, 4, self2D_plus_1_minus1_2_y, stride2D_y, "self_stride2D_y");


  sten_self2D_plus_1_2_minus1x = ops_decl_stencil( 2, 4, self2D_plus_1_2_minus1_x, "self2D_plus_1_2_minus1_x");
  sten_self2D_plus_1_2_minus1y = ops_decl_stencil( 2, 4, self2D_plus_1_2_minus1_y, "self2D_plus_1_2_minus1_y");


  sten_self_stride2D_xmax = ops_decl_strided_stencil( 2, 1, xmax2D, stride2D_y, "self_stride2D_xmax");
  sten_self_nullstride2D_xmax = ops_decl_strided_stencil( 2, 1, xmax2D, stride2D_null, "self_nullstride2D_xmax");
  sten_self_stride2D_ymax = ops_decl_strided_stencil( 2, 1, ymax2D, stride2D_x, "self_stride2D_ymax");
  sten_self_nullstride2D_ymax = ops_decl_strided_stencil( 2, 1, ymax2D, stride2D_null, "self_nullstride2D_ymax");


  //print ops blocks and dats details
  ops_diagnostic_output();

}
