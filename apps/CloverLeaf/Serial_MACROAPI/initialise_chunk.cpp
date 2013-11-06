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

/** @brief Driver for chunk initialisation.
 *  @author Wayne Gaudin, converted to OPS by Gihan Mudalige
 *  @details Invokes the user specified chunk initialisation kernel.
**/

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

// OPS header file
#include "ops_seq_opt.h"
#include "ops_seq_macro.h"

#include "data.h"
#include "definitions.h"

//Cloverleaf kernels
#include "initialise_chunk_kernel.h"


void initialise_chunk_kernel_x_macro(double *vertexx, int *xx, double *vertexdx) {

  int x_min=field->x_min;
  int x_max=field->x_max;
  int y_min=field->y_min;
  int y_max=field->y_max;

  double min_x, min_y, d_x, d_y;

  d_x = (grid->xmax - grid->xmin)/(double)grid->x_cells;
  d_y = (grid->ymax - grid->ymin)/(double)grid->y_cells;

  min_x=grid->xmin+d_x*field->left;
  min_y=grid->ymin+d_y*field->bottom;

  vertexx[OPS_ACC0(0,0)] = min_x + d_x * (xx[OPS_ACC1(0,0)] - x_min);
  vertexdx[OPS_ACC2(0,0)] = (double)d_x;
}

void initialise_chunk_kernel_y_macro(double *vertexy, int *yy, double *vertexdy) {

  int x_min=field->x_min;
  int x_max=field->x_max;
  int y_min=field->y_min;
  int y_max=field->y_max;

  double min_x, min_y, d_x, d_y;

  d_x = (grid->xmax - grid->xmin)/(double)grid->x_cells;
  d_y = (grid->ymax - grid->ymin)/(double)grid->y_cells;

  min_x=grid->xmin+d_x*field->left;
  min_y=grid->ymin+d_y*field->bottom;

  vertexy[OPS_ACC0(0,0)] = min_y + d_y * (yy[OPS_ACC1(0,0)] - y_min);
  vertexdy[OPS_ACC2(0,0)] = (double)d_y;
}



void initialise_chunk_kernel_cellx_macro(double *vertexx, double* cellx, double *celldx) {

  int x_min=field->x_min;
  int x_max=field->x_max;;
  int y_min=field->y_min;
  int y_max=field->y_max;

  double min_x, min_y, d_x, d_y;

  d_x = (grid->xmax - grid->xmin)/(double)grid->x_cells;
  d_y = (grid->ymax - grid->ymin)/(double)grid->y_cells;

  min_x=grid->xmin+d_x;
  min_y=grid->ymin+d_y;

  cellx[OPS_ACC1(0,0)]  = 0.5*( vertexx[OPS_ACC0(0,0)] + vertexx[OPS_ACC0(1,0)] );
  celldx[OPS_ACC2(0,0)]  = d_x;

}

void initialise_chunk_kernel_celly_macro(double *vertexy, double *celly, double *celldy) {

  int x_min=field->x_min;
  int x_max=field->x_max;;
  int y_min=field->y_min;
  int y_max=field->y_max;

  double min_x, min_y, d_x, d_y;

  d_x = (grid->xmax - grid->xmin)/(double)grid->x_cells;
  d_y = (grid->ymax - grid->ymin)/(double)grid->y_cells;

  min_x=grid->xmin+d_x;
  min_y=grid->ymin+d_y;

  celly[OPS_ACC1(0,0)] = 0.5*( vertexy[OPS_ACC0(0,0)]+ vertexy[OPS_ACC0(0,1)] );
  celldy[OPS_ACC2(0,0)] = d_y;


}

void initialise_chunk_kernel_volume_macro(double *volume, double *celldy, double *xarea,
                                         double *celldx, double *yarea) {

  double d_x, d_y;

  d_x = (grid->xmax - grid->xmin)/(double)grid->x_cells;
  d_y = (grid->ymax - grid->ymin)/(double)grid->y_cells;

  volume[OPS_ACC0(0,0)] = d_x*d_y;
  xarea[OPS_ACC2(0,0)] = celldy[OPS_ACC1(0,0)];
  yarea[OPS_ACC4(0,0)] = celldx[OPS_ACC3(0,0)];
}


void initialise_chunk()
{
  //initialize sizes using global values
  int x_cells = grid->x_cells;
  int y_cells = grid->y_cells;
  int x_min = field->x_min;
  int x_max = field->x_max;
  int y_min = field->y_min;
  int y_max = field->y_max;

  int self[] = {0,0};
  ops_stencil sten1 = ops_decl_stencil( 2, 1, self, "self");

  int rangex[] = {x_min-2, x_max+3, 0, 1};
  ops_par_loop_macro(initialise_chunk_kernel_x_macro, "initialise_chunk_kernel_x_macro", 2, rangex,
               ops_arg_dat(vertexx, S2D_00, "double", OPS_WRITE),
               ops_arg_dat(xx, S2D_00, "int", OPS_READ),
               ops_arg_dat(vertexdx, S2D_00, "double", OPS_WRITE));

  int rangey[] = {0, 1, y_min-2, y_max+3};
  ops_par_loop_macro(initialise_chunk_kernel_y_macro, "initialise_chunk_kernel_y_macro", 2, rangey,
               ops_arg_dat(vertexy, S2D_00, "double", OPS_WRITE),
               ops_arg_dat(yy, S2D_00, "int", OPS_READ),
               ops_arg_dat(vertexdy, S2D_00, "double", OPS_WRITE));

  rangex[0] = x_min-2; rangex[1] = x_max+2; rangex[2] = 0; rangex[3] = 1;
  ops_par_loop_macro(initialise_chunk_kernel_cellx_macro, "initialise_chunk_kernel_cellx_macro", 2, rangex,
               ops_arg_dat(vertexx, S2D_00_P10, "double", OPS_READ),
               ops_arg_dat(cellx, S2D_00, "double", OPS_WRITE),
               ops_arg_dat(celldx, S2D_00, "double", OPS_WRITE));

  rangey[0] = 0; rangey[1] = 1; rangey[2] = y_min-2; rangey[3] = y_max+2;
  ops_par_loop_macro(initialise_chunk_kernel_celly_macro, "initialise_chunk_kernel_celly_macro", 2, rangey,
               ops_arg_dat(vertexy, S2D_00_0P1, "double", OPS_READ),
               ops_arg_dat(celly, S2D_00, "double", OPS_WRITE),
               ops_arg_dat(celldy, S2D_00, "double", OPS_WRITE));

  int rangexy[] = {x_min-2,x_max+2,y_min-2,y_max+2};
  ops_par_loop_macro(initialise_chunk_kernel_volume_macro, "initialise_chunk_kernel_volume_macro", 2, rangexy,
    ops_arg_dat(volume, S2D_00, "double", OPS_WRITE),
    ops_arg_dat(celldy, S2D_00_STRID2D_Y, "double", OPS_READ),
    ops_arg_dat(xarea, S2D_00, "double", OPS_WRITE),
    ops_arg_dat(celldx, S2D_00_STRID2D_X, "double", OPS_READ),
    ops_arg_dat(yarea, S2D_00, "double", OPS_WRITE));
}
