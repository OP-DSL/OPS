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


#include "ops_seq.h"

#include "data.h"
#include "definitions.h"

//Cloverleaf kernels
#include "initialise_chunk_kernel.h"


void initialise_chunk()
{
  //initialize sizes using global values
  int x_cells = grid.x_cells;
  int y_cells = grid.y_cells;
  int x_min = field.x_min;
  int x_max = field.x_max;
  int y_min = field.y_min;
  int y_max = field.y_max;

  //int rangex[] = {x_min-2, x_max+3, 0, 1};
  //int rangey[] = {0, 1, y_min-2, y_max+3};

  int rangex[] = {x_min-2, x_max+3, y_min-2, y_max+3};
  int rangey[] = {x_min-2, x_max+3, y_min-2, y_max+3};
  int rangefull[] = {-2, x_cells+8, -2, y_cells+8};

  ops_par_loop(initialise_chunk_kernel_xx, "initialise_chunk_kernel_xx", clover_grid, 2, rangefull,
               ops_arg_dat(xx, S2D_00_STRID2D_X, "int", OPS_WRITE),
               ops_arg_idx());

  ops_par_loop(initialise_chunk_kernel_yy, "initialise_chunk_kernel_yy", clover_grid, 2, rangefull,
               ops_arg_dat(yy, S2D_00_STRID2D_Y, "int", OPS_WRITE),
               ops_arg_idx());

  ops_par_loop(initialise_chunk_kernel_x, "initialise_chunk_kernel_x", clover_grid, 2, rangex,
               ops_arg_dat(vertexx, S2D_00_STRID2D_X, "double", OPS_WRITE),
               ops_arg_dat(xx, S2D_00_STRID2D_X, "int", OPS_READ),
               ops_arg_dat(vertexdx, S2D_00_STRID2D_X, "double", OPS_WRITE));

  ops_par_loop(initialise_chunk_kernel_y, "initialise_chunk_kernel_y", clover_grid, 2, rangey,
               ops_arg_dat(vertexy, S2D_00_STRID2D_Y, "double", OPS_WRITE),
               ops_arg_dat(yy, S2D_00_STRID2D_Y, "int", OPS_READ),
               ops_arg_dat(vertexdy, S2D_00_STRID2D_Y, "double", OPS_WRITE));

  rangex[0] = x_min-2; rangex[1] = x_max+2; rangex[2] = y_min-2; rangex[3] = y_max+2;
  ops_par_loop(initialise_chunk_kernel_cellx, "initialise_chunk_kernel_cellx", clover_grid, 2, rangex,
               ops_arg_dat(vertexx, S2D_00_P10_STRID2D_X, "double", OPS_READ),
               ops_arg_dat(cellx, S2D_00_STRID2D_X, "double", OPS_WRITE),
               ops_arg_dat(celldx, S2D_00_STRID2D_X, "double", OPS_WRITE));

  //ops_print_dat_to_txtfile_core(vertexx, "cloverdats.dat");
  //ops_print_dat_to_txtfile_core(cellx, "cloverdats.dat");

  rangey[0] = x_min-2; rangey[1] = x_max+2; rangey[2] = y_min-2; rangey[3] = y_max+2;
  ops_par_loop(initialise_chunk_kernel_celly, "initialise_chunk_kernel_celly", clover_grid, 2, rangey,
               ops_arg_dat(vertexy, S2D_00_0P1_STRID2D_Y, "double", OPS_READ),
               ops_arg_dat(celly, S2D_00_STRID2D_Y, "double", OPS_WRITE),
               ops_arg_dat(celldy, S2D_00_STRID2D_Y, "double", OPS_WRITE));



  int rangexy[] = {x_min-2,x_max+2,y_min-2,y_max+2};
  ops_par_loop(initialise_chunk_kernel_volume, "initialise_chunk_kernel_volume", clover_grid, 2, rangexy,
    ops_arg_dat(volume, S2D_00, "double", OPS_WRITE),
    ops_arg_dat(celldy, S2D_00_STRID2D_Y, "double", OPS_READ),
    ops_arg_dat(xarea, S2D_00, "double", OPS_WRITE),
    ops_arg_dat(celldx, S2D_00_STRID2D_X, "double", OPS_READ),
    ops_arg_dat(yarea, S2D_00, "double", OPS_WRITE));

  //printf("\n");
  //ops_exit();//exit for now
  //exit(0);
}
