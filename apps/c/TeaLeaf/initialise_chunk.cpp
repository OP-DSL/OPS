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
#define OPS_2D
#include <ops_seq.h>

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

  int range[] = {x_min, x_max, y_min, y_max};
  ops_par_loop(initialise_chunk_kernel_zero, "initialise_chunk_kernel_zero", tea_grid, 2, range,
               ops_arg_dat(density      , 1, S2D_00, "double", OPS_WRITE));
  ops_par_loop(initialise_chunk_kernel_zero, "initialise_chunk_kernel_zero", tea_grid, 2, range,
               ops_arg_dat(energy0      , 1, S2D_00, "double", OPS_WRITE));
  ops_par_loop(initialise_chunk_kernel_zero, "initialise_chunk_kernel_zero", tea_grid, 2, range,
               ops_arg_dat(energy1      , 1, S2D_00, "double", OPS_WRITE));
  ops_par_loop(initialise_chunk_kernel_zero, "initialise_chunk_kernel_zero", tea_grid, 2, range,
               ops_arg_dat(u            , 1, S2D_00, "double", OPS_WRITE));
  ops_par_loop(initialise_chunk_kernel_zero, "initialise_chunk_kernel_zero", tea_grid, 2, range,
               ops_arg_dat(u0           , 1, S2D_00, "double", OPS_WRITE));
  ops_par_loop(initialise_chunk_kernel_zero, "initialise_chunk_kernel_zero", tea_grid, 2, range,
               ops_arg_dat(vector_r     , 1, S2D_00, "double", OPS_WRITE));
  ops_par_loop(initialise_chunk_kernel_zero, "initialise_chunk_kernel_zero", tea_grid, 2, range,
               ops_arg_dat(vector_rstore, 1, S2D_00, "double", OPS_WRITE));
  ops_par_loop(initialise_chunk_kernel_zero, "initialise_chunk_kernel_zero", tea_grid, 2, range,
               ops_arg_dat(vector_rtemp , 1, S2D_00, "double", OPS_WRITE));
  ops_par_loop(initialise_chunk_kernel_zero, "initialise_chunk_kernel_zero", tea_grid, 2, range,
               ops_arg_dat(vector_Mi    , 1, S2D_00, "double", OPS_WRITE));
  ops_par_loop(initialise_chunk_kernel_zero, "initialise_chunk_kernel_zero", tea_grid, 2, range,
               ops_arg_dat(vector_w     , 1, S2D_00, "double", OPS_WRITE));
  ops_par_loop(initialise_chunk_kernel_zero, "initialise_chunk_kernel_zero", tea_grid, 2, range,
               ops_arg_dat(vector_z     , 1, S2D_00, "double", OPS_WRITE));
  ops_par_loop(initialise_chunk_kernel_zero, "initialise_chunk_kernel_zero", tea_grid, 2, range,
               ops_arg_dat(vector_utemp , 1, S2D_00, "double", OPS_WRITE));
  ops_par_loop(initialise_chunk_kernel_zero, "initialise_chunk_kernel_zero", tea_grid, 2, range,
               ops_arg_dat(vector_Kx    , 1, S2D_00, "double", OPS_WRITE));
  ops_par_loop(initialise_chunk_kernel_zero, "initialise_chunk_kernel_zero", tea_grid, 2, range,
               ops_arg_dat(vector_Ky    , 1, S2D_00, "double", OPS_WRITE));
  ops_par_loop(initialise_chunk_kernel_zero, "initialise_chunk_kernel_zero", tea_grid, 2, range,
               ops_arg_dat(vector_p     , 1, S2D_00, "double", OPS_WRITE));
  ops_par_loop(initialise_chunk_kernel_zero, "initialise_chunk_kernel_zero", tea_grid, 2, range,
               ops_arg_dat(vector_sd    , 1, S2D_00, "double", OPS_WRITE));
  ops_par_loop(initialise_chunk_kernel_zero, "initialise_chunk_kernel_zero", tea_grid, 2, range,
               ops_arg_dat(tri_cp       , 1, S2D_00, "double", OPS_WRITE));
  ops_par_loop(initialise_chunk_kernel_zero, "initialise_chunk_kernel_zero", tea_grid, 2, range,
               ops_arg_dat(tri_bfp      , 1, S2D_00, "double", OPS_WRITE));

  int rangefull1[] = {-2, x_cells+2, -2, y_cells+2};
  ops_par_loop(initialise_chunk_kernel_zero, "initialise_chunk_kernel_zero", tea_grid, 2, rangefull1,
               ops_arg_dat(volume    , 1, S2D_00, "double", OPS_WRITE));
  int rangefull2[] = {-2, x_cells+3, -2, y_cells+2};
  ops_par_loop(initialise_chunk_kernel_zero, "initialise_chunk_kernel_zero", tea_grid, 2, rangefull2,
               ops_arg_dat(xarea       , 1, S2D_00, "double", OPS_WRITE));
  int rangefull3[] = {-2, x_cells+2, -2, y_cells+3};
  ops_par_loop(initialise_chunk_kernel_zero, "initialise_chunk_kernel_zero", tea_grid, 2, rangefull3,
               ops_arg_dat(yarea      , 1, S2D_00, "double", OPS_WRITE));

  

//edge datasets
  int rangex[] = {x_min-2, x_max+2, y_min-2, y_max+2};
  ops_par_loop(initialise_chunk_kernel_zero_x, "initialise_chunk_kernel_zero_x", tea_grid, 2, rangex,
               ops_arg_dat(cellx       , 1, S2D_00_STRID2D_X, "double", OPS_WRITE));
  ops_par_loop(initialise_chunk_kernel_zero_x, "initialise_chunk_kernel_zero_x", tea_grid, 2, rangex,
               ops_arg_dat(celldx      , 1, S2D_00_STRID2D_X, "double", OPS_WRITE));
  rangex[1]++;
  ops_par_loop(initialise_chunk_kernel_zero_x, "initialise_chunk_kernel_zero_x", tea_grid, 2, rangex,
               ops_arg_dat(vertexx     , 1, S2D_00_STRID2D_X, "double", OPS_WRITE));
  ops_par_loop(initialise_chunk_kernel_zero_x, "initialise_chunk_kernel_zero_x", tea_grid, 2, rangex,
               ops_arg_dat(vertexdx    , 1, S2D_00_STRID2D_X, "double", OPS_WRITE));  

  int rangey2[] = {x_min-2, x_max+2, y_min-2, y_max+2};
  ops_par_loop(initialise_chunk_kernel_zero_y, "initialise_chunk_kernel_zero_y", tea_grid, 2, rangey2,
               ops_arg_dat(celly       , 1, S2D_00_STRID2D_Y, "double", OPS_WRITE));
  ops_par_loop(initialise_chunk_kernel_zero_y, "initialise_chunk_kernel_zero_y", tea_grid, 2, rangey2,
               ops_arg_dat(celldy      , 1, S2D_00_STRID2D_Y, "double", OPS_WRITE));
  rangey2[3]++;
  ops_par_loop(initialise_chunk_kernel_zero_y, "initialise_chunk_kernel_zero_y", tea_grid, 2, rangey2,
               ops_arg_dat(vertexy     , 1, S2D_00_STRID2D_Y, "double", OPS_WRITE));
  ops_par_loop(initialise_chunk_kernel_zero_y, "initialise_chunk_kernel_zero_y", tea_grid, 2, rangey2,
               ops_arg_dat(vertexdy    , 1, S2D_00_STRID2D_Y, "double", OPS_WRITE));  




  int rangefull[] = {x_min-2, x_max+3, y_min-2, y_max+3};
  ops_execute(vertexy->block->instance); //The following 4 loops slightly break the 
                 //abstraction to initialise edge datasets
                 //tiling dependency analysis does not like that

  ops_par_loop(initialise_chunk_kernel_xx, "initialise_chunk_kernel_xx", tea_grid, 2, rangefull,
               ops_arg_dat(xx, 1, S2D_00_STRID2D_X, "int", OPS_WRITE),
               ops_arg_idx());

  ops_par_loop(initialise_chunk_kernel_yy, "initialise_chunk_kernel_yy", tea_grid, 2, rangefull,
               ops_arg_dat(yy, 1, S2D_00_STRID2D_Y, "int", OPS_WRITE),
               ops_arg_idx());

  ops_par_loop(initialise_chunk_kernel_x, "initialise_chunk_kernel_x", tea_grid, 2, rangex,
               ops_arg_dat(vertexx, 1, S2D_00_STRID2D_X, "double", OPS_WRITE),
               ops_arg_dat(xx, 1, S2D_00_STRID2D_X, "int", OPS_READ),
               ops_arg_dat(vertexdx, 1, S2D_00_STRID2D_X, "double", OPS_WRITE));

  ops_par_loop(initialise_chunk_kernel_y, "initialise_chunk_kernel_y", tea_grid, 2, rangey2,
               ops_arg_dat(vertexy, 1, S2D_00_STRID2D_Y, "double", OPS_WRITE),
               ops_arg_dat(yy, 1, S2D_00_STRID2D_Y, "int", OPS_READ),
               ops_arg_dat(vertexdy, 1, S2D_00_STRID2D_Y, "double", OPS_WRITE));

  rangex[0] = x_min-2; rangex[1] = x_max+2; rangex[2] = y_min-2; rangex[3] = y_max+2;
  ops_par_loop(initialise_chunk_kernel_cellx, "initialise_chunk_kernel_cellx", tea_grid, 2, rangex,
               ops_arg_dat(vertexx, 1, S2D_00_P10_STRID2D_X, "double", OPS_READ),
               ops_arg_dat(cellx, 1, S2D_00_STRID2D_X, "double", OPS_WRITE),
               ops_arg_dat(celldx, 1, S2D_00_STRID2D_X, "double", OPS_WRITE));

  int rangey[] = {x_min-2, x_max+3, y_min-2, y_max+2};
  ops_par_loop(initialise_chunk_kernel_celly, "initialise_chunk_kernel_celly", tea_grid, 2, rangey,
               ops_arg_dat(vertexy, 1, S2D_00_0P1_STRID2D_Y, "double", OPS_READ),
               ops_arg_dat(celly, 1, S2D_00_STRID2D_Y, "double", OPS_WRITE),
               ops_arg_dat(celldy, 1, S2D_00_STRID2D_Y, "double", OPS_WRITE));



  int rangexy[] = {x_min-2,x_max+2,y_min-2,y_max+2};
  ops_par_loop(initialise_chunk_kernel_volume, "initialise_chunk_kernel_volume", tea_grid, 2, rangexy,
    ops_arg_dat(volume, 1, S2D_00, "double", OPS_WRITE),
    ops_arg_dat(celldy, 1, S2D_00_STRID2D_Y, "double", OPS_READ),
    ops_arg_dat(xarea, 1, S2D_00, "double", OPS_WRITE),
    ops_arg_dat(celldx, 1, S2D_00_STRID2D_X, "double", OPS_READ),
    ops_arg_dat(yarea, 1, S2D_00, "double", OPS_WRITE));

  //printf("\n");
  //ops_exit();//exit for now
  //exit(0);
}
