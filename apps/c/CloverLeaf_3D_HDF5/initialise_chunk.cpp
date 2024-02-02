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
#define OPS_3D
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
  int z_cells = grid.z_cells;
  int x_min = field.x_min;
  int x_max = field.x_max;
  int y_min = field.y_min;
  int y_max = field.y_max;
  int z_min = field.z_min;
  int z_max = field.z_max;

//TODO: this should be fixed:
  int rangex[] = {x_min-2, x_max+3, y_min-2, y_max+3, z_min-2, z_max+3};
  int rangey[] = {x_min-2, x_max+3, y_min-2, y_max+3, z_min-2, z_max+3};
  int rangez[] = {x_min-2, x_max+3, y_min-2, y_max+3, z_min-2, z_max+3};

  int rangefull[] = {-2, x_cells+8, -2, y_cells+8, -2, z_cells+8};

  ops_par_loop(initialise_chunk_kernel_xx, "initialise_chunk_kernel_xx", clover_grid, 3, rangefull,
               ops_arg_dat(xx, 1, S3D_000_STRID3D_X, "int", OPS_WRITE),
               ops_arg_idx());

  ops_par_loop(initialise_chunk_kernel_yy, "initialise_chunk_kernel_yy", clover_grid, 3, rangefull,
               ops_arg_dat(yy, 1, S3D_000_STRID3D_Y, "int", OPS_WRITE),
               ops_arg_idx());

  ops_par_loop(initialise_chunk_kernel_zz, "initialise_chunk_kernel_zz", clover_grid, 3, rangefull,
               ops_arg_dat(zz, 1, S3D_000_STRID3D_Z, "int", OPS_WRITE),
               ops_arg_idx());

  ops_par_loop(initialise_chunk_kernel_x, "initialise_chunk_kernel_x", clover_grid, 3, rangex,
               ops_arg_dat(vertexx, 1, S3D_000_STRID3D_X, "double", OPS_WRITE),
               ops_arg_dat(xx, 1, S3D_000_STRID3D_X, "int", OPS_READ),
               ops_arg_dat(vertexdx, 1, S3D_000_STRID3D_X, "double", OPS_WRITE));

  ops_par_loop(initialise_chunk_kernel_y, "initialise_chunk_kernel_y", clover_grid, 3, rangey,
               ops_arg_dat(vertexy, 1, S3D_000_STRID3D_Y, "double", OPS_WRITE),
               ops_arg_dat(yy, 1, S3D_000_STRID3D_Y, "int", OPS_READ),
               ops_arg_dat(vertexdy, 1, S3D_000_STRID3D_Y, "double", OPS_WRITE));

  ops_par_loop(initialise_chunk_kernel_z, "initialise_chunk_kernel_z", clover_grid, 3, rangez,
               ops_arg_dat(vertexz, 1, S3D_000_STRID3D_Z, "double", OPS_WRITE),
               ops_arg_dat(zz, 1, S3D_000_STRID3D_Z, "int", OPS_READ),
               ops_arg_dat(vertexdz, 1, S3D_000_STRID3D_Z, "double", OPS_WRITE));


  rangex[0] = x_min-2; rangex[1] = x_max+2;
  ops_par_loop(initialise_chunk_kernel_cellx, "initialise_chunk_kernel_cellx", clover_grid, 3, rangex,
               ops_arg_dat(vertexx, 1, S3D_000_P100_STRID3D_X, "double", OPS_READ),
               ops_arg_dat(cellx, 1, S3D_000_STRID3D_X, "double", OPS_WRITE),
               ops_arg_dat(celldx, 1, S3D_000_STRID3D_X, "double", OPS_WRITE));

  rangey[2] = y_min-2; rangey[3] = y_max+2;
  ops_par_loop(initialise_chunk_kernel_celly, "initialise_chunk_kernel_celly", clover_grid, 3, rangey,
               ops_arg_dat(vertexy, 1, S3D_000_0P10_STRID3D_Y, "double", OPS_READ),
               ops_arg_dat(celly, 1, S3D_000_STRID3D_Y, "double", OPS_WRITE),
               ops_arg_dat(celldy, 1, S3D_000_STRID3D_Y, "double", OPS_WRITE));

  rangez[4] = z_min-2; rangez[5] = z_max+2;
  ops_par_loop(initialise_chunk_kernel_cellz, "initialise_chunk_kernel_cellz", clover_grid, 3, rangez,
               ops_arg_dat(vertexz, 1, S3D_000_00P1_STRID3D_Z, "double", OPS_READ),
               ops_arg_dat(cellz, 1, S3D_000_STRID3D_Z, "double", OPS_WRITE),
               ops_arg_dat(celldz, 1, S3D_000_STRID3D_Z, "double", OPS_WRITE));

  int rangexyz[] = {x_min-2,x_max+2,y_min-2,y_max+2,z_min-2,z_max+2};
  ops_par_loop(initialise_chunk_kernel_volume, "initialise_chunk_kernel_volume", clover_grid, 3, rangexyz,
               ops_arg_dat(volume, 1, S3D_000, "double", OPS_WRITE),
               ops_arg_dat(celldy, 1, S3D_000_STRID3D_Y, "double", OPS_READ),
               ops_arg_dat(xarea, 1, S3D_000, "double", OPS_WRITE),
               ops_arg_dat(celldx, 1, S3D_000_STRID3D_X, "double", OPS_READ),
               ops_arg_dat(yarea, 1, S3D_000, "double", OPS_WRITE),
               ops_arg_dat(celldz, 1, S3D_000_STRID3D_Z, "double", OPS_READ),
               ops_arg_dat(zarea, 1, S3D_000, "double", OPS_WRITE));
}
