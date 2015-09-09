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

/** @brief Mesh chunk generation driver
 *  @author Wayne Gaudin, converted to OPS by Gihan Mudalige
 *  @details Invoked the users specified chunk generator.
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
#include "generate_chunk_kernel.h"


void generate_hdf5()
{
  //initialize sizes using global values
  int x_min = field.x_min;
  int x_max = field.x_max;
  int y_min = field.y_min;
  int y_max = field.y_max;
  int z_min = field.z_min;
  int z_max = field.z_max;

  int rangexyz[] = {x_min-2,x_max+2,y_min-2,y_max+2,z_min-2,z_max+2};

  ops_par_loop(generate_chunk_kernel, "generate_chunk_kernel", clover_grid, 3, rangexyz,
    ops_arg_dat(vertexx,  1, S3D_000_P100_STRID3D_X, "double", OPS_READ),
    ops_arg_dat(vertexy,  1, S3D_000_0P10_STRID3D_Y, "double", OPS_READ),
    ops_arg_dat(vertexz,  1, S3D_000_00P1_STRID3D_Z, "double", OPS_READ),
    ops_arg_dat(energy0,  1, S3D_000, "double", OPS_WRITE),
    ops_arg_dat(density0, 1, S3D_000, "double", OPS_WRITE),
    ops_arg_dat(xvel0,    1, S3D_000_fP1P1P1, "double", OPS_WRITE),
    ops_arg_dat(yvel0,    1, S3D_000_fP1P1P1, "double", OPS_WRITE),
    ops_arg_dat(zvel0,    1, S3D_000_fP1P1P1, "double", OPS_WRITE),
    ops_arg_dat(cellx,    1, S3D_000_STRID3D_X, "double", OPS_READ),
    ops_arg_dat(celly,    1, S3D_000_STRID3D_Y, "double", OPS_READ),
    ops_arg_dat(cellz,    1, S3D_000_STRID3D_Z, "double", OPS_READ));

    ops_fetch_dat_hdf5_file(density0, "cloverdata.h5");
    ops_fetch_dat_hdf5_file(energy0, "cloverdata.h5");
    ops_fetch_dat_hdf5_file(xvel0, "cloverdata.h5");
    ops_fetch_dat_hdf5_file(yvel0, "cloverdata.h5");
    ops_fetch_dat_hdf5_file(zvel0, "cloverdata.h5");
}