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
#define OPS_2D
#include "ops_seq_v2.h"

#include "data.h"
#include "definitions.h"

//Cloverleaf kernels
#include "generate_chunk_kernel.h"


void generate()
{
  //initialize sizes using global values
  int x_min = field.x_min;
  int x_max = field.x_max;
  int y_min = field.y_min;
  int y_max = field.y_max;

  int rangexy[] = {x_min-2,x_max+2,y_min-2,y_max+2};

  ops_par_loop(generate_chunk_kernel, "generate_chunk_kernel", clover_grid, 2, rangexy,
    ops_arg_dat(vertexx, 1, S2D_00_P10_M10_STRID2D_X, "double", OPS_READ),
    ops_arg_dat(vertexy, 1, S2D_00_0P1_0M1_STRID2D_Y, "double", OPS_READ),
    ops_arg_dat(energy0, 1, S2D_00, "double", OPS_WRITE),
    ops_arg_dat(density0, 1, S2D_00, "double", OPS_WRITE),
    ops_arg_dat(xvel0, 1, S2D_00, "double", OPS_WRITE),
    ops_arg_dat(yvel0, 1, S2D_00, "double", OPS_WRITE),
    ops_arg_dat(cellx, 1, S2D_00_P10_M10_STRID2D_X, "double", OPS_READ),
    ops_arg_dat(celly, 1, S2D_00_0P1_0M1_STRID2D_Y, "double", OPS_READ));

}
