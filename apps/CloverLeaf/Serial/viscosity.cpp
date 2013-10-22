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

/** @brief call the viscosity kernels
 *  @author Wayne Gaudin
 *  @details Calculates an artificial viscosity using the Wilkin's method to
 *  smooth out shock front and prevent oscillations around discontinuities.
 *  Only cells in compression will have a non-zero value.
**/

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

// OPS header file
#include "ops_seq_opt.h"

#include "data.h"
#include "definitions.h"

#include "viscosity_kernel.h"

void viscosity_func()
{
  //initialize sizes using global values
  int x_cells = grid->x_cells;
  int y_cells = grid->y_cells;
  int x_min = field->x_min;
  int x_max = field->x_max;
  int y_min = field->y_min;
  int y_max = field->y_max;

  int rangexy_inner[] = {x_min,x_max,y_min,y_max}; // inner range without border

  ops_par_loop_opt(viscosity_kernel, "viscosity_kernel", 2, rangexy_inner,
      ops_arg_dat(xvel0, S2D_00_P10_0P1_P1P1, "double", OPS_READ),
      ops_arg_dat(yvel0, S2D_00_P10_0P1_P1P1, "double", OPS_READ),
      ops_arg_dat(celldx, s2D_00_P10_STRID2D_X, "double", OPS_READ),
      ops_arg_dat(celldy, S2D_00_0P1_STRID2D_Y, "double", OPS_READ),
      ops_arg_dat(pressure, S2D_10_M10_01_0M1, "double", OPS_READ),
      ops_arg_dat(density0, S2D_00, "double", OPS_READ),
      ops_arg_dat(viscosity, S2D_00, "double", OPS_WRITE));
}
