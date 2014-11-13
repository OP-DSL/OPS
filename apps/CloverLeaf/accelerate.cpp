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

/** @brief acceleration kernels
 *  @author Wayne Gaudin, converted to OPS by Gihan Mudalige
 *  @details Calls user requested kernel
**/

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

// OPS header file

#include "ops_seq.h"

#include "data.h"
#include "definitions.h"

#include "accelerate_kernel.h"


void accelerate()
{
  error_condition = 0; // Not used yet due to issue with OpenA reduction

  //initialize sizes using global values
  int x_min = field.x_min;
  int x_max = field.x_max;
  int y_min = field.y_min;
  int y_max = field.y_max;


  int rangexy_inner_plus1[] = {x_min,x_max+1,y_min,y_max+1}; // inner range plus 1
  
  ops_par_loop(accelerate_kernel, "accelerate_kernel", clover_grid, 2, rangexy_inner_plus1,
    ops_arg_dat(density0, 1, S2D_00_M10_0M1_M1M1, "double", OPS_READ),
    ops_arg_dat(volume, 1, S2D_00_M10_0M1_M1M1, "double", OPS_READ),
    ops_arg_dat(work_array1, 1, S2D_00, "double", OPS_WRITE),
    ops_arg_dat(xvel0, 1, S2D_00, "double", OPS_READ),
    ops_arg_dat(xvel1, 1, S2D_00, "double", OPS_INC),
    ops_arg_dat(xarea, 1, S2D_00_0M1, "double", OPS_READ),
    ops_arg_dat(pressure, 1, S2D_00_M10_0M1_M1M1, "double", OPS_READ),
    ops_arg_dat(yvel0, 1, S2D_00, "double", OPS_READ),
    ops_arg_dat(yvel1, 1, S2D_00, "double", OPS_INC),
    ops_arg_dat(yarea, 1, S2D_00_M10, "double", OPS_READ),
    ops_arg_dat(viscosity, 1, S2D_00_M10_0M1_M1M1, "double", OPS_READ));
}
