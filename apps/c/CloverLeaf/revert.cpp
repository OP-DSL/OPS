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

/** @brief revert kernel.
 *  @author Wayne Gaudin, converted to OPS by Gihan Mudalige
 *  @details Takes the half step field data used in the predictor and reverts
 *  it to the start of step data, ready for the corrector.
 *  Note that this does not seem necessary in this proxy-app but should be
 *  left in to remain relevant to the full method.
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

#include "revert_kernel.h"

void revert()
{
  error_condition = 0; // Not used yet due to issue with OpenA reduction

  //initialize sizes using global values
  int x_min = field.x_min;
  int x_max = field.x_max;
  int y_min = field.y_min;
  int y_max = field.y_max;

  int rangexy_inner[] = {x_min,x_max,y_min,y_max}; // inner range without border

  ops_par_loop(revert_kernel, "revert_kernel", clover_grid, 2, rangexy_inner,
    ops_arg_dat(density0, 1, S2D_00, "double", OPS_READ),
    ops_arg_dat(density1, 1, S2D_00, "double", OPS_WRITE),
    ops_arg_dat(energy0, 1, S2D_00, "double", OPS_READ),
    ops_arg_dat(energy1, 1, S2D_00, "double", OPS_WRITE));
}
