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

/** @brief Driver for the flux kernels
 *  @author Wayne Gaudin, converted to OPS by Gihan Mudalige
 *  @details Invokes the used specified flux kernel
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

#include "flux_calc_kernel.h"


void flux_calc_kernelx_macro( double *vol_flux_x, double *xarea,
                        double *xvel0, double *xvel1) {

  //{0,0, 0,1};
  vol_flux_x[OPS_ACC0(0,0)] = 0.25 * dt * (xarea[OPS_ACC1(0,0)]) *
  ( (xvel0[OPS_ACC2(0,0)]) + (xvel0[OPS_ACC2(0,1)]) + (xvel1[OPS_ACC3(0,0)]) + (xvel1[OPS_ACC3(0,1)]) );

}

void flux_calc_kernely_macro( double *vol_flux_y, double *yarea,
                        double *yvel0, double *yvel1) {

    //{0,0, 1,0};
  vol_flux_y[OPS_ACC0(0,0)] = 0.25 * dt * (yarea[OPS_ACC1(0,0)]) *
  ( (yvel0[OPS_ACC2(0,0)]) + (yvel0[OPS_ACC2(1,0)]) + (yvel1[OPS_ACC3(0,0)]) + (yvel1[OPS_ACC3(1,0)]) );

}

void flux_calc()
{
  error_condition = 0; // Not used yet due to issue with OpenA reduction

  //initialize sizes using global values
  int x_cells = grid->x_cells;
  int y_cells = grid->y_cells;
  int x_min = field->x_min;
  int x_max = field->x_max;
  int y_min = field->y_min;
  int y_max = field->y_max;

  int rangexy_inner_plus1x[] = {x_min,x_max+1,y_min,y_max};

  ops_par_loop_macro(flux_calc_kernelx_macro, "flux_calc_kernelx_macro", 2, rangexy_inner_plus1x,
    ops_arg_dat(vol_flux_x, S2D_00, "double", OPS_WRITE),
    ops_arg_dat(xarea, S2D_00, "double", OPS_READ),
    ops_arg_dat(xvel0, S2D_00_0P1, "double", OPS_READ),
    ops_arg_dat(xvel1, S2D_00_0P1, "double", OPS_READ));

  int rangexy_inner_plus1y[] = {x_min,x_max,y_min,y_max+1};

  ops_par_loop_macro(flux_calc_kernely_macro, "flux_calc_kernely_macro", 2, rangexy_inner_plus1y,
    ops_arg_dat(vol_flux_y, S2D_00, "double", OPS_WRITE),
    ops_arg_dat(yarea, S2D_00, "double", OPS_READ),
    ops_arg_dat(yvel0, S2D_00_P10, "double", OPS_READ),
    ops_arg_dat(yvel1, S2D_00_P10, "double", OPS_READ));

}
