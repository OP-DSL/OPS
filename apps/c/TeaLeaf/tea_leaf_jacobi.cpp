/*Crown Copyright 2014 AWE.

 This file is part of TeaLeaf.

 TeaLeaf is free software: you can redistribute it and/or modify it under
 the terms of the GNU General Public License as published by the
 Free Software Foundation, either version 3 of the License, or (at your option)
 any later version.

 TeaLeaf is distributed in the hope that it will be useful, but
 WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
 FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more
 details.

 You should have received a copy of the GNU General Public License along with
 TeaLeaf. If not, see http://www.gnu.org/licenses/. */

// @brief Controls the main diffusion cycle.
// @author Istvan Reguly, David Beckingsale, Wayne Gaudin
// @details Implicitly calculates the change in temperature using a Jacobi iteration


#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

// OPS header file
#define OPS_2D
#include <ops_seq.h>


#include "data.h"
#include "definitions.h"

#include "tea_leaf.h"
#include "tea_leaf_jacobi_kernels.h"

void tea_leaf_jacobi_solve(
  double rx, double ry,
	ops_dat Kx,
	ops_dat Ky,
  double *error,
	ops_dat u0,
	ops_dat u1,
	ops_dat un)
{
  //initialize sizes using global values
  int x_min = field.x_min;
  int x_max = field.x_max;
  int y_min = field.y_min;
  int y_max = field.y_max;

  int rangexy[] = {x_min,x_max,y_min,y_max};

  *error = 0.0;

  ops_par_loop(tea_leaf_yeqx_kernel, "tea_leaf_yeqx_kernel", tea_grid, 2, rangexy,
      ops_arg_dat(un, 1, S2D_00, "double", OPS_WRITE),
      ops_arg_dat(u1, 1, S2D_00, "double", OPS_READ));

  ops_par_loop(tea_leaf_jacobi_kernel, "tea_leaf_jacobi_kernel", tea_grid, 2, rangexy,
      ops_arg_dat(u1, 1, S2D_00, "double", OPS_WRITE),
      ops_arg_dat(Kx, 1, S2D_00_P10, "double", OPS_READ),
      ops_arg_dat(Ky, 1, S2D_00_0P1, "double", OPS_READ),
      ops_arg_dat(un, 1, S2D_00_0M1_M10_P10_0P1, "double", OPS_READ),
      ops_arg_dat(u0, 1, S2D_00, "double", OPS_READ),
      ops_arg_gbl(&rx, 1, "double", OPS_READ),
      ops_arg_gbl(&ry, 1, "double", OPS_READ),
      ops_arg_reduce(red_temp, 1, "double", OPS_INC));

  ops_reduction_result(red_temp,error);

}
