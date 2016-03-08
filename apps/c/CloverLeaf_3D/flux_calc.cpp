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
#define OPS_3D
#include "ops_seq.h"

#include "data.h"
#include "definitions.h"

#include "flux_calc_kernel.h"


void flux_calc()
{
  error_condition = 0; // Not used yet due to issue with OpenA reduction

  //initialize sizes using global values
  int x_min = field.x_min;
  int x_max = field.x_max;
  int y_min = field.y_min;
  int y_max = field.y_max;
  int z_min = field.z_min;
  int z_max = field.z_max;

  int rangexyz_inner_plus1x[] = {x_min,x_max+1,y_min,y_max,z_min,z_max};

  ops_par_loop(flux_calc_kernelx, "flux_calc_kernelx", clover_grid, 3, rangexyz_inner_plus1x,
    ops_arg_dat(vol_flux_x, 1, S3D_000, "double", OPS_WRITE),
    ops_arg_dat(xarea, 1, S3D_000, "double", OPS_READ),
    ops_arg_dat(xvel0, 1, S3D_000_f0P1P1, "double", OPS_READ),
    ops_arg_dat(xvel1, 1, S3D_000_f0P1P1, "double", OPS_READ));

  int rangexyz_inner_plus1y[] = {x_min,x_max,y_min,y_max+1,z_min,z_max};

  ops_par_loop(flux_calc_kernely, "flux_calc_kernely", clover_grid, 3, rangexyz_inner_plus1y,
    ops_arg_dat(vol_flux_y, 1, S3D_000, "double", OPS_WRITE),
    ops_arg_dat(yarea, 1, S3D_000, "double", OPS_READ),
    ops_arg_dat(yvel0, 1, S3D_000_fP10P1, "double", OPS_READ),
    ops_arg_dat(yvel1, 1, S3D_000_fP10P1, "double", OPS_READ));

  int rangexyz_inner_plus1z[] = {x_min,x_max,y_min,y_max,z_min,z_max+1};

  ops_par_loop(flux_calc_kernelz, "flux_calc_kernelz", clover_grid, 3, rangexyz_inner_plus1z,
    ops_arg_dat(vol_flux_z, 1, S3D_000, "double", OPS_WRITE),
    ops_arg_dat(zarea, 1, S3D_000, "double", OPS_READ),
    ops_arg_dat(zvel0, 1, S3D_000_fP1P10, "double", OPS_READ),
    ops_arg_dat(zvel1, 1, S3D_000_fP1P10, "double", OPS_READ));
}
