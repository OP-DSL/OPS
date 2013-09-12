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

/** @brief Top level initialisation routine
 *  @author Wayne Gaudin
 *  @details Checks for the user input and either invokes the input reader or
 *  switches to the internal test problem. It processes the input and strips
 *  comments before writing a final input file.
 *  It then calls the start routine.
**/

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

// OPS header file
#include "ops_seq.h"

#include "data.h"
#include "definitions.h"

//Cloverleaf kernels
#include  "update_halo_kernel.h"

void update_halo(int* fields, int depth)
{
  //initialize sizes using global values
  int x_cells = grid->x_cells;
  int y_cells = grid->y_cells;
  int x_min = field->x_min;
  int x_max = field->x_max;
  int y_min = field->y_min;
  int y_max = field->y_max;


  if(depth == 2) {
  int rangexy_bottom2[] = {x_min-2,x_max+2,y_min-2,y_min-1};
  ops_par_loop(update_halo_kernel, "update_halo_kernel", 2, rangexy_bottom2,
      ops_arg_dat(density0, sten_self2D_plus2y, OPS_RW),
      ops_arg_dat(density1, sten_self2D_plus2y, OPS_RW),
      ops_arg_dat(energy0, sten_self2D_plus2y, OPS_RW),
      ops_arg_dat(energy1, sten_self2D_plus2y, OPS_RW),
      ops_arg_dat(pressure, sten_self2D_plus2y, OPS_RW),
      ops_arg_dat(viscosity, sten_self2D_plus2y, OPS_RW),
      ops_arg_dat(soundspeed, sten_self2D_plus2y, OPS_RW),
      ops_arg_dat(xvel0, sten_self2D_plus3y, OPS_RW),
      ops_arg_dat(xvel1, sten_self2D_plus3y, OPS_RW),
      ops_arg_dat(yvel0, sten_self2D_plus3y, OPS_RW),
      ops_arg_dat(yvel1, sten_self2D_plus3y, OPS_RW),
      ops_arg_dat(vol_flux_x, sten_self2D_plus3y, OPS_RW),
      ops_arg_dat(mass_flux_x, sten_self2D_plus3y, OPS_RW),
      ops_arg_dat(vol_flux_y, sten_self2D_plus3y, OPS_RW),
      ops_arg_dat(mass_flux_y, sten_self2D_plus3y, OPS_RW));

  int rangexy_top2[] = {x_min-2,x_max+2,y_max+1,y_max+2};
  ops_par_loop(update_halo_kernel, "update_halo_kernel", 2, rangexy_top2,
      ops_arg_dat(density0, sten_self2D_minus2y, OPS_RW),
      ops_arg_dat(density1, sten_self2D_minus2y, OPS_RW),
      ops_arg_dat(energy0, sten_self2D_minus2y, OPS_RW),
      ops_arg_dat(energy1, sten_self2D_minus2y, OPS_RW),
      ops_arg_dat(pressure, sten_self2D_minus2y, OPS_RW),
      ops_arg_dat(viscosity, sten_self2D_minus2y, OPS_RW),
      ops_arg_dat(soundspeed, sten_self2D_minus2y, OPS_RW),
      ops_arg_dat(xvel0, sten_self2D_minus3y, OPS_RW),
      ops_arg_dat(xvel1, sten_self2D_minus3y, OPS_RW),
      ops_arg_dat(yvel0, sten_self2D_minus3y, OPS_RW),
      ops_arg_dat(yvel1, sten_self2D_minus3y, OPS_RW),
      ops_arg_dat(vol_flux_x,  sten_self2D_minus3y, OPS_RW),
      ops_arg_dat(mass_flux_x, sten_self2D_minus3y, OPS_RW),
      ops_arg_dat(vol_flux_y,  sten_self2D_minus3y, OPS_RW),
      ops_arg_dat(mass_flux_y, sten_self2D_minus3y, OPS_RW));

  int rangexy_left2[] = {x_min-2,x_min-1,y_min-2,y_max+2};
  ops_par_loop(update_halo_kernel, "update_halo_kernel", 2, rangexy_left2,
      ops_arg_dat(density0, sten_self2D_plus2x, OPS_RW),
      ops_arg_dat(density1, sten_self2D_plus2x, OPS_RW),
      ops_arg_dat(energy0, sten_self2D_plus2x, OPS_RW),
      ops_arg_dat(energy1, sten_self2D_plus2x, OPS_RW),
      ops_arg_dat(pressure, sten_self2D_plus2x, OPS_RW),
      ops_arg_dat(viscosity, sten_self2D_plus2x, OPS_RW),
      ops_arg_dat(soundspeed, sten_self2D_plus2x, OPS_RW),
      ops_arg_dat(xvel0, sten_self2D_plus3x, OPS_RW),
      ops_arg_dat(xvel1, sten_self2D_plus3x, OPS_RW),
      ops_arg_dat(yvel0, sten_self2D_plus3x, OPS_RW),
      ops_arg_dat(yvel1, sten_self2D_plus3x, OPS_RW),
      ops_arg_dat(vol_flux_x, sten_self2D_plus3x, OPS_RW),
      ops_arg_dat(mass_flux_x, sten_self2D_plus3x, OPS_RW),
      ops_arg_dat(vol_flux_y, sten_self2D_plus3x, OPS_RW),
      ops_arg_dat(mass_flux_y, sten_self2D_plus3x, OPS_RW));

  int rangexy_right2[] = {x_max+1,x_max+2,y_min-2,y_max+2};
  ops_par_loop(update_halo_kernel, "update_halo_kernel", 2, rangexy_right2,
      ops_arg_dat(density0, sten_self2D_minus2x, OPS_RW),
      ops_arg_dat(density1, sten_self2D_minus2x, OPS_RW),
      ops_arg_dat(energy0, sten_self2D_minus2x, OPS_RW),
      ops_arg_dat(energy1, sten_self2D_minus2x, OPS_RW),
      ops_arg_dat(pressure, sten_self2D_minus2x, OPS_RW),
      ops_arg_dat(viscosity, sten_self2D_minus2x, OPS_RW),
      ops_arg_dat(soundspeed, sten_self2D_minus2x, OPS_RW),
      ops_arg_dat(xvel0, sten_self2D_minus3x, OPS_RW),
      ops_arg_dat(xvel1, sten_self2D_minus3x, OPS_RW),
      ops_arg_dat(yvel0, sten_self2D_minus3x, OPS_RW),
      ops_arg_dat(yvel1, sten_self2D_minus3x, OPS_RW),
      ops_arg_dat(vol_flux_x,  sten_self2D_minus3x, OPS_RW),
      ops_arg_dat(mass_flux_x, sten_self2D_minus3x, OPS_RW),
      ops_arg_dat(vol_flux_y,  sten_self2D_minus3x, OPS_RW),
      ops_arg_dat(mass_flux_y, sten_self2D_minus3x, OPS_RW));
  }


  int rangexy_bottom1[] = {x_min-2,x_max+2,y_min-1,y_min};
  ops_par_loop(update_halo_kernel, "update_halo_kernel", 2, rangexy_bottom1,
      ops_arg_dat(density0, sten_self2D_plus1y, OPS_RW),
      ops_arg_dat(density1, sten_self2D_plus1y, OPS_RW),
      ops_arg_dat(energy0, sten_self2D_plus1y, OPS_RW),
      ops_arg_dat(energy1, sten_self2D_plus1y, OPS_RW),
      ops_arg_dat(pressure, sten_self2D_plus1y, OPS_RW),
      ops_arg_dat(viscosity, sten_self2D_plus1y, OPS_RW),
      ops_arg_dat(soundspeed, sten_self2D_plus1y, OPS_RW),
      ops_arg_dat(xvel0, sten_self2D_plus2y, OPS_RW),
      ops_arg_dat(xvel1, sten_self2D_plus2y, OPS_RW),
      ops_arg_dat(yvel0, sten_self2D_plus2y, OPS_RW),
      ops_arg_dat(yvel1, sten_self2D_plus2y, OPS_RW),
      ops_arg_dat(vol_flux_x,  sten_self2D_plus2y, OPS_RW),
      ops_arg_dat(mass_flux_x, sten_self2D_plus2y, OPS_RW),
      ops_arg_dat(vol_flux_y,  sten_self2D_plus2y, OPS_RW),
      ops_arg_dat(mass_flux_y, sten_self2D_plus2y, OPS_RW));

  int rangexy_top1[] = {x_min-2,x_max+2,y_max,y_max+1};
  ops_par_loop(update_halo_kernel, "update_halo_kernel", 2, rangexy_top1,
      ops_arg_dat(density0, sten_self2D_minus1y, OPS_RW),
      ops_arg_dat(density1, sten_self2D_minus1y, OPS_RW),
      ops_arg_dat(energy0, sten_self2D_minus1y, OPS_RW),
      ops_arg_dat(energy1, sten_self2D_minus1y, OPS_RW),
      ops_arg_dat(pressure, sten_self2D_minus1y, OPS_RW),
      ops_arg_dat(viscosity, sten_self2D_minus1y, OPS_RW),
      ops_arg_dat(soundspeed, sten_self2D_minus1y, OPS_RW),
      ops_arg_dat(xvel0, sten_self2D_minus2y, OPS_RW),
      ops_arg_dat(xvel1, sten_self2D_minus2y, OPS_RW),
      ops_arg_dat(yvel0, sten_self2D_minus2y, OPS_RW),
      ops_arg_dat(yvel1, sten_self2D_minus2y, OPS_RW),
      ops_arg_dat(vol_flux_x, sten_self2D_minus2y, OPS_RW),
      ops_arg_dat(mass_flux_x, sten_self2D_minus2y, OPS_RW),
      ops_arg_dat(vol_flux_y, sten_self2D_minus2y, OPS_RW),
      ops_arg_dat(mass_flux_y, sten_self2D_minus2y, OPS_RW));

  int rangexy_left1[] = {x_min-1,x_min,y_min-2,y_max+2};
  ops_par_loop(update_halo_kernel, "update_halo_kernel", 2, rangexy_left1,
      ops_arg_dat(density0, sten_self2D_plus1x, OPS_RW),
      ops_arg_dat(density1, sten_self2D_plus1x, OPS_RW),
      ops_arg_dat(energy0, sten_self2D_plus1x, OPS_RW),
      ops_arg_dat(energy1, sten_self2D_plus1x, OPS_RW),
      ops_arg_dat(pressure, sten_self2D_plus1x, OPS_RW),
      ops_arg_dat(viscosity, sten_self2D_plus1x, OPS_RW),
      ops_arg_dat(soundspeed, sten_self2D_plus1x, OPS_RW),
      ops_arg_dat(xvel0, sten_self2D_plus2x, OPS_RW),
      ops_arg_dat(xvel1, sten_self2D_plus2x, OPS_RW),
      ops_arg_dat(yvel0, sten_self2D_plus2x, OPS_RW),
      ops_arg_dat(yvel1, sten_self2D_plus2x, OPS_RW),
      ops_arg_dat(vol_flux_x, sten_self2D_plus2x, OPS_RW),
      ops_arg_dat(mass_flux_x, sten_self2D_plus2x, OPS_RW),
      ops_arg_dat(vol_flux_y, sten_self2D_plus2x, OPS_RW),
      ops_arg_dat(mass_flux_y, sten_self2D_plus2x, OPS_RW));

  int rangexy_right1[] = {x_max,x_max+1,y_min-2,y_max+2};
  ops_par_loop(update_halo_kernel, "update_halo_kernel", 2, rangexy_right1,
      ops_arg_dat(density0, sten_self2D_minus1x, OPS_RW),
      ops_arg_dat(density1, sten_self2D_minus1x, OPS_RW),
      ops_arg_dat(energy0, sten_self2D_minus1x, OPS_RW),
      ops_arg_dat(energy1, sten_self2D_minus1x, OPS_RW),
      ops_arg_dat(pressure, sten_self2D_minus1x, OPS_RW),
      ops_arg_dat(viscosity, sten_self2D_minus1x, OPS_RW),
      ops_arg_dat(soundspeed, sten_self2D_minus1x, OPS_RW),
      ops_arg_dat(xvel0, sten_self2D_minus2x, OPS_RW),
      ops_arg_dat(xvel1, sten_self2D_minus2x, OPS_RW),
      ops_arg_dat(yvel0, sten_self2D_minus2x, OPS_RW),
      ops_arg_dat(yvel1, sten_self2D_minus2x, OPS_RW),
      ops_arg_dat(vol_flux_x, sten_self2D_minus2x, OPS_RW),
      ops_arg_dat(mass_flux_x, sten_self2D_minus2x, OPS_RW),
      ops_arg_dat(vol_flux_y,  sten_self2D_minus2x, OPS_RW),
      ops_arg_dat(mass_flux_y, sten_self2D_minus2x, OPS_RW));
}
