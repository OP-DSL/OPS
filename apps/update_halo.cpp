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

/** @brief Update the external halo cells in a chunk.
 *  @author Wayne Gaudin, converted to OPS by Gihan Mudalige
 *  @details Updates halo cells for the required fields at the required depth
 *  for any halo cells that lie on an external boundary. The location and type
 *  of data governs how this is carried out. External boundaries are always
 *  reflective.
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

  //
  //density0, energy0, density1, energy1, pressure, viscosity and soundspeed
  // all has the same boundary ranges
  //

  int rangexy_b2a[] = {x_min-depth,x_max+depth,y_min-2,y_min-1};
  if(depth ==2)
  ops_par_loop(update_halo_kernel1, "update_halo_kernel1", 2, rangexy_b2a,
              ops_arg_dat(density0, sten_self2D_plus3y, OPS_RW),
              ops_arg_dat(density1, sten_self2D_plus3y, OPS_RW),
              ops_arg_dat(energy0, sten_self2D_plus3y, OPS_RW),
              ops_arg_dat(energy1, sten_self2D_plus3y, OPS_RW),
              ops_arg_dat(pressure, sten_self2D_plus3y, OPS_RW),
              ops_arg_dat(viscosity, sten_self2D_plus3y, OPS_RW),
              ops_arg_dat(soundspeed, sten_self2D_plus3y, OPS_RW));

  int rangexy_b1a[] = {x_min-depth,x_max+depth,y_min-1,y_min};
  ops_par_loop(update_halo_kernel1, "update_halo_kernel1", 2, rangexy_b1a,
              ops_arg_dat(density0, sten_self2D_plus1y, OPS_RW),
              ops_arg_dat(density1, sten_self2D_plus1y, OPS_RW),
              ops_arg_dat(energy0, sten_self2D_plus1y, OPS_RW),
              ops_arg_dat(energy1, sten_self2D_plus1y, OPS_RW),
              ops_arg_dat(pressure, sten_self2D_plus1y, OPS_RW),
              ops_arg_dat(viscosity, sten_self2D_plus1y, OPS_RW),
              ops_arg_dat(soundspeed, sten_self2D_plus1y, OPS_RW));

  int rangexy_t2a[] = {x_min-depth,x_max+depth,y_max+1,y_max+2};
  if(depth ==2)
  ops_par_loop(update_halo_kernel1, "update_halo_kernel1", 2, rangexy_t2a,
              ops_arg_dat(density0, sten_self2D_minus3y, OPS_RW),
              ops_arg_dat(density1, sten_self2D_minus3y, OPS_RW),
              ops_arg_dat(energy0, sten_self2D_minus3y, OPS_RW),
              ops_arg_dat(energy1, sten_self2D_minus3y, OPS_RW),
              ops_arg_dat(pressure, sten_self2D_minus3y, OPS_RW),
              ops_arg_dat(viscosity, sten_self2D_minus3y, OPS_RW),
              ops_arg_dat(soundspeed, sten_self2D_minus3y, OPS_RW));

  int rangexy_t1a[] = {x_min-depth,x_max+depth,y_max,y_max+1};
  ops_par_loop(update_halo_kernel1, "update_halo_kernel1", 2, rangexy_t1a,
              ops_arg_dat(density0, sten_self2D_minus1y, OPS_RW),
              ops_arg_dat(density1, sten_self2D_minus1y, OPS_RW),
              ops_arg_dat(energy0, sten_self2D_minus1y, OPS_RW),
              ops_arg_dat(energy1, sten_self2D_minus1y, OPS_RW),
              ops_arg_dat(pressure, sten_self2D_minus1y, OPS_RW),
              ops_arg_dat(viscosity, sten_self2D_minus1y, OPS_RW),
              ops_arg_dat(soundspeed, sten_self2D_minus1y, OPS_RW));

  int rangexy_l2a[] = {x_min-2,x_min-1,y_min-depth,y_max+depth};
  if(depth ==2)
  ops_par_loop(update_halo_kernel1, "update_halo_kernel", 2, rangexy_l2a,
              ops_arg_dat(density0, sten_self2D_plus3x, OPS_RW),
              ops_arg_dat(density1, sten_self2D_plus3x, OPS_RW),
              ops_arg_dat(energy0, sten_self2D_plus3x, OPS_RW),
              ops_arg_dat(energy1, sten_self2D_plus3x, OPS_RW),
              ops_arg_dat(pressure, sten_self2D_plus3x, OPS_RW),
              ops_arg_dat(viscosity, sten_self2D_plus3x, OPS_RW),
              ops_arg_dat(soundspeed, sten_self2D_plus3x, OPS_RW));

  int rangexy_l1a[] = {x_min-1,x_min,y_min-depth,y_max+depth};
  ops_par_loop(update_halo_kernel1, "update_halo_kernel", 2, rangexy_l1a,
              ops_arg_dat(density0, sten_self2D_plus1x, OPS_RW),
              ops_arg_dat(density1, sten_self2D_plus1x, OPS_RW),
              ops_arg_dat(energy0, sten_self2D_plus1x, OPS_RW),
              ops_arg_dat(energy1, sten_self2D_plus1x, OPS_RW),
              ops_arg_dat(pressure, sten_self2D_plus1x, OPS_RW),
              ops_arg_dat(viscosity, sten_self2D_plus1x, OPS_RW),
              ops_arg_dat(soundspeed, sten_self2D_plus1x, OPS_RW));

  int rangexy_r2a[] = {x_max+1,x_max+2,y_min-depth,y_max+depth};

  if(depth ==2)
  ops_par_loop(update_halo_kernel1, "update_halo_kernel", 2, rangexy_r2a,
              ops_arg_dat(density0, sten_self2D_minus3x, OPS_RW),
              ops_arg_dat(density1, sten_self2D_minus3x, OPS_RW),
              ops_arg_dat(energy0, sten_self2D_minus3x, OPS_RW),
              ops_arg_dat(energy1, sten_self2D_minus3x, OPS_RW),
              ops_arg_dat(pressure, sten_self2D_minus3x, OPS_RW),
              ops_arg_dat(viscosity, sten_self2D_minus3x, OPS_RW),
              ops_arg_dat(soundspeed, sten_self2D_minus3x, OPS_RW));

  int rangexy_r1a[] = {x_max,x_max+1,y_min-depth,y_max+depth};
  ops_par_loop(update_halo_kernel1, "update_halo_kernel", 2, rangexy_r1a,
              ops_arg_dat(density0, sten_self2D_minus1x, OPS_RW),
              ops_arg_dat(density1, sten_self2D_minus1x, OPS_RW),
              ops_arg_dat(energy0, sten_self2D_minus1x, OPS_RW),
              ops_arg_dat(energy1, sten_self2D_minus1x, OPS_RW),
              ops_arg_dat(pressure, sten_self2D_minus1x, OPS_RW),
              ops_arg_dat(viscosity, sten_self2D_minus1x, OPS_RW),
              ops_arg_dat(soundspeed, sten_self2D_minus1x, OPS_RW));

  //
  //xvel0, xvel1 has the same boundary ranges and assignment
  //


  int rangexy_b2b[] = {x_min-depth,x_max+1+depth,y_min-2,y_min-1};
  if(depth == 2)
  ops_par_loop(update_halo_kernel2_xvel_plus, "update_halo_kernel2_xvel_plus", 2, rangexy_b2b,
              ops_arg_dat(xvel0, sten_self2D_plus4y, OPS_RW),
              ops_arg_dat(xvel1, sten_self2D_plus4y, OPS_RW));

  int rangexy_b1b[] = {x_min-depth,x_max+1+depth,y_min-1,y_min};
  ops_par_loop(update_halo_kernel2_xvel_plus, "update_halo_kernel2_xvel_plus", 2, rangexy_b1b,
              ops_arg_dat(xvel0, sten_self2D_plus2y, OPS_RW),
              ops_arg_dat(xvel1, sten_self2D_plus2y, OPS_RW));

  //int rangexy_t2b[] = {x_min-depth,x_max+1+depth,y_max+1,y_max+2};
  int rangexy_t2b[] = {x_min-depth,x_max+1+depth,y_max+2,y_max+3};
  if(depth == 2)
  ops_par_loop(update_halo_kernel2_xvel_plus, "update_halo_kernel2_xvel_plus", 2, rangexy_t2b,
              ops_arg_dat(xvel0, sten_self2D_minus4y, OPS_RW),
              ops_arg_dat(xvel1, sten_self2D_minus4y, OPS_RW));

  //int rangexy_t1b[] = {x_min-depth,x_max+1+depth,y_max,y_max+1};
  int rangexy_t1b[] = {x_min-depth,x_max+1+depth,y_max+1,y_max+2};
  ops_par_loop(update_halo_kernel2_xvel_plus, "update_halo_kernel2_xvel_plus", 2, rangexy_t1b,
              ops_arg_dat(xvel0, sten_self2D_minus2y, OPS_RW),
              ops_arg_dat(xvel1, sten_self2D_minus2y, OPS_RW));


  int rangexy_l2b[] = {x_min-2,x_min-1,y_min-depth,y_max+1+depth};
  if(depth ==2)
  ops_par_loop(update_halo_kernel2_xvel_minus, "update_halo_kernel2_xvel_minus", 2, rangexy_l2b,
              ops_arg_dat(xvel0, sten_self2D_plus4x, OPS_RW),
              ops_arg_dat(xvel1, sten_self2D_plus4x, OPS_RW));

  int rangexy_l1b[] = {x_min-1,x_min,y_min-depth,y_max+1+depth};
    ops_par_loop(update_halo_kernel2_xvel_minus, "update_halo_kernel2_xvel_minus", 2, rangexy_l1b,
              ops_arg_dat(xvel0, sten_self2D_plus2x, OPS_RW),
              ops_arg_dat(xvel1, sten_self2D_plus2x, OPS_RW));


  //int rangexy_r2b[] = {x_max+1,x_max+2,y_min-depth,y_max+1+depth};
  int rangexy_r2b[] = {x_max+2,x_max+3,y_min-depth,y_max+1+depth};
  if(depth ==2)
  ops_par_loop(update_halo_kernel2_xvel_minus, "update_halo_kernel2", 2, rangexy_r2b,
              ops_arg_dat(xvel0, sten_self2D_minus4x, OPS_RW),
              ops_arg_dat(xvel1, sten_self2D_minus4x, OPS_RW));

  //int rangexy_r1b[] = {x_max,x_max+1,y_min-depth,y_max+1+depth};
  int rangexy_r1b[] = {x_max+1,x_max+2,y_min-depth,y_max+1+depth};
  ops_par_loop(update_halo_kernel2_xvel_minus, "update_halo_kernel2", 2, rangexy_r1b,
              ops_arg_dat(xvel0, sten_self2D_minus2x, OPS_RW),
              ops_arg_dat(xvel1, sten_self2D_minus2x, OPS_RW));


  //
  //yvel0, yvel1 has the same boundary ranges and assignment
  //

  if(depth == 2)
  ops_par_loop(update_halo_kernel2_yvel_minus, "update_halo_kernel2_yvel_minus", 2, rangexy_b2b,
              ops_arg_dat(yvel0, sten_self2D_plus4y, OPS_RW),
              ops_arg_dat(yvel1, sten_self2D_plus4y, OPS_RW));

  ops_par_loop(update_halo_kernel2_yvel_minus, "update_halo_kernel2_yvel_minus", 2, rangexy_b1b,
              ops_arg_dat(yvel0, sten_self2D_plus2y, OPS_RW),
              ops_arg_dat(yvel1, sten_self2D_plus2y, OPS_RW));

  if(depth == 2)
  ops_par_loop(update_halo_kernel2_yvel_minus, "update_halo_kernel2_yvel_minus", 2, rangexy_t2b,
              ops_arg_dat(yvel0, sten_self2D_minus4y, OPS_RW),
              ops_arg_dat(yvel1, sten_self2D_minus4y, OPS_RW));

  ops_par_loop(update_halo_kernel2_yvel_minus, "update_halo_kernel2_yvel_minus", 2, rangexy_t1b,
              ops_arg_dat(yvel0, sten_self2D_minus2y, OPS_RW),
              ops_arg_dat(yvel1, sten_self2D_minus2y, OPS_RW));

  if(depth ==2)
  ops_par_loop(update_halo_kernel2_yvel_plus, "update_halo_kernel2_yvel_plus", 2, rangexy_l2b,
              ops_arg_dat(yvel0, sten_self2D_plus4x, OPS_RW),
              ops_arg_dat(yvel1, sten_self2D_plus4x, OPS_RW));

  ops_par_loop(update_halo_kernel2_yvel_plus, "update_halo_kernel2_yvel_plus", 2, rangexy_l1b,
              ops_arg_dat(yvel0, sten_self2D_plus2x, OPS_RW),
              ops_arg_dat(yvel1, sten_self2D_plus2x, OPS_RW));

  if(depth ==2)
  ops_par_loop(update_halo_kernel2_yvel_plus, "update_halo_kernel2_yvel_plus", 2, rangexy_r2b,
              ops_arg_dat(yvel0, sten_self2D_minus4x, OPS_RW),
              ops_arg_dat(yvel1, sten_self2D_minus4x, OPS_RW));

  ops_par_loop(update_halo_kernel2_yvel_plus, "update_halo_kernel2_yvel_plus", 2, rangexy_r1b,
              ops_arg_dat(yvel0, sten_self2D_minus2x, OPS_RW),
              ops_arg_dat(yvel1, sten_self2D_minus2x, OPS_RW));



  //
  //vol_flux_x, mass_flux_x has the same ranges
  //

  int rangexy_b2c[] = {x_min-depth,x_max+1+depth,y_min-2,y_min-1};
  if(depth ==2)
  ops_par_loop(update_halo_kernel3_plus, "update_halo_kernel3", 2, rangexy_b2c,
              ops_arg_dat(vol_flux_x, sten_self2D_plus4y, OPS_RW),
              ops_arg_dat(mass_flux_x, sten_self2D_plus4y, OPS_RW));

  int rangexy_b1c[] = {x_min-depth,x_max+1+depth,y_min-1,y_min};
  ops_par_loop(update_halo_kernel3_plus, "update_halo_kernel3", 2, rangexy_b1c,
            ops_arg_dat(vol_flux_x, sten_self2D_plus2y, OPS_RW),
            ops_arg_dat(mass_flux_x, sten_self2D_plus2y, OPS_RW));


  int rangexy_t2c[] = {x_min-depth,x_max+1+depth,y_max+1,y_max+2};
  if(depth ==2)
  ops_par_loop(update_halo_kernel3_plus, "update_halo_kernel3", 2, rangexy_t2c,
            ops_arg_dat(vol_flux_x, sten_self2D_minus4y, OPS_RW),
            ops_arg_dat(mass_flux_x, sten_self2D_minus4y, OPS_RW));

  int rangexy_t1c[] = {x_min-depth,x_max+1+depth,y_max,y_max+1};
  ops_par_loop(update_halo_kernel3_plus, "update_halo_kernel3", 2, rangexy_t1c,
            ops_arg_dat(vol_flux_x, sten_self2D_minus2y, OPS_RW),
            ops_arg_dat(mass_flux_x, sten_self2D_minus2y, OPS_RW));


  int rangexy_l2c[] = {x_min-2,x_min-1,y_min-depth,y_max+depth};
  if(depth ==2)
  ops_par_loop(update_halo_kernel3_minus, "update_halo_kernel3_minus", 2, rangexy_l2c,
            ops_arg_dat(vol_flux_x, sten_self2D_plus4x, OPS_RW),
            ops_arg_dat(mass_flux_x, sten_self2D_plus4x, OPS_RW));

  int rangexy_l1c[] = {x_min-1,x_min,y_min-depth,y_max+depth};
  ops_par_loop(update_halo_kernel3_minus, "update_halo_kernel3_minus", 2, rangexy_l1c,
            ops_arg_dat(vol_flux_x, sten_self2D_plus2x, OPS_RW),
            ops_arg_dat(mass_flux_x, sten_self2D_plus2x, OPS_RW));

  int rangexy_r2c[] = {x_max+2,x_max+3,y_min-depth,y_max+depth}; //
  if(depth ==2)
  ops_par_loop(update_halo_kernel3_minus, "update_halo_kernel3_minus", 2, rangexy_r2c,
            ops_arg_dat(vol_flux_x, sten_self2D_minus4x, OPS_RW),
            ops_arg_dat(mass_flux_x, sten_self2D_minus4x, OPS_RW));

  int rangexy_r1c[] = {x_max+1,x_max+2,y_min-depth,y_max+depth}; //
  ops_par_loop(update_halo_kernel3_minus, "update_halo_kernel3_minus", 2, rangexy_r1c,
            ops_arg_dat(vol_flux_x, sten_self2D_minus2x, OPS_RW),
            ops_arg_dat(mass_flux_x, sten_self2D_minus2x, OPS_RW));

  //
  //vol_flux_y, mass_flux_y has the same ranges
  //

  int rangexy_b2d[] = {x_min-depth,x_max+depth,y_min-2,y_min-1};
  if(depth ==2)
  ops_par_loop(update_halo_kernel4_minus, "update_halo_kernel4", 2, rangexy_b2d,
              ops_arg_dat(vol_flux_y, sten_self2D_plus4y, OPS_RW),
              ops_arg_dat(mass_flux_y, sten_self2D_plus4y, OPS_RW));

  int rangexy_b1d[] = {x_min-depth,x_max+depth,y_min-1,y_min};
  ops_par_loop(update_halo_kernel4_minus, "update_halo_kernel4", 2, rangexy_b1d,
              ops_arg_dat(vol_flux_y, sten_self2D_plus2y, OPS_RW),
              ops_arg_dat(mass_flux_y, sten_self2D_plus2y, OPS_RW));

  int rangexy_t2d[] = {x_min-depth,x_max+depth,y_max+2,y_max+3}; //
  if(depth ==2)
  ops_par_loop(update_halo_kernel4_minus, "update_halo_kernel4", 2, rangexy_t2d,
              ops_arg_dat(vol_flux_y, sten_self2D_minus4y, OPS_RW),
              ops_arg_dat(mass_flux_y, sten_self2D_minus4y, OPS_RW));

  int rangexy_t1d[] = {x_min-depth,x_max+depth,y_max+1,y_max+2}; //
  ops_par_loop(update_halo_kernel4_minus, "update_halo_kernel4", 2, rangexy_t1d,
              ops_arg_dat(vol_flux_y, sten_self2D_minus2y, OPS_RW),
              ops_arg_dat(mass_flux_y, sten_self2D_minus2y, OPS_RW));

  int rangexy_l2d[] = {x_min-2,x_min-1,y_min-depth,y_max+1+depth};
  if(depth ==2)
  ops_par_loop(update_halo_kernel4_plus, "update_halo_kernel4", 2, rangexy_l2d,
              ops_arg_dat(vol_flux_y, sten_self2D_plus4x, OPS_RW),
              ops_arg_dat(mass_flux_y, sten_self2D_plus4x, OPS_RW));

  int rangexy_l1d[] = {x_min-1,x_min,y_min-depth,y_max+1+depth};
  ops_par_loop(update_halo_kernel4_plus, "update_halo_kernel4", 2, rangexy_l1d,
              ops_arg_dat(vol_flux_y, sten_self2D_plus2x, OPS_RW),
              ops_arg_dat(mass_flux_y, sten_self2D_plus2x, OPS_RW));

  int rangexy_r2d[] = {x_max+1,x_max+2,y_min-depth,y_max+1+depth};
  if(depth ==2)
  ops_par_loop(update_halo_kernel4_plus, "update_halo_kernel4", 2, rangexy_r2d,
              ops_arg_dat(vol_flux_y, sten_self2D_minus4x, OPS_RW),
              ops_arg_dat(mass_flux_y, sten_self2D_minus4x, OPS_RW));

  int rangexy_r1d[] = {x_max,x_max+1,y_min-depth,y_max+1+depth};
  ops_par_loop(update_halo_kernel4_plus, "update_halo_kernel4", 2, rangexy_r1d,
            ops_arg_dat(vol_flux_y, sten_self2D_minus2x, OPS_RW),
            ops_arg_dat(mass_flux_y, sten_self2D_minus2x, OPS_RW));

}
