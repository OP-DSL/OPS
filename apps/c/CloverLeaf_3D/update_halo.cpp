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

#define OPS_3D
#include "ops_seq.h"

#include "data.h"
#include "definitions.h"

//Cloverleaf kernels
#include  "update_halo_kernel.h"


void update_halo(int* fields, int depth)
{
  //initialize sizes using global values
  int x_cells = grid.x_cells;
  int y_cells = grid.y_cells;
  int x_min = field.x_min;
  int x_max = field.x_max;
  int y_min = field.y_min;
  int y_max = field.y_max;
  int z_min = field.z_min;
  int z_max = field.z_max;

  //
  //density0, energy0, density1, energy1, pressure, viscosity and soundspeed
  // all has the same boundary ranges
  //


  int rangexy_b2a[] = {x_min-depth,x_max+depth,y_min-2,y_min-1,z_min-depth,z_max+depth};
  if(depth ==2)
  ops_par_loop(update_halo_kernel1_b2, "update_halo_kernel1", clover_grid, 3, rangexy_b2a,
              ops_arg_dat_opt(density0, 1, S3D_000_0P30, "double", OPS_RW, fields[FIELD_DENSITY0]),
              ops_arg_dat_opt(density1, 1, S3D_000_0P30, "double", OPS_RW, fields[FIELD_DENSITY1]),
              ops_arg_dat_opt(energy0, 1, S3D_000_0P30, "double", OPS_RW, fields[FIELD_ENERGY0]),
              ops_arg_dat_opt(energy1, 1, S3D_000_0P30, "double", OPS_RW, fields[FIELD_ENERGY1]),
              ops_arg_dat_opt(pressure, 1, S3D_000_0P30, "double", OPS_RW, fields[FIELD_PRESSURE]),
              ops_arg_dat_opt(viscosity, 1, S3D_000_0P30, "double", OPS_RW, fields[FIELD_VISCOSITY]),
              ops_arg_dat_opt(soundspeed, 1, S3D_000_0P30, "double", OPS_RW, fields[FIELD_SOUNDSPEED]),
              ops_arg_gbl(fields, NUM_FIELDS, "int", OPS_READ));

  int rangexy_b1a[] = {x_min-depth,x_max+depth,y_min-1,y_min,z_min-depth,z_max+depth};
  ops_par_loop(update_halo_kernel1_b1, "update_halo_kernel1", clover_grid, 3, rangexy_b1a,
              ops_arg_dat_opt(density0, 1, S3D_000_0P10, "double", OPS_RW, fields[FIELD_DENSITY0]),
              ops_arg_dat_opt(density1, 1, S3D_000_0P10, "double", OPS_RW, fields[FIELD_DENSITY1]),
              ops_arg_dat_opt(energy0, 1, S3D_000_0P10, "double", OPS_RW, fields[FIELD_ENERGY0]),
              ops_arg_dat_opt(energy1, 1, S3D_000_0P10, "double", OPS_RW, fields[FIELD_ENERGY1]),
              ops_arg_dat_opt(pressure, 1, S3D_000_0P10, "double", OPS_RW, fields[FIELD_PRESSURE]),
              ops_arg_dat_opt(viscosity, 1, S3D_000_0P10, "double", OPS_RW, fields[FIELD_VISCOSITY]),
              ops_arg_dat_opt(soundspeed, 1, S3D_000_0P10, "double", OPS_RW, fields[FIELD_SOUNDSPEED]),
              ops_arg_gbl(fields, NUM_FIELDS, "int", OPS_READ));

  int rangexy_t2a[] = {x_min-depth,x_max+depth,y_max+1,y_max+2,z_min-depth,z_max+depth};
  if(depth ==2)
  ops_par_loop(update_halo_kernel1_t2, "update_halo_kernel1", clover_grid, 3, rangexy_t2a,
              ops_arg_dat_opt(density0, 1, S3D_000_0M30, "double", OPS_RW, fields[FIELD_DENSITY0]),
              ops_arg_dat_opt(density1, 1, S3D_000_0M30, "double", OPS_RW, fields[FIELD_DENSITY1]),
              ops_arg_dat_opt(energy0, 1, S3D_000_0M30, "double", OPS_RW, fields[FIELD_ENERGY0]),
              ops_arg_dat_opt(energy1, 1, S3D_000_0M30, "double", OPS_RW, fields[FIELD_ENERGY1]),
              ops_arg_dat_opt(pressure, 1, S3D_000_0M30, "double", OPS_RW, fields[FIELD_PRESSURE]),
              ops_arg_dat_opt(viscosity, 1, S3D_000_0M30, "double", OPS_RW, fields[FIELD_VISCOSITY]),
              ops_arg_dat_opt(soundspeed, 1, S3D_000_0M30, "double", OPS_RW, fields[FIELD_SOUNDSPEED]),
              ops_arg_gbl(fields, NUM_FIELDS, "int", OPS_READ));

  int rangexy_t1a[] = {x_min-depth,x_max+depth,y_max,y_max+1,z_min-depth,z_max+depth};
  ops_par_loop(update_halo_kernel1_t1, "update_halo_kernel1", clover_grid, 3, rangexy_t1a,
               ops_arg_dat_opt(density0, 1, S3D_000_0M10, "double", OPS_RW, fields[FIELD_DENSITY0]),
               ops_arg_dat_opt(density1, 1, S3D_000_0M10, "double", OPS_RW, fields[FIELD_DENSITY1]),
               ops_arg_dat_opt(energy0, 1, S3D_000_0M10, "double", OPS_RW, fields[FIELD_ENERGY0]),
               ops_arg_dat_opt(energy1, 1, S3D_000_0M10, "double", OPS_RW, fields[FIELD_ENERGY1]),
               ops_arg_dat_opt(pressure, 1, S3D_000_0M10, "double", OPS_RW, fields[FIELD_PRESSURE]),
               ops_arg_dat_opt(viscosity, 1, S3D_000_0M10, "double", OPS_RW, fields[FIELD_VISCOSITY]),
               ops_arg_dat_opt(soundspeed, 1, S3D_000_0M10, "double", OPS_RW, fields[FIELD_SOUNDSPEED]),
               ops_arg_gbl(fields, NUM_FIELDS, "int", OPS_READ));

  int rangexy_l2a[] = {x_min-2,x_min-1,y_min-depth,y_max+depth,z_min-depth,z_max+depth};
  if(depth ==2)
  ops_par_loop(update_halo_kernel1_l2, "update_halo_kernel", clover_grid, 3, rangexy_l2a,
               ops_arg_dat_opt(density0, 1, S3D_000_P300, "double", OPS_RW, fields[FIELD_DENSITY0]),
               ops_arg_dat_opt(density1, 1, S3D_000_P300, "double", OPS_RW, fields[FIELD_DENSITY1]),
               ops_arg_dat_opt(energy0, 1, S3D_000_P300, "double", OPS_RW, fields[FIELD_ENERGY0]),
               ops_arg_dat_opt(energy1, 1, S3D_000_P300, "double", OPS_RW, fields[FIELD_ENERGY1]),
               ops_arg_dat_opt(pressure, 1, S3D_000_P300, "double", OPS_RW, fields[FIELD_PRESSURE]),
               ops_arg_dat_opt(viscosity, 1, S3D_000_P300, "double", OPS_RW, fields[FIELD_VISCOSITY]),
               ops_arg_dat_opt(soundspeed, 1, S3D_000_P300, "double", OPS_RW, fields[FIELD_SOUNDSPEED]),
               ops_arg_gbl(fields, NUM_FIELDS, "int", OPS_READ));

  int rangexy_l1a[] = {x_min-1,x_min,y_min-depth,y_max+depth,z_min-depth,z_max+depth};
  ops_par_loop(update_halo_kernel1_l1, "update_halo_kernel", clover_grid, 3, rangexy_l1a,
               ops_arg_dat_opt(density0, 1, S3D_000_P100, "double", OPS_RW, fields[FIELD_DENSITY0]),
               ops_arg_dat_opt(density1, 1, S3D_000_P100, "double", OPS_RW, fields[FIELD_DENSITY1]),
               ops_arg_dat_opt(energy0, 1, S3D_000_P100, "double", OPS_RW, fields[FIELD_ENERGY0]),
               ops_arg_dat_opt(energy1, 1, S3D_000_P100, "double", OPS_RW, fields[FIELD_ENERGY1]),
               ops_arg_dat_opt(pressure, 1, S3D_000_P100, "double", OPS_RW, fields[FIELD_PRESSURE]),
               ops_arg_dat_opt(viscosity, 1, S3D_000_P100, "double", OPS_RW, fields[FIELD_VISCOSITY]),
               ops_arg_dat_opt(soundspeed, 1, S3D_000_P100, "double", OPS_RW, fields[FIELD_SOUNDSPEED]),
               ops_arg_gbl(fields, NUM_FIELDS, "int", OPS_READ));

  int rangexy_r2a[] = {x_max+1,x_max+2,y_min-depth,y_max+depth,z_min-depth,z_max+depth};
  if(depth ==2)
  ops_par_loop(update_halo_kernel1_r2, "update_halo_kernel", clover_grid, 3, rangexy_r2a,
               ops_arg_dat_opt(density0, 1, S3D_000_M300, "double", OPS_RW, fields[FIELD_DENSITY0]),
               ops_arg_dat_opt(density1, 1, S3D_000_M300, "double", OPS_RW, fields[FIELD_DENSITY1]),
               ops_arg_dat_opt(energy0, 1, S3D_000_M300, "double", OPS_RW, fields[FIELD_ENERGY0]),
               ops_arg_dat_opt(energy1, 1, S3D_000_M300, "double", OPS_RW, fields[FIELD_ENERGY1]),
               ops_arg_dat_opt(pressure, 1, S3D_000_M300, "double", OPS_RW, fields[FIELD_PRESSURE]),
               ops_arg_dat_opt(viscosity, 1, S3D_000_M300, "double", OPS_RW, fields[FIELD_VISCOSITY]),
               ops_arg_dat_opt(soundspeed, 1, S3D_000_M300, "double", OPS_RW, fields[FIELD_SOUNDSPEED]),
               ops_arg_gbl(fields, NUM_FIELDS, "int", OPS_READ));

  int rangexy_r1a[] = {x_max,x_max+1,y_min-depth,y_max+depth,z_min-depth,z_max+depth};
  ops_par_loop(update_halo_kernel1_r1, "update_halo_kernel", clover_grid, 3, rangexy_r1a,
               ops_arg_dat_opt(density0, 1, S3D_000_M100, "double", OPS_RW, fields[FIELD_DENSITY0]),
               ops_arg_dat_opt(density1, 1, S3D_000_M100, "double", OPS_RW, fields[FIELD_DENSITY1]),
               ops_arg_dat_opt(energy0, 1, S3D_000_M100, "double", OPS_RW, fields[FIELD_ENERGY1]),
               ops_arg_dat_opt(energy1, 1, S3D_000_M100, "double", OPS_RW, fields[FIELD_DENSITY0]),
               ops_arg_dat_opt(pressure, 1, S3D_000_M100, "double", OPS_RW, fields[FIELD_PRESSURE]),
               ops_arg_dat_opt(viscosity, 1, S3D_000_M100, "double", OPS_RW, fields[FIELD_VISCOSITY]),
               ops_arg_dat_opt(soundspeed, 1, S3D_000_M100, "double", OPS_RW, fields[FIELD_SOUNDSPEED]),
               ops_arg_gbl(fields, NUM_FIELDS, "int", OPS_READ));

  int rangexy_ba2a[] = {x_min-depth,x_max+depth,y_min-depth,y_max+depth,z_min-2,z_min-1};
  if(depth ==2)
  ops_par_loop(update_halo_kernel1_ba2, "update_halo_kernel", clover_grid, 3, rangexy_ba2a,
              ops_arg_dat_opt(density0, 1, S3D_000_00P3, "double", OPS_RW, fields[FIELD_DENSITY0]),
              ops_arg_dat_opt(density1, 1, S3D_000_00P3, "double", OPS_RW, fields[FIELD_DENSITY1]),
              ops_arg_dat_opt(energy0, 1, S3D_000_00P3, "double", OPS_RW, fields[FIELD_ENERGY0]),
              ops_arg_dat_opt(energy1, 1, S3D_000_00P3, "double", OPS_RW, fields[FIELD_ENERGY1]),
              ops_arg_dat_opt(pressure, 1, S3D_000_00P3, "double", OPS_RW, fields[FIELD_PRESSURE]),
              ops_arg_dat_opt(viscosity, 1, S3D_000_00P3, "double", OPS_RW, fields[FIELD_VISCOSITY]),
              ops_arg_dat_opt(soundspeed, 1, S3D_000_00P3, "double", OPS_RW, fields[FIELD_SOUNDSPEED]),
              ops_arg_gbl(fields, NUM_FIELDS, "int", OPS_READ));

  int rangexy_ba1a[] = {x_min-depth,x_max+depth,y_min-depth,y_max+depth,z_min-1,z_min};
  ops_par_loop(update_halo_kernel1_ba1, "update_halo_kernel", clover_grid, 3, rangexy_ba1a,
              ops_arg_dat_opt(density0, 1, S3D_000_00P1, "double", OPS_RW, fields[FIELD_DENSITY0]),
              ops_arg_dat_opt(density1, 1, S3D_000_00P1, "double", OPS_RW, fields[FIELD_DENSITY1]),
              ops_arg_dat_opt(energy0, 1, S3D_000_00P1, "double", OPS_RW, fields[FIELD_ENERGY0]),
              ops_arg_dat_opt(energy1, 1, S3D_000_00P1, "double", OPS_RW, fields[FIELD_ENERGY1]),
              ops_arg_dat_opt(pressure, 1, S3D_000_00P1, "double", OPS_RW, fields[FIELD_PRESSURE]),
              ops_arg_dat_opt(viscosity, 1, S3D_000_00P1, "double", OPS_RW, fields[FIELD_VISCOSITY]),
              ops_arg_dat_opt(soundspeed, 1, S3D_000_00P1, "double", OPS_RW, fields[FIELD_SOUNDSPEED]),
              ops_arg_gbl(fields, NUM_FIELDS, "int", OPS_READ));

  int rangexy_fr2a[] = {x_min-depth,x_max+depth,y_min-depth,y_max+depth,z_max+1,z_max+2};
  if(depth ==2)
  ops_par_loop(update_halo_kernel1_fr2, "update_halo_kernel", clover_grid, 3, rangexy_fr2a,
              ops_arg_dat_opt(density0, 1, S3D_000_00M3, "double", OPS_RW, fields[FIELD_DENSITY0]),
              ops_arg_dat_opt(density1, 1, S3D_000_00M3, "double", OPS_RW, fields[FIELD_DENSITY1]),
              ops_arg_dat_opt(energy0, 1, S3D_000_00M3, "double", OPS_RW, fields[FIELD_ENERGY0]),
              ops_arg_dat_opt(energy1, 1, S3D_000_00M3, "double", OPS_RW, fields[FIELD_ENERGY1]),
              ops_arg_dat_opt(pressure, 1, S3D_000_00M3, "double", OPS_RW, fields[FIELD_PRESSURE]),
              ops_arg_dat_opt(viscosity, 1, S3D_000_00M3, "double", OPS_RW, fields[FIELD_VISCOSITY]),
              ops_arg_dat_opt(soundspeed, 1, S3D_000_00M3, "double", OPS_RW, fields[FIELD_SOUNDSPEED]),
              ops_arg_gbl(fields, NUM_FIELDS, "int", OPS_READ));

  int rangexy_fr1a[] = {x_min-depth,x_max+depth,y_min-depth,y_max+depth,z_max,z_max+1};
  ops_par_loop(update_halo_kernel1_fr1, "update_halo_kernel", clover_grid, 3, rangexy_fr1a,
              ops_arg_dat_opt(density0, 1, S3D_000_00M1, "double", OPS_RW, fields[FIELD_DENSITY0]),
              ops_arg_dat_opt(density1, 1, S3D_000_00M1, "double", OPS_RW, fields[FIELD_DENSITY1]),
              ops_arg_dat_opt(energy0, 1, S3D_000_00M1, "double", OPS_RW, fields[FIELD_ENERGY1]),
              ops_arg_dat_opt(energy1, 1, S3D_000_00M1, "double", OPS_RW, fields[FIELD_DENSITY0]),
              ops_arg_dat_opt(pressure, 1, S3D_000_00M1, "double", OPS_RW, fields[FIELD_PRESSURE]),
              ops_arg_dat_opt(viscosity, 1, S3D_000_00M1, "double", OPS_RW, fields[FIELD_VISCOSITY]),
              ops_arg_dat_opt(soundspeed, 1, S3D_000_00M1, "double", OPS_RW, fields[FIELD_SOUNDSPEED]),
              ops_arg_gbl(fields, NUM_FIELDS, "int", OPS_READ));



  //
  //xvel0, xvel1 has the same boundary ranges and assignment
  //


  int rangexy_b2b[] = {x_min-depth,x_max+1+depth,y_min-2,y_min-1,z_min-depth,z_max+1+depth};
  if(depth == 2)
  ops_par_loop(update_halo_kernel2_xvel_plus_4_bot, "update_halo_kernel2_xvel_plus_4_bot", clover_grid, 3, rangexy_b2b,
               ops_arg_dat_opt(xvel0, 1, S3D_000_0P40, "double", OPS_RW, fields[FIELD_XVEL0]),
               ops_arg_dat_opt(xvel1, 1, S3D_000_0P40, "double", OPS_RW, fields[FIELD_XVEL1]),
               ops_arg_gbl(fields, NUM_FIELDS, "int", OPS_READ));

  int rangexy_b1b[] = {x_min-depth,x_max+1+depth,y_min-1,y_min,z_min-depth,z_max+1+depth};
  ops_par_loop(update_halo_kernel2_xvel_plus_2_bot, "update_halo_kernel2_xvel_plus_2_bot", clover_grid, 3, rangexy_b1b,
               ops_arg_dat_opt(xvel0, 1, S3D_000_0P20, "double", OPS_RW, fields[FIELD_XVEL0]),
               ops_arg_dat_opt(xvel1, 1, S3D_000_0P20, "double", OPS_RW, fields[FIELD_XVEL1]),
               ops_arg_gbl(fields, NUM_FIELDS, "int", OPS_READ));

  int rangexy_t2b[] = {x_min-depth,x_max+1+depth,y_max+2,y_max+3,z_min-depth,z_max+1+depth};
  if(depth == 2)
  ops_par_loop(update_halo_kernel2_xvel_plus_4_top, "update_halo_kernel2_xvel_minus_4_top", clover_grid, 3, rangexy_t2b,
               ops_arg_dat_opt(xvel0, 1, S3D_000_0M40, "double", OPS_RW, fields[FIELD_XVEL0]),
               ops_arg_dat_opt(xvel1, 1, S3D_000_0M40, "double", OPS_RW, fields[FIELD_XVEL1]),
               ops_arg_gbl(fields, NUM_FIELDS, "int", OPS_READ));

  int rangexy_t1b[] = {x_min-depth,x_max+1+depth,y_max+1,y_max+2,z_min-depth,z_max+1+depth};
  ops_par_loop(update_halo_kernel2_xvel_plus_2_top, "update_halo_kernel2_xvel_minus_2_top", clover_grid, 3, rangexy_t1b,
               ops_arg_dat_opt(xvel0, 1, S3D_000_0M20, "double", OPS_RW, fields[FIELD_XVEL0]),
               ops_arg_dat_opt(xvel1, 1, S3D_000_0M20, "double", OPS_RW, fields[FIELD_XVEL1]),
               ops_arg_gbl(fields, NUM_FIELDS, "int", OPS_READ));

  int rangexy_l2b[] = {x_min-2,x_min-1,y_min-depth,y_max+1+depth,z_min-depth,z_max+1+depth};
  if(depth ==2)
  ops_par_loop(update_halo_kernel2_xvel_minus_4_left, "update_halo_kernel2_xvel_plus_4_left", clover_grid, 3, rangexy_l2b,
               ops_arg_dat_opt(xvel0, 1, S3D_000_P400, "double", OPS_RW, fields[FIELD_XVEL0]),
               ops_arg_dat_opt(xvel1, 1, S3D_000_P400, "double", OPS_RW, fields[FIELD_XVEL1]),
               ops_arg_gbl(fields, NUM_FIELDS, "int", OPS_READ));

  int rangexy_l1b[] = {x_min-1,x_min,y_min-depth,y_max+1+depth,z_min-depth,z_max+1+depth};
  ops_par_loop(update_halo_kernel2_xvel_minus_2_left, "update_halo_kernel2_xvel_plus_2_left", clover_grid, 3, rangexy_l1b,
               ops_arg_dat_opt(xvel0, 1, S3D_000_P200, "double", OPS_RW, fields[FIELD_XVEL0]),
               ops_arg_dat_opt(xvel1, 1, S3D_000_P200, "double", OPS_RW, fields[FIELD_XVEL1]),
               ops_arg_gbl(fields, NUM_FIELDS, "int", OPS_READ));

  int rangexy_r2b[] = {x_max+2,x_max+3,y_min-depth,y_max+1+depth,z_min-depth,z_max+1+depth};
  if(depth ==2)
  ops_par_loop(update_halo_kernel2_xvel_minus_4_right, "update_halo_kernel2_xvel_minus_4_right", clover_grid, 3, rangexy_r2b,
               ops_arg_dat_opt(xvel0, 1, S3D_000_M400, "double", OPS_RW, fields[FIELD_XVEL0]),
               ops_arg_dat_opt(xvel1, 1, S3D_000_M400, "double", OPS_RW, fields[FIELD_XVEL1]),
               ops_arg_gbl(fields, NUM_FIELDS, "int", OPS_READ));

  int rangexy_r1b[] = {x_max+1,x_max+2,y_min-depth,y_max+1+depth,z_min-depth,z_max+1+depth};
  ops_par_loop(update_halo_kernel2_xvel_minus_2_right, "update_halo_kernel2_xvel_minus_2_right", clover_grid, 3, rangexy_r1b,
               ops_arg_dat_opt(xvel0, 1, S3D_000_M200, "double", OPS_RW, fields[FIELD_XVEL0]),
               ops_arg_dat_opt(xvel1, 1, S3D_000_M200, "double", OPS_RW, fields[FIELD_XVEL1]),
               ops_arg_gbl(fields, NUM_FIELDS, "int", OPS_READ));

  int rangexy_ba2b[] = {x_min-depth,x_max+1+depth,y_min-depth,y_max+1+depth,z_min-2,z_min-1};
  if(depth ==2)
  ops_par_loop(update_halo_kernel2_xvel_plus_4_back, "update_halo_kernel2_xvel_plus_4_back", clover_grid, 3, rangexy_ba2b,
              ops_arg_dat_opt(xvel0, 1, S3D_000_00P4, "double", OPS_RW, fields[FIELD_XVEL0]),
              ops_arg_dat_opt(xvel1, 1, S3D_000_00P4, "double", OPS_RW, fields[FIELD_XVEL1]),
              ops_arg_gbl(fields, NUM_FIELDS, "int", OPS_READ));

  int rangexy_ba1b[] = {x_min-depth,x_max+1+depth,y_min-depth,y_max+1+depth,z_min-1,z_min};
  ops_par_loop(update_halo_kernel2_xvel_plus_2_back, "update_halo_kernel2_xvel_plus_2_back", clover_grid, 3, rangexy_ba1b,
              ops_arg_dat_opt(xvel0, 1, S3D_000_00P2, "double", OPS_RW, fields[FIELD_XVEL0]),
              ops_arg_dat_opt(xvel1, 1, S3D_000_00P2, "double", OPS_RW, fields[FIELD_XVEL1]),
              ops_arg_gbl(fields, NUM_FIELDS, "int", OPS_READ));

  int rangexy_fr2b[] = {x_min-depth,x_max+1+depth,y_min-depth,y_max+1+depth,z_max+2,z_max+3};
  if(depth ==2)
  ops_par_loop(update_halo_kernel2_xvel_plus_4_front, "update_halo_kernel2_xvel_minus_4_front", clover_grid, 3, rangexy_fr2b,
              ops_arg_dat_opt(xvel0, 1, S3D_000_00M4, "double", OPS_RW, fields[FIELD_XVEL0]),
              ops_arg_dat_opt(xvel1, 1, S3D_000_00M4, "double", OPS_RW, fields[FIELD_XVEL1]),
              ops_arg_gbl(fields, NUM_FIELDS, "int", OPS_READ));

  int rangexy_fr1b[] = {x_min-depth,x_max+1+depth,y_min-depth,y_max+1+depth,z_max+1,z_max+2};
  ops_par_loop(update_halo_kernel2_xvel_plus_2_front, "update_halo_kernel2_xvel_minus_2_front", clover_grid, 3, rangexy_fr1b,
              ops_arg_dat_opt(xvel0, 1, S3D_000_00M2, "double", OPS_RW, fields[FIELD_XVEL0]),
              ops_arg_dat_opt(xvel1, 1, S3D_000_00M2, "double", OPS_RW, fields[FIELD_XVEL1]),
              ops_arg_gbl(fields, NUM_FIELDS, "int", OPS_READ));

  //
  //yvel0, yvel1 has the same boundary ranges and assignment
  //

  if(depth == 2)
  ops_par_loop(update_halo_kernel2_yvel_minus_4_bot, "update_halo_kernel2_yvel_plus_4_bot", clover_grid, 3, rangexy_b2b,
               ops_arg_dat_opt(yvel0, 1, S3D_000_0P40, "double", OPS_RW, fields[FIELD_YVEL0]),
               ops_arg_dat_opt(yvel1, 1, S3D_000_0P40, "double", OPS_RW, fields[FIELD_YVEL1]),
               ops_arg_gbl(fields, NUM_FIELDS, "int", OPS_READ));

  ops_par_loop(update_halo_kernel2_yvel_minus_2_bot, "update_halo_kernel2_yvel_plus_2_bot", clover_grid, 3, rangexy_b1b,
               ops_arg_dat_opt(yvel0, 1, S3D_000_0P20, "double", OPS_RW, fields[FIELD_YVEL0]),
               ops_arg_dat_opt(yvel1, 1, S3D_000_0P20, "double", OPS_RW, fields[FIELD_YVEL1]),
               ops_arg_gbl(fields, NUM_FIELDS, "int", OPS_READ));

  if(depth == 2)
  ops_par_loop(update_halo_kernel2_yvel_minus_4_top, "update_halo_kernel2_yvel_minus_4_top", clover_grid, 3, rangexy_t2b,
               ops_arg_dat_opt(yvel0, 1, S3D_000_0M40, "double", OPS_RW, fields[FIELD_YVEL0]),
               ops_arg_dat_opt(yvel1, 1, S3D_000_0M40, "double", OPS_RW, fields[FIELD_YVEL1]),
               ops_arg_gbl(fields, NUM_FIELDS, "int", OPS_READ));

  ops_par_loop(update_halo_kernel2_yvel_minus_2_top, "update_halo_kernel2_yvel_minus_2_top", clover_grid, 3, rangexy_t1b,
               ops_arg_dat_opt(yvel0, 1, S3D_000_0M20, "double", OPS_RW, fields[FIELD_YVEL0]),
               ops_arg_dat_opt(yvel1, 1, S3D_000_0M20, "double", OPS_RW, fields[FIELD_YVEL1]),
               ops_arg_gbl(fields, NUM_FIELDS, "int", OPS_READ));

  if(depth ==2)
  ops_par_loop(update_halo_kernel2_yvel_plus_4_left, "update_halo_kernel2_yvel_plus_4_left", clover_grid, 3, rangexy_l2b,
               ops_arg_dat_opt(yvel0, 1, S3D_000_P400, "double", OPS_RW, fields[FIELD_YVEL0]),
               ops_arg_dat_opt(yvel1, 1, S3D_000_P400, "double", OPS_RW, fields[FIELD_YVEL1]),
               ops_arg_gbl(fields, NUM_FIELDS, "int", OPS_READ));

  ops_par_loop(update_halo_kernel2_yvel_plus_2_left, "update_halo_kernel2_yvel_plus_2_left", clover_grid, 3, rangexy_l1b,
               ops_arg_dat_opt(yvel0, 1, S3D_000_P200, "double", OPS_RW, fields[FIELD_YVEL0]),
               ops_arg_dat_opt(yvel1, 1, S3D_000_P200, "double", OPS_RW, fields[FIELD_YVEL1]),
               ops_arg_gbl(fields, NUM_FIELDS, "int", OPS_READ));

  if(depth ==2)
  ops_par_loop(update_halo_kernel2_yvel_plus_4_right, "update_halo_kernel2_yvel_minus_4_right", clover_grid, 3, rangexy_r2b,
               ops_arg_dat_opt(yvel0, 1, S3D_000_M400, "double", OPS_RW, fields[FIELD_YVEL0]),
               ops_arg_dat_opt(yvel1, 1, S3D_000_M400, "double", OPS_RW, fields[FIELD_YVEL1]),
               ops_arg_gbl(fields, NUM_FIELDS, "int", OPS_READ));


  ops_par_loop(update_halo_kernel2_yvel_plus_2_right, "update_halo_kernel2_yvel_minus_2_right", clover_grid, 3, rangexy_r1b,
               ops_arg_dat_opt(yvel0, 1, S3D_000_M200, "double", OPS_RW, fields[FIELD_YVEL0]),
               ops_arg_dat_opt(yvel1, 1, S3D_000_M200, "double", OPS_RW, fields[FIELD_YVEL1]),
               ops_arg_gbl(fields, NUM_FIELDS, "int", OPS_READ));

  if(depth ==2)
  ops_par_loop(update_halo_kernel2_yvel_plus_4_back, "update_halo_kernel2_yvel_plus_4_back", clover_grid, 3, rangexy_ba2b,
             ops_arg_dat_opt(yvel0, 1, S3D_000_00P4, "double", OPS_RW, fields[FIELD_YVEL0]),
             ops_arg_dat_opt(yvel1, 1, S3D_000_00P4, "double", OPS_RW, fields[FIELD_YVEL1]),
             ops_arg_gbl(fields, NUM_FIELDS, "int", OPS_READ));

  ops_par_loop(update_halo_kernel2_yvel_plus_2_back, "update_halo_kernel2_yvel_plus_2_back", clover_grid, 3, rangexy_ba1b,
             ops_arg_dat_opt(yvel0, 1, S3D_000_00P2, "double", OPS_RW, fields[FIELD_YVEL0]),
             ops_arg_dat_opt(yvel1, 1, S3D_000_00P2, "double", OPS_RW, fields[FIELD_YVEL1]),
             ops_arg_gbl(fields, NUM_FIELDS, "int", OPS_READ));

  if(depth ==2)
  ops_par_loop(update_halo_kernel2_yvel_plus_4_front, "update_halo_kernel2_yvel_minus_4_front", clover_grid, 3, rangexy_fr2b,
             ops_arg_dat_opt(yvel0, 1, S3D_000_00M4, "double", OPS_RW, fields[FIELD_YVEL0]),
             ops_arg_dat_opt(yvel1, 1, S3D_000_00M4, "double", OPS_RW, fields[FIELD_YVEL1]),
             ops_arg_gbl(fields, NUM_FIELDS, "int", OPS_READ));

  ops_par_loop(update_halo_kernel2_yvel_plus_2_front, "update_halo_kernel2_yvel_minus_2_front", clover_grid, 3, rangexy_fr1b,
             ops_arg_dat_opt(yvel0, 1, S3D_000_00M2, "double", OPS_RW, fields[FIELD_YVEL0]),
             ops_arg_dat_opt(yvel1, 1, S3D_000_00M2, "double", OPS_RW, fields[FIELD_YVEL1]),
             ops_arg_gbl(fields, NUM_FIELDS, "int", OPS_READ));

  //
  //zvel0, zvel1 has the same boundary ranges and assignment
  //

  if(depth == 2)
  ops_par_loop(update_halo_kernel2_zvel_plus_4_bot, "update_halo_kernel2_zvel_plus_4_bot", clover_grid, 3, rangexy_b2b,
               ops_arg_dat_opt(zvel0, 1, S3D_000_0P40, "double", OPS_RW, fields[FIELD_ZVEL0]),
               ops_arg_dat_opt(zvel1, 1, S3D_000_0P40, "double", OPS_RW, fields[FIELD_ZVEL1]),
               ops_arg_gbl(fields, NUM_FIELDS, "int", OPS_READ));

  ops_par_loop(update_halo_kernel2_zvel_plus_2_bot, "update_halo_kernel2_zvel_plus_2_bot", clover_grid, 3, rangexy_b1b,
               ops_arg_dat_opt(zvel0, 1, S3D_000_0P20, "double", OPS_RW, fields[FIELD_ZVEL0]),
               ops_arg_dat_opt(zvel1, 1, S3D_000_0P20, "double", OPS_RW, fields[FIELD_ZVEL1]),
               ops_arg_gbl(fields, NUM_FIELDS, "int", OPS_READ));

  if(depth == 2)
  ops_par_loop(update_halo_kernel2_zvel_plus_4_top, "update_halo_kernel2_zvel_minus_4_top", clover_grid, 3, rangexy_t2b,
               ops_arg_dat_opt(zvel0, 1, S3D_000_0M40, "double", OPS_RW, fields[FIELD_ZVEL0]),
               ops_arg_dat_opt(zvel1, 1, S3D_000_0M40, "double", OPS_RW, fields[FIELD_ZVEL1]),
               ops_arg_gbl(fields, NUM_FIELDS, "int", OPS_READ));

  ops_par_loop(update_halo_kernel2_zvel_plus_2_top, "update_halo_kernel2_zvel_minus_2_top", clover_grid, 3, rangexy_t1b,
               ops_arg_dat_opt(zvel0, 1, S3D_000_0M20, "double", OPS_RW, fields[FIELD_ZVEL0]),
               ops_arg_dat_opt(zvel1, 1, S3D_000_0M20, "double", OPS_RW, fields[FIELD_ZVEL1]),
               ops_arg_gbl(fields, NUM_FIELDS, "int", OPS_READ));

  if(depth ==2)
  ops_par_loop(update_halo_kernel2_zvel_plus_4_left, "update_halo_kernel2_zvel_plus_4_left", clover_grid, 3, rangexy_l2b,
               ops_arg_dat_opt(zvel0, 1, S3D_000_P400, "double", OPS_RW, fields[FIELD_ZVEL0]),
               ops_arg_dat_opt(zvel1, 1, S3D_000_P400, "double", OPS_RW, fields[FIELD_ZVEL1]),
               ops_arg_gbl(fields, NUM_FIELDS, "int", OPS_READ));

  ops_par_loop(update_halo_kernel2_zvel_plus_2_left, "update_halo_kernel2_zvel_plus_2_left", clover_grid, 3, rangexy_l1b,
               ops_arg_dat_opt(zvel0, 1, S3D_000_P200, "double", OPS_RW, fields[FIELD_ZVEL0]),
               ops_arg_dat_opt(zvel1, 1, S3D_000_P200, "double", OPS_RW, fields[FIELD_ZVEL1]),
               ops_arg_gbl(fields, NUM_FIELDS, "int", OPS_READ));

  if(depth ==2)
  ops_par_loop(update_halo_kernel2_zvel_plus_4_right, "update_halo_kernel2_zvel_minus_4_right", clover_grid, 3, rangexy_r2b,
               ops_arg_dat_opt(zvel0, 1, S3D_000_M400, "double", OPS_RW, fields[FIELD_ZVEL0]),
               ops_arg_dat_opt(zvel1, 1, S3D_000_M400, "double", OPS_RW, fields[FIELD_ZVEL1]),
               ops_arg_gbl(fields, NUM_FIELDS, "int", OPS_READ));


  ops_par_loop(update_halo_kernel2_zvel_plus_2_right, "update_halo_kernel2_zvel_minus_2_right", clover_grid, 3, rangexy_r1b,
               ops_arg_dat_opt(zvel0, 1, S3D_000_M200, "double", OPS_RW, fields[FIELD_ZVEL0]),
               ops_arg_dat_opt(zvel1, 1, S3D_000_M200, "double", OPS_RW, fields[FIELD_ZVEL1]),
               ops_arg_gbl(fields, NUM_FIELDS, "int", OPS_READ));

  if(depth ==2)
  ops_par_loop(update_halo_kernel2_zvel_minus_4_back, "update_halo_kernel2_zvel_plus_4_back", clover_grid, 3, rangexy_ba2b,
             ops_arg_dat_opt(zvel0, 1, S3D_000_00P4, "double", OPS_RW, fields[FIELD_ZVEL0]),
             ops_arg_dat_opt(zvel1, 1, S3D_000_00P4, "double", OPS_RW, fields[FIELD_ZVEL1]),
             ops_arg_gbl(fields, NUM_FIELDS, "int", OPS_READ));

  ops_par_loop(update_halo_kernel2_zvel_minus_2_back, "update_halo_kernel2_zvel_plus_2_back", clover_grid, 3, rangexy_ba1b,
             ops_arg_dat_opt(zvel0, 1, S3D_000_00P2, "double", OPS_RW, fields[FIELD_ZVEL0]),
             ops_arg_dat_opt(zvel1, 1, S3D_000_00P2, "double", OPS_RW, fields[FIELD_ZVEL1]),
             ops_arg_gbl(fields, NUM_FIELDS, "int", OPS_READ));

  if(depth ==2)
  ops_par_loop(update_halo_kernel2_zvel_minus_4_front, "update_halo_kernel2_zvel_minus_4_front", clover_grid, 3, rangexy_fr2b,
             ops_arg_dat_opt(zvel0, 1, S3D_000_00M4, "double", OPS_RW, fields[FIELD_ZVEL0]),
             ops_arg_dat_opt(zvel1, 1, S3D_000_00M4, "double", OPS_RW, fields[FIELD_ZVEL1]),
             ops_arg_gbl(fields, NUM_FIELDS, "int", OPS_READ));

  ops_par_loop(update_halo_kernel2_zvel_minus_2_front, "update_halo_kernel2_zvel_minus_2_front", clover_grid, 3, rangexy_fr1b,
             ops_arg_dat_opt(zvel0, 1, S3D_000_00M2, "double", OPS_RW, fields[FIELD_ZVEL0]),
             ops_arg_dat_opt(zvel1, 1, S3D_000_00M2, "double", OPS_RW, fields[FIELD_ZVEL1]),
             ops_arg_gbl(fields, NUM_FIELDS, "int", OPS_READ));

  //
  //vol_flux_x, mass_flux_x has the same ranges
  //

  int rangexy_b2c[] = {x_min-depth,x_max+1+depth,y_min-2,y_min-1,z_min-depth,z_max+depth};
  if(depth ==2)
  ops_par_loop(update_halo_kernel3_plus_4_a, "update_halo_kernel3_plus_4_a", clover_grid, 3, rangexy_b2c,
               ops_arg_dat_opt(vol_flux_x, 1, S3D_000_0P40, "double", OPS_RW, fields[FIELD_VOL_FLUX_X]),
               ops_arg_dat_opt(mass_flux_x, 1, S3D_000_0P40, "double", OPS_RW, fields[FIELD_MASS_FLUX_X]),
               ops_arg_gbl(fields, NUM_FIELDS, "int", OPS_READ));

  int rangexy_b1c[] = {x_min-depth,x_max+1+depth,y_min-1,y_min,z_min-depth,z_max+depth};
  ops_par_loop(update_halo_kernel3_plus_2_a, "update_halo_kernel3_plus_2_a", clover_grid, 3, rangexy_b1c,
               ops_arg_dat_opt(vol_flux_x, 1, S3D_000_0P20, "double", OPS_RW, fields[FIELD_VOL_FLUX_X]),
               ops_arg_dat_opt(mass_flux_x, 1, S3D_000_0P20, "double", OPS_RW, fields[FIELD_MASS_FLUX_X]),
               ops_arg_gbl(fields, NUM_FIELDS, "int", OPS_READ));


  int rangexy_t2c[] = {x_min-depth,x_max+1+depth,y_max+1,y_max+2,z_min-depth,z_max+depth};
  if(depth ==2)
  ops_par_loop(update_halo_kernel3_plus_4_b, "update_halo_kernel3_plus_4_b", clover_grid, 3, rangexy_t2c,
               ops_arg_dat_opt(vol_flux_x, 1, S3D_000_0M40, "double", OPS_RW, fields[FIELD_VOL_FLUX_X]),
               ops_arg_dat_opt(mass_flux_x, 1, S3D_000_0M40, "double", OPS_RW, fields[FIELD_MASS_FLUX_X]),
               ops_arg_gbl(fields, NUM_FIELDS, "int", OPS_READ));

  int rangexy_t1c[] = {x_min-depth,x_max+1+depth,y_max,y_max+1,z_min-depth,z_max+depth};
  ops_par_loop(update_halo_kernel3_plus_2_b, "update_halo_kernel3_plus_2_b", clover_grid, 3, rangexy_t1c,
               ops_arg_dat_opt(vol_flux_x, 1, S3D_000_0M20, "double", OPS_RW, fields[FIELD_VOL_FLUX_X]),
               ops_arg_dat_opt(mass_flux_x, 1, S3D_000_0M20, "double", OPS_RW, fields[FIELD_MASS_FLUX_X]),
               ops_arg_gbl(fields, NUM_FIELDS, "int", OPS_READ));


  int rangexy_l2c[] = {x_min-2,x_min-1,y_min-depth,y_max+depth,z_min-depth,z_max+depth};
  if(depth ==2)
  ops_par_loop(update_halo_kernel3_minus_4_a, "update_halo_kernel3_minus_4_a", clover_grid, 3, rangexy_l2c,
               ops_arg_dat_opt(vol_flux_x, 1, S3D_000_P400, "double", OPS_RW, fields[FIELD_VOL_FLUX_X]),
               ops_arg_dat_opt(mass_flux_x, 1, S3D_000_P400, "double", OPS_RW, fields[FIELD_MASS_FLUX_X]),
               ops_arg_gbl(fields, NUM_FIELDS, "int", OPS_READ));

  int rangexy_l1c[] = {x_min-1,x_min,y_min-depth,y_max+depth,z_min-depth,z_max+depth};
  ops_par_loop(update_halo_kernel3_minus_2_a, "update_halo_kernel3_minus_2_a", clover_grid, 3, rangexy_l1c,
               ops_arg_dat_opt(vol_flux_x, 1, S3D_000_P200, "double", OPS_RW, fields[FIELD_VOL_FLUX_X]),
               ops_arg_dat_opt(mass_flux_x, 1, S3D_000_P200, "double", OPS_RW, fields[FIELD_MASS_FLUX_X]),
               ops_arg_gbl(fields, NUM_FIELDS, "int", OPS_READ));

  int rangexy_r2c[] = {x_max+2,x_max+3,y_min-depth,y_max+depth,z_min-depth,z_max+depth}; //
  if(depth ==2)
  ops_par_loop(update_halo_kernel3_minus_4_b, "update_halo_kernel3_minus_4_b", clover_grid, 3, rangexy_r2c,
               ops_arg_dat_opt(vol_flux_x, 1, S3D_000_M400, "double", OPS_RW, fields[FIELD_VOL_FLUX_X]),
               ops_arg_dat_opt(mass_flux_x, 1, S3D_000_M400, "double", OPS_RW, fields[FIELD_MASS_FLUX_X]),
               ops_arg_gbl(fields, NUM_FIELDS, "int", OPS_READ));

  int rangexy_r1c[] = {x_max+1,x_max+2,y_min-depth,y_max+depth,z_min-depth,z_max+depth}; //
  ops_par_loop(update_halo_kernel3_minus_2_b, "update_halo_kernel3_minus_2_b", clover_grid, 3, rangexy_r1c,
               ops_arg_dat_opt(vol_flux_x, 1, S3D_000_M200, "double", OPS_RW, fields[FIELD_VOL_FLUX_X]),
               ops_arg_dat_opt(mass_flux_x, 1, S3D_000_M200, "double", OPS_RW, fields[FIELD_MASS_FLUX_X]),
               ops_arg_gbl(fields, NUM_FIELDS, "int", OPS_READ));

  int rangexy_back2c[] = {x_min-depth,x_max+depth,y_min-depth,y_max+depth,z_min-2,z_min-1};
  if(depth ==2)
  ops_par_loop(update_halo_kernel3_plus_4_back, "update_halo_kernel3_plus_4_back", clover_grid, 3, rangexy_back2c,
              ops_arg_dat_opt(vol_flux_x, 1, S3D_000_00P4, "double", OPS_RW, fields[FIELD_VOL_FLUX_X]),
              ops_arg_dat_opt(mass_flux_x, 1, S3D_000_00P4, "double", OPS_RW, fields[FIELD_MASS_FLUX_X]),
              ops_arg_gbl(fields, NUM_FIELDS, "int", OPS_READ));

  int rangexy_back1c[] = {x_min-depth,x_max+depth,y_min-depth,y_max+depth,z_min-1,z_min};
  ops_par_loop(update_halo_kernel3_plus_2_back, "update_halo_kernel3_plus_2_back", clover_grid, 3, rangexy_back1c,
              ops_arg_dat_opt(vol_flux_x, 1, S3D_000_00P2, "double", OPS_RW, fields[FIELD_VOL_FLUX_X]),
              ops_arg_dat_opt(mass_flux_x, 1, S3D_000_00P2, "double", OPS_RW, fields[FIELD_MASS_FLUX_X]),
              ops_arg_gbl(fields, NUM_FIELDS, "int", OPS_READ));


  int rangexy_front2c[] = {x_min-depth,x_max+depth,y_min-depth,y_max+depth,z_max+1,z_max+2};
  if(depth ==2)
  ops_par_loop(update_halo_kernel3_plus_4_front, "update_halo_kernel3_plus_4_front", clover_grid, 3, rangexy_front2c,
              ops_arg_dat_opt(vol_flux_x, 1, S3D_000_00M4, "double", OPS_RW, fields[FIELD_VOL_FLUX_X]),
              ops_arg_dat_opt(mass_flux_x, 1, S3D_000_00M4, "double", OPS_RW, fields[FIELD_MASS_FLUX_X]),
              ops_arg_gbl(fields, NUM_FIELDS, "int", OPS_READ));

  int rangexy_front1c[] = {x_min-depth,x_max+depth,y_min-depth,y_max+depth,z_max,z_max+1};
  ops_par_loop(update_halo_kernel3_plus_2_front, "update_halo_kernel3_plus_2_front", clover_grid, 3, rangexy_front1c,
              ops_arg_dat_opt(vol_flux_x, 1, S3D_000_00M2, "double", OPS_RW, fields[FIELD_VOL_FLUX_X]),
              ops_arg_dat_opt(mass_flux_x, 1, S3D_000_00M2, "double", OPS_RW, fields[FIELD_MASS_FLUX_X]),
              ops_arg_gbl(fields, NUM_FIELDS, "int", OPS_READ));

  //
  //vol_flux_y, mass_flux_y has the same ranges
  //

  int rangexy_b2d[] = {x_min-depth,x_max+depth,y_min-2,y_min-1,z_min-depth,z_max+depth};
  if(depth ==2)
  ops_par_loop(update_halo_kernel4_minus_4_a, "update_halo_kernel4_minus_4_a", clover_grid, 3, rangexy_b2d,
               ops_arg_dat_opt(vol_flux_y, 1, S3D_000_0P40, "double", OPS_RW, fields[FIELD_VOL_FLUX_Y]),
               ops_arg_dat_opt(mass_flux_y, 1, S3D_000_0P40, "double", OPS_RW, fields[FIELD_MASS_FLUX_Y]),
               ops_arg_gbl(fields, NUM_FIELDS, "int", OPS_READ));

  int rangexy_b1d[] = {x_min-depth,x_max+depth,y_min-1,y_min,z_min-depth,z_max+depth};
  ops_par_loop(update_halo_kernel4_minus_2_a, "update_halo_kernel4_minus_2_a", clover_grid, 3, rangexy_b1d,
               ops_arg_dat_opt(vol_flux_y, 1, S3D_000_0P20, "double", OPS_RW, fields[FIELD_VOL_FLUX_Y]),
               ops_arg_dat_opt(mass_flux_y, 1, S3D_000_0P20, "double", OPS_RW, fields[FIELD_MASS_FLUX_Y]),
               ops_arg_gbl(fields, NUM_FIELDS, "int", OPS_READ));

  int rangexy_t2d[] = {x_min-depth,x_max+depth,y_max+2,y_max+3,z_min-depth,z_max+depth}; //
  if(depth ==2)
  ops_par_loop(update_halo_kernel4_minus_4_b, "update_halo_kernel4_minus_4_b", clover_grid, 3, rangexy_t2d,
               ops_arg_dat_opt(vol_flux_y, 1, S3D_000_0M40, "double", OPS_RW, fields[FIELD_VOL_FLUX_Y]),
               ops_arg_dat_opt(mass_flux_y, 1, S3D_000_0M40, "double", OPS_RW, fields[FIELD_MASS_FLUX_Y]),
               ops_arg_gbl(fields, NUM_FIELDS, "int", OPS_READ));

  int rangexy_t1d[] = {x_min-depth,x_max+depth,y_max+1,y_max+2,z_min-depth,z_max+depth}; //
  ops_par_loop(update_halo_kernel4_minus_2_b, "update_halo_kernel4_minus_2_b", clover_grid, 3, rangexy_t1d,
               ops_arg_dat_opt(vol_flux_y, 1, S3D_000_0M20, "double", OPS_RW, fields[FIELD_VOL_FLUX_Y]),
               ops_arg_dat_opt(mass_flux_y, 1, S3D_000_0M20, "double", OPS_RW, fields[FIELD_MASS_FLUX_Y]),
               ops_arg_gbl(fields, NUM_FIELDS, "int", OPS_READ));

  int rangexy_l2d[] = {x_min-2,x_min-1,y_min-depth,y_max+1+depth,z_min-depth,z_max+depth};
  if(depth ==2)
  ops_par_loop(update_halo_kernel4_plus_4_a, "update_halo_kernel4_plus_4_a", clover_grid, 3, rangexy_l2d,
               ops_arg_dat_opt(vol_flux_y, 1, S3D_000_P400, "double", OPS_RW, fields[FIELD_VOL_FLUX_Y]),
               ops_arg_dat_opt(mass_flux_y, 1, S3D_000_P400, "double", OPS_RW, fields[FIELD_MASS_FLUX_Y]),
               ops_arg_gbl(fields, NUM_FIELDS, "int", OPS_READ));

  int rangexy_l1d[] = {x_min-1,x_min,y_min-depth,y_max+1+depth,z_min-depth,z_max+depth};
  ops_par_loop(update_halo_kernel4_plus_2_a, "update_halo_kernel4_plus_2_a", clover_grid, 3, rangexy_l1d,
               ops_arg_dat_opt(vol_flux_y, 1, S3D_000_P200, "double", OPS_RW, fields[FIELD_VOL_FLUX_Y]),
               ops_arg_dat_opt(mass_flux_y, 1, S3D_000_P200, "double", OPS_RW, fields[FIELD_MASS_FLUX_Y]),
               ops_arg_gbl(fields, NUM_FIELDS, "int", OPS_READ));

  int rangexy_r2d[] = {x_max+1,x_max+2,y_min-depth,y_max+1+depth,z_min-depth,z_max+depth};
  if(depth ==2)
  ops_par_loop(update_halo_kernel4_plus_4_b, "update_halo_kernel4_plus_4_b", clover_grid, 3, rangexy_r2d,
               ops_arg_dat_opt(vol_flux_y, 1, S3D_000_M400, "double", OPS_RW, fields[FIELD_VOL_FLUX_Y]),
               ops_arg_dat_opt(mass_flux_y, 1, S3D_000_M400, "double", OPS_RW, fields[FIELD_MASS_FLUX_Y]),
               ops_arg_gbl(fields, NUM_FIELDS, "int", OPS_READ));

  int rangexy_r1d[] = {x_max,x_max+1,y_min-depth,y_max+1+depth,z_min-depth,z_max+depth};
  ops_par_loop(update_halo_kernel4_plus_2_b, "update_halo_kernel4_plus_2_b",clover_grid, 3, rangexy_r1d,
               ops_arg_dat_opt(vol_flux_y, 1, S3D_000_M200, "double", OPS_RW, fields[FIELD_VOL_FLUX_Y]),
               ops_arg_dat_opt(mass_flux_y, 1, S3D_000_M200, "double", OPS_RW, fields[FIELD_MASS_FLUX_Y]),
               ops_arg_gbl(fields, NUM_FIELDS, "int", OPS_READ));

   if(depth ==2)
   ops_par_loop(update_halo_kernel4_plus_4_back, "update_halo_kernel4_plus_4_back", clover_grid, 3, rangexy_back2c,
               ops_arg_dat_opt(vol_flux_y, 1, S3D_000_00P4, "double", OPS_RW, fields[FIELD_VOL_FLUX_Y]),
               ops_arg_dat_opt(mass_flux_y, 1, S3D_000_00P4, "double", OPS_RW, fields[FIELD_MASS_FLUX_Y]),
               ops_arg_gbl(fields, NUM_FIELDS, "int", OPS_READ));

   ops_par_loop(update_halo_kernel4_plus_2_back, "update_halo_kernel4_plus_2_back", clover_grid, 3, rangexy_back1c,
               ops_arg_dat_opt(vol_flux_y, 1, S3D_000_00P2, "double", OPS_RW, fields[FIELD_VOL_FLUX_Y]),
               ops_arg_dat_opt(mass_flux_y, 1, S3D_000_00P2, "double", OPS_RW, fields[FIELD_MASS_FLUX_Y]),
               ops_arg_gbl(fields, NUM_FIELDS, "int", OPS_READ));


   if(depth ==2)
   ops_par_loop(update_halo_kernel4_plus_4_front, "update_halo_kernel4_plus_4_front", clover_grid, 3, rangexy_front2c,
               ops_arg_dat_opt(vol_flux_y, 1, S3D_000_00M4, "double", OPS_RW, fields[FIELD_VOL_FLUX_Y]),
               ops_arg_dat_opt(mass_flux_y, 1, S3D_000_00M4, "double", OPS_RW, fields[FIELD_MASS_FLUX_Y]),
               ops_arg_gbl(fields, NUM_FIELDS, "int", OPS_READ));

   ops_par_loop(update_halo_kernel4_plus_2_front, "update_halo_kernel4_plus_2_front", clover_grid, 3, rangexy_front1c,
               ops_arg_dat_opt(vol_flux_y, 1, S3D_000_00M2, "double", OPS_RW, fields[FIELD_VOL_FLUX_Y]),
               ops_arg_dat_opt(mass_flux_y, 1, S3D_000_00M2, "double", OPS_RW, fields[FIELD_MASS_FLUX_Y]),
               ops_arg_gbl(fields, NUM_FIELDS, "int", OPS_READ));

  //
  //vol_flux_z, mass_flux_z has the same ranges
  //

  int rangexy_b2e[] = {x_min-depth,x_max+depth,y_min-2,y_min-1,z_min-depth,z_max+1+depth};
  if(depth ==2)
  ops_par_loop(update_halo_kernel5_plus_4_a, "update_halo_kernel5_plus_4_a", clover_grid, 3, rangexy_b2e,
               ops_arg_dat_opt(vol_flux_z, 1, S3D_000_0P40, "double", OPS_RW, fields[FIELD_VOL_FLUX_Z]),
               ops_arg_dat_opt(mass_flux_z, 1, S3D_000_0P40, "double", OPS_RW, fields[FIELD_MASS_FLUX_Z]),
               ops_arg_gbl(fields, NUM_FIELDS, "int", OPS_READ));

  int rangexy_b1e[] = {x_min-depth,x_max+depth,y_min-1,y_min,z_min-depth,z_max+1+depth};
  ops_par_loop(update_halo_kernel5_plus_2_a, "update_halo_kernel5_plus_2_a", clover_grid, 3, rangexy_b1e,
               ops_arg_dat_opt(vol_flux_z, 1, S3D_000_0P20, "double", OPS_RW, fields[FIELD_VOL_FLUX_Z]),
               ops_arg_dat_opt(mass_flux_z, 1, S3D_000_0P20, "double", OPS_RW, fields[FIELD_MASS_FLUX_Z]),
               ops_arg_gbl(fields, NUM_FIELDS, "int", OPS_READ));

  int rangexy_t2e[] = {x_min-depth,x_max+depth,y_max+1,y_max+2,z_min-depth,z_max+1+depth};
  if(depth ==2)
  ops_par_loop(update_halo_kernel5_plus_4_b, "update_halo_kernel5_plus_4_b", clover_grid, 3, rangexy_t2e,
               ops_arg_dat_opt(vol_flux_z, 1, S3D_000_0M40, "double", OPS_RW, fields[FIELD_VOL_FLUX_Z]),
               ops_arg_dat_opt(mass_flux_z, 1, S3D_000_0M40, "double", OPS_RW, fields[FIELD_MASS_FLUX_Z]),
               ops_arg_gbl(fields, NUM_FIELDS, "int", OPS_READ));

  int rangexy_t1e[] = {x_min-depth,x_max+depth,y_max+0,y_max+1,z_min-depth,z_max+1+depth};
  ops_par_loop(update_halo_kernel5_plus_2_b, "update_halo_kernel5_plus_2_b", clover_grid, 3, rangexy_t1e,
               ops_arg_dat_opt(vol_flux_z, 1, S3D_000_0M20, "double", OPS_RW, fields[FIELD_VOL_FLUX_Z]),
               ops_arg_dat_opt(mass_flux_z, 1, S3D_000_0M20, "double", OPS_RW, fields[FIELD_MASS_FLUX_Z]),
               ops_arg_gbl(fields, NUM_FIELDS, "int", OPS_READ));

  int rangexy_l2e[] = {x_min-2,x_min-1,y_min-depth,y_max+depth,z_min-depth,z_max+1+depth};
  if(depth ==2)
  ops_par_loop(update_halo_kernel5_plus_4_left, "update_halo_kernel5_plus_4_left", clover_grid, 3, rangexy_l2e,
               ops_arg_dat_opt(vol_flux_z, 1, S3D_000_P400, "double", OPS_RW, fields[FIELD_VOL_FLUX_Z]),
               ops_arg_dat_opt(mass_flux_z, 1, S3D_000_P400, "double", OPS_RW, fields[FIELD_MASS_FLUX_Z]),
               ops_arg_gbl(fields, NUM_FIELDS, "int", OPS_READ));

  int rangexy_l1e[] = {x_min-1,x_min,y_min-depth,y_max+depth,z_min-depth,z_max+1+depth};
  ops_par_loop(update_halo_kernel5_plus_2_left, "update_halo_kernel5_plus_2_left", clover_grid, 3, rangexy_l1e,
               ops_arg_dat_opt(vol_flux_z, 1, S3D_000_P200, "double", OPS_RW, fields[FIELD_VOL_FLUX_Z]),
               ops_arg_dat_opt(mass_flux_z, 1, S3D_000_P200, "double", OPS_RW, fields[FIELD_MASS_FLUX_Z]),
               ops_arg_gbl(fields, NUM_FIELDS, "int", OPS_READ));

  int rangexy_r2e[] = {x_max+1,x_max+2,y_min-depth,y_max+depth,z_min-depth,z_max+1+depth};
  if(depth ==2)
  ops_par_loop(update_halo_kernel5_plus_4_right, "update_halo_kernel5_plus_4_right", clover_grid, 3, rangexy_r2e,
               ops_arg_dat_opt(vol_flux_z, 1, S3D_000_M400, "double", OPS_RW, fields[FIELD_VOL_FLUX_Z]),
               ops_arg_dat_opt(mass_flux_z, 1, S3D_000_M400, "double", OPS_RW, fields[FIELD_MASS_FLUX_Z]),
               ops_arg_gbl(fields, NUM_FIELDS, "int", OPS_READ));

  int rangexy_r1e[] = {x_max,x_max+1,y_min-depth,y_max+depth,z_min-depth,z_max+1+depth};
  ops_par_loop(update_halo_kernel5_plus_2_right, "update_halo_kernel5_plus_2_right",clover_grid, 3, rangexy_r1e,
               ops_arg_dat_opt(vol_flux_z, 1, S3D_000_M200, "double", OPS_RW, fields[FIELD_VOL_FLUX_Z]),
               ops_arg_dat_opt(mass_flux_z, 1, S3D_000_M200, "double", OPS_RW, fields[FIELD_MASS_FLUX_Z]),
               ops_arg_gbl(fields, NUM_FIELDS, "int", OPS_READ));

   if(depth ==2) //TODO: is this really the same range? x should have +1
   ops_par_loop(update_halo_kernel5_minus_4_back, "update_halo_kernel5_minus_4_back", clover_grid, 3, rangexy_back2c,
               ops_arg_dat_opt(vol_flux_z, 1, S3D_000_00P4, "double", OPS_RW, fields[FIELD_VOL_FLUX_Z]),
               ops_arg_dat_opt(mass_flux_z, 1, S3D_000_00P4, "double", OPS_RW, fields[FIELD_MASS_FLUX_Z]),
               ops_arg_gbl(fields, NUM_FIELDS, "int", OPS_READ));

   ops_par_loop(update_halo_kernel5_minus_2_back, "update_halo_kernel5_minus_2_back", clover_grid, 3, rangexy_back1c,
               ops_arg_dat_opt(vol_flux_z, 1, S3D_000_00P2, "double", OPS_RW, fields[FIELD_VOL_FLUX_Z]),
               ops_arg_dat_opt(mass_flux_z, 1, S3D_000_00P2, "double", OPS_RW, fields[FIELD_MASS_FLUX_Z]),
               ops_arg_gbl(fields, NUM_FIELDS, "int", OPS_READ));

   int rangexy_front2d[] = {x_min-depth,x_max+depth,y_min-depth,y_max+depth,z_max+2,z_max+3};
   if(depth ==2)
   ops_par_loop(update_halo_kernel5_minus_4_front, "update_halo_kernel5_minus_4_front", clover_grid, 3, rangexy_front2d,
               ops_arg_dat_opt(vol_flux_z, 1, S3D_000_00M4, "double", OPS_RW, fields[FIELD_VOL_FLUX_Z]),
               ops_arg_dat_opt(mass_flux_z, 1, S3D_000_00M4, "double", OPS_RW, fields[FIELD_MASS_FLUX_Z]),
               ops_arg_gbl(fields, NUM_FIELDS, "int", OPS_READ));
   int rangexy_front1d[] = {x_min-depth,x_max+depth,y_min-depth,y_max+depth,z_max+1,z_max+2};
   ops_par_loop(update_halo_kernel5_minus_2_front, "update_halo_kernel5_minus_2_front", clover_grid, 3, rangexy_front1d,
               ops_arg_dat_opt(vol_flux_z, 1, S3D_000_00M2, "double", OPS_RW, fields[FIELD_VOL_FLUX_Z]),
               ops_arg_dat_opt(mass_flux_z, 1, S3D_000_00M2, "double", OPS_RW, fields[FIELD_MASS_FLUX_Z]),
               ops_arg_gbl(fields, NUM_FIELDS, "int", OPS_READ));
}
