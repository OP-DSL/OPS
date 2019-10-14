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
#define OPS_2D
#include "ops_seq_v2.h"

#include "data.h"
#include "definitions.h"

//Cloverleaf kernels
#include  "update_halo_kernel.h"


void update_halo(int* fields, int depth)
{
  //initialize sizes using global values
  int x_min = field.x_min;
  int x_max = field.x_max;
  int y_min = field.y_min;
  int y_max = field.y_max;

  if (fields[FIELD_DENSITY0] || fields[FIELD_DENSITY1] || fields[FIELD_ENERGY0] || fields[FIELD_ENERGY1] ||
      fields[FIELD_PRESSURE] || fields[FIELD_VISCOSITY] || fields[FIELD_SOUNDSPEED]) {
    int rangexy_b2a[] = {x_min-depth,x_max+depth,y_min-2,y_min-1};
    if(depth ==2)
    ops_par_loop(update_halo_kernel1_b2, "update_halo_kernel1", clover_grid, 2, rangexy_b2a,
              ops_arg_dat_opt(density0, 1, S2D_00_0P3, "double", OPS_RW, fields[FIELD_DENSITY0]),
              ops_arg_dat_opt(density1, 1, S2D_00_0P3, "double", OPS_RW, fields[FIELD_DENSITY1]),
              ops_arg_dat_opt(energy0, 1, S2D_00_0P3, "double", OPS_RW, fields[FIELD_ENERGY0]),
              ops_arg_dat_opt(energy1, 1, S2D_00_0P3, "double", OPS_RW, fields[FIELD_ENERGY1]),
              ops_arg_dat_opt(pressure, 1, S2D_00_0P3, "double", OPS_RW, fields[FIELD_PRESSURE]),
              ops_arg_dat_opt(viscosity, 1, S2D_00_0P3, "double", OPS_RW, fields[FIELD_VISCOSITY]),
              ops_arg_dat_opt(soundspeed, 1, S2D_00_0P3, "double", OPS_RW, fields[FIELD_SOUNDSPEED]),
              ops_arg_gbl(fields, NUM_FIELDS, "int", OPS_READ));

    int rangexy_b1a[] = {x_min-depth,x_max+depth,y_min-1,y_min};
    ops_par_loop(update_halo_kernel1_b1, "update_halo_kernel1", clover_grid, 2, rangexy_b1a,
              ops_arg_dat_opt(density0, 1, S2D_00_0P1, "double", OPS_RW, fields[FIELD_DENSITY0]),
              ops_arg_dat_opt(density1, 1, S2D_00_0P1, "double", OPS_RW, fields[FIELD_DENSITY1]),
              ops_arg_dat_opt(energy0, 1, S2D_00_0P1, "double", OPS_RW, fields[FIELD_ENERGY0]),
              ops_arg_dat_opt(energy1, 1, S2D_00_0P1, "double", OPS_RW, fields[FIELD_ENERGY1]),
              ops_arg_dat_opt(pressure, 1, S2D_00_0P1, "double", OPS_RW, fields[FIELD_PRESSURE]),
              ops_arg_dat_opt(viscosity, 1, S2D_00_0P1, "double", OPS_RW, fields[FIELD_VISCOSITY]),
              ops_arg_dat_opt(soundspeed, 1, S2D_00_0P1, "double", OPS_RW, fields[FIELD_SOUNDSPEED]),
              ops_arg_gbl(fields, NUM_FIELDS, "int", OPS_READ));

    int rangexy_t2a[] = {x_min-depth,x_max+depth,y_max+1,y_max+2};
    if(depth ==2)
    ops_par_loop(update_halo_kernel1_t2, "update_halo_kernel1", clover_grid, 2, rangexy_t2a,
              ops_arg_dat_opt(density0, 1, S2D_00_0M3, "double", OPS_RW, fields[FIELD_DENSITY0]),
              ops_arg_dat_opt(density1, 1, S2D_00_0M3, "double", OPS_RW, fields[FIELD_DENSITY1]),
              ops_arg_dat_opt(energy0, 1, S2D_00_0M3, "double", OPS_RW, fields[FIELD_ENERGY0]),
              ops_arg_dat_opt(energy1, 1, S2D_00_0M3, "double", OPS_RW, fields[FIELD_ENERGY1]),
              ops_arg_dat_opt(pressure, 1, S2D_00_0M3, "double", OPS_RW, fields[FIELD_PRESSURE]),
              ops_arg_dat_opt(viscosity, 1, S2D_00_0M3, "double", OPS_RW, fields[FIELD_VISCOSITY]),
              ops_arg_dat_opt(soundspeed, 1, S2D_00_0M3, "double", OPS_RW, fields[FIELD_SOUNDSPEED]),
              ops_arg_gbl(fields, NUM_FIELDS, "int", OPS_READ));

    int rangexy_t1a[] = {x_min-depth,x_max+depth,y_max,y_max+1};
    ops_par_loop(update_halo_kernel1_t1, "update_halo_kernel1", clover_grid, 2, rangexy_t1a,
               ops_arg_dat_opt(density0, 1, S2D_00_0M1, "double", OPS_RW, fields[FIELD_DENSITY0]),
               ops_arg_dat_opt(density1, 1, S2D_00_0M1, "double", OPS_RW, fields[FIELD_DENSITY1]),
               ops_arg_dat_opt(energy0, 1, S2D_00_0M1, "double", OPS_RW, fields[FIELD_ENERGY0]),
               ops_arg_dat_opt(energy1, 1, S2D_00_0M1, "double", OPS_RW, fields[FIELD_ENERGY1]),
               ops_arg_dat_opt(pressure, 1, S2D_00_0M1, "double", OPS_RW, fields[FIELD_PRESSURE]),
               ops_arg_dat_opt(viscosity, 1, S2D_00_0M1, "double", OPS_RW, fields[FIELD_VISCOSITY]),
               ops_arg_dat_opt(soundspeed, 1, S2D_00_0M1, "double", OPS_RW, fields[FIELD_SOUNDSPEED]),
               ops_arg_gbl(fields, NUM_FIELDS, "int", OPS_READ));

    int rangexy_l2a[] = {x_min-2,x_min-1,y_min-depth,y_max+depth};
    if(depth ==2)
    ops_par_loop(update_halo_kernel1_l2, "update_halo_kernel", clover_grid, 2, rangexy_l2a,
               ops_arg_dat_opt(density0, 1, S2D_00_P30, "double", OPS_RW, fields[FIELD_DENSITY0]),
               ops_arg_dat_opt(density1, 1, S2D_00_P30, "double", OPS_RW, fields[FIELD_DENSITY1]),
               ops_arg_dat_opt(energy0, 1, S2D_00_P30, "double", OPS_RW, fields[FIELD_ENERGY0]),
               ops_arg_dat_opt(energy1, 1, S2D_00_P30, "double", OPS_RW, fields[FIELD_ENERGY1]),
               ops_arg_dat_opt(pressure, 1, S2D_00_P30, "double", OPS_RW, fields[FIELD_PRESSURE]),
               ops_arg_dat_opt(viscosity, 1, S2D_00_P30, "double", OPS_RW, fields[FIELD_VISCOSITY]),
               ops_arg_dat_opt(soundspeed, 1, S2D_00_P30, "double", OPS_RW, fields[FIELD_SOUNDSPEED]),
               ops_arg_gbl(fields, NUM_FIELDS, "int", OPS_READ));

    int rangexy_l1a[] = {x_min-1,x_min,y_min-depth,y_max+depth};
    ops_par_loop(update_halo_kernel1_l1, "update_halo_kernel", clover_grid, 2, rangexy_l1a,
               ops_arg_dat_opt(density0, 1, S2D_00_P10, "double", OPS_RW, fields[FIELD_DENSITY0]),
               ops_arg_dat_opt(density1, 1, S2D_00_P10, "double", OPS_RW, fields[FIELD_DENSITY1]),
               ops_arg_dat_opt(energy0, 1, S2D_00_P10, "double", OPS_RW, fields[FIELD_ENERGY0]),
               ops_arg_dat_opt(energy1, 1, S2D_00_P10, "double", OPS_RW, fields[FIELD_ENERGY1]),
               ops_arg_dat_opt(pressure, 1, S2D_00_P10, "double", OPS_RW, fields[FIELD_PRESSURE]),
               ops_arg_dat_opt(viscosity, 1, S2D_00_P10, "double", OPS_RW, fields[FIELD_VISCOSITY]),
               ops_arg_dat_opt(soundspeed, 1, S2D_00_P10, "double", OPS_RW, fields[FIELD_SOUNDSPEED]),
               ops_arg_gbl(fields, NUM_FIELDS, "int", OPS_READ));

    int rangexy_r2a[] = {x_max+1,x_max+2,y_min-depth,y_max+depth};

    if(depth ==2)
    ops_par_loop(update_halo_kernel1_r2, "update_halo_kernel", clover_grid, 2, rangexy_r2a,
               ops_arg_dat_opt(density0, 1, S2D_00_M30, "double", OPS_RW, fields[FIELD_DENSITY0]),
               ops_arg_dat_opt(density1, 1, S2D_00_M30, "double", OPS_RW, fields[FIELD_DENSITY1]),
               ops_arg_dat_opt(energy0, 1, S2D_00_M30, "double", OPS_RW, fields[FIELD_ENERGY0]),
               ops_arg_dat_opt(energy1, 1, S2D_00_M30, "double", OPS_RW, fields[FIELD_ENERGY1]),
               ops_arg_dat_opt(pressure, 1, S2D_00_M30, "double", OPS_RW, fields[FIELD_PRESSURE]),
               ops_arg_dat_opt(viscosity, 1, S2D_00_M30, "double", OPS_RW, fields[FIELD_VISCOSITY]),
               ops_arg_dat_opt(soundspeed, 1, S2D_00_M30, "double", OPS_RW, fields[FIELD_SOUNDSPEED]),
               ops_arg_gbl(fields, NUM_FIELDS, "int", OPS_READ));

    int rangexy_r1a[] = {x_max,x_max+1,y_min-depth,y_max+depth};
    ops_par_loop(update_halo_kernel1_r1, "update_halo_kernel", clover_grid, 2, rangexy_r1a,
               ops_arg_dat_opt(density0, 1, S2D_00_M10, "double", OPS_RW, fields[FIELD_DENSITY0]),
               ops_arg_dat_opt(density1, 1, S2D_00_M10, "double", OPS_RW, fields[FIELD_DENSITY1]),
               ops_arg_dat_opt(energy0, 1, S2D_00_M10, "double", OPS_RW, fields[FIELD_ENERGY0]),
               ops_arg_dat_opt(energy1, 1, S2D_00_M10, "double", OPS_RW, fields[FIELD_ENERGY1]),
               ops_arg_dat_opt(pressure, 1, S2D_00_M10, "double", OPS_RW, fields[FIELD_PRESSURE]),
               ops_arg_dat_opt(viscosity, 1, S2D_00_M10, "double", OPS_RW, fields[FIELD_VISCOSITY]),
               ops_arg_dat_opt(soundspeed, 1, S2D_00_M10, "double", OPS_RW, fields[FIELD_SOUNDSPEED]),
               ops_arg_gbl(fields, NUM_FIELDS, "int", OPS_READ));


  }

  //
  //xvel0, xvel1 has the same boundary ranges and assignment
  //

  if (fields[FIELD_XVEL0] || fields[FIELD_XVEL1]) {
    int rangexy_b2b[] = {x_min-depth,x_max+1+depth,y_min-2,y_min-1};
    if(depth == 2)
    ops_par_loop(update_halo_kernel2_xvel_plus_4_a, "update_halo_kernel2_xvel_plus_4_a", clover_grid, 2, rangexy_b2b,
                 ops_arg_dat_opt(xvel0, 1, S2D_00_0P4, "double", OPS_RW, fields[FIELD_XVEL0]),
                 ops_arg_dat_opt(xvel1, 1, S2D_00_0P4, "double", OPS_RW, fields[FIELD_XVEL1]),
                 ops_arg_gbl(fields, NUM_FIELDS, "int", OPS_READ));

    int rangexy_b1b[] = {x_min-depth,x_max+1+depth,y_min-1,y_min};
    ops_par_loop(update_halo_kernel2_xvel_plus_2_a, "update_halo_kernel2_xvel_plus_2_a", clover_grid, 2, rangexy_b1b,
                 ops_arg_dat_opt(xvel0, 1, S2D_00_0P2, "double", OPS_RW, fields[FIELD_XVEL0]),
                 ops_arg_dat_opt(xvel1, 1, S2D_00_0P2, "double", OPS_RW, fields[FIELD_XVEL1]),
                 ops_arg_gbl(fields, NUM_FIELDS, "int", OPS_READ));

    //int rangexy_t2b[] = {x_min-depth,x_max+1+depth,y_max+1,y_max+2};
    int rangexy_t2b[] = {x_min-depth,x_max+1+depth,y_max+2,y_max+3};
    if(depth == 2)
    ops_par_loop(update_halo_kernel2_xvel_plus_4_b, "update_halo_kernel2_xvel_plus_4_b", clover_grid, 2, rangexy_t2b,
                 ops_arg_dat_opt(xvel0, 1, S2D_00_0M4, "double", OPS_RW, fields[FIELD_XVEL0]),
                 ops_arg_dat_opt(xvel1, 1, S2D_00_0M4, "double", OPS_RW, fields[FIELD_XVEL1]),
                 ops_arg_gbl(fields, NUM_FIELDS, "int", OPS_READ));

    //int rangexy_t1b[] = {x_min-depth,x_max+1+depth,y_max,y_max+1};
    int rangexy_t1b[] = {x_min-depth,x_max+1+depth,y_max+1,y_max+2};
    ops_par_loop(update_halo_kernel2_xvel_plus_2_b, "update_halo_kernel2_xvel_plus_2_b", clover_grid, 2, rangexy_t1b,
                 ops_arg_dat_opt(xvel0, 1, S2D_00_0M2, "double", OPS_RW, fields[FIELD_XVEL0]),
                 ops_arg_dat_opt(xvel1, 1, S2D_00_0M2, "double", OPS_RW, fields[FIELD_XVEL1]),
                 ops_arg_gbl(fields, NUM_FIELDS, "int", OPS_READ));

    int rangexy_l2b[] = {x_min-2,x_min-1,y_min-depth,y_max+1+depth};
    if(depth ==2)
    ops_par_loop(update_halo_kernel2_xvel_minus_4_a, "update_halo_kernel2_xvel_minus_4_a", clover_grid, 2, rangexy_l2b,
                 ops_arg_dat_opt(xvel0, 1, S2D_00_P40, "double", OPS_RW, fields[FIELD_XVEL0]),
                 ops_arg_dat_opt(xvel1, 1, S2D_00_P40, "double", OPS_RW, fields[FIELD_XVEL1]),
                 ops_arg_gbl(fields, NUM_FIELDS, "int", OPS_READ));

    int rangexy_l1b[] = {x_min-1,x_min,y_min-depth,y_max+1+depth};
    ops_par_loop(update_halo_kernel2_xvel_minus_2_a, "update_halo_kernel2_xvel_minus_2_a", clover_grid, 2, rangexy_l1b,
                 ops_arg_dat_opt(xvel0, 1, S2D_00_P20, "double", OPS_RW, fields[FIELD_XVEL0]),
                 ops_arg_dat_opt(xvel1, 1, S2D_00_P20, "double", OPS_RW, fields[FIELD_XVEL1]),
                 ops_arg_gbl(fields, NUM_FIELDS, "int", OPS_READ));

    //int rangexy_r2b[] = {x_max+1,x_max+2,y_min-depth,y_max+1+depth};
    int rangexy_r2b[] = {x_max+2,x_max+3,y_min-depth,y_max+1+depth};
    if(depth ==2)
    ops_par_loop(update_halo_kernel2_xvel_minus_4_b, "update_halo_kernel2_xvel_minus_4_b", clover_grid, 2, rangexy_r2b,
                 ops_arg_dat_opt(xvel0, 1, S2D_00_M40, "double", OPS_RW, fields[FIELD_XVEL0]),
                 ops_arg_dat_opt(xvel1, 1, S2D_00_M40, "double", OPS_RW, fields[FIELD_XVEL1]),
                 ops_arg_gbl(fields, NUM_FIELDS, "int", OPS_READ));

    //int rangexy_r1b[] = {x_max,x_max+1,y_min-depth,y_max+1+depth};
    int rangexy_r1b[] = {x_max+1,x_max+2,y_min-depth,y_max+1+depth};
    ops_par_loop(update_halo_kernel2_xvel_minus_2_b, "update_halo_kernel2_xvel_minus_2_b", clover_grid, 2, rangexy_r1b,
                 ops_arg_dat_opt(xvel0, 1, S2D_00_M20, "double", OPS_RW, fields[FIELD_XVEL0]),
                 ops_arg_dat_opt(xvel1, 1, S2D_00_M20, "double", OPS_RW, fields[FIELD_XVEL1]),
                 ops_arg_gbl(fields, NUM_FIELDS, "int", OPS_READ));
  }

  //
  //yvel0, yvel1 has the same boundary ranges and assignment
  //

  if (fields[FIELD_YVEL0] || fields[FIELD_YVEL1]) {
    int rangexy_b2b[] = {x_min-depth,x_max+1+depth,y_min-2,y_min-1};
    if(depth == 2)
    ops_par_loop(update_halo_kernel2_yvel_minus_4_a, "update_halo_kernel2_yvel_minus_4_a", clover_grid, 2, rangexy_b2b,
                 ops_arg_dat_opt(yvel0, 1, S2D_00_0P4, "double", OPS_RW, fields[FIELD_YVEL0]),
                 ops_arg_dat_opt(yvel1, 1, S2D_00_0P4, "double", OPS_RW, fields[FIELD_YVEL1]),
                 ops_arg_gbl(fields, NUM_FIELDS, "int", OPS_READ));

    int rangexy_b1b[] = {x_min-depth,x_max+1+depth,y_min-1,y_min};
    ops_par_loop(update_halo_kernel2_yvel_minus_2_a, "update_halo_kernel2_yvel_minus_2_a", clover_grid, 2, rangexy_b1b,
                 ops_arg_dat_opt(yvel0, 1, S2D_00_0P2, "double", OPS_RW, fields[FIELD_YVEL0]),
                 ops_arg_dat_opt(yvel1, 1, S2D_00_0P2, "double", OPS_RW, fields[FIELD_YVEL1]),
                 ops_arg_gbl(fields, NUM_FIELDS, "int", OPS_READ));

    int rangexy_t2b[] = {x_min-depth,x_max+1+depth,y_max+2,y_max+3};
    if(depth == 2)
    ops_par_loop(update_halo_kernel2_yvel_minus_4_b, "update_halo_kernel2_yvel_minus_4_b", clover_grid, 2, rangexy_t2b,
                 ops_arg_dat_opt(yvel0, 1, S2D_00_0M4, "double", OPS_RW, fields[FIELD_YVEL0]),
                 ops_arg_dat_opt(yvel1, 1, S2D_00_0M4, "double", OPS_RW, fields[FIELD_YVEL1]),
                 ops_arg_gbl(fields, NUM_FIELDS, "int", OPS_READ));

    int rangexy_t1b[] = {x_min-depth,x_max+1+depth,y_max+1,y_max+2};
    ops_par_loop(update_halo_kernel2_yvel_minus_2_b, "update_halo_kernel2_yvel_minus_2_b", clover_grid, 2, rangexy_t1b,
                 ops_arg_dat_opt(yvel0, 1, S2D_00_0M2, "double", OPS_RW, fields[FIELD_YVEL0]),
                 ops_arg_dat_opt(yvel1, 1, S2D_00_0M2, "double", OPS_RW, fields[FIELD_YVEL1]),
                 ops_arg_gbl(fields, NUM_FIELDS, "int", OPS_READ));

    int rangexy_l2b[] = {x_min-2,x_min-1,y_min-depth,y_max+1+depth};
    if(depth ==2)
    ops_par_loop(update_halo_kernel2_yvel_plus_4_a, "update_halo_kernel2_yvel_plus_4_a", clover_grid, 2, rangexy_l2b,
                 ops_arg_dat_opt(yvel0, 1, S2D_00_P40, "double", OPS_RW, fields[FIELD_YVEL0]),
                 ops_arg_dat_opt(yvel1, 1, S2D_00_P40, "double", OPS_RW, fields[FIELD_YVEL1]),
                 ops_arg_gbl(fields, NUM_FIELDS, "int", OPS_READ));

    int rangexy_l1b[] = {x_min-1,x_min,y_min-depth,y_max+1+depth};
    ops_par_loop(update_halo_kernel2_yvel_plus_2_a, "update_halo_kernel2_yvel_plus_2_a", clover_grid, 2, rangexy_l1b,
                 ops_arg_dat_opt(yvel0, 1, S2D_00_P20, "double", OPS_RW, fields[FIELD_YVEL0]),
                 ops_arg_dat_opt(yvel1, 1, S2D_00_P20, "double", OPS_RW, fields[FIELD_YVEL1]),
                 ops_arg_gbl(fields, NUM_FIELDS, "int", OPS_READ));

    int rangexy_r2b[] = {x_max+2,x_max+3,y_min-depth,y_max+1+depth};
    if(depth ==2)
    ops_par_loop(update_halo_kernel2_yvel_plus_4_b, "update_halo_kernel2_yvel_plus_4_b", clover_grid, 2, rangexy_r2b,
                 ops_arg_dat_opt(yvel0, 1, S2D_00_M40, "double", OPS_RW, fields[FIELD_YVEL0]),
                 ops_arg_dat_opt(yvel1, 1, S2D_00_M40, "double", OPS_RW, fields[FIELD_YVEL1]),
                 ops_arg_gbl(fields, NUM_FIELDS, "int", OPS_READ));


    int rangexy_r1b[] = {x_max+1,x_max+2,y_min-depth,y_max+1+depth};
    ops_par_loop(update_halo_kernel2_yvel_plus_2_b, "update_halo_kernel2_yvel_plus_2_b", clover_grid, 2, rangexy_r1b,
                 ops_arg_dat_opt(yvel0, 1, S2D_00_M20, "double", OPS_RW, fields[FIELD_YVEL0]),
                 ops_arg_dat_opt(yvel1, 1, S2D_00_M20, "double", OPS_RW, fields[FIELD_YVEL1]),
                 ops_arg_gbl(fields, NUM_FIELDS, "int", OPS_READ));

  }
  //
  //vol_flux_x, mass_flux_x has the same ranges
  //
  if (fields[FIELD_MASS_FLUX_X] || fields[FIELD_VOL_FLUX_X]) {
    int rangexy_b2c[] = {x_min-depth,x_max+1+depth,y_min-2,y_min-1};
    if(depth ==2)
    ops_par_loop(update_halo_kernel3_plus_4_a, "update_halo_kernel3_plus_4_a", clover_grid, 2, rangexy_b2c,
                 ops_arg_dat_opt(vol_flux_x, 1, S2D_00_0P4, "double", OPS_RW, fields[FIELD_VOL_FLUX_X]),
                 ops_arg_dat_opt(mass_flux_x, 1, S2D_00_0P4, "double", OPS_RW, fields[FIELD_MASS_FLUX_X]),
                 ops_arg_gbl(fields, NUM_FIELDS, "int", OPS_READ));

    int rangexy_b1c[] = {x_min-depth,x_max+1+depth,y_min-1,y_min};
    ops_par_loop(update_halo_kernel3_plus_2_a, "update_halo_kernel3_plus_2_a", clover_grid, 2, rangexy_b1c,
                 ops_arg_dat_opt(vol_flux_x, 1, S2D_00_0P2, "double", OPS_RW, fields[FIELD_VOL_FLUX_X]),
                 ops_arg_dat_opt(mass_flux_x, 1, S2D_00_0P2, "double", OPS_RW, fields[FIELD_MASS_FLUX_X]),
                 ops_arg_gbl(fields, NUM_FIELDS, "int", OPS_READ));


    int rangexy_t2c[] = {x_min-depth,x_max+1+depth,y_max+1,y_max+2};
    if(depth ==2)
    ops_par_loop(update_halo_kernel3_plus_4_b, "update_halo_kernel3_plus_4_b", clover_grid, 2, rangexy_t2c,
                 ops_arg_dat_opt(vol_flux_x, 1, S2D_00_0M4, "double", OPS_RW, fields[FIELD_VOL_FLUX_X]),
                 ops_arg_dat_opt(mass_flux_x, 1, S2D_00_0M4, "double", OPS_RW, fields[FIELD_MASS_FLUX_X]),
                 ops_arg_gbl(fields, NUM_FIELDS, "int", OPS_READ));

    int rangexy_t1c[] = {x_min-depth,x_max+1+depth,y_max,y_max+1};
    ops_par_loop(update_halo_kernel3_plus_2_b, "update_halo_kernel3_plus_2_b", clover_grid, 2, rangexy_t1c,
                 ops_arg_dat_opt(vol_flux_x, 1, S2D_00_0M2, "double", OPS_RW, fields[FIELD_VOL_FLUX_X]),
                 ops_arg_dat_opt(mass_flux_x, 1, S2D_00_0M2, "double", OPS_RW, fields[FIELD_MASS_FLUX_X]),
                 ops_arg_gbl(fields, NUM_FIELDS, "int", OPS_READ));


    int rangexy_l2c[] = {x_min-2,x_min-1,y_min-depth,y_max+depth};
    if(depth ==2)
    ops_par_loop(update_halo_kernel3_minus_4_a, "update_halo_kernel3_minus_4_a", clover_grid, 2, rangexy_l2c,
                 ops_arg_dat_opt(vol_flux_x, 1, S2D_00_P40, "double", OPS_RW, fields[FIELD_VOL_FLUX_X]),
                 ops_arg_dat_opt(mass_flux_x, 1, S2D_00_P40, "double", OPS_RW, fields[FIELD_MASS_FLUX_X]),
                 ops_arg_gbl(fields, NUM_FIELDS, "int", OPS_READ));

    int rangexy_l1c[] = {x_min-1,x_min,y_min-depth,y_max+depth};
    ops_par_loop(update_halo_kernel3_minus_2_a, "update_halo_kernel3_minus_2_a", clover_grid, 2, rangexy_l1c,
                 ops_arg_dat_opt(vol_flux_x, 1, S2D_00_P20, "double", OPS_RW, fields[FIELD_VOL_FLUX_X]),
                 ops_arg_dat_opt(mass_flux_x, 1, S2D_00_P20, "double", OPS_RW, fields[FIELD_MASS_FLUX_X]),
                 ops_arg_gbl(fields, NUM_FIELDS, "int", OPS_READ));

    int rangexy_r2c[] = {x_max+2,x_max+3,y_min-depth,y_max+depth}; //
    if(depth ==2)
    ops_par_loop(update_halo_kernel3_minus_4_b, "update_halo_kernel3_minus_4_b", clover_grid, 2, rangexy_r2c,
                 ops_arg_dat_opt(vol_flux_x, 1, S2D_00_M40, "double", OPS_RW, fields[FIELD_VOL_FLUX_X]),
                 ops_arg_dat_opt(mass_flux_x, 1, S2D_00_M40, "double", OPS_RW, fields[FIELD_MASS_FLUX_X]),
                 ops_arg_gbl(fields, NUM_FIELDS, "int", OPS_READ));

    int rangexy_r1c[] = {x_max+1,x_max+2,y_min-depth,y_max+depth}; //
    ops_par_loop(update_halo_kernel3_minus_2_b, "update_halo_kernel3_minus_2_b", clover_grid, 2, rangexy_r1c,
                 ops_arg_dat_opt(vol_flux_x, 1, S2D_00_M20, "double", OPS_RW, fields[FIELD_VOL_FLUX_X]),
                 ops_arg_dat_opt(mass_flux_x, 1, S2D_00_M20, "double", OPS_RW, fields[FIELD_MASS_FLUX_X]),
                 ops_arg_gbl(fields, NUM_FIELDS, "int", OPS_READ));
  }

  //
  //vol_flux_y, mass_flux_y has the same ranges
  //

  if (fields[FIELD_MASS_FLUX_Y] || fields[FIELD_VOL_FLUX_Y]) {
    int rangexy_b2d[] = {x_min-depth,x_max+depth,y_min-2,y_min-1};
    if(depth ==2)
    ops_par_loop(update_halo_kernel4_minus_4_a, "update_halo_kernel4_minus_4_a", clover_grid, 2, rangexy_b2d,
                 ops_arg_dat_opt(vol_flux_y, 1, S2D_00_0P4, "double", OPS_RW, fields[FIELD_VOL_FLUX_Y]),
                 ops_arg_dat_opt(mass_flux_y, 1, S2D_00_0P4, "double", OPS_RW, fields[FIELD_MASS_FLUX_Y]),
                 ops_arg_gbl(fields, NUM_FIELDS, "int", OPS_READ));

    int rangexy_b1d[] = {x_min-depth,x_max+depth,y_min-1,y_min};
    ops_par_loop(update_halo_kernel4_minus_2_a, "update_halo_kernel4_minus_2_a", clover_grid, 2, rangexy_b1d,
                 ops_arg_dat_opt(vol_flux_y, 1, S2D_00_0P2, "double", OPS_RW, fields[FIELD_VOL_FLUX_Y]),
                 ops_arg_dat_opt(mass_flux_y, 1, S2D_00_0P2, "double", OPS_RW, fields[FIELD_MASS_FLUX_Y]),
                 ops_arg_gbl(fields, NUM_FIELDS, "int", OPS_READ));

    int rangexy_t2d[] = {x_min-depth,x_max+depth,y_max+2,y_max+3}; //
    if(depth ==2)
    ops_par_loop(update_halo_kernel4_minus_4_b, "update_halo_kernel4_minus_4_b", clover_grid, 2, rangexy_t2d,
                 ops_arg_dat_opt(vol_flux_y, 1, S2D_00_0M4, "double", OPS_RW, fields[FIELD_VOL_FLUX_Y]),
                 ops_arg_dat_opt(mass_flux_y, 1, S2D_00_0M4, "double", OPS_RW, fields[FIELD_MASS_FLUX_Y]),
                 ops_arg_gbl(fields, NUM_FIELDS, "int", OPS_READ));

    int rangexy_t1d[] = {x_min-depth,x_max+depth,y_max+1,y_max+2}; //
    ops_par_loop(update_halo_kernel4_minus_2_b, "update_halo_kernel4_minus_2_b", clover_grid, 2, rangexy_t1d,
                 ops_arg_dat_opt(vol_flux_y, 1, S2D_00_0M2, "double", OPS_RW, fields[FIELD_VOL_FLUX_Y]),
                 ops_arg_dat_opt(mass_flux_y, 1, S2D_00_0M2, "double", OPS_RW, fields[FIELD_MASS_FLUX_Y]),
                 ops_arg_gbl(fields, NUM_FIELDS, "int", OPS_READ));

    int rangexy_l2d[] = {x_min-2,x_min-1,y_min-depth,y_max+1+depth};
    if(depth ==2)
    ops_par_loop(update_halo_kernel4_plus_4_a, "update_halo_kernel4_plus_4_a", clover_grid, 2, rangexy_l2d,
                 ops_arg_dat_opt(vol_flux_y, 1, S2D_00_P40, "double", OPS_RW, fields[FIELD_VOL_FLUX_Y]),
                 ops_arg_dat_opt(mass_flux_y, 1, S2D_00_P40, "double", OPS_RW, fields[FIELD_MASS_FLUX_Y]),
                 ops_arg_gbl(fields, NUM_FIELDS, "int", OPS_READ));

    int rangexy_l1d[] = {x_min-1,x_min,y_min-depth,y_max+1+depth};
    ops_par_loop(update_halo_kernel4_plus_2_a, "update_halo_kernel4_plus_2_a", clover_grid, 2, rangexy_l1d,
                 ops_arg_dat_opt(vol_flux_y, 1, S2D_00_P20, "double", OPS_RW, fields[FIELD_VOL_FLUX_Y]),
                 ops_arg_dat_opt(mass_flux_y, 1, S2D_00_P20, "double", OPS_RW, fields[FIELD_MASS_FLUX_Y]),
                 ops_arg_gbl(fields, NUM_FIELDS, "int", OPS_READ));

    int rangexy_r2d[] = {x_max+1,x_max+2,y_min-depth,y_max+1+depth};
    if(depth ==2)
    ops_par_loop(update_halo_kernel4_plus_4_b, "update_halo_kernel4_plus_4_b", clover_grid, 2, rangexy_r2d,
                 ops_arg_dat_opt(vol_flux_y, 1, S2D_00_M40, "double", OPS_RW, fields[FIELD_VOL_FLUX_Y]),
                 ops_arg_dat_opt(mass_flux_y, 1, S2D_00_M40, "double", OPS_RW, fields[FIELD_MASS_FLUX_Y]),
                 ops_arg_gbl(fields, NUM_FIELDS, "int", OPS_READ));

    int rangexy_r1d[] = {x_max,x_max+1,y_min-depth,y_max+1+depth};
    ops_par_loop(update_halo_kernel4_plus_2_b, "update_halo_kernel4_plus_2_b",clover_grid, 2, rangexy_r1d,
                 ops_arg_dat_opt(vol_flux_y, 1, S2D_00_M20, "double", OPS_RW, fields[FIELD_VOL_FLUX_Y]),
                 ops_arg_dat_opt(mass_flux_y, 1, S2D_00_M20, "double", OPS_RW, fields[FIELD_MASS_FLUX_Y]),
                 ops_arg_gbl(fields, NUM_FIELDS, "int", OPS_READ));
  }

}
