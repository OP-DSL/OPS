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
#include "ops_seq_opt.h"
#include "ops_seq_macro.h"

#include "data.h"
#include "definitions.h"

//Cloverleaf kernels
#include  "update_halo_kernel.h"

void update_halo_kernel1_b2_macro(double *density0, double *density1,
                          double *energy0, double *energy1,
                          double *pressure, double *viscosity,
                          double *soundspeed ) {
  if(fields[FIELD_DENSITY0] == 1) density0[OPS_ACC0(0,0)] = density0[OPS_ACC0(0,3)];
  if(fields[FIELD_DENSITY1] == 1) density1[OPS_ACC1(0,0)] = density1[OPS_ACC0(0,3)];
  if(fields[FIELD_ENERGY0] == 1) energy0[OPS_ACC2(0,0)] = energy0[OPS_ACC0(0,3)];
  if(fields[FIELD_ENERGY1] == 1) energy1[OPS_ACC3(0,0)] = energy1[OPS_ACC0(0,3)];
  if(fields[FIELD_PRESSURE] == 1) pressure[OPS_ACC4(0,0)] = pressure[OPS_ACC0(0,3)];
  if(fields[FIELD_VISCOSITY] == 1) viscosity[OPS_ACC5(0,0)] = viscosity[OPS_ACC0(0,3)];
  if(fields[FIELD_SOUNDSPEED] == 1) soundspeed[OPS_ACC6(0,0)] = soundspeed[OPS_ACC0(0,3)];

}

void update_halo_kernel_b1_macro(double *density0, double *density1,
                          double *energy0, double *energy1,
                          double *pressure, double *viscosity,
                          double *soundspeed ) {
  if(fields[FIELD_DENSITY0] == 1) density0[OPS_ACC0(0,0)] = density0[OPS_ACC0(0,1)];
  if(fields[FIELD_DENSITY1] == 1) density1[OPS_ACC1(0,0)] = density1[OPS_ACC0(0,1)];
  if(fields[FIELD_ENERGY0] == 1) energy0[OPS_ACC2(0,0)] = energy0[OPS_ACC0(0,1)];
  if(fields[FIELD_ENERGY1] == 1) energy1[OPS_ACC3(0,0)] = energy1[OPS_ACC0(0,1)];
  if(fields[FIELD_PRESSURE] == 1) pressure[OPS_ACC4(0,0)] = pressure[OPS_ACC0(0,1)];
  if(fields[FIELD_VISCOSITY] == 1) viscosity[OPS_ACC5(0,0)] = viscosity[OPS_ACC0(0,1)];
  if(fields[FIELD_SOUNDSPEED] == 1) soundspeed[OPS_ACC6(0,0)] = soundspeed[OPS_ACC0(0,1)];

}

void update_halo_kernel_t2_macro(double *density0, double *density1,
                          double *energy0, double *energy1,
                          double *pressure, double *viscosity,
                          double *soundspeed ) {
  if(fields[FIELD_DENSITY0] == 1) density0[OPS_ACC0(0,0)] = density0[OPS_ACC0(0,-3)];
  if(fields[FIELD_DENSITY1] == 1) density1[OPS_ACC1(0,0)] = density1[OPS_ACC0(0,-3)];
  if(fields[FIELD_ENERGY0] == 1) energy0[OPS_ACC2(0,0)] = energy0[OPS_ACC0(0,-3)];
  if(fields[FIELD_ENERGY1] == 1) energy1[OPS_ACC3(0,0)] = energy1[OPS_ACC0(0,-3)];
  if(fields[FIELD_PRESSURE] == 1) pressure[OPS_ACC4(0,0)] = pressure[OPS_ACC0(0,-3)];
  if(fields[FIELD_VISCOSITY] == 1) viscosity[OPS_ACC5(0,0)] = viscosity[OPS_ACC0(0,-3)];
  if(fields[FIELD_SOUNDSPEED] == 1) soundspeed[OPS_ACC6(0,0)] = soundspeed[OPS_ACC0(0,-3)];

}

void update_halo_kernel_t1_macro(double *density0, double *density1,
                          double *energy0, double *energy1,
                          double *pressure, double *viscosity,
                          double *soundspeed ) {
  if(fields[FIELD_DENSITY0] == 1) density0[OPS_ACC0(0,0)] = density0[OPS_ACC0(0,-1)];
  if(fields[FIELD_DENSITY1] == 1) density1[OPS_ACC1(0,0)] = density1[OPS_ACC0(0,-1)];
  if(fields[FIELD_ENERGY0] == 1) energy0[OPS_ACC2(0,0)] = energy0[OPS_ACC0(0,-1)];
  if(fields[FIELD_ENERGY1] == 1) energy1[OPS_ACC3(0,0)] = energy1[OPS_ACC0(0,-1)];
  if(fields[FIELD_PRESSURE] == 1) pressure[OPS_ACC4(0,0)] = pressure[OPS_ACC0(0,-1)];
  if(fields[FIELD_VISCOSITY] == 1) viscosity[OPS_ACC5(0,0)] = viscosity[OPS_ACC0(0,-1)];
  if(fields[FIELD_SOUNDSPEED] == 1) soundspeed[OPS_ACC6(0,0)] = soundspeed[OPS_ACC0(0,-1)];

}

//////////

void update_halo_kernel1_l2_macro(double *density0, double *density1,
                          double *energy0, double *energy1,
                          double *pressure, double *viscosity,
                          double *soundspeed ) {
  if(fields[FIELD_DENSITY0] == 1) density0[OPS_ACC0(0,0)] = density0[OPS_ACC0(3,0)];
  if(fields[FIELD_DENSITY1] == 1) density1[OPS_ACC1(0,0)] = density1[OPS_ACC0(3,0)];
  if(fields[FIELD_ENERGY0] == 1) energy0[OPS_ACC2(0,0)] = energy0[OPS_ACC0(3,0)];
  if(fields[FIELD_ENERGY1] == 1) energy1[OPS_ACC3(0,0)] = energy1[OPS_ACC0(3,0)];
  if(fields[FIELD_PRESSURE] == 1) pressure[OPS_ACC4(0,0)] = pressure[OPS_ACC0(3,0)];
  if(fields[FIELD_VISCOSITY] == 1) viscosity[OPS_ACC5(0,0)] = viscosity[OPS_ACC0(3,0)];
  if(fields[FIELD_SOUNDSPEED] == 1) soundspeed[OPS_ACC6(0,0)] = soundspeed[OPS_ACC0(3,0)];

}

void update_halo_kernel1_l1_macro(double *density0, double *density1,
                          double *energy0, double *energy1,
                          double *pressure, double *viscosity,
                          double *soundspeed ) {
  if(fields[FIELD_DENSITY0] == 1) density0[OPS_ACC0(0,0)] = density0[OPS_ACC0(1,0)];
  if(fields[FIELD_DENSITY1] == 1) density1[OPS_ACC1(0,0)] = density1[OPS_ACC0(1,0)];
  if(fields[FIELD_ENERGY0] == 1) energy0[OPS_ACC2(0,0)] = energy0[OPS_ACC0(1,0)];
  if(fields[FIELD_ENERGY1] == 1) energy1[OPS_ACC3(0,0)] = energy1[OPS_ACC0(1,0)];
  if(fields[FIELD_PRESSURE] == 1) pressure[OPS_ACC4(0,0)] = pressure[OPS_ACC0(1,0)];
  if(fields[FIELD_VISCOSITY] == 1) viscosity[OPS_ACC5(0,0)] = viscosity[OPS_ACC0(1,0)];
  if(fields[FIELD_SOUNDSPEED] == 1) soundspeed[OPS_ACC6(0,0)] = soundspeed[OPS_ACC0(1,0)];

}

void update_halo_kernel1_r2_macro(double *density0, double *density1,
                          double *energy0, double *energy1,
                          double *pressure, double *viscosity,
                          double *soundspeed ) {
  if(fields[FIELD_DENSITY0] == 1) density0[OPS_ACC0(0,0)] = density0[OPS_ACC0(-3,0)];
  if(fields[FIELD_DENSITY1] == 1) density1[OPS_ACC1(0,0)] = density1[OPS_ACC0(-3,0)];
  if(fields[FIELD_ENERGY0] == 1) energy0[OPS_ACC2(0,0)] = energy0[OPS_ACC0(-3,0)];
  if(fields[FIELD_ENERGY1] == 1) energy1[OPS_ACC3(0,0)] = energy1[OPS_ACC0(-3,0)];
  if(fields[FIELD_PRESSURE] == 1) pressure[OPS_ACC4(0,0)] = pressure[OPS_ACC0(-3,0)];
  if(fields[FIELD_VISCOSITY] == 1) viscosity[OPS_ACC5(0,0)] = viscosity[OPS_ACC0(-3,0)];
  if(fields[FIELD_SOUNDSPEED] == 1) soundspeed[OPS_ACC6(0,0)] = soundspeed[OPS_ACC0(-3,0)];

}

void update_halo_kernel1_r1_macro(double *density0, double *density1,
                          double *energy0, double *energy1,
                          double *pressure, double *viscosity,
                          double *soundspeed ) {
  if(fields[FIELD_DENSITY0] == 1) density0[OPS_ACC0(0,0)] = density0[OPS_ACC0(-1,0)];
  if(fields[FIELD_DENSITY1] == 1) density1[OPS_ACC1(0,0)] = density1[OPS_ACC0(-1,0)];
  if(fields[FIELD_ENERGY0] == 1) energy0[OPS_ACC2(0,0)] = energy0[OPS_ACC0(-1,0)];
  if(fields[FIELD_ENERGY1] == 1) energy1[OPS_ACC3(0,0)] = energy1[OPS_ACC0(-1,0)];
  if(fields[FIELD_PRESSURE] == 1) pressure[OPS_ACC4(0,0)] = pressure[OPS_ACC0(-1,0)];
  if(fields[FIELD_VISCOSITY] == 1) viscosity[OPS_ACC5(0,0)] = viscosity[OPS_ACC0(-1,0)];
  if(fields[FIELD_SOUNDSPEED] == 1) soundspeed[OPS_ACC6(0,0)] = soundspeed[OPS_ACC0(-1,0)];

}
////

void update_halo_kernel2_xvel_plus_4_a_macro(double *xvel0, double *xvel1){
  if(fields[FIELD_XVEL0] == 1) xvel0[OPS_ACC0(0,0)] = xvel0[OPS_ACC0(0,4)];
  if(fields[FIELD_XVEL1] == 1) xvel1[OPS_ACC1(0,0)] = xvel1[OPS_ACC1(0,4)];
}
void update_halo_kernel2_xvel_plus_2_a_macro(double *xvel0, double *xvel1){
  if(fields[FIELD_XVEL0] == 1) xvel0[OPS_ACC0(0,0)] = xvel0[OPS_ACC0(0,2)];
  if(fields[FIELD_XVEL1] == 1) xvel1[OPS_ACC1(0,0)] = xvel1[OPS_ACC1(0,2)];
}

void update_halo_kernel2_xvel_plus_4_b_macro(double *xvel0, double *xvel1){
  if(fields[FIELD_XVEL0] == 1) xvel0[OPS_ACC0(0,0)] = xvel0[OPS_ACC0(0,-4)];
  if(fields[FIELD_XVEL1] == 1) xvel1[OPS_ACC1(0,0)] = xvel1[OPS_ACC1(0,-4)];
}
void update_halo_kernel2_xvel_plus_2_b_macro(double *xvel0, double *xvel1){
  if(fields[FIELD_XVEL0] == 1) xvel0[OPS_ACC0(0,0)] = xvel0[OPS_ACC0(0,-2)];
  if(fields[FIELD_XVEL1] == 1) xvel1[OPS_ACC1(0,0)] = xvel1[OPS_ACC1(0,-2)];
}

///

void update_halo_kernel2_xvel_minus_4_a_macro(double *xvel0, double *xvel1){
  if(fields[FIELD_XVEL0] == 1) xvel0[OPS_ACC0(0,0)] = -xvel0[OPS_ACC0(4,0)];
  if(fields[FIELD_XVEL1] == 1) xvel1[OPS_ACC1(0,0)] = -xvel1[OPS_ACC1(4,0)];
}
void update_halo_kernel2_xvel_minus_2_a_macro(double *xvel0, double *xvel1){
  if(fields[FIELD_XVEL0] == 1) xvel0[OPS_ACC0(0,0)] = -xvel0[OPS_ACC0(2,0)];
  if(fields[FIELD_XVEL1] == 1) xvel1[OPS_ACC1(0,0)] = -xvel1[OPS_ACC1(2,0)];
}

void update_halo_kernel2_xvel_minus_4_b_macro(double *xvel0, double *xvel1){
  if(fields[FIELD_XVEL0] == 1) xvel0[OPS_ACC0(0,0)] = -xvel0[OPS_ACC0(-4,0)];
  if(fields[FIELD_XVEL1] == 1) xvel1[OPS_ACC1(0,0)] = -xvel1[OPS_ACC1(-4,0)];
}
void update_halo_kernel2_xvel_minus_2_b_macro(double *xvel0, double *xvel1){
  if(fields[FIELD_XVEL0] == 1) xvel0[OPS_ACC0(0,0)] = -xvel0[OPS_ACC0(-2,0)];
  if(fields[FIELD_XVEL1] == 1) xvel1[OPS_ACC1(0,0)] = -xvel1[OPS_ACC1(-2,0)];
}


///

void update_halo_kernel2_yvel_plus_4_a_macro(double *yvel0, double *yvel1){
  if(fields[FIELD_XVEL0] == 1) yvel0[OPS_ACC0(0,0)] = yvel0[OPS_ACC0(4,0)];
  if(fields[FIELD_XVEL1] == 1) yvel1[OPS_ACC1(0,0)] = yvel1[OPS_ACC1(4,0)];
}
void update_halo_kernel2_yvel_plus_2_a_macro(double *yvel0, double *yvel1){
  if(fields[FIELD_XVEL0] == 1) yvel0[OPS_ACC0(0,0)] = yvel0[OPS_ACC0(2,0)];
  if(fields[FIELD_XVEL1] == 1) yvel1[OPS_ACC1(0,0)] = yvel1[OPS_ACC1(2,0)];
}

void update_halo_kernel2_yvel_plus_4_b_macro(double *yvel0, double *yvel1){
  if(fields[FIELD_XVEL0] == 1) yvel0[OPS_ACC0(0,0)] = yvel0[OPS_ACC0(-4,0)];
  if(fields[FIELD_XVEL1] == 1) yvel1[OPS_ACC1(0,0)] = yvel1[OPS_ACC1(-4,0)];
}
void update_halo_kernel2_yvel_plus_2_b_macro(double *yvel0, double *yvel1){
  if(fields[FIELD_XVEL0] == 1) yvel0[OPS_ACC0(0,0)] = yvel0[OPS_ACC0(-2,0)];
  if(fields[FIELD_XVEL1] == 1) yvel1[OPS_ACC1(0,0)] = yvel1[OPS_ACC1(-2,0)];
}

///

void update_halo_kernel2_yvel_minus_4_a_macro(double *yvel0, double *yvel1){
  if(fields[FIELD_XVEL0] == 1) yvel0[OPS_ACC0(0,0)] = -yvel0[OPS_ACC0(0,4)];
  if(fields[FIELD_XVEL1] == 1) yvel1[OPS_ACC1(0,0)] = -yvel1[OPS_ACC1(0,4)];
}
void update_halo_kernel2_yvel_minus_2_a_macro(double *yvel0, double *yvel1){
  if(fields[FIELD_XVEL0] == 1) yvel0[OPS_ACC0(0,0)] = -yvel0[OPS_ACC0(0,2)];
  if(fields[FIELD_XVEL1] == 1) yvel1[OPS_ACC1(0,0)] = -yvel1[OPS_ACC1(0,2)];
}

void update_halo_kernel2_yvel_minus_4_b_macro(double *yvel0, double *yvel1){
  if(fields[FIELD_XVEL0] == 1) yvel0[OPS_ACC0(0,0)] = -yvel0[OPS_ACC0(0,-4)];
  if(fields[FIELD_XVEL1] == 1) yvel1[OPS_ACC1(0,0)] = -yvel1[OPS_ACC1(0,-4)];
}
void update_halo_kernel2_yvel_minus_2_b_macro(double *yvel0, double *yvel1){
  if(fields[FIELD_XVEL0] == 1) yvel0[OPS_ACC0(0,0)] = -yvel0[OPS_ACC0(0,-2)];
  if(fields[FIELD_XVEL1] == 1) yvel1[OPS_ACC1(0,0)] = -yvel1[OPS_ACC1(0,-2)];
}


///


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
  ops_par_loop_macro(update_halo_kernel1_b2_macro, "update_halo_kernel1_macro", 2, rangexy_b2a,
              ops_arg_dat(density0, S2D_00_0P3, "double", OPS_READ),
              ops_arg_dat(density1, S2D_00_0P3, "double", OPS_READ),
              ops_arg_dat(energy0, S2D_00_0P3, "double", OPS_READ),
              ops_arg_dat(energy1, S2D_00_0P3, "double", OPS_READ),
              ops_arg_dat(pressure, S2D_00_0P3, "double", OPS_READ),
              ops_arg_dat(viscosity, S2D_00_0P3, "double", OPS_READ),
              ops_arg_dat(soundspeed, S2D_00_0P3, "double", OPS_READ));

  int rangexy_b1a[] = {x_min-depth,x_max+depth,y_min-1,y_min};
  ops_par_loop_macro(update_halo_kernel_b1_macro, "update_halo_kernel1_macro", 2, rangexy_b1a,
              ops_arg_dat(density0, S2D_00_0P1, "double", OPS_READ),
              ops_arg_dat(density1, S2D_00_0P1, "double", OPS_READ),
              ops_arg_dat(energy0, S2D_00_0P1, "double", OPS_READ),
              ops_arg_dat(energy1, S2D_00_0P1, "double", OPS_READ),
              ops_arg_dat(pressure, S2D_00_0P1, "double", OPS_READ),
              ops_arg_dat(viscosity, S2D_00_0P1, "double", OPS_READ),
              ops_arg_dat(soundspeed, S2D_00_0P1, "double", OPS_READ));

  int rangexy_t2a[] = {x_min-depth,x_max+depth,y_max+1,y_max+2};
  if(depth ==2)
  ops_par_loop_macro(update_halo_kernel_t2_macro, "update_halo_kernel1", 2, rangexy_t2a,
              ops_arg_dat(density0, S2D_00_0M3, "double", OPS_READ),
              ops_arg_dat(density1, S2D_00_0M3, "double", OPS_READ),
              ops_arg_dat(energy0, S2D_00_0M3, "double", OPS_READ),
              ops_arg_dat(energy1, S2D_00_0M3, "double", OPS_READ),
              ops_arg_dat(pressure, S2D_00_0M3, "double", OPS_READ),
              ops_arg_dat(viscosity, S2D_00_0M3, "double", OPS_READ),
              ops_arg_dat(soundspeed, S2D_00_0M3, "double", OPS_READ));

  int rangexy_t1a[] = {x_min-depth,x_max+depth,y_max,y_max+1};
  ops_par_loop_macro(update_halo_kernel_t1_macro, "update_halo_kernel1", 2, rangexy_t1a,
              ops_arg_dat(density0, S2D_00_0M1, "double", OPS_READ),
              ops_arg_dat(density1, S2D_00_0M1, "double", OPS_READ),
              ops_arg_dat(energy0, S2D_00_0M1, "double", OPS_READ),
              ops_arg_dat(energy1, S2D_00_0M1, "double", OPS_READ),
              ops_arg_dat(pressure, S2D_00_0M1, "double", OPS_READ),
              ops_arg_dat(viscosity, S2D_00_0M1, "double", OPS_READ),
              ops_arg_dat(soundspeed, S2D_00_0M1, "double", OPS_READ));

  int rangexy_l2a[] = {x_min-2,x_min-1,y_min-depth,y_max+depth};
  if(depth ==2)
  ops_par_loop_macro(update_halo_kernel1_l2_macro, "update_halo_kernel", 2, rangexy_l2a,
              ops_arg_dat(density0, S2D_00_P30, "double", OPS_READ),
              ops_arg_dat(density1, S2D_00_P30, "double", OPS_READ),
              ops_arg_dat(energy0, S2D_00_P30, "double", OPS_READ),
              ops_arg_dat(energy1, S2D_00_P30, "double", OPS_READ),
              ops_arg_dat(pressure, S2D_00_P30, "double", OPS_READ),
              ops_arg_dat(viscosity, S2D_00_P30, "double", OPS_READ),
              ops_arg_dat(soundspeed, S2D_00_P30, "double", OPS_READ));

  int rangexy_l1a[] = {x_min-1,x_min,y_min-depth,y_max+depth};
  ops_par_loop_macro(update_halo_kernel1_l1_macro, "update_halo_kernel", 2, rangexy_l1a,
              ops_arg_dat(density0, S2D_00_P10, "double", OPS_READ),
              ops_arg_dat(density1, S2D_00_P10, "double", OPS_READ),
              ops_arg_dat(energy0, S2D_00_P10, "double", OPS_READ),
              ops_arg_dat(energy1, S2D_00_P10, "double", OPS_READ),
              ops_arg_dat(pressure, S2D_00_P10, "double", OPS_READ),
              ops_arg_dat(viscosity, S2D_00_P10, "double", OPS_READ),
              ops_arg_dat(soundspeed, S2D_00_P10, "double", OPS_READ));

  int rangexy_r2a[] = {x_max+1,x_max+2,y_min-depth,y_max+depth};

  if(depth ==2)
  ops_par_loop_macro(update_halo_kernel1_r2_macro, "update_halo_kernel", 2, rangexy_r2a,
              ops_arg_dat(density0, S2D_00_M30, "double", OPS_READ),
              ops_arg_dat(density1, S2D_00_M30, "double", OPS_READ),
              ops_arg_dat(energy0, S2D_00_M30, "double", OPS_READ),
              ops_arg_dat(energy1, S2D_00_M30, "double", OPS_READ),
              ops_arg_dat(pressure, S2D_00_M30, "double", OPS_READ),
              ops_arg_dat(viscosity, S2D_00_M30, "double", OPS_READ),
              ops_arg_dat(soundspeed, S2D_00_M30, "double", OPS_READ));

  int rangexy_r1a[] = {x_max,x_max+1,y_min-depth,y_max+depth};
  ops_par_loop_macro(update_halo_kernel1_r1_macro, "update_halo_kernel", 2, rangexy_r1a,
              ops_arg_dat(density0, S2D_00_M10, "double", OPS_READ),
              ops_arg_dat(density1, S2D_00_M10, "double", OPS_READ),
              ops_arg_dat(energy0, S2D_00_M10, "double", OPS_READ),
              ops_arg_dat(energy1, S2D_00_M10, "double", OPS_READ),
              ops_arg_dat(pressure, S2D_00_M10, "double", OPS_READ),
              ops_arg_dat(viscosity, S2D_00_M10, "double", OPS_READ),
              ops_arg_dat(soundspeed, S2D_00_M10, "double", OPS_READ));

  //
  //xvel0, xvel1 has the same boundary ranges and assignment
  //


  int rangexy_b2b[] = {x_min-depth,x_max+1+depth,y_min-2,y_min-1};
  if(depth == 2)
  ops_par_loop_macro(update_halo_kernel2_xvel_plus_4_a_macro, "update_halo_kernel2_xvel_plus_4_a_macro", 2, rangexy_b2b,
              ops_arg_dat(xvel0, S2D_00_0P4, "double", OPS_READ),
              ops_arg_dat(xvel1, S2D_00_0P4, "double", OPS_READ));

  int rangexy_b1b[] = {x_min-depth,x_max+1+depth,y_min-1,y_min};
  ops_par_loop_macro(update_halo_kernel2_xvel_plus_2_a_macro, "update_halo_kernel2_xvel_plus_2_a_macro", 2, rangexy_b1b,
              ops_arg_dat(xvel0, S2D_00_0P2, "double", OPS_READ),
              ops_arg_dat(xvel1, S2D_00_0P2, "double", OPS_READ));

  //int rangexy_t2b[] = {x_min-depth,x_max+1+depth,y_max+1,y_max+2};
  int rangexy_t2b[] = {x_min-depth,x_max+1+depth,y_max+2,y_max+3};
  if(depth == 2)
  ops_par_loop_macro(update_halo_kernel2_xvel_plus_4_b_macro, "update_halo_kernel2_xvel_plus_4_b_macro", 2, rangexy_t2b,
              ops_arg_dat(xvel0, S2D_00_0M4, "double", OPS_READ),
              ops_arg_dat(xvel1, S2D_00_0M4, "double", OPS_READ));

  //int rangexy_t1b[] = {x_min-depth,x_max+1+depth,y_max,y_max+1};
  int rangexy_t1b[] = {x_min-depth,x_max+1+depth,y_max+1,y_max+2};
  ops_par_loop_macro(update_halo_kernel2_xvel_plus_2_b_macro, "update_halo_kernel2_xvel_plus_2_b_macro", 2, rangexy_t1b,
              ops_arg_dat(xvel0, S2D_00_0M2, "double", OPS_READ),
              ops_arg_dat(xvel1, S2D_00_0M2, "double", OPS_READ));


  int rangexy_l2b[] = {x_min-2,x_min-1,y_min-depth,y_max+1+depth};
  if(depth ==2)
  ops_par_loop_macro(update_halo_kernel2_xvel_minus_4_a_macro, "update_halo_kernel2_xvel_minus_4_a_macro", 2, rangexy_l2b,
              ops_arg_dat(xvel0, S2D_00_P40, "double", OPS_READ),
              ops_arg_dat(xvel1, S2D_00_P40, "double", OPS_READ));

  int rangexy_l1b[] = {x_min-1,x_min,y_min-depth,y_max+1+depth};
  ops_par_loop_macro(update_halo_kernel2_xvel_minus_2_a_macro, "update_halo_kernel2_xvel_minus_2_a_macro", 2, rangexy_l1b,
              ops_arg_dat(xvel0, S2D_00_P20, "double", OPS_READ),
              ops_arg_dat(xvel1, S2D_00_P20, "double", OPS_READ));


  //int rangexy_r2b[] = {x_max+1,x_max+2,y_min-depth,y_max+1+depth};
  int rangexy_r2b[] = {x_max+2,x_max+3,y_min-depth,y_max+1+depth};
  if(depth ==2)
  ops_par_loop_macro(update_halo_kernel2_xvel_minus_4_b_macro, "update_halo_kernel2_xvel_minus_4_b_macro", 2, rangexy_r2b,
              ops_arg_dat(xvel0, S2D_00_M40, "double", OPS_READ),
              ops_arg_dat(xvel1, S2D_00_M40, "double", OPS_READ));

  //int rangexy_r1b[] = {x_max,x_max+1,y_min-depth,y_max+1+depth};
  int rangexy_r1b[] = {x_max+1,x_max+2,y_min-depth,y_max+1+depth};
  ops_par_loop_macro(update_halo_kernel2_xvel_minus_2_b_macro, "update_halo_kernel2_xvel_minus_2_b_macro", 2, rangexy_r1b,
              ops_arg_dat(xvel0, S2D_00_M20, "double", OPS_READ),
              ops_arg_dat(xvel1, S2D_00_M20, "double", OPS_READ));


  //
  //yvel0, yvel1 has the same boundary ranges and assignment
  //

  if(depth == 2)
  ops_par_loop_macro(update_halo_kernel2_yvel_minus_4_a_macro, "update_halo_kernel2_yvel_minus_4_a_macro", 2, rangexy_b2b,
              ops_arg_dat(yvel0, S2D_00_0P4, "double", OPS_READ),
              ops_arg_dat(yvel1, S2D_00_0P4, "double", OPS_READ));

  ops_par_loop_macro(update_halo_kernel2_yvel_minus_2_a_macro, "update_halo_kernel2_yvel_minus_2_a_macro", 2, rangexy_b1b,
              ops_arg_dat(yvel0, S2D_00_0P2, "double", OPS_READ),
              ops_arg_dat(yvel1, S2D_00_0P2, "double", OPS_READ));

  if(depth == 2)
  ops_par_loop_macro(update_halo_kernel2_yvel_minus_4_b_macro, "update_halo_kernel2_yvel_minus_4_b_macro", 2, rangexy_t2b,
              ops_arg_dat(yvel0, S2D_00_0M4, "double", OPS_READ),
              ops_arg_dat(yvel1, S2D_00_0M4, "double", OPS_READ));

  ops_par_loop_macro(update_halo_kernel2_yvel_minus_2_b_macro, "update_halo_kernel2_yvel_minus_2_b_macro", 2, rangexy_t1b,
              ops_arg_dat(yvel0, S2D_00_0M2, "double", OPS_READ),
              ops_arg_dat(yvel1, S2D_00_0M2, "double", OPS_READ));

  if(depth ==2)
  ops_par_loop_macro(update_halo_kernel2_yvel_plus_4_a_macro, "update_halo_kernel2_yvel_plus_4_a_macro", 2, rangexy_l2b,
              ops_arg_dat(yvel0, S2D_00_P40, "double", OPS_READ),
              ops_arg_dat(yvel1, S2D_00_P40, "double", OPS_READ));

  ops_par_loop_macro(update_halo_kernel2_yvel_plus_2_a_macro, "update_halo_kernel2_yvel_plus_2_a_macro", 2, rangexy_l1b,
              ops_arg_dat(yvel0, S2D_00_P20, "double", OPS_READ),
              ops_arg_dat(yvel1, S2D_00_P20, "double", OPS_READ));

  if(depth ==2)
  ops_par_loop_macro(update_halo_kernel2_yvel_plus_4_b_macro, "update_halo_kernel2_yvel_plus_4_b_macro", 2, rangexy_r2b,
              ops_arg_dat(yvel0, S2D_00_M40, "double", OPS_READ),
              ops_arg_dat(yvel1, S2D_00_M40, "double", OPS_READ));

  ops_par_loop_macro(update_halo_kernel2_yvel_plus_2_b_macro, "update_halo_kernel2_yvel_plus_2_b_macro", 2, rangexy_r1b,
              ops_arg_dat(yvel0, S2D_00_M20, "double", OPS_READ),
              ops_arg_dat(yvel1, S2D_00_M20, "double", OPS_READ));



  //
  //vol_flux_x, mass_flux_x has the same ranges
  //

  int rangexy_b2c[] = {x_min-depth,x_max+1+depth,y_min-2,y_min-1};
  if(depth ==2)
  ops_par_loop_opt(update_halo_kernel3_plus, "update_halo_kernel3", 2, rangexy_b2c,
              ops_arg_dat(vol_flux_x, S2D_00_0P4, "double", OPS_READ),
              ops_arg_dat(mass_flux_x, S2D_00_0P4, "double", OPS_READ));

  int rangexy_b1c[] = {x_min-depth,x_max+1+depth,y_min-1,y_min};
  ops_par_loop_opt(update_halo_kernel3_plus, "update_halo_kernel3", 2, rangexy_b1c,
            ops_arg_dat(vol_flux_x, S2D_00_0P2, "double", OPS_READ),
            ops_arg_dat(mass_flux_x, S2D_00_0P2, "double", OPS_READ));


  int rangexy_t2c[] = {x_min-depth,x_max+1+depth,y_max+1,y_max+2};
  if(depth ==2)
  ops_par_loop_opt(update_halo_kernel3_plus, "update_halo_kernel3", 2, rangexy_t2c,
            ops_arg_dat(vol_flux_x, S2D_00_0M4, "double", OPS_READ),
            ops_arg_dat(mass_flux_x, S2D_00_0M4, "double", OPS_READ));

  int rangexy_t1c[] = {x_min-depth,x_max+1+depth,y_max,y_max+1};
  ops_par_loop_opt(update_halo_kernel3_plus, "update_halo_kernel3", 2, rangexy_t1c,
            ops_arg_dat(vol_flux_x, S2D_00_0M2, "double", OPS_READ),
            ops_arg_dat(mass_flux_x, S2D_00_0M2, "double", OPS_READ));


  int rangexy_l2c[] = {x_min-2,x_min-1,y_min-depth,y_max+depth};
  if(depth ==2)
  ops_par_loop_opt(update_halo_kernel3_minus, "update_halo_kernel3_minus", 2, rangexy_l2c,
            ops_arg_dat(vol_flux_x, S2D_00_P40, "double", OPS_READ),
            ops_arg_dat(mass_flux_x, S2D_00_P40, "double", OPS_READ));

  int rangexy_l1c[] = {x_min-1,x_min,y_min-depth,y_max+depth};
  ops_par_loop_opt(update_halo_kernel3_minus, "update_halo_kernel3_minus", 2, rangexy_l1c,
            ops_arg_dat(vol_flux_x, S2D_00_P20, "double", OPS_READ),
            ops_arg_dat(mass_flux_x, S2D_00_P20, "double", OPS_READ));

  int rangexy_r2c[] = {x_max+2,x_max+3,y_min-depth,y_max+depth}; //
  if(depth ==2)
  ops_par_loop_opt(update_halo_kernel3_minus, "update_halo_kernel3_minus", 2, rangexy_r2c,
            ops_arg_dat(vol_flux_x, S2D_00_M40, "double", OPS_READ),
            ops_arg_dat(mass_flux_x, S2D_00_M40, "double", OPS_READ));

  int rangexy_r1c[] = {x_max+1,x_max+2,y_min-depth,y_max+depth}; //
  ops_par_loop_opt(update_halo_kernel3_minus, "update_halo_kernel3_minus", 2, rangexy_r1c,
            ops_arg_dat(vol_flux_x, S2D_00_M20, "double", OPS_READ),
            ops_arg_dat(mass_flux_x, S2D_00_M20, "double", OPS_READ));

  //
  //vol_flux_y, mass_flux_y has the same ranges
  //

  int rangexy_b2d[] = {x_min-depth,x_max+depth,y_min-2,y_min-1};
  if(depth ==2)
  ops_par_loop_opt(update_halo_kernel4_minus, "update_halo_kernel4", 2, rangexy_b2d,
              ops_arg_dat(vol_flux_y, S2D_00_0P4, "double", OPS_READ),
              ops_arg_dat(mass_flux_y, S2D_00_0P4, "double", OPS_READ));

  int rangexy_b1d[] = {x_min-depth,x_max+depth,y_min-1,y_min};
  ops_par_loop_opt(update_halo_kernel4_minus, "update_halo_kernel4", 2, rangexy_b1d,
              ops_arg_dat(vol_flux_y, S2D_00_0P2, "double", OPS_READ),
              ops_arg_dat(mass_flux_y, S2D_00_0P2, "double", OPS_READ));

  int rangexy_t2d[] = {x_min-depth,x_max+depth,y_max+2,y_max+3}; //
  if(depth ==2)
  ops_par_loop_opt(update_halo_kernel4_minus, "update_halo_kernel4", 2, rangexy_t2d,
              ops_arg_dat(vol_flux_y, S2D_00_0M4, "double", OPS_READ),
              ops_arg_dat(mass_flux_y, S2D_00_0M4, "double", OPS_READ));

  int rangexy_t1d[] = {x_min-depth,x_max+depth,y_max+1,y_max+2}; //
  ops_par_loop_opt(update_halo_kernel4_minus, "update_halo_kernel4", 2, rangexy_t1d,
              ops_arg_dat(vol_flux_y, S2D_00_0M2, "double", OPS_READ),
              ops_arg_dat(mass_flux_y, S2D_00_0M2, "double", OPS_READ));

  int rangexy_l2d[] = {x_min-2,x_min-1,y_min-depth,y_max+1+depth};
  if(depth ==2)
  ops_par_loop_opt(update_halo_kernel4_plus, "update_halo_kernel4", 2, rangexy_l2d,
              ops_arg_dat(vol_flux_y, S2D_00_P40, "double", OPS_READ),
              ops_arg_dat(mass_flux_y, S2D_00_P40, "double", OPS_READ));

  int rangexy_l1d[] = {x_min-1,x_min,y_min-depth,y_max+1+depth};
  ops_par_loop_opt(update_halo_kernel4_plus, "update_halo_kernel4", 2, rangexy_l1d,
              ops_arg_dat(vol_flux_y, S2D_00_P20, "double", OPS_READ),
              ops_arg_dat(mass_flux_y, S2D_00_P20, "double", OPS_READ));

  int rangexy_r2d[] = {x_max+1,x_max+2,y_min-depth,y_max+1+depth};
  if(depth ==2)
  ops_par_loop_opt(update_halo_kernel4_plus, "update_halo_kernel4", 2, rangexy_r2d,
              ops_arg_dat(vol_flux_y, S2D_00_M40, "double", OPS_READ),
              ops_arg_dat(mass_flux_y, S2D_00_M40, "double", OPS_READ));

  int rangexy_r1d[] = {x_max,x_max+1,y_min-depth,y_max+1+depth};
  ops_par_loop_opt(update_halo_kernel4_plus, "update_halo_kernel4", 2, rangexy_r1d,
            ops_arg_dat(vol_flux_y, S2D_00_M20, "double", OPS_READ),
            ops_arg_dat(mass_flux_y, S2D_00_M20, "double", OPS_READ));

}
