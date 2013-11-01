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

/** @brief PdV update
 *  @author Wayne Gaudin, converted to OPS by Gihan Mudalige
 *  @details Invokes the user specified kernel for the PdV update.
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

#include "PdV_kernel.h"

// Cloverleaf functions
void ideal_gas(int predict);
void update_halo(int* fields, int depth);
void revert();

void PdV_kernel_predict_macro(double *xarea, double *xvel0,
                double *yarea, double *yvel0,
                double *volume_change, double *volume,
                double *pressure,
                double *density0, double *density1,
                double *viscosity,
                double *energy0, double *energy1) {

  //xvel0, S2D_00_P10_0P1_P1P1

  double recip_volume, energy_change, min_cell_volume;
  double right_flux, left_flux, top_flux, bottom_flux, total_flux;

  left_flux = ( xarea[OPS_ACC0(0,0)] * ( xvel0[OPS_ACC1(0,0)] + xvel0[OPS_ACC1(0,1)] +
                                xvel0[OPS_ACC1(0,0)] + xvel0[OPS_ACC1(0,1)] ) ) * 0.25 * dt * 0.5;
  right_flux = ( xarea[OPS_ACC0(1,0)] * ( xvel0[OPS_ACC1(1,0)] + xvel0[OPS_ACC1(1,1)] +
                                 xvel0[OPS_ACC1(1,0)] + xvel0[OPS_ACC1(1,1)] ) ) * 0.25 * dt * 0.5;

  bottom_flux = ( yarea[OPS_ACC2(0,0)] * ( yvel0[OPS_ACC3(0,0)] + yvel0[OPS_ACC3(1,0)] +
                                  yvel0[OPS_ACC3(0,0)] + yvel0[OPS_ACC3(1,0)] ) ) * 0.25* dt * 0.5;
  top_flux = ( yarea[OPS_ACC2(0,1)] * ( yvel0[OPS_ACC3(0,1)] + yvel0[OPS_ACC3(1,1)] +
                               yvel0[OPS_ACC3(0,1)] + yvel0[OPS_ACC3(1,1)] ) ) * 0.25 * dt * 0.5;

  total_flux = right_flux - left_flux + top_flux - bottom_flux;

  volume_change[OPS_ACC4(0,0)] = (volume[OPS_ACC5(0,0)])/(volume[OPS_ACC5(0,0)] + total_flux);

  min_cell_volume = MIN( volume[OPS_ACC5(0,0)] + right_flux - left_flux + top_flux - bottom_flux ,
                           MIN(volume[OPS_ACC5(0,0)] + right_flux - left_flux,
                           volume[OPS_ACC5(0,0)] + top_flux - bottom_flux) );

  recip_volume = 1.0/volume[OPS_ACC5(0,0)];

  energy_change = ( pressure[OPS_ACC6(0,0)]/density0[OPS_ACC7(0,0)] +
                    viscosity[OPS_ACC9(0,0)]/density0[OPS_ACC7(0,0)] ) * total_flux * recip_volume;
  energy1[OPS_ACC11(0,0)] = energy0[OPS_ACC10(0,0)] - energy_change;
  density1[OPS_ACC8(0,0)] = density0[OPS_ACC7(0,0)] * volume_change[OPS_ACC4(0,0)];

}

void PdV_kernel_nopredict_macro(double *xarea, double *xvel0, double *xvel1,
                double *yarea, double *yvel0, double *yvel1,
                double *volume_change, double *volume,
                double *pressure,
                double *density0, double *density1,
                double *viscosity,
                double *energy0, double *energy1) {

  //xvel0, S2D_00_P10_0P1_P1P1

  double recip_volume, energy_change, min_cell_volume;
  double right_flux, left_flux, top_flux, bottom_flux, total_flux;

  left_flux = ( xarea[OPS_ACC0(0,0)] * ( xvel0[OPS_ACC1(0,0)] + xvel0[OPS_ACC1(0,1)] +
                                xvel1[OPS_ACC2(0,0)] + xvel1[OPS_ACC2(0,1)] ) ) * 0.25 * dt;
  right_flux = ( xarea[OPS_ACC0(1,0)] * ( xvel0[OPS_ACC1(1,0)] + xvel0[OPS_ACC1(1,1)] +
                                 xvel1[OPS_ACC2(1,0)] + xvel1[OPS_ACC2(1,1)] ) ) * 0.25 * dt;

  bottom_flux = ( yarea[OPS_ACC3(0,0)] * ( yvel0[OPS_ACC4(0,0)] + yvel0[OPS_ACC4(1,0)] +
                                  yvel1[OPS_ACC5(0,0)] + yvel1[OPS_ACC5(1,0)] ) ) * 0.25* dt;
  top_flux = ( yarea[OPS_ACC3(0,1)] * ( yvel0[OPS_ACC4(0,1)] + yvel0[OPS_ACC4(1,1)] +
                               yvel1[OPS_ACC5(0,1)] + yvel1[OPS_ACC5(1,1)] ) ) * 0.25 * dt;

  total_flux = right_flux - left_flux + top_flux - bottom_flux;

  volume_change[OPS_ACC6(0,0)] = (volume[OPS_ACC7(0,0)])/(volume[OPS_ACC7(0,0)] + total_flux);

  min_cell_volume = MIN( volume[OPS_ACC7(0,0)] + right_flux - left_flux + top_flux - bottom_flux ,
                           MIN(volume[OPS_ACC7(0,0)] + right_flux - left_flux,
                           volume[OPS_ACC7(0,0)] + top_flux - bottom_flux) );

  recip_volume = 1.0/volume[OPS_ACC7(0,0)];

  energy_change = ( pressure[OPS_ACC8(0,0)]/density0[OPS_ACC9(0,0)] +
                    viscosity[OPS_ACC11(0,0)]/density0[OPS_ACC9(0,0)] ) * total_flux * recip_volume;
  energy1[OPS_ACC13(0,0)] = energy0[OPS_ACC12(0,0)] - energy_change;
  density1[OPS_ACC10(0,0)] = density0[OPS_ACC9(0,0)] * volume_change[OPS_ACC6(0,0)];

}

void PdV(int predict)
{
  error_condition = 0; // Not used yet due to issue with OpenA reduction

  //initialize sizes using global values
  int x_cells = grid->x_cells;
  int y_cells = grid->y_cells;
  int x_min = field->x_min;
  int x_max = field->x_max;
  int y_min = field->y_min;
  int y_max = field->y_max;

  int rangexy_inner[] = {x_min,x_max,y_min,y_max}; // inner range without border

  if(predict == TRUE) {
  ops_par_loop_macro(PdV_kernel_predict_macro, "PdV_kernel_predict_macro", 2, rangexy_inner,
    ops_arg_dat(xarea, S2D_00_P10_0P1_P1P1, "double", OPS_READ),
    ops_arg_dat(xvel0, S2D_00_P10_0P1_P1P1, "double", OPS_READ),
    ops_arg_dat(yarea, S2D_00_P10_0P1_P1P1, "double", OPS_READ),
    ops_arg_dat(yvel0, S2D_00_P10_0P1_P1P1, "double", OPS_READ),
    ops_arg_dat(work_array1, S2D_00, "double", OPS_READ),
    ops_arg_dat(volume, S2D_00, "double", OPS_READ),
    ops_arg_dat(pressure, S2D_00, "double", OPS_READ),
    ops_arg_dat(density0, S2D_00, "double", OPS_READ),
    ops_arg_dat(density1, S2D_00, "double", OPS_READ),
    ops_arg_dat(viscosity, S2D_00, "double", OPS_READ),
    ops_arg_dat(energy0, S2D_00, "double", OPS_READ),
    ops_arg_dat(energy1, S2D_00, "double", OPS_READ));
  }
  else {
  ops_par_loop_macro(PdV_kernel_nopredict_macro, "PdV_kernel_nopredict_macro", 2, rangexy_inner,
    ops_arg_dat(xarea, S2D_00_P10_0P1_P1P1, "double", OPS_READ),
    ops_arg_dat(xvel0, S2D_00_P10_0P1_P1P1, "double", OPS_READ),
    ops_arg_dat(xvel1, S2D_00_P10_0P1_P1P1, "double", OPS_READ),
    ops_arg_dat(yarea, S2D_00_P10_0P1_P1P1, "double", OPS_READ),
    ops_arg_dat(yvel0, S2D_00_P10_0P1_P1P1, "double", OPS_READ),
    ops_arg_dat(yvel1, S2D_00_P10_0P1_P1P1, "double", OPS_READ),
    ops_arg_dat(work_array1, S2D_00, "double", OPS_READ),
    ops_arg_dat(volume, S2D_00, "double", OPS_READ),
    ops_arg_dat(pressure, S2D_00, "double", OPS_READ),
    ops_arg_dat(density0, S2D_00, "double", OPS_READ),
    ops_arg_dat(density1, S2D_00, "double", OPS_READ),
    ops_arg_dat(viscosity, S2D_00, "double", OPS_READ),
    ops_arg_dat(energy0, S2D_00, "double", OPS_READ),
    ops_arg_dat(energy1, S2D_00, "double", OPS_READ));
  }

  if(error_condition == 1) {
    ops_printf("PdV: error in PdV\n");
    exit(-2);
  }

  if(predict == TRUE) {
    ideal_gas(TRUE);

    fields[FIELD_DENSITY0]  = 0;
    fields[FIELD_ENERGY0]   = 0;
    fields[FIELD_PRESSURE]  = 1;
    fields[FIELD_VISCOSITY] = 0;
    fields[FIELD_DENSITY1]  = 0;
    fields[FIELD_ENERGY1]   = 0;
    fields[FIELD_XVEL0]     = 0;
    fields[FIELD_YVEL0]     = 0;
    fields[FIELD_XVEL1]     = 0;
    fields[FIELD_YVEL1]     = 0;
    fields[FIELD_VOL_FLUX_X] = 0;
    fields[FIELD_VOL_FLUX_Y] = 0;
    fields[FIELD_MASS_FLUX_X] = 0;
    fields[FIELD_MASS_FLUX_Y] = 0;
    update_halo(fields,1);

  }

  if(predict == TRUE) {
    revert();
  }

}
