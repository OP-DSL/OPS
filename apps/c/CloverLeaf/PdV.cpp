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
#define OPS_2D
#include "ops_seq_variadic.h"

#include "data.h"
#include "definitions.h"

#include "PdV_kernel.h"

// Cloverleaf functions
void ideal_gas(int predict);
void update_halo(int* fields, int depth);
void revert();

void PdV(int predict)
{
  error_condition = 0; // Not used yet due to issue with OpenA reduction

  //initialize sizes using global values
  int x_min = field.x_min;
  int x_max = field.x_max;
  int y_min = field.y_min;
  int y_max = field.y_max;

  int rangexy_inner[] = {x_min,x_max,y_min,y_max}; // inner range without border

  if(predict == TRUE) {
  ops_par_loop(PdV_kernel_predict, "PdV_kernel_predict", clover_grid, 2, rangexy_inner,
    ops_arg_dat(xarea, 1, S2D_00_P10_0P1_P1P1, "double", OPS_READ),
    ops_arg_dat(xvel0, 1, S2D_00_P10_0P1_P1P1, "double", OPS_READ),
    ops_arg_dat(yarea, 1, S2D_00_P10_0P1_P1P1, "double", OPS_READ),
    ops_arg_dat(yvel0, 1, S2D_00_P10_0P1_P1P1, "double", OPS_READ),
    ops_arg_dat(work_array1, 1, S2D_00, "double", OPS_WRITE),
    ops_arg_dat(volume, 1, S2D_00, "double", OPS_READ),
    ops_arg_dat(pressure, 1, S2D_00, "double", OPS_READ),
    ops_arg_dat(density0, 1, S2D_00, "double", OPS_READ),
    ops_arg_dat(density1, 1, S2D_00, "double", OPS_WRITE),
    ops_arg_dat(viscosity, 1, S2D_00, "double", OPS_READ),
    ops_arg_dat(energy0, 1, S2D_00, "double", OPS_READ),
    ops_arg_dat(energy1, 1, S2D_00, "double", OPS_WRITE));
  }
  else {
  ops_par_loop(PdV_kernel_nopredict, "PdV_kernel_nopredict", clover_grid, 2, rangexy_inner,
    ops_arg_dat(xarea, 1, S2D_00_P10_0P1_P1P1, "double", OPS_READ),
    ops_arg_dat(xvel0, 1, S2D_00_P10_0P1_P1P1, "double", OPS_READ),
    ops_arg_dat(xvel1, 1, S2D_00_P10_0P1_P1P1, "double", OPS_READ),
    ops_arg_dat(yarea, 1, S2D_00_P10_0P1_P1P1, "double", OPS_READ),
    ops_arg_dat(yvel0, 1, S2D_00_P10_0P1_P1P1, "double", OPS_READ),
    ops_arg_dat(yvel1, 1, S2D_00_P10_0P1_P1P1, "double", OPS_READ),
    ops_arg_dat(work_array1, 1, S2D_00, "double", OPS_WRITE),
    ops_arg_dat(volume, 1, S2D_00, "double", OPS_READ),
    ops_arg_dat(pressure, 1, S2D_00, "double", OPS_READ),
    ops_arg_dat(density0, 1, S2D_00, "double", OPS_READ),
    ops_arg_dat(density1, 1, S2D_00, "double", OPS_WRITE),
    ops_arg_dat(viscosity, 1, S2D_00, "double", OPS_READ),
    ops_arg_dat(energy0, 1, S2D_00, "double", OPS_READ),
    ops_arg_dat(energy1, 1, S2D_00, "double", OPS_WRITE));
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
