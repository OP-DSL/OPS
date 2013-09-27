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
#include "ops_seq.h"

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
  int x_cells = grid->x_cells;
  int y_cells = grid->y_cells;
  int x_min = field->x_min;
  int x_max = field->x_max;
  int y_min = field->y_min;
  int y_max = field->y_max;

  int rangexy_inner[] = {x_min,x_max,y_min,y_max}; // inner range without border

  if(predict) {
  ops_par_loop(PdV_kernel_predict, "PdV_kernel_predict", 2, rangexy_inner,
    ops_arg_dat(xarea, sten_self2D_plus1xy, OPS_READ),
    ops_arg_dat(xvel0, sten_self2D_plus1xy, OPS_READ),
    ops_arg_dat(yarea, sten_self2D_plus1xy, OPS_READ),
    ops_arg_dat(yvel0, sten_self2D_plus1xy, OPS_READ),
    ops_arg_dat(work_array1, sten_self_2D, OPS_WRITE),
    ops_arg_dat(volume, sten_self_2D, OPS_READ),
    ops_arg_dat(pressure, sten_self_2D, OPS_READ),
    ops_arg_dat(density0, sten_self_2D, OPS_READ),
    ops_arg_dat(density1, sten_self_2D, OPS_WRITE),
    ops_arg_dat(viscosity, sten_self_2D, OPS_READ),
    ops_arg_dat(energy0, sten_self_2D, OPS_READ),
    ops_arg_dat(energy1, sten_self_2D, OPS_WRITE));
  }
  else {
  ops_par_loop(PdV_kernel_nopredict, "PdV_kernel_nopredict", 2, rangexy_inner,
    ops_arg_dat(xarea, sten_self2D_plus1xy, OPS_READ),
    ops_arg_dat(xvel0, sten_self2D_plus1xy, OPS_READ),
    ops_arg_dat(xvel1, sten_self2D_plus1xy, OPS_READ),
    ops_arg_dat(yarea, sten_self2D_plus1xy, OPS_READ),
    ops_arg_dat(yvel0, sten_self2D_plus1xy, OPS_READ),
    ops_arg_dat(yvel1, sten_self2D_plus1xy, OPS_READ),
    ops_arg_dat(work_array1, sten_self_2D, OPS_WRITE),
    ops_arg_dat(volume, sten_self_2D, OPS_READ),
    ops_arg_dat(pressure, sten_self_2D, OPS_READ),
    ops_arg_dat(density0, sten_self_2D, OPS_READ),
    ops_arg_dat(density1, sten_self_2D, OPS_WRITE),
    ops_arg_dat(viscosity, sten_self_2D, OPS_READ),
    ops_arg_dat(energy0, sten_self_2D, OPS_READ),
    ops_arg_dat(energy1, sten_self_2D, OPS_WRITE));
  }

  if(error_condition == 1) {
    ops_printf("PdV: error in PdV\n");
    exit(-2);
  }

  if(predict) {
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

  if(predict) {
    revert();
  }

}
