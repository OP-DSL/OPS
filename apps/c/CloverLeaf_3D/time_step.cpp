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

/** @brief Calculate the minimum timestep
 *  @author Wayne Gaudin, converted to OPS by Gihan Mudalige
 *  @details Invokes the kernels needed to calculate the timestep and finds
 *  the minimum across all chunks. Checks if the timestep falls below the
 *  user specified limit and outputs the timestep information.
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

void ideal_gas(int predict);
void update_halo(int* fields, int depth);
void viscosity_func();
void calc_dt(double*, char*,
             double*, double*, int*, int*, double*, int*);

void timestep()
{
  int jldt, kldt, lldt;
  double dtlp;
  double x_pos, y_pos, z_pos, xl_pos, yl_pos, zl_pos;
  char dt_control[8];
  char dtl_control[8];

  int small = 0;

  x_pos  = 0.0;
  y_pos  = 0.0;
  z_pos  = 0.0;
  xl_pos = 0.0;
  yl_pos = 0.0;
  zl_pos = 0.0;

  //initialize sizes using global values
  int x_cells = grid.x_cells;
  int y_cells = grid.y_cells;
  int x_min = field.x_min;
  int x_max = field.x_max;
  int y_min = field.y_min;
  int y_max = field.y_max;

  ideal_gas(FALSE);

  fields[FIELD_DENSITY0]  = 1;
  fields[FIELD_ENERGY0]   = 1;
  fields[FIELD_DENSITY1]  = 0;
  fields[FIELD_ENERGY1]   = 0;
  fields[FIELD_SOUNDSPEED] = 0;
  fields[FIELD_PRESSURE]  = 1;
  fields[FIELD_VISCOSITY] = 0;
  fields[FIELD_XVEL0]     = 1;
  fields[FIELD_YVEL0]     = 1;
  fields[FIELD_ZVEL0]     = 0;
  fields[FIELD_XVEL1]     = 0;
  fields[FIELD_YVEL1]     = 0;
  fields[FIELD_ZVEL1]     = 0;
  fields[FIELD_VOL_FLUX_X] = 0;
  fields[FIELD_VOL_FLUX_Y] = 0;
  fields[FIELD_VOL_FLUX_Z] = 0;
  fields[FIELD_MASS_FLUX_X] = 0;
  fields[FIELD_MASS_FLUX_Y] = 0;
  fields[FIELD_MASS_FLUX_Z] = 0;

  update_halo(fields,1);

  viscosity_func();

  fields[FIELD_DENSITY0]  = 0;
  fields[FIELD_ENERGY0]   = 0;
  fields[FIELD_DENSITY1]  = 0;
  fields[FIELD_ENERGY1]   = 0;
  fields[FIELD_SOUNDSPEED] = 0;
  fields[FIELD_PRESSURE]  = 0;
  fields[FIELD_VISCOSITY] = 1;
  fields[FIELD_XVEL0]     = 0;
  fields[FIELD_YVEL0]     = 0;
  fields[FIELD_ZVEL0]     = 0;
  fields[FIELD_XVEL1]     = 0;
  fields[FIELD_YVEL1]     = 0;
  fields[FIELD_ZVEL1]     = 0;
  fields[FIELD_VOL_FLUX_X] = 0;
  fields[FIELD_VOL_FLUX_Y] = 0;
  fields[FIELD_VOL_FLUX_Z] = 0;
  fields[FIELD_MASS_FLUX_X] = 0;
  fields[FIELD_MASS_FLUX_Y] = 0;
  fields[FIELD_MASS_FLUX_Z] = 0;


  update_halo(fields,1);

  //dtl_control = (char *)xmalloc(8*sizeof(char ));
  calc_dt(&dtlp, dtl_control, &xl_pos, &yl_pos, &jldt, &kldt, &zl_pos, &lldt);

  double dt_temp = g_big;
  if (dtlp <= dt_temp) {
      dt_temp = dtlp;
      //memcpy(dt_control, dtl_control, sizeof(char)*8);
      x_pos = xl_pos;
      y_pos = yl_pos;
      jdt = jldt;
      kdt = kldt;
  }

  dt_temp = MIN(MIN(dt_temp, (dtold * dtrise)), dtmax);
  ops_update_const("dt", 1, "double", &dt_temp);
  dt = dt_temp;

  if(dt < dtmin) small=1;
  ops_printf(
  "\n Step %d time %11.7lf control %s timestep  %3.2E  %d, %d x  %E  y %E",
    step,   clover_time,    dtl_control,dt,          jdt, kdt,  x_pos,y_pos);
  ops_fprintf(g_out,
  "\n Step %d time %11.7lf control %s timestep  %3.2E  %d, %d x  %E  y %E",
    step,   clover_time,    dtl_control,dt,          jdt, kdt,  x_pos,y_pos);

  if(small == 1) {
    ops_printf("timestep :small timestep\n");
    exit(-2);
  }

  dtold = dt;

}