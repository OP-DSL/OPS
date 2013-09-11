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

void ideal_gas(int predict);
void update_halo(int* fields, int depth);
void viscosity_func();
void calc_dt(double*, char*,
             double*, double*, int*, int*);

void timestep()
{
  int jldt, kldt;
  double dtlp;
  double x_pos, y_pos, xl_pos, yl_pos;
  char dt_control[8];
  char dtl_control[8];

  dt = g_big;
  int small = 0;

  //initialize sizes using global values
  int x_cells = grid->x_cells;
  int y_cells = grid->y_cells;
  int x_min = field->x_min;
  int x_max = field->x_max;
  int y_min = field->y_min;
  int y_max = field->y_max;

  ideal_gas(FALSE);

  fields[FIELD_PRESSURE] = 1;
  fields[FIELD_ENERGY0] = 1;
  fields[FIELD_DENSITY0] = 1;
  fields[FIELD_XVEL0] = 1;
  fields[FIELD_YVEL0] = 1;

  update_halo(fields,1);

  viscosity_func();
  //ops_print_dat_to_txtfile_core(viscosity, "cloverdats.dat");

  fields[FIELD_VISCOSITY] = 1;

  update_halo(fields,1);

  //dtl_control = (char *)xmalloc(8*sizeof(char ));
  calc_dt(&dtlp, dtl_control, &xl_pos, &yl_pos, &jldt, &kldt);

  if (dtlp <= dt) {
      dt = dtlp;
      //memcpy(dt_control, dtl_control, sizeof(char)*8);
      x_pos = xl_pos;
      y_pos = yl_pos;
      jdt = jldt;
      kdt = kldt;
  }

  dt = MIN(MIN(dt, (dtold * dtrise)), dtmax);
  //CALL clover_min(dt)

  if(dt < dtmin) small=1;
  ops_printf(
  " Step %d time %lf control %s timestep  %E  %d, %d x  %E  y %E\n",
    step,   time,    dtl_control,dt,          jdt, kdt,  x_pos,y_pos);
  ops_fprintf(g_out,
  " Step %d time %lf control %s timestep  %E  %d, %d x  %E  y %E\n",
    step,   time,    dtl_control,dt,          jdt, kdt,  x_pos,y_pos);

  if(small == 1) {
    ops_printf("timestep :small timestep\n");
    exit(-2);
  }

  dtold = dt;
}
