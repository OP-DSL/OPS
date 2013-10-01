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

/** @brief Driver for the timestep kernels
 *  @author Wayne Gaudin, converted to OPS by Gihan Mudalige
 *  @details Invokes the user specified timestep kernel.
**/

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

// OPS header file
#include "ops_seq.h"

#include "data.h"
#include "definitions.h"

#include "calc_dt_kernel.h"

void calc_dt(double* local_dt, char* local_control,
             double* xl_pos, double* yl_pos, int* jldt, int* kldt)
{
  int small;
  double jk_control = 1.1;

  *local_dt = g_big;
  small = 0;

  int dtl_control;

  //initialize sizes using global values
  int x_cells = grid->x_cells;
  int y_cells = grid->y_cells;
  int x_min = field->x_min;
  int x_max = field->x_max;
  int y_min = field->y_min;
  int y_max = field->y_max;

  int rangexy_inner[] = {x_min,x_max,y_min,y_max}; // inner range without border

  ops_par_loop(calc_dt_kernel, "calc_dt_kernel", 2, rangexy_inner,
    ops_arg_dat(celldx, sten_self_plus1_stride2D_x, OPS_READ),
    ops_arg_dat(celldy, sten_self_plus1_stride2D_y, OPS_READ),
    ops_arg_dat(soundspeed, S2D_00, OPS_READ),
    ops_arg_dat(viscosity, S2D_00, OPS_READ),
    ops_arg_dat(density0, S2D_00, OPS_READ),
    ops_arg_dat(xvel0, sten_self2D_plus1xy, OPS_READ),
    ops_arg_dat(xarea, S2D_00_P10, OPS_READ),
    ops_arg_dat(volume, S2D_00, OPS_READ),
    ops_arg_dat(yvel0, sten_self2D_plus1xy, OPS_READ),
    ops_arg_dat(yarea, S2D_00_0P1, OPS_READ),
    ops_arg_dat(work_array1, S2D_00, OPS_WRITE) );

  ops_par_loop(calc_dt_min_kernel, "calc_dt_min_kernel", 2, rangexy_inner,
    ops_arg_dat(work_array1, S2D_00, OPS_READ),
    ops_arg_gbl(local_dt, 1, OPS_WRITE));


  //Extract the mimimum timestep information
  dtl_control = 10.01 * (jk_control - (int)(jk_control));
  jk_control = jk_control - (jk_control - (int)(jk_control));
  *jldt = (int)jk_control%x_max;
  *kldt = 1 + (jk_control/x_max);

  int rangexy_getpoint[] = {*jldt-1,*jldt,*kldt-1,*kldt}; // inner range without border

  if(*local_dt < dtmin) small = 1;

  ops_par_loop(calc_dt_get_kernel, "calc_dt_get_kernel", 2, rangexy_getpoint,
    ops_arg_dat(cellx, sten_self_stride2D_x, OPS_READ),
    ops_arg_dat(celly, sten_self_stride2D_y, OPS_READ),
    ops_arg_gbl(xl_pos, 1, OPS_WRITE),
    ops_arg_gbl(yl_pos, 1, OPS_WRITE));

  if(small != 0) {
    ops_printf("Timestep information:\n");
    ops_printf("j, k                 : %d, %d\n",*jldt,*kldt);
    ops_printf("x, y                 : %lf, %lf\n",*xl_pos,*xl_pos);
    ops_printf("timestep : %lf\n",*local_dt);
      ops_par_loop(calc_dt_print_kernel, "calc_dt_print_kernel", 2, rangexy_getpoint,
    ops_arg_dat(cellx, sten_self_stride2D_x, OPS_READ),
    ops_arg_dat(celly, sten_self_stride2D_y, OPS_READ),
    ops_arg_dat(xvel0, sten_self2D_4point1xy, OPS_READ),
    ops_arg_dat(yvel0, sten_self2D_4point1xy, OPS_READ),
    ops_arg_dat(density0, S2D_00, OPS_READ),
    ops_arg_dat(energy0, S2D_00, OPS_READ),
    ops_arg_dat(pressure, S2D_00, OPS_READ),
    ops_arg_dat(soundspeed, S2D_00, OPS_READ));
  }

  if(dtl_control == 1) sprintf(local_control, "sound");
  if(dtl_control == 2) sprintf(local_control, "xvel");
  if(dtl_control == 3) sprintf(local_control, "yvel");
  if(dtl_control == 4) sprintf(local_control, "div");

}
