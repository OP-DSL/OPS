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
#include "ops_seq_opt.h"
#include "ops_seq_macro.h"

#include "data.h"
#include "definitions.h"

#include "calc_dt_kernel.h"



void calc_dt_kernel_macro(double *celldx, double *celldy, double *soundspeed,
                    double *viscosity, double *density0, double *xvel0,
                    double *xarea, double *volume, double *yvel0,
                    double *yarea, double *dt_min /*dt_min is work_array1*/) {

  double div, dsx, dsy, dtut, dtvt, dtct, dtdivt, cc, dv1, dv2, jk_control;

  dsx = celldx[OPS_ACC0(0,0)];
  dsy = celldy[OPS_ACC2(0,0)];

  cc = soundspeed[OPS_ACC2(0,0)] * soundspeed[OPS_ACC2(0,0)];
  cc = cc + 2.0 * viscosity[OPS_ACC3(0,0)]/density0[OPS_ACC4(0,0)];
  cc = MAX(sqrt(cc),g_small);

  dtct = dtc_safe * MIN(dsx,dsy)/cc;

  div=0.0;

  //00_10_01_11

  dv1 = (xvel0[OPS_ACC5(0,0)] + xvel0[OPS_ACC5(0,1)]) * xarea[OPS_ACC6(0,0)];
  dv2 = (xvel0[OPS_ACC5(1,0)] + xvel0[OPS_ACC5(1,1)]) * xarea[OPS_ACC6(1,0)];

  div = div + dv2 - dv1;

  dtut = dtu_safe * 2.0 * volume[OPS_ACC7(0,0)]/MAX(MAX(fabs(dv1), fabs(dv2)), g_small * volume[OPS_ACC7(0,0)]);

  dv1 = (yvel0[OPS_ACC8(0,0)] + yvel0[OPS_ACC8(1,0)]) * yarea[OPS_ACC9(0,0)];
  dv2 = (yvel0[OPS_ACC8(0,1)] + yvel0[OPS_ACC8(1,1)]) * yarea[OPS_ACC9(0,1)];

  div = div + dv2 - dv1;

  dtvt = dtv_safe * 2.0 * volume[OPS_ACC7(0,0)]/MAX(MAX(fabs(dv1),fabs(dv2)), g_small * volume[OPS_ACC7(0,0)]);

  div = div/(2.0 * volume[OPS_ACC7(0,0)]);

  if(div < -g_small)
    dtdivt = dtdiv_safe * (-1.0/div);
  else
    dtdivt = g_big;

  //dt_min is work_array1
  dt_min[OPS_ACC10(0,0)] = MIN(MIN(dtct, dtut), MIN(dtvt, dtdivt));
  //printf("dt_min %3.15e \n",**dt_min);
}

void calc_dt_kernel_print_macro(double *cellx, double *celly,
                        double *xvel0, double *yvel0,
                        double *density0, double *energy0,
                        double *pressure, double *soundspeed) {
  printf("Cell velocities:\n");
  printf("%E, %E \n",xvel0[OPS_ACC2(1,0)], yvel0[OPS_ACC3(1,0)]); //xvel0(jldt  ,kldt  ),yvel0(jldt  ,kldt  )
  printf("%E, %E \n",xvel0[OPS_ACC2(-1,0)], yvel0[OPS_ACC3(-1,0)]); //xvel0(jldt+1,kldt  ),yvel0(jldt+1,kldt  )
  printf("%E, %E \n",xvel0[OPS_ACC2(0,1)], yvel0[OPS_ACC3(0,1)]); //xvel0(jldt+1,kldt+1),yvel0(jldt+1,kldt+1)
  printf("%E, %E \n",xvel0[OPS_ACC2(0,-1)], yvel0[OPS_ACC3(0,-1)]); //xvel0(jldt  ,kldt+1),yvel0(jldt  ,kldt+1)

  printf("density, energy, pressure, soundspeed = %lf, %lf, %lf, %lf \n",
    density0[OPS_ACC4(0,0)], energy0[OPS_ACC5(0,0)], pressure[OPS_ACC6(0,0)], soundspeed[OPS_ACC7(0,0)]);
}


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

  ops_par_loop_macro(calc_dt_kernel_macro, "calc_dt_kernel_macro", 2, rangexy_inner,
    ops_arg_dat(celldx, s2D_00_P10_STRID2D_X, "double", OPS_READ),
    ops_arg_dat(celldy, S2D_00_0P1_STRID2D_Y, "double", OPS_READ),
    ops_arg_dat(soundspeed, S2D_00, "double", OPS_READ),
    ops_arg_dat(viscosity, S2D_00, "double", OPS_READ),
    ops_arg_dat(density0, S2D_00, "double", OPS_READ),
    ops_arg_dat(xvel0, S2D_00_P10_0P1_P1P1, "double", OPS_READ),
    ops_arg_dat(xarea, S2D_00_P10, "double", OPS_READ),
    ops_arg_dat(volume, S2D_00, "double", OPS_READ),
    ops_arg_dat(yvel0, S2D_00_P10_0P1_P1P1, "double", OPS_READ),
    ops_arg_dat(yarea, S2D_00_0P1, "double", OPS_READ),
    ops_arg_dat(work_array1, S2D_00, "double", OPS_WRITE) );

  ops_par_loop_opt(calc_dt_kernel_min, "calc_dt_kernel_min", 2, rangexy_inner,
    ops_arg_dat(work_array1, S2D_00, "double", OPS_READ),
    ops_arg_gbl(local_dt, 1, "double", OPS_MIN));


  //Extract the mimimum timestep information
  dtl_control = 10.01 * (jk_control - (int)(jk_control));
  jk_control = jk_control - (jk_control - (int)(jk_control));
  *jldt = (int)jk_control%x_max;
  *kldt = 1 + (jk_control/x_max);

  int rangexy_getpoint[] = {*jldt-1,*jldt,*kldt-1,*kldt}; // get point value

  if(*local_dt < dtmin) small = 1;

  ops_par_loop_opt(calc_dt_kernel_get, "calc_dt_kernel_get", 2, rangexy_getpoint,
    ops_arg_dat(cellx, S2D_00_STRID2D_X, "double", OPS_READ),
    ops_arg_dat(celly, S2D_00_STRID2D_Y, "double", OPS_READ),
    ops_arg_gbl(xl_pos, 1, "double", OPS_WRITE),
    ops_arg_gbl(yl_pos, 1, "double", OPS_WRITE));

  if(small != 0) {
    ops_printf("Timestep information:\n");
    ops_printf("j, k                 : %d, %d\n",*jldt,*kldt);
    ops_printf("x, y                 : %lf, %lf\n",*xl_pos,*xl_pos);
    ops_printf("timestep : %lf\n",*local_dt);

    ops_par_loop_macro(calc_dt_kernel_print_macro, "calc_dt_kernel_print_macro", 2, rangexy_getpoint,
    ops_arg_dat(cellx, S2D_00_STRID2D_X, "double", OPS_READ),
    ops_arg_dat(celly, S2D_00_STRID2D_Y, "double", OPS_READ),
    ops_arg_dat(xvel0, S2D_10_M10_01_0M1, "double", OPS_READ),
    ops_arg_dat(yvel0, S2D_10_M10_01_0M1, "double", OPS_READ),
    ops_arg_dat(density0, S2D_00, "double", OPS_READ),
    ops_arg_dat(energy0, S2D_00, "double", OPS_READ),
    ops_arg_dat(pressure, S2D_00, "double", OPS_READ),
    ops_arg_dat(soundspeed, S2D_00, "double", OPS_READ));
  }

  if(dtl_control == 1) sprintf(local_control, "sound");
  if(dtl_control == 2) sprintf(local_control, "xvel");
  if(dtl_control == 3) sprintf(local_control, "yvel");
  if(dtl_control == 4) sprintf(local_control, "div");

}
