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
#define OPS_3D
#include "ops_seq.h"

#include "data.h"
#include "definitions.h"

#include "calc_dt_kernel.h"

void calc_dt(double* local_dt, char* local_control,
             double* xl_pos, double* yl_pos, int* jldt, int* kldt, double *zl_pos, int *lldt)
{
  int small;
  double jk_control = 1.1;

  small = 0;

  int dtl_control;

  //initialize sizes using global values
  int x_min = field.x_min;
  int x_max = field.x_max;
  int y_min = field.y_min;
  int y_max = field.y_max;
  int z_min = field.z_min;
  int z_max = field.z_max;
  
  int rangexyz_inner[] = {x_min,x_max,y_min,y_max,z_min,z_max}; // inner range without border

  ops_par_loop(calc_dt_kernel, "calc_dt_kernel", clover_grid, 3, rangexyz_inner,
    ops_arg_dat(celldx, 1, S3D_000_P100_STRID3D_X, "double", OPS_READ),
    ops_arg_dat(celldy, 1, S3D_000_0P10_STRID3D_Y, "double", OPS_READ),
    ops_arg_dat(soundspeed, 1, S3D_000, "double", OPS_READ),
    ops_arg_dat(viscosity, 1, S3D_000, "double", OPS_READ),
    ops_arg_dat(density0, 1, S3D_000, "double", OPS_READ),
    ops_arg_dat(xvel0, 1, S3D_000_fP1P1P1, "double", OPS_READ),
    ops_arg_dat(xarea, 1, S3D_000, "double", OPS_READ),
    ops_arg_dat(volume, 1, S3D_000, "double", OPS_READ),
    ops_arg_dat(yvel0, 1, S3D_000_fP1P1P1, "double", OPS_READ),
    ops_arg_dat(yarea, 1, S3D_000, "double", OPS_READ),
    ops_arg_dat(work_array1, 1, S3D_000, "double", OPS_WRITE),
    ops_arg_dat(celldz, 1, S3D_000_00P1_STRID3D_Z, "double", OPS_READ),
    ops_arg_dat(zvel0, 1, S3D_000_fP1P1P1, "double", OPS_READ),
    ops_arg_dat(zarea, 1, S3D_000, "double", OPS_READ));

  ops_par_loop(calc_dt_kernel_min, "calc_dt_kernel_min", clover_grid, 3, rangexyz_inner,
    ops_arg_dat(work_array1, 1, S3D_000, "double", OPS_READ),
    ops_arg_reduce(red_local_dt, 1, "double", OPS_MIN));

  //printf("*local_dt = %lf\n",*local_dt);

  //Extract the mimimum timestep information
  dtl_control = 10.01 * (jk_control - (int)(jk_control));
  jk_control = jk_control - (jk_control - (int)(jk_control));
  //*jldt = ((int)jk_control)%x_max;
  //*kldt = 1 + (jk_control/x_max);
  *jldt = ((int)jk_control)%(x_max-2);
  *kldt = 1 + (jk_control/(x_max-2));
  *lldt = 1 + (jk_control/(x_max-2));

  int rangexyz_getpoint[] = {*jldt-1+2,*jldt+2,*kldt-1+2,*kldt+2,*lldt-1+2,*lldt+2}; // get point value //note +2 added due to boundary

  ops_par_loop(calc_dt_kernel_get, "calc_dt_kernel_getx", clover_grid, 3, rangexyz_getpoint,
    ops_arg_dat(cellx, 1, S3D_000_STRID3D_X, "double", OPS_READ),
    ops_arg_dat(celly, 1, S3D_000_STRID3D_Y, "double", OPS_READ),
    ops_arg_reduce(red_xl_pos, 1, "double", OPS_INC),
    ops_arg_reduce(red_yl_pos, 1, "double", OPS_INC),
    ops_arg_dat(cellz, 1, S3D_000_STRID3D_Z, "double", OPS_READ),
    ops_arg_reduce(red_zl_pos, 1, "double", OPS_INC));

  ops_reduction_result(red_local_dt, local_dt);
  ops_reduction_result(red_xl_pos, xl_pos);
  ops_reduction_result(red_yl_pos, yl_pos);
  *local_dt = MIN(*local_dt, g_big);

  if(*local_dt < dtmin) small = 1;

  if(small != 0) {
    ops_printf("Timestep information:\n");
    ops_printf("j, k                 : %d, %d\n",*jldt,*kldt);
    ops_printf("x, y                 : %lf, %lf\n",*xl_pos,*xl_pos);
    ops_printf("timestep : %lf\n",*local_dt);

  double output[28] = {0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,
                       0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0};    
  ops_par_loop(calc_dt_kernel_print, "calc_dt_kernel_print", clover_grid, 3,rangexyz_getpoint,
    ops_arg_dat(xvel0, 1, S3D_000_fP1P1P1, "double", OPS_READ),
    ops_arg_dat(yvel0, 1, S3D_000_fP1P1P1, "double", OPS_READ),
    ops_arg_dat(zvel0, 1, S3D_000_fP1P1P1, "double", OPS_READ),
    ops_arg_dat(density0, 1, S3D_000, "double", OPS_READ),
    ops_arg_dat(energy0, 1, S3D_000, "double", OPS_READ),
    ops_arg_dat(pressure, 1, S3D_000, "double", OPS_READ),
    ops_arg_dat(soundspeed, 1, S3D_000, "double", OPS_READ),
    ops_arg_reduce(red_output, 28, "double", OPS_INC));
  
    ops_reduction_result(red_output, output);

    printf("Cell velocities:\n");
    printf("%E, %E, %E \n",output[0], output[1], output[2]);
    printf("%E, %E, %E \n",output[3], output[4], output[5]);
    printf("%E, %E, %E \n",output[6], output[7], output[8]);
    printf("%E, %E, %E \n",output[9], output[10], output[11]);
    printf("%E, %E, %E \n",output[12], output[13], output[14]);
    printf("%E, %E, %E \n",output[15], output[16], output[17]);
    printf("%E, %E, %E \n",output[18], output[19], output[20]);
    printf("%E, %E, %E \n",output[21], output[22], output[23]);
    
    printf("density, energy, pressure, soundspeed = %lf, %lf, %lf, %lf \n",
    output[24], output[25],output[26],output[27]);
  }

  if(dtl_control == 1) sprintf(local_control, "sound");
  if(dtl_control == 2) sprintf(local_control, "xvel");
  if(dtl_control == 3) sprintf(local_control, "yvel");
  if(dtl_control == 4) sprintf(local_control, "div");

}
