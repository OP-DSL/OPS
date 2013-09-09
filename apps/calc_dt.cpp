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

#include "calc_dt_kernel.h"

void calc_dt(double* local_dt, char local_control[],
             double* xl_pos, double* yl_pos, int* jldt, int* kldt)
{
  int small;
  double jk_control = 1.1;

  *local_dt = g_big;
  small = 0;

  int dtl_control;

  ops_print_dat_to_txtfile_core(soundspeed, "cloverdats.dat");

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
    ops_arg_dat(soundspeed, sten_self_2D, OPS_READ),
    ops_arg_dat(viscosity, sten_self_2D, OPS_READ),
    ops_arg_dat(density0, sten_self_2D, OPS_READ),
    ops_arg_dat(xvel0, sten_self2D_plus1xy, OPS_READ),
    ops_arg_dat(xarea, sten_self2D_plus1x, OPS_READ),
    ops_arg_dat(volume, sten_self_2D, OPS_READ),
    ops_arg_dat(yvel0, sten_self2D_plus1xy, OPS_READ),
    ops_arg_dat(yarea, sten_self2D_plus1y, OPS_READ),
    ops_arg_dat(work_array1, sten_self_2D, OPS_WRITE)
    );




  ops_par_loop(calc_dt_min_kernel, "calc_dt_min_kernel", 2, rangexy_inner,
    ops_arg_dat(work_array1, sten_self_2D, OPS_READ),
    ops_arg_gbl(local_dt, 1, OPS_WRITE));


  //Extract the mimimum timestep information
  dtl_control = 10.01 * (jk_control - (int)(jk_control));
  jk_control = jk_control - (jk_control - (int)(jk_control));
  *jldt = (int)jk_control%x_max;
  *kldt = 1 + (jk_control/x_max);
  //*xl_pos = cellx(jldt)
  //*yl_pos = celly(kldt)

  if(*local_dt < dtmin) small = 1;

  //if(small != 0) {
    ops_printf("Timestep information:\n");
    ops_printf("j, k                 : %d, %d\n",*jldt,*kldt);
    //ops_printf(0,*) 'x, y                 : ',cellx(jldt),celly(kldt)
    ops_printf("timestep : %lf\n",*local_dt);



  //}



  if(dtl_control == 1) local_control = "sound";
  if(dtl_control == 2) local_control = "xvel";
  if(dtl_control == 3) local_control = "yvel";
  if(dtl_control == 4) local_control = "div";

  ops_printf("local_control : %s\n",local_control);

}
