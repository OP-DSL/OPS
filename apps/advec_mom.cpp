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
#include "advec_mom_kernel.h"

void advec_mom(int which_vel, int sweep_number, int dir)
{
  //initialize sizes using global values
  int x_cells = grid->x_cells;
  int y_cells = grid->y_cells;
  int x_min = field->x_min;
  int x_max = field->x_max;
  int y_min = field->y_min;
  int y_max = field->y_max;

  int rangexy[] = {x_min-2,x_max+2,y_min-2,y_max+2}; // full range over grid


  int mom_sweep = dir + 2*(sweep_number-1);
  ops_dat vel1;

  if( which_vel == 1)
    vel1 = xvel1;
  else
    vel1 = yvel1;


  if(mom_sweep == 1) { // x 1
      ops_par_loop(advec_mom_x1_kernel, "advec_mom_x1_kernel", 2, rangexy,
        ops_arg_dat(work_array6, sten_self_2D, OPS_WRITE),
        ops_arg_dat(work_array7, sten_self_2D, OPS_RW),
        ops_arg_dat(volume, sten_self_2D, OPS_READ),
        ops_arg_dat(vol_flux_x, sten_self2D_plus1x, OPS_READ),
        ops_arg_dat(vol_flux_y, sten_self2D_plus1y, OPS_READ));
  }
  else if(mom_sweep == 2) { // y 1
    ops_par_loop(advec_mom_y1_kernel, "advec_mom_y1_kernel", 2, rangexy,
        ops_arg_dat(work_array6, sten_self_2D, OPS_WRITE),
        ops_arg_dat(work_array7, sten_self_2D, OPS_RW),
        ops_arg_dat(volume, sten_self_2D, OPS_READ),
        ops_arg_dat(vol_flux_x, sten_self2D_plus1x, OPS_READ),
        ops_arg_dat(vol_flux_y, sten_self2D_plus1y, OPS_READ));
  }
  else if (mom_sweep == 3) { // x 2
    ops_par_loop(advec_mom_x2_kernel, "advec_mom_x2_kernel", 2, rangexy,
        ops_arg_dat(work_array6, sten_self_2D, OPS_WRITE),
        ops_arg_dat(work_array7, sten_self_2D, OPS_RW),
        ops_arg_dat(volume, sten_self_2D, OPS_READ),
        ops_arg_dat(vol_flux_y, sten_self2D_plus1y, OPS_READ));
  }
  else if (mom_sweep == 4) { // y 2
    ops_par_loop(advec_mom_y2_kernel, "advec_mom_y2_kernel", 2, rangexy,
        ops_arg_dat(work_array6, sten_self_2D, OPS_WRITE),
        ops_arg_dat(work_array7, sten_self_2D, OPS_RW),
        ops_arg_dat(volume, sten_self_2D, OPS_READ),
        ops_arg_dat(vol_flux_x, sten_self2D_plus1x, OPS_READ));
  }

  ops_print_dat_to_txtfile_core(work_array6, "cloverdats.dat");
  ops_print_dat_to_txtfile_core(work_array7, "cloverdats.dat");


  if (dir == 1) {

  }


}
