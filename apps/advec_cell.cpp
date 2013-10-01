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

/** @brief cell advection
 *  @author Wayne Gaudin, converted to OPS by Gihan Mudalige
 *  @details Performs a second order advective remap using van-Leer limiting
 *  with directional splitting.
**/

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

// OPS header file
#include "ops_seq.h"

#include "data.h"
#include "definitions.h"
#include "advec_cell_kernel.h"

void advec_cell(int sweep_number, int dir)
{
  //initialize sizes using global values
  int x_cells = grid->x_cells;
  int y_cells = grid->y_cells;
  int x_min = field->x_min;
  int x_max = field->x_max;
  int y_min = field->y_min;
  int y_max = field->y_max;

  int rangexy[] = {x_min-2,x_max+2,y_min-2,y_max+2}; // full range over grid
  int rangexy_inner[] = {x_min,x_max,y_min,y_max}; // inner range without border

  int rangexy_inner_plus2x[] = {x_min,x_max+2,y_min,y_max}; // inner range with +2 in x
  int rangexy_inner_plus2y[] = {x_min,x_max,y_min,y_max+2}; // inner range with +2 in y

  if(dir == g_xdir) {

    if(sweep_number == 1) {
      ops_par_loop(advec_cell_xdir_kernel1, "advec_cell_xdir_kernel1", 2, rangexy,
        ops_arg_dat(work_array1, S2D_00, OPS_RW),
        ops_arg_dat(work_array2, S2D_00, OPS_WRITE),
        ops_arg_dat(volume, S2D_00, OPS_READ),
        ops_arg_dat(vol_flux_x, sten_self2D_plus1x, OPS_READ),
        ops_arg_dat(vol_flux_y, sten_self2D_plus1y, OPS_READ)
        );
    }
    else {
      ops_par_loop(advec_cell_xdir_kernel2, "advec_cell_xdir_kernel2", 2, rangexy,
        ops_arg_dat(work_array1, S2D_00, OPS_WRITE),
        ops_arg_dat(work_array2, S2D_00, OPS_WRITE),
        ops_arg_dat(volume, S2D_00, OPS_READ),
        ops_arg_dat(vol_flux_x, sten_self2D_plus1x, OPS_READ)
        );
    }

    ops_par_loop(advec_cell_xdir_kernel3, "advec_cell_xdir_kernel3", 2, rangexy_inner_plus2x,
      ops_arg_dat(vol_flux_x, S2D_00, OPS_READ),
      ops_arg_dat(work_array1, sten_self2D_plus_1_minus1_2_x, OPS_READ),
      ops_arg_dat(xx, sten_self_plus1_stride2D_x, OPS_READ),
      ops_arg_dat(vertexdx, sten_self_plus_1_minus1_2_x_stride2D_x, OPS_READ),
      ops_arg_dat(density1, sten_self2D_plus_1_minus1_2_x, OPS_READ),
      ops_arg_dat(energy1, sten_self2D_plus_1_minus1_2_x, OPS_READ),
      ops_arg_dat(mass_flux_x, S2D_00, OPS_WRITE),
      ops_arg_dat(work_array7, S2D_00, OPS_WRITE)
      );

    /*ops_par_loop(advec_cell_xdir_kernel3a, "advec_cell_xdir_kernel3a", 2, rangexy_inner_plus2x,
      ops_arg_dat(vol_flux_x, S2D_00, OPS_READ),
      ops_arg_dat(work_array1, sten_self2D_plus_1_minus1_2_x, OPS_READ),
      ops_arg_dat(xx, sten_self_plus1_stride2D_x, OPS_READ),

      ops_arg_dat(vertexdx, sten_self_plus_1_minus1_2_x_stride2D_x, OPS_READ),
      ops_arg_dat(vertexdx, sten_self_nullstride2D_xmax, OPS_READ),

      ops_arg_dat(density1, sten_self2D_plus_1_minus1_2_x, OPS_READ),
      ops_arg_dat(density1, sten_self_stride2D_xmax, OPS_READ),

      ops_arg_dat(energy1, sten_self2D_plus_1_minus1_2_x, OPS_READ),
      ops_arg_dat(energy1, sten_self_stride2D_xmax, OPS_READ),

      ops_arg_dat(mass_flux_x, S2D_00, OPS_WRITE),
      ops_arg_dat(work_array7, S2D_00, OPS_WRITE)
      );*/

    ops_par_loop(advec_cell_xdir_kernel4, "advec_cell_xdir_kernel4", 2, rangexy_inner,
      ops_arg_dat(density1, S2D_00, OPS_RW),
      ops_arg_dat(energy1, S2D_00, OPS_RW),
      ops_arg_dat(mass_flux_x, sten_self2D_plus1x, OPS_READ),
      ops_arg_dat(vol_flux_x, sten_self2D_plus1x, OPS_READ),
      ops_arg_dat(work_array1, S2D_00, OPS_READ),
      ops_arg_dat(work_array2, S2D_00, OPS_READ),
      ops_arg_dat(work_array3, S2D_00, OPS_RW),
      ops_arg_dat(work_array4, S2D_00, OPS_RW),
      ops_arg_dat(work_array5, S2D_00, OPS_RW),
      ops_arg_dat(work_array6, S2D_00, OPS_RW),
      ops_arg_dat(work_array7, sten_self2D_plus1x, OPS_READ)
      );

  }
  else {

    if(sweep_number == 1) {
      ops_par_loop(advec_cell_ydir_kernel1, "advec_cell_ydir_kernel1", 2, rangexy,
        ops_arg_dat(work_array1, S2D_00, OPS_RW),
        ops_arg_dat(work_array2, S2D_00, OPS_WRITE),
        ops_arg_dat(volume, S2D_00, OPS_READ),
        ops_arg_dat(vol_flux_x, sten_self2D_plus1x, OPS_READ),
        ops_arg_dat(vol_flux_y, sten_self2D_plus1y, OPS_READ)
        );
    }
    else {
      ops_par_loop(advec_cell_ydir_kernel2, "advec_cell_ydir_kernel2", 2, rangexy,
        ops_arg_dat(work_array1, S2D_00, OPS_WRITE),
        ops_arg_dat(work_array2, S2D_00, OPS_WRITE),
        ops_arg_dat(volume, S2D_00, OPS_READ),
        ops_arg_dat(vol_flux_y, sten_self2D_plus1x, OPS_READ)
        );
    }

    ops_par_loop(advec_cell_ydir_kernel3, "advec_cell_ydir_kernel3", 2, rangexy_inner_plus2y,
      ops_arg_dat(vol_flux_y, S2D_00, OPS_READ),
      ops_arg_dat(work_array1, sten_self2D_plus_1_minus1_2_y, OPS_READ),
      ops_arg_dat(yy, sten_self_plus1_stride2D_y, OPS_READ),
      ops_arg_dat(vertexdy, sten_self_plus_1_minus1_2_y_stride2D_y, OPS_READ),
      ops_arg_dat(density1, sten_self2D_plus_1_minus1_2_y, OPS_READ),
      ops_arg_dat(energy1, sten_self2D_plus_1_minus1_2_y, OPS_READ),
      ops_arg_dat(mass_flux_y, S2D_00, OPS_WRITE),
      ops_arg_dat(work_array7, S2D_00, OPS_WRITE)
      );

    /*ops_par_loop(advec_cell_ydir_kernel3a, "advec_cell_ydir_kernel3a", 2, rangexy_inner_plus2y,
      ops_arg_dat(vol_flux_y, S2D_00, OPS_READ),
      ops_arg_dat(work_array1, sten_self2D_plus_1_minus1_2_y, OPS_READ),
      ops_arg_dat(yy, sten_self_plus1_stride2D_y, OPS_READ),

      ops_arg_dat(vertexdy, sten_self_plus_1_minus1_2_y_stride2D_y, OPS_READ),
      ops_arg_dat(vertexdy, sten_self_nullstride2D_ymax, OPS_READ),

      ops_arg_dat(density1, sten_self2D_plus_1_minus1_2_y, OPS_READ),
      ops_arg_dat(density1, sten_self_stride2D_ymax, OPS_READ),

      ops_arg_dat(energy1, sten_self2D_plus_1_minus1_2_y, OPS_READ),
      ops_arg_dat(energy1, sten_self_stride2D_ymax, OPS_READ),

      ops_arg_dat(mass_flux_y, S2D_00, OPS_WRITE),
      ops_arg_dat(work_array7, S2D_00, OPS_WRITE)
      );*/


    ops_par_loop(advec_cell_ydir_kernel4, "advec_cell_ydir_kernel4", 2, rangexy_inner,
      ops_arg_dat(density1, S2D_00, OPS_RW),
      ops_arg_dat(energy1, S2D_00, OPS_RW),
      ops_arg_dat(mass_flux_y, sten_self2D_plus1y, OPS_READ),
      ops_arg_dat(vol_flux_y, sten_self2D_plus1y, OPS_READ),
      ops_arg_dat(work_array1, S2D_00, OPS_READ),
      ops_arg_dat(work_array2, S2D_00, OPS_READ),
      ops_arg_dat(work_array3, S2D_00, OPS_RW),
      ops_arg_dat(work_array4, S2D_00, OPS_RW),
      ops_arg_dat(work_array5, S2D_00, OPS_RW),
      ops_arg_dat(work_array6, S2D_00, OPS_RW),
      ops_arg_dat(work_array7, sten_self2D_plus1y, OPS_READ)
      );

  }

}
