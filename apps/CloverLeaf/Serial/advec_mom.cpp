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

/** @brief momentum advection
 *  @author Wayne Gaudin, converted to OPS by Gihan Mudalige
 *  @details Performs a second order advective remap on the vertex momentum
 *  using van-Leer limiting and directional splitting.
 *  Note that although pre_vol is only set and not used in the update, please
 *  leave it in the method.
**/

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

// OPS header file
#include "ops_seq.h"
#include "ops_seq_opt.h"

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

  int mom_sweep;
  ops_dat vel1;

  int vector = TRUE; //currently always use vector loops .. need to set this in input

  if( which_vel == 1) {
    vel1 = xvel1;
  }
  else {
    vel1 = yvel1;
  }

  mom_sweep = dir + 2*(sweep_number-1);
  //printf("mom_sweep %d direction: %d sweep_number: %d\n",mom_sweep, dir, sweep_number);

  if(mom_sweep == 1) { // x 1
      ops_par_loop_opt(advec_mom_x1_kernel, "advec_mom_x1_kernel", 2, rangexy,
        ops_arg_dat(work_array6, S2D_00, OPS_WRITE),
        ops_arg_dat(work_array7, S2D_00, OPS_RW),
        ops_arg_dat(volume, S2D_00, OPS_READ),
        ops_arg_dat(vol_flux_x, S2D_00_P10, OPS_READ),
        ops_arg_dat(vol_flux_y, S2D_00_0P1, OPS_READ));
  }
  else if(mom_sweep == 2) { // y 1
    ops_par_loop_opt(advec_mom_y1_kernel, "advec_mom_y1_kernel", 2, rangexy,
        ops_arg_dat(work_array6, S2D_00, OPS_WRITE),
        ops_arg_dat(work_array7, S2D_00, OPS_RW),
        ops_arg_dat(volume, S2D_00, OPS_READ),
        ops_arg_dat(vol_flux_x, S2D_00_P10, OPS_READ),
        ops_arg_dat(vol_flux_y, S2D_00_0P1, OPS_READ));
  }
  else if (mom_sweep == 3) { // x 2
    ops_par_loop_opt(advec_mom_x2_kernel, "advec_mom_x2_kernel", 2, rangexy,
        ops_arg_dat(work_array6, S2D_00, OPS_WRITE),
        ops_arg_dat(work_array7, S2D_00, OPS_RW),
        ops_arg_dat(volume, S2D_00, OPS_READ),
        ops_arg_dat(vol_flux_y, S2D_00_0P1, OPS_READ));
  }
  else if (mom_sweep == 4) { // y 2
    ops_par_loop_opt(advec_mom_y2_kernel, "advec_mom_y2_kernel", 2, rangexy,
        ops_arg_dat(work_array6, S2D_00, OPS_WRITE),
        ops_arg_dat(work_array7, S2D_00, OPS_RW),
        ops_arg_dat(volume, S2D_00, OPS_READ),
        ops_arg_dat(vol_flux_x, S2D_00_P10, OPS_READ));
  }

  int range_fullx_party_1[] = {x_min-2,x_max+2,y_min,y_max+1}; // full x range partial y range
  int range_partx_party_1[] = {x_min-1,x_max+2,y_min,y_max+1}; // partial x range partial y range

  int range_fully_party_1[] = {x_min,x_max+1,y_min-2,y_max+2}; // full y range partial x range
  int range_partx_party_2[] = {x_min,x_max+1,y_min-1,y_max+2}; // partial x range partial y range

  if (dir == 1) {

    //Find staggered mesh mass fluxes, nodal masses and volumes.
    ops_par_loop_opt(advec_mom_mass_flux_kernel_x, "advec_mom_mass_flux_kernel_x", 2, range_fullx_party_1,
        ops_arg_dat(work_array1, S2D_00, OPS_WRITE),
        ops_arg_dat(mass_flux_x, sten_self2D_plus1x_minus1y, OPS_READ));

    //Staggered cell mass post advection
    ops_par_loop_opt(advec_mom_post_advec_kernel_x, "advec_mom_post_advec_kernel_x", 2, range_partx_party_1,
        ops_arg_dat(work_array2, S2D_00, OPS_WRITE),
        ops_arg_dat(work_array7, sten_self2D_minus1xy, OPS_READ),
        ops_arg_dat(density1, sten_self2D_minus1xy, OPS_READ));

    //Stagered cell mass pre advection
    ops_par_loop_opt(advec_mom_pre_advec_kernel_x, "advec_mom_pre_advec_kernel_x", 2, range_partx_party_1,
        ops_arg_dat(work_array3/*node_mass_pre*/, S2D_00, OPS_WRITE),
        ops_arg_dat(work_array2/*node_mass_post*/, S2D_00, OPS_READ),
        ops_arg_dat(work_array1/*node_flux*/, S2D_00_M10, OPS_READ));


    int range_plus1xy_minus1x[] = {x_min-1,x_max+1,y_min,y_max+1}; // partial x range partial y range
    if(vector) {

      ops_par_loop_opt(advec_mom_kernel1_x, "advec_mom_kernel1_x", 2, range_plus1xy_minus1x,
        ops_arg_dat(work_array1/*node_flux*/, S2D_00, OPS_READ),
        ops_arg_dat(work_array3/*node_mass_pre*/, S2D_00_P10, OPS_WRITE),
        ops_arg_dat(work_array4/*advec_vel*/, S2D_00, OPS_RW),
        ops_arg_dat(work_array5/*mom_flux*/, S2D_00, OPS_WRITE),
        ops_arg_dat(celldx, sten_self_plus_1_minus1_2_x_stride2D_x, OPS_READ),
        ops_arg_dat(vel1, sten_self2D_plus_1_2_minus1x, OPS_READ));

    }
    else {
      //currently ignor this section
    }

    int range_partx_party_2[] = {x_min,x_max+1,y_min,y_max+1}; // full x range partial y range
    ops_par_loop_opt(advec_mom_kernel2_x, "advec_mom_kernel2_x", 2, range_partx_party_2,
        ops_arg_dat(vel1, S2D_00, OPS_WRITE),
        ops_arg_dat(work_array2/*node_mass_post*/, S2D_00, OPS_READ),
        ops_arg_dat(work_array3/*node_mass_pre*/, S2D_00, OPS_READ),
        ops_arg_dat(work_array5/*mom_flux*/, S2D_00_M10, OPS_READ)
        );
  }
  else if (dir == 2) {

    //Find staggered mesh mass fluxes, nodal masses and volumes.
    ops_par_loop_opt(advec_mom_mass_flux_kernel_y, "advec_mom_mass_flux_kernel_y", 2, range_fully_party_1,
        ops_arg_dat(work_array1, S2D_00, OPS_WRITE),
        ops_arg_dat(mass_flux_y, sten_self2D_plus1y_minus1x, OPS_READ));

    //Staggered cell mass post advection
    ops_par_loop_opt(advec_mom_post_advec_kernel_y, "advec_mom_post_advec_kernel", 2, range_partx_party_2,
        ops_arg_dat(work_array2, S2D_00, OPS_WRITE),
        ops_arg_dat(work_array7, sten_self2D_minus1xy, OPS_READ),
        ops_arg_dat(density1, sten_self2D_minus1xy, OPS_READ));

    //Stagered cell mass pre advection
    ops_par_loop_opt(advec_mom_pre_advec_kernel_y, "advec_mom_pre_advec_kernel_y", 2, range_partx_party_2,
        ops_arg_dat(work_array3/*node_mass_pre*/, S2D_00, OPS_WRITE),
        ops_arg_dat(work_array2/*node_mass_post*/, S2D_00, OPS_READ),
        ops_arg_dat(work_array1/*node_flux*/, S2D_00_0M1, OPS_READ));

    int range_plus1xy_minus1y[] = {x_min,x_max+1,y_min-1,y_max+1}; // partial x range partial y range
    if(vector) {
        ops_par_loop_opt(advec_mom_kernel1_y, "advec_mom_kernel1_y", 2, range_plus1xy_minus1y,
        ops_arg_dat(work_array1/*node_flux*/, S2D_00, OPS_READ),
        ops_arg_dat(work_array3/*node_mass_pre*/, S2D_00_0P1, OPS_WRITE),
        ops_arg_dat(work_array4/*advec_vel*/, S2D_00, OPS_RW),
        ops_arg_dat(work_array5/*mom_flux*/, S2D_00, OPS_WRITE),
        ops_arg_dat(celldy, sten_self_plus_1_minus1_2_y_stride2D_y, OPS_READ),
        ops_arg_dat(vel1, sten_self2D_plus_1_2_minus1y, OPS_READ));
    }
    else {
      //currently ignor this section
    }

    int range_partx_party_2[] = {x_min,x_max+1,y_min,y_max+1}; // full x range partial y range
    ops_par_loop_opt(advec_mom_kernel2_y, "advec_mom_kernel2_y", 2, range_partx_party_2,
        ops_arg_dat(vel1, S2D_00, OPS_WRITE),
        ops_arg_dat(work_array2/*node_mass_post*/, S2D_00, OPS_READ),
        ops_arg_dat(work_array3/*node_mass_pre*/, S2D_00, OPS_READ),
        ops_arg_dat(work_array5/*mom_flux*/, S2D_00_0M1, OPS_READ)
        );

  }

}
