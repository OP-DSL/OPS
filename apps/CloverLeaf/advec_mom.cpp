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

#include "data.h"
#include "definitions.h"
#include "advec_mom_kernel.h"


void advec_mom(int which_vel, int sweep_number, int dir)
{
  //initialize sizes using global values
  int x_min = field.x_min;
  int x_max = field.x_max;
  int y_min = field.y_min;
  int y_max = field.y_max;

  int rangexy[] = {x_min-2,x_max+2,y_min-2,y_max+2}; // full range over grid

  int mom_sweep;
  ops_dat vel1;

  if( which_vel == 1) {
    vel1 = xvel1;
  }
  else {
    vel1 = yvel1;
  }

  mom_sweep = dir + 2*(sweep_number-1);
  //printf("mom_sweep %d direction: %d sweep_number: %d\n",mom_sweep, dir, sweep_number);

  if(mom_sweep == 1) { // x 1
      ops_par_loop(advec_mom_kernel_x1, "advec_mom_kernel_x1", clover_grid, 2, rangexy,
        ops_arg_dat(work_array6, 1, S2D_00, "double", OPS_WRITE),
        ops_arg_dat(work_array7, 1, S2D_00, "double", OPS_WRITE),
        ops_arg_dat(volume, 1, S2D_00, "double", OPS_READ),
        ops_arg_dat(vol_flux_x, 1, S2D_00_P10, "double", OPS_READ),
        ops_arg_dat(vol_flux_y, 1, S2D_00_0P1, "double", OPS_READ));
  }
  else if(mom_sweep == 2) { // y 1
    ops_par_loop(advec_mom_kernel_y1, "advec_mom_kernel_y1", clover_grid, 2, rangexy,
        ops_arg_dat(work_array6, 1, S2D_00, "double", OPS_WRITE),
        ops_arg_dat(work_array7, 1, S2D_00, "double", OPS_WRITE),
        ops_arg_dat(volume, 1, S2D_00, "double", OPS_READ),
        ops_arg_dat(vol_flux_x, 1, S2D_00_P10, "double", OPS_READ),
        ops_arg_dat(vol_flux_y, 1, S2D_00_0P1, "double", OPS_READ));
  }
  else if (mom_sweep == 3) { // x 2
    ops_par_loop(advec_mom_kernel_x2, "advec_mom_kernel_x2", clover_grid, 2, rangexy,
        ops_arg_dat(work_array6, 1, S2D_00, "double", OPS_WRITE),
        ops_arg_dat(work_array7, 1, S2D_00, "double", OPS_WRITE),
        ops_arg_dat(volume, 1, S2D_00, "double", OPS_READ),
        ops_arg_dat(vol_flux_y, 1, S2D_00_0P1, "double", OPS_READ));
  }
  else if (mom_sweep == 4) { // y 2
    ops_par_loop(advec_mom_kernel_y2, "advec_mom_kernel_y2", clover_grid, 2, rangexy,
        ops_arg_dat(work_array6, 1, S2D_00, "double", OPS_WRITE),
        ops_arg_dat(work_array7, 1, S2D_00, "double", OPS_WRITE),
        ops_arg_dat(volume, 1, S2D_00, "double", OPS_READ),
        ops_arg_dat(vol_flux_x, 1, S2D_00_P10, "double", OPS_READ));
  }

  int range_fullx_party_1[] = {x_min-2,x_max+2,y_min,y_max+1}; // full x range partial y range
  int range_partx_party_1[] = {x_min-1,x_max+2,y_min,y_max+1}; // partial x range partial y range

  int range_fully_party_1[] = {x_min,x_max+1,y_min-2,y_max+2}; // full y range partial x range
  int range_partx_party_2[] = {x_min,x_max+1,y_min-1,y_max+2}; // partial x range partial y range

  if (dir == 1) {

    //Find staggered mesh mass fluxes, nodal masses and volumes.
    ops_par_loop(advec_mom_kernel_mass_flux_x, "advec_mom_kernel_mass_flux_x", clover_grid, 2, range_fullx_party_1,
        ops_arg_dat(work_array1, 1, S2D_00, "double", OPS_WRITE),
        ops_arg_dat(mass_flux_x, 1, S2D_00_P10_0M1_P1M1, "double", OPS_READ));

    //Staggered cell mass post and pre advection
    ops_par_loop(advec_mom_kernel_post_pre_advec_x, "advec_mom_kernel_post_pre_advec_x", clover_grid, 2, range_partx_party_1,
        ops_arg_dat(work_array2/*node_mass_post*/, 1, S2D_00, "double", OPS_WRITE),
        ops_arg_dat(work_array7, 1, S2D_00_M10_0M1_M1M1, "double", OPS_READ),
        ops_arg_dat(density1, 1, S2D_00_M10_0M1_M1M1, "double", OPS_READ),
        ops_arg_dat(work_array3/*node_mass_pre*/, 1, S2D_00, "double", OPS_WRITE),
        ops_arg_dat(work_array1/*node_flux*/, 1, S2D_00_M10, "double", OPS_READ));

    int range_plus1xy_minus1x[] = {x_min-1,x_max+1,y_min,y_max+1}; // partial x range partial y range
    ops_par_loop(advec_mom_kernel1_x_nonvector, "advec_mom_kernel1_x", clover_grid, 2, range_plus1xy_minus1x,
        ops_arg_dat(work_array1/*node_flux*/, 1, S2D_00, "double", OPS_READ),
        ops_arg_dat(work_array3/*node_mass_pre*/, 1, S2D_00_P10, "double", OPS_READ),
        ops_arg_dat(work_array5/*mom_flux*/, 1, S2D_00, "double", OPS_WRITE),
        ops_arg_dat(celldx, 1, S2D_00_P10_M10_STRID2D_X, "double", OPS_READ),
        ops_arg_dat(vel1, 1, S2D_00_P10_P20_M10, "double", OPS_READ));

    int range_partx_party_2[] = {x_min,x_max+1,y_min,y_max+1}; // full x range partial y range
    ops_par_loop(advec_mom_kernel2_x, "advec_mom_kernel2_x", clover_grid, 2, range_partx_party_2,
        ops_arg_dat(vel1, 1, S2D_00, "double", OPS_RW),
        ops_arg_dat(work_array2/*node_mass_post*/, 1, S2D_00, "double", OPS_READ),
        ops_arg_dat(work_array3/*node_mass_pre*/, 1, S2D_00, "double", OPS_READ),
        ops_arg_dat(work_array5/*mom_flux*/, 1, S2D_00_M10, "double", OPS_READ)
        );
  }
  else if (dir == 2) {

    //Find staggered mesh mass fluxes, nodal masses and volumes.
    ops_par_loop(advec_mom_kernel_mass_flux_y, "advec_mom_kernel_mass_flux_y", clover_grid, 2, range_fully_party_1,
        ops_arg_dat(work_array1, 1, S2D_00, "double", OPS_WRITE),
        ops_arg_dat(mass_flux_y, 1, S2D_00_0P1_M10_M1P1, "double", OPS_READ));

    //Staggered cell mass post and pre advection
    ops_par_loop(advec_mom_kernel_post_pre_advec_y, "advec_mom_kernel_post_pre_advec_y", clover_grid, 2, range_partx_party_2,
        ops_arg_dat(work_array2/*node_mass_post*/, 1, S2D_00, "double", OPS_WRITE),
        ops_arg_dat(work_array7, 1, S2D_00_M10_0M1_M1M1, "double", OPS_READ),
        ops_arg_dat(density1, 1, S2D_00_M10_0M1_M1M1, "double", OPS_READ),
        ops_arg_dat(work_array3/*node_mass_pre*/, 1, S2D_00, "double", OPS_WRITE),
        ops_arg_dat(work_array1/*node_flux*/, 1, S2D_00_0M1, "double", OPS_READ));

    int range_plus1xy_minus1y[] = {x_min,x_max+1,y_min-1,y_max+1}; // partial x range partial y range
    ops_par_loop(advec_mom_kernel1_y_nonvector, "advec_mom_kernel1_y", clover_grid, 2, range_plus1xy_minus1y,
        ops_arg_dat(work_array1/*node_flux*/, 1, S2D_00, "double", OPS_READ),
        ops_arg_dat(work_array3/*node_mass_pre*/, 1, S2D_00_0P1, "double", OPS_READ),
        ops_arg_dat(work_array5/*mom_flux*/, 1, S2D_00, "double", OPS_WRITE),
        ops_arg_dat(celldy, 1, S2D_00_0P1_0M1_STRID2D_Y, "double", OPS_READ),
        ops_arg_dat(vel1, 1, S2D_00_0P1_0P2_0M1, "double", OPS_READ));

    int range_partx_party_2[] = {x_min,x_max+1,y_min,y_max+1}; // full x range partial y range
    ops_par_loop(advec_mom_kernel2_y, "advec_mom_kernel2_y", clover_grid, 2, range_partx_party_2,
        ops_arg_dat(vel1, 1, S2D_00, "double", OPS_RW),
        ops_arg_dat(work_array2/*node_mass_post*/, 1, S2D_00, "double", OPS_READ),
        ops_arg_dat(work_array3/*node_mass_pre*/, 1, S2D_00, "double", OPS_READ),
        ops_arg_dat(work_array5/*mom_flux*/, 1, S2D_00_0M1, "double", OPS_READ));

  }

}
