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
 *  using van-Leer limiting and diral splitting.
 *  Note that although pre_vol is only set and not used in the update, please
 *  leave it in the method.
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
#include "advec_mom_kernel.h"


void advec_mom(int which_vel, int sweep_number, int dir)
{
  //initialize sizes using global values

  int x_min = field.x_min;
  int x_max = field.x_max;
  int y_min = field.y_min;
  int y_max = field.y_max;
  int z_min = field.z_min;
  int z_max = field.z_max;

  int rangexyz[] = {x_min-2,x_max+2,y_min-2,y_max+2,z_min-2,z_max+2}; // full range over grid
  ops_dat vel1;

  if( which_vel == 1) {
    vel1 = xvel1;
  }
  else if( which_vel == 2) {
    vel1 = yvel1;
  }
  else if( which_vel == 3) {
    vel1 = zvel1;
  }

  if(sweep_number==1 && dir == 1) {
      ops_par_loop(advec_mom_kernel_x1, "advec_mom_kernel_x1", clover_grid, 3, rangexyz,
        ops_arg_dat(work_array6, 1, S3D_000, "double", OPS_WRITE),
        ops_arg_dat(work_array7, 1, S3D_000, "double", OPS_WRITE),
        ops_arg_dat(volume, 1, S3D_000, "double", OPS_READ),
        ops_arg_dat(vol_flux_x, 1, S3D_000_P100, "double", OPS_READ),
        ops_arg_dat(vol_flux_y, 1, S3D_000_0P10, "double", OPS_READ),
        ops_arg_dat(vol_flux_z, 1, S3D_000_00P1, "double", OPS_READ));
  }
  else if(sweep_number==1 && dir == 3) {
    ops_par_loop(advec_mom_kernel_z1, "advec_mom_kernel_z1", clover_grid, 3, rangexyz,
        ops_arg_dat(work_array6, 1, S3D_000, "double", OPS_WRITE),
        ops_arg_dat(work_array7, 1, S3D_000, "double", OPS_WRITE),
        ops_arg_dat(volume, 1, S3D_000, "double", OPS_READ),
        ops_arg_dat(vol_flux_x, 1, S3D_000_P100, "double", OPS_READ),
        ops_arg_dat(vol_flux_y, 1, S3D_000_0P10, "double", OPS_READ),
        ops_arg_dat(vol_flux_z, 1, S3D_000_00P1, "double", OPS_READ));
  }
  else if (sweep_number==2 && advect_x) {
    ops_par_loop(advec_mom_kernel_x2, "advec_mom_kernel_x2", clover_grid, 3, rangexyz,
        ops_arg_dat(work_array6, 1, S3D_000, "double", OPS_WRITE),
        ops_arg_dat(work_array7, 1, S3D_000, "double", OPS_WRITE),
        ops_arg_dat(volume, 1, S3D_000, "double", OPS_READ),
        ops_arg_dat(vol_flux_y, 1, S3D_000_0P10, "double", OPS_READ),
        ops_arg_dat(vol_flux_z, 1, S3D_000_00P1, "double", OPS_READ));
  }
  else if (sweep_number==2 && !advect_x) {
    ops_par_loop(advec_mom_kernel_y2, "advec_mom_kernel_y2", clover_grid, 3, rangexyz,
        ops_arg_dat(work_array6, 1, S3D_000, "double", OPS_WRITE),
        ops_arg_dat(work_array7, 1, S3D_000, "double", OPS_WRITE),
        ops_arg_dat(volume, 1, S3D_000, "double", OPS_READ),
        ops_arg_dat(vol_flux_x, 1, S3D_000_P100, "double", OPS_READ),
        ops_arg_dat(vol_flux_y, 1, S3D_000_0P10, "double", OPS_READ));
  }
  else if (sweep_number==3 && dir == 1) {
    ops_par_loop(advec_mom_kernel_x3, "advec_mom_kernel_x3", clover_grid, 3, rangexyz,
        ops_arg_dat(work_array6, 1, S3D_000, "double", OPS_WRITE),
        ops_arg_dat(work_array7, 1, S3D_000, "double", OPS_WRITE),
        ops_arg_dat(volume, 1, S3D_000, "double", OPS_READ),
        ops_arg_dat(vol_flux_x, 1, S3D_000_P100, "double", OPS_READ));
  }
  else if (sweep_number==3 && dir == 3) {
    ops_par_loop(advec_mom_kernel_z3, "advec_mom_kernel_z3", clover_grid, 3, rangexyz,
        ops_arg_dat(work_array6, 1, S3D_000, "double", OPS_WRITE),
        ops_arg_dat(work_array7, 1, S3D_000, "double", OPS_WRITE),
        ops_arg_dat(volume, 1, S3D_000, "double", OPS_READ),
        ops_arg_dat(vol_flux_z, 1, S3D_000_00P1, "double", OPS_READ));
  }

  if (dir == 1) {
    if (which_vel == 1) {
      //Find staggered mesh mass fluxes, nodal masses and volumes.
      int range_fullx_party_partz_1[] = {x_min-2,x_max+2,y_min,y_max+1,z_min,z_max+1}; // full x range partial y,z range
      ops_par_loop(advec_mom_kernel_mass_flux_x, "advec_mom_kernel_mass_flux_x", clover_grid, 3, range_fullx_party_partz_1,
          ops_arg_dat(work_array1, 1, S3D_000, "double", OPS_WRITE),
          ops_arg_dat(mass_flux_x, 1, S3D_000_fP1M1M1, "double", OPS_READ));

      //Staggered cell mass post and pre advection
      int range_partx_party_partz_1[] = {x_min-1,x_max+2,y_min,y_max+1,z_min,z_max+1}; // partial x,y,z range
      ops_par_loop(advec_mom_kernel_post_pre_advec_x, "advec_mom_kernel_post_pre_advec_x", clover_grid, 3, range_partx_party_partz_1,
          ops_arg_dat(work_array2/*node_mass_post*/, 1, S3D_000, "double", OPS_WRITE),
          ops_arg_dat(work_array7/*post_vol*/, 1, S3D_000_fM1M1M1, "double", OPS_READ),
          ops_arg_dat(density1, 1, S3D_000_fM1M1M1, "double", OPS_READ),
          ops_arg_dat(work_array3/*node_mass_pre*/, 1, S3D_000, "double", OPS_WRITE),
          ops_arg_dat(work_array1/*node_flux*/, 1, S3D_000_M100, "double", OPS_READ));
    }

    int range_innder_plus1xyz_minus1x[] = {x_min-1,x_max+1,y_min,y_max+1,z_min,z_max+1}; // partial x range partial y range
    ops_par_loop(advec_mom_kernel1_x_nonvector, "advec_mom_kernel1_x", clover_grid, 3, range_innder_plus1xyz_minus1x,
        ops_arg_dat(work_array1/*node_flux*/, 1, S3D_000, "double", OPS_READ),
        ops_arg_dat(work_array3/*node_mass_pre*/, 1, S3D_000_P100, "double", OPS_READ),
        ops_arg_dat(work_array5/*mom_flux*/, 1, S3D_000, "double", OPS_WRITE),
        ops_arg_dat(celldx, 1, S3D_000_P100_M100_STRID3D_X, "double", OPS_READ),
        ops_arg_dat(vel1, 1, S3D_000_P100_P200_M100, "double", OPS_READ));

    int range_partx_party_partz_2[] = {x_min,x_max+1,y_min,y_max+1,z_min,z_max+1};
    ops_par_loop(advec_mom_kernel2_x, "advec_mom_kernel2_x", clover_grid, 3, range_partx_party_partz_2,
        ops_arg_dat(vel1, 1, S3D_000, "double", OPS_RW),
        ops_arg_dat(work_array2/*node_mass_post*/, 1, S3D_000, "double", OPS_READ),
        ops_arg_dat(work_array3/*node_mass_pre*/, 1, S3D_000, "double", OPS_READ),
        ops_arg_dat(work_array5/*mom_flux*/, 1, S3D_000_M100, "double", OPS_READ)
        );
  }
  else if (dir == 2) {
    if (which_vel == 1) {
      //Find staggered mesh mass fluxes, nodal masses and volumes.
      int range_fully_partx_partz_1[] = {x_min,x_max+1,y_min-2,y_max+2,z_min,z_max+1}; // full x range partial y,z range
      ops_par_loop(advec_mom_kernel_mass_flux_y, "advec_mom_kernel_mass_flux_y", clover_grid, 3, range_fully_partx_partz_1,
          ops_arg_dat(work_array1, 1, S3D_000, "double", OPS_WRITE),
          ops_arg_dat(mass_flux_y, 1, S3D_000_fM1P1M1, "double", OPS_READ));

      //Staggered cell mass post and pre advection
      int range_party_partx_partz_1[] = {x_min,x_max+1,y_min-1,y_max+2,z_min,z_max+1}; // full x range partial y,z range
      ops_par_loop(advec_mom_kernel_post_pre_advec_y, "advec_mom_kernel_post_pre_advec_y", clover_grid, 3, range_party_partx_partz_1,
          ops_arg_dat(work_array2/*node_mass_post*/, 1, S3D_000, "double", OPS_WRITE),
          ops_arg_dat(work_array7, 1, S3D_000_fM1M1M1, "double", OPS_READ),
          ops_arg_dat(density1, 1, S3D_000_fM1M1M1, "double", OPS_READ),
          ops_arg_dat(work_array3/*node_mass_pre*/, 1, S3D_000, "double", OPS_WRITE),
          ops_arg_dat(work_array1/*node_flux*/, 1, S3D_000_0M10, "double", OPS_READ));
    }
    int range_plus1xyz_minus1y[] = {x_min,x_max+1,y_min-1,y_max+1,z_min,z_max+1}; // partial x range partial y range
    ops_par_loop(advec_mom_kernel1_y_nonvector, "advec_mom_kernel1_y", clover_grid, 3, range_plus1xyz_minus1y,
        ops_arg_dat(work_array1/*node_flux*/, 1, S3D_000, "double", OPS_READ),
        ops_arg_dat(work_array3/*node_mass_pre*/, 1, S3D_000_0P10, "double", OPS_READ),
        ops_arg_dat(work_array5/*mom_flux*/, 1, S3D_000, "double", OPS_WRITE),
        ops_arg_dat(celldy, 1, S3D_000_0P10_0M10_STRID3D_Y, "double", OPS_READ),
        ops_arg_dat(vel1, 1, S3D_000_0P10_0P20_0M10, "double", OPS_READ));

    int range_partx_party_partz_2[] = {x_min,x_max+1,y_min,y_max+1,z_min,z_max+1};
    ops_par_loop(advec_mom_kernel2_y, "advec_mom_kernel2_y", clover_grid, 3, range_partx_party_partz_2,
        ops_arg_dat(vel1, 1, S3D_000, "double", OPS_RW),
        ops_arg_dat(work_array2/*node_mass_post*/, 1, S3D_000, "double", OPS_READ),
        ops_arg_dat(work_array3/*node_mass_pre*/, 1, S3D_000, "double", OPS_READ),
        ops_arg_dat(work_array5/*mom_flux*/, 1, S3D_000_0M10, "double", OPS_READ));

  }
  else if (dir == 3) {
    if (which_vel == 1) {
      //Find staggered mesh mass fluxes, nodal masses and volumes.
      int range_fullz_partx_party_1[] = {x_min,x_max+1,y_min,y_max+1,z_min-2,z_max+2}; // full x range partial y,z range
      ops_par_loop(advec_mom_kernel_mass_flux_z, "advec_mom_kernel_mass_flux_z", clover_grid, 3, range_fullz_partx_party_1,
          ops_arg_dat(work_array1, 1, S3D_000, "double", OPS_WRITE),
          ops_arg_dat(mass_flux_z, 1, S3D_000_fM1M1P1, "double", OPS_READ));

      //Staggered cell mass post and pre advection
      int range_party_partx_partz_1[] = {x_min,x_max+1,y_min,y_max+1,z_min-1,z_max+2}; // full x range partial y,z range
      ops_par_loop(advec_mom_kernel_post_pre_advec_z, "advec_mom_kernel_post_pre_advec_z", clover_grid, 3, range_party_partx_partz_1,
          ops_arg_dat(work_array2/*node_mass_post*/, 1, S3D_000, "double", OPS_WRITE),
          ops_arg_dat(work_array7, 1, S3D_000_fM1M1M1, "double", OPS_READ),
          ops_arg_dat(density1, 1, S3D_000_fM1M1M1, "double", OPS_READ),
          ops_arg_dat(work_array3/*node_mass_pre*/, 1, S3D_000, "double", OPS_WRITE),
          ops_arg_dat(work_array1/*node_flux*/, 1, S3D_000_00M1, "double", OPS_READ));
    }
    int range_plus1xyz_minus1z[] = {x_min,x_max+1,y_min,y_max+1,z_min-1,z_max+1}; // partial x range partial y range
    ops_par_loop(advec_mom_kernel1_z_nonvector, "advec_mom_kernel1_z", clover_grid, 3, range_plus1xyz_minus1z,
        ops_arg_dat(work_array1/*node_flux*/, 1, S3D_000, "double", OPS_READ),
        ops_arg_dat(work_array3/*node_mass_pre*/, 1, S3D_000_00P1, "double", OPS_READ),
        ops_arg_dat(work_array5/*mom_flux*/, 1, S3D_000, "double", OPS_WRITE),
        ops_arg_dat(celldz, 1, S3D_000_00P1_00M1_STRID3D_Z, "double", OPS_READ),
        ops_arg_dat(vel1, 1, S3D_000_00P1_00P2_00M1, "double", OPS_READ));

    int range_partx_party_partz_2[] = {x_min,x_max+1,y_min,y_max+1,z_min,z_max+1}; // full x range partial y range
    ops_par_loop(advec_mom_kernel2_z, "advec_mom_kernel2_z", clover_grid, 3, range_partx_party_partz_2,
        ops_arg_dat(vel1, 1, S3D_000, "double", OPS_RW),
        ops_arg_dat(work_array2/*node_mass_post*/, 1, S3D_000, "double", OPS_READ),
        ops_arg_dat(work_array3/*node_mass_pre*/, 1, S3D_000, "double", OPS_READ),
        ops_arg_dat(work_array5/*mom_flux*/, 1, S3D_000_00M1, "double", OPS_READ));

  }
}
