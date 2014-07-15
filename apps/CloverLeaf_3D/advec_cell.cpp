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

#define OPS_3D
#include "ops_seq.h"

#include "data.h"
#include "definitions.h"
#include "advec_cell_kernel.h"



void advec_cell(int sweep_number, int dir)
{
  //initialize sizes using global values
  int x_min = field.x_min;
  int x_max = field.x_max;
  int y_min = field.y_min;
  int y_max = field.y_max;
  int z_min = field.z_min;
  int z_max = field.z_max;
  
  int rangexyz[] = {x_min-2,x_max+2,y_min-2,y_max+2,z_min-2,z_max+2}; // full range over grid
  int rangexyz_inner[] = {x_min,x_max,y_min,y_max,z_min,z_max}; // inner range without border

  int rangexyz_inner_plus2x[] = {x_min,x_max+2,y_min,y_max,z_min,z_max}; // inner range with +2 in x
  int rangexyz_inner_plus2yz[] = {x_min,x_max,y_min,y_max+2,z_min,z_max+2}; // inner range with +2 in y and z
  int rangexyz_inner_plus2z[] = {x_min,x_max,y_min,y_max,z_min,z_max+2}; // inner range with +2 in z

  //printf("direction: %d sweep_number %d \n", dir, sweep_number);

  if(dir == g_xdir) {

    if(sweep_number == 1) {
      ops_par_loop(advec_cell_kernel1_xdir, "advec_cell_kernel1_xdir", clover_grid, 3, rangexyz,
        ops_arg_dat(work_array1, S3D_000, "double", OPS_WRITE),
        ops_arg_dat(work_array2, S3D_000, "double", OPS_WRITE),
        ops_arg_dat(volume, S3D_000, "double", OPS_READ),
        ops_arg_dat(vol_flux_x, S3D_000_P100, "double", OPS_READ),
        ops_arg_dat(vol_flux_y, S3D_000_0P10, "double", OPS_READ),
        ops_arg_dat(vol_flux_z, S3D_000_00P1, "double", OPS_READ));
    }
    else if (sweep_number == 3) {
      ops_par_loop(advec_cell_kernel2_xdir, "advec_cell_kernel2_xdir", clover_grid, 3, rangexyz,
        ops_arg_dat(work_array1, S3D_000, "double", OPS_WRITE),
        ops_arg_dat(work_array2, S3D_000, "double", OPS_WRITE),
        ops_arg_dat(volume, S3D_000, "double", OPS_READ),
        ops_arg_dat(vol_flux_x, S3D_000_P100, "double", OPS_READ));
    }

    ops_par_loop(advec_cell_kernel3_xdir, "advec_cell_kernel3_xdir", clover_grid, 3, rangexyz_inner_plus2x,
      ops_arg_dat(vol_flux_x, S3D_000, "double", OPS_READ),
      ops_arg_dat(work_array1, S3D_000_M100, "double", OPS_READ),
      ops_arg_dat(xx, S3D_000_P100_STRID3D_X, "int", OPS_READ),
      ops_arg_dat(vertexdx, S3D_000_P100_M100_STRID3D_X, "double", OPS_READ),
      ops_arg_dat(density1, S3D_000_P100_M100_M200, "double", OPS_READ),
      ops_arg_dat(energy1, S3D_000_P100_M100_M200, "double", OPS_READ),
      ops_arg_dat(mass_flux_x, S3D_000, "double", OPS_WRITE),
      ops_arg_dat(work_array7, S3D_000, "double", OPS_WRITE));

    ops_par_loop(advec_cell_kernel4_xdir, "advec_cell_kernel4_xdir", clover_grid, 3, rangexyz_inner,
      ops_arg_dat(density1, S3D_000, "double", OPS_RW),
      ops_arg_dat(energy1, S3D_000, "double", OPS_RW),
      ops_arg_dat(mass_flux_x, S3D_000_P100, "double", OPS_READ),
      ops_arg_dat(vol_flux_x, S3D_000_P100, "double", OPS_READ),
      ops_arg_dat(work_array1, S3D_000, "double", OPS_READ),
      ops_arg_dat(work_array2, S3D_000, "double", OPS_READ),
      ops_arg_dat(work_array3, S3D_000, "double", OPS_WRITE),
      ops_arg_dat(work_array4, S3D_000, "double", OPS_WRITE),
      ops_arg_dat(work_array5, S3D_000, "double", OPS_WRITE),
      ops_arg_dat(work_array6, S3D_000, "double", OPS_WRITE),
      ops_arg_dat(work_array7, S3D_000_P100, "double", OPS_READ));

  }
  else if(dir == g_ydir) {
    if(sweep_number == 2) {
      if (advect_x) {
      ops_par_loop(advec_cell_kernel1_ydir, "advec_cell_kernel1_ydir", clover_grid, 3, rangexyz,
        ops_arg_dat(work_array1, S3D_000, "double", OPS_WRITE),
        ops_arg_dat(work_array2, S3D_000, "double", OPS_WRITE),
        ops_arg_dat(volume, S3D_000, "double", OPS_READ),
        ops_arg_dat(vol_flux_z, S3D_000_00P1, "double", OPS_READ),
        ops_arg_dat(vol_flux_y, S3D_000_0P10, "double", OPS_READ));
    }
    else {
      ops_par_loop(advec_cell_kernel2_ydir, "advec_cell_kernel2_ydir", clover_grid, 3, rangexyz,
        ops_arg_dat(work_array1, S3D_000, "double", OPS_WRITE),
        ops_arg_dat(work_array2, S3D_000, "double", OPS_WRITE),
        ops_arg_dat(volume, S3D_000, "double", OPS_READ),
        ops_arg_dat(vol_flux_y, S3D_000_0P10, "double", OPS_READ),
        ops_arg_dat(vol_flux_x, S3D_000_P100, "double", OPS_READ));
    }
  }

    ops_par_loop(advec_cell_kernel3_ydir, "advec_cell_kernel3_ydir", clover_grid, 3, rangexyz_inner_plus2yz,
      ops_arg_dat(vol_flux_y, S3D_000, "double", OPS_READ),
      ops_arg_dat(work_array1, S3D_000_0M10, "double", OPS_READ),
      ops_arg_dat(yy, S3D_000_0P10_STRID3D_Y, "int", OPS_READ),
      ops_arg_dat(vertexdy, S3D_000_0P10_0M10_STRID3D_Y, "double", OPS_READ),
      ops_arg_dat(density1, S3D_000_0P10_0M10_0M20, "double", OPS_READ),
      ops_arg_dat(energy1, S3D_000_0P10_0M10_0M20, "double", OPS_READ),
      ops_arg_dat(mass_flux_y, S3D_000, "double", OPS_WRITE),
      ops_arg_dat(work_array7, S3D_000, "double", OPS_WRITE));


    ops_par_loop(advec_cell_kernel4_ydir, "advec_cell_kernel4_ydir", clover_grid, 3, rangexyz_inner,
      ops_arg_dat(density1, S3D_000, "double", OPS_RW),
      ops_arg_dat(energy1, S3D_000, "double", OPS_RW),
      ops_arg_dat(mass_flux_y, S3D_000_0P10, "double", OPS_READ),
      ops_arg_dat(vol_flux_y, S3D_000_0P10, "double", OPS_READ),
      ops_arg_dat(work_array1, S3D_000, "double", OPS_READ),
      ops_arg_dat(work_array2, S3D_000, "double", OPS_READ),
      ops_arg_dat(work_array3, S3D_000, "double", OPS_WRITE),
      ops_arg_dat(work_array4, S3D_000, "double", OPS_WRITE),
      ops_arg_dat(work_array5, S3D_000, "double", OPS_WRITE),
      ops_arg_dat(work_array6, S3D_000, "double", OPS_WRITE),
      ops_arg_dat(work_array7, S3D_000_0P10, "double", OPS_READ));

  }
  else if(dir == g_zdir) {

    if(sweep_number == 1) {
      ops_par_loop(advec_cell_kernel1_zdir, "advec_cell_kernel1_zdir", clover_grid, 3, rangexyz,
        ops_arg_dat(work_array1, S3D_000, "double", OPS_WRITE),
        ops_arg_dat(work_array2, S3D_000, "double", OPS_WRITE),
        ops_arg_dat(volume, S3D_000, "double", OPS_READ),
        ops_arg_dat(vol_flux_x, S3D_000_P100, "double", OPS_READ),
        ops_arg_dat(vol_flux_y, S3D_000_0P10, "double", OPS_READ),
        ops_arg_dat(vol_flux_z, S3D_000_00P1, "double", OPS_READ));
    }
    else if (sweep_number == 3) {
      ops_par_loop(advec_cell_kernel2_zdir, "advec_cell_kernel2_zdir", clover_grid, 3, rangexyz,
        ops_arg_dat(work_array1, S3D_000, "double", OPS_WRITE),
        ops_arg_dat(work_array2, S3D_000, "double", OPS_WRITE),
        ops_arg_dat(volume, S3D_000, "double", OPS_READ),
        ops_arg_dat(vol_flux_z, S3D_000_00P1, "double", OPS_READ));
    }

    ops_par_loop(advec_cell_kernel3_zdir, "advec_cell_kernel3_zdir", clover_grid, 3, rangexyz_inner_plus2z,
      ops_arg_dat(vol_flux_z, S3D_000, "double", OPS_READ),
      ops_arg_dat(work_array1, S3D_000_00M1, "double", OPS_READ),
      ops_arg_dat(zz, S3D_000_00P1_STRID3D_Z, "int", OPS_READ),
      ops_arg_dat(vertexdz, S3D_000_00P1_00M1_STRID3D_Z, "double", OPS_READ),
      ops_arg_dat(density1, S3D_000_00P1_00M1_00M2, "double", OPS_READ),
      ops_arg_dat(energy1, S3D_000_00P1_00M1_00M2, "double", OPS_READ),
      ops_arg_dat(mass_flux_z, S3D_000, "double", OPS_WRITE),
      ops_arg_dat(work_array7, S3D_000, "double", OPS_WRITE));

    ops_par_loop(advec_cell_kernel4_zdir, "advec_cell_kernel4_zdir", clover_grid, 3, rangexyz_inner,
      ops_arg_dat(density1, S3D_000, "double", OPS_RW),
      ops_arg_dat(energy1, S3D_000, "double", OPS_RW),
      ops_arg_dat(mass_flux_z, S3D_000_00P1, "double", OPS_READ),
      ops_arg_dat(vol_flux_z, S3D_000_00P1, "double", OPS_READ),
      ops_arg_dat(work_array1, S3D_000, "double", OPS_READ),
      ops_arg_dat(work_array2, S3D_000, "double", OPS_READ),
      ops_arg_dat(work_array3, S3D_000, "double", OPS_WRITE),
      ops_arg_dat(work_array4, S3D_000, "double", OPS_WRITE),
      ops_arg_dat(work_array5, S3D_000, "double", OPS_WRITE),
      ops_arg_dat(work_array6, S3D_000, "double", OPS_WRITE),
      ops_arg_dat(work_array7, S3D_000_00P1, "double", OPS_READ));

  }

}
