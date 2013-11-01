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
#include "ops_seq_opt.h"
#include "ops_seq_macro.h"

#include "data.h"
#include "definitions.h"
#include "advec_cell_kernel.h"

void advec_cell_kernel1_xdir_macro( double *pre_vol, double *post_vol, double *volume,
                        double *vol_flux_x, double *vol_flux_y) {

  pre_vol[OPS_ACC0(0,0)] = volume[OPS_ACC2(0,0)] + ( vol_flux_x[OPS_ACC3(1,0)] - vol_flux_x[OPS_ACC3(0,0)] +
                           vol_flux_y[OPS_ACC4(0,1)] - vol_flux_y[OPS_ACC4(0,1)]);
  post_vol[OPS_ACC1(0,0)] = pre_vol[OPS_ACC0(0,0)] - ( vol_flux_x[OPS_ACC3(1,0)] - vol_flux_x[OPS_ACC3(0,0)]);

}

void advec_cell_kernel2_xdir_macro( double *pre_vol, double *post_vol, double *volume,
                        double *vol_flux_x) {

  pre_vol[OPS_ACC0(0,0)] = volume[OPS_ACC2(0,0)] + vol_flux_x[OPS_ACC3(1,0)] - vol_flux_x[OPS_ACC3(0,0)];
  post_vol[OPS_ACC1(0,0)] = volume[OPS_ACC2(0,0)];

}


void advec_cell_kernel4_xdir_macro( double *density1, double *energy1,
                         double *mass_flux_x, double *vol_flux_x,
                         double *pre_vol, double *post_vol,
                         double *pre_mass, double *post_mass,
                         double *advec_vol, double *post_ener,
                         double *ener_flux) {

  pre_mass[OPS_ACC6(0,0)] = density1[OPS_ACC0(0,0)] * pre_vol[OPS_ACC4(0,0)];
  post_mass[OPS_ACC7(0,0)] = pre_mass[OPS_ACC6(0,0)] + mass_flux_x[OPS_ACC2(0,0)] - mass_flux_x[OPS_ACC2(1,0)];
  post_ener[OPS_ACC9(0,0)] = ( energy1[OPS_ACC1(0,0)] * pre_mass[OPS_ACC6(0,0)] + ener_flux[OPS_ACC10(0,0)] - ener_flux[OPS_ACC10(1,0)])/post_mass[OPS_ACC7(0,0)];
  advec_vol[OPS_ACC8(0,0)] = pre_vol[OPS_ACC4(0,0)] + vol_flux_x[OPS_ACC3(0,0)] - vol_flux_x[OPS_ACC3(1,0)];
  density1[OPS_ACC0(0,0)] = post_mass[OPS_ACC7(0,0)]/advec_vol[OPS_ACC8(0,0)];
  energy1[OPS_ACC1(0,0)] = post_ener[OPS_ACC9(0,0)];

}


void advec_cell_kernel1_ydir_macro( double *pre_vol, double *post_vol, double *volume,
                        double *vol_flux_x, double *vol_flux_y) {

  pre_vol[OPS_ACC0(0,0)] = volume[OPS_ACC2(0,0)] + ( vol_flux_y[OPS_ACC3(0,1)] - vol_flux_y[OPS_ACC3(0,0)] +
                           vol_flux_x[OPS_ACC4(1,0)] - vol_flux_x[OPS_ACC4(1,0)]);
  post_vol[OPS_ACC1(0,0)] = pre_vol[OPS_ACC0(0,0)] - ( vol_flux_y[OPS_ACC3(0,1)] - vol_flux_y[OPS_ACC3(0,0)]);

}

void advec_cell_kernel2_ydir_macro( double *pre_vol, double *post_vol, double *volume,
                        double *vol_flux_y) {

  pre_vol[OPS_ACC0(0,0)] = volume[OPS_ACC2(0,0)] + vol_flux_y[OPS_ACC3(0,1)] - vol_flux_y[OPS_ACC3(0,0)];
  post_vol[OPS_ACC1(0,0)] = volume[OPS_ACC2(0,0)];

}

void advec_cell_kernel4_ydir_macro( double *density1, double *energy1,
                         double *mass_flux_y, double *vol_flux_y,
                         double *pre_vol, double *post_vol,
                         double *pre_mass, double *post_mass,
                         double *advec_vol, double *post_ener,
                         double *ener_flux) {

  pre_mass[OPS_ACC6(0,0)] = density1[OPS_ACC0(0,0)] * pre_vol[OPS_ACC4(0,0)];
  post_mass[OPS_ACC7(0,0)] = pre_mass[OPS_ACC6(0,0)] + mass_flux_y[OPS_ACC2(0,0)] - mass_flux_y[OPS_ACC2(0,1)];
  post_ener[OPS_ACC9(0,0)] = ( energy1[OPS_ACC1(0,0)] * pre_mass[OPS_ACC6(0,0)] + ener_flux[OPS_ACC10(0,0)] - ener_flux[OPS_ACC10(0,1)])/post_mass[OPS_ACC7(0,0)];
  advec_vol[OPS_ACC8(0,0)] = pre_vol[OPS_ACC4(0,0)] + vol_flux_y[OPS_ACC3(0,0)] - vol_flux_y[OPS_ACC3(0,1)];
  density1[OPS_ACC0(0,0)] = post_mass[OPS_ACC7(0,0)]/advec_vol[OPS_ACC8(0,0)];
  energy1[OPS_ACC1(0,0)] = post_ener[OPS_ACC9(0,0)];

}

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

  //printf("direction: %d sweep_number %d \n", dir, sweep_number);

  if(dir == g_xdir) {

    if(sweep_number == 1) {
      ops_par_loop_macro(advec_cell_kernel1_xdir_macro, "advec_cell_kernel1_xdir_macro", 2, rangexy,
        ops_arg_dat(work_array1, S2D_00, "double", OPS_READ),
        ops_arg_dat(work_array2, S2D_00, "double", OPS_READ),
        ops_arg_dat(volume, S2D_00, "double", OPS_READ),
        ops_arg_dat(vol_flux_x, S2D_00_P10, "double", OPS_READ),
        ops_arg_dat(vol_flux_y, S2D_00_0P1, "double", OPS_READ)
        );
    }
    else {
      ops_par_loop_macro(advec_cell_kernel2_xdir_macro, "advec_cell_kernel2_xdir_macro", 2, rangexy,
        ops_arg_dat(work_array1, S2D_00, "double", OPS_READ),
        ops_arg_dat(work_array2, S2D_00, "double", OPS_READ),
        ops_arg_dat(volume, S2D_00, "double", OPS_READ),
        ops_arg_dat(vol_flux_x, S2D_00_P10, "double", OPS_READ)
        );
    }


    ops_par_loop_opt(advec_cell_kernel3_xdir, "advec_cell_kernel3_xdir", 2, rangexy_inner_plus2x,
      ops_arg_dat(vol_flux_x, S2D_00, "double", OPS_READ),
      ops_arg_dat(work_array1, S2D_00_P10_M10_M20, "double", OPS_READ),
      ops_arg_dat(xx, s2D_00_P10_STRID2D_X, "int", OPS_READ),
      ops_arg_dat(vertexdx, S2D_00_P10_M10_M20_STRID2D_X, "double", OPS_READ),
      ops_arg_dat(density1, S2D_00_P10_M10_M20, "double", OPS_READ),
      ops_arg_dat(energy1, S2D_00_P10_M10_M20, "double", OPS_READ),
      ops_arg_dat(mass_flux_x, S2D_00, "double", OPS_READ),
      ops_arg_dat(work_array7, S2D_00, "double", OPS_READ)
      );

    ops_par_loop_macro(advec_cell_kernel4_xdir_macro, "advec_cell_kernel4_xdir_macro", 2, rangexy_inner,
      ops_arg_dat(density1, S2D_00, "double", OPS_READ),
      ops_arg_dat(energy1, S2D_00, "double", OPS_READ),
      ops_arg_dat(mass_flux_x, S2D_00_P10, "double", OPS_READ),
      ops_arg_dat(vol_flux_x, S2D_00_P10, "double", OPS_READ),
      ops_arg_dat(work_array1, S2D_00, "double", OPS_READ),
      ops_arg_dat(work_array2, S2D_00, "double", OPS_READ),
      ops_arg_dat(work_array3, S2D_00, "double", OPS_READ),
      ops_arg_dat(work_array4, S2D_00, "double", OPS_READ),
      ops_arg_dat(work_array5, S2D_00, "double", OPS_READ),
      ops_arg_dat(work_array6, S2D_00, "double", OPS_READ),
      ops_arg_dat(work_array7, S2D_00_P10, "double", OPS_READ)
      );

  }
  else {



    if(sweep_number == 1) {
      ops_par_loop_macro(advec_cell_kernel1_ydir_macro, "advec_cell_kernel1_ydir_macro", 2, rangexy,
        ops_arg_dat(work_array1, S2D_00, "double", OPS_READ),
        ops_arg_dat(work_array2, S2D_00, "double", OPS_READ),
        ops_arg_dat(volume, S2D_00, "double", OPS_READ),
        ops_arg_dat(vol_flux_x, S2D_00_P10, "double", OPS_READ),
        ops_arg_dat(vol_flux_y, S2D_00_0P1, "double", OPS_READ)
        );
    }
    else {


      ops_par_loop_macro(advec_cell_kernel2_ydir_macro, "advec_cell_kernel2_ydir_macro", 2, rangexy,
        ops_arg_dat(work_array1, S2D_00, "double", OPS_READ),
        ops_arg_dat(work_array2, S2D_00, "double", OPS_READ),
        ops_arg_dat(volume, S2D_00, "double", OPS_READ),
        ops_arg_dat(vol_flux_y, S2D_00_0P1, "double", OPS_READ)
        );

  }

    ops_par_loop_opt(advec_cell_kernel3_ydir, "advec_cell_kernel3_ydir", 2, rangexy_inner_plus2y,
      ops_arg_dat(vol_flux_y, S2D_00, "double", OPS_READ),
      ops_arg_dat(work_array1, S2D_00_0P1_0M1_0M2, "double", OPS_READ),
      ops_arg_dat(yy, S2D_00_0P1_STRID2D_Y, "int", OPS_READ),
      ops_arg_dat(vertexdy, S2D_00_0P1_0M1_0M2_STRID2D_Y, "double", OPS_READ),
      ops_arg_dat(density1, S2D_00_0P1_0M1_0M2, "double", OPS_READ),
      ops_arg_dat(energy1, S2D_00_0P1_0M1_0M2, "double", OPS_READ),
      ops_arg_dat(mass_flux_y, S2D_00, "double", OPS_READ),
      ops_arg_dat(work_array7, S2D_00, "double", OPS_READ)
      );


    ops_par_loop_macro(advec_cell_kernel4_ydir_macro, "advec_cell_kernel4_ydir_macro", 2, rangexy_inner,
      ops_arg_dat(density1, S2D_00, "double", OPS_READ),
      ops_arg_dat(energy1, S2D_00, "double", OPS_READ),
      ops_arg_dat(mass_flux_y, S2D_00_0P1, "double", OPS_READ),
      ops_arg_dat(vol_flux_y, S2D_00_0P1, "double", OPS_READ),
      ops_arg_dat(work_array1, S2D_00, "double", OPS_READ),
      ops_arg_dat(work_array2, S2D_00, "double", OPS_READ),
      ops_arg_dat(work_array3, S2D_00, "double", OPS_READ),
      ops_arg_dat(work_array4, S2D_00, "double", OPS_READ),
      ops_arg_dat(work_array5, S2D_00, "double", OPS_READ),
      ops_arg_dat(work_array6, S2D_00, "double", OPS_READ),
      ops_arg_dat(work_array7, S2D_00_0P1, "double", OPS_READ)
      );

  }

}
