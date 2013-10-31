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

/** @brief acceleration kernels
 *  @author Wayne Gaudin, converted to OPS by Gihan Mudalige
 *  @details Calls user requested kernel
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

#include "accelerate_kernel.h"

void accelerate_kernel_stepbymass_macro( double *density0, double *volume,
                double *stepbymass) {

  double nodal_mass;

  //{0,0, -1,0, 0,-1, -1,-1};
  nodal_mass = ( density0[OPS_ACC0(-1,-1)] * volume[OPS_ACC1(-1,-1)]
    + density0[OPS_ACC0(0,-1)] * volume[OPS_ACC1(0,-1)]
    + density0[OPS_ACC0(0,0)] * volume[OPS_ACC1(0,0)]
    + density0[OPS_ACC0(-1,0)] * volume[OPS_ACC1(-1,0)] ) * 0.25;

  stepbymass[OPS_ACC2(0,0)] = 0.5*dt / nodal_mass;

}

void accelerate_kernelx1_macro( double *xvel0, double *xvel1,
                        double *stepbymass,
                        double *xarea, double *pressure) {
  //{0,0, -1,0, 0,-1, -1,-1};
  //{0,0, 0,-1};

  xvel1[OPS_ACC1(0,0)] = xvel0[OPS_ACC0(0,0)] - stepbymass[OPS_ACC2(0,0)] *
            ( xarea[OPS_ACC3(0,0)]  * ( pressure[OPS_ACC4(0,0)] - pressure[OPS_ACC4(-1,0)] ) +
              xarea[OPS_ACC3(0,-1)] * ( pressure[OPS_ACC4(0,-1)] - pressure[OPS_ACC4(-1,-1)] ) );
}

void accelerate_kernely1_macro( double *yvel0, double *yvel1,
                        double *stepbymass,
                        double *yarea, double *pressure) {

  //{0,0, -1,0, 0,-1, -1,-1};
  //{0,0, -1,0};

  yvel1[OPS_ACC1(0,0)] = yvel0[OPS_ACC0(0,0)] - stepbymass[OPS_ACC2(0,0)] *
            ( yarea[OPS_ACC3(0,0)]  * ( pressure[OPS_ACC4(0,0)] - pressure[OPS_ACC4(0,-1)] ) +
              yarea[OPS_ACC3(-1,0)] * ( pressure[OPS_ACC4(-1,0)] - pressure[OPS_ACC4(-1,-1)] ) );

}


void accelerate_kernelx2_macro( double *xvel1, double *stepbymass,
                        double *xarea, double *viscosity) {

  //{0,0, -1,0, 0,-1, -1,-1};
  //{0,0, 0,-1};

  xvel1[OPS_ACC0(0,0)] = xvel1[OPS_ACC0(0,0)] - stepbymass[OPS_ACC1(0,0)] *
            ( xarea[OPS_ACC2(0,0)] * ( viscosity[OPS_ACC3(0,0)] - viscosity[OPS_ACC3(-1,0)] ) +
              xarea[OPS_ACC2(0,-1)] * ( viscosity[OPS_ACC3(0,-1)] - viscosity[OPS_ACC3(-1,-1)] ) );
}

void accelerate_kernely2_macro( double *yvel1, double *stepbymass,
                        double *yarea, double *viscosity) {

  //{0,0, -1,0, 0,-1, -1,-1};
  //{0,0, -1,0};

  yvel1[OPS_ACC0(0,0)] = yvel1[OPS_ACC0(0,0)] - stepbymass[OPS_ACC1(0,0)] *
            ( yarea[OPS_ACC2(0,0)] * ( viscosity[OPS_ACC3(0,0)] - viscosity[OPS_ACC3(0,-1)] ) +
              yarea[OPS_ACC2(0,-1)] * ( viscosity[OPS_ACC3(-1,0)] - viscosity[OPS_ACC3(-1,-1)] ) );

}

void accelerate()
{
  error_condition = 0; // Not used yet due to issue with OpenA reduction

  //initialize sizes using global values
  int x_cells = grid->x_cells;
  int y_cells = grid->y_cells;
  int x_min = field->x_min;
  int x_max = field->x_max;
  int y_min = field->y_min;
  int y_max = field->y_max;


  int rangexy_inner_plus1[] = {x_min,x_max+1,y_min,y_max+1}; // inner range plus 1

  ops_par_loop_macro(accelerate_kernel_stepbymass_macro, "accelerate_kernel_stepbymass_macro", 2, rangexy_inner_plus1,
    ops_arg_dat(density0, S2D_00_M10_0M1_M1M1, "double", OPS_READ),
    ops_arg_dat(volume, S2D_00_M10_0M1_M1M1, "double", OPS_READ),
    ops_arg_dat(work_array1, S2D_00, "double", OPS_WRITE));

  ops_par_loop_macro(accelerate_kernelx1_macro, "accelerate_kernelx1_macro", 2, rangexy_inner_plus1,
    ops_arg_dat(xvel0, S2D_00, "double", OPS_READ),
    ops_arg_dat(xvel1, S2D_00, "double", OPS_WRITE),
    ops_arg_dat(work_array1, S2D_00, "double", OPS_READ),
    ops_arg_dat(xarea, S2D_00_0M1, "double", OPS_READ),
    ops_arg_dat(pressure, S2D_00_M10_0M1_M1M1, "double", OPS_READ));

  ops_par_loop_macro(accelerate_kernely1_macro, "accelerate_kernely1_macro", 2, rangexy_inner_plus1,
    ops_arg_dat(yvel0, S2D_00, "double", OPS_READ),
    ops_arg_dat(yvel1, S2D_00, "double", OPS_WRITE),
    ops_arg_dat(work_array1, S2D_00, "double", OPS_READ),
    ops_arg_dat(yarea, S2D_00_M10, "double", OPS_READ),
    ops_arg_dat(pressure, S2D_00_M10_0M1_M1M1, "double", OPS_READ));

  ops_par_loop_macro(accelerate_kernelx2_macro, "accelerate_kernelx2", 2, rangexy_inner_plus1,
    ops_arg_dat(xvel1, S2D_00, "double", OPS_INC),
    ops_arg_dat(work_array1, S2D_00, "double", OPS_READ),
    ops_arg_dat(xarea, S2D_00_0M1, "double", OPS_READ),
    ops_arg_dat(viscosity, S2D_00_M10_0M1_M1M1, "double", OPS_READ));

  ops_par_loop_macro(accelerate_kernely2_macro, "accelerate_kernely2", 2, rangexy_inner_plus1,
    ops_arg_dat(yvel1, S2D_00, "double", OPS_INC),
    ops_arg_dat(work_array1, S2D_00, "double", OPS_READ),
    ops_arg_dat(yarea, S2D_00_M10, "double", OPS_READ),
    ops_arg_dat(viscosity, S2D_00_M10_0M1_M1M1, "double", OPS_READ));

}
