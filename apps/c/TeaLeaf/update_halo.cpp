/*Crown Copyright 2014 AWE.

 This file is part of TeaLeaf.

 TeaLeaf is free software: you can redistribute it and/or modify it under
 the terms of the GNU General Public License as published by the
 Free Software Foundation, either version 3 of the License, or (at your option)
 any later version.

 TeaLeaf is distributed in the hope that it will be useful, but
 WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
 FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more
 details.

 You should have received a copy of the GNU General Public License along with
 TeaLeaf. if not, see http://www.gnu.org/licenses/. */

// @brief Controls the main diffusion cycle.
// @author Istvan Reguly, David Beckingsale, Wayne Gaudin
// @details Implicitly calculates the change in temperature using a Jacobi iteration


#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

// OPS header file
#define OPS_2D
#include <ops_seq.h>


#include "data.h"
#include "definitions.h"

#include "update_halo_kernels.h"

void update_halo_kernel(
	ops_dat density,
	ops_dat energy0,
	ops_dat energy1,
	ops_dat u,
	ops_dat p,
	ops_dat sd,
	int *fields,
	int depth) {

	//initialize sizes using global values
  int x_min = field.x_min;
  int x_max = field.x_max;
  int y_min = field.y_min;
  int y_max = field.y_max;

  if (fields[FIELD_DENSITY] || fields[FIELD_ENERGY0] || fields[FIELD_ENERGY1] ||
      fields[FIELD_U] || fields[FIELD_P] || fields[FIELD_SD]) {
    int rangexy_b2a[] = {x_min-depth,x_max+depth,y_min-2,y_min-1};
    if(depth ==2)
    ops_par_loop(update_halo_kernel1_b2, "update_halo_kernel1", tea_grid, 2, rangexy_b2a,
              ops_arg_dat_opt(density, 1, S2D_00_0P3, "double", OPS_RW, fields[FIELD_DENSITY]),
              ops_arg_dat_opt(energy0, 1, S2D_00_0P3, "double", OPS_RW, fields[FIELD_ENERGY0]),
              ops_arg_dat_opt(energy1, 1, S2D_00_0P3, "double", OPS_RW, fields[FIELD_ENERGY1]),
              ops_arg_dat_opt(u, 1, S2D_00_0P3, "double", OPS_RW, fields[FIELD_U]),
              ops_arg_dat_opt(p, 1, S2D_00_0P3, "double", OPS_RW, fields[FIELD_P]),
              ops_arg_dat_opt(sd, 1, S2D_00_0P3, "double", OPS_RW, fields[FIELD_SD]),
              ops_arg_gbl(fields, NUM_FIELDS, "int", OPS_READ));

    int rangexy_b1a[] = {x_min-depth,x_max+depth,y_min-1,y_min};
    ops_par_loop(update_halo_kernel1_b1, "update_halo_kernel1", tea_grid, 2, rangexy_b1a,
              ops_arg_dat_opt(density, 1, S2D_00_0P1, "double", OPS_RW, fields[FIELD_DENSITY]),
              ops_arg_dat_opt(energy0, 1, S2D_00_0P1, "double", OPS_RW, fields[FIELD_ENERGY0]),
              ops_arg_dat_opt(energy1, 1, S2D_00_0P1, "double", OPS_RW, fields[FIELD_ENERGY1]),
              ops_arg_dat_opt(u, 1, S2D_00_0P1, "double", OPS_RW, fields[FIELD_U]),
              ops_arg_dat_opt(p, 1, S2D_00_0P1, "double", OPS_RW, fields[FIELD_P]),
              ops_arg_dat_opt(sd, 1, S2D_00_0P1, "double", OPS_RW, fields[FIELD_SD]),
              ops_arg_gbl(fields, NUM_FIELDS, "int", OPS_READ));

    int rangexy_t2a[] = {x_min-depth,x_max+depth,y_max+1,y_max+2};
    if(depth ==2)
    ops_par_loop(update_halo_kernel1_t2, "update_halo_kernel1", tea_grid, 2, rangexy_t2a,
              ops_arg_dat_opt(density, 1, S2D_00_0M3, "double", OPS_RW, fields[FIELD_DENSITY]),
              ops_arg_dat_opt(energy0, 1, S2D_00_0M3, "double", OPS_RW, fields[FIELD_ENERGY0]),
              ops_arg_dat_opt(energy1, 1, S2D_00_0M3, "double", OPS_RW, fields[FIELD_ENERGY1]),
              ops_arg_dat_opt(u, 1, S2D_00_0M3, "double", OPS_RW, fields[FIELD_U]),
              ops_arg_dat_opt(p, 1, S2D_00_0M3, "double", OPS_RW, fields[FIELD_P]),
              ops_arg_dat_opt(sd, 1, S2D_00_0M3, "double", OPS_RW, fields[FIELD_SD]),
              ops_arg_gbl(fields, NUM_FIELDS, "int", OPS_READ));

    int rangexy_t1a[] = {x_min-depth,x_max+depth,y_max,y_max+1};
    ops_par_loop(update_halo_kernel1_t1, "update_halo_kernel1", tea_grid, 2, rangexy_t1a,
               ops_arg_dat_opt(density, 1, S2D_00_0M1, "double", OPS_RW, fields[FIELD_DENSITY]),
               ops_arg_dat_opt(energy0, 1, S2D_00_0M1, "double", OPS_RW, fields[FIELD_ENERGY0]),
               ops_arg_dat_opt(energy1, 1, S2D_00_0M1, "double", OPS_RW, fields[FIELD_ENERGY1]),
               ops_arg_dat_opt(u, 1, S2D_00_0M1, "double", OPS_RW, fields[FIELD_U]),
               ops_arg_dat_opt(p, 1, S2D_00_0M1, "double", OPS_RW, fields[FIELD_P]),
               ops_arg_dat_opt(sd, 1, S2D_00_0M1, "double", OPS_RW, fields[FIELD_SD]),
               ops_arg_gbl(fields, NUM_FIELDS, "int", OPS_READ));

    int rangexy_l2a[] = {x_min-2,x_min-1,y_min-depth,y_max+depth};
    if(depth ==2)
    ops_par_loop(update_halo_kernel1_l2, "update_halo_kernel", tea_grid, 2, rangexy_l2a,
               ops_arg_dat_opt(density, 1, S2D_00_P30, "double", OPS_RW, fields[FIELD_DENSITY]),
               ops_arg_dat_opt(energy0, 1, S2D_00_P30, "double", OPS_RW, fields[FIELD_ENERGY0]),
               ops_arg_dat_opt(energy1, 1, S2D_00_P30, "double", OPS_RW, fields[FIELD_ENERGY1]),
               ops_arg_dat_opt(u, 1, S2D_00_P30, "double", OPS_RW, fields[FIELD_U]),
               ops_arg_dat_opt(p, 1, S2D_00_P30, "double", OPS_RW, fields[FIELD_P]),
               ops_arg_dat_opt(sd, 1, S2D_00_P30, "double", OPS_RW, fields[FIELD_SD]),
               ops_arg_gbl(fields, NUM_FIELDS, "int", OPS_READ));

    int rangexy_l1a[] = {x_min-1,x_min,y_min-depth,y_max+depth};
    ops_par_loop(update_halo_kernel1_l1, "update_halo_kernel", tea_grid, 2, rangexy_l1a,
               ops_arg_dat_opt(density, 1, S2D_00_P10, "double", OPS_RW, fields[FIELD_DENSITY]),
               ops_arg_dat_opt(energy0, 1, S2D_00_P10, "double", OPS_RW, fields[FIELD_ENERGY0]),
               ops_arg_dat_opt(energy1, 1, S2D_00_P10, "double", OPS_RW, fields[FIELD_ENERGY1]),
               ops_arg_dat_opt(u, 1, S2D_00_P10, "double", OPS_RW, fields[FIELD_U]),
               ops_arg_dat_opt(p, 1, S2D_00_P10, "double", OPS_RW, fields[FIELD_P]),
               ops_arg_dat_opt(sd, 1, S2D_00_P10, "double", OPS_RW, fields[FIELD_SD]),
               ops_arg_gbl(fields, NUM_FIELDS, "int", OPS_READ));

    int rangexy_r2a[] = {x_max+1,x_max+2,y_min-depth,y_max+depth};

    if(depth ==2)
    ops_par_loop(update_halo_kernel1_r2, "update_halo_kernel", tea_grid, 2, rangexy_r2a,
               ops_arg_dat_opt(density, 1, S2D_00_M30, "double", OPS_RW, fields[FIELD_DENSITY]),
               ops_arg_dat_opt(energy0, 1, S2D_00_M30, "double", OPS_RW, fields[FIELD_ENERGY0]),
               ops_arg_dat_opt(energy1, 1, S2D_00_M30, "double", OPS_RW, fields[FIELD_ENERGY1]),
               ops_arg_dat_opt(u, 1, S2D_00_M30, "double", OPS_RW, fields[FIELD_U]),
               ops_arg_dat_opt(p, 1, S2D_00_M30, "double", OPS_RW, fields[FIELD_P]),
               ops_arg_dat_opt(sd, 1, S2D_00_M30, "double", OPS_RW, fields[FIELD_SD]),
               ops_arg_gbl(fields, NUM_FIELDS, "int", OPS_READ));

    int rangexy_r1a[] = {x_max,x_max+1,y_min-depth,y_max+depth};
    ops_par_loop(update_halo_kernel1_r1, "update_halo_kernel", tea_grid, 2, rangexy_r1a,
               ops_arg_dat_opt(density, 1, S2D_00_M10, "double", OPS_RW, fields[FIELD_DENSITY]),
               ops_arg_dat_opt(energy0, 1, S2D_00_M10, "double", OPS_RW, fields[FIELD_ENERGY0]),
               ops_arg_dat_opt(energy1, 1, S2D_00_M10, "double", OPS_RW, fields[FIELD_ENERGY1]),
               ops_arg_dat_opt(u, 1, S2D_00_M10, "double", OPS_RW, fields[FIELD_U]),
               ops_arg_dat_opt(p, 1, S2D_00_M10, "double", OPS_RW, fields[FIELD_P]),
               ops_arg_dat_opt(sd, 1, S2D_00_M10, "double", OPS_RW, fields[FIELD_SD]),
               ops_arg_gbl(fields, NUM_FIELDS, "int", OPS_READ));


  }

}
void update_halo(int *fields, int depth) {
	if (reflective_boundary == 1) {
		update_halo_kernel(density,energy0,energy1,u, vector_p, vector_sd, fields, depth);
	}
}
