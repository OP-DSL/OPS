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

//Cloverleaf kernels
#include "field_summary_kernel.h"
#include "ideal_gas_kernel.h"


void field_summary()
{
  //initialize sizes using global values
  int x_cells = grid->x_cells;
  int y_cells = grid->y_cells;
  int x_min = field->x_min;
  int x_max = field->x_max;
  int y_min = field->y_min;
  int y_max = field->y_max;

  int self2d[]  = {0,0};
  ops_stencil sten2D = ops_decl_stencil( 2, 1, self2d, "self2d");
  int rangexy_inner[] = {x_min,x_max,y_min,y_max}; // inner range without border

  //call ideal_gas again here
  ops_par_loop(ideal_gas_kernel, "ideal_gas_kernel", 2, rangexy_inner,
      ops_arg_dat(density0, sten2D, OPS_READ),
      ops_arg_dat(energy0, sten2D, OPS_READ),
      ops_arg_dat(pressure, sten2D, OPS_RW),
      ops_arg_dat(soundspeed, sten2D, OPS_WRITE));

  double vol= 0.0 , mass = 0.0, ie = 0.0, ke = 0.0, press = 0.0;

  int four_point[]  = {0,0, 1,0, 0,1, 1,1};
  ops_stencil sten2D_4point = ops_decl_stencil( 2, 4, four_point, "sten2D_4point");

  ops_par_loop(field_summary_kernel, "field_summary_kernel", 2, rangexy_inner,
      ops_arg_dat(volume, sten2D, OPS_READ),
      ops_arg_dat(density0, sten2D, OPS_READ),
      ops_arg_dat(energy0, sten2D, OPS_READ),
      ops_arg_dat(pressure, sten2D, OPS_READ),
      ops_arg_dat(xvel0, sten2D_4point, OPS_READ),
      ops_arg_dat(yvel0, sten2D_4point, OPS_READ),
      ops_arg_gbl(&vol, 1, OPS_WRITE),
      ops_arg_gbl(&mass, 1, OPS_WRITE),
      ops_arg_gbl(&ie, 1, OPS_WRITE),
      ops_arg_gbl(&ke, 1, OPS_WRITE),
      ops_arg_gbl(&press, 1, OPS_WRITE));

  ops_fprintf(g_out,"\n");
  ops_fprintf(g_out," Problem initialised and generated\n");
  ops_fprintf(g_out,"\n");

  ops_fprintf(g_out,"              %-10s  %-10s  %-10s  %-10s  %-15s  %-15s  %-15s\n",
  "Volume","Mass","Density","Pressure","Internal Energy","Kinetic Energy","Total Energy");
  ops_fprintf(g_out,"step:   %3d   %-10.3E  %-10.3E  %-10.3E  %-10.3E  %-15.3E  %-15.3E  %-15.3E\n\n",
          step, vol, mass, mass/vol, press/vol, ie, ke, ie+ke);

}
