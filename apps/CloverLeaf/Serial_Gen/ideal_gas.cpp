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

/** @brief Ideal gas kernel driver
 *  @author Wayne Gaudin, converted to OPS by Gihan Mudalige
 *  @details Invokes the user specified kernel for the ideal gas equation of
 *  state using the specified time level data.
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

//Cloverleaf kernels
#include "ideal_gas_kernel.h"


void ops_par_loop_idealgas (char const * name, int dim, int *range,
                  ops_arg arg0, ops_arg arg1, ops_arg arg2, ops_arg arg3  ) {

  char  **p_a[4];
  int   offs[4][2];
  int   count[dim];

  ops_arg args[4] = {arg0, arg1, arg2, arg3};

  for(int i=0; i<4; i++) {
    if (args[i].stencil!=NULL) {
      offs[i][0] = 1;  //unit step in x dimension
      offs[i][1] = ops_offs_set(range[0],range[2]+1, args[i]) - ops_offs_set(range[1],range[2], args[i]);// +1;

      if (args[i].stencil->stride[0] == 0) { //stride in y as x stride is 0
        offs[i][0] = 0;
        offs[i][1] = args[i].dat->block_size[0];
      }
      else if (args[i].stencil->stride[1] == 0) {//stride in x as y stride is 0
        offs[i][0] = 1;
        offs[i][1] = -( range[1] - range[0] );// +1;
      }
    }
  }

  int non_gbl[] = {0,0,0,0}; //store index of non_gbl args
  int g = 0; int a = 0;
  for (int i = 0; i < 4; i++) {
    if (args[i].argtype == OPS_ARG_DAT) {
      p_a[i] = (char **)malloc(args[i].stencil->points * sizeof(char *));
      non_gbl[g++] = i;
    }
    else if (args[i].argtype == OPS_ARG_GBL)
      p_a[i] = (char **)malloc(args[i].dim * sizeof(char *));
  }

  //set up initial pointers
  ops_args_set(range[0], range[2],4,args,p_a);

  for (int n_y = range[2]; n_y < range[3]; n_y++) {
    for (int n_x = range[0]; n_x < range[1]; n_x++) {
      // call kernel function, passing in pointers to data
      ideal_gas_kernel( (double **)p_a[0], (double ** )p_a[1], (double ** )p_a[2], ( double **)p_a[3] );

      for (int np=0; np<args[0].stencil->points; np++) p_a[0][np] += (args[0].dat->size * offs[0][0]);
      for (int np=0; np<args[1].stencil->points; np++) p_a[1][np] += (args[1].dat->size * offs[1][0]);
      for (int np=0; np<args[2].stencil->points; np++) p_a[2][np] += (args[2].dat->size * offs[2][0]);
      for (int np=0; np<args[3].stencil->points; np++) p_a[3][np] += (args[3].dat->size * offs[3][0]);
    }

    for (int np=0; np<args[0].stencil->points; np++) p_a[0][np] += (args[0].dat->size * offs[0][1]);
    for (int np=0; np<args[1].stencil->points; np++) p_a[1][np] += (args[1].dat->size * offs[1][1]);
    for (int np=0; np<args[2].stencil->points; np++) p_a[2][np] += (args[2].dat->size * offs[2][1]);
    for (int np=0; np<args[3].stencil->points; np++) p_a[3][np] += (args[3].dat->size * offs[3][1]);
  }

  for (int i = 0; i < 4; i++)
    free(p_a[i]);
}

void ideal_gas(int predict)
{
  //initialize sizes using global values
  int x_cells = grid->x_cells;
  int y_cells = grid->y_cells;
  int x_min = field->x_min;
  int x_max = field->x_max;
  int y_min = field->y_min;
  int y_max = field->y_max;

  int rangexy_inner[] = {x_min,x_max,y_min,y_max}; // inner range without border

  if(!predict) {
     //ops_par_loop_opt2(ideal_gas_kernel, "ideal_gas_kernel", 2, rangexy_inner,
    ops_par_loop_idealgas("ideal_gas_kernel", 2, rangexy_inner,
      ops_arg_dat(density0, S2D_00, OPS_READ),
      ops_arg_dat(energy0, S2D_00, OPS_READ),
      ops_arg_dat(pressure, S2D_00, OPS_RW),
      ops_arg_dat(soundspeed, S2D_00, OPS_WRITE));
  }
  else {
    //ops_par_loop_opt2(ideal_gas_kernel, "ideal_gas_kernel", 2, rangexy_inner,
    ops_par_loop_idealgas("ideal_gas_kernel", 2, rangexy_inner,
      ops_arg_dat(density1, S2D_00, OPS_READ),
      ops_arg_dat(energy1, S2D_00, OPS_READ),
      ops_arg_dat(pressure, S2D_00, OPS_RW),
      ops_arg_dat(soundspeed, S2D_00, OPS_WRITE));
  }

}
