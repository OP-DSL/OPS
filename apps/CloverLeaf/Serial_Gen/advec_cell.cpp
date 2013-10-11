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
#include "ops_seq_opt.h"

#include "data.h"
#include "definitions.h"
#include "advec_cell_kernel.h"

void ops_par_loop_advec_cell_xdir_kernel1(char const * name, int dim, int *range,
                  ops_arg arg0, ops_arg arg1, ops_arg arg2, ops_arg arg3, ops_arg arg4) {

  char  **p_a[5];
  int   offs[5][2];
  int   count[dim];

  ops_arg args[5] = {arg0, arg1, arg2, arg3, arg4};

  for(int i=0; i<5; i++) {
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

  for (int i = 0; i < 5; i++) {
    if (args[i].argtype == OPS_ARG_DAT) {
      p_a[i] = (char **)malloc(args[i].stencil->points * sizeof(char *));
    }
    else if (args[i].argtype == OPS_ARG_GBL)
      p_a[i] = (char **)malloc(args[i].dim * sizeof(char *));
  }

  //set up initial pointers
  ops_args_set(range[0], range[2],5,args,p_a);

  for (int n_y = range[2]; n_y < range[3]; n_y++) {
    for (int n_x = range[0]; n_x < range[1]; n_x++) {
      // call kernel function, passing in pointers to data
      advec_cell_xdir_kernel1( (double **)p_a[0], (double ** )p_a[1], (double ** )p_a[2],
                                     (double ** )p_a[3], (double ** )p_a[4]);

      for (int np=0; np<args[0].stencil->points; np++) p_a[0][np] += (args[0].dat->size * offs[0][0]);
      for (int np=0; np<args[1].stencil->points; np++) p_a[1][np] += (args[1].dat->size * offs[1][0]);
      for (int np=0; np<args[2].stencil->points; np++) p_a[2][np] += (args[2].dat->size * offs[2][0]);
      for (int np=0; np<args[3].stencil->points; np++) p_a[3][np] += (args[3].dat->size * offs[3][0]);
      for (int np=0; np<args[4].stencil->points; np++) p_a[4][np] += (args[4].dat->size * offs[4][0]);
    }

    for (int np=0; np<args[0].stencil->points; np++) p_a[0][np] += (args[0].dat->size * offs[0][1]);
    for (int np=0; np<args[1].stencil->points; np++) p_a[1][np] += (args[1].dat->size * offs[1][1]);
    for (int np=0; np<args[2].stencil->points; np++) p_a[2][np] += (args[2].dat->size * offs[2][1]);
    for (int np=0; np<args[3].stencil->points; np++) p_a[3][np] += (args[3].dat->size * offs[3][1]);
    for (int np=0; np<args[4].stencil->points; np++) p_a[4][np] += (args[4].dat->size * offs[4][1]);
  }

  for (int i = 0; i < 5; i++)
    free(p_a[i]);
}

void ops_par_loop_advec_cell_xdir_kernel2(char const * name, int dim, int *range,
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
      advec_cell_xdir_kernel2( (double **)p_a[0], (double ** )p_a[1], (double ** )p_a[2], ( double **)p_a[3] );

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



void ops_par_loop_advec_cell_xdir_kernel3(char const * name, int dim, int *range,
                  ops_arg arg0, ops_arg arg1, ops_arg arg2, ops_arg arg3,
                  ops_arg arg4, ops_arg arg5, ops_arg arg6, ops_arg arg7) {

  char  **p_a[8];
  int   offs[8][2];
  int   count[dim];

  ops_arg args[8] = {arg0, arg1, arg2, arg3, arg4, arg5, arg6, arg7};

  for(int i=0; i<8; i++) {
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

  for (int i = 0; i < 8; i++) {
    if (args[i].argtype == OPS_ARG_DAT) {
      p_a[i] = (char **)malloc(args[i].stencil->points * sizeof(char *));
    }
    else if (args[i].argtype == OPS_ARG_GBL)
      p_a[i] = (char **)malloc(args[i].dim * sizeof(char *));
  }

  //set up initial pointers
  ops_args_set(range[0], range[2],8,args,p_a);

  for (int n_y = range[2]; n_y < range[3]; n_y++) {
    for (int n_x = range[0]; n_x < range[1]; n_x++) {
      // call kernel function, passing in pointers to data
      advec_cell_xdir_kernel3( (double **)p_a[0], (double ** )p_a[1], (int ** )p_a[2],
                           (double ** )p_a[3], (double ** )p_a[4], (double ** )p_a[5],
                           (double ** )p_a[6], (double ** )p_a[7]);

      for (int np=0; np<args[0].stencil->points; np++) p_a[0][np] += (args[0].dat->size * offs[0][0]);
      for (int np=0; np<args[1].stencil->points; np++) p_a[1][np] += (args[1].dat->size * offs[1][0]);
      for (int np=0; np<args[2].stencil->points; np++) p_a[2][np] += (args[2].dat->size * offs[2][0]);
      for (int np=0; np<args[3].stencil->points; np++) p_a[3][np] += (args[3].dat->size * offs[3][0]);
      for (int np=0; np<args[4].stencil->points; np++) p_a[4][np] += (args[4].dat->size * offs[4][0]);
      for (int np=0; np<args[5].stencil->points; np++) p_a[5][np] += (args[5].dat->size * offs[5][0]);
      for (int np=0; np<args[6].stencil->points; np++) p_a[6][np] += (args[6].dat->size * offs[6][0]);
      for (int np=0; np<args[7].stencil->points; np++) p_a[7][np] += (args[7].dat->size * offs[7][0]);
    }

    for (int np=0; np<args[0].stencil->points; np++) p_a[0][np] += (args[0].dat->size * offs[0][1]);
    for (int np=0; np<args[1].stencil->points; np++) p_a[1][np] += (args[1].dat->size * offs[1][1]);
    for (int np=0; np<args[2].stencil->points; np++) p_a[2][np] += (args[2].dat->size * offs[2][1]);
    for (int np=0; np<args[3].stencil->points; np++) p_a[3][np] += (args[3].dat->size * offs[3][1]);
    for (int np=0; np<args[4].stencil->points; np++) p_a[4][np] += (args[4].dat->size * offs[4][1]);
    for (int np=0; np<args[5].stencil->points; np++) p_a[5][np] += (args[5].dat->size * offs[5][1]);
    for (int np=0; np<args[6].stencil->points; np++) p_a[6][np] += (args[6].dat->size * offs[6][1]);
    for (int np=0; np<args[7].stencil->points; np++) p_a[7][np] += (args[7].dat->size * offs[7][1]);
  }

  for (int i = 0; i < 8; i++)
    free(p_a[i]);
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

  if(dir == g_xdir) {

    if(sweep_number == 1) {
      ops_par_loop_advec_cell_xdir_kernel1("advec_cell_xdir_kernel1", 2, rangexy,
        ops_arg_dat(work_array1, S2D_00, OPS_RW),
        ops_arg_dat(work_array2, S2D_00, OPS_WRITE),
        ops_arg_dat(volume, S2D_00, OPS_READ),
        ops_arg_dat(vol_flux_x, S2D_00_P10, OPS_READ),
        ops_arg_dat(vol_flux_y, S2D_00_0P1, OPS_READ)
        );
    }
    else {
      ops_par_loop_advec_cell_xdir_kernel2("advec_cell_xdir_kernel2", 2, rangexy,
        ops_arg_dat(work_array1, S2D_00, OPS_WRITE),
        ops_arg_dat(work_array2, S2D_00, OPS_WRITE),
        ops_arg_dat(volume, S2D_00, OPS_READ),
        ops_arg_dat(vol_flux_x, S2D_00_P10, OPS_READ)
        );
    }

    ops_par_loop_advec_cell_xdir_kernel3("advec_cell_xdir_kernel3", 2, rangexy_inner_plus2x,
      ops_arg_dat(vol_flux_x, S2D_00, OPS_READ),
      ops_arg_dat(work_array1, sten_self2D_plus_1_minus1_2_x, OPS_READ),
      ops_arg_dat(xx, sten_self_plus1_stride2D_x, OPS_READ),
      ops_arg_dat(vertexdx, sten_self_plus_1_minus1_2_x_stride2D_x, OPS_READ),
      ops_arg_dat(density1, sten_self2D_plus_1_minus1_2_x, OPS_READ),
      ops_arg_dat(energy1, sten_self2D_plus_1_minus1_2_x, OPS_READ),
      ops_arg_dat(mass_flux_x, S2D_00, OPS_WRITE),
      ops_arg_dat(work_array7, S2D_00, OPS_WRITE)
      );

    ops_par_loop(advec_cell_xdir_kernel3a, "advec_cell_xdir_kernel3a", 2, rangexy_inner_plus2x,
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
      );

    ops_par_loop_opt(advec_cell_xdir_kernel4, "advec_cell_xdir_kernel4", 2, rangexy_inner,
      ops_arg_dat(density1, S2D_00, OPS_RW),
      ops_arg_dat(energy1, S2D_00, OPS_RW),
      ops_arg_dat(mass_flux_x, S2D_00_P10, OPS_READ),
      ops_arg_dat(vol_flux_x, S2D_00_P10, OPS_READ),
      ops_arg_dat(work_array1, S2D_00, OPS_READ),
      ops_arg_dat(work_array2, S2D_00, OPS_READ),
      ops_arg_dat(work_array3, S2D_00, OPS_RW),
      ops_arg_dat(work_array4, S2D_00, OPS_RW),
      ops_arg_dat(work_array5, S2D_00, OPS_RW),
      ops_arg_dat(work_array6, S2D_00, OPS_RW),
      ops_arg_dat(work_array7, S2D_00_P10, OPS_READ)
      );

  }
  else {

    if(sweep_number == 1) {
      ops_par_loop_opt(advec_cell_ydir_kernel1, "advec_cell_ydir_kernel1", 2, rangexy,
        ops_arg_dat(work_array1, S2D_00, OPS_RW),
        ops_arg_dat(work_array2, S2D_00, OPS_WRITE),
        ops_arg_dat(volume, S2D_00, OPS_READ),
        ops_arg_dat(vol_flux_x, S2D_00_P10, OPS_READ),
        ops_arg_dat(vol_flux_y, S2D_00_0P1, OPS_READ)
        );
    }
    else {
      ops_par_loop_opt(advec_cell_ydir_kernel2, "advec_cell_ydir_kernel2", 2, rangexy,
        ops_arg_dat(work_array1, S2D_00, OPS_WRITE),
        ops_arg_dat(work_array2, S2D_00, OPS_WRITE),
        ops_arg_dat(volume, S2D_00, OPS_READ),
        ops_arg_dat(vol_flux_y, S2D_00_P10, OPS_READ)
        );
    }

    ops_par_loop_opt(advec_cell_ydir_kernel3, "advec_cell_ydir_kernel3", 2, rangexy_inner_plus2y,
      ops_arg_dat(vol_flux_y, S2D_00, OPS_READ),
      ops_arg_dat(work_array1, sten_self2D_plus_1_minus1_2_y, OPS_READ),
      ops_arg_dat(yy, sten_self_plus1_stride2D_y, OPS_READ),
      ops_arg_dat(vertexdy, sten_self_plus_1_minus1_2_y_stride2D_y, OPS_READ),
      ops_arg_dat(density1, sten_self2D_plus_1_minus1_2_y, OPS_READ),
      ops_arg_dat(energy1, sten_self2D_plus_1_minus1_2_y, OPS_READ),
      ops_arg_dat(mass_flux_y, S2D_00, OPS_WRITE),
      ops_arg_dat(work_array7, S2D_00, OPS_WRITE)
      );

    ops_par_loop(advec_cell_ydir_kernel3a, "advec_cell_ydir_kernel3a", 2, rangexy_inner_plus2y,
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
      );


    ops_par_loop_opt(advec_cell_ydir_kernel4, "advec_cell_ydir_kernel4", 2, rangexy_inner,
      ops_arg_dat(density1, S2D_00, OPS_RW),
      ops_arg_dat(energy1, S2D_00, OPS_RW),
      ops_arg_dat(mass_flux_y, S2D_00_0P1, OPS_READ),
      ops_arg_dat(vol_flux_y, S2D_00_0P1, OPS_READ),
      ops_arg_dat(work_array1, S2D_00, OPS_READ),
      ops_arg_dat(work_array2, S2D_00, OPS_READ),
      ops_arg_dat(work_array3, S2D_00, OPS_RW),
      ops_arg_dat(work_array4, S2D_00, OPS_RW),
      ops_arg_dat(work_array5, S2D_00, OPS_RW),
      ops_arg_dat(work_array6, S2D_00, OPS_RW),
      ops_arg_dat(work_array7, S2D_00_0P1, OPS_READ)
      );

  }

}
