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

/** @brief Driver for chunk initialisation.
 *  @author Wayne Gaudin, converted to OPS by Gihan Mudalige
 *  @details Invokes the user specified chunk initialisation kernel.
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
#include "initialise_chunk_kernel.h"


void ops_par_loop_initialise_chunk_kernel_x(char const * name, int dim, int *range,
                  ops_arg arg0, ops_arg arg1, ops_arg arg2) {

  char  **p_a[3];
  int   offs[3][2];
  int   count[dim];

  ops_arg args[3] = {arg0, arg1, arg2};

  for(int i=0; i<3; i++) {
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

  for (int i = 0; i < 3; i++) {
    if (args[i].argtype == OPS_ARG_DAT) {
      p_a[i] = (char **)malloc(args[i].stencil->points * sizeof(char *));
    }
    else if (args[i].argtype == OPS_ARG_GBL)
      p_a[i] = (char **)malloc(args[i].dim * sizeof(char *));
  }

  //set up initial pointers
  ops_args_set(range[0], range[2],3,args,p_a);

  for (int n_y = range[2]; n_y < range[3]; n_y++) {
    for (int n_x = range[0]; n_x < range[1]; n_x++) {
      // call kernel function, passing in pointers to data
      initialise_chunk_kernel_x( (double **)p_a[0], (int ** )p_a[1], (double ** )p_a[2]);

      for (int np=0; np<args[0].stencil->points; np++) p_a[0][np] += (args[0].dat->size * offs[0][0]);
      for (int np=0; np<args[1].stencil->points; np++) p_a[1][np] += (args[1].dat->size * offs[1][0]);
      for (int np=0; np<args[2].stencil->points; np++) p_a[2][np] += (args[2].dat->size * offs[2][0]);
    }

    for (int np=0; np<args[0].stencil->points; np++) p_a[0][np] += (args[0].dat->size * offs[0][1]);
    for (int np=0; np<args[1].stencil->points; np++) p_a[1][np] += (args[1].dat->size * offs[1][1]);
    for (int np=0; np<args[2].stencil->points; np++) p_a[2][np] += (args[2].dat->size * offs[2][1]);
  }

  for (int i = 0; i < 3; i++)
    free(p_a[i]);
}

void ops_par_loop_initialise_chunk_kernel_y(char const * name, int dim, int *range,
                  ops_arg arg0, ops_arg arg1, ops_arg arg2) {

  char  **p_a[3];
  int   offs[3][2];
  int   count[dim];

  ops_arg args[3] = {arg0, arg1, arg2};

  for(int i=0; i<3; i++) {
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

  for (int i = 0; i < 3; i++) {
    if (args[i].argtype == OPS_ARG_DAT) {
      p_a[i] = (char **)malloc(args[i].stencil->points * sizeof(char *));
    }
    else if (args[i].argtype == OPS_ARG_GBL)
      p_a[i] = (char **)malloc(args[i].dim * sizeof(char *));
  }

  //set up initial pointers
  ops_args_set(range[0], range[2],3,args,p_a);

  for (int n_y = range[2]; n_y < range[3]; n_y++) {
    for (int n_x = range[0]; n_x < range[1]; n_x++) {
      // call kernel function, passing in pointers to data
      initialise_chunk_kernel_y( (double **)p_a[0], (int ** )p_a[1], (double ** )p_a[2]);

      for (int np=0; np<args[0].stencil->points; np++) p_a[0][np] += (args[0].dat->size * offs[0][0]);
      for (int np=0; np<args[1].stencil->points; np++) p_a[1][np] += (args[1].dat->size * offs[1][0]);
      for (int np=0; np<args[2].stencil->points; np++) p_a[2][np] += (args[2].dat->size * offs[2][0]);
    }

    for (int np=0; np<args[0].stencil->points; np++) p_a[0][np] += (args[0].dat->size * offs[0][1]);
    for (int np=0; np<args[1].stencil->points; np++) p_a[1][np] += (args[1].dat->size * offs[1][1]);
    for (int np=0; np<args[2].stencil->points; np++) p_a[2][np] += (args[2].dat->size * offs[2][1]);
  }

  for (int i = 0; i < 3; i++)
    free(p_a[i]);
}

void ops_par_loop_initialise_chunk_kernel_cellx(char const * name, int dim, int *range,
                  ops_arg arg0, ops_arg arg1, ops_arg arg2) {

  char  **p_a[3];
  int   offs[3][2];
  int   count[dim];

  ops_arg args[3] = {arg0, arg1, arg2};

  for(int i=0; i<3; i++) {
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

  for (int i = 0; i < 3; i++) {
    if (args[i].argtype == OPS_ARG_DAT) {
      p_a[i] = (char **)malloc(args[i].stencil->points * sizeof(char *));
    }
    else if (args[i].argtype == OPS_ARG_GBL)
      p_a[i] = (char **)malloc(args[i].dim * sizeof(char *));
  }

  //set up initial pointers
  ops_args_set(range[0], range[2],3,args,p_a);

  for (int n_y = range[2]; n_y < range[3]; n_y++) {
    for (int n_x = range[0]; n_x < range[1]; n_x++) {
      // call kernel function, passing in pointers to data
      initialise_chunk_kernel_cellx( (double **)p_a[0], (double ** )p_a[1], (double ** )p_a[2]);

      for (int np=0; np<args[0].stencil->points; np++) p_a[0][np] += (args[0].dat->size * offs[0][0]);
      for (int np=0; np<args[1].stencil->points; np++) p_a[1][np] += (args[1].dat->size * offs[1][0]);
      for (int np=0; np<args[2].stencil->points; np++) p_a[2][np] += (args[2].dat->size * offs[2][0]);
    }

    for (int np=0; np<args[0].stencil->points; np++) p_a[0][np] += (args[0].dat->size * offs[0][1]);
    for (int np=0; np<args[1].stencil->points; np++) p_a[1][np] += (args[1].dat->size * offs[1][1]);
    for (int np=0; np<args[2].stencil->points; np++) p_a[2][np] += (args[2].dat->size * offs[2][1]);
  }

  for (int i = 0; i < 3; i++)
    free(p_a[i]);
}

void ops_par_loop_initialise_chunk_kernel_celly(char const * name, int dim, int *range,
                  ops_arg arg0, ops_arg arg1, ops_arg arg2) {

  char  **p_a[3];
  int   offs[3][2];
  int   count[dim];

  ops_arg args[3] = {arg0, arg1, arg2};

  for(int i=0; i<3; i++) {
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

  for (int i = 0; i < 3; i++) {
    if (args[i].argtype == OPS_ARG_DAT) {
      p_a[i] = (char **)malloc(args[i].stencil->points * sizeof(char *));
    }
    else if (args[i].argtype == OPS_ARG_GBL)
      p_a[i] = (char **)malloc(args[i].dim * sizeof(char *));
  }

  //set up initial pointers
  ops_args_set(range[0], range[2],3,args,p_a);

  for (int n_y = range[2]; n_y < range[3]; n_y++) {
    for (int n_x = range[0]; n_x < range[1]; n_x++) {
      // call kernel function, passing in pointers to data
      initialise_chunk_kernel_celly( (double **)p_a[0], (double ** )p_a[1], (double ** )p_a[2]);

      for (int np=0; np<args[0].stencil->points; np++) p_a[0][np] += (args[0].dat->size * offs[0][0]);
      for (int np=0; np<args[1].stencil->points; np++) p_a[1][np] += (args[1].dat->size * offs[1][0]);
      for (int np=0; np<args[2].stencil->points; np++) p_a[2][np] += (args[2].dat->size * offs[2][0]);
    }

    for (int np=0; np<args[0].stencil->points; np++) p_a[0][np] += (args[0].dat->size * offs[0][1]);
    for (int np=0; np<args[1].stencil->points; np++) p_a[1][np] += (args[1].dat->size * offs[1][1]);
    for (int np=0; np<args[2].stencil->points; np++) p_a[2][np] += (args[2].dat->size * offs[2][1]);
  }

  for (int i = 0; i < 3; i++)
    free(p_a[i]);
}


void ops_par_loop_initialise_volume_xarea_yarea(char const * name, int dim, int *range,
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
      initialise_volume_xarea_yarea( (double **)p_a[0], (double ** )p_a[1], (double ** )p_a[2],
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

void initialise_chunk()
{
  //initialize sizes using global values
  int x_cells = grid->x_cells;
  int y_cells = grid->y_cells;
  int x_min = field->x_min;
  int x_max = field->x_max;
  int y_min = field->y_min;
  int y_max = field->y_max;

  int self[] = {0,0};
  ops_stencil sten1 = ops_decl_stencil( 2, 1, self, "self");

  int rangex[] = {x_min-2, x_max+3, 0, 1};
  ops_par_loop_initialise_chunk_kernel_x("initialise_chunk_kernel_x", 2, rangex,
               ops_arg_dat(vertexx, S2D_00, OPS_WRITE),
               ops_arg_dat(xx, S2D_00, OPS_READ),
               ops_arg_dat(vertexdx, S2D_00, OPS_WRITE));

  int rangey[] = {0, 1, y_min-2, y_max+3};
  ops_par_loop_initialise_chunk_kernel_y("initialise_chunk_kernel_y", 2, rangey,
               ops_arg_dat(vertexy, S2D_00, OPS_WRITE),
               ops_arg_dat(yy, S2D_00, OPS_READ),
               ops_arg_dat(vertexdy, S2D_00, OPS_WRITE));

  rangex[0] = x_min-2; rangex[1] = x_max+2; rangex[2] = 0; rangex[3] = 1;
  ops_par_loop_initialise_chunk_kernel_cellx("initialise_chunk_kernel_cellx", 2, rangex,
               ops_arg_dat(vertexx, S2D_00_P10, OPS_READ),
               ops_arg_dat(cellx, S2D_00, OPS_WRITE),
               ops_arg_dat(celldx, S2D_00, OPS_WRITE));

  rangey[0] = 0; rangey[1] = 1; rangey[2] = y_min-2; rangey[3] = y_max+2;
  ops_par_loop_initialise_chunk_kernel_celly("initialise_chunk_kernel_celly", 2, rangey,
               ops_arg_dat(vertexy, S2D_00_0P1, OPS_READ),
               ops_arg_dat(celly, S2D_00, OPS_WRITE),
               ops_arg_dat(celldy, S2D_00, OPS_WRITE));

  int rangexy[] = {x_min-2,x_max+2,y_min-2,y_max+2};
  ops_par_loop_initialise_volume_xarea_yarea("initialise_volume_xarea_yarea", 2, rangexy,
    ops_arg_dat(volume, S2D_00, OPS_WRITE),
    ops_arg_dat(celldy, sten_self_stride2D_y, OPS_READ),
    ops_arg_dat(xarea, S2D_00, OPS_WRITE),
    ops_arg_dat(celldx, sten_self_stride2D_x, OPS_READ),
    ops_arg_dat(yarea, S2D_00, OPS_WRITE));
}
