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

#include "data.h"
#include "definitions.h"

#include "accelerate_kernel.h"

#define OPS_ACC0(x,y) (x+xdim0*y)
#define OPS_ACC1(x,y) (x+xdim1*y)
#define OPS_ACC2(x,y) (x+xdim2*y)
#define OPS_ACC3(x,y) (x+xdim3*y)

int xdim0 = 0;
int xdim1 = 0;
int xdim2 = 0;
int xdim3 = 0;

void accelerate_kernel_stepbymass_test( double *density0, double *volume,
                double *stepbymass, double *pressure) {

  double nodal_mass;

  //{0,0, -1,0, 0,-1, -1,-1};
  nodal_mass = ( density0[OPS_ACC0(-1,-1)] * volume[OPS_ACC1(-1,-1)]
    + density0[OPS_ACC0(0,-1)] * volume[OPS_ACC1(0,-1)]
    + density0[OPS_ACC0(0,0)] * volume[OPS_ACC1(0,0)]
    + density0[OPS_ACC0(-1,0)] * volume[OPS_ACC1(-1,0)] ) * 0.25;

  stepbymass[OPS_ACC2(0,0)] = 0.5*dt / nodal_mass;

}

inline void ops_arg_set_test(int n_x,
                        int n_y, ops_arg arg, char *p_arg){
  //if (arg.stencil!=NULL) {
      p_arg = (
        arg.data + //base of 2D array
        //y dimension -- get to the correct y line
        arg.dat->size * arg.dat->block_size[0] * ( //multiply by the number of
                                                    //bytes per element and xdim block size
        n_y - arg.dat->offset[1] ) // calculate the offset from index 0 for y dim
        +
        //x dimension - get to the correct x point on the y line
        arg.dat->size * ( //multiply by the number of bytes per element
        n_x - arg.dat->offset[0] //calculate the offset from index 0 for x dim
      ) );
    //}
}

inline void ops_args_set_test(int iter_x,
                         int iter_y,
                         int nargs, ops_arg *args, char **p_a){
  for (int n=0; n<nargs; n++) {
    ops_arg_set_test(iter_x, iter_y, args[n], p_a[n]);
  }
}


//
//ops_par_loop routine for 4 arguments
//
template <class T0,class T1,class T2,class T3>
void ops_par_loop_test(void (*kernel)(T0*, T1*, T2*, T3*),
     char const * name, int dim, int *range,
     ops_arg arg0, ops_arg arg1, ops_arg arg2, ops_arg arg3) {

  char **p_a;
  int  offs[4][2];

  int  count[dim];
  ops_arg args[4] = { arg0, arg1, arg2, arg3};

  for (int i = 0; i<4;i++) {
    if(args[i].stencil!=NULL) {
      offs[i][0] = 1;  //unit step in x dimension
      offs[i][1] = ops_offs_set(range[0],range[2]+1, args[i]) - ops_offs_set(range[1],range[2], args[i]) +1;
      if (args[i].stencil->stride[0] == 0) { //stride in y as x stride is 0
        offs[i][0] = 0;
        offs[i][1] = args[i].dat->block_size[0];
      }
      else if (args[i].stencil->stride[1] == 0) {//stride in x as y stride is 0
        offs[i][0] = 1;
        offs[i][1] = -( range[1] - range[0] ) +1;
      }
    }
  }

  p_a = (char **)malloc(4 * sizeof(char *));


  int total_range = 1;
  for (int m=0; m<dim; m++) {
    count[m] = range[2*m+1]-range[2*m];  // number in each dimension
    total_range *= count[m];
  }
  count[dim-1]++;     // extra in last to ensure correct termination

  xdim0 = args[0].dat->block_size[0];
  xdim1 = args[1].dat->block_size[0];
  xdim2 = args[2].dat->block_size[0];
  xdim3 = args[3].dat->block_size[0];

  //set up initial pointers
  ops_args_set_test(range[0], range[2], 4, args, p_a); //set up the initial possition

  for (int nt=0; nt<total_range; nt++) {
    // call kernel function, passing in pointers to data

    kernel(  (T0 *)p_a[0], (T1 *)p_a[1], (T2 *)p_a[2], (T3 *)p_a[3] );

    count[0]--;   // decrement counter
    int m = 0;    // max dimension with changed index
    while (count[m]==0) {
      count[m] = range[2*m+1]-range[2*m]; // reset counter
      m++;                                // next dimension
      count[m]--;                         // decrement counter
    }

    // shift pointers to data
    for (int a = 0; a<4;a++) {
      p_a[a] = p_a[a] + (args[a].dat->size * offs[a][m]);
    }
  }

  free(p_a);

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

  ops_par_loop_test(accelerate_kernel_stepbymass_test, "accelerate_kernel_stepbymass_test", 2, rangexy_inner_plus1,
    ops_arg_dat(density0, S2D_00_M10_0M1_M1M1, "double", OPS_READ),
    ops_arg_dat(volume, S2D_00_M10_0M1_M1M1, "double", OPS_READ),
    ops_arg_dat(work_array1, S2D_00, "double", OPS_WRITE),
    ops_arg_dat(pressure, S2D_00_M10_0M1_M1M1, "double", OPS_READ));

  ops_par_loop_opt(accelerate_kernelx1, "accelerate_kernelx1", 2, rangexy_inner_plus1,
    ops_arg_dat(xvel0, S2D_00, "double", OPS_READ),
    ops_arg_dat(xvel1, S2D_00, "double", OPS_WRITE),
    ops_arg_dat(work_array1, S2D_00, "double", OPS_READ),
    ops_arg_dat(xarea, S2D_00_0M1, "double", OPS_READ),
    ops_arg_dat(pressure, S2D_00_M10_0M1_M1M1, "double", OPS_READ));

  ops_par_loop_opt(accelerate_kernely1, "accelerate_kernely1", 2, rangexy_inner_plus1,
    ops_arg_dat(yvel0, S2D_00, "double", OPS_READ),
    ops_arg_dat(yvel1, S2D_00, "double", OPS_WRITE),
    ops_arg_dat(work_array1, S2D_00, "double", OPS_READ),
    ops_arg_dat(yarea, S2D_00_M10, "double", OPS_READ),
    ops_arg_dat(pressure, S2D_00_M10_0M1_M1M1, "double", OPS_READ));

  ops_par_loop_opt(accelerate_kernelx2, "accelerate_kernelx2", 2, rangexy_inner_plus1,
    ops_arg_dat(xvel1, S2D_00, "double", OPS_INC),
    ops_arg_dat(work_array1, S2D_00, "double", OPS_READ),
    ops_arg_dat(xarea, S2D_00_0M1, "double", OPS_READ),
    ops_arg_dat(viscosity, S2D_00_M10_0M1_M1M1, "double", OPS_READ));

  ops_par_loop_opt(accelerate_kernely2, "accelerate_kernely2", 2, rangexy_inner_plus1,
    ops_arg_dat(yvel1, S2D_00, "double", OPS_INC),
    ops_arg_dat(work_array1, S2D_00, "double", OPS_READ),
    ops_arg_dat(yarea, S2D_00_M10, "double", OPS_READ),
    ops_arg_dat(viscosity, S2D_00_M10_0M1_M1M1, "double", OPS_READ));

}
