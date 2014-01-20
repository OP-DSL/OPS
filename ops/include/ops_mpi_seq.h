
/*
* Open source copyright declaration based on BSD open source template:
* http://www.opensource.org/licenses/bsd-license.php
*
* This file is part of the OPS distribution.
*
* Copyright (c) 2013, Mike Giles and others. Please see the AUTHORS file in
* the main source directory for a full list of copyright holders.
* All rights reserved.
*
* Redistribution and use in source and binary forms, with or without
* modification, are permitted provided that the following conditions are met:
* * Redistributions of source code must retain the above copyright
* notice, this list of conditions and the following disclaimer.
* * Redistributions in binary form must reproduce the above copyright
* notice, this list of conditions and the following disclaimer in the
* documentation and/or other materials provided with the distribution.
* * The name of Mike Giles may not be used to endorse or promote products
* derived from this software without specific prior written permission.
*
* THIS SOFTWARE IS PROVIDED BY Mike Giles ''AS IS'' AND ANY
* EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
* WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
* DISCLAIMED. IN NO EVENT SHALL Mike Giles BE LIABLE FOR ANY
* DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
* (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
* LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
* ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
* (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
* SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

/** @brief header file declaring the functions for the ops sequential backend
  * @author Gihan Mudalige, Istvan Reguly
  * @details Declares the OPS API calls for the sequential backend
  */

#include "ops_lib_cpp.h"
#include "ops_mpi_core.h"

#ifndef OPS_ACC_MACROS
#ifndef OPS_DEBUG
#define OPS_ACC0(x,y) (x+xdim0*(y))
#define OPS_ACC1(x,y) (x+xdim1*(y))
#define OPS_ACC2(x,y) (x+xdim2*(y))
#define OPS_ACC3(x,y) (x+xdim3*(y))
#define OPS_ACC4(x,y) (x+xdim4*(y))
#define OPS_ACC5(x,y) (x+xdim5*(y))
#define OPS_ACC6(x,y) (x+xdim6*(y))
#define OPS_ACC7(x,y) (x+xdim7*(y))
#define OPS_ACC8(x,y) (x+xdim8*(y))
#define OPS_ACC9(x,y) (x+xdim9*(y))
#define OPS_ACC10(x,y) (x+xdim10*(y))
#define OPS_ACC11(x,y) (x+xdim11*(y))
#define OPS_ACC12(x,y) (x+xdim12*(y))
#define OPS_ACC13(x,y) (x+xdim13*(y))
#define OPS_ACC14(x,y) (x+xdim14*(y))
#define OPS_ACC15(x,y) (x+xdim15*(y))
#else
#define OPS_ACC0(x,y) (ops_stencil_check_2d(0, x, y, xdim0, -1))
#define OPS_ACC1(x,y) (ops_stencil_check_2d(1, x, y, xdim1, -1))
#define OPS_ACC2(x,y) (ops_stencil_check_2d(2, x, y, xdim2, -1))
#define OPS_ACC3(x,y) (ops_stencil_check_2d(3, x, y, xdim3, -1))
#define OPS_ACC4(x,y) (ops_stencil_check_2d(4, x, y, xdim4, -1))
#define OPS_ACC5(x,y) (ops_stencil_check_2d(5, x, y, xdim5, -1))
#define OPS_ACC6(x,y) (ops_stencil_check_2d(6, x, y, xdim6, -1))
#define OPS_ACC7(x,y) (ops_stencil_check_2d(7, x, y, xdim7, -1))
#define OPS_ACC8(x,y) (ops_stencil_check_2d(8, x, y, xdim8, -1))
#define OPS_ACC9(x,y) (ops_stencil_check_2d(9, x, y, xdim8, -1))
#define OPS_ACC10(x,y) (ops_stencil_check_2d(10, x, y, xdim10, -1))
#define OPS_ACC11(x,y) (ops_stencil_check_2d(11, x, y, xdim11, -1))
#define OPS_ACC12(x,y) (ops_stencil_check_2d(12, x, y, xdim12, -1))
#define OPS_ACC13(x,y) (ops_stencil_check_2d(13, x, y, xdim13, -1))
#define OPS_ACC14(x,y) (ops_stencil_check_2d(14, x, y, xdim14, -1))
#define OPS_ACC15(x,y) (ops_stencil_check_2d(15, x, y, xdim15, -1))
#endif
#endif

extern int xdim0;
extern int xdim1;
extern int xdim2;
extern int xdim3;
extern int xdim4;
extern int xdim5;
extern int xdim6;
extern int xdim7;
extern int xdim8;
extern int xdim9;
extern int xdim10;
extern int xdim11;
extern int xdim12;
extern int xdim13;
extern int xdim14;
extern int xdim15;



inline int mult(int* co, int* s, int r)
{
  int result = 1;
  if(r > 0) {
    for(int i = 0; i<r;i++) result *= s[i];
  }
  result = result * co[r];
  return result;
}

inline int add(int* co, int* s, int r)
{
  int result = co[0];
  for(int i = 1; i<=r;i++) result += mult(co,s,i);
  return result;
}


inline int off(int ndim, int r, int* ps, int* pe, int* size)
{

  int i = 0;
  int* c1 = (int*) xmalloc(sizeof(int)*ndim);
  int* c2 = (int*) xmalloc(sizeof(int)*ndim);

  for(i=0; i<ndim; i++) c1[i] = ps[i];
  c1[r] = ps[r] + 1;

  for(i = 0; i<r; i++) c2[i] = pe[i];
  for(i=r; i<ndim; i++) c2[i] = ps[i];

  int off =  add(c1, size, r) - add(c2, size, r) + 1; //plus 1 to get the next element

  free(c1);free(c2);
  return off;
}



//
//ops_par_loop routine for 1 arguments
//
template <class T0>
void ops_par_loop_mpi(void (*kernel)(T0*),
     char const * name, ops_block block, int dim, int *range,
     ops_arg arg0) {

  char *p_a[1];
  int  offs[1][2];

  int  count[dim];
  ops_arg args[1] = { arg0};

  sub_dat_list sd0 = OPS_sub_dat_list[arg0.dat->index];
  sub_block_list sb = OPS_sub_block_list[block->index];

  //compute localy allocated range for the sub-block
  int ndim = sb->ndim;
  int ps[ndim];
  int pe[ndim];
  int* start = (int*) xmalloc(sizeof(int)*ndim);
  int* end = (int*) xmalloc(sizeof(int)*ndim);

  for(int n=0; n<sb->ndim; n++) {
    ps[n] = sb->istart[n];
    pe[n] = sb->iend[n]+1; //+1 is for C indexing
    //printf("ps[n] : %d, pe[n] : %d elements: %d\n",ps[n],pe[n], pe[n]-ps[n]);
  }


  //determin the valid range to iterate over on this MPI process
  for(int n=0; n<ndim; n++) {
    //printf("range[ndim*n] : %d, range[ndim*n+1] : %d elements: %d\n",
      //range[ndim*n],range[ndim*n+1], range[ndim*n+1]-range[ndim*n]);
    if(pe[n] >= ps[n]) {
      if(ps[n] >= range[ndim*n] && pe[n] <= range[ndim*n + 1]) {
        start[n] = ps[n] - arg0.dat->offset[n];
        end[n]   = pe[n] - arg0.dat->offset[n];
      }
      else if (ps[n] < range[ndim*n] && pe[n] <= range[ndim*n + 1]) {
        start[n] = range[ndim*n] - arg0.dat->offset[n];
        end[n]   = pe[n] - arg0.dat->offset[n];
      }
      else if (ps[n] < range[ndim*n] && pe[n] > range[ndim*n + 1]) {
        start[n] = range[ndim*n]  - arg0.dat->offset[n];
        end[n]   = range[ndim*n+1] - arg0.dat->offset[n];
      }
      else if (ps[n] < range[ndim*n] && pe[n] < range[ndim*n + 1]) {
        start[n] = 0;
        end[n]   = 0;
      }
      else if (ps[n] > range[ndim*n] && pe[n] > range[ndim*n + 1]) {
        start[n] = 0;
        end[n]   = 0;
      }
    }
  }

  for (int i = 0; i<1;i++) {
    if(args[i].stencil!=NULL) {
      offs[i][0] = 1;  //unit step in x dimension
      for(int n=1; n<ndim; n++) {
        offs[i][1] = off(2, 1, start, end, args[i].dat->block_size);
      }

      int correct = ops_offs_set(range[0],range[2]+1, args[i]) - ops_offs_set(range[1],range[2], args[i]) +1;

      printf("correct = %d, test = %d\n",correct, offs[i][1]);
      offs[i][1] = correct;

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

  //set up initial pointers
  for (int i = 0; i < 1; i++) {
    if (args[i].argtype == OPS_ARG_DAT) {
      p_a[i] = (char *)args[i].data //base of 2D array
      +
      //y dimension -- get to the correct y line
      args[i].dat->size * args[i].dat->block_size[0] * ( range[2] * args[i].stencil->stride[1] - args[i].dat->offset[1] )
      +
      //x dimension - get to the correct x point on the y line
      args[i].dat->size * ( range[0] * args[i].stencil->stride[0] - args[i].dat->offset[0] );
    }
    else if (args[i].argtype == OPS_ARG_GBL)
      p_a[i] = (char *)args[i].data;
  }

  int total_range = 1;
  for (int m=0; m<dim; m++) {
    count[m] = range[2*m+1]-range[2*m];  // number in each dimension
    total_range *= count[m];
  }
  count[dim-1]++;     // extra in last to ensure correct termination

  if (args[0].argtype == OPS_ARG_DAT)  xdim0 = args[0].dat->block_size[0];

  for (int nt=0; nt<total_range; nt++) {
    // call kernel function, passing in pointers to data

    kernel(  (T0 *)p_a[0] );

    count[0]--;   // decrement counter
    int m = 0;    // max dimension with changed index
    while (count[m]==0) {
      count[m] = range[2*m+1]-range[2*m]; // reset counter
      m++;                                // next dimension
      count[m]--;                         // decrement counter
    }

    int a = 0;
    // shift pointers to data
    for (int i=0; i<1; i++) {
      if (args[i].argtype == OPS_ARG_DAT)
        p_a[i] = p_a[i] + (args[i].dat->size * offs[i][m]);
    }
  }
}


/*
template <class T0>
void ops_par_loop_mpi(void (*kernel)(T0*),
     char const * name, ops_block block, int dim, int *range,
     ops_arg arg0) {

  char *p_a[1];
  int  offs[1][2];

  int  count[dim];
  ops_arg args[1] = { arg0};
  sub_dat_list sd0 = OPS_sub_dat_list[arg0.dat->index];

  sub_block_list sb = OPS_sub_block_list[block->index];

  //check if all args, if they are dats are defined on this block
  for(int i=0; i<1; i++) {
    //if dat is not defined for this block -- exit with error message
    /**TO DO
  }

  //compute localy allocated range for the sub-block
  int ndim = sb->ndim;
  int ps[ndim];
  int pe[ndim];
  for(int n=0; n<sb->ndim; n++) {
    ps[n] = sb->istart[n];
    pe[n] = sb->iend[n];
  }

  //determin the valid range to iterate over on this MPI process
  int start[ndim];
  int end[ndim];
  for(int n=0; n<ndim; n++) {
    if(pe[n] >= ps[n]) {
      if(ps[n] >= range[ndim*n] && pe[n] <= range[ndim*n + 1]) {
        start[n] = ps[n]; end[n] = pe[n];
      }
    }
  }



  //compute offsets -- need a way to generalize the calculation for any dimension

  //setup initial pointers -- need a way to generalize the calculation for any dimension


}*/
