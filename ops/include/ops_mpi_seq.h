
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
#define OPS_ACC16(x,y) (x+xdim16*(y))
#define OPS_ACC17(x,y) (x+xdim17*(y))
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
#define OPS_ACC9(x,y) (ops_stencil_check_2d(9, x, y, xdim9, -1))
#define OPS_ACC10(x,y) (ops_stencil_check_2d(10, x, y, xdim10, -1))
#define OPS_ACC11(x,y) (ops_stencil_check_2d(11, x, y, xdim11, -1))
#define OPS_ACC12(x,y) (ops_stencil_check_2d(12, x, y, xdim12, -1))
#define OPS_ACC13(x,y) (ops_stencil_check_2d(13, x, y, xdim13, -1))
#define OPS_ACC14(x,y) (ops_stencil_check_2d(14, x, y, xdim14, -1))
#define OPS_ACC15(x,y) (ops_stencil_check_2d(15, x, y, xdim15, -1))
#define OPS_ACC16(x,y) (ops_stencil_check_2d(16, x, y, xdim16, -1))
#define OPS_ACC17(x,y) (ops_stencil_check_2d(17, x, y, xdim17, -1))
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
extern int xdim16;
extern int xdim17;

inline int mult(int* s, int r)
{
  int result = 1;
  if(r > 0) {
    for(int i = 0; i<r;i++) result *= s[i];
  }
  return result;
}

inline int add(int* co, int* s, int r)
{
  int result = co[0];
  for(int i = 1; i<=r;i++) result += co[i]*mult(s,i);
  return result;
}


inline int off(int ndim, int r, int* ps, int* pe, int* size, int* std)
{

  int i = 0;
  int* c1 = (int*) xmalloc(sizeof(int)*ndim);
  int* c2 = (int*) xmalloc(sizeof(int)*ndim);

  for(i=0; i<ndim; i++) c1[i] = ps[i];
  c1[r] = ps[r] + 1*std[r];

  for(i = 0; i<r; i++) std[i]!=0 ? c2[i] = pe[i]:c2[i] = ps[i]+1;
  for(i=r; i<ndim; i++) c2[i] = ps[i];

  int off =  add(c1, size, r) - add(c2, size, r) + 1; //plus 1 to get the next element

  free(c1);free(c2);
  return off;
}

inline int address(int ndim, int dat_size, int* ps, int* size, int* std, int* off)
{
  int base = 0;
  for(int i=0; i<ndim; i++) {
    base = base + dat_size * mult(size, i) * (ps[i] * std[i] - off[i]);
  }
  return base;
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

  sub_block_list sb = OPS_sub_block_list[block->index];


  //compute localy allocated range for the sub-block
  int ndim = sb->ndim;
  int* start = (int*) xmalloc(sizeof(int)*ndim*1);
  int* end = (int*) xmalloc(sizeof(int)*ndim*1);

  int s[ndim];
  int e[ndim];

  for (int n=0; n<ndim; n++) {
    s[n] = sb->istart[n];e[n] = sb->iend[n]+1;
    if(s[n] >= range[2*n]) s[n] = 0;
    else s[n] = range[2*n] - s[n];
    if(e[n] >= range[2*n+1]) e[n] = range[2*n+1] - sb->istart[n];
    else e[n] = sb->sizes[n];
  }
  for(int i = 0; i<1; i++) {
    for(int n=0; n<ndim; n++) {
      start[i*ndim+n] = s[n] ;//range[2*n];- args[i].dat->offset[n];
      end[i*ndim+n]   = e[n] ;//range[2*n+1];- args[i].dat->offset[n];
    }
  }

  #ifdef OPS_DEBUG
  ops_register_args(args, name);
  #endif

  for (int i = 0; i<1;i++) {
    if(args[i].stencil!=NULL) {
      offs[i][0] = args[i].stencil->stride[0]*1;  //unit step in x dimension
      for(int n=1; n<ndim; n++) {
        offs[i][n] = off(ndim, n, &start[i*ndim], &end[i*ndim],
                         args[i].dat->block_size, args[i].stencil->stride);
      }
    }
  }

  //set up initial pointers
  for (int i = 0; i < 1; i++) {
    if (args[i].argtype == OPS_ARG_DAT) {
      p_a[i] = (char *)args[i].data //base of 2D array
      + address(ndim, args[i].dat->size, &start[i*ndim],
        args[i].dat->block_size, args[i].stencil->stride, args[i].dat->offset);
    }
    else if (args[i].argtype == OPS_ARG_GBL)
      p_a[i] = (char *)args[i].data;
  }

  free(start);free(end);

  int total_range = 1;
  for (int n=0; n<ndim; n++) {
    count[n] = e[n]-s[n];  // number in each dimension
    total_range *= count[n];
  }
  count[dim-1]++;     // extra in last to ensure correct termination

  if (args[0].argtype == OPS_ARG_DAT)  xdim0 = args[0].dat->block_size[0];

  for (int nt=0; nt<total_range; nt++) {
    // call kernel function, passing in pointers to data

    kernel(  (T0 *)p_a[0] );

    count[0]--;   // decrement counter
    int m = 0;    // max dimension with changed index
    while (count[m]==0) {
      count[m] =  e[m]-s[m];// reset counter
      m++;                        // next dimension
      count[m]--;                 // decrement counter
    }

    int a = 0;
    // shift pointers to data
    for (int i=0; i<1; i++) {
      if (args[i].argtype == OPS_ARG_DAT)
        p_a[i] = p_a[i] + (args[i].dat->size * offs[i][m]);
    }
  }

}

//
//ops_par_loop routine for 2 arguments
//
template <class T0,class T1>
void ops_par_loop_mpi(void (*kernel)(T0*, T1*),
     char const * name, ops_block block, int dim, int *range,
     ops_arg arg0, ops_arg arg1) {

  char *p_a[2];
  int  offs[2][2];

  int  count[dim];
  ops_arg args[2] = { arg0, arg1};

  sub_block_list sb = OPS_sub_block_list[block->index];


  //compute localy allocated range for the sub-block
  int ndim = sb->ndim;
  int* start = (int*) xmalloc(sizeof(int)*ndim*2);
  int* end = (int*) xmalloc(sizeof(int)*ndim*2);

  int s[ndim];
  int e[ndim];

  for (int n=0; n<ndim; n++) {
    s[n] = sb->istart[n];e[n] = sb->iend[n]+1;
    if(s[n] >= range[2*n]) s[n] = 0;
    else s[n] = range[2*n] - s[n];
    if(e[n] >= range[2*n+1]) e[n] = range[2*n+1] - sb->istart[n];
    else e[n] = sb->sizes[n];
  }
  for(int i = 0; i<2; i++) {
    for(int n=0; n<ndim; n++) {
      start[i*ndim+n] = s[n] ;//range[2*n];- args[i].dat->offset[n];
      end[i*ndim+n]   = e[n] ;//range[2*n+1];- args[i].dat->offset[n];
    }
  }

  #ifdef OPS_DEBUG
  ops_register_args(args, name);
  #endif

  for (int i = 0; i<2;i++) {
    if(args[i].stencil!=NULL) {
      offs[i][0] = args[i].stencil->stride[0]*1;  //unit step in x dimension
      for(int n=1; n<ndim; n++) {
        offs[i][n] = off(ndim, n, &start[i*ndim], &end[i*ndim],
                         args[i].dat->block_size, args[i].stencil->stride);
      }
    }
  }

  //set up initial pointers
  for (int i = 0; i < 2; i++) {
    if (args[i].argtype == OPS_ARG_DAT) {
      p_a[i] = (char *)args[i].data //base of 2D array
      + address(ndim, args[i].dat->size, &start[i*ndim],
        args[i].dat->block_size, args[i].stencil->stride, args[i].dat->offset);
    }
    else if (args[i].argtype == OPS_ARG_GBL)
      p_a[i] = (char *)args[i].data;
  }

  free(start);free(end);

  int total_range = 1;
  for (int n=0; n<ndim; n++) {
    count[n] = e[n]-s[n];  // number in each dimension
    total_range *= count[n];
  }
  count[dim-1]++;     // extra in last to ensure correct termination

  if (args[0].argtype == OPS_ARG_DAT)  xdim0 = args[0].dat->block_size[0];
  if (args[1].argtype == OPS_ARG_DAT)  xdim1 = args[1].dat->block_size[0];

  for (int nt=0; nt<total_range; nt++) {
    // call kernel function, passing in pointers to data

    kernel(  (T0 *)p_a[0], (T1 *)p_a[1] );

    count[0]--;   // decrement counter
    int m = 0;    // max dimension with changed index
    while (count[m]==0) {
      count[m] =  e[m]-s[m];// reset counter
      m++;                        // next dimension
      count[m]--;                 // decrement counter
    }

    int a = 0;
    // shift pointers to data
    for (int i=0; i<2; i++) {
      if (args[i].argtype == OPS_ARG_DAT)
        p_a[i] = p_a[i] + (args[i].dat->size * offs[i][m]);
    }
  }

}

//
//ops_par_loop routine for 3 arguments
//
template <class T0,class T1,class T2>
void ops_par_loop_mpi(void (*kernel)(T0*, T1*, T2*),
     char const * name, ops_block block, int dim, int *range,
     ops_arg arg0, ops_arg arg1, ops_arg arg2) {

  char *p_a[3];
  int  offs[3][2];

  int  count[dim];
  ops_arg args[3] = { arg0, arg1, arg2};

  sub_block_list sb = OPS_sub_block_list[block->index];


  //compute localy allocated range for the sub-block
  int ndim = sb->ndim;
  int* start = (int*) xmalloc(sizeof(int)*ndim*3);
  int* end = (int*) xmalloc(sizeof(int)*ndim*3);

  int s[ndim];
  int e[ndim];

  for (int n=0; n<ndim; n++) {
    s[n] = sb->istart[n];e[n] = sb->iend[n]+1;
    if(s[n] >= range[2*n]) s[n] = 0;
    else s[n] = range[2*n] - s[n];
    if(e[n] >= range[2*n+1]) e[n] = range[2*n+1] - sb->istart[n];
    else e[n] = sb->sizes[n];
  }
  for(int i = 0; i<3; i++) {
    for(int n=0; n<ndim; n++) {
      start[i*ndim+n] = s[n] ;//range[2*n];- args[i].dat->offset[n];
      end[i*ndim+n]   = e[n] ;//range[2*n+1];- args[i].dat->offset[n];
    }
  }

  #ifdef OPS_DEBUG
  ops_register_args(args, name);
  #endif

  for (int i = 0; i<3;i++) {
    if(args[i].stencil!=NULL) {
      offs[i][0] = args[i].stencil->stride[0]*1;  //unit step in x dimension
      for(int n=1; n<ndim; n++) {
        offs[i][n] = off(ndim, n, &start[i*ndim], &end[i*ndim],
                         args[i].dat->block_size, args[i].stencil->stride);
      }
    }
  }

  //set up initial pointers
  for (int i = 0; i < 3; i++) {
    if (args[i].argtype == OPS_ARG_DAT) {
      p_a[i] = (char *)args[i].data //base of 2D array
      + address(ndim, args[i].dat->size, &start[i*ndim],
        args[i].dat->block_size, args[i].stencil->stride, args[i].dat->offset);
    }
    else if (args[i].argtype == OPS_ARG_GBL)
      p_a[i] = (char *)args[i].data;
  }

  free(start);free(end);

  int total_range = 1;
  for (int n=0; n<ndim; n++) {
    count[n] = e[n]-s[n];  // number in each dimension
    total_range *= count[n];
  }
  count[dim-1]++;     // extra in last to ensure correct termination

  if (args[0].argtype == OPS_ARG_DAT)  xdim0 = args[0].dat->block_size[0];
  if (args[1].argtype == OPS_ARG_DAT)  xdim1 = args[1].dat->block_size[0];
  if (args[2].argtype == OPS_ARG_DAT)  xdim2 = args[2].dat->block_size[0];

  for (int nt=0; nt<total_range; nt++) {
    // call kernel function, passing in pointers to data

    kernel(  (T0 *)p_a[0], (T1 *)p_a[1], (T2 *)p_a[2] );

    count[0]--;   // decrement counter
    int m = 0;    // max dimension with changed index
    while (count[m]==0) {
      count[m] =  e[m]-s[m];// reset counter
      m++;                        // next dimension
      count[m]--;                 // decrement counter
    }

    int a = 0;
    // shift pointers to data
    for (int i=0; i<3; i++) {
      if (args[i].argtype == OPS_ARG_DAT)
        p_a[i] = p_a[i] + (args[i].dat->size * offs[i][m]);
    }
  }

}

//
//ops_par_loop routine for 4 arguments
//
template <class T0,class T1,class T2,class T3>
void ops_par_loop_mpi(void (*kernel)(T0*, T1*, T2*, T3*),
     char const * name, ops_block block, int dim, int *range,
     ops_arg arg0, ops_arg arg1, ops_arg arg2, ops_arg arg3) {

  char *p_a[4];
  int  offs[4][2];

  int  count[dim];
  ops_arg args[4] = { arg0, arg1, arg2, arg3};

  sub_block_list sb = OPS_sub_block_list[block->index];


  //compute localy allocated range for the sub-block
  int ndim = sb->ndim;
  int* start = (int*) xmalloc(sizeof(int)*ndim*4);
  int* end = (int*) xmalloc(sizeof(int)*ndim*4);

  int s[ndim];
  int e[ndim];

  for (int n=0; n<ndim; n++) {
    s[n] = sb->istart[n];e[n] = sb->iend[n]+1;
    if(s[n] >= range[2*n]) s[n] = 0;
    else s[n] = range[2*n] - s[n];
    if(e[n] >= range[2*n+1]) e[n] = range[2*n+1] - sb->istart[n];
    else e[n] = sb->sizes[n];
  }
  for(int i = 0; i<4; i++) {
    for(int n=0; n<ndim; n++) {
      start[i*ndim+n] = s[n] ;//range[2*n];- args[i].dat->offset[n];
      end[i*ndim+n]   = e[n] ;//range[2*n+1];- args[i].dat->offset[n];
    }
  }

  #ifdef OPS_DEBUG
  ops_register_args(args, name);
  #endif

  for (int i = 0; i<4;i++) {
    if(args[i].stencil!=NULL) {
      offs[i][0] = args[i].stencil->stride[0]*1;  //unit step in x dimension
      for(int n=1; n<ndim; n++) {
        offs[i][n] = off(ndim, n, &start[i*ndim], &end[i*ndim],
                         args[i].dat->block_size, args[i].stencil->stride);
      }
    }
  }

  //set up initial pointers
  for (int i = 0; i < 4; i++) {
    if (args[i].argtype == OPS_ARG_DAT) {
      p_a[i] = (char *)args[i].data //base of 2D array
      + address(ndim, args[i].dat->size, &start[i*ndim],
        args[i].dat->block_size, args[i].stencil->stride, args[i].dat->offset);
    }
    else if (args[i].argtype == OPS_ARG_GBL)
      p_a[i] = (char *)args[i].data;
  }

  free(start);free(end);

  int total_range = 1;
  for (int n=0; n<ndim; n++) {
    count[n] = e[n]-s[n];  // number in each dimension
    total_range *= count[n];
  }
  count[dim-1]++;     // extra in last to ensure correct termination

  if (args[0].argtype == OPS_ARG_DAT)  xdim0 = args[0].dat->block_size[0];
  if (args[1].argtype == OPS_ARG_DAT)  xdim1 = args[1].dat->block_size[0];
  if (args[2].argtype == OPS_ARG_DAT)  xdim2 = args[2].dat->block_size[0];
  if (args[3].argtype == OPS_ARG_DAT)  xdim3 = args[3].dat->block_size[0];

  for (int nt=0; nt<total_range; nt++) {
    // call kernel function, passing in pointers to data

    kernel(  (T0 *)p_a[0], (T1 *)p_a[1], (T2 *)p_a[2], (T3 *)p_a[3] );

    count[0]--;   // decrement counter
    int m = 0;    // max dimension with changed index
    while (count[m]==0) {
      count[m] =  e[m]-s[m];// reset counter
      m++;                        // next dimension
      count[m]--;                 // decrement counter
    }

    int a = 0;
    // shift pointers to data
    for (int i=0; i<4; i++) {
      if (args[i].argtype == OPS_ARG_DAT)
        p_a[i] = p_a[i] + (args[i].dat->size * offs[i][m]);
    }
  }

}

//
//ops_par_loop routine for 5 arguments
//
template <class T0,class T1,class T2,class T3,
class T4>
void ops_par_loop_mpi(void (*kernel)(T0*, T1*, T2*, T3*,
                           T4*),
     char const * name, ops_block block, int dim, int *range,
     ops_arg arg0, ops_arg arg1, ops_arg arg2, ops_arg arg3,
     ops_arg arg4) {

  char *p_a[5];
  int  offs[5][2];

  int  count[dim];
  ops_arg args[5] = { arg0, arg1, arg2, arg3,
                     arg4};

  sub_block_list sb = OPS_sub_block_list[block->index];


  //compute localy allocated range for the sub-block
  int ndim = sb->ndim;
  int* start = (int*) xmalloc(sizeof(int)*ndim*5);
  int* end = (int*) xmalloc(sizeof(int)*ndim*5);

  int s[ndim];
  int e[ndim];

  for (int n=0; n<ndim; n++) {
    s[n] = sb->istart[n];e[n] = sb->iend[n]+1;
    if(s[n] >= range[2*n]) s[n] = 0;
    else s[n] = range[2*n] - s[n];
    if(e[n] >= range[2*n+1]) e[n] = range[2*n+1] - sb->istart[n];
    else e[n] = sb->sizes[n];
  }
  for(int i = 0; i<5; i++) {
    for(int n=0; n<ndim; n++) {
      start[i*ndim+n] = s[n] ;//range[2*n];- args[i].dat->offset[n];
      end[i*ndim+n]   = e[n] ;//range[2*n+1];- args[i].dat->offset[n];
    }
  }

  #ifdef OPS_DEBUG
  ops_register_args(args, name);
  #endif

  for (int i = 0; i<5;i++) {
    if(args[i].stencil!=NULL) {
      offs[i][0] = args[i].stencil->stride[0]*1;  //unit step in x dimension
      for(int n=1; n<ndim; n++) {
        offs[i][n] = off(ndim, n, &start[i*ndim], &end[i*ndim],
                         args[i].dat->block_size, args[i].stencil->stride);
      }
    }
  }

  //set up initial pointers
  for (int i = 0; i < 5; i++) {
    if (args[i].argtype == OPS_ARG_DAT) {
      p_a[i] = (char *)args[i].data //base of 2D array
      + address(ndim, args[i].dat->size, &start[i*ndim],
        args[i].dat->block_size, args[i].stencil->stride, args[i].dat->offset);
    }
    else if (args[i].argtype == OPS_ARG_GBL)
      p_a[i] = (char *)args[i].data;
  }

  free(start);free(end);

  int total_range = 1;
  for (int n=0; n<ndim; n++) {
    count[n] = e[n]-s[n];  // number in each dimension
    total_range *= count[n];
  }
  count[dim-1]++;     // extra in last to ensure correct termination

  if (args[0].argtype == OPS_ARG_DAT)  xdim0 = args[0].dat->block_size[0];
  if (args[1].argtype == OPS_ARG_DAT)  xdim1 = args[1].dat->block_size[0];
  if (args[2].argtype == OPS_ARG_DAT)  xdim2 = args[2].dat->block_size[0];
  if (args[3].argtype == OPS_ARG_DAT)  xdim3 = args[3].dat->block_size[0];
  if (args[4].argtype == OPS_ARG_DAT)  xdim4 = args[4].dat->block_size[0];

  for (int nt=0; nt<total_range; nt++) {
    // call kernel function, passing in pointers to data

    kernel(  (T0 *)p_a[0], (T1 *)p_a[1], (T2 *)p_a[2], (T3 *)p_a[3],
           (T4 *)p_a[4] );

    count[0]--;   // decrement counter
    int m = 0;    // max dimension with changed index
    while (count[m]==0) {
      count[m] =  e[m]-s[m];// reset counter
      m++;                        // next dimension
      count[m]--;                 // decrement counter
    }

    int a = 0;
    // shift pointers to data
    for (int i=0; i<5; i++) {
      if (args[i].argtype == OPS_ARG_DAT)
        p_a[i] = p_a[i] + (args[i].dat->size * offs[i][m]);
    }
  }

}

//
//ops_par_loop routine for 6 arguments
//
template <class T0,class T1,class T2,class T3,
class T4,class T5>
void ops_par_loop_mpi(void (*kernel)(T0*, T1*, T2*, T3*,
                           T4*, T5*),
     char const * name, ops_block block, int dim, int *range,
     ops_arg arg0, ops_arg arg1, ops_arg arg2, ops_arg arg3,
     ops_arg arg4, ops_arg arg5) {

  char *p_a[6];
  int  offs[6][2];

  int  count[dim];
  ops_arg args[6] = { arg0, arg1, arg2, arg3,
                     arg4, arg5};

  sub_block_list sb = OPS_sub_block_list[block->index];


  //compute localy allocated range for the sub-block
  int ndim = sb->ndim;
  int* start = (int*) xmalloc(sizeof(int)*ndim*6);
  int* end = (int*) xmalloc(sizeof(int)*ndim*6);

  int s[ndim];
  int e[ndim];

  for (int n=0; n<ndim; n++) {
    s[n] = sb->istart[n];e[n] = sb->iend[n]+1;
    if(s[n] >= range[2*n]) s[n] = 0;
    else s[n] = range[2*n] - s[n];
    if(e[n] >= range[2*n+1]) e[n] = range[2*n+1] - sb->istart[n];
    else e[n] = sb->sizes[n];
  }
  for(int i = 0; i<6; i++) {
    for(int n=0; n<ndim; n++) {
      start[i*ndim+n] = s[n] ;//range[2*n];- args[i].dat->offset[n];
      end[i*ndim+n]   = e[n] ;//range[2*n+1];- args[i].dat->offset[n];
    }
  }

  #ifdef OPS_DEBUG
  ops_register_args(args, name);
  #endif

  for (int i = 0; i<6;i++) {
    if(args[i].stencil!=NULL) {
      offs[i][0] = args[i].stencil->stride[0]*1;  //unit step in x dimension
      for(int n=1; n<ndim; n++) {
        offs[i][n] = off(ndim, n, &start[i*ndim], &end[i*ndim],
                         args[i].dat->block_size, args[i].stencil->stride);
      }
    }
  }

  //set up initial pointers
  for (int i = 0; i < 6; i++) {
    if (args[i].argtype == OPS_ARG_DAT) {
      p_a[i] = (char *)args[i].data //base of 2D array
      + address(ndim, args[i].dat->size, &start[i*ndim],
        args[i].dat->block_size, args[i].stencil->stride, args[i].dat->offset);
    }
    else if (args[i].argtype == OPS_ARG_GBL)
      p_a[i] = (char *)args[i].data;
  }

  free(start);free(end);

  int total_range = 1;
  for (int n=0; n<ndim; n++) {
    count[n] = e[n]-s[n];  // number in each dimension
    total_range *= count[n];
  }
  count[dim-1]++;     // extra in last to ensure correct termination

  if (args[0].argtype == OPS_ARG_DAT)  xdim0 = args[0].dat->block_size[0];
  if (args[1].argtype == OPS_ARG_DAT)  xdim1 = args[1].dat->block_size[0];
  if (args[2].argtype == OPS_ARG_DAT)  xdim2 = args[2].dat->block_size[0];
  if (args[3].argtype == OPS_ARG_DAT)  xdim3 = args[3].dat->block_size[0];
  if (args[4].argtype == OPS_ARG_DAT)  xdim4 = args[4].dat->block_size[0];
  if (args[5].argtype == OPS_ARG_DAT)  xdim5 = args[5].dat->block_size[0];

  for (int nt=0; nt<total_range; nt++) {
    // call kernel function, passing in pointers to data

    kernel(  (T0 *)p_a[0], (T1 *)p_a[1], (T2 *)p_a[2], (T3 *)p_a[3],
           (T4 *)p_a[4], (T5 *)p_a[5] );

    count[0]--;   // decrement counter
    int m = 0;    // max dimension with changed index
    while (count[m]==0) {
      count[m] =  e[m]-s[m];// reset counter
      m++;                        // next dimension
      count[m]--;                 // decrement counter
    }

    int a = 0;
    // shift pointers to data
    for (int i=0; i<6; i++) {
      if (args[i].argtype == OPS_ARG_DAT)
        p_a[i] = p_a[i] + (args[i].dat->size * offs[i][m]);
    }
  }

}

//
//ops_par_loop routine for 7 arguments
//
template <class T0,class T1,class T2,class T3,
class T4,class T5,class T6>
void ops_par_loop_mpi(void (*kernel)(T0*, T1*, T2*, T3*,
                           T4*, T5*, T6*),
     char const * name, ops_block block, int dim, int *range,
     ops_arg arg0, ops_arg arg1, ops_arg arg2, ops_arg arg3,
     ops_arg arg4, ops_arg arg5, ops_arg arg6) {

  char *p_a[7];
  int  offs[7][2];

  int  count[dim];
  ops_arg args[7] = { arg0, arg1, arg2, arg3,
                     arg4, arg5, arg6};

  sub_block_list sb = OPS_sub_block_list[block->index];


  //compute localy allocated range for the sub-block
  int ndim = sb->ndim;
  int* start = (int*) xmalloc(sizeof(int)*ndim*7);
  int* end = (int*) xmalloc(sizeof(int)*ndim*7);

  int s[ndim];
  int e[ndim];

  for (int n=0; n<ndim; n++) {
    s[n] = sb->istart[n];e[n] = sb->iend[n]+1;
    if(s[n] >= range[2*n]) s[n] = 0;
    else s[n] = range[2*n] - s[n];
    if(e[n] >= range[2*n+1]) e[n] = range[2*n+1] - sb->istart[n];
    else e[n] = sb->sizes[n];
  }
  for(int i = 0; i<7; i++) {
    for(int n=0; n<ndim; n++) {
      start[i*ndim+n] = s[n] ;//range[2*n];- args[i].dat->offset[n];
      end[i*ndim+n]   = e[n] ;//range[2*n+1];- args[i].dat->offset[n];
    }
  }

  #ifdef OPS_DEBUG
  ops_register_args(args, name);
  #endif

  for (int i = 0; i<7;i++) {
    if(args[i].stencil!=NULL) {
      offs[i][0] = args[i].stencil->stride[0]*1;  //unit step in x dimension
      for(int n=1; n<ndim; n++) {
        offs[i][n] = off(ndim, n, &start[i*ndim], &end[i*ndim],
                         args[i].dat->block_size, args[i].stencil->stride);
      }
    }
  }

  //set up initial pointers
  for (int i = 0; i < 7; i++) {
    if (args[i].argtype == OPS_ARG_DAT) {
      p_a[i] = (char *)args[i].data //base of 2D array
      + address(ndim, args[i].dat->size, &start[i*ndim],
        args[i].dat->block_size, args[i].stencil->stride, args[i].dat->offset);
    }
    else if (args[i].argtype == OPS_ARG_GBL)
      p_a[i] = (char *)args[i].data;
  }

  free(start);free(end);

  int total_range = 1;
  for (int n=0; n<ndim; n++) {
    count[n] = e[n]-s[n];  // number in each dimension
    total_range *= count[n];
  }
  count[dim-1]++;     // extra in last to ensure correct termination

  if (args[0].argtype == OPS_ARG_DAT)  xdim0 = args[0].dat->block_size[0];
  if (args[1].argtype == OPS_ARG_DAT)  xdim1 = args[1].dat->block_size[0];
  if (args[2].argtype == OPS_ARG_DAT)  xdim2 = args[2].dat->block_size[0];
  if (args[3].argtype == OPS_ARG_DAT)  xdim3 = args[3].dat->block_size[0];
  if (args[4].argtype == OPS_ARG_DAT)  xdim4 = args[4].dat->block_size[0];
  if (args[5].argtype == OPS_ARG_DAT)  xdim5 = args[5].dat->block_size[0];
  if (args[6].argtype == OPS_ARG_DAT)  xdim6 = args[6].dat->block_size[0];

  for (int nt=0; nt<total_range; nt++) {
    // call kernel function, passing in pointers to data

    kernel(  (T0 *)p_a[0], (T1 *)p_a[1], (T2 *)p_a[2], (T3 *)p_a[3],
           (T4 *)p_a[4], (T5 *)p_a[5], (T6 *)p_a[6] );

    count[0]--;   // decrement counter
    int m = 0;    // max dimension with changed index
    while (count[m]==0) {
      count[m] =  e[m]-s[m];// reset counter
      m++;                        // next dimension
      count[m]--;                 // decrement counter
    }

    int a = 0;
    // shift pointers to data
    for (int i=0; i<7; i++) {
      if (args[i].argtype == OPS_ARG_DAT)
        p_a[i] = p_a[i] + (args[i].dat->size * offs[i][m]);
    }
  }

}

//
//ops_par_loop routine for 8 arguments
//
template <class T0,class T1,class T2,class T3,
class T4,class T5,class T6,class T7>
void ops_par_loop_mpi(void (*kernel)(T0*, T1*, T2*, T3*,
                           T4*, T5*, T6*, T7*),
     char const * name, ops_block block, int dim, int *range,
     ops_arg arg0, ops_arg arg1, ops_arg arg2, ops_arg arg3,
     ops_arg arg4, ops_arg arg5, ops_arg arg6, ops_arg arg7) {

  char *p_a[8];
  int  offs[8][2];

  int  count[dim];
  ops_arg args[8] = { arg0, arg1, arg2, arg3,
                     arg4, arg5, arg6, arg7};

  sub_block_list sb = OPS_sub_block_list[block->index];


  //compute localy allocated range for the sub-block
  int ndim = sb->ndim;
  int* start = (int*) xmalloc(sizeof(int)*ndim*8);
  int* end = (int*) xmalloc(sizeof(int)*ndim*8);

  int s[ndim];
  int e[ndim];

  for (int n=0; n<ndim; n++) {
    s[n] = sb->istart[n];e[n] = sb->iend[n]+1;
    if(s[n] >= range[2*n]) s[n] = 0;
    else s[n] = range[2*n] - s[n];
    if(e[n] >= range[2*n+1]) e[n] = range[2*n+1] - sb->istart[n];
    else e[n] = sb->sizes[n];
  }
  for(int i = 0; i<8; i++) {
    for(int n=0; n<ndim; n++) {
      start[i*ndim+n] = s[n] ;//range[2*n];- args[i].dat->offset[n];
      end[i*ndim+n]   = e[n] ;//range[2*n+1];- args[i].dat->offset[n];
    }
  }

  #ifdef OPS_DEBUG
  ops_register_args(args, name);
  #endif

  for (int i = 0; i<8;i++) {
    if(args[i].stencil!=NULL) {
      offs[i][0] = args[i].stencil->stride[0]*1;  //unit step in x dimension
      for(int n=1; n<ndim; n++) {
        offs[i][n] = off(ndim, n, &start[i*ndim], &end[i*ndim],
                         args[i].dat->block_size, args[i].stencil->stride);
      }
    }
  }

  //set up initial pointers
  for (int i = 0; i < 8; i++) {
    if (args[i].argtype == OPS_ARG_DAT) {
      p_a[i] = (char *)args[i].data //base of 2D array
      + address(ndim, args[i].dat->size, &start[i*ndim],
        args[i].dat->block_size, args[i].stencil->stride, args[i].dat->offset);
    }
    else if (args[i].argtype == OPS_ARG_GBL)
      p_a[i] = (char *)args[i].data;
  }

  free(start);free(end);

  int total_range = 1;
  for (int n=0; n<ndim; n++) {
    count[n] = e[n]-s[n];  // number in each dimension
    total_range *= count[n];
  }
  count[dim-1]++;     // extra in last to ensure correct termination

  if (args[0].argtype == OPS_ARG_DAT)  xdim0 = args[0].dat->block_size[0];
  if (args[1].argtype == OPS_ARG_DAT)  xdim1 = args[1].dat->block_size[0];
  if (args[2].argtype == OPS_ARG_DAT)  xdim2 = args[2].dat->block_size[0];
  if (args[3].argtype == OPS_ARG_DAT)  xdim3 = args[3].dat->block_size[0];
  if (args[4].argtype == OPS_ARG_DAT)  xdim4 = args[4].dat->block_size[0];
  if (args[5].argtype == OPS_ARG_DAT)  xdim5 = args[5].dat->block_size[0];
  if (args[6].argtype == OPS_ARG_DAT)  xdim6 = args[6].dat->block_size[0];
  if (args[7].argtype == OPS_ARG_DAT)  xdim7 = args[7].dat->block_size[0];

  ops_exchange_halo(&arg2,2);
  ops_exchange_halo(&arg3,2);
  ops_exchange_halo(&arg4,2);
  ops_exchange_halo(&arg5,2);

  for (int nt=0; nt<total_range; nt++) {
    // call kernel function, passing in pointers to data

    kernel(  (T0 *)p_a[0], (T1 *)p_a[1], (T2 *)p_a[2], (T3 *)p_a[3],
           (T4 *)p_a[4], (T5 *)p_a[5], (T6 *)p_a[6], (T7 *)p_a[7] );

    count[0]--;   // decrement counter
    int m = 0;    // max dimension with changed index
    while (count[m]==0) {
      count[m] =  e[m]-s[m];// reset counter
      m++;                        // next dimension
      count[m]--;                 // decrement counter
    }

    int a = 0;
    // shift pointers to data
    for (int i=0; i<8; i++) {
      if (args[i].argtype == OPS_ARG_DAT)
        p_a[i] = p_a[i] + (args[i].dat->size * offs[i][m]);
    }
  }
}

//
//ops_par_loop routine for 9 arguments
//
template <class T0,class T1,class T2,class T3,
class T4,class T5,class T6,class T7,
class T8>
void ops_par_loop_mpi(void (*kernel)(T0*, T1*, T2*, T3*,
                           T4*, T5*, T6*, T7*,
                           T8*),
     char const * name, ops_block block, int dim, int *range,
     ops_arg arg0, ops_arg arg1, ops_arg arg2, ops_arg arg3,
     ops_arg arg4, ops_arg arg5, ops_arg arg6, ops_arg arg7,
     ops_arg arg8) {

  char *p_a[9];
  int  offs[9][2];

  int  count[dim];
  ops_arg args[9] = { arg0, arg1, arg2, arg3,
                     arg4, arg5, arg6, arg7,
                     arg8};

  sub_block_list sb = OPS_sub_block_list[block->index];


  //compute localy allocated range for the sub-block
  int ndim = sb->ndim;
  int* start = (int*) xmalloc(sizeof(int)*ndim*9);
  int* end = (int*) xmalloc(sizeof(int)*ndim*9);

  int s[ndim];
  int e[ndim];

  for (int n=0; n<ndim; n++) {
    s[n] = sb->istart[n];e[n] = sb->iend[n]+1;
    if(s[n] >= range[2*n]) s[n] = 0;
    else s[n] = range[2*n] - s[n];
    if(e[n] >= range[2*n+1]) e[n] = range[2*n+1] - sb->istart[n];
    else e[n] = sb->sizes[n];
  }
  for(int i = 0; i<9; i++) {
    for(int n=0; n<ndim; n++) {
      start[i*ndim+n] = s[n] ;//range[2*n];- args[i].dat->offset[n];
      end[i*ndim+n]   = e[n] ;//range[2*n+1];- args[i].dat->offset[n];
    }
  }

  #ifdef OPS_DEBUG
  ops_register_args(args, name);
  #endif

  for (int i = 0; i<9;i++) {
    if(args[i].stencil!=NULL) {
      offs[i][0] = args[i].stencil->stride[0]*1;  //unit step in x dimension
      for(int n=1; n<ndim; n++) {
        offs[i][n] = off(ndim, n, &start[i*ndim], &end[i*ndim],
                         args[i].dat->block_size, args[i].stencil->stride);
      }
    }
  }

  //set up initial pointers
  for (int i = 0; i < 9; i++) {
    if (args[i].argtype == OPS_ARG_DAT) {
      p_a[i] = (char *)args[i].data //base of 2D array
      + address(ndim, args[i].dat->size, &start[i*ndim],
        args[i].dat->block_size, args[i].stencil->stride, args[i].dat->offset);
    }
    else if (args[i].argtype == OPS_ARG_GBL)
      p_a[i] = (char *)args[i].data;
  }

  free(start);free(end);

  int total_range = 1;
  for (int n=0; n<ndim; n++) {
    count[n] = e[n]-s[n];  // number in each dimension
    total_range *= count[n];
  }
  count[dim-1]++;     // extra in last to ensure correct termination

  if (args[0].argtype == OPS_ARG_DAT)  xdim0 = args[0].dat->block_size[0];
  if (args[1].argtype == OPS_ARG_DAT)  xdim1 = args[1].dat->block_size[0];
  if (args[2].argtype == OPS_ARG_DAT)  xdim2 = args[2].dat->block_size[0];
  if (args[3].argtype == OPS_ARG_DAT)  xdim3 = args[3].dat->block_size[0];
  if (args[4].argtype == OPS_ARG_DAT)  xdim4 = args[4].dat->block_size[0];
  if (args[5].argtype == OPS_ARG_DAT)  xdim5 = args[5].dat->block_size[0];
  if (args[6].argtype == OPS_ARG_DAT)  xdim6 = args[6].dat->block_size[0];
  if (args[7].argtype == OPS_ARG_DAT)  xdim7 = args[7].dat->block_size[0];
  if (args[8].argtype == OPS_ARG_DAT)  xdim8 = args[8].dat->block_size[0];

  for (int nt=0; nt<total_range; nt++) {
    // call kernel function, passing in pointers to data

    kernel(  (T0 *)p_a[0], (T1 *)p_a[1], (T2 *)p_a[2], (T3 *)p_a[3],
           (T4 *)p_a[4], (T5 *)p_a[5], (T6 *)p_a[6], (T7 *)p_a[7],
           (T8 *)p_a[8] );

    count[0]--;   // decrement counter
    int m = 0;    // max dimension with changed index
    while (count[m]==0) {
      count[m] =  e[m]-s[m];// reset counter
      m++;                        // next dimension
      count[m]--;                 // decrement counter
    }

    int a = 0;
    // shift pointers to data
    for (int i=0; i<9; i++) {
      if (args[i].argtype == OPS_ARG_DAT)
        p_a[i] = p_a[i] + (args[i].dat->size * offs[i][m]);
    }
  }

}

//
//ops_par_loop routine for 10 arguments
//
template <class T0,class T1,class T2,class T3,
class T4,class T5,class T6,class T7,
class T8,class T9>
void ops_par_loop_mpi(void (*kernel)(T0*, T1*, T2*, T3*,
                           T4*, T5*, T6*, T7*,
                           T8*, T9*),
     char const * name, ops_block block, int dim, int *range,
     ops_arg arg0, ops_arg arg1, ops_arg arg2, ops_arg arg3,
     ops_arg arg4, ops_arg arg5, ops_arg arg6, ops_arg arg7,
     ops_arg arg8, ops_arg arg9) {

  char *p_a[10];
  int  offs[10][2];

  int  count[dim];
  ops_arg args[10] = { arg0, arg1, arg2, arg3,
                     arg4, arg5, arg6, arg7,
                     arg8, arg9};

  sub_block_list sb = OPS_sub_block_list[block->index];


  //compute localy allocated range for the sub-block
  int ndim = sb->ndim;
  int* start = (int*) xmalloc(sizeof(int)*ndim*10);
  int* end = (int*) xmalloc(sizeof(int)*ndim*10);

  int s[ndim];
  int e[ndim];

  for (int n=0; n<ndim; n++) {
    s[n] = sb->istart[n];e[n] = sb->iend[n]+1;
    if(s[n] >= range[2*n]) s[n] = 0;
    else s[n] = range[2*n] - s[n];
    if(e[n] >= range[2*n+1]) e[n] = range[2*n+1] - sb->istart[n];
    else e[n] = sb->sizes[n];
  }
  for(int i = 0; i<10; i++) {
    for(int n=0; n<ndim; n++) {
      start[i*ndim+n] = s[n] ;//range[2*n];- args[i].dat->offset[n];
      end[i*ndim+n]   = e[n] ;//range[2*n+1];- args[i].dat->offset[n];
    }
  }

  #ifdef OPS_DEBUG
  ops_register_args(args, name);
  #endif

  for (int i = 0; i<10;i++) {
    if(args[i].stencil!=NULL) {
      offs[i][0] = args[i].stencil->stride[0]*1;  //unit step in x dimension
      for(int n=1; n<ndim; n++) {
        offs[i][n] = off(ndim, n, &start[i*ndim], &end[i*ndim],
                         args[i].dat->block_size, args[i].stencil->stride);
      }
    }
  }

  //set up initial pointers
  for (int i = 0; i < 10; i++) {
    if (args[i].argtype == OPS_ARG_DAT) {
      p_a[i] = (char *)args[i].data //base of 2D array
      + address(ndim, args[i].dat->size, &start[i*ndim],
        args[i].dat->block_size, args[i].stencil->stride, args[i].dat->offset);
    }
    else if (args[i].argtype == OPS_ARG_GBL)
      p_a[i] = (char *)args[i].data;
  }

  free(start);free(end);

  int total_range = 1;
  for (int n=0; n<ndim; n++) {
    count[n] = e[n]-s[n];  // number in each dimension
    total_range *= count[n];
  }
  count[dim-1]++;     // extra in last to ensure correct termination

  if (args[0].argtype == OPS_ARG_DAT)  xdim0 = args[0].dat->block_size[0];
  if (args[1].argtype == OPS_ARG_DAT)  xdim1 = args[1].dat->block_size[0];
  if (args[2].argtype == OPS_ARG_DAT)  xdim2 = args[2].dat->block_size[0];
  if (args[3].argtype == OPS_ARG_DAT)  xdim3 = args[3].dat->block_size[0];
  if (args[4].argtype == OPS_ARG_DAT)  xdim4 = args[4].dat->block_size[0];
  if (args[5].argtype == OPS_ARG_DAT)  xdim5 = args[5].dat->block_size[0];
  if (args[6].argtype == OPS_ARG_DAT)  xdim6 = args[6].dat->block_size[0];
  if (args[7].argtype == OPS_ARG_DAT)  xdim7 = args[7].dat->block_size[0];
  if (args[8].argtype == OPS_ARG_DAT)  xdim8 = args[8].dat->block_size[0];
  if (args[9].argtype == OPS_ARG_DAT)  xdim9 = args[9].dat->block_size[0];

  for (int nt=0; nt<total_range; nt++) {
    // call kernel function, passing in pointers to data

    kernel(  (T0 *)p_a[0], (T1 *)p_a[1], (T2 *)p_a[2], (T3 *)p_a[3],
           (T4 *)p_a[4], (T5 *)p_a[5], (T6 *)p_a[6], (T7 *)p_a[7],
           (T8 *)p_a[8], (T9 *)p_a[9] );

    count[0]--;   // decrement counter
    int m = 0;    // max dimension with changed index
    while (count[m]==0) {
      count[m] =  e[m]-s[m];// reset counter
      m++;                        // next dimension
      count[m]--;                 // decrement counter
    }

    int a = 0;
    // shift pointers to data
    for (int i=0; i<10; i++) {
      if (args[i].argtype == OPS_ARG_DAT)
        p_a[i] = p_a[i] + (args[i].dat->size * offs[i][m]);
    }
  }

}

//
//ops_par_loop routine for 11 arguments
//
template <class T0,class T1,class T2,class T3,
class T4,class T5,class T6,class T7,
class T8,class T9,class T10>
void ops_par_loop_mpi(void (*kernel)(T0*, T1*, T2*, T3*,
                           T4*, T5*, T6*, T7*,
                           T8*, T9*, T10*),
     char const * name, ops_block block, int dim, int *range,
     ops_arg arg0, ops_arg arg1, ops_arg arg2, ops_arg arg3,
     ops_arg arg4, ops_arg arg5, ops_arg arg6, ops_arg arg7,
     ops_arg arg8, ops_arg arg9, ops_arg arg10) {

  char *p_a[11];
  int  offs[11][2];

  int  count[dim];
  ops_arg args[11] = { arg0, arg1, arg2, arg3,
                     arg4, arg5, arg6, arg7,
                     arg8, arg9, arg10};

  sub_block_list sb = OPS_sub_block_list[block->index];


  //compute localy allocated range for the sub-block
  int ndim = sb->ndim;
  int* start = (int*) xmalloc(sizeof(int)*ndim*11);
  int* end = (int*) xmalloc(sizeof(int)*ndim*11);

  int s[ndim];
  int e[ndim];

  for (int n=0; n<ndim; n++) {
    s[n] = sb->istart[n];e[n] = sb->iend[n]+1;
    if(s[n] >= range[2*n]) s[n] = 0;
    else s[n] = range[2*n] - s[n];
    if(e[n] >= range[2*n+1]) e[n] = range[2*n+1] - sb->istart[n];
    else e[n] = sb->sizes[n];
  }
  for(int i = 0; i<11; i++) {
    for(int n=0; n<ndim; n++) {
      start[i*ndim+n] = s[n] ;//range[2*n];- args[i].dat->offset[n];
      end[i*ndim+n]   = e[n] ;//range[2*n+1];- args[i].dat->offset[n];
    }
  }

  #ifdef OPS_DEBUG
  ops_register_args(args, name);
  #endif

  for (int i = 0; i<11;i++) {
    if(args[i].stencil!=NULL) {
      offs[i][0] = args[i].stencil->stride[0]*1;  //unit step in x dimension
      for(int n=1; n<ndim; n++) {
        offs[i][n] = off(ndim, n, &start[i*ndim], &end[i*ndim],
                         args[i].dat->block_size, args[i].stencil->stride);
      }
    }
  }

  //set up initial pointers
  for (int i = 0; i < 11; i++) {
    if (args[i].argtype == OPS_ARG_DAT) {
      p_a[i] = (char *)args[i].data //base of 2D array
      + address(ndim, args[i].dat->size, &start[i*ndim],
        args[i].dat->block_size, args[i].stencil->stride, args[i].dat->offset);
    }
    else if (args[i].argtype == OPS_ARG_GBL)
      p_a[i] = (char *)args[i].data;
  }

  free(start);free(end);

  int total_range = 1;
  for (int n=0; n<ndim; n++) {
    count[n] = e[n]-s[n];  // number in each dimension
    total_range *= count[n];
  }
  count[dim-1]++;     // extra in last to ensure correct termination

  if (args[0].argtype == OPS_ARG_DAT)  xdim0 = args[0].dat->block_size[0];
  if (args[1].argtype == OPS_ARG_DAT)  xdim1 = args[1].dat->block_size[0];
  if (args[2].argtype == OPS_ARG_DAT)  xdim2 = args[2].dat->block_size[0];
  if (args[3].argtype == OPS_ARG_DAT)  xdim3 = args[3].dat->block_size[0];
  if (args[4].argtype == OPS_ARG_DAT)  xdim4 = args[4].dat->block_size[0];
  if (args[5].argtype == OPS_ARG_DAT)  xdim5 = args[5].dat->block_size[0];
  if (args[6].argtype == OPS_ARG_DAT)  xdim6 = args[6].dat->block_size[0];
  if (args[7].argtype == OPS_ARG_DAT)  xdim7 = args[7].dat->block_size[0];
  if (args[8].argtype == OPS_ARG_DAT)  xdim8 = args[8].dat->block_size[0];
  if (args[9].argtype == OPS_ARG_DAT)  xdim9 = args[9].dat->block_size[0];
  if (args[10].argtype == OPS_ARG_DAT)  xdim10 = args[10].dat->block_size[0];

  for (int nt=0; nt<total_range; nt++) {
    // call kernel function, passing in pointers to data

    kernel(  (T0 *)p_a[0], (T1 *)p_a[1], (T2 *)p_a[2], (T3 *)p_a[3],
           (T4 *)p_a[4], (T5 *)p_a[5], (T6 *)p_a[6], (T7 *)p_a[7],
           (T8 *)p_a[8], (T9 *)p_a[9], (T10 *)p_a[10] );

    count[0]--;   // decrement counter
    int m = 0;    // max dimension with changed index
    while (count[m]==0) {
      count[m] =  e[m]-s[m];// reset counter
      m++;                        // next dimension
      count[m]--;                 // decrement counter
    }

    int a = 0;
    // shift pointers to data
    for (int i=0; i<11; i++) {
      if (args[i].argtype == OPS_ARG_DAT)
        p_a[i] = p_a[i] + (args[i].dat->size * offs[i][m]);
    }
  }

}

//
//ops_par_loop routine for 12 arguments
//
template <class T0,class T1,class T2,class T3,
class T4,class T5,class T6,class T7,
class T8,class T9,class T10,class T11>
void ops_par_loop_mpi(void (*kernel)(T0*, T1*, T2*, T3*,
                           T4*, T5*, T6*, T7*,
                           T8*, T9*, T10*, T11*),
     char const * name, ops_block block, int dim, int *range,
     ops_arg arg0, ops_arg arg1, ops_arg arg2, ops_arg arg3,
     ops_arg arg4, ops_arg arg5, ops_arg arg6, ops_arg arg7,
     ops_arg arg8, ops_arg arg9, ops_arg arg10, ops_arg arg11) {

  char *p_a[12];
  int  offs[12][2];

  int  count[dim];
  ops_arg args[12] = { arg0, arg1, arg2, arg3,
                     arg4, arg5, arg6, arg7,
                     arg8, arg9, arg10, arg11};

  sub_block_list sb = OPS_sub_block_list[block->index];


  //compute localy allocated range for the sub-block
  int ndim = sb->ndim;
  int* start = (int*) xmalloc(sizeof(int)*ndim*12);
  int* end = (int*) xmalloc(sizeof(int)*ndim*12);

  int s[ndim];
  int e[ndim];

  for (int n=0; n<ndim; n++) {
    s[n] = sb->istart[n];e[n] = sb->iend[n]+1;
    if(s[n] >= range[2*n]) s[n] = 0;
    else s[n] = range[2*n] - s[n];
    if(e[n] >= range[2*n+1]) e[n] = range[2*n+1] - sb->istart[n];
    else e[n] = sb->sizes[n];
  }
  for(int i = 0; i<12; i++) {
    for(int n=0; n<ndim; n++) {
      start[i*ndim+n] = s[n] ;//range[2*n];- args[i].dat->offset[n];
      end[i*ndim+n]   = e[n] ;//range[2*n+1];- args[i].dat->offset[n];
    }
  }

  #ifdef OPS_DEBUG
  ops_register_args(args, name);
  #endif

  for (int i = 0; i<12;i++) {
    if(args[i].stencil!=NULL) {
      offs[i][0] = args[i].stencil->stride[0]*1;  //unit step in x dimension
      for(int n=1; n<ndim; n++) {
        offs[i][n] = off(ndim, n, &start[i*ndim], &end[i*ndim],
                         args[i].dat->block_size, args[i].stencil->stride);
      }
    }
  }

  //set up initial pointers
  for (int i = 0; i < 12; i++) {
    if (args[i].argtype == OPS_ARG_DAT) {
      p_a[i] = (char *)args[i].data //base of 2D array
      + address(ndim, args[i].dat->size, &start[i*ndim],
        args[i].dat->block_size, args[i].stencil->stride, args[i].dat->offset);
    }
    else if (args[i].argtype == OPS_ARG_GBL)
      p_a[i] = (char *)args[i].data;
  }

  free(start);free(end);

  int total_range = 1;
  for (int n=0; n<ndim; n++) {
    count[n] = e[n]-s[n];  // number in each dimension
    total_range *= count[n];
  }
  count[dim-1]++;     // extra in last to ensure correct termination

  if (args[0].argtype == OPS_ARG_DAT)  xdim0 = args[0].dat->block_size[0];
  if (args[1].argtype == OPS_ARG_DAT)  xdim1 = args[1].dat->block_size[0];
  if (args[2].argtype == OPS_ARG_DAT)  xdim2 = args[2].dat->block_size[0];
  if (args[3].argtype == OPS_ARG_DAT)  xdim3 = args[3].dat->block_size[0];
  if (args[4].argtype == OPS_ARG_DAT)  xdim4 = args[4].dat->block_size[0];
  if (args[5].argtype == OPS_ARG_DAT)  xdim5 = args[5].dat->block_size[0];
  if (args[6].argtype == OPS_ARG_DAT)  xdim6 = args[6].dat->block_size[0];
  if (args[7].argtype == OPS_ARG_DAT)  xdim7 = args[7].dat->block_size[0];
  if (args[8].argtype == OPS_ARG_DAT)  xdim8 = args[8].dat->block_size[0];
  if (args[9].argtype == OPS_ARG_DAT)  xdim9 = args[9].dat->block_size[0];
  if (args[10].argtype == OPS_ARG_DAT)  xdim10 = args[10].dat->block_size[0];
  if (args[11].argtype == OPS_ARG_DAT)  xdim11 = args[11].dat->block_size[0];

  for (int nt=0; nt<total_range; nt++) {
    // call kernel function, passing in pointers to data

    kernel(  (T0 *)p_a[0], (T1 *)p_a[1], (T2 *)p_a[2], (T3 *)p_a[3],
           (T4 *)p_a[4], (T5 *)p_a[5], (T6 *)p_a[6], (T7 *)p_a[7],
           (T8 *)p_a[8], (T9 *)p_a[9], (T10 *)p_a[10], (T11 *)p_a[11] );

    count[0]--;   // decrement counter
    int m = 0;    // max dimension with changed index
    while (count[m]==0) {
      count[m] =  e[m]-s[m];// reset counter
      m++;                        // next dimension
      count[m]--;                 // decrement counter
    }

    int a = 0;
    // shift pointers to data
    for (int i=0; i<12; i++) {
      if (args[i].argtype == OPS_ARG_DAT)
        p_a[i] = p_a[i] + (args[i].dat->size * offs[i][m]);
    }
  }

}

//
//ops_par_loop routine for 13 arguments
//
template <class T0,class T1,class T2,class T3,
class T4,class T5,class T6,class T7,
class T8,class T9,class T10,class T11,
class T12>
void ops_par_loop_mpi(void (*kernel)(T0*, T1*, T2*, T3*,
                           T4*, T5*, T6*, T7*,
                           T8*, T9*, T10*, T11*,
                           T12*),
     char const * name, ops_block block, int dim, int *range,
     ops_arg arg0, ops_arg arg1, ops_arg arg2, ops_arg arg3,
     ops_arg arg4, ops_arg arg5, ops_arg arg6, ops_arg arg7,
     ops_arg arg8, ops_arg arg9, ops_arg arg10, ops_arg arg11,
     ops_arg arg12) {

  char *p_a[13];
  int  offs[13][2];

  int  count[dim];
  ops_arg args[13] = { arg0, arg1, arg2, arg3,
                     arg4, arg5, arg6, arg7,
                     arg8, arg9, arg10, arg11,
                     arg12};

  sub_block_list sb = OPS_sub_block_list[block->index];


  //compute localy allocated range for the sub-block
  int ndim = sb->ndim;
  int* start = (int*) xmalloc(sizeof(int)*ndim*13);
  int* end = (int*) xmalloc(sizeof(int)*ndim*13);

  int s[ndim];
  int e[ndim];

  for (int n=0; n<ndim; n++) {
    s[n] = sb->istart[n];e[n] = sb->iend[n]+1;
    if(s[n] >= range[2*n]) s[n] = 0;
    else s[n] = range[2*n] - s[n];
    if(e[n] >= range[2*n+1]) e[n] = range[2*n+1] - sb->istart[n];
    else e[n] = sb->sizes[n];
  }
  for(int i = 0; i<13; i++) {
    for(int n=0; n<ndim; n++) {
      start[i*ndim+n] = s[n] ;//range[2*n];- args[i].dat->offset[n];
      end[i*ndim+n]   = e[n] ;//range[2*n+1];- args[i].dat->offset[n];
    }
  }

  #ifdef OPS_DEBUG
  ops_register_args(args, name);
  #endif

  for (int i = 0; i<13;i++) {
    if(args[i].stencil!=NULL) {
      offs[i][0] = args[i].stencil->stride[0]*1;  //unit step in x dimension
      for(int n=1; n<ndim; n++) {
        offs[i][n] = off(ndim, n, &start[i*ndim], &end[i*ndim],
                         args[i].dat->block_size, args[i].stencil->stride);
      }
    }
  }

  //set up initial pointers
  for (int i = 0; i < 13; i++) {
    if (args[i].argtype == OPS_ARG_DAT) {
      p_a[i] = (char *)args[i].data //base of 2D array
      + address(ndim, args[i].dat->size, &start[i*ndim],
        args[i].dat->block_size, args[i].stencil->stride, args[i].dat->offset);
    }
    else if (args[i].argtype == OPS_ARG_GBL)
      p_a[i] = (char *)args[i].data;
  }

  free(start);free(end);

  int total_range = 1;
  for (int n=0; n<ndim; n++) {
    count[n] = e[n]-s[n];  // number in each dimension
    total_range *= count[n];
  }
  count[dim-1]++;     // extra in last to ensure correct termination

  if (args[0].argtype == OPS_ARG_DAT)  xdim0 = args[0].dat->block_size[0];
  if (args[1].argtype == OPS_ARG_DAT)  xdim1 = args[1].dat->block_size[0];
  if (args[2].argtype == OPS_ARG_DAT)  xdim2 = args[2].dat->block_size[0];
  if (args[3].argtype == OPS_ARG_DAT)  xdim3 = args[3].dat->block_size[0];
  if (args[4].argtype == OPS_ARG_DAT)  xdim4 = args[4].dat->block_size[0];
  if (args[5].argtype == OPS_ARG_DAT)  xdim5 = args[5].dat->block_size[0];
  if (args[6].argtype == OPS_ARG_DAT)  xdim6 = args[6].dat->block_size[0];
  if (args[7].argtype == OPS_ARG_DAT)  xdim7 = args[7].dat->block_size[0];
  if (args[8].argtype == OPS_ARG_DAT)  xdim8 = args[8].dat->block_size[0];
  if (args[9].argtype == OPS_ARG_DAT)  xdim9 = args[9].dat->block_size[0];
  if (args[10].argtype == OPS_ARG_DAT)  xdim10 = args[10].dat->block_size[0];
  if (args[11].argtype == OPS_ARG_DAT)  xdim11 = args[11].dat->block_size[0];
  if (args[12].argtype == OPS_ARG_DAT)  xdim12 = args[12].dat->block_size[0];

  for (int nt=0; nt<total_range; nt++) {
    // call kernel function, passing in pointers to data

    kernel(  (T0 *)p_a[0], (T1 *)p_a[1], (T2 *)p_a[2], (T3 *)p_a[3],
           (T4 *)p_a[4], (T5 *)p_a[5], (T6 *)p_a[6], (T7 *)p_a[7],
           (T8 *)p_a[8], (T9 *)p_a[9], (T10 *)p_a[10], (T11 *)p_a[11],
           (T12 *)p_a[12] );

    count[0]--;   // decrement counter
    int m = 0;    // max dimension with changed index
    while (count[m]==0) {
      count[m] =  e[m]-s[m];// reset counter
      m++;                        // next dimension
      count[m]--;                 // decrement counter
    }

    int a = 0;
    // shift pointers to data
    for (int i=0; i<13; i++) {
      if (args[i].argtype == OPS_ARG_DAT)
        p_a[i] = p_a[i] + (args[i].dat->size * offs[i][m]);
    }
  }

}

//
//ops_par_loop routine for 14 arguments
//
template <class T0,class T1,class T2,class T3,
class T4,class T5,class T6,class T7,
class T8,class T9,class T10,class T11,
class T12,class T13>
void ops_par_loop_mpi(void (*kernel)(T0*, T1*, T2*, T3*,
                           T4*, T5*, T6*, T7*,
                           T8*, T9*, T10*, T11*,
                           T12*, T13*),
     char const * name, ops_block block, int dim, int *range,
     ops_arg arg0, ops_arg arg1, ops_arg arg2, ops_arg arg3,
     ops_arg arg4, ops_arg arg5, ops_arg arg6, ops_arg arg7,
     ops_arg arg8, ops_arg arg9, ops_arg arg10, ops_arg arg11,
     ops_arg arg12, ops_arg arg13) {

  char *p_a[14];
  int  offs[14][2];

  int  count[dim];
  ops_arg args[14] = { arg0, arg1, arg2, arg3,
                     arg4, arg5, arg6, arg7,
                     arg8, arg9, arg10, arg11,
                     arg12, arg13};

  sub_block_list sb = OPS_sub_block_list[block->index];


  //compute localy allocated range for the sub-block
  int ndim = sb->ndim;
  int* start = (int*) xmalloc(sizeof(int)*ndim*14);
  int* end = (int*) xmalloc(sizeof(int)*ndim*14);

  int s[ndim];
  int e[ndim];

  for (int n=0; n<ndim; n++) {
    s[n] = sb->istart[n];e[n] = sb->iend[n]+1;
    if(s[n] >= range[2*n]) s[n] = 0;
    else s[n] = range[2*n] - s[n];
    if(e[n] >= range[2*n+1]) e[n] = range[2*n+1] - sb->istart[n];
    else e[n] = sb->sizes[n];
  }
  for(int i = 0; i<14; i++) {
    for(int n=0; n<ndim; n++) {
      start[i*ndim+n] = s[n] ;//range[2*n];- args[i].dat->offset[n];
      end[i*ndim+n]   = e[n] ;//range[2*n+1];- args[i].dat->offset[n];
    }
  }

  #ifdef OPS_DEBUG
  ops_register_args(args, name);
  #endif

  for (int i = 0; i<14;i++) {
    if(args[i].stencil!=NULL) {
      offs[i][0] = args[i].stencil->stride[0]*1;  //unit step in x dimension
      for(int n=1; n<ndim; n++) {
        offs[i][n] = off(ndim, n, &start[i*ndim], &end[i*ndim],
                         args[i].dat->block_size, args[i].stencil->stride);
      }
    }
  }

  //set up initial pointers
  for (int i = 0; i < 14; i++) {
    if (args[i].argtype == OPS_ARG_DAT) {
      p_a[i] = (char *)args[i].data //base of 2D array
      + address(ndim, args[i].dat->size, &start[i*ndim],
        args[i].dat->block_size, args[i].stencil->stride, args[i].dat->offset);
    }
    else if (args[i].argtype == OPS_ARG_GBL)
      p_a[i] = (char *)args[i].data;
  }

  free(start);free(end);

  int total_range = 1;
  for (int n=0; n<ndim; n++) {
    count[n] = e[n]-s[n];  // number in each dimension
    total_range *= count[n];
  }
  count[dim-1]++;     // extra in last to ensure correct termination

  if (args[0].argtype == OPS_ARG_DAT)  xdim0 = args[0].dat->block_size[0];
  if (args[1].argtype == OPS_ARG_DAT)  xdim1 = args[1].dat->block_size[0];
  if (args[2].argtype == OPS_ARG_DAT)  xdim2 = args[2].dat->block_size[0];
  if (args[3].argtype == OPS_ARG_DAT)  xdim3 = args[3].dat->block_size[0];
  if (args[4].argtype == OPS_ARG_DAT)  xdim4 = args[4].dat->block_size[0];
  if (args[5].argtype == OPS_ARG_DAT)  xdim5 = args[5].dat->block_size[0];
  if (args[6].argtype == OPS_ARG_DAT)  xdim6 = args[6].dat->block_size[0];
  if (args[7].argtype == OPS_ARG_DAT)  xdim7 = args[7].dat->block_size[0];
  if (args[8].argtype == OPS_ARG_DAT)  xdim8 = args[8].dat->block_size[0];
  if (args[9].argtype == OPS_ARG_DAT)  xdim9 = args[9].dat->block_size[0];
  if (args[10].argtype == OPS_ARG_DAT)  xdim10 = args[10].dat->block_size[0];
  if (args[11].argtype == OPS_ARG_DAT)  xdim11 = args[11].dat->block_size[0];
  if (args[12].argtype == OPS_ARG_DAT)  xdim12 = args[12].dat->block_size[0];
  if (args[13].argtype == OPS_ARG_DAT)  xdim13 = args[13].dat->block_size[0];

  for (int nt=0; nt<total_range; nt++) {
    // call kernel function, passing in pointers to data

    kernel(  (T0 *)p_a[0], (T1 *)p_a[1], (T2 *)p_a[2], (T3 *)p_a[3],
           (T4 *)p_a[4], (T5 *)p_a[5], (T6 *)p_a[6], (T7 *)p_a[7],
           (T8 *)p_a[8], (T9 *)p_a[9], (T10 *)p_a[10], (T11 *)p_a[11],
           (T12 *)p_a[12], (T13 *)p_a[13] );

    count[0]--;   // decrement counter
    int m = 0;    // max dimension with changed index
    while (count[m]==0) {
      count[m] =  e[m]-s[m];// reset counter
      m++;                        // next dimension
      count[m]--;                 // decrement counter
    }

    int a = 0;
    // shift pointers to data
    for (int i=0; i<14; i++) {
      if (args[i].argtype == OPS_ARG_DAT)
        p_a[i] = p_a[i] + (args[i].dat->size * offs[i][m]);
    }
  }

}

//
//ops_par_loop routine for 15 arguments
//
template <class T0,class T1,class T2,class T3,
class T4,class T5,class T6,class T7,
class T8,class T9,class T10,class T11,
class T12,class T13,class T14>
void ops_par_loop_mpi(void (*kernel)(T0*, T1*, T2*, T3*,
                           T4*, T5*, T6*, T7*,
                           T8*, T9*, T10*, T11*,
                           T12*, T13*, T14*),
     char const * name, ops_block block, int dim, int *range,
     ops_arg arg0, ops_arg arg1, ops_arg arg2, ops_arg arg3,
     ops_arg arg4, ops_arg arg5, ops_arg arg6, ops_arg arg7,
     ops_arg arg8, ops_arg arg9, ops_arg arg10, ops_arg arg11,
     ops_arg arg12, ops_arg arg13, ops_arg arg14) {

  char *p_a[15];
  int  offs[15][2];

  int  count[dim];
  ops_arg args[15] = { arg0, arg1, arg2, arg3,
                     arg4, arg5, arg6, arg7,
                     arg8, arg9, arg10, arg11,
                     arg12, arg13, arg14};

  sub_block_list sb = OPS_sub_block_list[block->index];


  //compute localy allocated range for the sub-block
  int ndim = sb->ndim;
  int* start = (int*) xmalloc(sizeof(int)*ndim*15);
  int* end = (int*) xmalloc(sizeof(int)*ndim*15);

  int s[ndim];
  int e[ndim];

  for (int n=0; n<ndim; n++) {
    s[n] = sb->istart[n];e[n] = sb->iend[n]+1;
    if(s[n] >= range[2*n]) s[n] = 0;
    else s[n] = range[2*n] - s[n];
    if(e[n] >= range[2*n+1]) e[n] = range[2*n+1] - sb->istart[n];
    else e[n] = sb->sizes[n];
  }
  for(int i = 0; i<15; i++) {
    for(int n=0; n<ndim; n++) {
      start[i*ndim+n] = s[n] ;//range[2*n];- args[i].dat->offset[n];
      end[i*ndim+n]   = e[n] ;//range[2*n+1];- args[i].dat->offset[n];
    }
  }

  #ifdef OPS_DEBUG
  ops_register_args(args, name);
  #endif

  for (int i = 0; i<15;i++) {
    if(args[i].stencil!=NULL) {
      offs[i][0] = args[i].stencil->stride[0]*1;  //unit step in x dimension
      for(int n=1; n<ndim; n++) {
        offs[i][n] = off(ndim, n, &start[i*ndim], &end[i*ndim],
                         args[i].dat->block_size, args[i].stencil->stride);
      }
    }
  }

  //set up initial pointers
  for (int i = 0; i < 15; i++) {
    if (args[i].argtype == OPS_ARG_DAT) {
      p_a[i] = (char *)args[i].data //base of 2D array
      + address(ndim, args[i].dat->size, &start[i*ndim],
        args[i].dat->block_size, args[i].stencil->stride, args[i].dat->offset);
    }
    else if (args[i].argtype == OPS_ARG_GBL)
      p_a[i] = (char *)args[i].data;
  }

  free(start);free(end);

  int total_range = 1;
  for (int n=0; n<ndim; n++) {
    count[n] = e[n]-s[n];  // number in each dimension
    total_range *= count[n];
  }
  count[dim-1]++;     // extra in last to ensure correct termination

  if (args[0].argtype == OPS_ARG_DAT)  xdim0 = args[0].dat->block_size[0];
  if (args[1].argtype == OPS_ARG_DAT)  xdim1 = args[1].dat->block_size[0];
  if (args[2].argtype == OPS_ARG_DAT)  xdim2 = args[2].dat->block_size[0];
  if (args[3].argtype == OPS_ARG_DAT)  xdim3 = args[3].dat->block_size[0];
  if (args[4].argtype == OPS_ARG_DAT)  xdim4 = args[4].dat->block_size[0];
  if (args[5].argtype == OPS_ARG_DAT)  xdim5 = args[5].dat->block_size[0];
  if (args[6].argtype == OPS_ARG_DAT)  xdim6 = args[6].dat->block_size[0];
  if (args[7].argtype == OPS_ARG_DAT)  xdim7 = args[7].dat->block_size[0];
  if (args[8].argtype == OPS_ARG_DAT)  xdim8 = args[8].dat->block_size[0];
  if (args[9].argtype == OPS_ARG_DAT)  xdim9 = args[9].dat->block_size[0];
  if (args[10].argtype == OPS_ARG_DAT)  xdim10 = args[10].dat->block_size[0];
  if (args[11].argtype == OPS_ARG_DAT)  xdim11 = args[11].dat->block_size[0];
  if (args[12].argtype == OPS_ARG_DAT)  xdim12 = args[12].dat->block_size[0];
  if (args[13].argtype == OPS_ARG_DAT)  xdim13 = args[13].dat->block_size[0];
  if (args[14].argtype == OPS_ARG_DAT)  xdim14 = args[14].dat->block_size[0];

  for (int nt=0; nt<total_range; nt++) {
    // call kernel function, passing in pointers to data

    kernel(  (T0 *)p_a[0], (T1 *)p_a[1], (T2 *)p_a[2], (T3 *)p_a[3],
           (T4 *)p_a[4], (T5 *)p_a[5], (T6 *)p_a[6], (T7 *)p_a[7],
           (T8 *)p_a[8], (T9 *)p_a[9], (T10 *)p_a[10], (T11 *)p_a[11],
           (T12 *)p_a[12], (T13 *)p_a[13], (T14 *)p_a[14] );

    count[0]--;   // decrement counter
    int m = 0;    // max dimension with changed index
    while (count[m]==0) {
      count[m] =  e[m]-s[m];// reset counter
      m++;                        // next dimension
      count[m]--;                 // decrement counter
    }

    int a = 0;
    // shift pointers to data
    for (int i=0; i<15; i++) {
      if (args[i].argtype == OPS_ARG_DAT)
        p_a[i] = p_a[i] + (args[i].dat->size * offs[i][m]);
    }
  }

}

//
//ops_par_loop routine for 16 arguments
//
template <class T0,class T1,class T2,class T3,
class T4,class T5,class T6,class T7,
class T8,class T9,class T10,class T11,
class T12,class T13,class T14,class T15>
void ops_par_loop_mpi(void (*kernel)(T0*, T1*, T2*, T3*,
                           T4*, T5*, T6*, T7*,
                           T8*, T9*, T10*, T11*,
                           T12*, T13*, T14*, T15*),
     char const * name, ops_block block, int dim, int *range,
     ops_arg arg0, ops_arg arg1, ops_arg arg2, ops_arg arg3,
     ops_arg arg4, ops_arg arg5, ops_arg arg6, ops_arg arg7,
     ops_arg arg8, ops_arg arg9, ops_arg arg10, ops_arg arg11,
     ops_arg arg12, ops_arg arg13, ops_arg arg14, ops_arg arg15) {

  char *p_a[16];
  int  offs[16][2];

  int  count[dim];
  ops_arg args[16] = { arg0, arg1, arg2, arg3,
                     arg4, arg5, arg6, arg7,
                     arg8, arg9, arg10, arg11,
                     arg12, arg13, arg14, arg15};

  sub_block_list sb = OPS_sub_block_list[block->index];


  //compute localy allocated range for the sub-block
  int ndim = sb->ndim;
  int* start = (int*) xmalloc(sizeof(int)*ndim*16);
  int* end = (int*) xmalloc(sizeof(int)*ndim*16);

  int s[ndim];
  int e[ndim];

  for (int n=0; n<ndim; n++) {
    s[n] = sb->istart[n];e[n] = sb->iend[n]+1;
    if(s[n] >= range[2*n]) s[n] = 0;
    else s[n] = range[2*n] - s[n];
    if(e[n] >= range[2*n+1]) e[n] = range[2*n+1] - sb->istart[n];
    else e[n] = sb->sizes[n];
  }
  for(int i = 0; i<16; i++) {
    for(int n=0; n<ndim; n++) {
      start[i*ndim+n] = s[n] ;//range[2*n];- args[i].dat->offset[n];
      end[i*ndim+n]   = e[n] ;//range[2*n+1];- args[i].dat->offset[n];
    }
  }

  #ifdef OPS_DEBUG
  ops_register_args(args, name);
  #endif

  for (int i = 0; i<16;i++) {
    if(args[i].stencil!=NULL) {
      offs[i][0] = args[i].stencil->stride[0]*1;  //unit step in x dimension
      for(int n=1; n<ndim; n++) {
        offs[i][n] = off(ndim, n, &start[i*ndim], &end[i*ndim],
                         args[i].dat->block_size, args[i].stencil->stride);
      }
    }
  }

  //set up initial pointers
  for (int i = 0; i < 16; i++) {
    if (args[i].argtype == OPS_ARG_DAT) {
      p_a[i] = (char *)args[i].data //base of 2D array
      + address(ndim, args[i].dat->size, &start[i*ndim],
        args[i].dat->block_size, args[i].stencil->stride, args[i].dat->offset);
    }
    else if (args[i].argtype == OPS_ARG_GBL)
      p_a[i] = (char *)args[i].data;
  }

  free(start);free(end);

  int total_range = 1;
  for (int n=0; n<ndim; n++) {
    count[n] = e[n]-s[n];  // number in each dimension
    total_range *= count[n];
  }
  count[dim-1]++;     // extra in last to ensure correct termination

  if (args[0].argtype == OPS_ARG_DAT)  xdim0 = args[0].dat->block_size[0];
  if (args[1].argtype == OPS_ARG_DAT)  xdim1 = args[1].dat->block_size[0];
  if (args[2].argtype == OPS_ARG_DAT)  xdim2 = args[2].dat->block_size[0];
  if (args[3].argtype == OPS_ARG_DAT)  xdim3 = args[3].dat->block_size[0];
  if (args[4].argtype == OPS_ARG_DAT)  xdim4 = args[4].dat->block_size[0];
  if (args[5].argtype == OPS_ARG_DAT)  xdim5 = args[5].dat->block_size[0];
  if (args[6].argtype == OPS_ARG_DAT)  xdim6 = args[6].dat->block_size[0];
  if (args[7].argtype == OPS_ARG_DAT)  xdim7 = args[7].dat->block_size[0];
  if (args[8].argtype == OPS_ARG_DAT)  xdim8 = args[8].dat->block_size[0];
  if (args[9].argtype == OPS_ARG_DAT)  xdim9 = args[9].dat->block_size[0];
  if (args[10].argtype == OPS_ARG_DAT)  xdim10 = args[10].dat->block_size[0];
  if (args[11].argtype == OPS_ARG_DAT)  xdim11 = args[11].dat->block_size[0];
  if (args[12].argtype == OPS_ARG_DAT)  xdim12 = args[12].dat->block_size[0];
  if (args[13].argtype == OPS_ARG_DAT)  xdim13 = args[13].dat->block_size[0];
  if (args[14].argtype == OPS_ARG_DAT)  xdim14 = args[14].dat->block_size[0];
  if (args[15].argtype == OPS_ARG_DAT)  xdim15 = args[15].dat->block_size[0];

  for (int nt=0; nt<total_range; nt++) {
    // call kernel function, passing in pointers to data

    kernel(  (T0 *)p_a[0], (T1 *)p_a[1], (T2 *)p_a[2], (T3 *)p_a[3],
           (T4 *)p_a[4], (T5 *)p_a[5], (T6 *)p_a[6], (T7 *)p_a[7],
           (T8 *)p_a[8], (T9 *)p_a[9], (T10 *)p_a[10], (T11 *)p_a[11],
           (T12 *)p_a[12], (T13 *)p_a[13], (T14 *)p_a[14], (T15 *)p_a[15] );

    count[0]--;   // decrement counter
    int m = 0;    // max dimension with changed index
    while (count[m]==0) {
      count[m] =  e[m]-s[m];// reset counter
      m++;                        // next dimension
      count[m]--;                 // decrement counter
    }

    int a = 0;
    // shift pointers to data
    for (int i=0; i<16; i++) {
      if (args[i].argtype == OPS_ARG_DAT)
        p_a[i] = p_a[i] + (args[i].dat->size * offs[i][m]);
    }
  }

}

//
//ops_par_loop routine for 17 arguments
//
template <class T0,class T1,class T2,class T3,
class T4,class T5,class T6,class T7,
class T8,class T9,class T10,class T11,
class T12,class T13,class T14,class T15,
class T16>
void ops_par_loop_mpi(void (*kernel)(T0*, T1*, T2*, T3*,
                           T4*, T5*, T6*, T7*,
                           T8*, T9*, T10*, T11*,
                           T12*, T13*, T14*, T15*,
                           T16*),
     char const * name, ops_block block, int dim, int *range,
     ops_arg arg0, ops_arg arg1, ops_arg arg2, ops_arg arg3,
     ops_arg arg4, ops_arg arg5, ops_arg arg6, ops_arg arg7,
     ops_arg arg8, ops_arg arg9, ops_arg arg10, ops_arg arg11,
     ops_arg arg12, ops_arg arg13, ops_arg arg14, ops_arg arg15,
     ops_arg arg16) {

  char *p_a[17];
  int  offs[17][2];

  int  count[dim];
  ops_arg args[17] = { arg0, arg1, arg2, arg3,
                     arg4, arg5, arg6, arg7,
                     arg8, arg9, arg10, arg11,
                     arg12, arg13, arg14, arg15,
                     arg16};

  sub_block_list sb = OPS_sub_block_list[block->index];


  //compute localy allocated range for the sub-block
  int ndim = sb->ndim;
  int* start = (int*) xmalloc(sizeof(int)*ndim*17);
  int* end = (int*) xmalloc(sizeof(int)*ndim*17);

  int s[ndim];
  int e[ndim];

  for (int n=0; n<ndim; n++) {
    s[n] = sb->istart[n];e[n] = sb->iend[n]+1;
    if(s[n] >= range[2*n]) s[n] = 0;
    else s[n] = range[2*n] - s[n];
    if(e[n] >= range[2*n+1]) e[n] = range[2*n+1] - sb->istart[n];
    else e[n] = sb->sizes[n];
  }
  for(int i = 0; i<17; i++) {
    for(int n=0; n<ndim; n++) {
      start[i*ndim+n] = s[n] ;//range[2*n];- args[i].dat->offset[n];
      end[i*ndim+n]   = e[n] ;//range[2*n+1];- args[i].dat->offset[n];
    }
  }

  #ifdef OPS_DEBUG
  ops_register_args(args, name);
  #endif

  for (int i = 0; i<17;i++) {
    if(args[i].stencil!=NULL) {
      offs[i][0] = args[i].stencil->stride[0]*1;  //unit step in x dimension
      for(int n=1; n<ndim; n++) {
        offs[i][n] = off(ndim, n, &start[i*ndim], &end[i*ndim],
                         args[i].dat->block_size, args[i].stencil->stride);
      }
    }
  }

  //set up initial pointers
  for (int i = 0; i < 17; i++) {
    if (args[i].argtype == OPS_ARG_DAT) {
      p_a[i] = (char *)args[i].data //base of 2D array
      + address(ndim, args[i].dat->size, &start[i*ndim],
        args[i].dat->block_size, args[i].stencil->stride, args[i].dat->offset);
    }
    else if (args[i].argtype == OPS_ARG_GBL)
      p_a[i] = (char *)args[i].data;
  }

  free(start);free(end);

  int total_range = 1;
  for (int n=0; n<ndim; n++) {
    count[n] = e[n]-s[n];  // number in each dimension
    total_range *= count[n];
  }
  count[dim-1]++;     // extra in last to ensure correct termination

  if (args[0].argtype == OPS_ARG_DAT)  xdim0 = args[0].dat->block_size[0];
  if (args[1].argtype == OPS_ARG_DAT)  xdim1 = args[1].dat->block_size[0];
  if (args[2].argtype == OPS_ARG_DAT)  xdim2 = args[2].dat->block_size[0];
  if (args[3].argtype == OPS_ARG_DAT)  xdim3 = args[3].dat->block_size[0];
  if (args[4].argtype == OPS_ARG_DAT)  xdim4 = args[4].dat->block_size[0];
  if (args[5].argtype == OPS_ARG_DAT)  xdim5 = args[5].dat->block_size[0];
  if (args[6].argtype == OPS_ARG_DAT)  xdim6 = args[6].dat->block_size[0];
  if (args[7].argtype == OPS_ARG_DAT)  xdim7 = args[7].dat->block_size[0];
  if (args[8].argtype == OPS_ARG_DAT)  xdim8 = args[8].dat->block_size[0];
  if (args[9].argtype == OPS_ARG_DAT)  xdim9 = args[9].dat->block_size[0];
  if (args[10].argtype == OPS_ARG_DAT)  xdim10 = args[10].dat->block_size[0];
  if (args[11].argtype == OPS_ARG_DAT)  xdim11 = args[11].dat->block_size[0];
  if (args[12].argtype == OPS_ARG_DAT)  xdim12 = args[12].dat->block_size[0];
  if (args[13].argtype == OPS_ARG_DAT)  xdim13 = args[13].dat->block_size[0];
  if (args[14].argtype == OPS_ARG_DAT)  xdim14 = args[14].dat->block_size[0];
  if (args[15].argtype == OPS_ARG_DAT)  xdim15 = args[15].dat->block_size[0];
  if (args[16].argtype == OPS_ARG_DAT)  xdim16 = args[16].dat->block_size[0];

  for (int nt=0; nt<total_range; nt++) {
    // call kernel function, passing in pointers to data

    kernel(  (T0 *)p_a[0], (T1 *)p_a[1], (T2 *)p_a[2], (T3 *)p_a[3],
           (T4 *)p_a[4], (T5 *)p_a[5], (T6 *)p_a[6], (T7 *)p_a[7],
           (T8 *)p_a[8], (T9 *)p_a[9], (T10 *)p_a[10], (T11 *)p_a[11],
           (T12 *)p_a[12], (T13 *)p_a[13], (T14 *)p_a[14], (T15 *)p_a[15],
           (T16 *)p_a[16] );

    count[0]--;   // decrement counter
    int m = 0;    // max dimension with changed index
    while (count[m]==0) {
      count[m] =  e[m]-s[m];// reset counter
      m++;                        // next dimension
      count[m]--;                 // decrement counter
    }

    int a = 0;
    // shift pointers to data
    for (int i=0; i<17; i++) {
      if (args[i].argtype == OPS_ARG_DAT)
        p_a[i] = p_a[i] + (args[i].dat->size * offs[i][m]);
    }
  }

}

//
//ops_par_loop routine for 18 arguments
//
template <class T0,class T1,class T2,class T3,
class T4,class T5,class T6,class T7,
class T8,class T9,class T10,class T11,
class T12,class T13,class T14,class T15,
class T16,class T17>
void ops_par_loop_mpi(void (*kernel)(T0*, T1*, T2*, T3*,
                           T4*, T5*, T6*, T7*,
                           T8*, T9*, T10*, T11*,
                           T12*, T13*, T14*, T15*,
                           T16*, T17*),
     char const * name, ops_block block, int dim, int *range,
     ops_arg arg0, ops_arg arg1, ops_arg arg2, ops_arg arg3,
     ops_arg arg4, ops_arg arg5, ops_arg arg6, ops_arg arg7,
     ops_arg arg8, ops_arg arg9, ops_arg arg10, ops_arg arg11,
     ops_arg arg12, ops_arg arg13, ops_arg arg14, ops_arg arg15,
     ops_arg arg16, ops_arg arg17) {

  char *p_a[18];
  int  offs[18][2];

  int  count[dim];
  ops_arg args[18] = { arg0, arg1, arg2, arg3,
                     arg4, arg5, arg6, arg7,
                     arg8, arg9, arg10, arg11,
                     arg12, arg13, arg14, arg15,
                     arg16, arg17};

  sub_block_list sb = OPS_sub_block_list[block->index];


  //compute localy allocated range for the sub-block
  int ndim = sb->ndim;
  int* start = (int*) xmalloc(sizeof(int)*ndim*18);
  int* end = (int*) xmalloc(sizeof(int)*ndim*18);

  int s[ndim];
  int e[ndim];

  for (int n=0; n<ndim; n++) {
    s[n] = sb->istart[n];e[n] = sb->iend[n]+1;
    if(s[n] >= range[2*n]) s[n] = 0;
    else s[n] = range[2*n] - s[n];
    if(e[n] >= range[2*n+1]) e[n] = range[2*n+1] - sb->istart[n];
    else e[n] = sb->sizes[n];
  }
  for(int i = 0; i<18; i++) {
    for(int n=0; n<ndim; n++) {
      start[i*ndim+n] = s[n] ;//range[2*n];- args[i].dat->offset[n];
      end[i*ndim+n]   = e[n] ;//range[2*n+1];- args[i].dat->offset[n];
    }
  }

  #ifdef OPS_DEBUG
  ops_register_args(args, name);
  #endif

  for (int i = 0; i<18;i++) {
    if(args[i].stencil!=NULL) {
      offs[i][0] = args[i].stencil->stride[0]*1;  //unit step in x dimension
      for(int n=1; n<ndim; n++) {
        offs[i][n] = off(ndim, n, &start[i*ndim], &end[i*ndim],
                         args[i].dat->block_size, args[i].stencil->stride);
      }
    }
  }

  //set up initial pointers
  for (int i = 0; i < 18; i++) {
    if (args[i].argtype == OPS_ARG_DAT) {
      p_a[i] = (char *)args[i].data //base of 2D array
      + address(ndim, args[i].dat->size, &start[i*ndim],
        args[i].dat->block_size, args[i].stencil->stride, args[i].dat->offset);
    }
    else if (args[i].argtype == OPS_ARG_GBL)
      p_a[i] = (char *)args[i].data;
  }

  free(start);free(end);

  int total_range = 1;
  for (int n=0; n<ndim; n++) {
    count[n] = e[n]-s[n];  // number in each dimension
    total_range *= count[n];
  }
  count[dim-1]++;     // extra in last to ensure correct termination

  if (args[0].argtype == OPS_ARG_DAT)  xdim0 = args[0].dat->block_size[0];
  if (args[1].argtype == OPS_ARG_DAT)  xdim1 = args[1].dat->block_size[0];
  if (args[2].argtype == OPS_ARG_DAT)  xdim2 = args[2].dat->block_size[0];
  if (args[3].argtype == OPS_ARG_DAT)  xdim3 = args[3].dat->block_size[0];
  if (args[4].argtype == OPS_ARG_DAT)  xdim4 = args[4].dat->block_size[0];
  if (args[5].argtype == OPS_ARG_DAT)  xdim5 = args[5].dat->block_size[0];
  if (args[6].argtype == OPS_ARG_DAT)  xdim6 = args[6].dat->block_size[0];
  if (args[7].argtype == OPS_ARG_DAT)  xdim7 = args[7].dat->block_size[0];
  if (args[8].argtype == OPS_ARG_DAT)  xdim8 = args[8].dat->block_size[0];
  if (args[9].argtype == OPS_ARG_DAT)  xdim9 = args[9].dat->block_size[0];
  if (args[10].argtype == OPS_ARG_DAT)  xdim10 = args[10].dat->block_size[0];
  if (args[11].argtype == OPS_ARG_DAT)  xdim11 = args[11].dat->block_size[0];
  if (args[12].argtype == OPS_ARG_DAT)  xdim12 = args[12].dat->block_size[0];
  if (args[13].argtype == OPS_ARG_DAT)  xdim13 = args[13].dat->block_size[0];
  if (args[14].argtype == OPS_ARG_DAT)  xdim14 = args[14].dat->block_size[0];
  if (args[15].argtype == OPS_ARG_DAT)  xdim15 = args[15].dat->block_size[0];
  if (args[16].argtype == OPS_ARG_DAT)  xdim16 = args[16].dat->block_size[0];
  if (args[17].argtype == OPS_ARG_DAT)  xdim17 = args[17].dat->block_size[0];

  for (int nt=0; nt<total_range; nt++) {
    // call kernel function, passing in pointers to data

    kernel(  (T0 *)p_a[0], (T1 *)p_a[1], (T2 *)p_a[2], (T3 *)p_a[3],
           (T4 *)p_a[4], (T5 *)p_a[5], (T6 *)p_a[6], (T7 *)p_a[7],
           (T8 *)p_a[8], (T9 *)p_a[9], (T10 *)p_a[10], (T11 *)p_a[11],
           (T12 *)p_a[12], (T13 *)p_a[13], (T14 *)p_a[14], (T15 *)p_a[15],
           (T16 *)p_a[16], (T17 *)p_a[17] );

    count[0]--;   // decrement counter
    int m = 0;    // max dimension with changed index
    while (count[m]==0) {
      count[m] =  e[m]-s[m];// reset counter
      m++;                        // next dimension
      count[m]--;                 // decrement counter
    }

    int a = 0;
    // shift pointers to data
    for (int i=0; i<18; i++) {
      if (args[i].argtype == OPS_ARG_DAT)
        p_a[i] = p_a[i] + (args[i].dat->size * offs[i][m]);
    }
  }

}