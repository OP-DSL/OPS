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

/** @brief headder file declaring the functions for the ops sequential backend
  * @author Gihan Mudalige, Istvan Reguly
  * @details Declares the OPS API calls for the sequential backend
  */

#include "ops_lib_cpp.h"

inline void ops_arg_set(int n_x, ops_arg arg, char **p_arg){
  if (arg.stencil!=NULL) {
    for (int i = 0; i < arg.stencil->points; i++){
      p_arg[i] =
         arg.data + //base of 1D array
         arg.dat->size * ( //multiply by the number of bytes per element
         (n_x - arg.dat->offset[0]) * //calculate the offset from index 0
         arg.stencil->stride[0]  + // jump in strides ??
         arg.stencil->stencil[i * arg.stencil->dims + 0] //get the value at the ith stencil point
         );
    }
  } else {
    *p_arg = arg.data;
  }
}

inline void ops_arg_set(int n_x,
                        int n_y, ops_arg arg, char **p_arg){
  if (arg.stencil!=NULL) {
    for (int i = 0; i < arg.stencil->points; i++)

      p_arg[i] =
        arg.data + //base of 2D array
        //y dimension -- get to the correct y line
        arg.dat->size * arg.dat->block->size[0] * ( //multiply by the number of
                                                    //bytes per element and xdim block size
        (n_y - arg.dat->offset[1]) * // calculate the offset from index 0 for y dim
        arg.stencil->stride[1] + // jump in strides in y dim ??
        arg.stencil->stencil[i*arg.stencil->dims + 1]) //get the value at the ith
                                                       //stencil point "+ 1" is the y dim
        +
        //x dimension - get to the correct x point on the y line
        arg.dat->size * ( //multiply by the number of bytes per element
        (n_x - arg.dat->offset[0]) * //calculate the offset from index 0 for x dim
        arg.stencil->stride[0] + // jump in strides in x dim ??
        arg.stencil->stencil[i*arg.stencil->dims + 0] //get the value at the ith
                                                      //stencil point "+ 0" is the x dim
      );
  } else {
    *p_arg = arg.data;
  }
}

inline void ops_arg_set(int n_x,
                        int n_y,
                        int n_z, ops_arg arg, char **p_arg){
  if (arg.stencil!=NULL) {
    for (int i = 0; i < arg.stencil->points; i++)
      p_arg[i] =
      arg.data +
      //z dimension - get to the correct z plane
      arg.dat->size * arg.dat->block->size[1] * arg.dat->block->size[0] * (
      (n_z - arg.dat->offset[2]) *
      arg.stencil->stride[2] +
      arg.stencil->stencil[i*arg.stencil->dims + 2])
      +
      //y dimension -- get to the correct y line on the z plane
      arg.dat->size * arg.dat->block->size[0] * (
      (n_y - arg.dat->offset[1]) *
      arg.stencil->stride[1] +
      arg.stencil->stencil[i*arg.stencil->dims + 1])
      +
      //x dimension - get to the correct x point on the y line
      arg.dat->size * (
      (n_x - arg.dat->offset[0]) *
      arg.stencil->stride[0] +
      arg.stencil->stencil[i*arg.stencil->dims + 0]
      );
  } else {
    *p_arg = arg.data;
  }
}



inline void ops_args_set(int iter_x,
                         int nargs, ops_arg *args, char ***p_a){
  for (int n=0; n<nargs; n++) {
    ops_arg_set(iter_x, args[n], p_a[n]);
  }
}

inline void ops_args_set(int iter_x,
                         int iter_y,
                         int nargs, ops_arg *args, char ***p_a){
  for (int n=0; n<nargs; n++) {
    ops_arg_set(iter_x, iter_y, args[n], p_a[n]);
  }
}

inline void ops_args_set(int iter_x,
                         int iter_y,
                         int iter_z, int nargs, ops_arg *args, char ***p_a){
  for (int n=0; n<nargs; n++) {
    ops_arg_set(iter_x, iter_y, iter_z, args[n], p_a[n]);
  }
}


template < class T0 >
void ops_par_loop(void (*kernel)( T0* ),
  char const * name, int dim, int *range,
  ops_arg arg0 ) {

  char** p_a[1];
  ops_arg args[1] = {arg0};

  // consistency checks
  // ops_args_check(1,args,name);

  for (int i = 0; i < 1; i++) {
    if (args[i].argtype == OPS_ARG_DAT)
      p_a[i] = (char **)malloc(args[i].stencil->points * sizeof(char *));
  }

  // loop over set elements

  if (dim == 1) {
    for (int n_x = range[0]; n_x < range[1]; n_x++) {
      ops_args_set(n_x, 1,args,p_a);
      // call kernel function, passing in pointers to data
      kernel( (T0 *)p_a[0] );
    }
  }
  else if (dim == 2) {
    for (int n_y = range[2]; n_y < range[3]; n_y++) {
      for (int n_x = range[0]; n_x < range[1]; n_x++) {
        ops_args_set(n_x, n_y,1,args,p_a);
        // call kernel function, passing in pointers to data
        kernel( (T0 *)p_a[0] );
      }
    }
  }
  else if (dim == 3) {
    for (int n_z = range[4]; n_z < range[5]; n_z++) {
      for (int n_y = range[2]; n_y < range[3]; n_y++) {
        for (int n_x = range[0]; n_x < range[1]; n_x++) {
          ops_args_set(n_x, n_y, n_z,1,args,p_a);
          // call kernel function, passing in pointers to data
          kernel( (T0 *)p_a[0] );
        }
      }
    }
  }

  for (int i = 0; i < 1; i++) {
    if (args[i].argtype == OPS_ARG_DAT)
      free(p_a[i]);
  }
}


template < class T0, class T1 >
void ops_par_loop(void (*kernel)( T0*, T1* ),
  char const * name, int dim, int *range,
  ops_arg arg0,
  ops_arg arg1) {

  char** p_a[2];
  ops_arg args[2] = {arg0, arg1};

  // consistency checks
  // ops_args_check(2,args,name);

  for (int i = 0; i < 2; i++) {
    if (args[i].argtype == OPS_ARG_DAT)
      p_a[i] = (char **)malloc(args[i].stencil->points * sizeof(char *));
  }

  // loop over set elements

  if (dim == 1) {
    for (int n_x = range[0]; n_x < range[1]; n_x++) {
      ops_args_set(n_x, 2, args,p_a);
      // call kernel function, passing in pointers to data
      kernel( (T0 *)p_a[0], (T1 *)p_a[1]);
    }
  }
  else if (dim == 2) {
    for (int n_y = range[2]; n_y < range[3]; n_y++) {
      for (int n_x = range[0]; n_x < range[1]; n_x++) {
        ops_args_set(n_x, n_y, 2, args,p_a);
        // call kernel function, passing in pointers to data
        kernel( (T0 *)p_a[0], (T1 *)p_a[1] );
      }
    }
  }
  else if (dim == 3) {
    for (int n_z = range[4]; n_z < range[5]; n_z++) {
      for (int n_y = range[2]; n_y < range[3]; n_y++) {
        for (int n_x = range[0]; n_x < range[1]; n_x++) {
          ops_args_set(n_x, n_y, n_z, 2, args,p_a);
          // call kernel function, passing in pointers to data
          kernel( (T0 *)p_a[0], (T1 *)p_a[1] );
        }
      }
    }
  }

  for (int i = 0; i < 2; i++) {
    if (args[i].argtype == OPS_ARG_DAT)
      free(p_a[i]);
  }
}


template < class T0, class T1, class T2 >
void ops_par_loop(void (*kernel)( T0*, T1*, T2* ),
  char const * name, int dim, int *range,
  ops_arg arg0,
  ops_arg arg1,
  ops_arg arg2) {

  char** p_a[3];
  ops_arg args[3] = {arg0, arg1, arg2};

  // consistency checks
  // ops_args_check(3,args,name);

  for (int i = 0; i < 3; i++) {
    if (args[i].argtype == OPS_ARG_DAT)
      p_a[i] = (char **)malloc(args[i].stencil->points * sizeof(char *));
  }

  // loop over set elements

  if (dim == 1) {
    for (int n_x = range[0]; n_x < range[1]; n_x++) {
      ops_args_set(n_x, 3, args,p_a);
      // call kernel function, passing in pointers to data
      kernel( (T0 *)p_a[0], (T1 *)p_a[1], (T2 *)p_a[2]);
    }
  }
  else if (dim == 2) {
    for (int n_y = range[2]; n_y < range[3]; n_y++) {
      for (int n_x = range[0]; n_x < range[1]; n_x++) {
        ops_args_set(n_x, n_y, 3, args,p_a);
        // call kernel function, passing in pointers to data
        kernel( (T0 *)p_a[0], (T1 *)p_a[1], (T2 *)p_a[2] );
      }
    }
  }
  else if (dim == 3) {
    for (int n_z = range[4]; n_z < range[5]; n_z++) {
      for (int n_y = range[2]; n_y < range[3]; n_y++) {
        for (int n_x = range[0]; n_x < range[1]; n_x++) {
          ops_args_set(n_x, n_y, n_z, 3, args,p_a);
          // call kernel function, passing in pointers to data
          kernel( (T0 *)p_a[0], (T1 *)p_a[1], (T2 *)p_a[2] );
        }
      }
    }
  }

  for (int i = 0; i < 3; i++) {
    if (args[i].argtype == OPS_ARG_DAT)
      free(p_a[i]);
  }
}
