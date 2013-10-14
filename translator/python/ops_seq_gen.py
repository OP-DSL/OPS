#!/usr/bin/env python
#######################################################################
#                                                                     #
#       This Python routine generates the header file ops_seq.h        #
#                                                                     #
#######################################################################


#
# this sets the max number of arguments in ops_par_loop
#
maxargs = 16

#open/create file
f = open('./ops_seq_opt.h','w')

#
#first the top bit
#

top =  """
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


inline int ops_offs_set(int n_x,
                        int n_y, ops_arg arg){
        return
        arg.dat->block_size[0] * //multiply by the number of
        (n_y - arg.dat->offset[1])  // calculate the offset from index 0 for y dim
        +
        (n_x - arg.dat->offset[0]); //calculate the offset from index 0 for x dim
}

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
    for (int i = 0; i < arg.stencil->points; i++){
      p_arg[i] =
        arg.data + //base of 2D array
        //y dimension -- get to the correct y line
        arg.dat->size * arg.dat->block_size[0] * ( //multiply by the number of
                                                    //bytes per element and xdim block size

        //(n_y - arg.dat->offset[1]) * // calculate the offset from index 0 for y dim
        //arg.stencil->stride[1] + // jump in strides in y dim ??
        n_y * arg.stencil->stride[1] - arg.dat->offset[1] +
        arg.stencil->stencil[i*arg.stencil->dims + 1]) //get the value at the ith
                                                       //stencil point "+ 1" is the y dim
        +
        //x dimension - get to the correct x point on the y line
        arg.dat->size * ( //multiply by the number of bytes per element

        //(n_x - arg.dat->offset[0]) * //calculate the offset from index 0 for x dim
        //arg.stencil->stride[0] + // jump in strides in x dim ??
        n_x * arg.stencil->stride[0] - arg.dat->offset[0] +
        arg.stencil->stencil[i*arg.stencil->dims + 0] //get the value at the ith
                                                      //stencil point "+ 0" is the x dim
      );
    }
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
      arg.dat->size * arg.dat->block_size[1] * arg.dat->block_size[0] * (
      (n_z - arg.dat->offset[2]) *
      arg.stencil->stride[2] +
      arg.stencil->stencil[i*arg.stencil->dims + 2])
      +
      //y dimension -- get to the correct y line on the z plane
      arg.dat->size * arg.dat->block_size[0] * (
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

"""

f.write(top)

#
# now for ops_par_loop defns
#

#
# now for ops_par_loop defns
#

for nargs in range (1,maxargs+1):
    f.write('\n\n//\n')
    f.write('//ops_par_loop routine for '+str(nargs)+' arguments\n')
    f.write('//\n')

    n_per_line = 4

    f.write('template <')
    for n in range (0, nargs):
        f.write('class T'+str(n))
        if nargs <> 1 and n != nargs-1:
          f.write(',')
        else:
          f.write('>\n')
        if n%n_per_line == 3 and n <> nargs-1:
          f.write('\n')

    f.write('void ops_par_loop_opt(void (*kernel)(')
    for n in range (0, nargs):
        f.write('T'+str(n)+'*')
        if nargs <> 1 and n != nargs-1:
          f.write(',')
        else:
          f.write('),\n')
        if n%n_per_line == 3 and n <> nargs-1:
          f.write('\n                           ')
        else:
          f.write(' ')


    f.write('    char const * name, int dim, int *range,\n    ')
    for n in range (0, nargs):
        f.write(' ops_arg arg'+str(n))
        if nargs <> 1 and n != nargs-1:
          f.write(',')
        else:
          f.write(') {\n')
        if n%n_per_line == 3 and n <> nargs-1:
         f.write('\n    ')

    f.write('\n  char **p_a['+str(nargs)+'];')
    f.write('\n  int  offs['+str(nargs)+'][2];\n')
    f.write('\n  int  count[dim];\n')

    f.write('  ops_arg args['+str(nargs)+'] = {')
    for n in range (0, nargs):
        f.write(' arg'+str(n))
        if nargs <> 1 and n != nargs-1:
          f.write(',')
        else:
          f.write('};\n\n')
        if n%n_per_line == 3 and n <> nargs-1:
          f.write('\n                    ')


    f.write('  for (int i = 0; i<'+str(nargs)+';i++) {\n')
    f.write('    if(args[i].stencil!=NULL) {\n')
    f.write('      offs[i][0] = 1;  //unit step in x dimension\n')
    f.write('      offs[i][1] = ops_offs_set(range[0],range[2]+1, args[i]) - ops_offs_set(range[1],range[2], args[i]) +1;\n')

    f.write('      if (args[i].stencil->stride[0] == 0) { //stride in y as x stride is 0\n')
    f.write('        offs[i][0] = 0;\n')
    f.write('        offs[i][1] = args[i].dat->block_size[0];\n')
    f.write('      }\n')
    f.write('      else if (args[i].stencil->stride[1] == 0) {//stride in x as y stride is 0\n')
    f.write('        offs[i][0] = 1;\n')
    f.write('        offs[i][1] = -( range[1] - range[0] ) +1;\n')
    f.write('      }\n')
    f.write('    }\n')
    f.write('  }\n\n')

    f.write('  //store index of non_gbl args\n')
    f.write('  int non_gbl['+str(nargs)+'] = {')
    for n in range (0, nargs):
        f.write('0')
        if nargs <> 1 and n != nargs-1:
          f.write(', ')
        else:
          f.write('};\n\n')
        if n%n_per_line == 5 and n <> nargs-1:
          f.write('\n                    ')
    f.write('  int g = 0;\n')

    f.write('  for (int i = 0; i < '+str(nargs)+'; i++) {\n')
    f.write('    if (args[i].argtype == OPS_ARG_DAT) {\n')
    f.write('      p_a[i] = (char **)malloc(args[i].stencil->points * sizeof(char *));\n')
    f.write('      non_gbl[g++] = i;\n')
    f.write('    }\n')
    f.write('    else if (args[i].argtype == OPS_ARG_GBL)\n')
    f.write('      p_a[i] = (char **)malloc(args[i].dim * sizeof(char *));\n')
    f.write('  }\n\n')

    f.write('  int total_range = 1;\n')
    f.write('  for (int m=0; m<dim; m++) {\n')
    f.write('    count[m] = range[2*m+1]-range[2*m];  // number in each dimension\n')
    f.write('    total_range *= count[m];\n')
    f.write('  }\n')
    f.write('  count[dim-1]++;     // extra in last to ensure correct termination\n\n')


    f.write('  //set up initial pointers\n')
    f.write('  ops_args_set(range[0], range[2], '+str(nargs)+', args,p_a); //set up the initial possition\n\n')

    f.write('  for (int nt=0; nt<total_range; nt++) {\n')

    f.write('    // call kernel function, passing in pointers to data\n')
    f.write('\n    kernel( ')
    for n in range (0, nargs):
        f.write(' (T'+str(n)+' *)p_a['+str(n)+']')
        if nargs <> 1 and n != nargs-1:
          f.write(',')
        else:
          f.write(' );\n\n')
        if n%n_per_line == 3 and n <> nargs-1:
          f.write('\n          ')

    f.write('    count[0]--;   // decrement counter\n')
    f.write('    int m = 0;    // max dimension with changed index\n')

    f.write('    while (count[m]==0) {\n')
    f.write('      count[m] = range[2*m+1]-range[2*m]; // reset counter\n')
    f.write('      m++;                                // next dimension\n')
    f.write('      count[m]--;                         // decrement counter\n')
    f.write('    }\n\n')

    f.write('    int a = 0;\n')
    f.write('    // shift pointers to data\n')
    f.write('    for (int i=0; i<g; i++) {\n')
    f.write('      a = non_gbl[i];\n')
    f.write('      for (int np=0; np<args[a].stencil->points; np++) {\n')
    f.write('        p_a[a][np] = p_a[a][np] + (args[a].dat->size * offs[a][m]);\n')
    f.write('      }\n')
    f.write('    }\n')
    f.write('  }\n')

    f.write('  for (int i = 0; i < '+str(nargs)+'; i++)\n')
    f.write('    free(p_a[i]);\n')
    f.write('}')
