#!/usr/bin/env python

# Open source copyright declaration based on BSD open source template:
# http://www.opensource.org/licenses/bsd-license.php
#
# This file is part of the OPS distribution.
#
# Copyright (c) 2013, Mike Giles and others. Please see the AUTHORS file in
# the main source directory for a full list of copyright holders.
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
# Redistributions of source code must retain the above copyright
# notice, this list of conditions and the following disclaimer.
# Redistributions in binary form must reproduce the above copyright
# notice, this list of conditions and the following disclaimer in the
# documentation and/or other materials provided with the distribution.
# The name of Mike Giles may not be used to endorse or promote products
# derived from this software without specific prior written permission.
# THIS SOFTWARE IS PROVIDED BY Mike Giles ''AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL Mike Giles BE LIABLE FOR ANY
# DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
# ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#######################################################################
#                                                                     #
#       This Python routine generates the header file ops_seq.h       #
#                                                                     #
#######################################################################


#
# this sets the max number of arguments in ops_par_loop
#
maxargs = 18

#open/create file
f = open('./ops_mpi_seq.h','w')

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
#ifdef OPS_MPI
#include "ops_mpi_core.h"
#endif

"""

f.write(top)

#
# now define the macros and extern vars
#

f.write('#ifndef OPS_ACC_MACROS\n')
f.write('#ifdef OPS_3D\n')
f.write('#ifndef OPS_DEBUG\n')
for nargs in range (0,maxargs):
  f.write('#define OPS_ACC'+str(nargs)+'(x,y,z) (x+xdim'+str(nargs)+'*(y)+xdim'+str(nargs)+'*ydim'+str(nargs)+'*(z))\n')
f.write('#else\n\n')
for nargs in range (0,maxargs):
  f.write('#define OPS_ACC'+str(nargs)+'(x,y,z) (ops_stencil_check_3d('+str(nargs)+', x, y, z, xdim'+str(nargs)+', ydim'+str(nargs)+'))\n')
f.write('#endif\n')
f.write('#else\n')
f.write('#ifndef OPS_DEBUG\n')
for nargs in range (0,maxargs):
  f.write('#define OPS_ACC'+str(nargs)+'(x,y) (x+xdim'+str(nargs)+'*(y))\n')
f.write('#else\n\n')
for nargs in range (0,maxargs):
  f.write('#define OPS_ACC'+str(nargs)+'(x,y) (ops_stencil_check_2d('+str(nargs)+', x, y, xdim'+str(nargs)+', -1))\n')
f.write('#endif\n')
f.write('#endif\n')
f.write('#endif\n\n')

for nargs in range (0,maxargs):
  f.write('extern int xdim'+str(nargs)+';\n')

f.write('#ifdef OPS_3D\n')
for nargs in range (0,maxargs):
  f.write('extern int ydim'+str(nargs)+';\n')
f.write('#endif\n')
functions =  """

static int arg_idx[OPS_MAX_DIM];

inline int mult(int* size, int dim)
{
  int result = 1;
  if(dim > 0) {
    for(int i = 0; i<dim;i++) result *= size[i];
  }
  return result;
}

inline int add(int* coords, int* size, int dim)
{
  int result = coords[0];
  for(int i = 1; i<=dim;i++) result += coords[i]*mult(size,i);
  return result;
}


inline int off(int ndim, int dim , int* start, int* end, int* size, int* stride)
{

  int i = 0;
  int c1[3];
  int c2[3];

  for(i=0; i<=dim; i++) c1[i] = start[i]+1;
  for(i=dim+1; i<ndim; i++) c1[i] = start[i];

  for(i = 0; i<dim; i++) c2[i] = end[i];
  for(i=dim; i<ndim; i++) c2[i] = start[i];

  for (i = 0; i < ndim; i++) {
    c1[i] *= stride[i];
    c2[i] *= stride[i];
  }
  int off =  add(c1, size, dim) - add(c2, size, dim);

  return off;
}

inline int address(int ndim, int dat_size, int* start, int* size, int* stride, int* base_off, int *d_m)
{
  int base = 0;
  for(int i=0; i<ndim; i++) {
    base = base + dat_size * mult(size, i) * (start[i] * stride[i] - base_off[i] - d_m[i]);
  }
  return base;
}

inline void stencil_depth(ops_stencil sten, int* d_pos, int* d_neg)
{
  for(int dim = 0;dim<sten->dims;dim++){
    d_pos[dim] = 0; d_neg[dim] = 0;
  }
  for(int p=0;p<sten->points; p++) {
    for(int dim = 0;dim<sten->dims;dim++){
    d_pos[dim] = MAX(d_pos[dim],sten->stencil[sten->dims*p + dim]);
    d_neg[dim] = MIN(d_neg[dim],sten->stencil[sten->dims*p + dim]);
    }
  }
}

"""

f.write(functions)

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

    f.write('void ops_par_loop(void (*kernel)(')
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


    f.write('    char const * name, ops_block block, int dim, int *range,\n    ')
    for n in range (0, nargs):
        f.write(' ops_arg arg'+str(n))
        if nargs <> 1 and n != nargs-1:
          f.write(',')
        else:
          f.write(') {\n')
        if n%n_per_line == 3 and n <> nargs-1:
         f.write('\n    ')

    f.write('\n  char *p_a['+str(nargs)+'];')
    f.write('\n  int  offs['+str(nargs)+'][OPS_MAX_DIM];\n')
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

    f.write('\n  #ifdef CHECKPOINTING\n')
    f.write('  if (!ops_checkpointing_name_before(args,'+str(nargs)+',range,name)) return;\n')
    f.write('  #endif\n\n')
    f.write('  int start[OPS_MAX_DIM];\n');
    f.write('  int end[OPS_MAX_DIM];\n\n')

    f.write('  #ifdef OPS_MPI\n')
    f.write('  sub_block_list sb = OPS_sub_block_list[block->index];\n')
    f.write('  if (!sb->owned) return;\n')
    f.write('  //compute locally allocated range for the sub-block \n' +
            '  int ndim = sb->ndim;\n' )
    f.write('  for (int n=0; n<ndim; n++) {\n')
    f.write('    start[n] = sb->decomp_disp[n];end[n] = sb->decomp_disp[n]+sb->decomp_size[n];\n')
    f.write('    if (start[n] >= range[2*n]) start[n] = 0;\n')
    f.write('    else start[n] = range[2*n] - start[n];\n')
    f.write('    if (sb->id_m[n]==MPI_PROC_NULL && range[2*n] < 0) start[n] = range[2*n];\n')
    f.write('    if (end[n] >= range[2*n+1]) end[n] = range[2*n+1] - sb->decomp_disp[n];\n')
    f.write('    else end[n] = sb->decomp_size[n];\n')
    f.write('    if (sb->id_p[n]==MPI_PROC_NULL && (range[2*n+1] > sb->decomp_disp[n]+sb->decomp_size[n]))\n      end[n] += (range[2*n+1]-sb->decomp_disp[n]-sb->decomp_size[n]);\n')
    f.write('  }\n')
    f.write('  #else //!OPS_MPI\n')
    f.write('  int ndim = block->dims;\n')
    f.write('  for (int n=0; n<ndim; n++) {\n')
    f.write('    start[n] = range[2*n];end[n] = range[2*n+1];\n')
    f.write('  }\n')
    f.write('  #endif //OPS_MPI\n\n')


    #f.write('  double t1,t2,c1,c2;\n')
    #f.write('  ops_timing_hash(name);\n')

    #f.write('  ops_printf("%s\\n",name);\n')
    f.write('  #ifdef OPS_DEBUG\n')
    f.write('  ops_register_args(args, name);\n');
    f.write('  #endif\n\n')

    f.write('  for (int i = 0; i<'+str(nargs)+';i++) {\n')
    f.write('    if(args[i].stencil!=NULL) {\n')
    f.write('      offs[i][0] = args[i].stencil->stride[0]*1;  //unit step in x dimension\n')
    f.write('      for(int n=1; n<ndim; n++) {\n')
    f.write('        offs[i][n] = off(ndim, n, &start[0], &end[0],\n')
    f.write('                         args[i].dat->size, args[i].stencil->stride);\n')
    f.write('      }\n')
    f.write('    }\n')
    f.write('  }\n\n')

    f.write('  //set up initial pointers\n')
    f.write('  for (int i = 0; i < '+str(nargs)+'; i++) {\n')
    f.write('    if (args[i].argtype == OPS_ARG_DAT) {\n')
    f.write('      int d_m[OPS_MAX_DIM];\n')
    f.write('  #ifdef OPS_MPI\n')
    f.write('      for (int d = 0; d < dim; d++) d_m[d] = args[i].dat->d_m[d] + OPS_sub_dat_list[args[i].dat->index]->d_im[d];\n')
    f.write('  #else //OPS_MPI\n')
    f.write('      for (int d = 0; d < dim; d++) d_m[d] = args[i].dat->d_m[d];\n')
    f.write('  #endif //OPS_MPI\n')
    f.write('      p_a[i] = (char *)args[i].data //base of 2D array\n')
    f.write('      + address(ndim, args[i].dat->elem_size, &start[0], \n')
    f.write('        args[i].dat->size, args[i].stencil->stride, args[i].dat->base,\n')
    f.write('        d_m);\n')
    f.write('    }\n')
    f.write('    else if (args[i].argtype == OPS_ARG_GBL) {\n')
    f.write('      if (args[i].acc == OPS_READ) p_a[i] = args[i].data;\n')
    f.write('      else\n')
    f.write('  #ifdef OPS_MPI\n')
    f.write('        p_a[i] = ((ops_reduction)args[i].data)->data + ((ops_reduction)args[i].data)->size * block->index;\n')
    f.write('  #else //OPS_MPI\n')
    f.write('        p_a[i] = ((ops_reduction)args[i].data)->data;\n')
    f.write('  #endif //OPS_MPI\n')
    f.write('    } else if (args[i].argtype == OPS_ARG_IDX) {\n')
    f.write('  #ifdef OPS_MPI\n')
    f.write('      for (int d = 0; d < dim; d++) arg_idx[d] = sb->decomp_disp[d] + start[d];\n')
    f.write('  #else //OPS_MPI\n')
    f.write('      for (int d = 0; d < dim; d++) arg_idx[d] = start[d];\n')
    f.write('  #endif //OPS_MPI\n')
    f.write('      p_a[i] = (char *)arg_idx;\n')
    f.write('    }\n')
    f.write('  }\n\n')


    f.write('  int total_range = 1;\n')
    f.write('  for (int n=0; n<ndim; n++) {\n')
    f.write('    count[n] = end[n]-start[n];  // number in each dimension\n')
    f.write('    total_range *= count[n];\n')
    f.write('    total_range *= (count[n]<0?0:1);\n')
    f.write('  }\n')
    f.write('  count[dim-1]++;     // extra in last to ensure correct termination\n\n')


    for n in range (0, nargs):
      f.write('  if (args['+str(n)+'].argtype == OPS_ARG_DAT) {\n')
      f.write('    xdim'+str(n)+' = args['+str(n)+'].dat->size[0]*args['+str(n)+'].dat->dim;\n')
      f.write('    multi_d'+str(n)+' = args['+str(n)+'].dat->dim;\n')
      f.write('    #ifdef OPS_3D\n')
      f.write('    ydim'+str(n)+' = args['+str(n)+'].dat->size[1];\n')
      f.write('    #endif\n')
      f.write('  }\n')
    f.write('\n')

    #f.write('  //calculate max halodepth for each dat\n')
    #f.write('  for (int i = 0; i<'+str(nargs)+';i++) {\n')
    #f.write('    int max_depth[3];\n')
    #f.write('    max_depth[0] = 0;\n')
    #f.write('    max_depth[1] = 0;\n')
    #f.write('    if(args[i].stencil!=NULL) {\n')
    #f.write('      for(int d = 0; d<ndim; d++) {\n')
    #f.write('        for (int p = 0; p<args[i].stencil->points; p++) {\n')
    #f.write('          max_depth[d] = MAX(max_depth[d], abs(args[i].stencil->stencil[p*ndim+d]));\n')
    #f.write('          if(max_depth[d]>2) printf("larger halo %d\\n",max_depth[d]);\n')
    #f.write('        }\n')
    #f.write('      }\n')
    #f.write('      if(args[i].argtype == OPS_ARG_DAT)\n')
    #f.write('        ops_exchange_halo(&args[i],max_depth);\n')
    #f.write('    }\n')
    #f.write('  }\n\n')
    #f.write('   int d_pos[3];')
    #f.write('   int d_neg[3];')
    #f.write('  for (int i = 0; i < '+str(nargs)+'; i++) {\n')
    #f.write('    if(args[i].argtype == OPS_ARG_DAT) {\n')
    ##f.write('      stencil_depth(args[i].stencil, d_pos, d_neg);\n')
    ##f.write('      ops_exchange_halo2(&args[i],d_pos,d_neg);\n')
    #f.write('      ops_exchange_halo(&args[i],2);\n')
    #f.write('    }\n')
    #f.write('  }\n\n')


    f.write('  ops_H_D_exchanges_host(args, '+str(nargs)+');\n')
    f.write('  ops_halo_exchanges(args,'+str(nargs)+',range);\n')
    f.write('  ops_H_D_exchanges_host(args, '+str(nargs)+');\n')
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
    f.write('      count[m] =  end[m]-start[m];// reset counter\n')
    f.write('      m++;                        // next dimension\n')
    f.write('      count[m]--;                 // decrement counter\n')
    f.write('    }\n\n')

    f.write('    // shift pointers to data\n')
    f.write('    for (int i=0; i<'+str(nargs)+'; i++) {\n')
    f.write('      if (args[i].argtype == OPS_ARG_DAT)\n')
    f.write('        p_a[i] = p_a[i] + (args[i].dat->elem_size * offs[i][m]);\n')
    f.write('      else if (args[i].argtype == OPS_ARG_IDX) {\n')
    f.write('        arg_idx[m]++;\n')
    f.write('  #ifdef OPS_MPI\n')
    f.write('        for (int d = 0; d < m; d++) arg_idx[d] = sb->decomp_disp[d] + start[d];\n')
    f.write('  #else //OPS_MPI\n')
    f.write('        for (int d = 0; d < m; d++) arg_idx[d] = start[d];\n')
    f.write('  #endif //OPS_MPI\n')
    f.write('      }\n')
    f.write('    }\n')
    f.write('  }\n\n')

#    for n in range (0, nargs):
#      f.write('  if (args['+str(n)+'].argtype == OPS_ARG_GBL && args['+str(n)+'].acc != OPS_READ)')
#      f.write('  ops_mpi_reduce(&arg'+str(n)+',(T'+str(n)+' *)p_a['+str(n)+']);\n')
#    f.write('\n')

    f.write('  #ifdef OPS_DEBUG_DUMP\n')
    for n in range (0, nargs):
      f.write('  if (args['+str(n)+'].argtype == OPS_ARG_DAT && args['+str(n)+'].acc != OPS_READ) ops_dump3(args['+str(n)+'].dat,name);\n')
    f.write('  #endif\n')
    for n in range (0, nargs):
      f.write('  if (args['+str(n)+'].argtype == OPS_ARG_DAT && args['+str(n)+'].acc != OPS_READ)')
      f.write('  ops_set_halo_dirtybit3(&args['+str(n)+'],range);\n')
#      f.write('  ops_set_halo_dirtybit(&args['+str(n)+']);\n')
    f.write('  ops_set_dirtybit_host(args, '+str(nargs)+');\n')


    f.write('}\n')
