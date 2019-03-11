
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

## @file
## @brief OPS CUDA code generator
#
#  This routine is called by ops.py which parses the input files
#
#  It produces a file xxx_cuda_kernel.cpp for each kernel,
#  plus a master kernel file
#

"""
OPS CUDA code generator

This routine is called by ops.py which parses the input files

It produces a file xxx_cuda_kernel.cpp for each kernel,
plus a master kernel file

"""

import re
import datetime
import os
import glob

import util
import config

import ops_gen_common

para_parse = util.para_parse
comment_remover = util.comment_remover
remove_trailing_w_space = util.remove_trailing_w_space
parse_signature = util.parse_signature
complex_numbers_cuda = util.complex_numbers_cuda
check_accs = util.check_accs
mult = util.mult
convert_ACC = util.convert_ACC
clean_type = util.clean_type

comm = util.comm
code = util.code
FOR = util.FOR
FOR2 = util.FOR2
WHILE = util.WHILE
ENDWHILE = util.ENDWHILE
ENDFOR = util.ENDFOR
IF = util.IF
ELSEIF = util.ELSEIF
ELSE = util.ELSE
ENDIF = util.ENDIF


def ops_gen_mpi_cuda(master, date, consts, kernels, soa_set):

  OPS_ID   = 1;  OPS_GBL   = 2;

  OPS_READ = 1;  OPS_WRITE = 2;  OPS_RW  = 3;
  OPS_INC  = 4;  OPS_MAX   = 5;  OPS_MIN = 6;

  accsstring = ['OPS_READ','OPS_WRITE','OPS_RW','OPS_INC','OPS_MAX','OPS_MIN' ]

  NDIM = 2 #the dimension of the application is hardcoded here .. need to get this dynamically

  src_dir = os.path.dirname(master) or '.'
  master_basename = os.path.splitext(os.path.basename(master))

  gen_full_code = 1
##########################################################################
#  create new kernel file
##########################################################################

  for nk in range (0,len(kernels)):
    arg_typ  = kernels[nk]['arg_type']
    name  = kernels[nk]['name']
    nargs = kernels[nk]['nargs']
    dim   = kernels[nk]['dim']
    dims  = kernels[nk]['dims']
    stens = kernels[nk]['stens']
    var   = kernels[nk]['var']
    accs  = kernels[nk]['accs']
    typs  = kernels[nk]['typs']
    NDIM = int(dim)
    #parse stencil to locate strided access
    stride = [1] * nargs * (NDIM+1)
    restrict = [1] * nargs
    prolong = [1] * nargs

    stride = ops_gen_common.parse_strides(stens, nargs, stride, NDIM)

    ### Determine if this is a MULTI_GRID LOOP with
    ### either restrict or prolong
    MULTI_GRID = 0
    any_prolong = 0
    for n in range (0, nargs):
      restrict[n] = 0
      prolong[n] = 0
      if str(stens[n]).find('RESTRICT') > 0:
        restrict[n] = 1
        MULTI_GRID = 1
      if str(stens[n]).find('PROLONG') > 0 :
        prolong[n] = 1
        MULTI_GRID = 1
        any_prolong = 1

    ### Determine if this is a MULTI_GRID LOOP with
    ### either restrict or prolong
    MULTI_GRID = 0
    any_prolong = 0
    for n in range (0, nargs):
      restrict[n] = 0
      prolong[n] = 0
      if str(stens[n]).find('RESTRICT') > 0:
        restrict[n] = 1
        MULTI_GRID = 1
      if str(stens[n]).find('PROLONG') > 0 :
        prolong[n] = 1
        MULTI_GRID = 1
        any_prolong = 1

    reduct = 0
    for n in range (0, nargs):
      if arg_typ[n] == 'ops_arg_gbl' and accs[n] <> OPS_READ:
        reduct = 1

    config.file_text = ''
    config.depth = 0
    n_per_line = 4

    i = name.find('kernel')
    name2 = name[0:i-1]
    #print name2

    reduction = False
    ng_args = 0

    for n in range (0, nargs):
      if arg_typ[n] == 'ops_arg_gbl' and accs[n] <> OPS_READ:
        reduction = True
      else:
        ng_args = ng_args + 1

    arg_idx = -1
    for n in range (0, nargs):
      if arg_typ[n] == 'ops_arg_idx':
        arg_idx = n

##########################################################################
#  generate constants and MACROS
##########################################################################

    code('__constant__ int dims_'+name+' ['+str(nargs)+']['+str(NDIM+1)+'];')
    code('static int dims_'+name+'_h ['+str(nargs)+']['+str(NDIM+1)+'] = {0};')
    code('')

##########################################################################
#  generate header
##########################################################################

    comm('user function')

    ret = ops_gen_common.get_user_function(name, arg_typ, src_dir)
    arg_list = ret[1]
    kernel_text = ret[0]
    #new_code = complex_numbers_cuda(kernel_text)  # Handle complex numbers with the cuComplex.h CUDA library.
    #code(new_code)
    code('')
    code('')


##########################################################################
#  generate cuda kernel wrapper function
##########################################################################

    code('__global__ void ops_'+name+'(')
    for n in range (0, nargs):
      if arg_typ[n] == 'ops_arg_dat':
        code(typs[n]+'* __restrict '+clean_type(arg_list[n])+'_p,')
      elif arg_typ[n] == 'ops_arg_gbl':
        if accs[n] == OPS_READ:
          if dims[n].isdigit() and int(dims[n])==1:
            code('const '+typs[n]+' '+clean_type(arg_list[n])+'_p,')
          else:
            code('const '+typs[n]+'* __restrict '+clean_type(arg_list[n])+',')
        else:
          code(typs[n]+'* __restrict '+clean_type(arg_list[n])+'_p,')
      if restrict[n] or prolong[n]:
        if NDIM == 1:
          code('int stride_'+str(n)+'0,')
        if NDIM == 2:
          code('int stride_'+str(n)+'0, int stride_'+str(n)+'1,')
        if NDIM == 3:
          code('int stride_'+str(n)+'0, int stride_'+str(n)+'1, int stride_'+str(n)+'2,')

    if arg_idx>=0 or any_prolong:
        code('#ifdef OPS_MPI')
        if NDIM==1:
          code('int arg_idx0,')
        elif NDIM==2:
          code('int arg_idx0, int arg_idx1,')
        elif NDIM==3:
          code('int arg_idx0, int arg_idx1, int arg_idx2,')
        code('#endif')

    if NDIM==1:
      code('int bounds_0_l, int bounds_0_u, int bounds_1_l, int bounds_1_u ){')
    elif NDIM==2:
      code('int bounds_0_l, int bounds_0_u, int bounds_1_l, int bounds_1_u,')
      code('int bounds_2_l, int bounds_2_u) {')
    elif NDIM==3:
      code('int bounds_0_l, int bounds_0_u, int bounds_1_l, int bounds_1_u,')
      code('int bounds_2_l, int bounds_2_u, int bounds_3_l, int bounds_3_u) {')

    config.depth = config.depth + 2

    #local variable to hold reductions on GPU
    code('')
    ops_gen_common.generate_gbl_locals(nargs, arg_typ, accs, dims, typs, arg_list)
    for n in range(0,nargs):
        if arg_typ[n] == 'ops_arg_gbl' and accs[n] == OPS_READ and dims[n].isdigit() and int(dims[n])==1:
            code('const '+typs[n]+' * __restrict__ '+clean_type(arg_list[n])+' = &'+clean_type(arg_list[n])+'_p;')
    code('')
    ops_gen_common.generate_strides(nargs, stens, stride, NDIM)

    code('')
    if NDIM==3:
      code('int n_2 = bounds_2_l + blockDim.z * blockIdx.z + threadIdx.z;')
      code('int n_3 = n_2/(bounds_2_u-bounds_2_l);')
      code('#ifdef OPS_BATCHED')
      code('n_2 = n_2%(bounds_2_u-bounds_2_l) ')
      code('#endif')
    if NDIM==2:
      code('int n_2 = bounds_2_l + blockDim.z * blockIdx.z + threadIdx.z;')
    code('int n_1 = bounds_1_l + blockDim.y * blockIdx.y + threadIdx.y;')
    code('int n_0 = bounds_0_l + blockDim.x * blockIdx.x + threadIdx.x;')
    code('')
    if arg_idx>=0:
      code('int '+arg_list[arg_idx]+'['+str(NDIM+1)+']={0};')
      code('#ifdef OPS_MPI')
      code(arg_list[arg_idx]+'[0] = arg_idx0+n_0;')
      if NDIM>1:
        code(arg_list[arg_idx]+'[1] = arg_idx1+n_1;')
      if NDIM>2:
        code(arg_list[arg_idx]+'[2] = arg_idx2+n_2;')
      code('#else')
      code(arg_list[arg_idx]+'[0] = n_0;')
      code(arg_list[arg_idx]+'[1] = n_1;')
      if NDIM>1:
        code(arg_list[arg_idx]+'[2] = n_2;')
      if NDIM>2:
        code(arg_list[arg_idx]+'[3] = n_3;')
      code('#endif')


    if NDIM==1:
      IF('n_0 < bounds_0_u && n_1 < bounds_1_u')
    if NDIM==2:
      IF('n_0 < bounds_0_u && n_1 < bounds_1_u && n_2 < bounds_2_u')
    elif NDIM==3:
      IF('n_0 < bounds_0_u && n_1 < bounds_1_u && n_2 < bounds_2_u && n_3 < bounds_3_u')

    def dimstr(n,d):
      return 'dims_'+name+'['+str(n)+']['+str(d)+']'
    ops_gen_common.generate_accessors(nargs, arg_typ, dims, NDIM, stride, typs, accs, arg_list, restrict, prolong, dimstr)
    code(kernel_text)
    ENDIF()

    #reduction across blocks
    if NDIM==1:
      cont = '((blockIdx.x + blockIdx.y*gridDim.x) + n_1*gridDim.x )*'
    if NDIM==2:
      cont = '((blockIdx.x + blockIdx.y*gridDim.x) + n_2*gridDim.x*gridDim.y )*'
    elif NDIM==3:
      cont = '((blockIdx.x + blockIdx.y*gridDim.x + blockIdx.z*gridDim.x*gridDim.y) + n_3*gridDim.x*gridDim.y*(gridDim.z/(bounds_3_u-bounds_3_l)) )*'
    for n in range (0, nargs):
      if arg_typ[n] == 'ops_arg_gbl' and accs[n] == OPS_INC:
        code('for (int d=0; d<'+str(dims[n])+'; d++)')
        code('  ops_reduction_cuda<OPS_INC>(&'+clean_type(arg_list[n])+'_p[d+'+cont+str(dims[n])+'],'+clean_type(arg_list[n])+'[d]);')
      if arg_typ[n] == 'ops_arg_gbl' and accs[n] == OPS_MIN:
        code('for (int d=0; d<'+str(dims[n])+'; d++)')
        code('  ops_reduction_cuda<OPS_MIN>(&'+clean_type(arg_list[n])+'_p[d+'+cont+str(dims[n])+'],'+clean_type(arg_list[n])+'[d]);')
      if arg_typ[n] == 'ops_arg_gbl' and accs[n] == OPS_MAX:
        code('for (int d=0; d<'+str(dims[n])+'; d++)')
        code('  ops_reduction_cuda<OPS_MAX>(&'+clean_type(arg_list[n])+'_p[d+'+cont+str(dims[n])+'],'+clean_type(arg_list[n])+'[d]);')


    code('')
    config.depth = config.depth - 2
    code('}')


##########################################################################
#  now host stub
##########################################################################

    ops_gen_common.generate_header(nk, name, nargs, arg_typ, accs, arg_idx, NDIM, MULTI_GRID, gen_full_code)
    
    ops_gen_common.generate_sizes_bounds(nargs, arg_typ, NDIM, 0)

    condition = ''
    for n in range (0, nargs):
      if arg_typ[n] == 'ops_arg_dat':
        condition = condition + 'args['+str(n)+'].dat->size[0] != dims_'+name+'_h['+str(n)+'][0] || '
        condition = condition + 'args['+str(n)+'].dat->size[1] != dims_'+name+'_h['+str(n)+'][1] || '
        if NDIM>1:
          condition = condition + 'args['+str(n)+'].dat->size[2] != dims_'+name+'_h['+str(n)+'][2] || '
        if NDIM>2:
          condition = condition + 'args['+str(n)+'].dat->size[3] != dims_'+name+'_h['+str(n)+'][3] || '
    condition = condition[:-4]
    IF(condition)

    for n in range (0, nargs):
      if arg_typ[n] == 'ops_arg_dat':
        code('dims_'+name+'_h['+str(n)+'][0] = args['+str(n)+'].dat->size[0];')
        code('dims_'+name+'_h['+str(n)+'][1] = args['+str(n)+'].dat->size[1];')
        if NDIM>1:
          code('dims_'+name+'_h['+str(n)+'][2] = args['+str(n)+'].dat->size[2];')
        if NDIM>2:
          code('dims_'+name+'_h['+str(n)+'][2] = args['+str(n)+'].dat->size[3];')
    code('cutilSafeCall(cudaMemcpyToSymbol( dims_'+name+', dims_'+name+'_h, sizeof(dims_'+name+')));')
    ENDIF()

    code('')

    ops_gen_common.generate_pointers(nargs, arg_typ, accs, typs, arg_list, restrict, prolong,  dims, 0, '_d')

    code('')
    code('int x_size = MAX(0,bounds_0_u-bounds_0_l);')
    code('int y_size = MAX(0,bounds_1_u-bounds_1_l);')
    if NDIM>1:
      code('int z_size = MAX(0,bounds_2_u-bounds_2_l);')
    if NDIM>2:
      code('z_size *= MAX(0,bounds_3_u-bounds_3_l);')
    code('')

    #set up CUDA grid and thread blocks for kernel call
    if NDIM==1:
      code('dim3 grid( (x_size-1)/OPS_instance::getOPSInstance()->OPS_block_size_x+ 1, (y_size-1)/OPS_instance::getOPSInstance()->OPS_block_size_y + 1, 1);')
    if NDIM==2:
      code('dim3 grid( (x_size-1)/OPS_instance::getOPSInstance()->OPS_block_size_x+ 1, (y_size-1)/OPS_instance::getOPSInstance()->OPS_block_size_y + 1, (z_size-1)/OPS_instance::getOPSInstance()->OPS_block_size_z+1);')

    if NDIM==1:
      code('dim3 tblock(MIN(OPS_instance::getOPSInstance()->OPS_block_size_x, x_size), MIN(OPS_instance::getOPSInstance()->OPS_block_size_y, y_size),1);')
    else:
      code('dim3 tblock(MIN(OPS_instance::getOPSInstance()->OPS_block_size_x, x_size), MIN(OPS_instance::getOPSInstance()->OPS_block_size_y, y_size),MIN(OPS_instance::getOPSInstance()->OPS_block_size_z, z_size));')

    code('')

    GBL_READ = False
    GBL_READ_MDIM = False
    GBL_INC = False
    GBL_MAX = False
    GBL_MIN = False
    GBL_WRITE = False

    #set up reduction variables
    for n in range (0, nargs):
      if arg_typ[n] == 'ops_arg_gbl':
        if accs[n] == OPS_READ:
          GBL_READ = True
          if not dims[n].isdigit() or int(dims[n])>1:
            GBL_READ_MDIM = True
        if accs[n] == OPS_INC:
          GBL_INC = True
        if accs[n] == OPS_MAX:
          GBL_MAX = True
        if accs[n] == OPS_MIN:
          GBL_MIN = True
        if accs[n] == OPS_WRITE:
          GBL_WRITE = True

    if GBL_INC == True or GBL_MIN == True or GBL_MAX == True or GBL_WRITE == True:
      code('int nblocks = grid.x * grid.y * grid.z;')
      code('int reduct_bytes = 0;')
      code('int reduct_size = 0;')
      code('')

    if GBL_READ == True and GBL_READ_MDIM == True:
      code('int consts_bytes = 0;')
      code('')

    for n in range (0, nargs):
      if arg_typ[n] == 'ops_arg_gbl':
        if accs[n] == OPS_READ and (not dims[n].isdigit() or int(dims[n])>1):
          code('consts_bytes += ROUND_UP('+str(dims[n])+'*sizeof('+typs[n]+'));')
        elif accs[n] <> OPS_READ:
          code('reduct_bytes += ROUND_UP(nblocks*'+str(dims[n])+'*sizeof('+typs[n]+'));')
          code('reduct_size = MAX(reduct_size,sizeof('+typs[n]+')*'+str(dims[n])+');')
    code('')

    if GBL_READ == True and GBL_READ_MDIM == True:
      code('reallocConstArrays(consts_bytes);')
    if GBL_INC == True or GBL_MIN == True or GBL_MAX == True or GBL_WRITE == True:
      code('reallocReductArrays(reduct_bytes);')
      code('reduct_bytes = 0;')
      code('')

    for n in range (0, nargs):
      if arg_typ[n] == 'ops_arg_gbl' and accs[n] <> OPS_READ:
        code('arg'+str(n)+'.data = OPS_instance::getOPSInstance()->OPS_reduct_h + reduct_bytes;')
        code('arg'+str(n)+'.data_d = OPS_instance::getOPSInstance()->OPS_reduct_d + reduct_bytes;')
        code('for (int b=0; b<nblocks; b++)')
        if accs[n] == OPS_INC:
          code('for (int d=0; d<'+str(dims[n])+'; d++) (('+typs[n]+' *)arg'+str(n)+'.data)[d+b*'+str(dims[n])+'] = ZERO_'+typs[n]+';')
        if accs[n] == OPS_MAX:
          code('for (int d=0; d<'+str(dims[n])+'; d++) (('+typs[n]+' *)arg'+str(n)+'.data)[d+b*'+str(dims[n])+'] = -INFINITY_'+typs[n]+';')
        if accs[n] == OPS_MIN:
          code('for (int d=0; d<'+str(dims[n])+'; d++) (('+typs[n]+' *)arg'+str(n)+'.data)[d+b*'+str(dims[n])+'] = INFINITY_'+typs[n]+';')
        code('reduct_bytes += ROUND_UP(nblocks*'+str(dims[n])+'*sizeof('+typs[n]+'));')
        code('')

    code('')

    for n in range (0, nargs):
      if arg_typ[n] == 'ops_arg_gbl':
        if accs[n] == OPS_READ and (not dims[n].isdigit() or int(dims[n])>1):
          code('consts_bytes = 0;')
          code('arg'+str(n)+'.data = OPS_instance::getOPSInstance()->OPS_consts_h + consts_bytes;')
          code('arg'+str(n)+'.data_d = OPS_instance::getOPSInstance()->OPS_consts_d + consts_bytes;')
          code('for (int d=0; d<'+str(dims[n])+'; d++) (('+typs[n]+' *)arg'+str(n)+'.data)[d] = p_a'+str(n)+'[d];')
          code('consts_bytes += ROUND_UP('+str(dims[n])+'*sizeof(int));')
    if GBL_READ == True and GBL_READ_MDIM == True:
      code('mvConstArraysToDevice(consts_bytes);')

    if GBL_INC == True or GBL_MIN == True or GBL_MAX == True or GBL_WRITE == True:
      code('mvReductArraysToDevice(reduct_bytes);')


    ops_gen_common.generate_exchanges(nargs, nk, gen_full_code, 'device')

    #set up shared memory for reduction
    if GBL_INC == True or GBL_MIN == True or GBL_MAX == True or GBL_WRITE == True:
       code('int nshared = 0;')
       code('int nthread = OPS_instance::getOPSInstance()->OPS_block_size_x*OPS_instance::getOPSInstance()->OPS_block_size_y*OPS_instance::getOPSInstance()->OPS_block_size_z;')
       code('')
    for n in range (0, nargs):
      if arg_typ[n] == 'ops_arg_gbl' and accs[n] <> OPS_READ:
        code('nshared = MAX(nshared,sizeof('+typs[n]+')*'+str(dims[n])+');')
    code('')
    if GBL_INC == True or GBL_MIN == True or GBL_MAX == True or GBL_WRITE == True:
      code('nshared = MAX(nshared*nthread,reduct_size*nthread);')
      code('')


   #kernel call
    comm('call kernel wrapper function, passing in pointers to data')
    if NDIM==1:
      code('if (x_size > 0)')
    if NDIM==2:
      code('if (x_size > 0 && y_size > 0)')
    if NDIM==3:
      code('if (x_size > 0 && y_size > 0 && z_size > 0)')
    config.depth = config.depth + 2
    n_per_line = 4
    if GBL_INC == True or GBL_MIN == True or GBL_MAX == True or GBL_WRITE == True:
      text = 'ops_'+name+'<<<grid, tblock, nshared >>> ( '
    else:
      text = 'ops_'+name+'<<<grid, tblock >>> ( '
    for n in range (0, nargs):
      if arg_typ[n] == 'ops_arg_dat':
        text = text +' '+arg_list[n]+'_p,'
      elif arg_typ[n] == 'ops_arg_gbl':
        if dims[n].isdigit() and int(dims[n])==1 and accs[n]==OPS_READ:
          text = text +' *'+arg_list[n]+','
        else:
          text = text +' ('+typs[n]+' *)arg'+str(n)+'.data_d,'
      if restrict[n] or prolong[n]:
        if NDIM==1:
          text = text + 'stride_'+str(n)+'[0],'
        if NDIM==2:
          text = text + 'stride_'+str(n)+'[0],stride_'+str(n)+'[1],'
        if NDIM==3:
          text = text + 'stride_'+str(n)+'[0],stride_'+str(n)+'[1],stride_'+str(n)+'[2],'

      if n%n_per_line == 1 and n <> nargs-1:
        text = text +'\n        '
    if arg_idx>=0 or any_prolong:
        text = text + '\n#ifdef OPS_MPI\n'
        if NDIM==1:
          text = text + config.depth*' '+'     arg_idx[0],'
        elif NDIM==2:
          text = text + config.depth*' '+'     arg_idx[0], arg_idx[1],'
        elif NDIM==3:
          text = text + config.depth*' '+'     arg_idx[0], arg_idx[1], arg_idx[2],'
        text = text + '\n#endif\n'

    if NDIM==1:
      text = text +config.depth*' '+'     bounds_0_l, bounds_0_u, bounds_1_l, bounds_1_u );'
    if NDIM==2:
      text = text +config.depth*' '+'     bounds_0_l, bounds_0_u, bounds_1_l, bounds_1_u,\n'
      text = text +config.depth*' '+'     bounds_2_l, bounds_2_u);'
    elif NDIM==3:
      text = text +config.depth*' '+'     bounds_0_l, bounds_0_u, bounds_1_l, bounds_1_u,\n'
      text = text +config.depth*' '+'     bounds_2_l, bounds_2_u, bounds_3_l, bounds_3_u);'
    code(text);
    config.depth = config.depth - 2

    code('')
    code('cutilSafeCall(cudaGetLastError());')
    code('')

    #
    # Complete Reduction Operation by moving data onto host
    # and reducing over blocks
    #
    if GBL_INC == True or GBL_MIN == True or GBL_MAX == True or GBL_WRITE == True:
      code('mvReductArraysToHost(reduct_bytes);')

    for n in range (0, nargs):
      if arg_typ[n] == 'ops_arg_gbl' and accs[n] <> OPS_READ:
        FOR('b','0','nblocks')
        FOR('d','0',str(dims[n]))
        if accs[n] == OPS_INC:
          code('p_a'+str(n)+'[(b/(nblocks/block->count))*'+str(dims[n])+' + d] = p_a'+str(n)+'[(b/(nblocks/block->count))*'+str(dims[n])+' + d] + (('+typs[n]+' *)arg'+str(n)+'.data)[d+b*'+str(dims[n])+'];')
        elif accs[n] == OPS_MAX:
          code('p_a'+str(n)+'[(b/(nblocks/block->count))*'+str(dims[n])+' + d] = MAX(p_a'+str(n)+'[(b/(nblocks/block->count))*'+str(dims[n])+' + d],(('+typs[n]+' *)arg'+str(n)+'.data)[d+b*'+str(dims[n])+']);')
        elif accs[n] == OPS_MIN:
          code('p_a'+str(n)+'[(b/(nblocks/block->count))*'+str(dims[n])+' + d] = MIN(p_a'+str(n)+'[(b/(nblocks/block->count))*'+str(dims[n])+' + d],(('+typs[n]+' *)arg'+str(n)+'.data)[d+b*'+str(dims[n])+']);')
        ENDFOR()
        ENDFOR()
        code('arg'+str(n)+'.data = (char *)p_a'+str(n)+';')
        code('')

    IF('OPS_instance::getOPSInstance()->OPS_diags>1')
    code('cutilSafeCall(cudaDeviceSynchronize());')
    ENDIF()
    code('')
    
    ops_gen_common.generate_tail(nk, nargs, arg_typ, accs, gen_full_code, 'device')

    code('#ifdef OPS_LAZY')
    code('void ops_par_loop_'+name+'(char const *name, ops_block block, int dim, int* range,')
    text = ''
    for n in range (0, nargs):

      text = text +' ops_arg arg'+str(n)
      if nargs <> 1 and n != nargs-1:
        text = text +','
      else:
        text = text +') {'
      if n%n_per_line == 3 and n <> nargs-1:
         text = text +'\n'
    code(text);
    config.depth = 2
    code('ops_kernel_descriptor *desc = (ops_kernel_descriptor *)malloc(sizeof(ops_kernel_descriptor));')
    #code('desc->name = (char *)malloc(strlen(name)+1);')
    #code('strcpy(desc->name, name);')
    code('desc->name = name;')
    code('desc->block = block;')
    code('desc->dim = dim;')
    code('desc->device = 1;')
    code('desc->index = '+str(nk)+';')
    code('desc->hash = 5381;')
    code('desc->hash = ((desc->hash << 5) + desc->hash) + '+str(nk)+';')
    FOR('i','0',str(2*NDIM))
    code('desc->range[i] = range[i];')
    code('desc->orig_range[i] = range[i];')
    code('desc->hash = ((desc->hash << 5) + desc->hash) + range[i];')
    ENDFOR()

    code('desc->nargs = '+str(nargs)+';')
    code('desc->args = (ops_arg*)malloc('+str(nargs)+'*sizeof(ops_arg));')
    declared = 0
    for n in range (0, nargs):
      code('desc->args['+str(n)+'] = arg'+str(n)+';')
      if arg_typ[n] == 'ops_arg_dat':
        code('desc->hash = ((desc->hash << 5) + desc->hash) + arg'+str(n)+'.dat->index;')
      if arg_typ[n] == 'ops_arg_gbl' and accs[n] == OPS_READ:
        if declared == 0:
          code('char *tmp = (char*)malloc('+dims[n]+'*sizeof('+typs[n]+'));')
          declared = 1
        else:
          code('tmp = (char*)malloc('+dims[n]+'*sizeof('+typs[n]+'));')
        code('memcpy(tmp, arg'+str(n)+'.data,'+dims[n]+'*sizeof('+typs[n]+'));')
        code('desc->args['+str(n)+'].data = tmp;')
    code('desc->function = ops_par_loop_'+name+'_execute;')
    IF('OPS_instance::getOPSInstance()->OPS_diags > 1')
    code('ops_timing_realloc('+str(nk)+',"'+name+'");')
    ENDIF()
    code('ops_enqueue_kernel(desc);')
    config.depth = 0
    code('}')
    code('#endif')


##########################################################################
#  output individual kernel file
##########################################################################
    if not os.path.exists('./CUDA'):
      os.makedirs('./CUDA')
    fid = open('./CUDA/'+name+'_cuda_kernel.cu','w')
    date = datetime.datetime.now()
    fid.write('//\n// auto-generated by ops.py\n//\n')
    fid.write(config.file_text)
    fid.close()

# end of main kernel call loop

##########################################################################
#  output one master kernel file
##########################################################################

  config.file_text =''
  config.depth = 0
  comm('header')
  code('#define OPS_API 2')
  if NDIM==1:
    code('#define OPS_1D')
  if NDIM==2:
    code('#define OPS_2D')
  if NDIM==3:
    code('#define OPS_3D')
  if soa_set:
    code('#define OPS_SOA')
  code('#include "ops_lib_cpp.h"')
  code('')
  code('#include "ops_cuda_rt_support.h"')
  code('#include "ops_cuda_reduction.h"')
  code('')
  code('#include <cuComplex.h>')  # Include the CUDA complex numbers library, in case complex numbers are used anywhere.
  code('')
  if os.path.exists(os.path.join(src_dir,'user_types.h')):
    code('#define OPS_FUN_PREFIX __device__ __host__')
    code('#include "user_types.h"')
  code('#ifdef OPS_MPI')
  code('#include "ops_mpi_core.h"')
  code('#endif')

  comm(' global constants')
  for nc in range (0,len(consts)):
    if consts[nc]['dim'].isdigit() and int(consts[nc]['dim'])==1:
      code('__constant__ '+consts[nc]['type']+' '+(str(consts[nc]['name']).replace('"','')).strip()+';')
    else:
      if consts[nc]['dim'].isdigit() and consts[nc]['dim'] > 0:
        num = str(consts[nc]['dim'])
        code('__constant__ '+consts[nc]['type']+' '+(str(consts[nc]['name']).replace('"','')).strip()+'['+num+'];')
      else:
        code('__constant__ '+consts[nc]['type']+' *'+(str(consts[nc]['name']).replace('"','')).strip()+';')



  code('')
  code('void ops_init_backend() {}')
  code('')
  code('void ops_decl_const_char(int dim, char const *type,')
  code('int size, char *dat, char const *name){')
  config.depth = config.depth + 2

  for nc in range(0,len(consts)):
    IF('!strcmp(name,"'+(str(consts[nc]['name']).replace('"','')).strip()+'")')
    if consts[nc]['dim'].isdigit():
      code('cutilSafeCall(cudaMemcpyToSymbol('+(str(consts[nc]['name']).replace('"','')).strip()+', dat, dim*size));')
    else:
      code('char *temp; cutilSafeCall(cudaMalloc((void**)&temp,dim*size));')
      code('cutilSafeCall(cudaMemcpy(temp,dat,dim*size,cudaMemcpyHostToDevice));')
      code('cutilSafeCall(cudaMemcpyToSymbol('+(str(consts[nc]['name']).replace('"','')).strip()+', &temp, sizeof(char *)));')
    ENDIF()
    code('else')

  code('{')
  config.depth = config.depth + 2
  code('printf("error: unknown const name\\n"); exit(1);')
  ENDIF()

  config.depth = config.depth - 2
  code('}')
  code('')

  code('')
  comm('user kernel files')

  kernel_name_list = []

  for nk in range(0,len(kernels)):
    if kernels[nk]['name'] not in kernel_name_list :
      code('#include "'+kernels[nk]['name']+'_cuda_kernel.cu"')
      kernel_name_list.append(kernels[nk]['name'])

  fid = open('./CUDA/'+master_basename[0]+'_kernels.cu','w')
  fid.write('//\n// auto-generated by ops.py//\n\n')
  fid.write(config.file_text)
  fid.close()
