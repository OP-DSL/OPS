
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
import errno
import os

import config
from config import OPS_READ, OPS_WRITE, OPS_RW, OPS_INC, OPS_MAX, OPS_MIN

import util
from util import para_parse, parse_signature, complex_numbers_cuda, convert_ACC
from util import comm, code, FOR, ENDFOR, IF, ENDIF


def ops_gen_mpi_cuda(master, consts, kernels, soa_set):
  NDIM = 2 #the dimension of the application is hardcoded here .. need to get this dynamically

  src_dir = os.path.dirname(master) or '.'
  master_basename = os.path.splitext(os.path.basename(master))

  ##########################################################################
  #  create new kernel file
  ##########################################################################
  try:
    os.makedirs('./CUDA')
  except OSError as e:
    if e.errno != errno.EEXIST:
      raise
  for nk in range (0,len(kernels)):
    assert config.file_text == '' and config.depth == 0
    arg_typ  = kernels[nk]['arg_type']
    name  = kernels[nk]['name']
    nargs = kernels[nk]['nargs']
    dim   = kernels[nk]['dim']
    dims  = kernels[nk]['dims']
    stens = kernels[nk]['stens']
    accs  = kernels[nk]['accs']
    typs  = kernels[nk]['typs']
    NDIM = int(dim)
    #parse stencil to locate strided access
    stride = [1] * nargs * NDIM
    restrict = [1] * nargs
    prolong = [1] * nargs


    if NDIM == 2:
      for n in range (0, nargs):
        if str(stens[n]).find('STRID2D_X') > 0:
          stride[NDIM*n+1] = 0
        elif str(stens[n]).find('STRID2D_Y') > 0:
          stride[NDIM*n] = 0

    if NDIM == 3:
      for n in range (0, nargs):
        if str(stens[n]).find('STRID3D_XY') > 0:
          stride[NDIM*n+2] = 0
        elif str(stens[n]).find('STRID3D_YZ') > 0:
          stride[NDIM*n] = 0
        elif str(stens[n]).find('STRID3D_XZ') > 0:
          stride[NDIM*n+1] = 0
        elif str(stens[n]).find('STRID3D_X') > 0:
          stride[NDIM*n+1] = 0
          stride[NDIM*n+2] = 0
        elif str(stens[n]).find('STRID3D_Y') > 0:
          stride[NDIM*n] = 0
          stride[NDIM*n+2] = 0
        elif str(stens[n]).find('STRID3D_Z') > 0:
          stride[NDIM*n] = 0
          stride[NDIM*n+1] = 0

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
      if arg_typ[n] == 'ops_arg_gbl' and accs[n] != OPS_READ:
        reduct = 1

    n_per_line = 4

    i = name.find('kernel')

    reduction = False
    ng_args = 0

    for n in range (0, nargs):
      if arg_typ[n] == 'ops_arg_gbl' and accs[n] != OPS_READ:
        reduction = True
      else:
        ng_args = ng_args + 1

    arg_idx = 0
    for n in range (0, nargs):
      if arg_typ[n] == 'ops_arg_idx':
        arg_idx = 1

    needDimList = []
    for n in range(0,nargs):
      if arg_typ[n] == 'ops_arg_dat' or (arg_typ[n] == 'ops_arg_gbl' and accs[n] != OPS_READ):
        if not dims[n].isdigit():
          needDimList = needDimList + [n]

    ##########################################################################
    # generate constants and MACROS
    ##########################################################################

    num_dims = max(1, NDIM -1)
    if NDIM > 1 and soa_set:
        num_dims += 1;
    code(f'__constant__ int dims_{name} [{nargs}][{num_dims}];')
    code(f'static int dims_{name}_h [{nargs}][{num_dims}] = {{0}};')
    code('')

    ##########################################################################
    #  generate header
    ##########################################################################

    comm('user function')
    text = util.get_file_text_for_kernel(name, src_dir)

    text = re.sub(f'void\\s+\\b{name}\\b',f'void {name}_gpu',text)
    p = re.compile(f'void\\s+\\b{name}_gpu\\b')

    i = p.search(text).start()


    if(i < 0):
      print("\n********")
      print(f"Error: cannot locate user kernel function: {name} - Aborting code generation")
      exit(2)

    i2 = i
    i = text[0:i].rfind('\n') #reverse find
    if i < 0:
      i = 0
    j = text[i:].find('{')
    k = para_parse(text, i+j, '{', '}')
    arg_list = parse_signature(text[i2+len(name):i+j])
    code('__device__')

    new_code = complex_numbers_cuda(text[i:k+2])  # Handle complex numbers with the cuComplex.h CUDA library.
    code(convert_ACC(new_code, arg_typ))
    code('')
    code('')


    ##########################################################################
    #  generate cuda kernel wrapper function
    ##########################################################################

    code(f'__global__ void ops_{name}(')
    for n in range (0, nargs):
      if arg_typ[n] == 'ops_arg_dat':
        code(f'{typs[n]}* __restrict arg{n},')
        if n in needDimList:
          code(f'int arg{n}dim,')
      elif arg_typ[n] == 'ops_arg_gbl':
        if accs[n] == OPS_READ:
          if dims[n].isdigit() and int(dims[n])==1:
            code(f'const {typs[n]} arg{n},')
          else:
            code(f'const {typs[n]}* __restrict arg{n},')
            if n in needDimList:
              code(f'int arg{n}dim,')
        else:
          code(f'{typs[n]}* __restrict arg{n},')
      if restrict[n] or prolong[n]:
        if NDIM == 1:
          code(f'int stride_{n}0,')
        if NDIM == 2:
          code(f'int stride_{n}0, int stride_{n}1,')
        if NDIM == 3:
          code(f'int stride_{n}0, int stride_{n}1, int stride_{n}2,')

      elif arg_typ[n] == 'ops_arg_idx':
        if NDIM==1:
          code('int arg_idx0,')
        elif NDIM==2:
          code('int arg_idx0, int arg_idx1,')
        elif NDIM==3:
          code('int arg_idx0, int arg_idx1, int arg_idx2,')

    if any_prolong:
      if NDIM == 1:
        code('int global_idx0,')
      elif NDIM == 2:
        code('int global_idx0, int global_idx1,')
      elif NDIM == 3:
        code('int global_idx0, int global_idx1, int global_idx2,')
    if NDIM==1:
      code('int size0 ){')
    elif NDIM==2:
      code('int size0,')
      code('int size1 ){')
    elif NDIM==3:
      code('int size0,')
      code('int size1,')
      code('int size2 ){')

    config.depth = config.depth + 2

    #local variable to hold reductions on GPU
    code('')
    for n in range (0, nargs):
      if arg_typ[n] == 'ops_arg_gbl' and accs[n] != OPS_READ:
        code(f'{typs[n]} arg{n}_l[{dims[n]}];')

    # set local variables to 0 if OPS_INC, INF if OPS_MIN, -INF
    for n in range (0, nargs):
      if arg_typ[n] == 'ops_arg_gbl' and accs[n] == OPS_INC:
        code(f'for (int d=0; d<{dims[n]}; d++) arg{n}_l[d] = ZERO_{typs[n]};')
      if arg_typ[n] == 'ops_arg_gbl' and accs[n] == OPS_MIN:
        code(f'for (int d=0; d<{dims[n]}; d++) arg{n}_l[d] = INFINITY_{typs[n]};')
      if arg_typ[n] == 'ops_arg_gbl' and accs[n] == OPS_MAX:
        code(f'for (int d=0; d<{dims[n]}; d++) arg{n}_l[d] = -INFINITY_{typs[n]};')


    code('')
    if NDIM==3:
      code('int idx_z = blockDim.z * blockIdx.z + threadIdx.z;')
      code('int idx_y = blockDim.y * blockIdx.y + threadIdx.y;')
    if NDIM==2:
      code('int idx_y = blockDim.y * blockIdx.y + threadIdx.y;')
    code('int idx_x = blockDim.x * blockIdx.x + threadIdx.x;')
    code('')
    if arg_idx:
      code(f'int arg_idx[{NDIM}];')
      code('arg_idx[0] = arg_idx0+idx_x;')
      if NDIM==2:
        code('arg_idx[1] = arg_idx1+idx_y;')
      if NDIM==3:
        code('arg_idx[1] = arg_idx1+idx_y;')
        code('arg_idx[2] = arg_idx2+idx_z;')

    for n in range (0, nargs):
      if arg_typ[n] == 'ops_arg_dat':
        if restrict[n] == 1:
          n_x = f'idx_x*stride_{n}0'
          n_y = f'idx_y*stride_{n}1'
          n_z = f'idx_z*stride_{n}2'
        elif prolong[n] == 1:
          n_x = f'(idx_x+global_idx0%stride_{n}0)/stride_{n}0'
          n_y = f'(idx_y+global_idx1%stride_{n}1)/stride_{n}1'
          n_z = f'(idx_z+global_idx2%stride_{n}2)/stride_{n}2'
        else:
          n_x = 'idx_x'
          n_y = 'idx_y'
          n_z = 'idx_z'
        
        argdim = str(dims[n])
        if n in needDimList:
          argdim = f'arg{n}dim'
        if NDIM == 1:
          if soa_set:
            code(f'arg{n} += {n_x} * {stride[NDIM*n]};')
          else:
            code(f'arg{n} += {n_x} * {stride[NDIM*n]}*{argdim};')
        elif NDIM == 2:
          if soa_set:
            code(f'arg{n} += {n_x} * {stride[NDIM*n]} + {n_y} * {stride[NDIM*n+1]} * dims_{name}[{n}][0]'+';')
          else:
            code(f'arg{n} += {n_x} * {stride[NDIM*n]}*{argdim} + {n_y} * {stride[NDIM*n+1]}*{argdim} * dims_{name}[{n}][0]'+';')
        elif NDIM==3:
          if soa_set:
            code(f'arg{n} += {n_x} * {stride[NDIM*n]}+ {n_y} * {stride[NDIM*n+1]}* dims_{name}[{n}][0] + {n_z} * {stride[NDIM*n+2]} * dims_{name}[{n}][0] * dims_{name}[{n}][1];')
          else:
            code(f'arg{n} += {n_x} * {stride[NDIM*n]}*{argdim} + {n_y} * {stride[NDIM*n+1]}*{argdim} * dims_{name}[{n}][0] + {n_z} * {stride[NDIM*n+2]}*{argdim} * dims_{name}[{n}][0] * dims_{name}[{n}][1];')

    code('')
    n_per_line = 5
    if NDIM==1:
      IF('idx_x < size0')
    if NDIM==2:
      IF('idx_x < size0 && idx_y < size1')
    elif NDIM==3:
      IF('idx_x < size0 && idx_y < size1 && idx_z < size2')
    text = name+'_gpu('
    for n in range (0, nargs):
      if arg_typ[n] == 'ops_arg_dat':
        dim = ''
        sizelist = ''
        pre = ''
        extradim = 0
        if dims[n].isdigit() and int(dims[n])>1:
            dim = dims[n]+', '
            extradim = 1
        elif not dims[n].isdigit():
            dim = f'arg{n}dim, '
            extradim = 1
        for i in range(1,NDIM):
          sizelist += f'dims_{name}[{n}][{i-1}], '
        if extradim:
          if soa_set:
            sizelist += f'dims_{name}[{n}][{NDIM-1}], '
          else:
            sizelist += '0, '

        if accs[n] == OPS_READ:
            pre = 'const '

        code(f'{pre}ACC<{typs[n]}> argp{n}({dim+sizelist}arg{n});')
    for n in range(0,nargs):
      if arg_typ[n] == 'ops_arg_dat':
          text += f'argp{n}'
      elif arg_typ[n] == 'ops_arg_gbl' and accs[n] == OPS_READ:
        if dims[n].isdigit() and int(dims[n])==1:
          text += f'&arg{n}'
        else:
          text += f'arg{n}'
      elif arg_typ[n] == 'ops_arg_gbl' and accs[n] != OPS_READ:
        text += f'arg{n}_l'
      elif arg_typ[n] == 'ops_arg_idx':
        text += 'arg_idx'


      if nargs != 1 and n != nargs-1:
        if n%n_per_line != 3:
          text += ', '
        else:
          text += ','
      else:
        text += ');'
      if n%n_per_line == 3 and n != nargs-1:
         text += '\n                   '
    code(text)
    ENDIF()

    #reduction across blocks
    if NDIM==1:
      cont = '(blockIdx.x + blockIdx.y*gridDim.x)*'
    if NDIM==2:
      cont = '(blockIdx.x + blockIdx.y*gridDim.x)*'
    elif NDIM==3:
      cont = '(blockIdx.x + blockIdx.y*gridDim.x + blockIdx.z*gridDim.x*gridDim.y)*'
    for n in range (0, nargs):
      if arg_typ[n] == 'ops_arg_gbl' and accs[n] == OPS_INC:
        code(f'for (int d=0; d<{dims[n]}; d++)')
        code(f'  ops_reduction_cuda<OPS_INC>(&arg{n}[d+{cont}{dims[n]}],arg{n}_l[d]);')
      if arg_typ[n] == 'ops_arg_gbl' and accs[n] == OPS_MIN:
        code(f'for (int d=0; d<{dims[n]}; d++)')
        code(f'  ops_reduction_cuda<OPS_MIN>(&arg{n}[d+{cont}{dims[n]}],arg{n}_l[d]);')
      if arg_typ[n] == 'ops_arg_gbl' and accs[n] == OPS_MAX:
        code(f'for (int d=0; d<{dims[n]}; d++)')
        code(f'  ops_reduction_cuda<OPS_MAX>(&arg{n}[d+{cont}{dims[n]}],arg{n}_l[d]);')


    code('')
    config.depth = config.depth - 2
    code('}')


    ##########################################################################
    #  now host stub
    ##########################################################################
    code('')
    comm(' host stub function')
    code('#ifndef OPS_LAZY')
    code(f'void ops_par_loop_{name}(char const *name, ops_block block, int dim, int* range,')
    text = ''
    for n in range (0, nargs):

      text = text +' ops_arg arg'+str(n)
      if nargs != 1 and n != nargs-1:
        text = text +','
      else:
        text = text +') {'
      if n%n_per_line == 3 and n != nargs-1:
         text = text +'\n'
    code(text);
    code('#else')
    code(f'void ops_par_loop_{name}_execute(ops_kernel_descriptor *desc) {{')
    config.depth = 2
    code('int dim = desc->dim;')
    code('#if OPS_MPI')
    code('ops_block block = desc->block;')
    code('#endif')
    code('int *range = desc->range;')

    for n in range (0, nargs):
      code(f'ops_arg arg{n} = desc->args[{n}];')
    code('#endif')

    code('')
    comm('Timing')
    code('double t1,t2,c1,c2;')
    code('');

    text =f'ops_arg args[{nargs}] = {{'
    for n in range (0, nargs):
      text += f' arg{n}'
      if nargs != 1 and n != nargs-1:
        text += ','
      else:
        text += '};\n'
      if n%n_per_line == 5 and n != nargs-1:
        text += '\n                    '
    code(text);
    code('')
    code('#if CHECKPOINTING && !OPS_LAZY')
    code(f'if (!ops_checkpointing_before(args,{nargs},range,{nk})) return;')
    code('#endif')
    code('')

    IF('block->instance->OPS_diags > 1')
    code('#ifndef OPS_LAZY')
    code(f'ops_timing_realloc(block->instance,{nk},"{name}");')
    code(f'block->instance->OPS_kernels[{nk}].count++;')
    code('ops_timers_core(&c1,&t1);')
    code('#endif')
    ENDIF()

    code('')
    comm('compute locally allocated range for the sub-block')

    code(f'int start[{NDIM}];')
    code(f'int end[{NDIM}];')

    code('#if OPS_MPI && !OPS_LAZY')
    code('sub_block_list sb = OPS_sub_block_list[block->index];')
    code('#endif //OPS_MPI')

    code('')
    if not arg_idx:
      code('#ifdef OPS_MPI')
    code(f'int arg_idx[{NDIM}];')
    if not arg_idx:
      code('#endif')


    code('#if defined(OPS_LAZY) || !defined(OPS_MPI)')
    FOR('n','0',str(NDIM))
    code('start[n] = range[2*n];end[n] = range[2*n+1];')
    ENDFOR()
    code('#else')
    code(f'if (compute_ranges(args, {nargs},block, range, start, end, arg_idx) < 0) return;')
    code('#endif')

    code('')
    if arg_idx or MULTI_GRID:
      code('#if defined(OPS_MPI)')
      code('#if defined(OPS_LAZY)')
      code('sub_block_list sb = OPS_sub_block_list[block->index];')
      for n in range (0,NDIM):
        code(f'arg_idx[{n}] = sb->decomp_disp[{n}]+start[{n}];')
      code('#endif')
      code('#else //OPS_MPI')
      for n in range (0,NDIM):
        code(f'arg_idx[{n}] = start[{n}];')
      code('#endif //OPS_MPI')


    if MULTI_GRID:
      code(f'int global_idx[{NDIM}];')
      code('#ifdef OPS_MPI')
      for n in range (0,NDIM):
        code(f'global_idx[{n}] = arg_idx[{n}];')
      code('#else')
      for n in range (0,NDIM):
        code(f'global_idx[{n}] = start[{n}];')
      code('#endif')
      code('')

    for n in range (0, nargs):
      if arg_typ[n] == 'ops_arg_dat':
        code(f'int xdim{n} = args[{n}].dat->size[0];')
        if NDIM>2 or (NDIM==2 and soa_set):
          code(f'int ydim{n} = args[{n}].dat->size[1];')
        if NDIM>3 or (NDIM==3 and soa_set):
          code(f'int zdim{n} = args[{n}].dat->size[2];')
    code('')

    condition = ''
    for n in range (0, nargs):
      if arg_typ[n] == 'ops_arg_dat':
        condition += f'xdim{n} != dims_{name}_h[{n}][0] || '
        if NDIM>2 or (NDIM==2 and soa_set):
          condition += f'ydim{n} != dims_{name}_h[{n}][1] || '
        if NDIM>3 or (NDIM==3 and soa_set):
          condition += f'zdim{n} != dims_{name}_h[{n}][2] || '
    condition = condition[:-4]
    IF(condition)

    for n in range (0, nargs):
      if arg_typ[n] == 'ops_arg_dat':
        code(f'dims_{name}_h[{n}][0] = xdim{n};')
        if NDIM>2 or (NDIM==2 and soa_set):
          code(f'dims_{name}_h[{n}][1] = ydim{n};')
        if NDIM>3 or (NDIM==3 and soa_set):
          code(f'dims_{name}_h[{n}][2] = zdim{n};')
    code(f'cutilSafeCall(block->instance->ostream(), cudaMemcpyToSymbol( dims_{name}, dims_{name}_h, sizeof(dims_{name})));')
    ENDIF()

    code('')

    #setup reduction variables
    code('')
    if reduct and not arg_idx:
      code('#if defined(OPS_LAZY) && !defined(OPS_MPI)')
      code('ops_block block = desc->block;')
      code('#endif')
    for n in range (0, nargs):
        if arg_typ[n] == 'ops_arg_gbl' and (accs[n] != OPS_READ or (accs[n] == OPS_READ and (not dims[n].isdigit() or int(dims[n])>1))):
          if (accs[n] == OPS_READ):
            code(f'{typs[n]} *arg{n}h = ({typs[n]} *)arg{n}.data;')
          else:
            code('#ifdef OPS_MPI')
            code(f'{typs[n]} *arg{n}h = ({typs[n]} *)(((ops_reduction)args[{n}].data)->data + ((ops_reduction)args[{n}].data)->size * block->index);')
            code('#else')
            code(f'{typs[n]} *arg{n}h = ({typs[n]} *)(((ops_reduction)args[{n}].data)->data);')
            code('#endif')

    code('')
    code('int x_size = MAX(0,end[0]-start[0]);')
    if NDIM==2:
      code('int y_size = MAX(0,end[1]-start[1]);')
    if NDIM==3:
      code('int y_size = MAX(0,end[1]-start[1]);')
      code('int z_size = MAX(0,end[2]-start[2]);')
    code('')

    #set up CUDA grid and thread blocks for kernel call
    if NDIM==1:
      code('dim3 grid( (x_size-1)/block->instance->OPS_block_size_x+ 1, 1, 1);')
    if NDIM==2:
      code('dim3 grid( (x_size-1)/block->instance->OPS_block_size_x+ 1, (y_size-1)/block->instance->OPS_block_size_y + 1, 1);')
    if NDIM==3:
      code('dim3 grid( (x_size-1)/block->instance->OPS_block_size_x+ 1, (y_size-1)/block->instance->OPS_block_size_y + 1, (z_size-1)/block->instance->OPS_block_size_z +1);')

    if NDIM>1:
      code('dim3 tblock(block->instance->OPS_block_size_x,block->instance->OPS_block_size_y,block->instance->OPS_block_size_z);')
    else:
      code('dim3 tblock(block->instance->OPS_block_size_x,1,1);')

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
      if NDIM==1:
        code('int nblocks = ((x_size-1)/block->instance->OPS_block_size_x+ 1);')
      elif NDIM==2:
        code('int nblocks = ((x_size-1)/block->instance->OPS_block_size_x+ 1)*((y_size-1)/block->instance->OPS_block_size_y + 1);')
      elif NDIM==3:
        code('int nblocks = ((x_size-1)/block->instance->OPS_block_size_x+ 1)*((y_size-1)/block->instance->OPS_block_size_y + 1)*((z_size-1)/block->instance->OPS_block_size_z +1);')
      code('int maxblocks = nblocks;')
      code('int reduct_bytes = 0;')
      code('size_t reduct_size = 0;')
      code('')

    if GBL_READ == True and GBL_READ_MDIM == True:
      code('int consts_bytes = 0;')
      code('')

    for n in range (0, nargs):
      if arg_typ[n] == 'ops_arg_gbl':
        if accs[n] == OPS_READ and (not dims[n].isdigit() or int(dims[n])>1):
          code(f'consts_bytes += ROUND_UP(arg{n}.dim*sizeof({typs[n]}));')
        elif accs[n] != OPS_READ:
          code(f'reduct_bytes += ROUND_UP(maxblocks*{dims[n]}*sizeof({typs[n]}));')
          code(f'reduct_size = MAX(reduct_size,sizeof({typs[n]})*{dims[n]});')
    code('')

    if GBL_READ == True and GBL_READ_MDIM == True:
      code('reallocConstArrays(block->instance,consts_bytes);')
      code('consts_bytes = 0;')
    if GBL_INC == True or GBL_MIN == True or GBL_MAX == True or GBL_WRITE == True:
      code('reallocReductArrays(block->instance,reduct_bytes);')
      code('reduct_bytes = 0;')
      code('')

    for n in range (0, nargs):
      if arg_typ[n] == 'ops_arg_gbl' and accs[n] != OPS_READ:
        code(f'arg{n}.data = block->instance->OPS_reduct_h + reduct_bytes;')
        code(f'arg{n}.data_d = block->instance->OPS_reduct_d + reduct_bytes;')
        code('for (int b=0; b<maxblocks; b++)')
        if accs[n] == OPS_INC:
          code(f'for (int d=0; d<{dims[n]}; d++) (({typs[n]} *)arg{n}.data)[d+b*{dims[n]}] = ZERO_{typs[n]};')
        if accs[n] == OPS_MAX:
          code(f'for (int d=0; d<{dims[n]}; d++) (({typs[n]} *)arg{n}.data)[d+b*{dims[n]}] = -INFINITY_{typs[n]};')
        if accs[n] == OPS_MIN:
          code(f'for (int d=0; d<{dims[n]}; d++) (({typs[n]} *)arg{n}.data)[d+b*{dims[n]}] = INFINITY_{typs[n]};')
        code(f'reduct_bytes += ROUND_UP(maxblocks*{dims[n]}*sizeof({typs[n]}));')
        code('')

    code('')

    for n in range (0, nargs):
      if arg_typ[n] == 'ops_arg_gbl':
        if accs[n] == OPS_READ and (not dims[n].isdigit() or int(dims[n])>1):
          code(f'arg{n}.data = block->instance->OPS_consts_h + consts_bytes;')
          code(f'arg{n}.data_d = block->instance->OPS_consts_d + consts_bytes;')
          code(f'for (int d=0; d<arg{n}.dim; d++) (({typs[n]} *)arg{n}.data)[d] = arg{n}h[d];')
          code(f'consts_bytes += ROUND_UP(arg{n}.dim*sizeof({typs[n]}));')
    if GBL_READ == True and GBL_READ_MDIM == True:
      code('mvConstArraysToDevice(block->instance,consts_bytes);')

    if GBL_INC == True or GBL_MIN == True or GBL_MAX == True or GBL_WRITE == True:
      code('mvReductArraysToDevice(block->instance,reduct_bytes);')

    for n in range (0, nargs):
      if arg_typ[n] == 'ops_arg_dat':
        code(f'long long int dat{n} = (block->instance->OPS_soa ? args[{n}].dat->type_size : args[{n}].dat->elem_size);')

    code('')
    code(f'char *p_a[{nargs}];')

    #some custom logic for multigrid
    if MULTI_GRID:
      for n in range (0, nargs):
        if prolong[n] == 1 or restrict[n] == 1:
          comm('This arg has a prolong stencil - so create different ranges')
          code(f'int start_{n}[{NDIM}]; int end_{n}[{NDIM}]; int stride_{n}[{NDIM}];int d_size_{n}[{NDIM}];')
          code('#ifdef OPS_MPI')
          FOR('n','0',str(NDIM))
          code(f'sub_dat *sd{n} = OPS_sub_dat_list[args[{n}].dat->index];')
          code(f'stride_{n}[n] = args[{n}].stencil->mgrid_stride[n];')
          code(f'd_size_{n}[n] = args[{n}].dat->d_m[n] + sd{n}->decomp_size[n] - args[{n}].dat->d_p[n];')
          if restrict[n] == 1:
            code(f'start_{n}[n] = global_idx[n]*stride_{n}[n] - sd{n}->decomp_disp[n] + args[{n}].dat->d_m[n];')
          else:
            code(f'start_{n}[n] = global_idx[n]/stride_{n}[n] - sd{n}->decomp_disp[n] + args[{n}].dat->d_m[n];')
          code(f'end_{n}[n] = start_{n}[n] + d_size_{n}[n];')
          ENDFOR()
          code('#else')
          FOR('n','0',str(NDIM))
          code(f'stride_{n}[n] = args[{n}].stencil->mgrid_stride[n];')
          code(f'd_size_{n}[n] = args[{n}].dat->d_m[n] + args[{n}].dat->size[n] - args[{n}].dat->d_p[n];')
          if restrict[n] == 1:
            code(f'start_{n}[n] = global_idx[n]*stride_{n}[n];')
          else:
            code(f'start_{n}[n] = global_idx[n]/stride_{n}[n];')
          code(f'end_{n}[n] = start_{n}[n] + d_size_{n}[n];')
          ENDFOR()
          code('#endif')


    comm('')
    comm('set up initial pointers')
    for n in range (0, nargs):
      if arg_typ[n] == 'ops_arg_dat':
        if prolong[n] == 1 or restrict[n] == 1:
          starttext = 'start_'+str(n)
        else:
          starttext = 'start'
        code(f'long long int base{n} = args[{n}].dat->base_offset + ')
        code(f'         dat{n} * 1 * ({starttext}[0] * args[{n}].stencil->stride[0]);')
        for d in range (1, NDIM):
          line = f'base{n} = base{n}+ dat{n} *\n'
          for d2 in range (0,d):
            line += config.depth*' '+f'  args[{n}].dat->size[{d2}] *\n'
          code(line[:-1])
          code(f'  ({starttext}[{d}] * args[{n}].stencil->stride[{d}]);')

        code(f'p_a[{n}] = (char *)args[{n}].data_d + base{n};')
        code('')

    #halo exchange
    code('')
    code('#ifndef OPS_LAZY')
    code(f'ops_H_D_exchanges_device(args, {nargs});')
    code(f'ops_halo_exchanges(args,{nargs},range);')
    code('#endif')
    code('')
    IF('block->instance->OPS_diags > 1')
    code('ops_timers_core(&c2,&t2);')
    code(f'block->instance->OPS_kernels[{nk}].mpi_time += t2-t1;')
    ENDIF()
    code('')


    #set up shared memory for reduction
    if GBL_INC == True or GBL_MIN == True or GBL_MAX == True or GBL_WRITE == True:
       code('size_t nshared = 0;')
       code('int nthread = block->instance->OPS_block_size_x*block->instance->OPS_block_size_y*block->instance->OPS_block_size_z;')
       code('')
    for n in range (0, nargs):
      if arg_typ[n] == 'ops_arg_gbl' and accs[n] != OPS_READ:
        code(f'nshared = MAX(nshared,sizeof({typs[n]})*{dims[n]});')
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
    n_per_line = 2
    if GBL_INC == True or GBL_MIN == True or GBL_MAX == True or GBL_WRITE == True:
      text = f'ops_'+name+'<<<grid, tblock, nshared >>> ( '
    else:
      text = f'ops_{name}<<<grid, tblock >>> ( '
    for n in range (0, nargs):
      if arg_typ[n] == 'ops_arg_dat':
        text += f' ({typs[n]} *)p_a[{n}],'
        if n in needDimList:
          text += f' arg{n}.dim,'
      elif arg_typ[n] == 'ops_arg_gbl':
        if dims[n].isdigit() and int(dims[n])==1 and accs[n]==OPS_READ:
          text += f' *({typs[n]} *)arg{n}.data,'
        else:
          text += f' ({typs[n]} *)arg{n}.data_d,'
          if n in needDimList and accs[n] != OP_READ:
            text += ' arg{n}.dim,'
      elif arg_typ[n] == 'ops_arg_idx':
        if NDIM==1:
          text += ' arg_idx[0],'
        if NDIM==2:
          text += ' arg_idx[0], arg_idx[1],'
        elif NDIM==3:
          text += ' arg_idx[0], arg_idx[1], arg_idx[2],'
      if restrict[n] or prolong[n]:
        if NDIM==1:
          text += f'stride_{n}[0],'
        if NDIM==2:
          text += f'stride_{n}[0],stride_{n}[1],'
        if NDIM==3:
          text += f'stride_{n}[0],stride_{n}[1],stride_{n}[2],'

      if n%n_per_line == 1 and n != nargs-1:
        text = text +'\n        '
    if any_prolong:
      if NDIM==1:
        text = text + 'global_idx[0],'
      elif NDIM==2:
        text = text + 'global_idx[0], global_idx[1],'
      elif NDIM==3:
        text = text + 'global_idx[0], global_idx[1], global_idx[2],'

    if NDIM==1:
      text = text +'x_size);'
    if NDIM==2:
      text = text +'x_size, y_size);'
    elif NDIM==3:
      text = text +'x_size, y_size, z_size);'
    code(text);
    config.depth = config.depth - 2

    code('')
    code('cutilSafeCall(block->instance->ostream(), cudaGetLastError());')
    code('')

    #
    # Complete Reduction Operation by moving data onto host
    # and reducing over blocks
    #
    if GBL_INC == True or GBL_MIN == True or GBL_MAX == True or GBL_WRITE == True:
      code('mvReductArraysToHost(block->instance,reduct_bytes);')

    for n in range (0, nargs):
      if arg_typ[n] == 'ops_arg_gbl' and accs[n] != OPS_READ:
        FOR('b','0','maxblocks')
        FOR('d','0',str(dims[n]))
        if accs[n] == OPS_INC:
          code(f'arg{n}h[d] = arg{n}h[d] + (({typs[n]} *)arg{n}.data)[d+b*{dims[n]}];')
        elif accs[n] == OPS_MAX:
          code(f'arg{n}h[d] = MAX(arg{n}h[d],(({typs[n]} *)arg{n}.data)[d+b*{dims[n]}]);')
        elif accs[n] == OPS_MIN:
          code(f'arg{n}h[d] = MIN(arg{n}h[d],(({typs[n]} *)arg{n}.data)[d+b*{dims[n]}]);')
        ENDFOR()
        ENDFOR()
        code(f'arg{n}.data = (char *)arg{n}h;')
        code('')

    IF('block->instance->OPS_diags>1')
    code('cutilSafeCall(block->instance->ostream(), cudaDeviceSynchronize());')
    code('ops_timers_core(&c1,&t1);')
    code(f'block->instance->OPS_kernels[{nk}].time += t1-t2;')
    ENDIF()
    code('')


    code('#ifndef OPS_LAZY')
    code(f'ops_set_dirtybit_device(args, {nargs});')
    for n in range (0, nargs):
      if arg_typ[n] == 'ops_arg_dat' and (accs[n] == OPS_WRITE or accs[n] == OPS_RW or accs[n] == OPS_INC):
        code(f'ops_set_halo_dirtybit3(&args[{n}],range);')
    code('#endif')

    code('')
    IF('block->instance->OPS_diags > 1')
    comm('Update kernel record')
    code('ops_timers_core(&c2,&t2);')
    code(f'block->instance->OPS_kernels[{nk}].mpi_time += t2-t1;')
    for n in range (0, nargs):
      if arg_typ[n] == 'ops_arg_dat':
        code(f'block->instance->OPS_kernels[{nk}].transfer += ops_compute_transfer(dim, start, end, &arg{n});')
    ENDIF()
    config.depth = config.depth - 2
    code('}')
    code('')
    code('#ifdef OPS_LAZY')
    code(f'void ops_par_loop_{name}(char const *name, ops_block block, int dim, int* range,')
    text = ''
    for n in range (0, nargs):

      text = text +' ops_arg arg'+str(n)
      if nargs != 1 and n != nargs-1:
        text = text +','
      else:
        text = text +') {'
      if n%n_per_line == 3 and n != nargs-1:
         text = text +'\n'
    code(text);
    config.depth = 2

    text = 'ops_arg args[] = {'
    for n in range (0, nargs):
        text = text +' arg'+str(n)
        if nargs != 1 and n != nargs-1:
            text += ','
        else:
            text += ' };'

    code(text)

#    code('ops_kernel_descriptor *desc = ops_populate_kernel_descriptor(name, args, {nargs}, {nk}, dim, 0, range, block, ops_par_loop_{name}_execute);')
    text = 'ops_kernel_descriptor *desc = ops_populate_kernel_descriptor(name, args, '
    text = text + f'{nargs}, '
    text = text + f'{nk}, '
    text = text + 'dim, 1, range, block, '
    text = text + f'ops_par_loop_{name}_execute'
    text = text + ');'
    code(text)

    code('ops_enqueue_kernel(desc);')
    config.depth = 0
    code('}')
    code('#endif')


    ##########################################################################
    #  output individual kernel file
    ##########################################################################
    util.write_text_to_file(f"./CUDA/{name}_cuda_kernel.cu")

  # end of main kernel call loop

  ##########################################################################
  #  output one master kernel file
  ##########################################################################

  comm('header')
  code('#include <cuda.h>')
  code('#define OPS_API 2')
  if NDIM==1:
    code('#define OPS_1D')
  if NDIM==2:
    code('#define OPS_2D')
  if NDIM==3:
    code('#define OPS_3D')
  if soa_set:
    code('#define OPS_SOA')
  code('#include "ops_lib_core.h"')
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
      code(f"__constant__ {consts[nc]['type']} "+(str(consts[nc]['name']).replace('"','')).strip()+';')
    else:
      if consts[nc]['dim'].isdigit() and int(consts[nc]['dim']) > 0:
        num = str(consts[nc]['dim'])
        code(f"__constant__ {consts[nc]['type']} "+(str(consts[nc]['name']).replace('"','')).strip()+f'[{num}];')
      else:
        code(f"__constant__ {consts[nc]['type']} *"+(str(consts[nc]['name']).replace('"','')).strip()+';')



  code('')
  code('void ops_init_backend() {}')
  code('')
  code('void ops_decl_const_char(OPS_instance *instance, int dim, char const *type,')
  code('int size, char *dat, char const *name){')
  config.depth = config.depth + 2
  code('ops_execute(instance);')

  for nc in range(0,len(consts)):
    IF('!strcmp(name,"'+(str(consts[nc]['name']).replace('"','')).strip()+'")')
    if consts[nc]['dim'].isdigit():
      code('cutilSafeCall(instance->ostream(),cudaMemcpyToSymbol('+(str(consts[nc]['name']).replace('"','')).strip()+', dat, dim*size));')
    else:
      code('char *temp; cutilSafeCall(instance->ostream(),cudaMalloc((void**)&temp,dim*size));')
      code('cutilSafeCall(instance->ostream(),cudaMemcpy(temp,dat,dim*size,cudaMemcpyHostToDevice));')
      code('cutilSafeCall(instance->ostream(),cudaMemcpyToSymbol('+(str(consts[nc]['name']).replace('"','')).strip()+', &temp, sizeof(char *)));')
    ENDIF()
    code('else')

  code('{')
  config.depth = config.depth + 2
  code('throw OPSException(OPS_RUNTIME_ERROR, "error: unknown const name");')
  ENDIF()

  config.depth = config.depth - 2
  code('}')
  code('')

  code('')
  comm('user kernel files')

  for kernel_name in map(lambda kernel: kernel['name'], kernels):
      code(f"#include \"{kernel_name}_cuda_kernel.cu\"")

  util.write_text_to_file(f"./CUDA/{master_basename[0]}_kernels.cu")
