
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
import errno
import os
import glob

import util
import config
from config import OPS_READ, OPS_WRITE, OPS_RW, OPS_INC, OPS_MAX, OPS_MIN, OPS_accs_labels

para_parse = util.para_parse
comment_remover = util.comment_remover
remove_trailing_w_space = util.remove_trailing_w_space
parse_signature = util.parse_signature
complex_numbers_cuda = util.complex_numbers_cuda
check_accs = util.check_accs
mult = util.mult
convert_ACC = util.convert_ACC

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

    config.file_text = ''
    config.depth = 0
    n_per_line = 4

    i = name.find('kernel')
    name2 = name[0:i-1]
    #print name2

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
    code('__constant__ int dims_'+name+' ['+str(nargs)+']['+str(num_dims)+'];')
    code('static int dims_'+name+'_h ['+str(nargs)+']['+str(num_dims)+'] = {0};')
    code('')

    ##########################################################################
    #  generate header
    ##########################################################################

    comm('user function')

    found = 0
    for files in glob.glob( os.path.join(src_dir, "*.h") ):
      f = open( files, 'r' )
      for line in f:
        if name in line:
          file_name = f.name
          found = 1;
          break
      if found == 1:
        break;

    if found == 0:
      print(("COUND NOT FIND KERNEL", name))

    fid = open(file_name, 'r')
    text = fid.read()

    fid.close()
    text = comment_remover(text)
    text = remove_trailing_w_space(text)

    text = re.sub('void\\s+\\b'+name+'\\b','void '+name+'_gpu',text)
    p = re.compile('void\\s+\\b'+name+'_gpu\\b')

    i = p.search(text).start()


    if(i < 0):
      print("\n********")
      print(("Error: cannot locate user kernel function: "+name+" - Aborting code generation"))
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

    code('__global__ void ops_'+name+'(')
    for n in range (0, nargs):
      if arg_typ[n] == 'ops_arg_dat':
        code(typs[n]+'* __restrict arg'+str(n)+',')
        if n in needDimList:
          code('int arg'+str(n)+'dim,')
      elif arg_typ[n] == 'ops_arg_gbl':
        if accs[n] == OPS_READ:
          if dims[n].isdigit() and int(dims[n])==1:
            code('const '+typs[n]+' arg'+str(n)+',')
          else:
            code('const '+typs[n]+'* __restrict arg'+str(n)+',')
            if n in needDimList:
              code('int arg'+str(n)+'dim,')
        else:
          code(typs[n]+'* __restrict arg'+str(n)+',')
      if restrict[n] or prolong[n]:
        if NDIM == 1:
          code('int stride_'+str(n)+'0,')
        if NDIM == 2:
          code('int stride_'+str(n)+'0, int stride_'+str(n)+'1,')
        if NDIM == 3:
          code('int stride_'+str(n)+'0, int stride_'+str(n)+'1, int stride_'+str(n)+'2,')

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
        code(typs[n]+' arg'+str(n)+'_l['+str(dims[n])+'];')

    # set local variables to 0 if OPS_INC, INF if OPS_MIN, -INF
    for n in range (0, nargs):
      if arg_typ[n] == 'ops_arg_gbl' and accs[n] == OPS_INC:
        code('for (int d=0; d<'+str(dims[n])+'; d++) arg'+str(n)+'_l[d] = ZERO_'+typs[n]+';')
      if arg_typ[n] == 'ops_arg_gbl' and accs[n] == OPS_MIN:
        code('for (int d=0; d<'+str(dims[n])+'; d++) arg'+str(n)+'_l[d] = INFINITY_'+typs[n]+';')
      if arg_typ[n] == 'ops_arg_gbl' and accs[n] == OPS_MAX:
        code('for (int d=0; d<'+str(dims[n])+'; d++) arg'+str(n)+'_l[d] = -INFINITY_'+typs[n]+';')


    code('')
    if NDIM==3:
      code('int idx_z = blockDim.z * blockIdx.z + threadIdx.z;')
      code('int idx_y = blockDim.y * blockIdx.y + threadIdx.y;')
    if NDIM==2:
      code('int idx_y = blockDim.y * blockIdx.y + threadIdx.y;')
    code('int idx_x = blockDim.x * blockIdx.x + threadIdx.x;')
    code('')
    if arg_idx:
      code('int arg_idx['+str(NDIM)+'];')
      code('arg_idx[0] = arg_idx0+idx_x;')
      if NDIM==2:
        code('arg_idx[1] = arg_idx1+idx_y;')
      if NDIM==3:
        code('arg_idx[1] = arg_idx1+idx_y;')
        code('arg_idx[2] = arg_idx2+idx_z;')

    for n in range (0, nargs):
      if arg_typ[n] == 'ops_arg_dat':
        if restrict[n] == 1:
          n_x = 'idx_x*stride_'+str(n)+'0'
          n_y = 'idx_y*stride_'+str(n)+'1'
          n_z = 'idx_z*stride_'+str(n)+'2'
        elif prolong[n] == 1:
          n_x = '(idx_x+global_idx0%stride_'+str(n)+'0)/stride_'+str(n)+'0'
          n_y = '(idx_y+global_idx1%stride_'+str(n)+'1)/stride_'+str(n)+'1'
          n_z = '(idx_z+global_idx2%stride_'+str(n)+'2)/stride_'+str(n)+'2'
        else:
          n_x = 'idx_x'
          n_y = 'idx_y'
          n_z = 'idx_z'
        
        argdim = str(dims[n])
        if n in needDimList:
          argdim = 'arg'+str(n)+'dim'
        if NDIM == 1:
          if soa_set:
            code('arg'+str(n)+' += '+n_x+' * '+str(stride[NDIM*n])+';')
          else:
            code('arg'+str(n)+' += '+n_x+' * '+str(stride[NDIM*n])+'*'+argdim+';')
        elif NDIM == 2:
          if soa_set:
            code('arg'+str(n)+' += '+n_x+' * '+str(stride[NDIM*n])+' + '+n_y+' * '+str(stride[NDIM*n+1])+' * dims_'+name+'['+str(n)+'][0]'+';')
          else:
            code('arg'+str(n)+' += '+n_x+' * '+str(stride[NDIM*n])+'*'+argdim+' + '+n_y+' * '+str(stride[NDIM*n+1])+'*'+argdim+' * dims_'+name+'['+str(n)+'][0]'+';')
        elif NDIM==3:
          if soa_set:
            code('arg'+str(n)+' += '+n_x+' * '+str(stride[NDIM*n])+'+ '+n_y+' * '+str(stride[NDIM*n+1])+'* dims_'+name+'['+str(n)+'][0]'+' + '+n_z+' * '+str(stride[NDIM*n+2])+' * dims_'+name+'['+str(n)+'][0]'+' * dims_'+name+'['+str(n)+'][1]'+';')
          else:
            code('arg'+str(n)+' += '+n_x+' * '+str(stride[NDIM*n])+'*'+argdim+' + '+n_y+' * '+str(stride[NDIM*n+1])+'*'+argdim+' * dims_'+name+'['+str(n)+'][0]'+' + '+n_z+' * '+str(stride[NDIM*n+2])+'*'+argdim+' * dims_'+name+'['+str(n)+'][0]'+' * dims_'+name+'['+str(n)+'][1]'+';')

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
            dim = 'arg'+str(n)+'dim, '
            extradim = 1
        for i in range(1,NDIM):
          sizelist = sizelist + 'dims_'+name+'['+str(n)+']['+str(i-1)+'], '
        if extradim:
          if soa_set:
            sizelist = sizelist + 'dims_'+name+'['+str(n)+']['+str(NDIM-1)+'], '
          else:
            sizelist = sizelist + '0, '

        if accs[n] == OPS_READ:
            pre = 'const '

        code(pre+'ACC<'+typs[n]+'> argp'+str(n)+'('+dim+sizelist+'arg'+str(n)+');')
    for n in range(0,nargs):
      if arg_typ[n] == 'ops_arg_dat':
          text = text +'argp'+str(n)
      elif arg_typ[n] == 'ops_arg_gbl' and accs[n] == OPS_READ:
        if dims[n].isdigit() and int(dims[n])==1:
          text = text +'&arg'+str(n)
        else:
          text = text +'arg'+str(n)
      elif arg_typ[n] == 'ops_arg_gbl' and accs[n] != OPS_READ:
        text = text +'arg'+str(n)+'_l'
      elif arg_typ[n] == 'ops_arg_idx':
        text = text +'arg_idx'


      if nargs != 1 and n != nargs-1:
        if n%n_per_line != 3:
          text = text +', '
        else:
          text = text +','
      else:
        text = text +');'
      if n%n_per_line == 3 and n != nargs-1:
         text = text +'\n                   '
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
        code('for (int d=0; d<'+str(dims[n])+'; d++)')
        code('  ops_reduction_cuda<OPS_INC>(&arg'+str(n)+'[d+'+cont+str(dims[n])+'],arg'+str(n)+'_l[d]);')
      if arg_typ[n] == 'ops_arg_gbl' and accs[n] == OPS_MIN:
        code('for (int d=0; d<'+str(dims[n])+'; d++)')
        code('  ops_reduction_cuda<OPS_MIN>(&arg'+str(n)+'[d+'+cont+str(dims[n])+'],arg'+str(n)+'_l[d]);')
      if arg_typ[n] == 'ops_arg_gbl' and accs[n] == OPS_MAX:
        code('for (int d=0; d<'+str(dims[n])+'; d++)')
        code('  ops_reduction_cuda<OPS_MAX>(&arg'+str(n)+'[d+'+cont+str(dims[n])+'],arg'+str(n)+'_l[d]);')


    code('')
    config.depth = config.depth - 2
    code('}')


    ##########################################################################
    #  now host stub
    ##########################################################################
    code('')
    comm(' host stub function')
    code('#ifndef OPS_LAZY')
    code('void ops_par_loop_'+name+'(char const *name, ops_block block, int dim, int* range,')
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
    code('void ops_par_loop_'+name+'_execute(ops_kernel_descriptor *desc) {')
    config.depth = 2
    #code('char const *name = "'+name+'";')
    code('int dim = desc->dim;')
    code('#if OPS_MPI')
    code('ops_block block = desc->block;')
    code('#endif')
    code('int *range = desc->range;')

    for n in range (0, nargs):
      code('ops_arg arg'+str(n)+' = desc->args['+str(n)+'];')
    code('#endif')

    code('')
    comm('Timing')
    code('double t1,t2,c1,c2;')
    code('');

    text ='ops_arg args['+str(nargs)+'] = {'
    for n in range (0, nargs):
      text = text +' arg'+str(n)
      if nargs != 1 and n != nargs-1:
        text = text +','
      else:
        text = text +'};\n'
      if n%n_per_line == 5 and n != nargs-1:
        text = text +'\n                    '
    code(text);
    code('')
    code('#if CHECKPOINTING && !OPS_LAZY')
    code('if (!ops_checkpointing_before(args,'+str(nargs)+',range,'+str(nk)+')) return;')
    code('#endif')
    code('')

    IF('block->instance->OPS_diags > 1')
    code('ops_timing_realloc(block->instance,'+str(nk)+',"'+name+'");')
    code('block->instance->OPS_kernels['+str(nk)+'].count++;')
    code('ops_timers_core(&c1,&t1);')
    ENDIF()

    code('')
    comm('compute locally allocated range for the sub-block')

    code('int start['+str(NDIM)+'];')
    code('int end['+str(NDIM)+'];')

    code('#if OPS_MPI && !OPS_LAZY')
    code('sub_block_list sb = OPS_sub_block_list[block->index];')
    code('#endif //OPS_MPI')

    code('')
    if not arg_idx:
      code('#ifdef OPS_MPI')
    code('int arg_idx['+str(NDIM)+'];')
    if not arg_idx:
      code('#endif')


    code('#if defined(OPS_LAZY) || !defined(OPS_MPI)')
    FOR('n','0',str(NDIM))
    code('start[n] = range[2*n];end[n] = range[2*n+1];')
    ENDFOR()
    code('#else')
    code('if (compute_ranges(args, '+str(nargs)+',block, range, start, end, arg_idx) < 0) return;')
    code('#endif')

    code('')
    if arg_idx or MULTI_GRID:
      code('#if defined(OPS_MPI)')
      code('#if defined(OPS_LAZY)')
      code('sub_block_list sb = OPS_sub_block_list[block->index];')
      for n in range (0,NDIM):
        code('arg_idx['+str(n)+'] = sb->decomp_disp['+str(n)+']+start['+str(n)+'];')
      code('#endif')
      code('#else //OPS_MPI')
      for n in range (0,NDIM):
        code('arg_idx['+str(n)+'] = start['+str(n)+'];')
      code('#endif //OPS_MPI')


    if MULTI_GRID:
      code('int global_idx['+str(NDIM)+'];')
      code('#ifdef OPS_MPI')
      for n in range (0,NDIM):
        code('global_idx['+str(n)+'] = arg_idx['+str(n)+'];')
      code('#else')
      for n in range (0,NDIM):
        code('global_idx['+str(n)+'] = start['+str(n)+'];')
      code('#endif')
      code('')

    for n in range (0, nargs):
      if arg_typ[n] == 'ops_arg_dat':
        code('int xdim'+str(n)+' = args['+str(n)+'].dat->size[0];')
        if NDIM>2 or (NDIM==2 and soa_set):
          code('int ydim'+str(n)+' = args['+str(n)+'].dat->size[1];')
        if NDIM>3 or (NDIM==3 and soa_set):
          code('int zdim'+str(n)+' = args['+str(n)+'].dat->size[2];')
    code('')

    condition = ''
    for n in range (0, nargs):
      if arg_typ[n] == 'ops_arg_dat':
        condition = condition + 'xdim'+str(n)+' != dims_'+name+'_h['+str(n)+'][0] || '
        if NDIM>2 or (NDIM==2 and soa_set):
          condition = condition + 'ydim'+str(n)+' != dims_'+name+'_h['+str(n)+'][1] || '
        if NDIM>3 or (NDIM==3 and soa_set):
          condition = condition + 'zdim'+str(n)+' != dims_'+name+'_h['+str(n)+'][2] || '
    condition = condition[:-4]
    IF(condition)

    #    for n in range (0, nargs):
    #      if arg_typ[n] == 'ops_arg_dat':
    #        code('cudaMemcpyToSymbol( dims_'+name+'['+str(n)+'][0]'+', &xdim'+str(n)+', sizeof(int) );')
    #        code('dims_'+name+'_h['+str(n)+'][0] = xdim'+str(n)+';')
    #        if NDIM>2 or (NDIM==2 and soa_set):
    #          code('cudaMemcpyToSymbol( dims_'+name+'['+str(n)+'][1]'+', &ydim'+str(n)+', sizeof(int) );')
    #          code('dims_'+name+'_h['+str(n)+'][1] = ydim'+str(n)+';')
    #        if NDIM>3 or (NDIM==3 and soa_set):
    #          code('cudaMemcpyToSymbol( dims_'+name+'['+str(n)+'][2]'+', &zdim'+str(n)+', sizeof(int) );')
    #          code('dims_'+name+'_h['+str(n)+'][2] = zdim'+str(n)+';')
    for n in range (0, nargs):
      if arg_typ[n] == 'ops_arg_dat':
        code('dims_'+name+'_h['+str(n)+'][0] = xdim'+str(n)+';')
        if NDIM>2 or (NDIM==2 and soa_set):
          code('dims_'+name+'_h['+str(n)+'][1] = ydim'+str(n)+';')
        if NDIM>3 or (NDIM==3 and soa_set):
          code('dims_'+name+'_h['+str(n)+'][2] = zdim'+str(n)+';')
    code('cutilSafeCall(block->instance->ostream(), cudaMemcpyToSymbol( dims_'+name+', dims_'+name+'_h, sizeof(dims_'+name+')));')
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
            code(''+typs[n]+' *arg'+str(n)+'h = ('+typs[n]+' *)arg'+str(n)+'.data;')
          else:
            code('#ifdef OPS_MPI')
            code(typs[n]+' *arg'+str(n)+'h = ('+typs[n]+' *)(((ops_reduction)args['+str(n)+'].data)->data + ((ops_reduction)args['+str(n)+'].data)->size * block->index);')
            code('#else')
            code(typs[n]+' *arg'+str(n)+'h = ('+typs[n]+' *)(((ops_reduction)args['+str(n)+'].data)->data);')
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
          code('consts_bytes += ROUND_UP(arg'+str(n)+'.dim*sizeof('+typs[n]+'));')
        elif accs[n] != OPS_READ:
          code('reduct_bytes += ROUND_UP(maxblocks*'+str(dims[n])+'*sizeof('+typs[n]+'));')
          code('reduct_size = MAX(reduct_size,sizeof('+typs[n]+')*'+str(dims[n])+');')
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
        code('arg'+str(n)+'.data = block->instance->OPS_reduct_h + reduct_bytes;')
        code('arg'+str(n)+'.data_d = block->instance->OPS_reduct_d + reduct_bytes;')
        code('for (int b=0; b<maxblocks; b++)')
        if accs[n] == OPS_INC:
          code('for (int d=0; d<'+str(dims[n])+'; d++) (('+typs[n]+' *)arg'+str(n)+'.data)[d+b*'+str(dims[n])+'] = ZERO_'+typs[n]+';')
        if accs[n] == OPS_MAX:
          code('for (int d=0; d<'+str(dims[n])+'; d++) (('+typs[n]+' *)arg'+str(n)+'.data)[d+b*'+str(dims[n])+'] = -INFINITY_'+typs[n]+';')
        if accs[n] == OPS_MIN:
          code('for (int d=0; d<'+str(dims[n])+'; d++) (('+typs[n]+' *)arg'+str(n)+'.data)[d+b*'+str(dims[n])+'] = INFINITY_'+typs[n]+';')
        code('reduct_bytes += ROUND_UP(maxblocks*'+str(dims[n])+'*sizeof('+typs[n]+'));')
        code('')

    code('')

    for n in range (0, nargs):
      if arg_typ[n] == 'ops_arg_gbl':
        if accs[n] == OPS_READ and (not dims[n].isdigit() or int(dims[n])>1):
          code('arg'+str(n)+'.data = block->instance->OPS_consts_h + consts_bytes;')
          code('arg'+str(n)+'.data_d = block->instance->OPS_consts_d + consts_bytes;')
          code('for (int d=0; d<arg'+str(n)+'.dim; d++) (('+typs[n]+' *)arg'+str(n)+'.data)[d] = arg'+str(n)+'h[d];')
          code('consts_bytes += ROUND_UP(arg'+str(n)+'.dim*sizeof('+typs[n]+'));')
    if GBL_READ == True and GBL_READ_MDIM == True:
      code('mvConstArraysToDevice(block->instance,consts_bytes);')

    if GBL_INC == True or GBL_MIN == True or GBL_MAX == True or GBL_WRITE == True:
      code('mvReductArraysToDevice(block->instance,reduct_bytes);')

    for n in range (0, nargs):
      if arg_typ[n] == 'ops_arg_dat':
        code('long long int dat'+str(n)+' = (block->instance->OPS_soa ? args['+str(n)+'].dat->type_size : args['+str(n)+'].dat->elem_size);')

    code('')
    code('char *p_a['+str(nargs)+'];')

    #some custom logic for multigrid
    if MULTI_GRID:
      for n in range (0, nargs):
        if prolong[n] == 1 or restrict[n] == 1:
          comm('This arg has a prolong stencil - so create different ranges')
          code('int start_'+str(n)+'['+str(NDIM)+']; int end_'+str(n)+'['+str(NDIM)+']; int stride_'+str(n)+'['+str(NDIM)+'];int d_size_'+str(n)+'['+str(NDIM)+'];')
          code('#ifdef OPS_MPI')
          FOR('n','0',str(NDIM))
          code('sub_dat *sd'+str(n)+' = OPS_sub_dat_list[args['+str(n)+'].dat->index];')
          code('stride_'+str(n)+'[n] = args['+str(n)+'].stencil->mgrid_stride[n];')
          code('d_size_'+str(n)+'[n] = args['+str(n)+'].dat->d_m[n] + sd'+str(n)+'->decomp_size[n] - args['+str(n)+'].dat->d_p[n];')
          if restrict[n] == 1:
            code('start_'+str(n)+'[n] = global_idx[n]*stride_'+str(n)+'[n] - sd'+str(n)+'->decomp_disp[n] + args['+str(n)+'].dat->d_m[n];')
          else:
            code('start_'+str(n)+'[n] = global_idx[n]/stride_'+str(n)+'[n] - sd'+str(n)+'->decomp_disp[n] + args['+str(n)+'].dat->d_m[n];')
          code('end_'+str(n)+'[n] = start_'+str(n)+'[n] + d_size_'+str(n)+'[n];')
          ENDFOR()
          code('#else')
          FOR('n','0',str(NDIM))
          code('stride_'+str(n)+'[n] = args['+str(n)+'].stencil->mgrid_stride[n];')
          code('d_size_'+str(n)+'[n] = args['+str(n)+'].dat->d_m[n] + args['+str(n)+'].dat->size[n] - args['+str(n)+'].dat->d_p[n];')
          if restrict[n] == 1:
            code('start_'+str(n)+'[n] = global_idx[n]*stride_'+str(n)+'[n];')
          else:
            code('start_'+str(n)+'[n] = global_idx[n]/stride_'+str(n)+'[n];')
          code('end_'+str(n)+'[n] = start_'+str(n)+'[n] + d_size_'+str(n)+'[n];')
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
        code('long long int base'+str(n)+' = args['+str(n)+'].dat->base_offset + ')
        code('         dat'+str(n)+' * 1 * ('+starttext+'[0] * args['+str(n)+'].stencil->stride[0]);')
        for d in range (1, NDIM):
          line = 'base'+str(n)+' = base'+str(n)+'+ dat'+str(n)+' *\n'
          for d2 in range (0,d):
            line = line + config.depth*' '+'  args['+str(n)+'].dat->size['+str(d2)+'] *\n'
          code(line[:-1])
          code('  ('+starttext+'['+str(d)+'] * args['+str(n)+'].stencil->stride['+str(d)+']);')

        code('p_a['+str(n)+'] = (char *)args['+str(n)+'].data_d + base'+str(n)+';')
        code('')

    #halo exchange
    code('')
    code('#ifndef OPS_LAZY')
    code('ops_H_D_exchanges_device(args, '+str(nargs)+');')
    code('ops_halo_exchanges(args,'+str(nargs)+',range);')
    code('#endif')
    code('')
    IF('block->instance->OPS_diags > 1')
    code('ops_timers_core(&c2,&t2);')
    code('block->instance->OPS_kernels['+str(nk)+'].mpi_time += t2-t1;')
    ENDIF()
    code('')


    #set up shared memory for reduction
    if GBL_INC == True or GBL_MIN == True or GBL_MAX == True or GBL_WRITE == True:
       code('size_t nshared = 0;')
       code('int nthread = block->instance->OPS_block_size_x*block->instance->OPS_block_size_y*block->instance->OPS_block_size_z;')
       code('')
    for n in range (0, nargs):
      if arg_typ[n] == 'ops_arg_gbl' and accs[n] != OPS_READ:
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
    n_per_line = 2
    if GBL_INC == True or GBL_MIN == True or GBL_MAX == True or GBL_WRITE == True:
      text = 'ops_'+name+'<<<grid, tblock, nshared >>> ( '
    else:
      text = 'ops_'+name+'<<<grid, tblock >>> ( '
    for n in range (0, nargs):
      if arg_typ[n] == 'ops_arg_dat':
        text = text +' ('+typs[n]+' *)p_a['+str(n)+'],'
        if n in needDimList:
          text = text + ' arg'+str(n)+'.dim,'
      elif arg_typ[n] == 'ops_arg_gbl':
        if dims[n].isdigit() and int(dims[n])==1 and accs[n]==OPS_READ:
          text = text +' *('+typs[n]+' *)arg'+str(n)+'.data,'
        else:
          text = text +' ('+typs[n]+' *)arg'+str(n)+'.data_d,'
          if n in needDimList and accs[n] != OP_READ:
            text = text + ' arg'+str(n)+'.dim,'
      elif arg_typ[n] == 'ops_arg_idx':
        if NDIM==1:
          text = text + ' arg_idx[0],'
        if NDIM==2:
          text = text + ' arg_idx[0], arg_idx[1],'
        elif NDIM==3:
          text = text + ' arg_idx[0], arg_idx[1], arg_idx[2],'
      if restrict[n] or prolong[n]:
        if NDIM==1:
          text = text + 'stride_'+str(n)+'[0],'
        if NDIM==2:
          text = text + 'stride_'+str(n)+'[0],stride_'+str(n)+'[1],'
        if NDIM==3:
          text = text + 'stride_'+str(n)+'[0],stride_'+str(n)+'[1],stride_'+str(n)+'[2],'

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
          code('arg'+str(n)+'h[d] = arg'+str(n)+'h[d] + (('+typs[n]+' *)arg'+str(n)+'.data)[d+b*'+str(dims[n])+'];')
        elif accs[n] == OPS_MAX:
          code('arg'+str(n)+'h[d] = MAX(arg'+str(n)+'h[d],(('+typs[n]+' *)arg'+str(n)+'.data)[d+b*'+str(dims[n])+']);')
        elif accs[n] == OPS_MIN:
          code('arg'+str(n)+'h[d] = MIN(arg'+str(n)+'h[d],(('+typs[n]+' *)arg'+str(n)+'.data)[d+b*'+str(dims[n])+']);')
        ENDFOR()
        ENDFOR()
        code('arg'+str(n)+'.data = (char *)arg'+str(n)+'h;')
        code('')

    IF('block->instance->OPS_diags>1')
    code('cutilSafeCall(block->instance->ostream(), cudaDeviceSynchronize());')
    code('ops_timers_core(&c1,&t1);')
    code('block->instance->OPS_kernels['+str(nk)+'].time += t1-t2;')
    ENDIF()
    code('')

    # This is not doen any more due to redution_handles treatement under MPI
    # if reduction == 1 :
    #   for n in range (0, nargs):
    #     if arg_typ[n] == 'ops_arg_gbl' and accs[n] <> OPS_READ:
    #       #code('ops_mpi_reduce(&arg'+str(n)+',('+typs[n]+' *)p_a['+str(n)+']);')
    #   code('ops_timers_core(&c1,&t1);')
    #   code('block->instance->OPS_kernels['+str(nk)+'].mpi_time += t1-t2;')

    code('#ifndef OPS_LAZY')
    code('ops_set_dirtybit_device(args, '+str(nargs)+');')
    for n in range (0, nargs):
      if arg_typ[n] == 'ops_arg_dat' and (accs[n] == OPS_WRITE or accs[n] == OPS_RW or accs[n] == OPS_INC):
        code('ops_set_halo_dirtybit3(&args['+str(n)+'],range);')
    code('#endif')

    code('')
    IF('block->instance->OPS_diags > 1')
    comm('Update kernel record')
    code('ops_timers_core(&c2,&t2);')
    code('block->instance->OPS_kernels['+str(nk)+'].mpi_time += t2-t1;')
    for n in range (0, nargs):
      if arg_typ[n] == 'ops_arg_dat':
        code('block->instance->OPS_kernels['+str(nk)+'].transfer += ops_compute_transfer(dim, start, end, &arg'+str(n)+');')
    ENDIF()
    config.depth = config.depth - 2
    code('}')
    code('')
    code('#ifdef OPS_LAZY')
    code('void ops_par_loop_'+name+'(char const *name, ops_block block, int dim, int* range,')
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
    code('ops_kernel_descriptor *desc = (ops_kernel_descriptor *)calloc(1,sizeof(ops_kernel_descriptor));')
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
    code('desc->args = (ops_arg*)ops_malloc('+str(nargs)+'*sizeof(ops_arg));')
    declared = 0
    for n in range (0, nargs):
      code('desc->args['+str(n)+'] = arg'+str(n)+';')
      if arg_typ[n] == 'ops_arg_dat':
        code('desc->hash = ((desc->hash << 5) + desc->hash) + arg'+str(n)+'.dat->index;')
      if arg_typ[n] == 'ops_arg_gbl' and accs[n] == OPS_READ:
        if declared == 0:
          code('char *tmp = (char*)ops_malloc(arg'+str(n)+'.dim*sizeof('+typs[n]+'));')
          declared = 1
        else:
          code('tmp = (char*)ops_malloc(arg'+str(n)+'.dim*sizeof('+typs[n]+'));')
        code('memcpy(tmp, arg'+str(n)+'.data,arg'+str(n)+'.dim*sizeof('+typs[n]+'));')
        code('desc->args['+str(n)+'].data = tmp;')
    code('desc->function = ops_par_loop_'+name+'_execute;')
    IF('block->instance->OPS_diags > 1')
    code('ops_timing_realloc(block->instance,'+str(nk)+',"'+name+'");')
    ENDIF()
    code('ops_enqueue_kernel(desc);')
    config.depth = 0
    code('}')
    code('#endif')


    ##########################################################################
    #  output individual kernel file
    ##########################################################################
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
      code('__constant__ '+consts[nc]['type']+' '+(str(consts[nc]['name']).replace('"','')).strip()+';')
    else:
      if consts[nc]['dim'].isdigit() and int(consts[nc]['dim']) > 0:
        num = str(consts[nc]['dim'])
        code('__constant__ '+consts[nc]['type']+' '+(str(consts[nc]['name']).replace('"','')).strip()+'['+num+'];')
      else:
        code('__constant__ '+consts[nc]['type']+' *'+(str(consts[nc]['name']).replace('"','')).strip()+';')



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

  kernel_name_list = []

  for nk in range(0,len(kernels)):
    if kernels[nk]['name'] not in kernel_name_list :
      code('#include "'+kernels[nk]['name']+'_cuda_kernel.cu"')
      kernel_name_list.append(kernels[nk]['name'])

  fid = open('./CUDA/'+master_basename[0]+'_kernels.cu','w')
  fid.write('//\n// auto-generated by ops.py//\n\n')
  fid.write(config.file_text)
  fid.close()
