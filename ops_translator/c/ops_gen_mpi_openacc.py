
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
## @brief
#  OPS OpenMP code generator
#  This routine is called by ops.py which parses the input files
#
# It produces a file xxx_omp_kernel.cpp for each kernel,
# plus a master kernel file
#

"""
OPS OpenMP code generator

This routine is called by ops.py which parses the input files

It produces a file xxx_omp_kernel.cpp for each kernel,
plus a master kernel file

"""

import errno
import os

import config
from config import OPS_READ, OPS_WRITE, OPS_RW, OPS_INC, OPS_MAX, OPS_MIN

import util
from util import get_kernel_func_text, parse_signature, replace_ACC_kernel_body, parse_replace_ACC_signature, get_kernel_func_text
from util import comm, code, FOR, ENDFOR, IF, ENDIF

def ops_gen_mpi_openacc(master, consts, kernels, soa_set):
  NDIM = 2 #the dimension of the application is hardcoded here .. need to get this dynamically

  src_dir = os.path.dirname(master) or '.'
  master_basename = os.path.splitext(os.path.basename(master))


  ##########################################################################
  #  create new kernel file
  ##########################################################################
  try:
    os.makedirs('./OpenACC')
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
    for n in range (0, nargs):
      restrict[n] = 0
      prolong[n] = 0
      if str(stens[n]).find('RESTRICT') > 0:
        restrict[n] = 1
        MULTI_GRID = 1
      if str(stens[n]).find('PROLONG') > 0 :
        prolong[n] = 1
        MULTI_GRID = 1

    reduct = 0
    for n in range (0, nargs):
      if arg_typ[n] == 'ops_arg_gbl' and accs[n] != OPS_READ:
        reduct = 1

    n_per_line = 2

    if NDIM==3:
      n_per_line = 1

    i = name.find('kernel')

    reduction = False
    ng_args = 0

    for n in range (0, nargs):
      if arg_typ[n] == 'ops_arg_gbl':
        reduction = True
      else:
        ng_args = ng_args + 1


    arg_idx = 0
    for n in range (0, nargs):
      if arg_typ[n] == 'ops_arg_idx':
        arg_idx = 1

    ##########################################################################
    #  generate constants and MACROS
    ##########################################################################

    code('')
    if not 'calc_dt_kernel_print' in name:
      if not (NDIM==3 and 'field_summary' in name):
        code('#define OPS_GPU')
        code('')
    for n in range (0, nargs):
      if arg_typ[n] == 'ops_arg_dat':
        code(f'int xdim{n}_{name};')
        if NDIM>2 or (NDIM==2 and soa_set):
          code(f'int ydim{n}_{name};')
        if NDIM>3 or (NDIM==3 and soa_set):
          code(f'int zdim{n}_{name};')
    code('')

    ##########################################################################
    #  generate header
    ##########################################################################

    comm('user function')
    text = get_kernel_func_text(name, src_dir, arg_typ).rstrip()
    j = text.find('{')

    m = text.find(name)
    arg_list = parse_signature(text[m+len(name):j])

    text = (
        text[0 : m + len(name)]
        + parse_replace_ACC_signature(text[m + len(name) : j], arg_typ, dims)
        + replace_ACC_kernel_body(text[j:], arg_list, arg_typ, nargs)
    )

    l = text[0:m].find('inline')
    if(l<0):
      text = 'inline '+text
#    code('#pragma acc routine')
    code(text)
    code('')


    code('')

    ##########################################################################
    #  generate C wrapper
    ##########################################################################
    code(f'void {name}_c_wrapper(')
    config.depth = config.depth+2
    for n in range (0, nargs):
      if arg_typ[n] == 'ops_arg_gbl' and accs[n] == OPS_READ and dims[n].isdigit() and int(dims[n])==1:
        code(f'{typs[n]} p_a{n},')
      else:
        code(f'{typs[n]} *p_a{n},')
        if restrict[n] or prolong[n]:
          code(f'int *stride_{n},')
    if arg_idx:
      if NDIM == 1:
        code('int arg_idx0,')
      elif NDIM == 2:
        code('int arg_idx0, int arg_idx1,')
      elif NDIM == 3:
        code('int arg_idx0, int arg_idx1, int arg_idx2,')

    if MULTI_GRID:
      if NDIM == 1:
        code('int global_idx0,')
      elif NDIM == 2:
        code('int global_idx0, int global_idx1,')
      elif NDIM == 3:
        code('int global_idx0, int global_idx1, int global_idx2,')

    if NDIM == 1:
      code('int x_size) {')
    elif NDIM == 2:
      code('int x_size, int y_size) {')
    elif NDIM == 3:
      code('int x_size, int y_size, int z_size) {')


    for n in range (0,nargs):
      if arg_typ[n] == 'ops_arg_gbl':
        if accs[n] != OPS_READ:
          for d in range(0,int(dims[n])):
            code(f'{typs[n]} p_a{n}_{d} = p_a{n}[{d}];')
      if restrict[n] or prolong[n]:
        code(f'int stride_{n}0 = stride_{n}[0];')
        if NDIM >= 2:
          code(f'int stride_{n}1 = stride_{n}[1];')
        if NDIM >= 3:
          code(f'int stride_{n}2 = stride_{n}[2];')

    line = '#pragma acc parallel deviceptr('
    for n in range (0,nargs):
      if arg_typ[n] == 'ops_arg_dat':
        line += f'p_a{n},'
      if arg_typ[n] == 'ops_arg_gbl':
        if accs[n] == OPS_READ and (not dims[n].isdigit() or int(dims[n])>1):
          line += f'p_a{n},'
    line = line[:-1]+')'
    for n in range (0,nargs):
      if arg_typ[n] == 'ops_arg_gbl':
        if accs[n] == OPS_MIN:
          for d in range(0,int(dims[n])):
            line += f' reduction(min:p_a{n}_{d})'
        if accs[n] == OPS_MAX:
          for d in range(0,int(dims[n])):
            line += f' reduction(max:p_a{n}_{d})'
        if accs[n] == OPS_INC:
          for d in range(0,int(dims[n])):
            line += f' reduction(+:p_a{n}_{d})'
        if accs[n] == OPS_WRITE: #this may not be correct ..
          for d in range(0,int(dims[n])):
            line += f' reduction(+:p_a{n}_{d})'
    code('#ifdef OPS_GPU')
    code(line)
    line = '#pragma acc loop'
    for n in range (0,nargs):
      if arg_typ[n] == 'ops_arg_gbl':
        if accs[n] == OPS_MIN:
          for d in range(0,int(dims[n])):
            line += f' reduction(min:p_a{n}_{d})'
        if accs[n] == OPS_MAX:
          for d in range(0,int(dims[n])):
            line += f' reduction(max:p_a{n}_{d})'
        if accs[n] == OPS_INC:
          for d in range(0,int(dims[n])):
            line += f' reduction(+:p_a{n}_{d})'
        if accs[n] == OPS_WRITE: #this may not be correct
          for d in range(0,int(dims[n])):
            line += f' reduction(+:p_a{n}_{d})'
    code(line)
    code('#endif')
    if NDIM==3:
      FOR('n_z','0','z_size')
      code('#ifdef OPS_GPU')
      code(line)
      code('#endif')
      FOR('n_y','0','y_size')
      code('#ifdef OPS_GPU')
      code(line)
      code('#endif')
    if NDIM==2:
      FOR('n_y','0','y_size')
      code('#ifdef OPS_GPU')
      code(line)
      code('#endif')

    FOR('n_x','0','x_size')
    if arg_idx:
      if NDIM==1:
        code('int arg_idx[] = {arg_idx0+n_x};')
      elif NDIM==2:
        code('int arg_idx[] = {arg_idx0+n_x, arg_idx1+n_y};')
      elif NDIM==3:
        code('int arg_idx[] = {arg_idx0+n_x, arg_idx1+n_y, arg_idx2+n_z};')

    for n in range (0,nargs):
      if arg_typ[n] == 'ops_arg_gbl':
        if accs[n] == OPS_MIN and int(dims[n])>1:
          code(f'{typs[n]} p_a{n}_local[{dims[n]}];')
          for d in range(0,int(dims[n])):
            code(f'p_a{n}_local[{d}] = p_a{n}[{d}];') #need +INFINITY_ change to
        if accs[n] == OPS_MAX and int(dims[n])>1:
          code(f'{typs[n]} p_a{n}_local[{dims[n]}];')
          for d in range(0,int(dims[n])):
            code(f'p_a{n}_local[{d}] = p_a{n}[{d}];') #need -INFINITY_ change to
        if accs[n] == OPS_INC and int(dims[n])>1:
          code(f'{typs[n]} p_a{n}_local[{dims[n]}];')
          for d in range(0,int(dims[n])):
            code(f'p_a{n}_local[{d}] = ZERO_{typs[n]};')
        if accs[n] == OPS_WRITE and int(dims[n])>1: #this may not be correct
          code(f'{typs[n]} p_a{n}_local[{dims[n]}];')
          for d in range(0,int(dims[n])):
            code(f'p_a{n}_local[{d}] = ZERO_{typs[n]};')


    for n in range (0, nargs):
      if arg_typ[n] == 'ops_arg_dat':
        if restrict[n] == 1:
            n_x = f'n_x*stride_{n}0'
            n_y = f'n_y*stride_{n}1'
            n_z = f'n_z*stride_{n}2'
        elif prolong[n] == 1:
          n_x = f'(n_x+global_idx0%stride_{n}0)/stride_{n}0'
          n_y = f'(n_y+global_idx1%stride_{n}1)/stride_{n}1'
          n_z = f'(n_z+global_idx2%stride_{n}2)/stride_{n}2'
        else:
          n_x = 'n_x'
          n_y = 'n_y'
          n_z = 'n_z'

        if NDIM == 1:
          if soa_set:
            text = f' p_a{n} + {n_x}*{stride[NDIM*n]}'
          else:
            text = f' p_a{n} + {n_x}*{stride[NDIM*n]}*{dims[n]}'
        elif NDIM == 2:
          if soa_set:
            text = f' p_a{n} + {n_x}*{stride[NDIM*n]} + {n_y}*xdim{n}_{name}*{stride[NDIM*n+1]}'
          else:
            text = f' p_a{n} + {n_x}*{stride[NDIM*n]}*{dims[n]} + {n_y}*xdim{n}_{name}*{stride[NDIM*n+1]}*{dims[n]}'
        elif NDIM == 3:
          if soa_set:
            text = f' p_a{n} + {n_x}*{stride[NDIM*n]} + {n_y}*xdim{n}_{name}*{stride[NDIM*n+1]}'
            text += f' + {n_z}*xdim{n}_{name}*ydim{n}_{name}*{stride[NDIM*n+2]}'
          else:
            text = f' p_a{n} + {n_x}*{stride[NDIM*n]}*{dims[n]} + {n_y}*xdim{n}_{name}*{stride[NDIM*n+1]}*{dims[n]}'
            text += f' + {n_z}*xdim{n}_{name}*ydim{n}_{name}*{stride[NDIM*n+2]}*{dims[n]}'

        pre = ''
        if accs[n] == OPS_READ:
          pre = 'const '
        dim = ''
        sizelist = ''
        extradim = 0
        if dims[n].isdigit() and int(dims[n])>1:
            dim = dims[n]
            extradim = 1
        elif not dims[n].isdigit():
            dim = f'arg{n}.dim'
            extradim = 1
        dimlabels = 'xyzuv'
        for i in range(1,NDIM):
          sizelist += f'{dimlabels[i-1]}dim{n}_{name}, '
        extradim = f'{dimlabels[NDIM+extradim-2]}dim{n}_{name}'
        if dim == '':
          if NDIM==1:
            code(f'{pre}ptr_{typs[n]} ptr{n} = {{ {text} }};')
          else:
            code(f'{pre}ptr_{typs[n]} ptr{n} = {{ {text}, {sizelist[:-2]}}};')
        else:
          code('#ifdef OPS_SOA')
          code(f'{pre}ptrm_{typs[n]} ptr{n} = {{ {text}, {sizelist + extradim}}};')
          code('#else')
          code(f'{pre}ptrm_{typs[n]} ptr{n} = {{ {text}, {sizelist+dim}}};')
          code('#endif')


    text = name+'( '
    for n in range (0, nargs):
      if arg_typ[n] == 'ops_arg_dat':
        text += f'ptr{n}'
      elif arg_typ[n] == 'ops_arg_gbl':
        if accs[n] == OPS_READ:
          if dims[n].isdigit() and int(dims[n])==1:
            text += f' &p_a{n}'
          else:
            text += f' p_a{n}'
        else:
          if dims[n].isdigit() and int(dims[n]) == 1:
            text += f' &p_a{n}_0'
          else:
            text += f' p_a{n}_local'
      elif arg_typ[n] == 'ops_arg_idx':
        text += 'arg_idx'

      if nargs != 1 and n != nargs-1:
        text += ','
      else:
        text += ' );\n'
      if n%n_per_line == 0 and n != nargs-1:
        text += '\n          '
    code(text);

    for n in range (0,nargs):
      if arg_typ[n] == 'ops_arg_gbl':
        if accs[n] == OPS_MIN and int(dims[n])>1:
          for d in range(0,int(dims[n])):
            code(f'p_a{n}_{d} = MIN(p_a{n}_{d},p_a{n}_local[{d}]);')
        if accs[n] == OPS_MAX and int(dims[n])>1:
          for d in range(0,int(dims[n])):
            code(f'p_a{n}_{d} = MAX(p_a{n}_{d}p_a{n}_local[{d}]);')
        if accs[n] == OPS_INC and int(dims[n])>1:
          for d in range(0,int(dims[n])):
            code(f'p_a{n}_{d} +=p_a{n}_local[{d}];')
        if accs[n] == OPS_WRITE and int(dims[n])>1: #this may not be correct
          for d in range(0,int(dims[n])):
            code(f'p_a{n}_{d} +=p_a{n}_local[{d}];')


    ENDFOR()
    if NDIM==2:
      ENDFOR()
    if NDIM==3:
      ENDFOR()
      ENDFOR()

    for n in range (0, nargs):
      if arg_typ[n] == 'ops_arg_gbl':
        if accs[n] != OPS_READ:
          for d in range(0,int(dims[n])):
            code(f'p_a{n}[{d}] = p_a{n}_{d};')

    config.depth = config.depth-2
    code('}')

    ##########################################################################
    #  output individual kernel file
    ##########################################################################
    util.write_text_to_file(f"./OpenACC/{name}_openacc_kernel_c.c")
    ##########################################################################
    #  now host stub
    ##########################################################################

    code('')
    if not 'calc_dt_kernel_print' in name:
      if not (NDIM==3 and 'field_summary' in name):
        code('#define OPS_GPU')
        code('')
    for n in range (0, nargs):
      if arg_typ[n] == 'ops_arg_dat':
        code(f'extern int xdim{n}_{name};')
        code(f'int xdim{n}_{name}_h = -1;')
        if NDIM>2 or (NDIM==2 and soa_set):
          code(f'extern int ydim{n}_{name};')
          code(f'int ydim{n}_{name}_h = -1;')
        if NDIM>3 or (NDIM==3 and soa_set):
          code(f'extern int zdim{n}_{name};')
          code(f'int zdim{n}_{name}_h = -1;')
    code('')

    code('#ifdef __cplusplus')
    code('extern "C" {')
    code('#endif')
    code(f'void {name}_c_wrapper(')
    config.depth = config.depth+2
    for n in range (0, nargs):
      if arg_typ[n] == 'ops_arg_gbl' and accs[n] == OPS_READ and dims[n].isdigit() and int(dims[n])==1:
        code(f'{typs[n]} p_a{n},')
      else:
        code(f'{typs[n]} *p_a{n},')
        if restrict[n] or prolong[n]:
          code(f'int *stride_{n},')
    if arg_idx:
      if NDIM == 1:
        code('int arg_idx0,')
      elif NDIM == 2:
        code('int arg_idx0, int arg_idx1,')
      elif NDIM == 3:
        code('int arg_idx0, int arg_idx1, int arg_idx2,')

    if MULTI_GRID:
      if NDIM == 1:
        code('int global_idx0,')
      elif NDIM == 2:
        code('int global_idx0, int global_idx1,')
      elif NDIM == 3:
        code('int global_idx0, int global_idx1, int global_idx2,')

    if NDIM == 1:
      code('int x_size);')
    elif NDIM == 2:
      code('int x_size, int y_size);')
    elif NDIM == 3:
      code('int x_size, int y_size, int z_size);')
    config.depth = config.depth-2
    code('')
    code('#ifdef __cplusplus')
    code('}')
    code('#endif')
    code('')
    comm(' host stub function')

    code(f'void ops_par_loop_{name}(char const *name, ops_block block, int dim, int* range,')
    text = ''
    for n in range (0, nargs):

      text += f' ops_arg arg{n}'
      if nargs != 1 and n != nargs-1:
        text += ','
      else:
        text += ') {'
      if n%n_per_line == 3 and n != nargs-1:
         text += '\n'
    code(text);
    config.depth = 2

    code('');
    comm('Timing')
    code('double t1,t2,c1,c2;')

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
    code('#ifdef CHECKPOINTING')
    code(f'if (!ops_checkpointing_before(args,{nargs},range,{nk})) return;')
    code('#endif')
    code('')

    IF('block->instance->OPS_diags > 1')
    code(f'ops_timing_realloc(block->instance,{nk},"{name}");')
    code(f'block->instance->OPS_kernels[{nk}].count++;')
    code('ops_timers_core(&c1,&t1);')
    ENDIF()

    code('')
    comm('compute locally allocated range for the sub-block')
    comm('')
    code(f'int start[{NDIM}];')
    code(f'int end[{NDIM}];')


    code('#ifdef OPS_MPI')
    code('sub_block_list sb = OPS_sub_block_list[block->index];')
    code('#endif //OPS_MPI')

    code('')
    code(f'int arg_idx[{NDIM}];')
    code(f'int arg_idx_base[{NDIM}];')


    code('#ifdef OPS_MPI')
    code(f'if (compute_ranges(args, {nargs},block, range, start, end, arg_idx) < 0) return;')
    code('#else //OPS_MPI')
    FOR('n','0',str(NDIM))
    code('start[n] = range[2*n];end[n] = range[2*n+1];')
    code('arg_idx[n] = start[n];')
    ENDFOR()
    code('#endif')
    FOR('n','0',str(NDIM))
    code('arg_idx_base[n] = arg_idx[n];')
    ENDFOR()

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
        code(f'int dat{n} = args[{n}].dat->elem_size;')


    GBL_READ = False
    GBL_READ_MDIM = False
    GBL_INC = False
    GBL_MAX = False
    GBL_MIN = False
    GBL_WRITE = False

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

    code('')
    for n in range (0, nargs):
        if arg_typ[n] == 'ops_arg_gbl':
          if accs[n] != OPS_READ or (accs[n] == OPS_READ and (not dims[n].isdigit() or int(dims[n])>1)):
            if (accs[n] == OPS_READ):
              code(f'{typs[n]} *arg{n}h = ({typs[n]} *)arg{n}.data;')
            else:
              code('#ifdef OPS_MPI')
              code(f'{typs[n]} *arg{n}h = ({typs[n]} *)(((ops_reduction)args[{n}].data)->data + ((ops_reduction)args[{n}].data)->size * block->index);')
              code('#else')
              code(f'{typs[n]} *arg{n}h = ({typs[n]} *)(((ops_reduction)args[{n}].data)->data);')
              code('#endif')
    if GBL_READ == True and GBL_READ_MDIM == True:
      comm('Upload large globals')
      code('#ifdef OPS_GPU')
      code('int consts_bytes = 0;')
    for n in range (0, nargs):
      if arg_typ[n] == 'ops_arg_gbl':
        if accs[n] == OPS_READ and (not dims[n].isdigit() or int(dims[n])>1):
          code(f'consts_bytes += ROUND_UP({dims[n]}*sizeof({typs[n]}));')
    if GBL_READ == True and GBL_READ_MDIM == True:
      code('reallocConstArrays(block->instance,consts_bytes);')
    for n in range (0, nargs):
      if arg_typ[n] == 'ops_arg_gbl':
        if accs[n] == OPS_READ and (not dims[n].isdigit() or int(dims[n])>1):
          code('consts_bytes = 0;')
          code(f'args[{n}].data = block->instance->OPS_consts_h + consts_bytes;')
          code(f'args[{n}].data_d = block->instance->OPS_consts_d + consts_bytes;')
          code(f'for (int d=0; d<{dims[n]}; d++) (({typs[n]} *)args[{n}].data)[d] = arg{n}h[d];')
          code(f'consts_bytes += ROUND_UP({dims[n]}*sizeof(int));')
    if GBL_READ == True and GBL_READ_MDIM == True:
      code('mvConstArraysToDevice(block->instance,consts_bytes);')
      code('#endif //OPS_GPU')

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
          starttext = f'start_{n}'
        else:
          starttext = 'start'
        code(f'long long int base{n} = args[{n}].dat->base_offset + (long long int)(block->instance->OPS_soa ? args[{n}].dat->type_size : args[{n}].dat->elem_size) * {starttext}[0] * args[{n}].stencil->stride[0];')
        for d in range (1, NDIM):
          line = f'base{n} = base{n} + (long long int)(block->instance->OPS_soa ? args[{n}].dat->type_size : args[{n}].dat->elem_size) *\n'
          for d2 in range (0,d):
            line = line + config.depth*' '+f'  args[{n}].dat->size[{d2}] *\n'
          code(line[:-1])
          code(f'  {starttext}[{d}] * args[{n}].stencil->stride[{d}];')

        code('#ifdef OPS_GPU')
        code(f'{typs[n]} *p_a{n} = ({typs[n]} *)((char *)args[{n}].data_d + base{n});')
        code('#else')
        code(f'{typs[n]} *p_a{n} = ({typs[n]} *)((char *)args[{n}].data + base{n});')
        code('#endif')
        code('')
      elif arg_typ[n] == 'ops_arg_gbl':
        if accs[n] == OPS_READ:
          if dims[n].isdigit() and int(dims[n])==1:
            code(f'{typs[n]} *p_a{n} = ({typs[n]} *)args[{n}].data;')
          else:
            code('#ifdef OPS_GPU')
            code(f'{typs[n]} *p_a{n} = ({typs[n]} *)args[{n}].data_d;')
            code('#else')
            code(f'{typs[n]} *p_a{n} = arg{n}h;')
            code('#endif')
        else:
          code(f'{typs[n]} *p_a{n} = arg{n}h;')
      else:
        code(f'{typs[n]} *p_a{n} = NULL;')
        code('')

    #iteration range size
    code('')
    code('int x_size = MAX(0,end[0]-start[0]);')
    if NDIM==2:
      code('int y_size = MAX(0,end[1]-start[1]);')
    if NDIM==3:
      code('int y_size = MAX(0,end[1]-start[1]);')
      code('int z_size = MAX(0,end[2]-start[2]);')
    code('')

    comm("initialize global variable with the dimension of dats")
    #array sizes
    for n in range (0,nargs):
      if arg_typ[n] == 'ops_arg_dat':
        code(f'int xdim{n} = args[{n}].dat->size[0];')
        if NDIM==3:
          code(f'int ydim{n} = args[{n}].dat->size[1];')

    #array sizes - upload to GPU
    condition = ''
    for n in range (0, nargs):
      if arg_typ[n] == 'ops_arg_dat':
        condition += f'xdim{n} != xdim{n}_{name}_h || '
        if NDIM==3:
          condition += f'ydim{n} != ydim{n}_{name}_h || '
    condition = condition[:-4]
    IF(condition)

    #array sizes - upload to GPU
    for n in range (0, nargs):
      if arg_typ[n] == 'ops_arg_dat':
        code(f'xdim{n}_{name} = xdim{n};')
        code(f'xdim{n}_{name}_h = xdim{n};')
        if NDIM==3:
          code(f'ydim{n}_{name} = ydim{n};')
          code(f'ydim{n}_{name}_h = ydim{n};')
    ENDIF()
    code('')

    comm('Halo Exchanges')
    code('')
    code('#ifdef OPS_GPU')
    code(f'ops_H_D_exchanges_device(args, {nargs});')
    code('#else')
    code(f'ops_H_D_exchanges_host(args, {nargs});')
    code('#endif')
    code(f'ops_halo_exchanges(args,{nargs},range);')
    code('')
    code('#ifdef OPS_GPU')
    code(f'ops_H_D_exchanges_device(args, {nargs});')
    code('#else')
    code(f'ops_H_D_exchanges_host(args, {nargs});')
    code('#endif')

    IF('block->instance->OPS_diags > 1')
    code('ops_timers_core(&c2,&t2);')
    code(f'block->instance->OPS_kernels[{nk}].mpi_time += t2-t1;')
    ENDIF()
    code('')


    code(name+'_c_wrapper(')
    config.depth = config.depth+2
    for n in range (0, nargs):
      if arg_typ[n] == 'ops_arg_gbl' and accs[n] == OPS_READ and dims[n].isdigit() and int(dims[n])==1:
        code(f'*p_a{n},')
      else:
        code(f'p_a{n},')
        if restrict[n] or prolong[n]:
          code(f'stride_{n},')

    if arg_idx:
      if NDIM==1:
        code('arg_idx[0],')
      elif NDIM==2:
        code('arg_idx[0], arg_idx[1],')
      elif NDIM==3:
        code('arg_idx[0], arg_idx[1], arg_idx[2],')
    if MULTI_GRID:
      if NDIM==1:
        code('global_idx[0],')
      elif NDIM==2:
        code('global_idx[0], global_idx[1],')
      elif NDIM==3:
        code('global_idx[0], global_idx[1], global_idx[2],')

    if NDIM == 1:
      code('x_size);')
    if NDIM == 2:
      code('x_size, y_size);')
    if NDIM == 3:
      code('x_size, y_size, z_size);')

    config.depth = config.depth-2


    code('')
    IF('block->instance->OPS_diags > 1')
    code('ops_timers_core(&c1,&t1);')
    code(f'block->instance->OPS_kernels[{nk}].time += t1-t2;')
    ENDIF()

    code('#ifdef OPS_GPU')
    code(f'ops_set_dirtybit_device(args, {nargs});')
    code('#else')
    code(f'ops_set_dirtybit_host(args, {nargs});')
    code('#endif')
    for n in range (0, nargs):
      if arg_typ[n] == 'ops_arg_dat' and (accs[n] == OPS_WRITE or accs[n] == OPS_RW or accs[n] == OPS_INC):
        code(f'ops_set_halo_dirtybit3(&args[{n}],range);')

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


    ##########################################################################
    #  output individual kernel file
    ##########################################################################
    util.write_text_to_file(f"./OpenACC/{name}_openacc_kernel.cpp")

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
  code('#include <math.h>')
  code('#include "ops_macros.h"')
  code('#ifdef __cplusplus')
  code('#include "ops_lib_core.h"')
  code('#include "ops_cuda_rt_support.h"')
  code('#endif')
  code('#if defined(OPS_MPI) && defined(__cplusplus)')
  code('#include "ops_mpi_core.h"')
  code('#endif')
  if os.path.exists(os.path.join(src_dir,'user_types.h')):
    code('#include "user_types.h"')

  comm(' global constants')
  for nc in range (0,len(consts)):
    if consts[nc]['dim'].isdigit() and int(consts[nc]['dim'])==1:
      code(f"extern {consts[nc]['type']} "+(str(consts[nc]['name']).replace('"','')).strip()+';')
    else:
      if consts[nc]['dim'].isdigit() and int(consts[nc]['dim']) > 0:
        num = str(consts[nc]['dim'])
        code(f"extern {consts[nc]['type']} "+(str(consts[nc]['name']).replace('"','')).strip()+f'[{num}];')
      else:
        code(f"extern {consts[nc]['type']} *"+(str(consts[nc]['name']).replace('"','')).strip()+';')

  util.write_text_to_file(f"./OpenACC/{master_basename[0]}_common.h")

  code(f"#include \"./OpenACC/{master_basename[0]}_common.h\"")
  code('')
  code('#include <openacc.h>')
  code('')
  code('void ops_init_backend() {acc_set_device_num(ops_get_proc()%acc_get_num_devices(acc_device_nvidia),acc_device_nvidia); }')
  code('')
  code('void ops_decl_const_char(OPS_instance *instance, int dim, char const *type,')
  code('int size, char *dat, char const *name){')
  config.depth = config.depth + 2
  code('ops_execute(instance);')

  for nc in range(0,len(consts)):
    IF('!strcmp(name,"'+(str(consts[nc]['name']).replace('"','')).strip()+'")')
    if consts[nc]['dim'].isdigit() and int(consts[nc]['dim'])==1:
      code((str(consts[nc]['name']).replace('"','')).strip()+f" = *({consts[nc]['type']}*)dat;")
    else:
      FOR('d','0',consts[nc]['dim'])
      code((str(consts[nc]['name']).replace('"','')).strip()+f"[d] = (({consts[nc]['type']}*)dat)[d];")
      ENDFOR()
    ENDIF()
    code('else')

  code('{')
  config.depth = config.depth + 2
  code('throw OPSException(OPS_RUNTIME_ERROR, "error: unknown const name");')
  ENDIF()
  config.depth = config.depth - 2
  code('}')

  code('')
  comm('user kernel files')

  for kernel_name in map(lambda kernel: kernel['name'], kernels):
      code(f"#include \"{kernel_name}_openacc_kernel.cpp\"")


  util.write_text_to_file(f"./OpenACC/{master_basename[0]}_kernels.cpp")

  code(f"#include \"./OpenACC/{master_basename[0]}_common.h\"")
  code('#include <math.h>')
  code('#include "ops_macros.h"')
  code('#include <openacc.h>')
  code('')
  comm('user kernel files')

  for kernel_name in map(lambda kernel: kernel['name'], kernels):
      code(f"#include \"{kernel_name}_openacc_kernel_c.c\"")

  config.depth = config.depth - 2
  util.write_text_to_file(f"./OpenACC/{master_basename[0]}_kernels_c.c")
