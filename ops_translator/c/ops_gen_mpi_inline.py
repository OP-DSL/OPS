
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
## @brief OPS OpenMP code generator
#
#  OPS OpenMP code generator
#
#  This routine is called by ops.py which parses the input files
#
#  It produces a file xxx_omp_kernel.cpp for each kernel,
#  plus a master kernel file
#

"""
OPS OpenMP code generator

This routine is called by ops.py which parses the input files

It produces a file xxx_omp_kernel.cpp for each kernel,
plus a master kernel file

"""

import re
import errno
import os

import config
from config import OPS_READ, OPS_WRITE, OPS_RW, OPS_INC, OPS_MAX, OPS_MIN

import util
from util import para_parse, parse_signature, replace_ACC_kernel_body, convert_ACC_body
from util import comm, code, FOR, ENDFOR, IF, ENDIF


def ops_gen_mpi_inline(master, consts, kernels, soa_set):
  NDIM = 2 #the dimension of the application is hardcoded here .. need to get this dynamically

  src_dir = os.path.dirname(master) or '.'
  master_basename = os.path.splitext(os.path.basename(master))
  try:
    os.makedirs('./MPI_inline')
  except OSError as e:
    if e.errno != errno.EEXIST:
      raise

  ##########################################################################
  #  create new kernel file
  ##########################################################################

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
    stride = [1] * (nargs+4) * NDIM
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
    for n in range (0, nargs):
      if arg_typ[n] == 'ops_arg_dat':
        code(f'int xdim{n}_{name};')
        if NDIM>2 or (NDIM==2 and soa_set):
          code(f'int ydim{n}_{name};')
        if NDIM>3 or (NDIM==3 and soa_set):
          code(f'int zdim{n}_{name};')
    code('')
    code('')

    for n in range (0, nargs):
      if arg_typ[n] == 'ops_arg_dat':
        if restrict[n] == 1:
          n_x = f'n_x*stride_{n}[0]'
          n_y = f'n_y*stride_{n}[1]'
          n_z = f'n_z*stride_{n}[2]'
        elif prolong[n] == 1:
          n_x = f'(n_x+global_idx[0]%stride_{n}[0])/stride_{n}[0]'
          n_y = f'(n_y+global_idx[1]%stride_{n}[1])/stride_{n}[1]'
          n_z = f'(n_z+global_idx[2]%stride_{n}[2])/stride_{n}[2]'
        else:
          n_x = f'n_x*{str(stride[NDIM*n])}'
          n_y = f'n_y*{str(stride[NDIM*n+1])}'
          n_z = f'n_z*{str(stride[NDIM*n+2])}'
        s_y = f'xdim{n}_{name}'
        s_z = f'xdim{n}_{name}*ydim{n}_{name}'
        s_u = f'xdim{n}_{name}*ydim{n}_{name}*zdim{n}_{name}'


    ##########################################################################
    #  generate header
    ##########################################################################

    comm('user function')
    text = util.get_file_text_for_kernel(name, src_dir)

    p = re.compile(f'void\\s+\\b{name}\\b')

    i = p.search(text).start()

    if(i < 0):
      print("\n********")
      print(f"Error: cannot locate user kernel function: {name} - Aborting code generation")
      exit(2)

    i2 = text[i:].find(name)
    i2 = i+i2
    j = text[i:].find('{')
    k = para_parse(text, i+j, '{', '}')
    kernel_text = text[i+j+1:k]
    kernel_text = convert_ACC_body(kernel_text)
    m = text.find(name)
    arg_list = parse_signature(text[i2+len(name):i+j])

    kernel_text = replace_ACC_kernel_body(kernel_text, arg_list, arg_typ, nargs)

    l = text[i:m].find('inline')
    if(l<0):
      text = text[i:k+2]
    else:
      text = text[i+l:k+2]


    i = text.find('{')
    i = text[0:i].rfind(')')
    if (NDIM==1):
      itervar = ', const int n_x'
    if (NDIM==2):
      itervar = ', const int n_x, const int n_y'
    if (NDIM==3):
      itervar = ', const int n_x, const int n_y, const int n_z'
    text = text[0:i]+itervar+text[i:]
    code('')


    code('')
    code('')

    ##########################################################################
    #  generate C wrapper
    ##########################################################################
    code(f'void {name}_c_wrapper(')
    config.depth = config.depth+2
    for n in range (0, nargs):
      if accs[n] == OPS_READ:
        pre = 'const '
      else:
        pre = ''
      if arg_typ[n] == 'ops_arg_gbl' and accs[n] != OPS_READ:
        code(f'{pre+typs[n]} * restrict {arg_list[n]}_g,')
      else:
        if arg_typ[n] == 'ops_arg_dat':
          code(f'{typs[n]} * restrict {arg_list[n]}_p,')
        else:
          code(f'{pre+typs[n]} * restrict {arg_list[n]},')
    if MULTI_GRID:
      code('const int * restrict global_idx,')
    for n in range(0,nargs):
      if restrict[n] == 1 or prolong[n] == 1:
        code(f'const int * restrict stride_{n},')
    if arg_idx:
      if NDIM == 1:
        code('int arg_idx0, ')
      elif NDIM == 2:
        code('int arg_idx0, int arg_idx1,')
      elif NDIM == 3:
        code('int arg_idx0, int arg_idx1, int arg_idx2,')

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
            code(f'{typs[n]} {arg_list[n]}_{d} = {arg_list[n]}_g[{d}];')

    redlist=''
    for n in range (0, nargs):
      if arg_typ[n] == 'ops_arg_gbl' and accs[n] != OPS_READ:
        if accs[n] == OPS_INC:
          for d in range(0,int(dims[n])):
            redlist += f' reduction(+:{arg_list[n]}_{d})'
        elif accs[n] == OPS_MIN:
          for d in range(0,int(dims[n])):
            redlist += f' reduction(min:{arg_list[n]}_{d})'
        elif accs[n] == OPS_MAX:
          for d in range(0,int(dims[n])):
            redlist += f' reduction(max:{arg_list[n]}_{d})'

    code('#pragma omp parallel for'+redlist)
    if NDIM==3:
      FOR('n_z','0','z_size')
      FOR('n_y','0','y_size')
    if NDIM==2:
      FOR('n_y','0','y_size')

    FOR('n_x','0','x_size')
    for n in range (0, nargs):
      if arg_typ[n] == 'ops_arg_gbl' and accs[n] != OPS_READ:
        if accs[n] == OPS_MIN:
          code(f'{typs[n]} {arg_list[n]}[{dims[n]}];')
          for d in range(0,int(dims[n])):
            code(f'{arg_list[n]}[{d}] = {arg_list[n]}_g[{d}];') #need +INFINITY_ change to
        if accs[n] == OPS_MAX:
          code(f'{typs[n]} {arg_list[n]}[{dims[n]}];')
          for d in range(0,int(dims[n])):
            code(f'{arg_list[n]}[{d}] = {arg_list[n]}_g[{d}];') #need -INFINITY_ change to
        if accs[n] == OPS_INC:
          code(f'{typs[n]} {arg_list[n]}[{dims[n]}];')
          for d in range(0,int(dims[n])):
            code(f'{arg_list[n]}[{d}] = 0;')#ZERO_{typs[n]};')
        if accs[n] == OPS_WRITE: #this may not be correct
          code(f'{typs[n]} {arg_list[n]}[{dims[n]}];')
          for d in range(0,int(dims[n])):
            code(f'{arg_list[n]}[{d}] = 0;')#ZERO_{typs[n]};')

    for n in range (0, nargs):
      if arg_typ[n] == 'ops_arg_idx':
        if NDIM==1:
          code(f'int {arg_list[n]}[] = {{arg_idx0+n_x}};')
        elif NDIM==2:
          code(f'int {arg_list[n]}[] = {{arg_idx0+n_x, arg_idx1+n_y}};')
        elif NDIM==3:
          code(f'int {arg_list[n]}[] = {{arg_idx0+n_x, arg_idx1+n_y, arg_idx2+n_z}};')

    for n in range (0, nargs):
      if arg_typ[n] == 'ops_arg_dat':
        pre = ''
        if accs[n] == OPS_READ:
          pre = 'const '
        offset = ''
        dim = ''
        sizelist = ''
        extradim = 0
        if dims[n].isdigit() and int(dims[n])>1:
            dim = dims[n]
            extradim = 1
        elif not dims[n].isdigit():
            dim = f'arg{n}.dim'
            extradim = 1
        if restrict[n] == 1:
          n_x = f'n_x*stride_{n}[0]'
          n_y = f'n_y*stride_{n}[1]'
          n_z = f'n_z*stride_{n}[2]'
        elif prolong[n] == 1:
          n_x = f'(n_x+global_idx[0]%stride_{n}[0])/stride_{n}[0]'
          n_y = f'(n_y+global_idx[1]%stride_{n}[1])/stride_{n}[1]'
          n_z = f'(n_z+global_idx[2]%stride_{n}[2])/stride_{n}[2]'
        else:
          n_x = 'n_x'
          n_y = 'n_y'
          n_z = 'n_z'
        if NDIM > 0:
          offset += f'{n_x}*{stride[NDIM*n]}'
        if NDIM > 1:
          offset += f' + {n_y} * xdim{n}_{name}*{stride[NDIM*n+1]}'
        if NDIM > 2:
          offset += f' + {n_z} * xdim{n}_{name} * ydim{n}_{name}*{stride[NDIM*n+2]}'
        dimlabels = 'xyzuv'
        for i in range(1,NDIM):
          sizelist += f'{dimlabels[i-1]}dim{n}_{name}, '
        extradim = f'{dimlabels[NDIM+extradim-2]}dim{n}_{name}'
        if dim == '':
          if NDIM==1:
            code(f'{pre}ptr_{typs[n]} {arg_list[n]} = {{ {arg_list[n]}_p + {offset}}};')
          else:
            code(f'{pre}ptr_{typs[n]} {arg_list[n]} = {{ {arg_list[n]}_p + {offset}, {sizelist[:-2]}}};')
        else:
          code(f''+'#ifdef OPS_SOA')
          code(f'{pre}ptrm_{typs[n]} {arg_list[n]} = {{ {arg_list[n]}_p + {offset}, {sizelist+extradim}}};')
          code(f''+'#else')
          code(f'{pre}ptrm_{typs[n]} {arg_list[n]} = {{ {arg_list[n]}_p + {offset}, {sizelist+dim}}};')
          code('#endif')


    code(kernel_text)

    for n in range (0,nargs):
      if arg_typ[n] == 'ops_arg_gbl':
        if accs[n] == OPS_MIN:
          for d in range(0,int(dims[n])):
            code(f'{arg_list[n]}_{d} = MIN({arg_list[n]}_{d},{arg_list[n]}[{d}]);')
        if accs[n] == OPS_MAX:
          for d in range(0,int(dims[n])):
            code(f'{arg_list[n]}_{d} = MAX({arg_list[n]}_{d},{arg_list[n]}[{d}]);')
        if accs[n] == OPS_INC:
          for d in range(0,int(dims[n])):
            code(f'{arg_list[n]}_{d} +={arg_list[n]}[{d}];')
        if accs[n] == OPS_WRITE: #this may not be correct
          for d in range(0,int(dims[n])):
            code(f'{arg_list[n]}_{d} +={arg_list[n]}[{d}];')


    ENDFOR()
    if NDIM==2:
      ENDFOR()
    if NDIM==3:
      ENDFOR()
      ENDFOR()

    for n in range (0, nargs):
      if arg_typ[n] == 'ops_arg_gbl' and accs[n] != OPS_READ:
        for d in range(0,int(dims[n])):
          code(f'{arg_list[n]}_g[{d}] = {arg_list[n]}_{d};')

    config.depth = config.depth-2
    code('}')

    ##########################################################################
    #  output individual kernel file
    ##########################################################################
    try:
      os.makedirs('./MPI_inline')
    except OSError as e:
      if e.errno != errno.EEXIST:
        raise
    util.write_text_to_file(f"./MPI_inline/{name}_mpiinline_kernel_c.c")
    ##########################################################################
    #  now host stub
    ##########################################################################

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
      code(f'{typs[n]} *p_a{n},')
    if MULTI_GRID:
      code('int *global_idx,')
    for n in range(0,nargs):
      if restrict[n] == 1 or prolong[n] == 1:
        code(f'int *stride_{n},')
    if arg_idx:
      if NDIM == 1:
        code('int arg_idx0,')
      elif NDIM == 2:
        code('int arg_idx0, int arg_idx1,')
      elif NDIM == 3:
        code('int arg_idx0, int arg_idx1, int arg_idx2,')

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

      text = text +' ops_arg arg'+str(n)
      if nargs != 1 and n != nargs-1:
        text = text +','
      else:
        text = text +') {'
      if n%n_per_line == 3 and n != nargs-1:
         text = text +'\n'
    code(text);
    config.depth = 2

    code('');

    text = f'ops_arg args[{nargs}] = {{'
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
    ENDIF()
    code('')
    comm('compute localy allocated range for the sub-block')

    code(f'int start[{NDIM}];')
    code(f'int end[{NDIM}];')
    code(f'int arg_idx[{NDIM}];')
    code('')

    code('#ifdef OPS_MPI')
    code(f'if (compute_ranges(args, {nargs},block, range, start, end, arg_idx) < 0) return;')
    code('#else')
    FOR('n','0',str(NDIM))
    code('start[n] = range[2*n];end[n] = range[2*n+1];')
    code('arg_idx[n] = start[n];')
    ENDFOR()
    code('#endif')

    code('')
    code('int x_size = MAX(0,end[0]-start[0]);')
    if NDIM==2:
      code('int y_size = MAX(0,end[1]-start[1]);')
    if NDIM==3:
      code('int y_size = MAX(0,end[1]-start[1]);')
      code('int z_size = MAX(0,end[2]-start[2]);')
    code('')

    for n in range (0,nargs):
      if arg_typ[n] == 'ops_arg_dat':
        code(f'int xdim{n} = args[{n}].dat->size[0];')
        if NDIM>2 or (NDIM==2 and soa_set):
          code(f'int ydim{n} = args[{n}].dat->size[1];')
        if NDIM>3 or (NDIM==3 and soa_set):
          code(f'int zdim{n} = args[{n}].dat->size[2];')
    #timing structs
    code('')
    comm('Timing')
    code('double t1,t2,c1,c2;')
    IF('block->instance->OPS_diags > 1')
    code('ops_timers_core(&c2,&t2);')
    ENDIF()
    code('')
    condition = ''
    for n in range (0, nargs):
      if arg_typ[n] == 'ops_arg_dat':
        condition += f'xdim{n} != xdim{n}_{name}_h || '
        if NDIM>2 or (NDIM==2 and soa_set):
          condition += f'ydim{n} != ydim{n}_{name}_h || '
        if NDIM>3 or (NDIM==3 and soa_set):
          condition += f'zdim{n} != zdim{n}_{name}_h || '
    condition = condition[:-4]
    IF(condition)

    for n in range (0, nargs):
      if arg_typ[n] == 'ops_arg_dat':
        code(f'xdim{n}_{name} = xdim{n};')
        code(f'xdim{n}_{name}_h = xdim{n};')
        if NDIM>2 or (NDIM==2 and soa_set):
          code(f'ydim{n}_{name} = ydim{n};')
          code(f'ydim{n}_{name}_h = ydim{n};')
        if NDIM>3 or (NDIM==3 and soa_set):
          code(f'zdim{n}_{name} = zdim{n};')
          code(f'zdim{n}_{name}_h = zdim{n};')
    ENDIF()
    code('')

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

    if MULTI_GRID:
      code(f'int global_idx[{NDIM}];')
      code('#ifdef OPS_MPI')
      for n in range (0,NDIM):
        code(f'global_idx[{n}] = arg_idx[{n}];')
      code('#else //OPS_MPI')
      for n in range (0,NDIM):
        code(f'global_idx[{n}] = start[{n}];')
      code('#endif //OPS_MPI')
      code('')

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

    code('')
    comm('set up initial pointers and exchange halos if necessary')
    for n in range (0, nargs):
      if arg_typ[n] == 'ops_arg_dat':
        if prolong[n] == 1 or restrict[n] == 1:
          starttext = 'start_'+str(n)
        else:
          starttext = 'start'
        code(f'long long int base{n} = args[{n}].dat->base_offset + (long long int)(block->instance->OPS_soa ? args[{n}].dat->type_size : args[{n}].dat->elem_size) * {starttext}[0] * args[{n}].stencil->stride[0];')

        for d in range (1, NDIM):
          line = f'base{n} = base{n}+ (long long int)(block->instance->OPS_soa ? args[{n}].dat->type_size : args[{n}].dat->elem_size) *\n'
          for d2 in range (0,d):
            line += config.depth*' '+f'  args[{n}].dat->size[{d2}] *\n'
          code(line[:-1])
          code(f'  {starttext}[{d}] * args[{n}].stencil->stride[{d}];')

        code(f'{typs[n]} *p_a{n} = ({typs[n]} *)(args[{n}].data + base{n});')

      elif arg_typ[n] == 'ops_arg_gbl':
        if accs[n] == OPS_READ:
          code(f'{typs[n]} *p_a{n} = ({typs[n]} *)args[{n}].data;')
        else:
          code('#ifdef OPS_MPI')
          code(f'{typs[n]} *p_a{n} = ({typs[n]} *)(((ops_reduction)args[{n}].data)->data + ((ops_reduction)args[{n}].data)->size * block->index);')
          code('#else')
          code(f'{typs[n]} *p_a{n} = ({typs[n]} *)(((ops_reduction)args[{n}].data)->data);')
          code('#endif')
        code('')
      else:
        code(f'{typs[n]} *p_a{n} = NULL;')
      code('')
    code('')

    code('')
    code(f'ops_H_D_exchanges_host(args, {nargs});')
    code(f'ops_halo_exchanges(args,{nargs},range);')
    code('')
    IF('block->instance->OPS_diags > 1')
    code('ops_timers_core(&c1,&t1);')
    code(f'block->instance->OPS_kernels[{nk}].mpi_time += t1-t2;')
    ENDIF()
    code('')


    code(name+'_c_wrapper(')
    config.depth = config.depth+2
    for n in range (0, nargs):
      code(f'p_a{n},')
    if MULTI_GRID:
      code('global_idx,')
    for n in range(0,nargs):
      if restrict[n] == 1 or prolong[n] == 1:
        code(f'stride_{n},')
    if arg_idx:
      if NDIM==1:
        code('arg_idx[0],')
      elif NDIM==2:
        code('arg_idx[0], arg_idx[1],')
      elif NDIM==3:
        code('arg_idx[0], arg_idx[1], arg_idx[2],')

    if NDIM == 1:
      code('x_size);')
    if NDIM == 2:
      code('x_size, y_size);')
    if NDIM == 3:
      code('x_size, y_size, z_size);')

    config.depth = config.depth-2


    code('')
    IF('block->instance->OPS_diags > 1')
    code('ops_timers_core(&c2,&t2);')
    code(f'block->instance->OPS_kernels[{nk}].time += t2-t1;')
    ENDIF()


    code(f'ops_set_dirtybit_host(args, {nargs});')
    for n in range (0, nargs):
      if arg_typ[n] == 'ops_arg_dat' and (accs[n] == OPS_WRITE or accs[n] == OPS_RW or accs[n] == OPS_INC):
        code(f'ops_set_halo_dirtybit3(&args[{n}],range);')

    code('')
    comm('Update kernel record')
    IF('block->instance->OPS_diags > 1')
    for n in range (0, nargs):
      if arg_typ[n] == 'ops_arg_dat':
        code(f'block->instance->OPS_kernels[{nk}].transfer += ops_compute_transfer(dim, start, end, &arg{n});')
    ENDIF()
    config.depth = config.depth - 2
    code('}')


    ##########################################################################
    #  output individual kernel file
    ##########################################################################
    util.write_text_to_file(f"./MPI_inline/{name}_mpiinline_kernel.cpp")

  # end of main kernel call loop

  ##########################################################################
  #  output one master kernel file
  ##########################################################################

  comm('header')
  code('#include <math.h>')
  code('#define OPS_API 2')
  if NDIM==1:
    code('#define OPS_1D')
  if NDIM==2:
    code('#define OPS_2D')
  if NDIM==3:
    code('#define OPS_3D')
  if soa_set:
    code('#define OPS_SOA')
  code('#include "ops_macros.h"')
  code('#ifdef __cplusplus')
  code('#include "ops_lib_core.h"')
  code('#endif')
  code('#if defined(OPS_MPI) && defined(__cplusplus)')
  code('#include "ops_mpi_core.h"')
  code('#endif')
  if os.path.exists(os.path.join(src_dir,'user_types.h')):
    code('#include "user_types.h"')
  code('')

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

  util.write_text_to_file(f"./MPI_inline/{master_basename[0]}_common.h")

  code(f'#include "./MPI_inline/{master_basename[0]}_common.h"')
  code('')
  code('')
  code('void ops_init_backend() {}')
  code('')
  comm('user kernel files')

  for kernel_name in map(lambda kernel: kernel['name'], kernels):
      code(f"#include \"{kernel_name}_mpiinline_kernel.cpp\"")

  util.write_text_to_file(f"./MPI_inline/{master_basename[0]}_kernels.cpp")

  if NDIM==1:
    code('#define OPS_1D')
  if NDIM==2:
    code('#define OPS_2D')
  if NDIM==3:
    code('#define OPS_3D')
  code('#include <math.h>')
  code(f"#include \"./MPI_inline/{master_basename[0]}_common.h\"")
  comm('user kernel files')

  for kernel_name in map(lambda kernel: kernel['name'], kernels):
      code(f"#include \"{kernel_name}_mpiinline_kernel_c.c\"")

  util.write_text_to_file(f"./MPI_inline/{master_basename[0]}_kernels_c.c")
