
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
## @brief OPS MPI_seq code generator
#
#  This routine is called by ops.py which parses the input files
#
#  It produces a file xxx_cpu_kernel.cpp for each kernel,
#  plus a master kernel file
#

"""
OPS MPI_seq code generator

This routine is called by ops.py which parses the input files

It produces a file xxx_cpu_kernel.cpp for each kernel,
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
check_accs = util.check_accs
mult = util.mult
convert_ACC_body = util.convert_ACC_body
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



def ops_gen_mpi_lazy(master, date, consts, kernels, soa_set):

  global dims, stens
  global g_m, file_text, depth

  OPS_ID   = 1;  OPS_GBL   = 2;  OPS_MAP = 3;

  OPS_READ = 1;  OPS_WRITE = 2;  OPS_RW  = 3;
  OPS_INC  = 4;  OPS_MAX   = 5;  OPS_MIN = 6;

  accsstring = ['OPS_READ','OPS_WRITE','OPS_RW','OPS_INC','OPS_MAX','OPS_MIN' ]

  NDIM = 2 #the dimension of the application is hardcoded here .. need to get this dynamically

  gen_full_code = 1;

  src_dir = os.path.dirname(master) or '.'
  master_basename = os.path.splitext(os.path.basename(master))


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
    stride = [1] * (nargs+4) * (NDIM+1)
    restrict = [1] * nargs
    prolong = [1] * nargs

    stride = ops_gen_common.parse_strides(stens, nargs, stride, NDIM)
            
    ### Determine if this is a MULTI_GRID LOOP with
    ### either restrict or prolong
    MULTI_GRID = 0
    for n in range (0, nargs):
      dims[n] = str(dims[n])
      restrict[n] = 0
      prolong[n] = 0
      if str(stens[n]).find('RESTRICT') > 0:
        restrict[n] = 1
        MULTI_GRID = 1
      if str(stens[n]).find('PROLONG') > 0 :
        prolong[n] = 1
        MULTI_GRID = 1

    reduction = 0
    for n in range (0, nargs):
      if arg_typ[n] == 'ops_arg_gbl' and accs[n] <> OPS_READ:
        reduction = 1

    arg_idx = -1
    for n in range (0, nargs):
      if arg_typ[n] == 'ops_arg_idx':
        arg_idx = n

    config.file_text = ''
    config.depth = 0
    n_per_line = 4

    i = name.find('kernel')
    name2 = name[0:i-1]

##########################################################################
#  start with seq kernel function
##########################################################################

    code('')

    ret = ops_gen_common.get_user_function(name, arg_typ, src_dir)
    kernel_text = ret[0]
    arg_list = ret[1]
    
    ops_gen_common.generate_header(nk, name, nargs, arg_typ, accs, arg_idx, NDIM, MULTI_GRID, gen_full_code)
    
    ops_gen_common.generate_sizes_bounds(nargs, arg_typ, NDIM)

    stride = ops_gen_common.generate_strides(nargs, stens, stride, NDIM)

    code('')

    ops_gen_common.generate_exchanges(nargs, nk, gen_full_code, 'host')

    ops_gen_common.generate_pointers(nargs, arg_typ, accs, typs, arg_list, restrict, prolong, dims)

    code('#if defined(_OPENMP) && defined(OPS_BATCHED) && !defined(OPS_LAZY) && OPS_BATCHED=='+str(NDIM))
    code('#pragma omp parallel for')
    code('#endif')
    FOR('n_'+str(NDIM),'bounds_'+str(NDIM)+'_l','bounds_'+str(NDIM)+'_u')

    if reduction:
      code('#if OPS_BATCHED=='+str(NDIM)+' || !defined(OPS_BATCHED)')
      for n in range (0,nargs):
        if arg_typ[n] == 'ops_arg_gbl':
          if accs[n] <> OPS_READ:
            for d in range(0,int(dims[n])):
                code(typs[n]+' p_a'+str(n)+'_'+str(d)+' = p_a'+str(n)+'[n_'+str(NDIM)+'*'+dims[n]+'+'+str(d)+'];')
      code('#endif')

    line = ''
    for n in range (0,nargs):
      if arg_typ[n] == 'ops_arg_gbl':
        if accs[n] == OPS_MIN:
          for d in range(0,int(dims[n])):
            line = line + ' reduction(min:p_a'+str(n)+'_'+str(d)+')'
        if accs[n] == OPS_MAX:
          for d in range(0,int(dims[n])):
            line = line + ' reduction(max:p_a'+str(n)+'_'+str(d)+')'
        if accs[n] == OPS_INC:
          for d in range(0,int(dims[n])):
            line = line + ' reduction(+:p_a'+str(n)+'_'+str(d)+')'
        if accs[n] == OPS_WRITE: #this may not be correct ..
          for d in range(0,int(dims[n])):
            line = line + ' reduction(+:p_a'+str(n)+'_'+str(d)+')'
    if NDIM==3 and reduction==0:
      line2 = ' collapse(2)'
    else:
      line2 = line
    code('#if defined(_OPENMP) && !defined(OPS_BATCHED)')
    code('#pragma omp parallel for'+line2)
    code('#endif')
    if NDIM>2:
      FOR('n_2','bounds_2_l','bounds_2_u')

      if reduction:
        code('#if OPS_BATCHED==2')
        for n in range (0,nargs):
          if arg_typ[n] == 'ops_arg_gbl':
            if accs[n] <> OPS_READ:
              for d in range(0,int(dims[n])):
                code(typs[n]+' p_a'+str(n)+'_'+str(d)+' = p_a'+str(n)+'[n_'+str(2)+'*'+dims[n]+'+'+str(d)+'];')
        code('#endif')
    if NDIM>1:
      FOR('n_1','bounds_1_l','bounds_1_u')
      if reduction:
        code('#if OPS_BATCHED==1')
        for n in range (0,nargs):
          if arg_typ[n] == 'ops_arg_gbl':
            if accs[n] <> OPS_READ:
              for d in range(0,int(dims[n])):
                code(typs[n]+' p_a'+str(n)+'_'+str(d)+' = p_a'+str(n)+'[n_'+str(1)+'*'+dims[n]+'+'+str(d)+'];')
        code('#endif')

    line3 = ''
    for n in range (0,nargs):
      if arg_typ[n] == 'ops_arg_dat':
        line3 = line3 +arg_list[n]+','
    if NDIM>1:
      code('#ifdef __INTEL_COMPILER')
      code('#pragma loop_count(10000)')
      if reduction:
        code('#if defined(OPS_BATCHED) && OPS_BATCHED==0')
        code('#pragma omp simd') #+' aligned('+clean_type(line3[:-1])+')')
        code('#else')
        code('#pragma omp simd'+line) #+' aligned('+clean_type(line3[:-1])+')')
        code('#endif')
      else:
        code('#pragma omp simd'+line) #+' aligned('+clean_type(line3[:-1])+')')
      code('#elif defined(__clang__)')
      code('#pragma clang loop vectorize(assume_safety)')
      code('#elif defined(__GNUC__)')
      code('#pragma simd')
      code('#pragma GCC ivdep')
      code('#else')
      code('#pragma simd')
      code('#endif')
    FOR('n_0','bounds_0_l','bounds_0_u')
    if reduction:
      code('#if defined(OPS_BATCHED) && OPS_BATCHED==0')
      for n in range (0,nargs):
        if arg_typ[n] == 'ops_arg_gbl':
          if accs[n] <> OPS_READ:
            for d in range(0,int(dims[n])):
              code(typs[n]+' p_a'+str(n)+'_'+str(d)+' = p_a'+str(n)+'[n_'+str(0)+'*'+dims[n]+'+'+str(d)+'];')
      code('#endif')
    if arg_idx <> -1:
      ops_gen_common.generate_arg_idx(arg_idx, arg_list, NDIM)

    def dimstr(n, d):
      dimlabels = 'xyzuv'
      return dimlabels[d]+'dim'+str(n)

    ops_gen_common.generate_accessors(nargs, arg_typ, dims, NDIM, stride, typs, accs, arg_list, restrict, prolong, dimstr)

    ops_gen_common.generate_gbl_locals(nargs, arg_typ, accs, dims, typs, arg_list)
 
    #insert user kernel
    code(kernel_text);

    for n in range (0,nargs):
      if arg_typ[n] == 'ops_arg_gbl':
        if accs[n] == OPS_MIN:
          for d in range(0,int(dims[n])):
            code('p_a'+str(n)+'_'+str(d)+' = MIN(p_a'+str(n)+'_'+str(d)+','+arg_list[n]+'['+str(d)+']);')
        if accs[n] == OPS_MAX:
          for d in range(0,int(dims[n])):
            code('p_a'+str(n)+'_'+str(d)+' = MAX(p_a'+str(n)+'_'+str(d)+','+arg_list[n]+'['+str(d)+']);')
        if accs[n] == OPS_INC:
          for d in range(0,int(dims[n])):
            code('p_a'+str(n)+'_'+str(d)+' +='+arg_list[n]+'['+str(d)+'];')
        if accs[n] == OPS_WRITE: #this may not be correct
          for d in range(0,int(dims[n])):
            code('p_a'+str(n)+'_'+str(d)+' +='+arg_list[n]+'['+str(d)+'];')

    if reduction:
      code('#if defined(OPS_BATCHED) && OPS_BATCHED==0')
      for n in range (0, nargs):
        if arg_typ[n] == 'ops_arg_gbl':
          if accs[n] <> OPS_READ:
            for d in range(0,int(dims[n])):
              code('p_a'+str(n)+'[n_'+str(0)+'*'+dims[n]+'+'+str(d)+'] = p_a'+str(n)+'_'+str(d)+';')
      code('#endif')
    ENDFOR()
    if NDIM>1:
      if reduction:
        code('#if OPS_BATCHED==1')
        for n in range (0, nargs):
          if arg_typ[n] == 'ops_arg_gbl':
            if accs[n] <> OPS_READ:
              for d in range(0,int(dims[n])):
                code('p_a'+str(n)+'[n_'+str(1)+'*'+dims[n]+'+'+str(d)+'] = p_a'+str(n)+'_'+str(d)+';')
        code('#endif')
      ENDFOR()
    if NDIM>2:
      if reduction:
        code('#if OPS_BATCHED==2')
        for n in range (0, nargs):
          if arg_typ[n] == 'ops_arg_gbl':
            if accs[n] <> OPS_READ:
              for d in range(0,int(dims[n])):
                code('p_a'+str(n)+'[n_'+str(2)+'*'+dims[n]+'+'+str(d)+'] = p_a'+str(n)+'_'+str(d)+';')
        code('#endif')

      ENDFOR()


    code('#if OPS_BATCHED=='+str(NDIM)+' || !defined(OPS_BATCHED)')
    for n in range (0, nargs):
      if arg_typ[n] == 'ops_arg_gbl':
        if accs[n] <> OPS_READ:
          for d in range(0,int(dims[n])):
            code('p_a'+str(n)+'[n_'+str(NDIM)+'*'+dims[n]+'+'+str(d)+'] = p_a'+str(n)+'_'+str(d)+';')
    code('#endif')


    ENDFOR() #batches

    ops_gen_common.generate_tail(nk, nargs, arg_typ, accs, gen_full_code, 'host')

    code('')
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
    if not os.path.exists('./MPI_OpenMP'):
      os.makedirs('./MPI_OpenMP')
    fid = open('./MPI_OpenMP/'+name+'_cpu_kernel.cpp','w')
    date = datetime.datetime.now()
    fid.write('//\n// auto-generated by ops.py\n//\n')
    fid.write(config.file_text)
    fid.close()

# end of main kernel call loop

##########################################################################
#  output one master kernel file
##########################################################################
  config.depth = 0
  config.file_text =''
  comm('header')
  if NDIM==1:
    code('#define OPS_1D')
  if NDIM==2:
    code('#define OPS_2D')
  if NDIM==3:
    code('#define OPS_3D')
  if soa_set:
    code('#define OPS_SOA')
  code('#define OPS_API 2')
  code('#include "ops_lib_cpp.h"')
  code('#ifdef OPS_MPI')
  code('#include "ops_mpi_core.h"')
  code('#endif')
  if os.path.exists(os.path.join(src_dir, 'user_types.h')):
    code('#include "user_types.h"')
  code('')

  comm(' global constants')
  for nc in range (0,len(consts)):
    if consts[nc]['dim'].isdigit() and int(consts[nc]['dim'])==1:
      code('extern '+consts[nc]['type']+' '+(str(consts[nc]['name']).replace('"','')).strip()+';')
    else:
      if consts[nc]['dim'].isdigit() and consts[nc]['dim'] > 0:
        num = str(consts[nc]['dim'])
        code('extern '+consts[nc]['type']+' '+(str(consts[nc]['name']).replace('"','')).strip()+'['+num+'];')
      else:
        code('extern '+consts[nc]['type']+' *'+(str(consts[nc]['name']).replace('"','')).strip()+';')

  code('')
  code('void ops_init_backend() {}')
  code('')
  comm('user kernel files')

  kernel_name_list = []

  for nk in range(0,len(kernels)):
    if kernels[nk]['name'] not in kernel_name_list :
      code('#include "'+kernels[nk]['name']+'_cpu_kernel.cpp"')
      kernel_name_list.append(kernels[nk]['name'])

  fid = open('./MPI_OpenMP/'+master_basename[0]+'_cpu_kernels.cpp','w')
  fid.write('//\n// auto-generated by ops.py//\n\n')
  fid.write(config.file_text)
  fid.close()
