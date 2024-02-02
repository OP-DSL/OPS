

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
## @brief OPS MPI omp code generator with adjoint calculations
#
#  This routine is called by ops.py which parses the input files
#
#  It produces a file xxx_cpu_kernel.cpp for each kernel,
#  plus a master kernel file
#


"""
OPS MPI OpenMP adjoint code generator

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

verbose = util.verbose

para_parse = util.para_parse
comment_remover = util.comment_remover
remove_trailing_w_space = util.remove_trailing_w_space
parse_signature = util.parse_signature
check_accs = util.check_accs
mult = util.mult
convert_ACC_body = util.convert_ACC_body
find_kernel = util.find_kernel
find_kernel_with_retry = util.find_kernel_with_retry

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


def clean_type(arg):
    for qual in ['__restrict__', 'RESTRICT', '__volatile__']:
        arg = arg.replace(qual, '')
    return arg

def get_idxs(arg_list_with_adjoints, orig_arg_list):
    idxs = [0]*len(arg_list_with_adjoints)
    for i in range(0, len(idxs)):
        if arg_list_with_adjoints[i][-4:] == "_a1s":
            idxs[i] = - orig_arg_list.index(arg_list_with_adjoints[i][0:-4]) - 1
        else:
            idxs[i] = orig_arg_list.index(arg_list_with_adjoints[i])
    return idxs

def get_ad_idx(n, arg_idxs_with_adjoints):
  n_ad = -n-1;
  try:
    return arg_idxs_with_adjoints.index(n_ad)
  except ValueError:
    return -1

def gen_kernel(arg_typ, name, nargs, accs, typs, NDIM, stride, restrict, prolong, MULTI_GRID, reduction, arg_idx, gen_full_code, n_per_line, nk, soa_set, src_dir):
    global dims, stens
    global g_m, file_text, depth

    OPS_READ = 1;  OPS_WRITE = 2;  OPS_RW  = 3;
    OPS_INC  = 4;  OPS_MAX   = 5;  OPS_MIN = 6;
 
    accsstring = ['OPS_READ','OPS_WRITE','OPS_RW','OPS_INC','OPS_MAX','OPS_MIN' ]

    enable_mpi = False; # for clean codes to show other people and since we don't need MPI yet
    code('')
    comm('user function')
    
    kernel_text, arg_list = find_kernel(src_dir, name)
    comm('')
    comm(' host stub function')
    code('void ops_par_loop_'+name+'_execute(ops_kernel_descriptor *desc) {')
    config.depth = 2
    #code('char const *name = "'+name+'";')
    code('ops_block block = desc->block;')
    code('int dim = desc->dim;')
    code('int *range = desc->range;')
    
    for n in range (0, nargs):
      code('ops_arg arg'+str(n)+' = desc->args['+str(n)+'];')


    code('')
    if gen_full_code == 1:
      comm('Timing')
      code('double __t1,__t2,__c1,__c2;')
      code('')

    
    text ='ops_arg args['+str(nargs)+'] = {'
    for n in range (0, nargs):
      text = text +' arg'+str(n)
      if nargs != 1 and n != nargs-1:
        text = text +','
      else:
        text = text +'};\n\n'
      if n%n_per_line == 5 and n != nargs-1:
        text = text +'\n                    '
    code(text);
    code('')
    
    if gen_full_code:
      IF('block->instance->OPS_diags > 1')
      code('ops_timing_realloc(block->instance,'+str(nk)+',"'+name+'");')
      code('block->instance->OPS_kernels['+str(nk)+'].count++;')
      code('ops_timers_core(&__c2,&__t2);')
      ENDIF()
      code('')

    code('#ifdef OPS_DEBUG')
    code('ops_register_args(args, "'+name+'");')
    code('#endif')
    code('')

    comm('compute locally allocated range for the sub-block')
    code('int start['+str(NDIM)+'];')
    code('int end['+str(NDIM)+'];')
    if enable_mpi and not (arg_idx!=-1 or MULTI_GRID):
      code('#ifdef OPS_MPI')
    code('int arg_idx['+str(NDIM)+'];')
    if enable_mpi and not (arg_idx!=-1 or MULTI_GRID):
      code('#endif')
    
    if enable_mpi:
      code('#if defined(OPS_LAZY) || !defined(OPS_MPI)')
    
    FOR('n','0',str(NDIM))
    code('start[n] = range[2*n];end[n] = range[2*n+1];')
    ENDFOR()
    
    if enable_mpi:
      code('#else')
      code('if (compute_ranges(args, '+str(nargs)+',block, range, start, end, arg_idx) < 0) return;')
      code('#endif')

    code('')

    if arg_idx != -1 or MULTI_GRID:
      if enable_mpi:
        code('#ifdef OPS_MPI')
        arg_write = -1
        for n in range(0,nargs):
          if arg_typ[n] == 'ops_arg_dat' and accs[n] != OPS_READ:
            arg_write = n
        if arg_write == -1:
          code('sub_block_list sb = OPS_sub_block_list[block->index];')
          for n in range (0,NDIM):
            code('arg_idx['+str(n)+'] = sb->decomp_disp['+str(n)+'];')
        else:
          code('sub_dat_list sd = OPS_sub_dat_list[args['+str(arg_write)+'].dat->index];')
          for n in range (0,NDIM):
            code('arg_idx['+str(n)+'] = MAX(0,sd->decomp_disp['+str(n)+']);')

        code('#else //OPS_MPI')
      for n in range (0,NDIM):
        code('arg_idx['+str(n)+'] = 0;')
      if enable_mpi:
        code('#endif //OPS_MPI')


    code('')
    comm("initialize global variable with the dimension of dats")
    for n in range (0, nargs):
      if arg_typ[n] == 'ops_arg_dat':
        if NDIM>1:
          code('int xdim'+str(n)+'_'+name+' = args['+str(n)+'].dat->size[0];')#*args['+str(n)+'].dat->dim;')
        if NDIM>2 or (NDIM==2 and soa_set):
          code('int ydim'+str(n)+'_'+name+' = args['+str(n)+'].dat->size[1];')
        if NDIM>3 or (NDIM==3 and soa_set):
          code('int zdim'+str(n)+'_'+name+' = args['+str(n)+'].dat->size[2];')


    code('')
    comm('set up initial pointers and exchange halos if necessary')
    for n in range (0, nargs):
      if arg_typ[n] == 'ops_arg_dat':
          code('int base'+str(n)+' = args['+str(n)+'].dat->base_offset;')
          code(typs[n]+' * __restrict__ '+clean_type(arg_list[n])+'_p = ('+typs[n]+' *)(args['+str(n)+'].data + base'+str(n)+');')
          if restrict[n] == 1 or prolong[n] == 1:
            code('#ifdef OPS_MPI')
            code('sub_dat_list sd'+str(n)+' = OPS_sub_dat_list[args['+str(n)+'].dat->index];')
          if restrict[n] == 1:
            code(clean_type(arg_list[n])+' += arg_idx[0]*args['+str(n)+'].stencil->mgrid_stride[0] - sd'+str(n)+'->decomp_disp[0] + args['+str(n)+'].dat->d_m[0];')
            if NDIM>1:
              code(clean_type(arg_list[n])+' += (arg_idx[1]*args['+str(n)+'].stencil->mgrid_stride[1] - sd'+str(n)+'->decomp_disp[1] + args['+str(n)+'].dat->d_m[1])*xdim'+str(n)+'_'+name+';')
            if NDIM>2:
              code(clean_type(arg_list[n])+' += (arg_idx[2]*args['+str(n)+'].stencil->mgrid_stride[2] - sd'+str(n)+'->decomp_disp[2] + args['+str(n)+'].dat->d_m[2])*xdim'+str(n)+'_'+name+' * ydim'+str(n)+'_'+name+';')
          if prolong[n] == 1:
            code(clean_type(arg_list[n])+' += arg_idx[0]/args['+str(n)+'].stencil->mgrid_stride[0] - sd'+str(n)+'->decomp_disp[0] + args['+str(n)+'].dat->d_m[0];')
            if NDIM>1:
              code(clean_type(arg_list[n])+' += (arg_idx[1]/args['+str(n)+'].stencil->mgrid_stride[1] - sd'+str(n)+'->decomp_disp[1] + args['+str(n)+'].dat->d_m[1])*xdim'+str(n)+'_'+name+';')
            if NDIM>2:
              code(clean_type(arg_list[n])+' += (arg_idx[2]/args['+str(n)+'].stencil->mgrid_stride[2] - sd'+str(n)+'->decomp_disp[2] + args['+str(n)+'].dat->d_m[2])*xdim'+str(n)+'_'+name+' * ydim'+str(n)+'_'+name+';')

          if restrict[n] == 1 or prolong[n] == 1:
            code('#endif')
      elif arg_typ[n] == 'ops_arg_gbl':
        if accs[n] == OPS_READ:
          code(typs[n]+' * __restrict__ '+clean_type(arg_list[n])+' = ('+typs[n]+' *)args['+str(n)+'].data;')
        else:
          if enable_mpi:
            code('#ifdef OPS_MPI')
            code(typs[n]+' * __restrict__ p_a'+str(n)+' = ('+typs[n]+' *)(((ops_reduction)args['+str(n)+'].data)->data + ((ops_reduction)args['+str(n)+'].data)->size * block->index);')
            code('#else //OPS_MPI')
          code(typs[n]+' * __restrict__ p_a'+str(n)+' = ('+typs[n]+' *)((ops_reduction)args['+str(n)+'].data)->data;')
          if enable_mpi:
            code('#endif //OPS_MPI')
        code('')
      if arg_typ[n] == 'ops_arg_scalar':
        if accs[n] == OPS_READ:
          code(typs[n]+' * __restrict__ '+clean_type(arg_list[n])+' = ('+typs[n]+' *)((ops_scalar)args['+str(n)+'].data)->data;')
        else:
          print('Warning ops_scalar type with non read-only access not implemented\n')
      code('')
    code('')

    code('')
    if gen_full_code==1:
      IF('block->instance->OPS_diags > 1')
      code('ops_timers_core(&__c1,&__t1);')
      code('block->instance->OPS_kernels['+str(nk)+'].mpi_time += __t1-__t2;')
      ENDIF()
      code('')

    for n in range (0,nargs):
      if arg_typ[n] == 'ops_arg_gbl':
        if accs[n] != OPS_READ:
          for d in range(0,int(dims[n])):
            code(typs[n]+' p_a'+str(n)+'_'+str(d)+' = p_a'+str(n)+'['+str(d)+'];')

    line = ''
    if NDIM>2:
      FOR('n_z','start[2]','end[2]')
    if NDIM>1:
      FOR('n_y','start[1]','end[1]')

    FOR('n_x','start[0]','end[0]')
    if arg_idx != -1:
      if NDIM==1:
        code('int '+clean_type(arg_list[arg_idx])+'[] = {0};')
      elif NDIM==2:
        code('int '+clean_type(arg_list[arg_idx])+'[] = {0, arg_idx[1]+n_y};')
      elif NDIM==3:
        code('int '+clean_type(arg_list[arg_idx])+'[] = {0, arg_idx[1]+n_y, arg_idx[2]+n_z};')

    if arg_idx != -1:
      code(clean_type(arg_list[arg_idx])+'[0] = arg_idx[0]+n_x;')

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
            dim = dims[n]+', '
            extradim = 1
        elif not dims[n].isdigit():
            dim = 'arg'+str(n)+'.dim, '
            extradim = 1
        if NDIM > 0:
          offset = offset + 'n_x*'+str(stride[NDIM*n])
        if NDIM > 1:
          offset = offset + ' + n_y * xdim'+str(n)+'_'+name+'*'+str(stride[NDIM*n+1])
        if NDIM > 2:
          offset = offset + ' + n_z * xdim'+str(n)+'_'+name+' * ydim'+str(n)+'_'+name+'*'+str(stride[NDIM*n+2])
        dimlabels = 'xyzuv'
        for i in range(1,NDIM+extradim):
          sizelist = sizelist + dimlabels[i-1]+'dim'+str(n)+'_'+name+', '

        if not dims[n].isdigit() or int(dims[n])>1:
          code('#ifdef OPS_SOA')
        code(pre + 'ACC<'+typs[n]+'> '+arg_list[n]+'('+dim+sizelist+arg_list[n]+'_p + '+offset+');')
        if not dims[n].isdigit() or int(dims[n])>1:
          code('#else')
          code(pre + 'ACC<'+typs[n]+'> '+arg_list[n]+'('+dim+sizelist+arg_list[n]+'_p + '+dim[:-2]+'*('+offset+'));')
          code('#endif')
    for n in range (0,nargs):
      if arg_typ[n] == 'ops_arg_gbl':
        if accs[n] == OPS_MIN:
          code(typs[n]+' '+arg_list[n]+'['+str(dims[n])+'];')
          for d in range(0,int(dims[n])):
            code(arg_list[n]+'['+str(d)+'] = p_a'+str(n)+'['+str(d)+'];') #need +INFINITY_ change to
        if accs[n] == OPS_MAX:
          code(typs[n]+' '+arg_list[n]+'['+str(dims[n])+'];')
          for d in range(0,int(dims[n])):
            code(arg_list[n]+'['+str(d)+'] = p_a'+str(n)+'['+str(d)+'];') #need -INFINITY_ change to
        if accs[n] == OPS_INC:
          code(typs[n]+' '+arg_list[n]+'['+str(dims[n])+'];')
          for d in range(0,int(dims[n])):
            code(arg_list[n]+'['+str(d)+'] = ZERO_'+typs[n]+';')
        if accs[n] == OPS_WRITE: #this may not be correct
          code(typs[n]+' '+arg_list[n]+'['+str(dims[n])+'];')
          for d in range(0,int(dims[n])):
            code(arg_list[n]+'['+str(d)+'] = ZERO_'+typs[n]+';')

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

    ENDFOR()
    if NDIM>1:
      ENDFOR()
    if NDIM>2:
      ENDFOR()

    for n in range (0, nargs):
      if arg_typ[n] == 'ops_arg_gbl':
        if accs[n] != OPS_READ:
          for d in range(0,int(dims[n])):
            code('p_a'+str(n)+'['+str(d)+'] = p_a'+str(n)+'_'+str(d)+';')

    if gen_full_code==1:
      IF('block->instance->OPS_diags > 1')
      code('ops_timers_core(&__c2,&__t2);')
      code('block->instance->OPS_kernels['+str(nk)+'].time += __t2-__t1;')
      ENDIF()
      code('')
      IF('block->instance->OPS_diags > 1')
      comm('Update kernel record')
      code('ops_timers_core(&__c1,&__t1);')
      code('block->instance->OPS_kernels['+str(nk)+'].mpi_time += __t1-__t2;')
      for n in range (0, nargs):
        if arg_typ[n] == 'ops_arg_dat':
          code('block->instance->OPS_kernels['+str(nk)+'].transfer += ops_compute_transfer(dim, start, end, &arg'+str(n)+');')
      ENDIF()
    config.depth = config.depth - 2
    code('}')
    code('')

def gen_adjoint_kernel(arg_typ, name, nargs, accs, typs, NDIM, stride, restrict, prolong, MULTI_GRID, reduction, arg_idx, n_per_line, nk, soa_set, src_dir):
    global dims, stens
    global g_m, file_text, depth
 
    OPS_READ = 1;  OPS_WRITE = 2;  OPS_RW  = 3; # TODO globallll
    OPS_INC  = 4;  OPS_MAX   = 5;  OPS_MIN = 6;
 
    accsstring = ['OPS_READ','OPS_WRITE','OPS_RW','OPS_INC','OPS_MAX','OPS_MIN' ]

    code('')
    adjoint_name = name + "_adjoint"
    kernel_text, arg_list_with_adjoints = find_kernel_with_retry(src_dir, adjoint_name)
    orig_kernel_func, arg_list = find_kernel(src_dir, name)
    if kernel_text is None:
        if verbose:
            print(
                f"Couldn't find adjoint for kernel for {name}. Generate as passive loop."
            )
        return None
    arg_idxs_with_adjoints = get_idxs(arg_list_with_adjoints, arg_list) # an array can be indexed with the adjoint arg idx value: indexes from -n-1 to n. negative value -i stands for adjoint of the i-1th argument
    kernel_text = kernel_text[kernel_text.find('{')+1:-1]
    comm('')
    comm(' host stub function')
    code('void ops_par_loop_'+adjoint_name+'_execute(ops_kernel_descriptor *desc) {')
    config.depth = 2
    #code('char const *name = "'+name+'";')
    code('ops_block block = desc->block;')
    code('int dim = desc->dim;')
    code('int *range = desc->range;')
    
    code('ops_arg args['+str(nargs)+'] = {'+','.join(['desc->args['+str(i)+']' for i in range(0, nargs)]) + '};')
    code('')

    comm('compute locally allocated range for the sub-block')
    code('int start['+str(NDIM)+'];')
    code('int end['+str(NDIM)+'];')
    code('int arg_idx['+str(NDIM)+'] = {0'+(',0'*(NDIM-1))+'};')
    FOR('n','0',str(NDIM))
    code('start[n] = range[2*n];end[n] = range[2*n+1];')
    ENDFOR()
 
    code('')
    comm("initialize global variable with the dimension of dats")
    for n in range (0, nargs):
      if arg_typ[n] == 'ops_arg_dat':
        if NDIM>1:
          code('int xdim'+str(n)+'_'+name+' = args['+str(n)+'].dat->size[0];')#*args['+str(n)+'].dat->dim;')
        if NDIM>2 or (NDIM==2 and soa_set):
          code('int ydim'+str(n)+'_'+name+' = args['+str(n)+'].dat->size[1];')
        if NDIM>3 or (NDIM==3 and soa_set):
          code('int zdim'+str(n)+'_'+name+' = args['+str(n)+'].dat->size[2];')

    code('')
    comm('set up initial pointers and exchange halos if necessary')
    for n in range (0, nargs):
      n_ad = get_ad_idx(n, arg_idxs_with_adjoints)
      if arg_typ[n] == 'ops_arg_dat': #TODO correct types const/not_const
        code('int base'+str(n)+' = args['+str(n)+'].dat->base_offset;')
        code('int base'+str(n)+'_t = base'+str(n)+'/sizeof('+typs[n]+');')
        code(typs[n]+' *__restrict__ '+clean_type(arg_list[n])+'_p = ('+typs[n]+' *)(args['+str(n)+'].data + base'+str(n)+');')
        if n_ad != -1:
          code(typs[n]+' *__restrict__ '+clean_type(arg_list_with_adjoints[n_ad])+'_p = ('+typs[n]+' *)args['+str(n)+'].derivative + base'+str(n)+'_t;')
      elif arg_typ[n] == 'ops_arg_gbl':
        if accs[n] == OPS_READ:
          code(typs[n]+' * __restrict__ '+clean_type(arg_list[n])+' = ('+typs[n]+' *)args['+str(n)+'].data;')
          if n_ad != -1:
            print('Warning adjoints for arguments defined with ops_arg_gbl not supported\n')
        else:
          code(typs[n]+' *__restrict__ p_a'+str(n)+' = ('+typs[n]+' *)((ops_reduction)args['+str(n)+'].data)->data;')
          if n_ad != -1:
            code(typs[n]+' *__restrict__ p_a'+str(n)+'_a1s = ('+typs[n]+' *)((ops_reduction)args['+str(n)+'].data)->derivative;')
      elif arg_typ[n] == 'ops_arg_scalar':
        if accs[n] == OPS_READ:
          code(typs[n]+' *__restrict__ '+clean_type(arg_list[n])+' = ('+typs[n]+' *)((ops_scalar)args['+str(n)+'].data)->data;')
          if n_ad != -1:
            code(typs[n]+' *__restrict__ p_a'+str(n)+'_a1s = ('+typs[n]+' *)args['+str(n)+'].derivative;')
        else:
          print('Warning ops_scalar with non read-only access not implemented\n')
      code('')
    code('')
    
    for n in range (0,nargs): #TODO gbl adjoints for reductions!
      if arg_typ[n] == 'ops_arg_gbl':
        if accs[n] != OPS_READ:
          n_ad = get_ad_idx(n, arg_idxs_with_adjoints)
          for d in range(0,int(dims[n])):
            code(typs[n]+' p_a'+str(n)+'_'+str(d)+' = p_a'+str(n)+'['+str(d)+'];')
            if n_ad >-1:
              code(typs[n]+' p_a'+str(n)+'_a1s_'+str(d)+' = p_a'+str(n)+'_a1s['+str(d)+'];')
      elif arg_typ[n] == 'ops_arg_scalar':
        n_ad = get_ad_idx(n, arg_idxs_with_adjoints)
        if n_ad >-1:
          for d in range(0,int(dims[n])):
            code(typs[n]+' p_a'+str(n)+'_a1s_'+str(d)+' = p_a'+str(n)+'_a1s['+str(d)+'];')
    if NDIM>2:
      FOR('n_z','start[2]','end[2]')
    if NDIM>1:
      FOR('n_y','start[1]','end[1]')

    if arg_idx != -1:
      if NDIM==1:
        code('int '+clean_type(arg_list[arg_idx])+'[] = {0};')
      elif NDIM==2:
        code('int '+clean_type(arg_list[arg_idx])+'[] = {0, arg_idx[1]+n_y};')
      elif NDIM==3:
        code('int '+clean_type(arg_list[arg_idx])+'[] = {0, arg_idx[1]+n_y, arg_idx[2]+n_z};')

    FOR('n_x','start[0]','end[0]')
    
    if arg_idx != -1:
      code(clean_type(arg_list[arg_idx])+'[0] = arg_idx[0]+n_x;')

    for n in range (0, nargs):
      if arg_typ[n] == 'ops_arg_dat':
        pre = ''
        n_ad = get_ad_idx(n, arg_idxs_with_adjoints) # ad
        if accs[n] == OPS_READ:
          pre = 'const '
        offset = ''
        dim = ''
        sizelist = ''
        extradim = 0
        if dims[n].isdigit() and int(dims[n])>1:
            dim = dims[n]+', '
            extradim = 1
        elif not dims[n].isdigit():
            dim = 'arg'+str(n)+'.dim, '
            extradim = 1
        if NDIM > 0:
          offset = offset + 'n_x*'+str(stride[NDIM*n])
        if NDIM > 1:
          offset = offset + ' + n_y * xdim'+str(n)+'_'+name+'*'+str(stride[NDIM*n+1])
        if NDIM > 2:
          offset = offset + ' + n_z * xdim'+str(n)+'_'+name+' * ydim'+str(n)+'_'+name+'*'+str(stride[NDIM*n+2])
        dimlabels = 'xyzuv'
        for i in range(1,NDIM+extradim):
          sizelist = sizelist + dimlabels[i-1]+'dim'+str(n)+'_'+name+', '

        if not dims[n].isdigit() or int(dims[n])>1:
          code('#ifdef OPS_SOA')
        code(pre + 'ACC<'+typs[n]+'> '+arg_list[n]+'('+dim+sizelist+arg_list[n]+'_p + '+offset+');')
        if n_ad != -1:
          argname = arg_list_with_adjoints[n_ad]+'_p'
          code('ACC<'+typs[n]+'> '+arg_list_with_adjoints[n_ad]+'('+dim+sizelist+argname+' + '+offset+');')

        if not dims[n].isdigit() or int(dims[n])>1:
          code('#else')
          code(pre + 'ACC<'+typs[n]+'> '+arg_list[n]+'('+dim+sizelist+arg_list[n]+'_p + '+dim[:-2]+'*('+offset+'));')
          if n_ad != -1:
            argname = arg_list_with_adjoints[n_ad]+'_p'
            code('ACC<'+typs[n]+'> '+arg_list_with_adjoints[n_ad]+'('+dim+sizelist+argname+' + '+dim[:-2]+'*('+offset+'));')
          code('#endif')
    for n in range (0,nargs):
      if arg_typ[n] == 'ops_arg_gbl':
        n_ad = get_ad_idx(n, arg_idxs_with_adjoints) # ad
        if n_ad > -1 and accs[n] != OPS_READ:
          code(typs[n]+' '+arg_list_with_adjoints[n_ad]+'['+str(dims[n])+'];')
          for d in range(0,int(dims[n])): # TODO values!!
            code(typs[n]+' p_a'+str(n)+'_'+str(d)+' = p_a'+str(n)+'['+str(d)+'];')
            line = arg_list_with_adjoints[n_ad] + '['+str(d)+'] = '
            if accs[n] == OPS_MIN or accs[n] == OPS_MAX or accs[n] == OPS_INC or accs[n] == OPS_WRITE:
              line += 'p_a'+str(n)+'_a1s_'+str(d)+';'
            code(line)
        if accs[n] == OPS_MIN:
          code(typs[n]+' '+arg_list[n]+'['+str(dims[n])+'];')
          for d in range(0,int(dims[n])):
            code(arg_list[n]+'['+str(d)+'] = p_a'+str(n)+'['+str(d)+'];') #need +INFINITY_ change to
        if accs[n] == OPS_MAX:
          code(typs[n]+' '+arg_list[n]+'['+str(dims[n])+'];')
          for d in range(0,int(dims[n])):
            code(arg_list[n]+'['+str(d)+'] = p_a'+str(n)+'['+str(d)+'];') #need -INFINITY_ change to
        if accs[n] == OPS_INC:
          code(typs[n]+' '+arg_list[n]+'['+str(dims[n])+'];')
          for d in range(0,int(dims[n])):
            code(arg_list[n]+'['+str(d)+'] = ZERO_'+typs[n]+';')
        if accs[n] == OPS_WRITE: #this may not be correct
          code(typs[n]+' '+arg_list[n]+'['+str(dims[n])+'];')
          for d in range(0,int(dims[n])):
            code(arg_list[n]+'['+str(d)+'] = ZERO_'+typs[n]+';')
      elif arg_typ[n] == 'ops_arg_scalar':
        n_ad = get_ad_idx(n, arg_idxs_with_adjoints) # ad
        if n_ad > -1:
          code(typs[n]+' '+arg_list_with_adjoints[n_ad]+'['+str(dims[n])+'];')
          for d in range(0,int(dims[n])):
            code( arg_list_with_adjoints[n_ad] + '['+str(d)+'] = ZERO_'+typs[n]+';')

              


    #insert user kernel
    code(kernel_text);

    for n in range (0,nargs):
      if arg_typ[n] == 'ops_arg_gbl':
        n_ad = get_ad_idx(n, arg_idxs_with_adjoints) # ad
        #  if n_ad > -1 and accs[n] != OPS_READ:
        #    for d in range(0,int(dims[n])): # values..
        #      line = 'p_a'+str(n)+'_a1s_' + str(d)
        #      if accs[n] == OPS_MIN or accs[n] == OPS_MAX:
        #        line += '=p_a'+str(n)+'_a1s['+str(d)+'];'
        #      elif accs[n] == OPS_INC or accs[n] == OPS_WRITE:
        #        line += '+=' +arg_list_with_adjoints[n_ad]+'['+str(d)+'];'
        #      code(line)
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
      elif arg_typ[n] == 'ops_arg_scalar':
        n_ad = get_ad_idx(n, arg_idxs_with_adjoints)
        if n_ad != -1:
          for d in range(0,int(dims[n])):
            code('p_a'+str(n)+'_a1s_'+str(d)+' +='+arg_list_with_adjoints[n_ad]+'['+str(d)+'];')
        

    ENDFOR() # x
    if NDIM>1:
      ENDFOR() # y
    if NDIM>2:
      ENDFOR() # z

    for n in range (0, nargs):
      if arg_typ[n] == 'ops_arg_gbl':
        n_ad = get_ad_idx(n, arg_idxs_with_adjoints) # ad
        if accs[n] != OPS_READ:
          for d in range(0,int(dims[n])):
            code('p_a'+str(n)+'['+str(d)+'] = p_a'+str(n)+'_'+str(d)+';')
            if n_ad > -1:
              code('p_a'+str(n)+'_a1s['+str(d)+'] = p_a'+str(n)+'_a1s_'+str(d)+';')
      elif arg_typ[n] == 'ops_arg_scalar':
        n_ad = get_ad_idx(n, arg_idxs_with_adjoints)
        if n_ad != -1:
          for d in range(0,int(dims[n])):
            code('p_a'+str(n)+'_a1s['+str(d)+'] = p_a'+str(n)+'_a1s_'+str(d)+';')

    config.depth = config.depth - 2
    code('}')
    code('')

    return True;

def ops_gen_mpi_adjoint_seq(master, date, consts, kernels, soa_set):

  global dims, stens
  global g_m, file_text, depth

  OPS_ID   = 1;  OPS_GBL   = 2;  OPS_MAP = 3; # unused

  OPS_READ = 1;  OPS_WRITE = 2;  OPS_RW  = 3;
  OPS_INC  = 4;  OPS_MAX   = 5;  OPS_MIN = 6;

  accsstring = ['OPS_READ','OPS_WRITE','OPS_RW','OPS_INC','OPS_MAX','OPS_MIN' ] # unused

  NDIM = 2 #the dimension of the application is hardcoded here .. we set it for each kernel

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

    reduction = 0
    for n in range (0, nargs):
      if arg_typ[n] == 'ops_arg_gbl' and accs[n] != OPS_READ:
        reduction = 1

    arg_idx = -1
    for n in range (0, nargs):
      if arg_typ[n] == 'ops_arg_idx':
        arg_idx = n

    config.file_text = ''
    config.depth = 0
    n_per_line = 4

    i = name.find('kernel') # unused
    name2 = name[0:i-1] # unused

##########################################################################
#  start with seq kernel function
##########################################################################
    gen_kernel(arg_typ, name, nargs, accs, typs, NDIM, stride, restrict, prolong, MULTI_GRID, reduction, arg_idx, gen_full_code, n_per_line, nk, soa_set, src_dir)
    adjoint_enabled = gen_adjoint_kernel(arg_typ, name, nargs, accs, typs, NDIM, stride, restrict, prolong, MULTI_GRID, reduction, arg_idx, n_per_line, nk, soa_set, src_dir)

    code('')
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
    code('desc->adjoint_function = ops_par_loop_'+name+'_adjoint_execute;')
    IF('block->instance->OPS_diags > 1')
    code('ops_timing_realloc(block->instance,'+str(nk)+',"'+name+'");')
    ENDIF()
    code('ops_add_to_dag(desc);')
    config.depth = 0
    code('}')


##########################################################################
#  output individual kernel file
##########################################################################
    if not os.path.exists('./MPI_adjoint_seq'):
      os.makedirs('./MPI_adjoint_seq')
    fid = open('./MPI_adjoint_seq/'+name+'_cpu_kernel.cpp','w')
    date = datetime.datetime.now() # unused
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
  if NDIM==2:
    code('#define OPS_2D')
  if NDIM==3:
    code('#define OPS_3D')
  if soa_set:
    code('#define OPS_SOA')
  code('#include "ops_lib_core.h"')
  code('#include "ops_algodiff.hpp"')
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

  fid = open('./MPI_adjoint_seq/'+master_basename[0]+'_cpu_kernels.cpp','w')
  fid.write('//\n// auto-generated by ops.py//\n\n')
  fid.write(config.file_text)
  fid.close()
