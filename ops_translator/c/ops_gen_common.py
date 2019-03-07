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

para_parse = util.para_parse
comment_remover = util.comment_remover
remove_trailing_w_space = util.remove_trailing_w_space
parse_signature = util.parse_signature
check_accs = util.check_accs
mult = util.mult
convert_ACC_body = util.convert_ACC_body

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

OPS_ID   = 1;  OPS_GBL   = 2;  OPS_MAP = 3;

OPS_READ = 1;  OPS_WRITE = 2;  OPS_RW  = 3;
OPS_INC  = 4;  OPS_MAX   = 5;  OPS_MIN = 6;


def clean_type(arg):
    for qual in ['__restrict__', 'RESTRICT', '__volatile__']:
        arg = arg.replace(qual, '')
    return arg

def generate_header(nk, name, nargs, arg_typ, accs, arg_idx, NDIM, MULTI_GRID, gen_full_code):
    global g_m, file_text, depth
    n_per_line = 4

    comm('')
    comm(' host stub function')
    code('#ifndef OPS_LAZY')
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
    code('const int blockidx_start = 0; const int blockidx_end = block->count;')
    code('const int batch_size = block->count;')
    code('#else')
    #code('void ops_par_loop_'+name+'_execute(ops_kernel_descriptor *desc) {')
    code('void ops_par_loop_'+name+'_execute(const char *name, ops_block block, int blockidx_start, int blockidx_end, int dim, int *range, int nargs, ops_arg* args) {')
    config.depth = 2
    #code('char const *name = "'+name+'";')
    #code('ops_block block = desc->block;')
    #code('int dim = desc->dim;')
    #code('int *range = desc->range;')
    code('const int batch_size = OPS_BATCH_SIZE;')

    for n in range (0, nargs):
      code('ops_arg arg'+str(n)+' = args['+str(n)+'];')

    code('#endif')

    code('');
    comm('Timing')
    code('double __t1,__t2,__c1,__c2;')
    code('');

    code('#ifndef OPS_LAZY')
    text ='ops_arg args['+str(nargs)+'] = {'
    for n in range (0, nargs):
      text = text +' arg'+str(n)
      if nargs <> 1 and n != nargs-1:
        text = text +','
      else:
        text = text +'};\n\n'
      if n%n_per_line == 5 and n <> nargs-1:
        text = text +'\n                    '
    code(text);
    code('#endif')
    code('')
    code('#if defined(CHECKPOINTING) && !defined(OPS_LAZY)')
    code('if (!ops_checkpointing_before(args,'+str(nargs)+',range,'+str(nk)+')) return;')
    code('#endif')
    code('')

    if gen_full_code:
      IF('OPS_instance::getOPSInstance()->OPS_diags > 1')
      code('ops_timing_realloc('+str(nk)+',"'+name+'");')
      code('OPS_instance::getOPSInstance()->OPS_kernels['+str(nk)+'].count++;')
      code('ops_timers_core(&__c2,&__t2);')
      ENDIF()
      code('')

    code('#ifdef OPS_DEBUG')
    code('ops_register_args(args, "'+name+'");')
    code('#endif')
    code('')

    code('')

    comm('compute locally allocated range for the sub-block')
    code('int start['+str(NDIM)+'];')
    code('int end['+str(NDIM)+'];')
    if not (arg_idx<>-1 or MULTI_GRID):
      code('#ifdef OPS_MPI')
    code('int arg_idx['+str(NDIM)+'];')
    if not (arg_idx<>-1 or MULTI_GRID):
      code('#endif')

    code('#if defined(OPS_LAZY) || !defined(OPS_MPI)')
    FOR('n','0',str(NDIM))
    code('start[n] = range[2*n];end[n] = range[2*n+1];')
    ENDFOR()
    code('#else')
    code('if (compute_ranges(args, '+str(nargs)+',block, range, start, end, arg_idx) < 0) return;')
    code('#endif')

    code('')

    if arg_idx<>-1 or MULTI_GRID:
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
      code('#endif //OPS_MPI')

def generate_sizes_bounds(nargs, arg_typ, NDIM):
    global g_m, file_text, depth
    code('')
    comm("initialize variable with the dimension of dats")
    for n in range (0, nargs):
      if arg_typ[n] == 'ops_arg_dat':
        if NDIM>0:
          code('const int xdim'+str(n)+' = args['+str(n)+'].dat->size[0];')#*args['+str(n)+'].dat->dim;')
        if NDIM>1:
          code('const int ydim'+str(n)+' = args['+str(n)+'].dat->size[1];')
        if NDIM>2:
          code('const int zdim'+str(n)+' = args['+str(n)+'].dat->size[2];')

    code('#ifdef OPS_BATCHED')
    for d in range(0,NDIM+1):
      code('const int bounds_'+str(d)+'_l = OPS_BATCHED == '+str(d)+' ? 0 : start[(OPS_BATCHED>'+str(d)+')+'+str(d-1)+'];')
      code('const int bounds_'+str(d)+'_u = OPS_BATCHED == '+str(d)+' ? MIN(batch_size,block->count-blockidx_start) : end[(OPS_BATCHED>'+str(d)+')+'+str(d-1)+'];')
    code('#else')
    for d in range(0,NDIM):
      code('const int bounds_'+str(d)+'_l = start['+str(d)+'];')
      code('const int bounds_'+str(d)+'_u = end['+str(d)+'];')
    code('const int bounds_'+str(NDIM)+'_l = 0;')
    code('const int bounds_'+str(NDIM)+'_u = blockidx_end-blockidx_start;')
    code('#endif')

def generate_strides(nargs, stens, stride, NDIM):
    global g_m, file_text, depth
    for n in range (0, nargs):
      if str(stens[n]).find('STRID') > 0:
        code('#if defined(OPS_BATCHED) && OPS_BATCHED==0')
        code('const int stride'+str(n)+'_0 = 1;')
        code('const int stride'+str(n)+'_1 = '+str(stride[(NDIM+1)*n+0])+';')
        if NDIM>1:
          code('const int stride'+str(n)+'_2 = '+str(stride[(NDIM+1)*n+1])+';')
        if NDIM>2:
          code('const int stride'+str(n)+'_3 = '+str(stride[(NDIM+1)*n+2])+';')
        code('#elif OPS_BATCHED==1')
        code('const int stride'+str(n)+'_0 = '+str(stride[(NDIM+1)*n+0])+';')
        code('const int stride'+str(n)+'_1 = 1;')
        if NDIM>1:
          code('const int stride'+str(n)+'_2 = '+str(stride[(NDIM+1)*n+1])+';')
        if NDIM>2:
          code('const int stride'+str(n)+'_3 = '+str(stride[(NDIM+1)*n+2])+';')
        if NDIM>1:
          code('#elif OPS_BATCHED==2')
          code('const int stride'+str(n)+'_0 = '+str(stride[(NDIM+1)*n+0])+';')
          code('const int stride'+str(n)+'_1 = '+str(stride[(NDIM+1)*n+1])+';')
          code('const int stride'+str(n)+'_2 = 1;')
          if NDIM>2:
            code('const int stride'+str(n)+'_3 = '+str(stride[(NDIM+1)*n+2])+';')
        if NDIM>2:
          code('#elif OPS_BATCHED==3')
          code('const int stride'+str(n)+'_0 = '+str(stride[(NDIM+1)*n+0])+';')
          code('const int stride'+str(n)+'_1 = '+str(stride[(NDIM+1)*n+1])+';')
          code('const int stride'+str(n)+'_2 = '+str(stride[(NDIM+1)*n+2])+';')
          code('const int stride'+str(n)+'_3 = 1;')
        code('#else')
        code('const int stride'+str(n)+'_0 = '+str(stride[(NDIM+1)*n+0])+';')
        if NDIM>1:
          code('const int stride'+str(n)+'_1 = '+str(stride[(NDIM+1)*n+1])+';')
        if NDIM>2:
          code('const int stride'+str(n)+'_2 = '+str(stride[(NDIM+1)*n+2])+';')
        code('const int stride'+str(n)+'_'+str(NDIM)+' = 1;')

        code('#endif')
        for d in range(0,NDIM+1):
          stride[(NDIM+1)*n+d] = '*stride'+str(n)+'_'+str(d)
      else:
        for d in range(0,NDIM+1):
          stride[(NDIM+1)*n+d] = ''
    return stride

def generate_pointers(nargs, arg_typ, accs, typs, arg_list, restrict, prolong):
    comm('set up initial pointers')
    for n in range (0, nargs):
      if arg_typ[n] == 'ops_arg_dat':
          code(typs[n]+' * __restrict__ '+clean_type(arg_list[n])+'_p = ('+typs[n]+' *)(args['+str(n)+'].data + args['+str(n)+'].dat->base_offset + blockidx_start * args['+str(n)+'].dat->batch_offset);')
          if restrict[n] == 1 or prolong[n] == 1:
            code('#ifdef OPS_MPI')
            code('sub_dat_list sd'+str(n)+' = OPS_sub_dat_list[args['+str(n)+'].dat->index];')
          if restrict[n] == 1:
            code(clean_type(arg_list[n])+' += arg_idx[0]*args['+str(n)+'].stencil->mgrid_stride[0] - sd'+str(n)+'->decomp_disp[0] + args['+str(n)+'].dat->d_m[0];')
            if NDIM>1:
              code(clean_type(arg_list[n])+' += (arg_idx[1]*args['+str(n)+'].stencil->mgrid_stride[1] - sd'+str(n)+'->decomp_disp[1] + args['+str(n)+'].dat->d_m[1])*xdim'+str(n)+';')
            if NDIM>2:
              code(clean_type(arg_list[n])+' += (arg_idx[2]*args['+str(n)+'].stencil->mgrid_stride[2] - sd'+str(n)+'->decomp_disp[2] + args['+str(n)+'].dat->d_m[2])*xdim'+str(n)+' * ydim'+str(n)+';')
          if prolong[n] == 1:
            code(clean_type(arg_list[n])+' += arg_idx[0]/args['+str(n)+'].stencil->mgrid_stride[0] - sd'+str(n)+'->decomp_disp[0] + args['+str(n)+'].dat->d_m[0];')
            if NDIM>1:
              code(clean_type(arg_list[n])+' += (arg_idx[1]/args['+str(n)+'].stencil->mgrid_stride[1] - sd'+str(n)+'->decomp_disp[1] + args['+str(n)+'].dat->d_m[1])*xdim'+str(n)+';')
            if NDIM>2:
              code(clean_type(arg_list[n])+' += (arg_idx[2]/args['+str(n)+'].stencil->mgrid_stride[2] - sd'+str(n)+'->decomp_disp[2] + args['+str(n)+'].dat->d_m[2])*xdim'+str(n)+' * ydim'+str(n)+';')

          if restrict[n] == 1 or prolong[n] == 1:
            code('#endif')
      elif arg_typ[n] == 'ops_arg_gbl':
        if accs[n] == OPS_READ:
          code(typs[n]+' * __restrict__ '+clean_type(arg_list[n])+' = ('+typs[n]+' *)args['+str(n)+'].data;')
        else:
          code('#ifdef OPS_MPI')
          code(typs[n]+' * __restrict__ p_a'+str(n)+' = ('+typs[n]+' *)(((ops_reduction)args['+str(n)+'].data)->data + ((ops_reduction)args['+str(n)+'].data)->size * block->index + ((ops_reduction)args['+str(n)+'].data)->size * blockidx_start);')
          code('#else //OPS_MPI')
          code(typs[n]+' * __restrict__ p_a'+str(n)+' = ('+typs[n]+' *)(((ops_reduction)args['+str(n)+'].data)->data + ((ops_reduction)args['+str(n)+'].data)->size * blockidx_start);')
          code('#endif //OPS_MPI')
        code('')
      code('')
    code('')
