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

OPS_ID   = 1;  OPS_GBL   = 2;  OPS_MAP = 3;

OPS_READ = 1;  OPS_WRITE = 2;  OPS_RW  = 3;
OPS_INC  = 4;  OPS_MAX   = 5;  OPS_MIN = 6;


def clean_type(arg):
    for qual in ['__restrict__', 'RESTRICT', '__volatile__']:
        arg = arg.replace(qual, '')
    return arg

def parse_strides(stens, nargs, stride, NDIM):
    if NDIM == 2:
      for n in range (0, nargs):
        if str(stens[n]).find('STRID2D_X') > 0:
          stride[(NDIM+1)*n+1] = 0
        elif str(stens[n]).find('STRID2D_Y') > 0:
          stride[(NDIM+1)*n] = 0

    if NDIM == 3:
      for n in range (0, nargs):
        if str(stens[n]).find('STRID3D_XY') > 0:
          stride[(NDIM+1)*n+2] = 0
        elif str(stens[n]).find('STRID3D_YZ') > 0:
          stride[(NDIM+1)*n] = 0
        elif str(stens[n]).find('STRID3D_XZ') > 0:
          stride[(NDIM+1)*n+1] = 0
        elif str(stens[n]).find('STRID3D_X') > 0:
          stride[(NDIM+1)*n+1] = 0
          stride[(NDIM+1)*n+2] = 0
        elif str(stens[n]).find('STRID3D_Y') > 0:
          stride[(NDIM+1)*n] = 0
          stride[(NDIM+1)*n+2] = 0
        elif str(stens[n]).find('STRID3D_Z') > 0:
          stride[(NDIM+1)*n] = 0
          stride[(NDIM+1)*n+1] = 0
    return stride



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
    code('#ifdef OPS_BATCHED')
    code('const int batch_size = block->count;')
    code('#endif')
    code('#else')
    #code('void ops_par_loop_'+name+'_execute(ops_kernel_descriptor *desc) {')
    code('void ops_par_loop_'+name+'_execute(const char *name, ops_block block, int blockidx_start, int blockidx_end, int dim, int *range, int nargs, ops_arg* args) {')
    config.depth = 2
    #code('char const *name = "'+name+'";')
    #code('ops_block block = desc->block;')
    #code('int dim = desc->dim;')
    #code('int *range = desc->range;')
    code('#ifdef OPS_BATCHED')
    code('const int batch_size = OPS_BATCH_SIZE;')
    code('#endif')

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

def generate_sizes_bounds(nargs, arg_typ, NDIM, generate_dims=1):
    global g_m, file_text, depth
    code('')
    if generate_dims:
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

def generate_pointers(nargs, arg_typ, accs, typs, arg_list, restrict, prolong, dims, decl_read_1d=1, device = ''):
    comm('set up initial pointers')
    for n in range (0, nargs):
      if arg_typ[n] == 'ops_arg_dat':
          code(typs[n]+' * __restrict__ '+clean_type(arg_list[n])+'_p = ('+typs[n]+' *)(args['+str(n)+'].data'+device+' + args['+str(n)+'].dat->base_offset + blockidx_start * args['+str(n)+'].dat->batch_offset);')
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
          if decl_read_1d or (dims[n].isdigit() and int(dims[n])==1):
            code(typs[n]+' * __restrict__ '+clean_type(arg_list[n])+' = ('+typs[n]+' *)args['+str(n)+'].data;')
          else:
            code(typs[n]+' * __restrict__ p_a'+str(n)+' = ('+typs[n]+' *)args['+str(n)+'].data;')
        else:
          code('#ifdef OPS_MPI')
          code(typs[n]+' * __restrict__ p_a'+str(n)+' = ('+typs[n]+' *)(((ops_reduction)args['+str(n)+'].data)->data + ((ops_reduction)args['+str(n)+'].data)->size * block->index + ((ops_reduction)args['+str(n)+'].data)->size * blockidx_start);')
          code('#else //OPS_MPI')
          code(typs[n]+' * __restrict__ p_a'+str(n)+' = ('+typs[n]+' *)(((ops_reduction)args['+str(n)+'].data)->data + ((ops_reduction)args['+str(n)+'].data)->size * blockidx_start);')
          code('#endif //OPS_MPI')
        code('')
      code('')
    code('')


def generate_exchanges(nargs, nk, gen_full_code, HD):
    code('#ifndef OPS_LAZY')
    comm('Halo Exchanges')
    code('ops_H_D_exchanges_'+HD+'(args, '+str(nargs)+');')
    code('ops_halo_exchanges(args,'+str(nargs)+',range);')
    code('ops_H_D_exchanges_'+HD+'(args, '+str(nargs)+');')
    code('#endif')
    code('')
    if gen_full_code==1:
      IF('OPS_instance::getOPSInstance()->OPS_diags > 1')
      code('ops_timers_core(&__c1,&__t1);')
      code('OPS_instance::getOPSInstance()->OPS_kernels['+str(nk)+'].mpi_time += __t1-__t2;')
      ENDIF()
      code('')

def get_user_function(name, arg_typ, src_dir):
    found = 0
    for files in glob.glob( os.path.join(src_dir,"*.h") ):
      f = open( files, 'r' )
      for line in f:
        if name in line:
          file_name = f.name
          found = 1;
          break
      if found == 1:
        break;

    if found == 0:
      print "COUND NOT FIND KERNEL", name

    fid = open(file_name, 'r')
    text = fid.read()

    fid.close()
    text = comment_remover(text)
    text = remove_trailing_w_space(text)

    p = re.compile('void\\s+\\b'+name+'\\b')

    i = p.search(text).start()

    if(i < 0):
      print "\n********"
      print "Error: cannot locate user kernel function: "+name+" - Aborting code generation"
      exit(2)

    i2 = text[i:].find(name)
    i2 = i+i2
    j = text[i:].find('{')
    k = para_parse(text, i+j, '{', '}')
    full_text = util.convert_ACC(text[i:k+1], arg_typ)
    kernel_text = text[i+j+1:k]
    kernel_text = convert_ACC_body(kernel_text)
    arg_list = parse_signature(text[i2+len(name):i+j]) 

    return [kernel_text, arg_list, full_text]

def generate_arg_idx(arg_idx, arg_list, NDIM):
      if NDIM==1:
        code('#if defined(OPS_BATCHED) && OPS_BATCHED==0')
        code('int '+clean_type(arg_list[arg_idx])+'[] = {arg_idx[0]+n_1, blockidx_start + n_0};')
        code('#else')
        code('int '+clean_type(arg_list[arg_idx])+'[] = {arg_idx[0]+n_0, blockidx_start + n_1};')
        code('#endif')
      elif NDIM==2:
        code('#if defined(OPS_BATCHED) && OPS_BATCHED==0')
        code('int '+clean_type(arg_list[arg_idx])+'[] = {arg_idx[0]+n_1, arg_idx[1]+n_2, blockidx_start + n_0};')
        code('#elif OPS_BATCHED==1')
        code('int '+clean_type(arg_list[arg_idx])+'[] = {arg_idx[0]+n_0, arg_idx[1]+n_2, blockidx_start + n_1};')
        code('#else')
        code('int '+clean_type(arg_list[arg_idx])+'[] = {arg_idx[0]+n_0, arg_idx[1]+n_1, blockidx_start + n_2};')
        code('#endif')
      elif NDIM==3:
        code('#if defined(OPS_BATCHED) && OPS_BATCHED==0')
        code('int '+clean_type(arg_list[arg_idx])+'[] = {arg_idx[0]+n_1, arg_idx[1]+n_2, arg_idx[2]+n_3, blockidx_start + n_0};')
        code('#elif OPS_BATCHED==1')
        code('int '+clean_type(arg_list[arg_idx])+'[] = {arg_idx[0]+n_0, arg_idx[1]+n_2, arg_idx[2]+n_3, blockidx_start + n_1};')
        code('#elif OPS_BATCHED==2')
        code('int '+clean_type(arg_list[arg_idx])+'[] = {arg_idx[0]+n_0, arg_idx[1]+n_1, arg_idx[2]+n_3, blockidx_start + n_2};')
        code('#else')
        code('int '+clean_type(arg_list[arg_idx])+'[] = {arg_idx[0]+n_0, arg_idx[1]+n_1, arg_idx[2]+n_2, blockidx_start + n_3};')
        code('#endif')


def generate_accessors(nargs, arg_typ, dims, NDIM, stride, typs, accs, arg_list, restrict, prolong, dimstr):
    for n in range (0, nargs):
      if arg_typ[n] == 'ops_arg_dat':
        if restrict[n] == 1:
          n_0 = 'n_0*stride_'+str(n)+'0'
          n_1 = 'n_1*stride_'+str(n)+'1'
          n_2 = 'n_2*stride_'+str(n)+'2'
        elif prolong[n] == 1:
          n_0 = '(n_0+global_idx0%stride_'+str(n)+'0)/stride_'+str(n)+'0'
          n_1 = '(n_1+global_idx1%stride_'+str(n)+'1)/stride_'+str(n)+'1'
          n_2 = '(n_2+global_idx2%stride_'+str(n)+'2)/stride_'+str(n)+'2'
        else:
          n_0 = 'n_0'
          n_1 = 'n_1'
          n_2 = 'n_2'
          n_3 = 'n_3'
        pre = ''
        if accs[n] == OPS_READ:
          pre = 'const '
        offset = ''
        dim = ''
        sizelist = ''
        if dims[n].isdigit() and int(dims[n])>1:
            dim = dims[n]+', '
        elif not dims[n].isdigit():
            dim = 'arg'+str(n)+'.dim, '
        if NDIM >= 0:
          offset = offset + n_0 +str(stride[(NDIM+1)*n])
        if NDIM >= 1:
          offset = offset + ' + '+n_1+' * '+dimstr(n,0)+str(stride[(NDIM+1)*n+1])
        if NDIM >= 2:
          offset = offset + ' + '+n_2+' * '+dimstr(n,0)+' * '+dimstr(n,1)+str(stride[(NDIM+1)*n+2])
        if NDIM >= 3:
          offset = offset + ' + '+n_3+' * '+dimstr(n,0)+' * '+dimstr(n,1)+' * '+dimstr(n,2)+str(stride[(NDIM+1)*n+3])
        dimlabels = 'xyzuv'
        for i in range(0,NDIM):
          sizelist = sizelist + dimstr(n,i)+', '

        if not dims[n].isdigit() or int(dims[n])>1:
          code('#ifdef OPS_SOA')
        code(pre + 'ACC<'+typs[n]+'> '+clean_type(arg_list[n])+'('+dim+sizelist+arg_list[n]+'_p + '+offset+');')
        if not dims[n].isdigit() or int(dims[n])>1:
          code('#else')
          code(pre + 'ACC<'+typs[n]+'> '+clean_type(arg_list[n])+'('+dim+sizelist+arg_list[n]+'_p + '+dim[:-2]+'*('+offset+'));')
          code('#endif')

def generate_gbl_locals(nargs, arg_typ, accs, dims, typs, arg_list):
    for n in range (0,nargs):
      if arg_typ[n] == 'ops_arg_gbl':
        if accs[n] == OPS_MIN:
          code(typs[n]+' '+arg_list[n]+'['+str(dims[n])+'];')
          for d in range(0,int(dims[n])):
            code(arg_list[n]+'['+str(d)+'] = INFINITY_'+typs[n]+';')
        if accs[n] == OPS_MAX:
          code(typs[n]+' '+arg_list[n]+'['+str(dims[n])+'];')
          for d in range(0,int(dims[n])):
            code(arg_list[n]+'['+str(d)+'] = -INFINITY_'+typs[n]+';') 
        if accs[n] == OPS_INC:
          code(typs[n]+' '+arg_list[n]+'['+str(dims[n])+'];')
          for d in range(0,int(dims[n])):
            code(arg_list[n]+'['+str(d)+'] = ZERO_'+typs[n]+';')
        if accs[n] == OPS_WRITE: #this may not be correct
          code(typs[n]+' '+arg_list[n]+'['+str(dims[n])+'];')
          for d in range(0,int(dims[n])):
            code(arg_list[n]+'['+str(d)+'] = ZERO_'+typs[n]+';')

def generate_tail(nk, nargs, arg_typ, accs, gen_full_code, HD):

    if gen_full_code==1:
      IF('OPS_instance::getOPSInstance()->OPS_diags > 1')
      code('ops_timers_core(&__c2,&__t2);')
      code('OPS_instance::getOPSInstance()->OPS_kernels['+str(nk)+'].time += __t2-__t1;')
      ENDIF()

    code('#ifndef OPS_LAZY')
    code('ops_set_dirtybit_'+HD+'(args, '+str(nargs)+');')
    for n in range (0, nargs):
      if arg_typ[n] == 'ops_arg_dat' and (accs[n] == OPS_WRITE or accs[n] == OPS_RW or accs[n] == OPS_INC):
        code('ops_set_halo_dirtybit3(&args['+str(n)+'],range);')
    code('#endif')

    if gen_full_code==1:
      code('')
      IF('OPS_instance::getOPSInstance()->OPS_diags > 1')
      comm('Update kernel record')
      code('ops_timers_core(&__c1,&__t1);')
      code('OPS_instance::getOPSInstance()->OPS_kernels['+str(nk)+'].mpi_time += __t1-__t2;')
      for n in range (0, nargs):
        if arg_typ[n] == 'ops_arg_dat':
          code('OPS_instance::getOPSInstance()->OPS_kernels['+str(nk)+'].transfer += ops_compute_transfer(dim, start, end, &arg'+str(n)+');')
      ENDIF()
    config.depth = config.depth - 2
    code('}')
    code('')
