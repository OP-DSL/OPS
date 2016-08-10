
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

"""
OPS MPI_seq code generator

This routine is called by ops.py which parses the input files

It produces a file xxx_seq_kernel.cpp for each kernel,
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

def ops_gen_mpi_lazy(master, date, consts, kernels):

  global dims, stens
  global g_m, file_text, depth

  OPS_ID   = 1;  OPS_GBL   = 2;  OPS_MAP = 3;

  OPS_READ = 1;  OPS_WRITE = 2;  OPS_RW  = 3;
  OPS_INC  = 4;  OPS_MAX   = 5;  OPS_MIN = 6;

  accsstring = ['OPS_READ','OPS_WRITE','OPS_RW','OPS_INC','OPS_MAX','OPS_MIN' ]

  NDIM = 2 #the dimension of the application is hardcoded here .. need to get this dynamically

  gen_full_code = 1;

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
    stride = [1] * nargs * NDIM

    if NDIM == 2:
      for n in range (0, nargs):
        if str(stens[n]).find('STRID2D_X') > 0:
          stride[NDIM*n+1] = 0
        elif str(stens[n]).find('STRID2D_Y') > 0:
          stride[NDIM*n] = 0

    if NDIM == 3:
      for n in range (0, nargs):
        if str(stens[n]).find('STRID3D_X') > 0:
          stride[NDIM*n+1] = 0
          stride[NDIM*n+2] = 0
        elif str(stens[n]).find('STRID3D_Y') > 0:
          stride[NDIM*n] = 0
          stride[NDIM*n+2] = 0
        elif str(stens[n]).find('STRID3D_Z') > 0:
          stride[NDIM*n] = 0
          stride[NDIM*n+1] = 0


    reduction = 0
    for n in range (0, nargs):
      if arg_typ[n] == 'ops_arg_gbl' and accs[n] <> OPS_READ:
        reduction = 1

    arg_idx = 0
    for n in range (0, nargs):
      if arg_typ[n] == 'ops_arg_idx':
        arg_idx = n

    config.file_text = ''
    config.depth = 0
    n_per_line = 4

    i = name.find('kernel')
    name2 = name[0:i-1]

##########################################################################
#  generate MACROS
##########################################################################
    for n in range (0, nargs):
      if arg_typ[n] == 'ops_arg_dat':
        if int(dims[n]) == 1:
          if NDIM==1:
            code('#define OPS_ACC'+str(n)+'(x) (n_x*'+str(stride[NDIM*n])+'+x)')
          if NDIM==2:
            code('#define OPS_ACC'+str(n)+'(x,y) (n_x*'+str(stride[NDIM*n])+'+n_y*xdim'+str(n)+'_'+name+'*'+str(stride[NDIM*n+1])+'+x+xdim'+str(n)+'_'+name+'*(y))')
          if NDIM==3:
            code('#define OPS_ACC'+str(n)+'(x,y,z) (n_x*'+str(stride[NDIM*n])+'+n_y*xdim'+str(n)+'_'+name+'*'+str(stride[NDIM*n+1])+'+n_z*xdim'+str(n)+'_'+name+'*ydim'+str(n)+'_'+name+'*'+str(stride[NDIM*n+2])+'+x+xdim'+str(n)+'_'+name+'*(y)+xdim'+str(n)+'_'+name+'*ydim'+str(n)+'_'+name+'*(z))')
    code('')

    for n in range (0, nargs):
      if arg_typ[n] == 'ops_arg_dat':
        if int(dims[n]) > 1:
          if NDIM==1:
            code('#define OPS_ACC_MD'+str(n)+'(d,x) (n_x*'+str(stride[NDIM*n])+'*'+str(dims[n])+' +(x)*'+str(dims[n])+'+(d))')
          if NDIM==2:
            code('#define OPS_ACC_MD'+str(n)+'(d,x,y) (n_x*'+str(stride[NDIM*n])+'*'+str(dims[n])+' + n_y*xdim'+str(n)+'_'+name+'*'+str(stride[NDIM*n+1])+'*'+str(dims[n])+' + (x)*'+str(dims[n])+'+(d)+(xdim'+str(n)+'_'+name+'*(y)*'+str(dims[n])+'))')
          if NDIM==3:
            code('#define OPS_ACC_MD'+str(n)+'(d,x,y,z) (n_x*'+str(stride[NDIM*n])+'*'+str(dims[n])+' + n_y*xdim'+str(n)+'_'+name+'*'+str(stride[NDIM*n+1])+'*'+str(dims[n])+' + n_z*xdim'+str(n)+'_'+name+'*ydim'+str(n)+'_'+name+'*'+str(stride[NDIM*n+2])+'*'+str(dims[n])+' + (x)*'+str(dims[n])+'+(d)+(xdim'+str(n)+'_'+name+'*(y)*'+str(dims[n])+')+(xdim'+str(n)+'_'+name+'*ydim'+str(n)+'_'+name+'*(z)*'+str(dims[n])+'))')


##########################################################################
#  start with seq kernel function
##########################################################################

    code('')
    comm('user function')

    found = 0
    for files in glob.glob( "*.h" ):
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
    kernel_text = text[i+j+1:k]
    m = text.find(name)
    arg_list = parse_signature(text[i2+len(name):i+j])

    check_accs(name, arg_list, arg_typ, text[i+j:k])

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

    code('');
    comm('Timing')
    code('double t1,t2,c1,c2;')
    code('');

    #code('ops_printf("In loop \%s\\n","'+name+'");')

    
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
    code('')
    code('#ifdef CHECKPOINTING')
    code('if (!ops_checkpointing_before(args,'+str(nargs)+',range,'+str(nk)+')) return;')
    code('#endif')
    code('')

    if gen_full_code:
      IF('OPS_diags > 1')
      code('ops_timing_realloc('+str(nk)+',"'+name+'");')
      code('OPS_kernels['+str(nk)+'].count++;')
      code('ops_timers_core(&c2,&t2);')
      ENDIF()
      code('')


    comm('compute locally allocated range for the sub-block')
    code('int start['+str(NDIM)+'];')
    code('int end['+str(NDIM)+'];')
    code('')
    FOR('n','0',str(NDIM))
    code('start[n] = range[2*n];end[n] = range[2*n+1];')
    ENDFOR()
    code('')

    code('#ifdef OPS_DEBUG')
    code('ops_register_args(args, "'+name+'");')
    code('#endif')
    code('')

    code('')
    if arg_idx:
      code('int arg_idx['+str(NDIM)+'];')
      code('#ifdef OPS_MPI')
      code('sub_block_list sb = OPS_sub_block_list[block->index];')
      for n in range (0,NDIM):
        code('arg_idx['+str(n)+'] = sb->decomp_disp['+str(n)+'];')
      code('#else //OPS_MPI')
      for n in range (0,NDIM):
        code('arg_idx['+str(n)+'] = 0;')
      code('#endif //OPS_MPI')

    code('')
    comm('set up initial pointers and exchange halos if necessary')
    for n in range (0, nargs):
      pre = ''
      if accs[n] == OPS_READ:
        pre = 'const '
      if arg_typ[n] == 'ops_arg_dat':
        code('int base'+str(n)+' = args['+str(n)+'].dat->base_offset;')
        code(pre + typs[n]+' * __restrict__ '+arg_list[n]+' = ('+typs[n]+' *)(args['+str(n)+'].data + base'+str(n)+');')

      elif arg_typ[n] == 'ops_arg_gbl':
        if accs[n] == OPS_READ:
          code(pre + typs[n]+' * __restrict__ '+arg_list[n]+' = ('+typs[n]+' *)args['+str(n)+'].data;')
        else:
          code('#ifdef OPS_MPI')
          code(typs[n]+' * __restrict__ p_a'+str(n)+' = ('+typs[n]+' *)(((ops_reduction)args['+str(n)+'].data)->data + ((ops_reduction)args['+str(n)+'].data)->size * block->index);')
          code('#else //OPS_MPI')
          code(typs[n]+' * __restrict__ p_a'+str(n)+' = ('+typs[n]+' *)((ops_reduction)args['+str(n)+'].data)->data;')
          code('#endif //OPS_MPI')
        code('')
      code('')
    code('')


    comm("initialize global variable with the dimension of dats")
    for n in range (0, nargs):
      if arg_typ[n] == 'ops_arg_dat':
        if NDIM > 1:
          code('int xdim'+str(n)+'_'+name+' = args['+str(n)+'].dat->size[0];')#*args['+str(n)+'].dat->dim;')
        if NDIM==3:
          code('int ydim'+str(n)+'_'+name+' = args['+str(n)+'].dat->size[1];')
    code('')

    if gen_full_code==1:
      comm('Halo Exchanges')
      code('ops_H_D_exchanges_host(args, '+str(nargs)+');')
      code('ops_halo_exchanges(args,'+str(nargs)+',range);')
      code('ops_H_D_exchanges_host(args, '+str(nargs)+');')
      code('')
      IF('OPS_diags > 1')
      code('ops_timers_core(&c1,&t1);')
      code('OPS_kernels['+str(nk)+'].mpi_time += t1-t2;')
      ENDIF()
      code('')

    for n in range (0,nargs):
      if arg_typ[n] == 'ops_arg_gbl':
        if accs[n] <> OPS_READ:
          for d in range(0,int(dims[n])):
            code(typs[n]+' p_a'+str(n)+'_'+str(d)+' = p_a'+str(n)+'['+str(d)+'];')

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
    code('#pragma omp parallel for'+line2)
    if NDIM>2:
      FOR('n_z','start[2]','end[2]')
    if NDIM>1:
      FOR('n_y','start[1]','end[1]')
    code('#pragma omp simd'+line)
    FOR('n_x','start[0]','end[0]')
    if arg_idx:
      if NDIM==1:
        code('int '+arg_list[arg_idx]+'[] = {arg_idx[0]+n_x};')
      elif NDIM==2:
        code('int '+arg_list[arg_idx]+'[] = {arg_idx[0]+n_x, arg_idx[1]+n_y};')
      elif NDIM==3:
        code('int '+arg_list[arg_idx]+'[] = {arg_idx[0]+n_x, arg_idx[1]+n_y, arg_idx[2]+n_z};')

    for n in range (0,nargs):
      if arg_typ[n] == 'ops_arg_gbl':
        if accs[n] == OPS_MIN and int(dims[n])>1:
          code(typs[n]+' '+arg_list[n]+'['+str(dims[n])+'];')
          for d in range(0,int(dims[n])):
            code(arg_list[n]+'['+str(d)+'] = p_a'+str(n)+'['+str(d)+'];') #need +INFINITY_ change to
        if accs[n] == OPS_MAX and int(dims[n])>1:
          code(typs[n]+' '+arg_list[n]+'['+str(dims[n])+'];')
          for d in range(0,int(dims[n])):
            code(arg_list[n]+'['+str(d)+'] = p_a'+str(n)+'['+str(d)+'];') #need -INFINITY_ change to
        if accs[n] == OPS_INC and int(dims[n])>1:
          code(typs[n]+' '+arg_list[n]+'['+str(dims[n])+'];')
          for d in range(0,int(dims[n])):
            code(arg_list[n]+'['+str(d)+'] = ZERO_'+typs[n]+';')
        if accs[n] == OPS_WRITE and int(dims[n])>1: #this may not be correct
          code(typs[n]+' '+arg_list[n]+'['+str(dims[n])+'];')
          for d in range(0,int(dims[n])):
            code(arg_list[n]+'['+str(d)+'] = ZERO_'+typs[n]+';')
        if accs[n] <> OPS_READ and int(dims[n])==1:
          code(typs[n]+' *'+arg_list[n]+' = &p_a'+str(n)+'_0;')

    #insert user kernel
    code(kernel_text);

    for n in range (0,nargs):
      if arg_typ[n] == 'ops_arg_gbl':
        if accs[n] == OPS_MIN and int(dims[n])>1:
          for d in range(0,int(dims[n])):
            code('p_a'+str(n)+'_'+str(d)+' = MIN(p_a'+str(n)+'_'+str(d)+','+arg_list[n]+'['+str(d)+']);')
        if accs[n] == OPS_MAX and int(dims[n])>1:
          for d in range(0,int(dims[n])):
            code('p_a'+str(n)+'_'+str(d)+' = MAX(p_a'+str(n)+'_'+str(d)+''+arg_list[n]+'['+str(d)+']);')
        if accs[n] == OPS_INC and int(dims[n])>1:
          for d in range(0,int(dims[n])):
            code('p_a'+str(n)+'_'+str(d)+' +='+arg_list[n]+'['+str(d)+'];')
        if accs[n] == OPS_WRITE and int(dims[n])>1: #this may not be correct
          for d in range(0,int(dims[n])):
            code('p_a'+str(n)+'_'+str(d)+' +='+arg_list[n]+'['+str(d)+'];')

    ENDFOR()
    if NDIM>1:
      ENDFOR()
    if NDIM>2:
      ENDFOR()

    for n in range (0, nargs):
      if arg_typ[n] == 'ops_arg_gbl':
        if accs[n] <> OPS_READ:
          for d in range(0,int(dims[n])):
            code('p_a'+str(n)+'['+str(d)+'] = p_a'+str(n)+'_'+str(d)+';')

    if gen_full_code==1:
      IF('OPS_diags > 1')
      code('ops_timers_core(&c2,&t2);')
      code('OPS_kernels['+str(nk)+'].time += t2-t1;')
      ENDIF()

      code('ops_set_dirtybit_host(args, '+str(nargs)+');')
      for n in range (0, nargs):
        if arg_typ[n] == 'ops_arg_dat' and (accs[n] == OPS_WRITE or accs[n] == OPS_RW or accs[n] == OPS_INC):
          #code('ops_set_halo_dirtybit(&args['+str(n)+']);')
          code('ops_set_halo_dirtybit3(&args['+str(n)+'],range);')

      code('')
      IF('OPS_diags > 1')
      comm('Update kernel record')
      code('ops_timers_core(&c1,&t1);')
      code('OPS_kernels['+str(nk)+'].mpi_time += t1-t2;')
      for n in range (0, nargs):
        if arg_typ[n] == 'ops_arg_dat':
          code('OPS_kernels['+str(nk)+'].transfer += ops_compute_transfer(dim, start, end, &arg'+str(n)+');')
      ENDIF()
    config.depth = config.depth - 2
    code('}')
    for n in range (0, nargs):
      if arg_typ[n] == 'ops_arg_dat':
        if int(dims[n]) > 1:
          code('#undef OPS_ACC_MD'+str(n))
        else:
          code('#undef OPS_ACC'+str(n))
    code('')

    code('')
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
    code('desc->index = '+str(nk)+';')
    
    code('#ifdef OPS_MPI')
    code('sub_block_list sb = OPS_sub_block_list[block->index];')
    code('if (!sb->owned) return;')
    FOR('n','0',str(NDIM))
    code('desc->range[2*n] = sb->decomp_disp[n];desc->range[2*n+1] = sb->decomp_disp[n]+sb->decomp_size[n];')
    IF('desc->range[2*n] >= range[2*n]')
    code('desc->range[2*n] = 0;')
    ENDIF()
    ELSE()
    code('desc->range[2*n] = range[2*n] - desc->range[2*n];')
    ENDIF()
    code('if (sb->id_m[n]==MPI_PROC_NULL && range[2*n] < 0) desc->range[2*n] = range[2*n];')
    IF('desc->range[2*n+1] >= range[2*n+1]')
    code('desc->range[2*n+1] = range[2*n+1] - sb->decomp_disp[n];')
    ENDIF()
    ELSE()
    code('desc->range[2*n+1] = sb->decomp_size[n];')
    ENDIF()
    code('if (sb->id_p[n]==MPI_PROC_NULL && (range[2*n+1] > sb->decomp_disp[n]+sb->decomp_size[n]))')
    code('  desc->range[2*n+1] += (range[2*n+1]-sb->decomp_disp[n]-sb->decomp_size[n]);')
    ENDFOR()
    code('#else //OPS_MPI')
    FOR('i','0',str(2*NDIM))
    code('desc->range[i] = range[i];')
    ENDFOR()
    code('#endif //OPS_MPI')

    code('desc->nargs = '+str(nargs)+';')
    code('desc->args = (ops_arg*)malloc('+str(nargs)+'*sizeof(ops_arg));')
    declared = 0
    for n in range (0, nargs):
      code('desc->args['+str(n)+'] = arg'+str(n)+';')
      if arg_typ[n] == 'ops_arg_gbl' and accs[n] == OPS_READ:
        if declared == 0:
          code('char *tmp = (char*)malloc('+dims[n]+'*sizeof('+typs[n]+'));')
          declared = 1
        else:
          code('tmp = (char*)malloc('+dims[n]+'*sizeof('+typs[n]+'));')
        code('memcpy(tmp, arg'+str(n)+'.data,'+dims[n]+'*sizeof('+typs[n]+'));')
        code('desc->args['+str(n)+'].data = tmp;')
    code('desc->function = ops_par_loop_'+name+'_execute;')
    code('ops_enqueue_kernel(desc);')
    depth = 0
    code('}')

##########################################################################
#  output individual kernel file
##########################################################################
    if not os.path.exists('./MPI'):
      os.makedirs('./MPI')
    fid = open('./MPI/'+name+'_seq_kernel.cpp','w')
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
  if NDIM==2:
    code('#define OPS_2D')
  if NDIM==3:
    code('#define OPS_3D')
  code('#define OPS_ACC_MACROS')
  code('#define OPS_ACC_MD_MACROS')
  code('#include "ops_lib_cpp.h"')
  code('#ifdef OPS_MPI')
  code('#include "ops_mpi_core.h"')
  code('#endif')
  if os.path.exists('./user_types.h'):
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

  comm('user kernel files')

  kernel_name_list = []

  for nk in range(0,len(kernels)):
    if kernels[nk]['name'] not in kernel_name_list :
      code('#include "'+kernels[nk]['name']+'_seq_kernel.cpp"')
      kernel_name_list.append(kernels[nk]['name'])

  master = master.split('.')[0]
  fid = open('./MPI/'+master.split('.')[0]+'_seq_kernels.cpp','w')
  fid.write('//\n// auto-generated by ops.py//\n\n')
  fid.write(config.file_text)
  fid.close()
