
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
OPS OpenMP code generator

This routine is called by ops.py which parses the input files

It produces a file xxx_omp_kernel.cpp for each kernel,
plus a master kernel file

"""

import re
import datetime
import os

import util
import config

para_parse = util.para_parse
comment_remover = util.comment_remover
remove_trailing_w_space = util.remove_trailing_w_space
parse_signature = util.parse_signature_openacc
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

def ops_gen_mpi_inline(master, date, consts, kernels):

  OPS_ID   = 1;  OPS_GBL   = 2;

  OPS_READ = 1;  OPS_WRITE = 2;  OPS_RW  = 3;
  OPS_INC  = 4;  OPS_MAX   = 5;  OPS_MIN = 6;

  accsstring = ['OPS_READ','OPS_WRITE','OPS_RW','OPS_INC','OPS_MAX','OPS_MIN' ]

  NDIM = 2 #the dimension of the application is hardcoded here .. need to get this dynamically


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

    reduct = 0
    for n in range (0, nargs):
      if arg_typ[n] == 'ops_arg_gbl' and accs[n] <> OPS_READ:
        reduct = 1

    config.file_text = ''
    config.depth = 0
    n_per_line = 2

    if NDIM==3:
      n_per_line = 1

    i = name.find('kernel')
    name2 = name[0:i-1]

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

    code('#include "./MPI_inline/'+master.split('.')[0]+'_common.h"')
    code('')
    for n in range (0, nargs):
      if arg_typ[n] == 'ops_arg_dat':
        code('int xdim'+str(n)+'_'+name+';')
        if NDIM==3:
          code('int ydim'+str(n)+'_'+name+';')
    code('')

    #code('#define OPS_ACC_MACROS')
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
#  generate headder
##########################################################################

    comm('user function')
    fid = open(name2+'_kernel.h', 'r')
    text = fid.read()
    fid.close()
    text = comment_remover(text)

    text = remove_trailing_w_space(text)

    i = text.find(name)
    if(i < 0):
      print "\n********"
      print "Error: cannot locate user kernel function: "+name+" - Aborting code generation"
      exit(2)

    i2=i
    i = text[0:i].rfind('\n') #reverse find
    j = text[i:].find('{')
    k = para_parse(text, i+j, '{', '}')
    kernel_text = text[i+j+1:k]
    m = text.find(name)
    arg_list = parse_signature(text[i2+len(name):i+j])
    l = text[i:m].find('inline')
    if(l<0):
      text = text[i:k+2]
    else:
      text = text[i+l:k+2]


    #text = text[0:j].replace('*','* restrict ')+text[j:]
    i = text.find('{')
    i = text[0:i].rfind(')')
    if (NDIM==1):
      itervar = ', const int n_x'
    if (NDIM==2):
      itervar = ', const int n_x, const int n_y'
    if (NDIM==3):
      itervar = ', const int n_x, const int n_y, const int n_z'
    text = text[0:i]+itervar+text[i:]
#    code(text)
    code('')
    

    code('')
    code('')

##########################################################################
#  generate C wrapper
##########################################################################
    code('void '+name+'_c_wrapper(')
    config.depth = config.depth+2
    for n in range (0, nargs):
      if accs[n] == OPS_READ:
        pre = 'const '
      else:
        pre = ''
      if arg_typ[n] == 'ops_arg_gbl' and accs[n] == OPS_READ and dims[n].isdigit() and int(dims[n])==1:
        code(pre+typs[n]+' '+arg_list[n]+',')
      else:
        if arg_typ[n] == 'ops_arg_gbl' and accs[n] <> OPS_READ:
          code(pre+typs[n]+' * restrict '+arg_list[n]+'_g,')
        else:
          code(pre+typs[n]+' * restrict '+arg_list[n]+',')
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

    redlist=''
    for n in range (0, nargs):
      if arg_typ[n] == 'ops_arg_gbl' and accs[n] <> OPS_READ:
        code(typs[n]+' '+arg_list[n]+'_v = *'+arg_list[n]+'_g;')
        if accs[n] == OPS_INC:
          redlist = redlist + ' reduction(+:'+arg_list[n]+'_v)'
        elif accs[n] == OPS_MIN:
          redlist = redlist + ' reduction(min:'+arg_list[n]+'_v)'
        elif accs[n] == OPS_MAX:
          redlist = redlist + ' reduction(max:'+arg_list[n]+'_v)'

    code('#pragma omp parallel for'+redlist)
    if NDIM==3:
      FOR('n_z','0','z_size')
      FOR('n_y','0','y_size')
    if NDIM==2:
      FOR('n_y','0','y_size')

    FOR('n_x','0','x_size')
    for n in range (0, nargs):
      if arg_typ[n] == 'ops_arg_gbl' and accs[n] <> OPS_READ:
        code(typs[n]+' * restrict '+arg_list[n]+' = &'+arg_list[n]+'_v;')

    for n in range (0, nargs):
      if arg_typ[n] == 'ops_arg_idx':
        if NDIM==1:
          code('int '+arg_list[n]+'[] = {arg_idx0+n_x};')
        elif NDIM==2:
          code('int '+arg_list[n]+'[] = {arg_idx0+n_x, arg_idx1+n_y};')
        elif NDIM==3:
          code('int '+arg_list[n]+'[] = {arg_idx0+n_x, arg_idx1+n_y, arg_idx2+n_z};')

    text = name+'( '
    for n in range (0, nargs):
      if arg_typ[n] == 'ops_arg_dat':
        text = text +' p_a'+str(n)+''
      elif arg_typ[n] == 'ops_arg_gbl':
        if accs[n] == OPS_READ:
          if dims[n].isdigit() and int(dims[n])==1:
            text = text +' &p_a'+str(n)+''
          else:
            text = text +' p_a'+str(n)+''
        else:
          if dims[n].isdigit() and int(dims[n]) == 1:
            text = text +' &p_a'+str(n)+'_l'
          else:
            text = text +' p_a'+str(n)+'_l'
      elif arg_typ[n] == 'ops_arg_idx':
        text = text +'arg_idx'

      if nargs <> 1 and n != nargs-1:
        text = text + ','
      else:
        if NDIM==1:
          text = text +', n_x);\n'
        if NDIM==2:
          text = text +',n_x, n_y );\n'
        if NDIM==3:
          text = text +', n_x, n_y, n_z );\n'
      if n%n_per_line == 0 and n <> nargs-1:
        text = text +'\n          '
    code(kernel_text)
    ENDFOR()
    if NDIM==2:
      ENDFOR()
    if NDIM==3:
      ENDFOR()
      ENDFOR()
   
    for n in range (0, nargs):
      if arg_typ[n] == 'ops_arg_gbl' and accs[n] <> OPS_READ:
        code('*'+arg_list[n]+'_g = '+arg_list[n]+'_v;')

    config.depth = config.depth-2
    code('}')
    for n in range (0, nargs):
      if arg_typ[n] == 'ops_arg_dat':
        if int(dims[n]) == 1:
          code('#undef OPS_ACC'+str(n))
    code('')
    for n in range (0, nargs):
      if arg_typ[n] == 'ops_arg_dat':
        if int(dims[n]) > 1:
          code('#undef OPS_ACC_MD'+str(n))

##########################################################################
#  output individual kernel file
##########################################################################
    if not os.path.exists('./MPI_inline'):
      os.makedirs('./MPI_inline')
    fid = open('./MPI_inline/'+name+'_mpiinline_kernel_c.c','w')
    date = datetime.datetime.now()
    fid.write('//\n// auto-generated by ops.py\n//\n')
    fid.write(config.file_text)
    fid.close()
    config.file_text = ''
##########################################################################
#  now host stub
##########################################################################

    code('#include "./MPI_inline/'+master.split('.')[0]+'_common.h"')
    code('')
    for n in range (0, nargs):
      if arg_typ[n] == 'ops_arg_dat':
        code('extern int xdim'+str(n)+'_'+name+';')
        code('int xdim'+str(n)+'_'+name+'_h = -1;')
        if NDIM==3:
          code('extern int ydim'+str(n)+'_'+name+';')
          code('int ydim'+str(n)+'_'+name+'_h = -1;')
    code('')

    code('#ifdef __cplusplus')
    code('extern "C" {')
    code('#endif')
    code('void '+name+'_c_wrapper(')
    config.depth = config.depth+2
    for n in range (0, nargs):
      if arg_typ[n] == 'ops_arg_gbl' and accs[n] == OPS_READ and dims[n].isdigit() and int(dims[n])==1:
        code(typs[n]+' p_a'+str(n)+',')
      else:
        code(typs[n]+' *p_a'+str(n)+',')
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

    code('');

    text ='ops_arg args['+str(nargs)+'] = {'
    for n in range (0, nargs):
      text = text +' arg'+str(n)
      if nargs <> 1 and n != nargs-1:
        text = text +','
      else:
        text = text +'};\n'
      if n%n_per_line == 5 and n <> nargs-1:
        text = text +'\n                    '
    code(text);
    code('')
    code('#ifdef CHECKPOINTING')
    code('if (!ops_checkpointing_before(args,'+str(nargs)+',range,'+str(nk)+')) return;')
    code('#endif')
    code('')
    code('ops_timing_realloc('+str(nk)+',"'+name+'");')
    code('OPS_kernels['+str(nk)+'].count++;')
    code('')
    comm('compute localy allocated range for the sub-block')

    code('int start['+str(NDIM)+'];')
    code('int end['+str(NDIM)+'];')


    code('#ifdef OPS_MPI')
    code('sub_block_list sb = OPS_sub_block_list[block->index];')
    code('if (!sb->owned) return;')
    FOR('n','0',str(NDIM))
    code('start[n] = sb->decomp_disp[n];end[n] = sb->decomp_disp[n]+sb->decomp_size[n];')
    IF('start[n] >= range[2*n]')
    code('start[n] = 0;')
    ENDIF()
    ELSE()
    code('start[n] = range[2*n] - start[n];')
    ENDIF()
    code('if (sb->id_m[n]==MPI_PROC_NULL && range[2*n] < 0) start[n] = range[2*n];')
    IF('end[n] >= range[2*n+1]')
    code('end[n] = range[2*n+1] - sb->decomp_disp[n];')
    ENDIF()
    ELSE()
    code('end[n] = sb->decomp_size[n];')
    ENDIF()
    code('if (sb->id_p[n]==MPI_PROC_NULL && (range[2*n+1] > sb->decomp_disp[n]+sb->decomp_size[n]))')
    code('  end[n] += (range[2*n+1]-sb->decomp_disp[n]-sb->decomp_size[n]);')
    ENDFOR()
    code('#else')
    FOR('n','0',str(NDIM))
    code('start[n] = range[2*n];end[n] = range[2*n+1];')
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

    if arg_idx:
      code('int arg_idx['+str(NDIM)+'];')
      code('#ifdef OPS_MPI')
      for n in range (0,NDIM):
        code('arg_idx['+str(n)+'] = sb->decomp_disp['+str(n)+']+start['+str(n)+'];')
      code('#else')
      for n in range (0,NDIM):
        code('arg_idx['+str(n)+'] = start['+str(n)+'];')
      code('#endif')
    code('')

    for n in range (0,nargs):
      if arg_typ[n] == 'ops_arg_dat':
        code('xdim'+str(n)+' = args['+str(n)+'].dat->size[0];')#*args['+str(n)+'].dat->dim;')
        if NDIM==3:
          code('ydim'+str(n)+' = args['+str(n)+'].dat->size[1];')
    #timing structs
    code('')
    comm('Timing')
    code('double t1,t2,c1,c2;')
    code('ops_timers_core(&c2,&t2);')
    code('')
    condition = ''
    for n in range (0, nargs):
      if arg_typ[n] == 'ops_arg_dat':
        condition = condition + 'xdim'+str(n)+' != xdim'+str(n)+'_'+name+'_h || '
        if NDIM==3:
          condition = condition + 'ydim'+str(n)+' != ydim'+str(n)+'_'+name+'_h || '
    condition = condition[:-4]
    IF(condition)

    for n in range (0, nargs):
      if arg_typ[n] == 'ops_arg_dat':
        code('xdim'+str(n)+'_'+name+' = xdim'+str(n)+';')
        code('xdim'+str(n)+'_'+name+'_h = xdim'+str(n)+';')
        if NDIM==3:
          code('ydim'+str(n)+'_'+name+' = ydim'+str(n)+';')
          code('ydim'+str(n)+'_'+name+'_h = ydim'+str(n)+';')
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

    for n in range (0, nargs):
      if arg_typ[n] == 'ops_arg_dat':
        #code('int off'+str(n)+'_1 = offs['+str(n)+'][0];')
        #code('int off'+str(n)+'_2 = offs['+str(n)+'][1];')
        code('int dat'+str(n)+' = args['+str(n)+'].dat->elem_size;')

    code('')
    for n in range (0, nargs):
        if arg_typ[n] == 'ops_arg_gbl':
          if accs[n] <> OPS_READ or (accs[n] == OPS_READ and (not dims[n].isdigit() or int(dims[n])>1)):
            #code(''+typs[n]+' *arg'+str(n)+'h = ('+typs[n]+' *)args['+str(n)+'].data;')
            if (accs[n] == OPS_READ):
              code(''+typs[n]+' *arg'+str(n)+'h = ('+typs[n]+' *)arg'+str(n)+'.data;')
            else:
              code('#ifdef OPS_MPI')
              code(typs[n]+' *arg'+str(n)+'h = ('+typs[n]+' *)(((ops_reduction)args['+str(n)+'].data)->data + ((ops_reduction)args['+str(n)+'].data)->size * block->index);')
              code('#else')
              code(typs[n]+' *arg'+str(n)+'h = ('+typs[n]+' *)(((ops_reduction)args['+str(n)+'].data)->data);')
              code('#endif')


    comm('')
    comm('set up initial pointers')
    code('int d_m[OPS_MAX_DIM];')
    for n in range (0, nargs):
      if arg_typ[n] == 'ops_arg_dat':
        code('#ifdef OPS_MPI')
        code('for (int d = 0; d < dim; d++) d_m[d] = args['+str(n)+'].dat->d_m[d] + OPS_sub_dat_list[args['+str(n)+'].dat->index]->d_im[d];')
        code('#else')
        code('for (int d = 0; d < dim; d++) d_m[d] = args['+str(n)+'].dat->d_m[d];')
        code('#endif')
        code('int base'+str(n)+' = dat'+str(n)+' * 1 * ')
        code('  (start[0] * args['+str(n)+'].stencil->stride[0] - args['+str(n)+'].dat->base[0] - d_m[0]);')
        for d in range (1, NDIM):
          line = 'base'+str(n)+' = base'+str(n)+'+ dat'+str(n)+' *\n'
          for d2 in range (0,d):
            line = line + config.depth*' '+'  args['+str(n)+'].dat->size['+str(d2)+'] *\n'
          code(line[:-1])
          code('  (start['+str(d)+'] * args['+str(n)+'].stencil->stride['+str(d)+'] - args['+str(n)+'].dat->base['+str(d)+'] - d_m['+str(d)+']);')

        code(typs[n]+' *p_a'+str(n)+' = ('+typs[n]+' *)((char *)args['+str(n)+'].data + base'+str(n)+');')
        code('')
      elif arg_typ[n] == 'ops_arg_gbl':
        if accs[n] == OPS_READ:
          if dims[n].isdigit() and int(dims[n])==1:
            code(typs[n]+' *p_a'+str(n)+' = ('+typs[n]+' *)args['+str(n)+'].data;')
          else:
            code(typs[n]+' *p_a'+str(n)+' = arg'+str(n)+'h;')
        else:
          code(typs[n]+' *p_a'+str(n)+' = arg'+str(n)+'h;')
      else:
        code(typs[n]+' *p_a'+str(n)+' = NULL;')
        code('')

    code('')
    code('ops_H_D_exchanges_host(args, '+str(nargs)+');')
    code('ops_halo_exchanges(args,'+str(nargs)+',range);')
    code('')
    code('ops_timers_core(&c1,&t1);')
    code('OPS_kernels['+str(nk)+'].mpi_time += t1-t2;')
    code('')


    code(name+'_c_wrapper(')
    config.depth = config.depth+2
    for n in range (0, nargs):
      if arg_typ[n] == 'ops_arg_gbl' and accs[n] == OPS_READ and dims[n].isdigit() and int(dims[n])==1:
        code('*p_a'+str(n)+',')
      else:
        code('p_a'+str(n)+',')
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





    # for n in range (0, nargs):
    #   if arg_typ[n] == 'ops_arg_gbl':
    #     if accs[n] <> OPS_READ:
    #       code('*('+typs[n]+' *)args['+str(n)+'].data = *p_a'+str(n)+';')
    code('')
    code('ops_timers_core(&c2,&t2);')
    code('OPS_kernels['+str(nk)+'].time += t2-t1;')

    # if reduction == 1 :
    #   for n in range (0, nargs):
    #     if arg_typ[n] == 'ops_arg_gbl' and accs[n] != OPS_READ:
    #       #code('ops_mpi_reduce(&arg'+str(n)+',('+typs[n]+' *)args['+str(n)+'].data);')
    #   code('ops_timers_core(&c1,&t1);')
    #   code('OPS_kernels['+str(nk)+'].mpi_time += t1-t2;')

    code('ops_set_dirtybit_host(args, '+str(nargs)+');')
    for n in range (0, nargs):
      if arg_typ[n] == 'ops_arg_dat' and (accs[n] == OPS_WRITE or accs[n] == OPS_RW or accs[n] == OPS_INC):
        code('ops_set_halo_dirtybit3(&args['+str(n)+'],range);')

    code('')
    comm('Update kernel record')
    for n in range (0, nargs):
      if arg_typ[n] == 'ops_arg_dat':
        code('OPS_kernels['+str(nk)+'].transfer += ops_compute_transfer(dim, start, end, &arg'+str(n)+');')
    config.depth = config.depth - 2
    code('}')


##########################################################################
#  output individual kernel file
##########################################################################
    if not os.path.exists('./MPI_inline'):
      os.makedirs('./MPI_inline')
    fid = open('./MPI_inline/'+name+'_mpiinline_kernel.cpp','w')
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
  code('#define OPS_ACC_MD_MACROS')
  if NDIM==2:
    code('#define OPS_2D')
  if NDIM==3:
    code('#define OPS_3D')
  code('#ifdef __cplusplus')
  code('#include "ops_lib_cpp.h"')
  code('#endif')
  code('#ifdef OPS_MPI')
  code('#include "ops_mpi_core.h"')
  code('#endif')
  if os.path.exists('./user_types.h'):
    code('#include "user_types.h"')
  code('')

  for n in range(0,17):
    code('#undef OPS_ACC'+str(n))

  comm(' global constants')
  for nc in range (0,len(consts)):
    if consts[nc]['dim'].isdigit() and int(consts[nc]['dim'])==1:
      code('extern '+consts[nc]['type']+' '+(str(consts[nc]['name']).replace('"','')).strip()+';')
#      code('#pragma acc declare create('+(str(consts[nc]['name']).replace('"','')).strip()+')')
    else:
      if consts[nc]['dim'].isdigit() and consts[nc]['dim'] > 0:
        num = str(consts[nc]['dim'])
        code('extern '+consts[nc]['type']+' '+(str(consts[nc]['name']).replace('"','')).strip()+'['+num+'];')
#        code('#pragma acc declare create('+(str(consts[nc]['name']).replace('"','')).strip()+')')
      else:
        code('extern '+consts[nc]['type']+' *'+(str(consts[nc]['name']).replace('"','')).strip()+';')
#        code('#pragma acc declare create('+(str(consts[nc]['name']).replace('"','')).strip()+')')

  fid = open('./MPI_inline/'+master.split('.')[0]+'_common.h','w')
  fid.write('//\n// auto-generated by ops.py\n//\n')
  fid.write(config.file_text)
  fid.close()
  config.file_text =''
  code('#include "./MPI_inline/'+master.split('.')[0]+'_common.h"')
  code('')
  code('')
  code('void ops_decl_const_char2(int dim, char const *type,')
  code('int size, char *dat, char const *name){')
  config.depth = config.depth + 2

  for nc in range(0,len(consts)):
    IF('!strcmp(name,"'+(str(consts[nc]['name']).replace('"','')).strip()+'")')
    if consts[nc]['dim'].isdigit() and int(consts[nc]['dim'])==1:
      code((str(consts[nc]['name']).replace('"','')).strip()+' = *('+consts[nc]['type']+'*)dat;')
    else:
      code((str(consts[nc]['name']).replace('"','')).strip()+' = ('+consts[nc]['type']+'*)dat;')
    ENDIF()
    code('else')

  code('{')
  config.depth = config.depth + 2
  code('printf("error: unknown const name\\n"); exit(1);')
  ENDIF()

  config.depth = config.depth - 2
  code('}')
  code('')
  #code('')
  #comm('user kernel files')

  #kernel_name_list = ['generate_chunk_kernel']

  #for nk in range(0,len(kernels)):
  #  if kernels[nk]['name'] not in kernel_name_list :
  #    code('#include "'+kernels[nk]['name']+'_openacc_kernel.cpp"')
  #    kernel_name_list.append(kernels[nk]['name'])

  fid = open('./MPI_inline/'+master.split('.')[0]+'_kernels.cpp','w')
  fid.write('//\n// auto-generated by ops.py//\n\n')
  fid.write(config.file_text)
  fid.close()
