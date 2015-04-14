
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
import glob

import util
import config

para_parse = util.para_parse
comment_remover = util.comment_remover
remove_trailing_w_space = util.remove_trailing_w_space
parse_signature = util.parse_signature_cuda
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


def ops_gen_mpi_cuda(master, date, consts, kernels):

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
    n_per_line = 4

    i = name.find('kernel')
    name2 = name[0:i-1]
    #print name2

    reduction = False
    ng_args = 0

    for n in range (0, nargs):
      if arg_typ[n] == 'ops_arg_gbl' and accs[n] <> OPS_READ:
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



    for n in range (0, nargs):
      if arg_typ[n] == 'ops_arg_dat':
        code('__constant__ int xdim'+str(n)+'_'+name+';')
        code('int xdim'+str(n)+'_'+name+'_h = -1;')
        if NDIM==3:
          code('__constant__ int ydim'+str(n)+'_'+name+';')
        code('int ydim'+str(n)+'_'+name+'_h = -1;')
    code('')

    #code('#define OPS_ACC_MACROS')
    for n in range (0, nargs):
      if arg_typ[n] == 'ops_arg_dat':
        if int(dims[n]) == 1:
          if NDIM==1:
            code('#define OPS_ACC'+str(n)+'(x) (x)')
          if NDIM==2:
            code('#define OPS_ACC'+str(n)+'(x,y) (x+xdim'+str(n)+'_'+name+'*(y))')
          if NDIM==3:
            code('#define OPS_ACC'+str(n)+'(x,y,z) (x+xdim'+str(n)+'_'+name+'*(y)+xdim'+str(n)+'_'+name+'*ydim'+str(n)+'_'+name+'*(z))')
    code('')

    for n in range (0, nargs):
      if arg_typ[n] == 'ops_arg_dat':
        if int(dims[n]) > 1:
          if NDIM==1:
            code('#define OPS_ACC_MD'+str(n)+'(d,x) ((x)*'+str(dims[n])+'+(d))')
          if NDIM==2:
            code('#define OPS_ACC_MD'+str(n)+'(d,x,y) ((x)*'+str(dims[n])+'+(d)+(xdim'+str(n)+'_'+name+'*(y)*'+str(dims[n])+'))')
          if NDIM==3:
            code('#define OPS_ACC_MD'+str(n)+'(d,x,y,z) ((x)*'+str(dims[n])+'+(d)+(xdim'+str(n)+'_'+name+'*(y)*'+str(dims[n])+')+(xdim'+str(n)+'_'+name+'*ydim'+str(n)+'_'+name+'*(z)*'+str(dims[n])+'))')


##########################################################################
#  generate headder
##########################################################################

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

    i2 = i
    i = text[0:i].rfind('\n') #reverse find
    j = text[i:].find('{')
    k = para_parse(text, i+j, '{', '}')
    arg_list = parse_signature(text[i2+len(name):i+j])
    check_accs(name, arg_list, arg_typ, text[i+j:k])
    code('__device__')
    code(text[i:k+2])
    code('')
    code('')
    for n in range (0, nargs):
      if arg_typ[n] == 'ops_arg_dat':
        if int(dims[n]) == 1:
          code('#undef OPS_ACC'+str(n))
    code('')
    for n in range (0, nargs):
      if arg_typ[n] == 'ops_arg_dat':
        if int(dims[n]) > 1:
          code('#undef OPS_ACC_MD'+str(n))
    code('')


##########################################################################
#  generate cuda kernel wrapper function
##########################################################################

    code('__global__ void ops_'+name+'(')
    for n in range (0, nargs):
      if arg_typ[n] == 'ops_arg_dat'and accs[n] == OPS_READ:
        code('const '+typs[n]+'* __restrict arg'+str(n)+',')
      elif arg_typ[n] == 'ops_arg_dat'and (accs[n] == OPS_WRITE or accs[n] == OPS_RW or accs[n] == OPS_INC) :
        code(typs[n]+'* __restrict arg'+str(n)+',')
      elif arg_typ[n] == 'ops_arg_gbl':
        if accs[n] == OPS_READ:
          if dims[n].isdigit() and int(dims[n])==1:
            code('const '+typs[n]+' arg'+str(n)+',')
          else:
            code('const '+typs[n]+'* __restrict arg'+str(n)+',')
        else:
          code(typs[n]+'* __restrict arg'+str(n)+',')
      elif arg_typ[n] == 'ops_arg_idx':
        if NDIM==1:
          code('int arg_idx0,')
        elif NDIM==2:
          code('int arg_idx0, int arg_idx1,')
        elif NDIM==3:
          code('int arg_idx0, int arg_idx1, int arg_idx2,')


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
      if arg_typ[n] == 'ops_arg_gbl' and accs[n] <> OPS_READ:
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
        if NDIM == 1:
          code('arg'+str(n)+' += idx_x * '+str(stride[NDIM*n])+'*'+str(dims[n])+';')
        elif NDIM == 2:
          code('arg'+str(n)+' += idx_x * '+str(stride[NDIM*n])+'*'+str(dims[n])+' + idx_y * '+str(stride[NDIM*n+1])+'*'+str(dims[n])+' * xdim'+str(n)+'_'+name+';')
        elif NDIM==3:
          code('arg'+str(n)+' += idx_x * '+str(stride[NDIM*n])+'*'+str(dims[n])+' + idx_y * '+str(stride[NDIM*n+1])+'*'+str(dims[n])+' * xdim'+str(n)+'_'+name+' + idx_z * '+str(stride[NDIM*n+2])+'*'+str(dims[n])+' * xdim'+str(n)+'_'+name+' * ydim'+str(n)+'_'+name+';')

    code('')
    n_per_line = 5
    if NDIM==1:
      IF('idx_x < size0')
    if NDIM==2:
      IF('idx_x < size0 && idx_y < size1')
    elif NDIM==3:
      IF('idx_x < size0 && idx_y < size1 && idx_z < size2')
    text = name+'('
    for n in range (0, nargs):
      if arg_typ[n] == 'ops_arg_dat':
        text = text +'arg'+str(n)
      elif arg_typ[n] == 'ops_arg_gbl' and accs[n] == OPS_READ:
        if dims[n].isdigit() and int(dims[n])==1:
          text = text +'&arg'+str(n)
        else:
          text = text +'arg'+str(n)
      elif arg_typ[n] == 'ops_arg_gbl' and accs[n] <> OPS_READ:
        text = text +'arg'+str(n)+'_l'
      elif arg_typ[n] == 'ops_arg_idx':
        text = text +'arg_idx'


      if nargs <> 1 and n <> nargs-1:
        if n%n_per_line <> 3:
          text = text +', '
        else:
          text = text +','
      else:
        text = text +');'
      if n%n_per_line == 3 and n <> nargs-1:
         text = text +'\n                   '
    code(text)
    ENDIF()

    #reduction across blocks
    if NDIM==1:
      cont = 'blockIdx.x + blockIdx.y*gridDim.x'
    if NDIM==2:
      cont = 'blockIdx.x + blockIdx.y*gridDim.x'
    elif NDIM==3:
      cont = 'blockIdx.x + blockIdx.y*gridDim.x + blockIdx.z*gridDim.x*gridDim.y'
    for n in range (0, nargs):
      if arg_typ[n] == 'ops_arg_gbl' and accs[n] == OPS_INC:
        code('for (int d=0; d<'+str(dims[n])+'; d++)')
        code('  ops_reduction_cuda<OPS_INC>(&arg'+str(n)+'[d+'+cont+'],arg'+str(n)+'_l[d]);')
      if arg_typ[n] == 'ops_arg_gbl' and accs[n] == OPS_MIN:
        code('for (int d=0; d<'+str(dims[n])+'; d++)')
        code('  ops_reduction_cuda<OPS_MIN>(&arg'+str(n)+'[d+'+cont+'],arg'+str(n)+'_l[d]);')
      if arg_typ[n] == 'ops_arg_gbl' and accs[n] == OPS_MAX:
        code('for (int d=0; d<'+str(dims[n])+'; d++)')
        code('  ops_reduction_cuda<OPS_MAX>(&arg'+str(n)+'[d+'+cont+'],arg'+str(n)+'_l[d]);')


    code('')
    config.depth = config.depth - 2
    code('}')


##########################################################################
#  now host stub
##########################################################################
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
    comm('compute locally allocated range for the sub-block')

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
    code('#else //OPS_MPI')
    FOR('n','0',str(NDIM))
    code('start[n] = range[2*n];end[n] = range[2*n+1];')
    ENDFOR()
    code('#endif //OPS_MPI')

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
      code('#else //OPS_MPI')
      for n in range (0,NDIM):
        code('arg_idx['+str(n)+'] = start['+str(n)+'];')
      code('#endif //OPS_MPI')

    for n in range (0, nargs):
      if arg_typ[n] == 'ops_arg_dat':
        code('int xdim'+str(n)+' = args['+str(n)+'].dat->size[0];')#*args['+str(n)+'].dat->dim;')
        if NDIM==3:
          code('int ydim'+str(n)+' = args['+str(n)+'].dat->size[1];')
    code('')

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
        code('cudaMemcpyToSymbol( xdim'+str(n)+'_'+name+', &xdim'+str(n)+', sizeof(int) );')
        code('xdim'+str(n)+'_'+name+'_h = xdim'+str(n)+';')
        if NDIM==3:
          code('cudaMemcpyToSymbol( ydim'+str(n)+'_'+name+', &ydim'+str(n)+', sizeof(int) );')
          code('ydim'+str(n)+'_'+name+'_h = ydim'+str(n)+';')
    ENDIF()

    code('')

    #setup reduction variables
    code('')
    for n in range (0, nargs):
        if arg_typ[n] == 'ops_arg_gbl' and (accs[n] <> OPS_READ or (accs[n] == OPS_READ and (not dims[n].isdigit() or int(dims[n])>1))):
          if (accs[n] == OPS_READ):
            code(''+typs[n]+' *arg'+str(n)+'h = ('+typs[n]+' *)arg'+str(n)+'.data;')
          else:
            code('#ifdef OPS_MPI')
            code(typs[n]+' *arg'+str(n)+'h = ('+typs[n]+' *)(((ops_reduction)args['+str(n)+'].data)->data + ((ops_reduction)args['+str(n)+'].data)->size * block->index);')
            code('#else //OPS_MPI')
            code(typs[n]+' *arg'+str(n)+'h = ('+typs[n]+' *)(((ops_reduction)args['+str(n)+'].data)->data);')
            code('#endif //OPS_MPI')

    code('')

    #set up CUDA grid and thread blocks for kernel call
    if NDIM==1:
      code('dim3 grid( (x_size-1)/OPS_block_size_x+ 1, 1, 1);')
    if NDIM==2:
      code('dim3 grid( (x_size-1)/OPS_block_size_x+ 1, (y_size-1)/OPS_block_size_y + 1, 1);')
    if NDIM==3:
      code('dim3 grid( (x_size-1)/OPS_block_size_x+ 1, (y_size-1)/OPS_block_size_y + 1, z_size);')

    if NDIM>1:
      code('dim3 tblock(OPS_block_size_x,OPS_block_size_y,1);')
    else:
      code('dim3 tblock(OPS_block_size_x,1,1);')

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
        code('int nblocks = ((x_size-1)/OPS_block_size_x+ 1);')
      elif NDIM==2:
        code('int nblocks = ((x_size-1)/OPS_block_size_x+ 1)*((y_size-1)/OPS_block_size_y + 1);')
      elif NDIM==3:
        code('int nblocks = ((x_size-1)/OPS_block_size_x+ 1)*((y_size-1)/OPS_block_size_y + 1)*z_size;')
      code('int maxblocks = nblocks;')
      code('int reduct_bytes = 0;')
      code('int reduct_size = 0;')
      code('')

    if GBL_READ == True and GBL_READ_MDIM == True:
      code('int consts_bytes = 0;')
      code('')

    for n in range (0, nargs):
      if arg_typ[n] == 'ops_arg_gbl':
        if accs[n] == OPS_READ and (not dims[n].isdigit() or int(dims[n])>1):
          code('consts_bytes += ROUND_UP('+str(dims[n])+'*sizeof('+typs[n]+'));')
        elif accs[n] <> OPS_READ:
          code('reduct_bytes += ROUND_UP(maxblocks*'+str(dims[n])+'*sizeof('+typs[n]+'));')
          code('reduct_size = MAX(reduct_size,sizeof('+typs[n]+')*'+str(dims[n])+');')
    code('')

    if GBL_READ == True and GBL_READ_MDIM == True:
      code('reallocConstArrays(consts_bytes);')
    if GBL_INC == True or GBL_MIN == True or GBL_MAX == True or GBL_WRITE == True:
      code('reallocReductArrays(reduct_bytes);')
      code('reduct_bytes = 0;')
      code('')

    for n in range (0, nargs):
      if arg_typ[n] == 'ops_arg_gbl' and accs[n] <> OPS_READ:
        code('arg'+str(n)+'.data = OPS_reduct_h + reduct_bytes;')
        code('arg'+str(n)+'.data_d = OPS_reduct_d + reduct_bytes;')
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
          code('consts_bytes = 0;')
          code('arg'+str(n)+'.data = OPS_consts_h + consts_bytes;')
          code('arg'+str(n)+'.data_d = OPS_consts_d + consts_bytes;')
          code('for (int d=0; d<'+str(dims[n])+'; d++) (('+typs[n]+' *)arg'+str(n)+'.data)[d] = arg'+str(n)+'h[d];')
          code('consts_bytes += ROUND_UP('+str(dims[n])+'*sizeof(int));')
    if GBL_READ == True and GBL_READ_MDIM == True:
      code('mvConstArraysToDevice(consts_bytes);')

    if GBL_INC == True or GBL_MIN == True or GBL_MAX == True or GBL_WRITE == True:
      code('mvReductArraysToDevice(reduct_bytes);')

    for n in range (0, nargs):
      if arg_typ[n] == 'ops_arg_dat':
        code('int dat'+str(n)+' = args['+str(n)+'].dat->elem_size;')

    code('')
    code('char *p_a['+str(nargs)+'];')



    comm('')
    comm('set up initial pointers')
    code('int d_m[OPS_MAX_DIM];')
    for n in range (0, nargs):
      if arg_typ[n] == 'ops_arg_dat':
        code('#ifdef OPS_MPI')
        code('for (int d = 0; d < dim; d++) d_m[d] = args['+str(n)+'].dat->d_m[d] + OPS_sub_dat_list[args['+str(n)+'].dat->index]->d_im[d];')
        code('#else //OPS_MPI')
        code('for (int d = 0; d < dim; d++) d_m[d] = args['+str(n)+'].dat->d_m[d];')
        code('#endif //OPS_MPI')
        code('int base'+str(n)+' = dat'+str(n)+' * 1 *')
        code('(start[0] * args['+str(n)+'].stencil->stride[0] - args['+str(n)+'].dat->base[0] - d_m[0]);')
        for d in range (1, NDIM):
          line = 'base'+str(n)+' = base'+str(n)+'+ dat'+str(n)+' *\n'
          for d2 in range (0,d):
            line = line + config.depth*' '+'  args['+str(n)+'].dat->size['+str(d2)+'] *\n'
          code(line[:-1])
          code('  (start['+str(d)+'] * args['+str(n)+'].stencil->stride['+str(d)+'] - args['+str(n)+'].dat->base['+str(d)+'] - d_m['+str(d)+']);')
        #code('printf("base'+str(n)+' = %d, d_m[0] = %d\\n",base'+str(n)+'/dat'+str(n)+',d_m[0]);')
        code('p_a['+str(n)+'] = (char *)args['+str(n)+'].data_d + base'+str(n)+';')
        code('')

    #halo exchange
    code('')
    code('ops_H_D_exchanges_device(args, '+str(nargs)+');')
    code('ops_halo_exchanges(args,'+str(nargs)+',range);')
    code('')
    code('ops_timers_core(&c1,&t1);')
    code('OPS_kernels['+str(nk)+'].mpi_time += t1-t2;')
    code('')


    #set up shared memory for reduction
    if GBL_INC == True or GBL_MIN == True or GBL_MAX == True or GBL_WRITE == True:
       code('int nshared = 0;')
       code('int nthread = OPS_block_size_x*OPS_block_size_y;')
       code('')
    for n in range (0, nargs):
      if arg_typ[n] == 'ops_arg_gbl' and accs[n] <> OPS_READ:
        code('nshared = MAX(nshared,sizeof('+typs[n]+')*'+str(dims[n])+');')
    code('')
    if GBL_INC == True or GBL_MIN == True or GBL_MAX == True or GBL_WRITE == True:
      code('nshared = MAX(nshared*nthread,reduct_size*nthread);')
      code('')


   #kernel call
    comm('call kernel wrapper function, passing in pointers to data')
    n_per_line = 2
    if GBL_INC == True or GBL_MIN == True or GBL_MAX == True or GBL_WRITE == True:
      text = 'ops_'+name+'<<<grid, tblock, nshared >>> ( '
    else:
      text = 'ops_'+name+'<<<grid, tblock >>> ( '
    for n in range (0, nargs):
      if arg_typ[n] == 'ops_arg_dat':
        text = text +' ('+typs[n]+' *)p_a['+str(n)+'],'
      elif arg_typ[n] == 'ops_arg_gbl':
        if dims[n].isdigit() and int(dims[n])==1 and accs[n]==OPS_READ:
          text = text +' *('+typs[n]+' *)arg'+str(n)+'.data,'
        else:
          text = text +' ('+typs[n]+' *)arg'+str(n)+'.data_d,'
      elif arg_typ[n] == 'ops_arg_idx':
        if NDIM==1:
          text = text + ' arg_idx[0],'
        if NDIM==2:
          text = text + ' arg_idx[0], arg_idx[1],'
        elif NDIM==3:
          text = text + ' arg_idx[0], arg_idx[1], arg_idx[2],'

      if n%n_per_line == 1 and n <> nargs-1:
        text = text +'\n          '
    if NDIM==1:
      text = text +'x_size);'
    if NDIM==2:
      text = text +'x_size, y_size);'
    elif NDIM==3:
      text = text +'x_size, y_size, z_size);'
    code(text);

    code('')

    #
    # Complete Reduction Operation by moving data onto host
    # and reducing over blocks
    #
    if GBL_INC == True or GBL_MIN == True or GBL_MAX == True or GBL_WRITE == True:
      code('mvReductArraysToHost(reduct_bytes);')

    for n in range (0, nargs):
      if arg_typ[n] == 'ops_arg_gbl' and accs[n] <> OPS_READ:
        FOR('b','0','maxblocks')
        FOR('d','0',str(dims[n]))
        if accs[n] == OPS_INC:
          code('arg'+str(n)+'h[d] = arg'+str(n)+'h[d] + ((double *)arg'+str(n)+'.data)[d+b*'+str(dims[n])+'];')
        elif accs[n] == OPS_MAX:
          code('arg'+str(n)+'h[d] = MAX(arg'+str(n)+'h[d],((double *)arg'+str(n)+'.data)[d+b*'+str(dims[n])+']);')
        elif accs[n] == OPS_MIN:
          code('arg'+str(n)+'h[d] = MIN(arg'+str(n)+'h[d],((double *)arg'+str(n)+'.data)[d+b*'+str(dims[n])+']);')
        ENDFOR()
        ENDFOR()
        code('arg'+str(n)+'.data = (char *)arg'+str(n)+'h;')
        code('')

    IF('OPS_diags>1')
    code('cutilSafeCall(cudaDeviceSynchronize());')
    ENDIF()
    code('ops_timers_core(&c2,&t2);')
    code('OPS_kernels['+str(nk)+'].time += t2-t1;')

    # This is not doen any more due to redution_handles treatement under MPI
    # if reduction == 1 :
    #   for n in range (0, nargs):
    #     if arg_typ[n] == 'ops_arg_gbl' and accs[n] <> OPS_READ:
    #       #code('ops_mpi_reduce(&arg'+str(n)+',('+typs[n]+' *)p_a['+str(n)+']);')
    #   code('ops_timers_core(&c1,&t1);')
    #   code('OPS_kernels['+str(nk)+'].mpi_time += t1-t2;')

    code('ops_set_dirtybit_device(args, '+str(nargs)+');')
    for n in range (0, nargs):
      if arg_typ[n] == 'ops_arg_dat' and (accs[n] == OPS_WRITE or accs[n] == OPS_RW or accs[n] == OPS_INC):
        code('ops_set_halo_dirtybit3(&args['+str(n)+'],range);')

    code('')
    comm('Update kernel record')
    for n in range (0, nargs):
      if arg_typ[n] == 'ops_arg_dat':
        code('OPS_kernels['+str(nk)+'].transfer += ops_compute_transfer(dim, range, &arg'+str(n)+');')
    config.depth = config.depth - 2
    code('}')


##########################################################################
#  output individual kernel file
##########################################################################
    if not os.path.exists('./CUDA'):
      os.makedirs('./CUDA')
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
  code('#define OPS_ACC_MD_MACROS')
  if NDIM==2:
    code('#define OPS_2D')
  if NDIM==3:
    code('#define OPS_3D')
  code('#include "ops_lib_cpp.h"')
  code('#include "ops_cuda_rt_support.h"')
  code('#include "ops_cuda_reduction.h"')
  if os.path.exists('./user_types.h'):
    code('#include "user_types.h"')
  code('#ifdef OPS_MPI')
  code('#include "ops_mpi_core.h"')
  code('#endif')
  for n in range (0,18):
   code('#undef OPS_ACC'+str(n))
  code('')
  comm(' global constants')
  for nc in range (0,len(consts)):
    if consts[nc]['dim'].isdigit() and int(consts[nc]['dim'])==1:
      code('__constant__ '+consts[nc]['type']+' '+(str(consts[nc]['name']).replace('"','')).strip()+';')
    else:
      if consts[nc]['dim'].isdigit() and consts[nc]['dim'] > 0:
        num = str(consts[nc]['dim'])
        code('__constant__ '+consts[nc]['type']+' '+(str(consts[nc]['name']).replace('"','')).strip()+'['+num+'];')
      else:
        code('__constant__ '+consts[nc]['type']+' *'+(str(consts[nc]['name']).replace('"','')).strip()+';')



  code('')
  code('void ops_decl_const_char(int dim, char const *type,')
  code('int size, char *dat, char const *name){')
  config.depth = config.depth + 2

  for nc in range(0,len(consts)):
    IF('!strcmp(name,"'+(str(consts[nc]['name']).replace('"','')).strip()+'")')
    if consts[nc]['dim'].isdigit():
      code('cutilSafeCall(cudaMemcpyToSymbol('+(str(consts[nc]['name']).replace('"','')).strip()+', dat, dim*size));')
    else:
      code('char *temp; cutilSafeCall(cudaMalloc((void**)&temp,dim*size));')
      code('cutilSafeCall(cudaMemcpy(temp,dat,dim*size,cudaMemcpyHostToDevice));')
      code('cutilSafeCall(cudaMemcpyToSymbol('+(str(consts[nc]['name']).replace('"','')).strip()+', &temp, sizeof(char *)));')
    ENDIF()
    code('else')

  code('{')
  config.depth = config.depth + 2
  code('printf("error: unknown const name\\n"); exit(1);')
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

  master = master.split('.')[0]
  fid = open('./CUDA/'+master.split('.')[0]+'_kernels.cu','w')
  fid.write('//\n// auto-generated by ops.py//\n\n')
  fid.write(config.file_text)
  fid.close()
