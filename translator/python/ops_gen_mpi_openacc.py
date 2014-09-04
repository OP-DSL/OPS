
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

def comm(line):
  global file_text, FORTRAN, CPP
  global depth
  prefix = ' '*depth
  if len(line) == 0:
    file_text +='\n'
  else:
    file_text +=prefix+'//'+line+'\n'

def code(text):
  global file_text, g_m
  global depth
  prefix = ''
  if len(text) != 0:
    prefix = ' '*depth
  #file_text += prefix+rep(text,g_m)+'\n'
  file_text += prefix+text+'\n'

def FOR(i,start,finish):
  global file_text
  global depth
  code('for ( int '+i+'='+start+'; '+i+'<'+finish+'; '+i+'++ ){')
  depth += 2

def FOR2(i,start,finish,increment):
  global file_text
  global depth
  code('for ( int '+i+'='+start+'; '+i+'<'+finish+'; '+i+'+='+increment+' ){')
  depth += 2

def WHILE(line):
  global file_text
  global depth
  code('while ( '+ line+ ' ){')
  depth += 2

def ENDWHILE():
  global file_text
  global depth
  depth -= 2
  code('}')

def ENDFOR():
  global file_text
  global depth
  depth -= 2
  code('}')

def IF(line):
  global file_text
  global depth
  code('if ('+ line + ') {')
  depth += 2

def ELSEIF(line):
  global file_text
  global depth
  code('else if ('+ line + ') {')
  depth += 2

def ELSE():
  global file_text
  global depth
  code('else {')
  depth += 2

def ENDIF():
  global file_text
  global depth
  depth -= 2
  code('}')


def para_parse(text, j, op_b, cl_b):
    """Parsing code block, i.e. text to find the correct closing brace"""

    depth = 0
    loc2 = j

    while 1:
      if text[loc2] == op_b:
            depth = depth + 1

      elif text[loc2] == cl_b:
            depth = depth - 1
            if depth == 0:
                return loc2
      loc2 = loc2 + 1

def comment_remover(text):
    """Remove comments from text"""

    def replacer(match):
        s = match.group(0)
        if s.startswith('/'):
            return ''
        else:
            return s
    pattern = re.compile(
        r'//.*?$|/\*.*?\*/|\'(?:\\.|[^\\\'])*\'|"(?:\\.|[^\\"])*"',
        re.DOTALL | re.MULTILINE
    )
    return re.sub(pattern, replacer, text)


def remove_trailing_w_space(text):
  line_start = 0
  line = ""
  line_end = 0
  striped_test = ''
  count = 0
  while 1:
    line_end =  text.find("\n",line_start+1)
    line = text[line_start:line_end]
    line = line.rstrip()
    striped_test = striped_test + line +'\n'
    line_start = line_end + 1
    line = ""
    if line_end < 0:
      return striped_test


def ops_gen_mpi_openacc(master, date, consts, kernels):

  global dims, stens
  global g_m, file_text, depth

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


    g_m = 0;
    file_text = ''
    depth = 0
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

    code('#include "./OpenACC/'+master.split('.')[0]+'_common.h"')
    code('')
    if not (('generate_chunk' in name) or ('calc_dt_kernel_print' in name)):
      code('#define OPS_GPU')
      code('')
    for n in range (0, nargs):
      if arg_typ[n] == 'ops_arg_dat':
        code('int xdim'+str(n)+'_'+name+';')
        if NDIM==3:
          code('int ydim'+str(n)+'_'+name+';')
#        code('#pragma acc declare create(xdim'+str(n)+'_'+name+')')
    code('')

    #code('#define OPS_ACC_MACROS')
    for n in range (0, nargs):
      if arg_typ[n] == 'ops_arg_dat':
        if NDIM==2:
          code('#define OPS_ACC'+str(n)+'(x,y) (x+xdim'+str(n)+'_'+name+'*(y))')
        if NDIM==3:
          code('#define OPS_ACC'+str(n)+'(x,y,z) (x+xdim'+str(n)+'_'+name+'*(y)+xdim'+str(n)+'_'+name+'*ydim'+str(n)+'_'+name+'*(z))')
    code('')


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

    i = text[0:i].rfind('\n') #reverse find
    j = text[i:].find('{')
    k = para_parse(text, i+j, '{', '}')
    m = text.find(name)
    l = text[i:m].find('inline')
    if(l<0):
      code('inline '+text[i:k+2])
    else:
      code(text[i:k+2])
    code('')
    code('')
    for n in range (0, nargs):
      if arg_typ[n] == 'ops_arg_dat':
        code('#undef OPS_ACC'+str(n))
    code('')
    code('')

##########################################################################
#  generate C wrapper
##########################################################################
    code('void '+name+'_c_wrapper(')
    depth = depth+2
    for n in range (0, nargs):
      if arg_typ[n] == 'ops_arg_gbl' and accs[n] == OPS_READ and dims[n].isdigit() and int(dims[n])==1:
        code(typs[n]+' p_a'+str(n)+',')
      else:
        code(typs[n]+' *p_a'+str(n)+',')
    if arg_idx:
      if NDIM == 2:
        code('int arg_idx0, int arg_idx1,')
      elif NDIM == 3:
        code('int arg_idx0, int arg_idx1, int arg_idx2,')

    if NDIM == 2:
      code('int x_size, int y_size) {')
    if NDIM == 3:
      code('int x_size, int y_size, int z_size) {')


    for n in range (0,nargs):
      if arg_typ[n] == 'ops_arg_gbl':
        if accs[n] <> OPS_READ:
          if dims[n].isdigit() and int(dims[n]) == 1:
            code(typs[n]+' p_a'+str(n)+'_l = *p_a'+str(n)+';')
          else:
            code(typs[n]+' p_a'+str(n)+'_l['+str(dims[n])+'];')
            code('for (int d = 0; d < '+str(dims[n])+'; d++) p_a'+str(n)+'_l[d] = p_a'+str(n)+'[d];')
    line = '#pragma acc parallel deviceptr('
    for n in range (0,nargs):
      if arg_typ[n] == 'ops_arg_dat':
        line = line + 'p_a'+str(n)+','
      if arg_typ[n] == 'ops_arg_gbl':
        if accs[n] == OPS_READ and (not dims[n].isdigit() or int(dims[n])>1):
          line = line + 'p_a'+str(n)+','
    line = line[:-1]+')'
    for n in range (0,nargs):
      if arg_typ[n] == 'ops_arg_gbl':
        if accs[n] == OPS_MIN:
          line = line + ' reduction(min:p_a'+str(n)+'_l)'
        if accs[n] == OPS_MAX:
          line = line + ' reduction(max:p_a'+str(n)+'_l)'
        if accs[n] == OPS_INC:
          line = line + ' reduction(+:p_a'+str(n)+'_l)'
        if accs[n] == OPS_WRITE: #this may not be correct
          line = line + ' reduction(+:p_a'+str(n)+'_l)'
    code('#ifdef OPS_GPU')
    code(line)
    line = '#pragma acc loop'
    for n in range (0,nargs):
      if arg_typ[n] == 'ops_arg_gbl':
        if accs[n] == OPS_MIN:
          line = line + ' reduction(min:p_a'+str(n)+'_l)'
        if accs[n] == OPS_MAX:
          line = line + ' reduction(max:p_a'+str(n)+'_l)'
        if accs[n] == OPS_INC:
          line = line + ' reduction(+:p_a'+str(n)+'_l)'
        if accs[n] == OPS_WRITE: #this may not be correct
          line = line + ' reduction(+:p_a'+str(n)+'_l)'
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
    FOR('n_x','0','x_size')
    if arg_idx:
      if NDIM==2:
        code('int arg_idx[] = {arg_idx0+n_x, arg_idx1+n_y};')
      elif NDIM==3:
        code('int arg_idx[] = {arg_idx0+n_x, arg_idx1+n_y, arg_idx2+n_z};')

    text = name+'( '
    for n in range (0, nargs):
      if arg_typ[n] == 'ops_arg_dat':
        text = text +' p_a'+str(n)+' + n_x*'+str(stride[NDIM*n])+' + n_y*xdim'+str(n)+'_'+name+'*'+str(stride[NDIM*n+1])
        if NDIM == 3:
          text = text + ' + n_z*xdim'+str(n)+'_'+name+'*ydim'+str(n)+'_'+name+'*'+str(stride[NDIM*n+2])
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
        text = text +' );\n'
      if n%n_per_line == 0 and n <> nargs-1:
        text = text +'\n          '
    code(text);
    ENDFOR()
    ENDFOR()
    if NDIM==3:
      ENDFOR()
    for n in range (0, nargs):
      if arg_typ[n] == 'ops_arg_gbl':
        if accs[n] <> OPS_READ:
          if dims[n].isdigit() and int(dims[n]) == 1:
            code('*p_a'+str(n)+' = p_a'+str(n)+'_l;')
          else:
            code('for (int d = 0; d < '+str(dims[n])+'; d++) p_a'+str(n)+'[d] = p_a'+str(n)+'_l[d];')
    depth = depth-2
    code('}')

##########################################################################
#  output individual kernel file
##########################################################################
    if not os.path.exists('./OpenACC'):
      os.makedirs('./OpenACC')
    fid = open('./OpenACC/'+name+'_openacc_kernel_c.c','w')
    date = datetime.datetime.now()
    fid.write('//\n// auto-generated by ops.py\n//\n')
    fid.write(file_text)
    fid.close()
    file_text = ''
##########################################################################
#  now host stub
##########################################################################

    code('#include "./OpenACC/'+master.split('.')[0]+'_common.h"')
    code('')
    if not (('generate_chunk' in name) or ('calc_dt_kernel_print' in name)):
      code('#define OPS_GPU')
      code('')
    for n in range (0, nargs):
      if arg_typ[n] == 'ops_arg_dat':
        code('extern int xdim'+str(n)+'_'+name+';')
        code('int xdim'+str(n)+'_'+name+'_h;')
        if NDIM==3:
          code('extern int ydim'+str(n)+'_'+name+';')
          code('int ydim'+str(n)+'_'+name+'_h;')
    code('')

    code('#ifdef __cplusplus')
    code('extern "C" {')
    code('#endif')
    code('void '+name+'_c_wrapper(')
    depth = depth+2
    for n in range (0, nargs):
      if arg_typ[n] == 'ops_arg_gbl' and accs[n] == OPS_READ and dims[n].isdigit() and int(dims[n])==1:
        code(typs[n]+' p_a'+str(n)+',')
      else:
        code(typs[n]+' *p_a'+str(n)+',')
    if arg_idx:
      if NDIM == 2:
        code('int arg_idx0, int arg_idx1,')
      elif NDIM == 3:
        code('int arg_idx0, int arg_idx1, int arg_idx2,')
    if NDIM == 2:
      code('int x_size, int y_size);')
    if NDIM == 3:
      code('int x_size, int y_size, int z_size);')
    depth = depth-2
    code('')
    code('#ifdef __cplusplus')
    code('}')
    code('#endif')
    code('')
    comm(' host stub function')

    code('void ops_par_loop_'+name+'(char const *name, ops_block Block, int dim, int* range,')
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
    depth = 2

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
    code('#else //OPS_MPI')
    FOR('n','0',str(NDIM))
    code('start[n] = range[2*n];end[n] = range[2*n+1];')
    ENDFOR()
    code('#endif //OPS_MPI')

    code('')
    code('int x_size = MAX(0,end[0]-start[0]);')
    code('int y_size = MAX(0,end[1]-start[1]);')
    if NDIM==3:
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
        code('xdim'+str(n)+'_'+name+' = args['+str(n)+'].dat->size[0]*args['+str(n)+'].dat->dim;')
        code('xdim'+str(n)+'_'+name+'_h = xdim'+str(n)+';')
        if NDIM==3:
          code('ydim'+str(n)+'_'+name+' = args['+str(n)+'].dat->size[1];')
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
              code('#else //OPS_MPI')
              code(typs[n]+' *arg'+str(n)+'h = ('+typs[n]+' *)(((ops_reduction)args['+str(n)+'].data)->data);')
              code('#endif //OPS_MPI')
    if GBL_READ == True and GBL_READ_MDIM == True:
      comm('Upload large globals')
      code('int consts_bytes = 0;')
    for n in range (0, nargs):
      if arg_typ[n] == 'ops_arg_gbl':
        if accs[n] == OPS_READ and (not dims[n].isdigit() or int(dims[n])>1):
          code('consts_bytes += ROUND_UP('+str(dims[n])+'*sizeof('+typs[n]+'));')
    if GBL_READ == True and GBL_READ_MDIM == True:
      code('reallocConstArrays(consts_bytes);')
    for n in range (0, nargs):
      if arg_typ[n] == 'ops_arg_gbl':
        if accs[n] == OPS_READ and (not dims[n].isdigit() or int(dims[n])>1):
          code('consts_bytes = 0;')
          code('args['+str(n)+'].data = OPS_consts_h + consts_bytes;')
          code('args['+str(n)+'].data_d = OPS_consts_d + consts_bytes;')
          code('for (int d=0; d<'+str(dims[n])+'; d++) (('+typs[n]+' *)args['+str(n)+'].data)[d] = arg'+str(n)+'h[d];')
          code('consts_bytes += ROUND_UP('+str(dims[n])+'*sizeof(int));')
    if GBL_READ == True and GBL_READ_MDIM == True:
      code('mvConstArraysToDevice(consts_bytes);')


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
        code('int base'+str(n)+' = dat'+str(n)+' * 1 * ')
        code('  (start[0] * args['+str(n)+'].stencil->stride[0] - args['+str(n)+'].dat->base[0] - d_m[0]);')
        for d in range (1, NDIM):
          line = 'base'+str(n)+' = base'+str(n)+'+ dat'+str(n)+' *\n'
          for d2 in range (0,d):
            line = line + depth*' '+'  args['+str(n)+'].dat->size['+str(d2)+'] *\n'
          code(line[:-1])
          code('  (start['+str(d)+'] * args['+str(n)+'].stencil->stride['+str(d)+'] - args['+str(n)+'].dat->base['+str(d)+'] - d_m['+str(d)+']);')

        code('#ifdef OPS_GPU')
        code(typs[n]+' *p_a'+str(n)+' = ('+typs[n]+' *)((char *)args['+str(n)+'].data_d + base'+str(n)+');')
        code('#else')
        code(typs[n]+' *p_a'+str(n)+' = ('+typs[n]+' *)((char *)args['+str(n)+'].data + base'+str(n)+');')
        #code('char *p_a'+str(n)+' = (char *)args['+str(n)+'].data + base'+str(n)+';')
        code('#endif')
        code('')
      elif arg_typ[n] == 'ops_arg_gbl':
        if accs[n] == OPS_READ:
          if dims[n].isdigit() and int(dims[n])==1:
            code(typs[n]+' *p_a'+str(n)+' = ('+typs[n]+' *)args['+str(n)+'].data;')
          else:
            code('#ifdef OPS_GPU')
            code(typs[n]+' *p_a'+str(n)+' = ('+typs[n]+' *)args['+str(n)+'].data_d;')
            code('#else')
            code(typs[n]+' *p_a'+str(n)+' = arg'+str(n)+'h;')
            code('#endif')
        else:
          code(typs[n]+' *p_a'+str(n)+' = arg'+str(n)+'h;')
      else:
        code(typs[n]+' *p_a'+str(n)+' = NULL;')
        code('')

    code('')
    code('#ifdef OPS_GPU')
    code('ops_H_D_exchanges_device(args, '+str(nargs)+');')
    code('#else')
    code('ops_H_D_exchanges_host(args, '+str(nargs)+');')
    code('#endif')
    code('ops_halo_exchanges(args,'+str(nargs)+',range);')
    code('')
    code('ops_timers_core(&c1,&t1);')
    code('OPS_kernels['+str(nk)+'].mpi_time += t1-t2;')
    code('')


    code(name+'_c_wrapper(')
    depth = depth+2
    for n in range (0, nargs):
      if arg_typ[n] == 'ops_arg_gbl' and accs[n] == OPS_READ and dims[n].isdigit() and int(dims[n])==1:
        code('*p_a'+str(n)+',')
      else:
        code('p_a'+str(n)+',')
    if arg_idx:
      if NDIM==2:
        code('arg_idx[0], arg_idx[1],')
      elif NDIM==3:
        code('arg_idx[0], arg_idx[1], arg_idx[2],')
    if NDIM == 2:
      code('x_size, y_size);')
    if NDIM == 3:
      code('x_size, y_size, z_size);')

    depth = depth-2





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

    code('#ifdef OPS_GPU')
    code('ops_set_dirtybit_device(args, '+str(nargs)+');')
    code('#else')
    code('ops_set_dirtybit_host(args, '+str(nargs)+');')
    code('#endif')
    for n in range (0, nargs):
      if arg_typ[n] == 'ops_arg_dat' and (accs[n] == OPS_WRITE or accs[n] == OPS_RW or accs[n] == OPS_INC):
        code('ops_set_halo_dirtybit3(&args['+str(n)+'],range);')

    code('')
    comm('Update kernel record')
    for n in range (0, nargs):
      if arg_typ[n] == 'ops_arg_dat':
        code('OPS_kernels['+str(nk)+'].transfer += ops_compute_transfer(dim, range, &arg'+str(n)+');')
    depth = depth - 2
    code('}')


##########################################################################
#  output individual kernel file
##########################################################################
    if not os.path.exists('./OpenACC'):
      os.makedirs('./OpenACC')
    fid = open('./OpenACC/'+name+'_openacc_kernel.cpp','w')
    date = datetime.datetime.now()
    fid.write('//\n// auto-generated by ops.py\n//\n')
    fid.write(file_text)
    fid.close()

# end of main kernel call loop

##########################################################################
#  output one master kernel file
##########################################################################

  file_text =''
  comm('header')
  code('#ifdef __cplusplus')
  code('#include "ops_lib_cpp.h"')
  code('#endif')
  code('#ifdef OPS_MPI')
  code('#include "ops_mpi_core.h"')
  code('#endif')
  code('#include "ops_cuda_rt_support.h"')
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

  fid = open('./OpenACC/'+master.split('.')[0]+'_common.h','w')
  fid.write('//\n// auto-generated by ops.py\n//\n')
  fid.write(file_text)
  fid.close()
  file_text =''
  code('#include "./OpenACC/'+master.split('.')[0]+'_common.h"')
  code('')
  code('')
  code('void ops_decl_const_char(int dim, char const *type,')
  code('int size, char *dat, char const *name){')
  depth = depth + 2

  for nc in range(0,len(consts)):
    IF('!strcmp(name,"'+(str(consts[nc]['name']).replace('"','')).strip()+'")')
    if consts[nc]['dim'].isdigit() and int(consts[nc]['dim'])==1:
      code((str(consts[nc]['name']).replace('"','')).strip()+' = *('+consts[nc]['type']+'*)dat;')
    else:
      code((str(consts[nc]['name']).replace('"','')).strip()+' = ('+consts[nc]['type']+'*)dat;')
    ENDIF()
    code('else')

  code('{')
  depth = depth + 2
  code('printf("error: unknown const name\\n"); exit(1);')
  ENDIF()

  depth = depth - 2
  code('}')
  code('')
  #code('')
  #comm('user kernel files')

  #kernel_name_list = ['generate_chunk_kernel']

  #for nk in range(0,len(kernels)):
  #  if kernels[nk]['name'] not in kernel_name_list :
  #    code('#include "'+kernels[nk]['name']+'_openacc_kernel.cpp"')
  #    kernel_name_list.append(kernels[nk]['name'])

  fid = open('./OpenACC/'+master.split('.')[0]+'_kernels.cpp','w')
  fid.write('//\n// auto-generated by ops.py//\n\n')
  fid.write(file_text)
  fid.close()
