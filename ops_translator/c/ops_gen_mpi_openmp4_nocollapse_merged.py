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

def ops_gen_mpi_openmp4(master, date, consts, kernels):

  OPS_ID   = 1;  OPS_GBL   = 2;

  OPS_READ = 1;  OPS_WRITE = 2;  OPS_RW  = 3;
  OPS_INC  = 4;  OPS_MAX   = 5;  OPS_MIN = 6;

  accsstring = ['OPS_READ','OPS_WRITE','OPS_RW','OPS_INC','OPS_MAX','OPS_MIN' ]

  NDIM = 2 #the dimension of the application is hardcoded here .. need to get this dynamically


##########################################################################
#  create new kernel file
##########################################################################
  compiler = os.environ.get("OPS_COMPILER")
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

    code('#include "./OpenMP4/'+master.split('.')[0]+'_common.h"')
    code('#include <omp.h>')
#    code('#define module(A,B) A%B')
#    code('#define div(A,B) A/B')
    if not (('calc_dt_kernel_print' in name)):
      if not (NDIM==3 and 'field_summary' in name):
        code('#define OPS_GPU')
        code('')
    for n in range (0, nargs):
      if arg_typ[n] == 'ops_arg_dat':
        code('extern int xdim'+str(n)+'_'+name+';')
        if NDIM==3:
          code('extern int ydim'+str(n)+'_'+name+';')
#        code('#pragma acc declare create(xdim'+str(n)+'_'+name+')')
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

    #code('#define OPS_ACC_MACROS')
    for n in range (0, nargs):
      if arg_typ[n] == 'ops_arg_dat':
        if int(dims[n]) == 1:
          if NDIM==1:
            code('#define OPS_ACC'+str(n)+'(x) (x)')
          if NDIM==2:
            #code('#define OPS_ACC'+str(n)+'(x,y) (x+xdim'+str(n)+'_'+name+'*(y))')
            code('#define OPS_ACC'+str(n)+'(x,y) ( n_x*'+str(stride[NDIM*n])+'*'+str(dims[n])+ '+n_y*xdim'+str(n)+'_'+name+'*'+str(stride[NDIM*n+1])+'*'+str(dims[n])+'+x+xdim'+str(n)+'_'+name+'*(y))')
          if NDIM==3:
            #code('#define OPS_ACC'+str(n)+'(x,y,z) (x+xdim'+str(n)+'_'+name+'*(y)+xdim'+str(n)+'_'+name+'*ydim'+str(n)+'_'+name+'*(z))')
	    code('#define OPS_ACC'+str(n)+'(x,y,z) ( n_x*'+str(stride[NDIM*n])+'*'+str(dims[n])+'+ n_y*xdim'+str(n)+'_'+name+'*'+str(stride[NDIM*n+1])+'*'+str(dims[n])+'+n_z*xdim'+str(n)+'_'+name+'*ydim'+str(n)+'_'+name+'*'+str(stride[NDIM*n+2]) +'+ x+xdim'+str(n)+'_'+name+'*(y)+xdim'+str(n)+'_'+name+'*ydim'+str(n)+'_'+name+'*(z))')
    
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

    i = text[0:i].rfind('\n') #reverse find
    j = text[i:].find('{')
    k = para_parse(text, i+j, '{', '}')
    m = text.find(name)
    l = text[i:m].find('inline')
    signature_text = text[i:i+j]
    lr = signature_text[0:].find('(')
    mt = para_parse(signature_text, 0, '(', ')')
    signature_text = signature_text[lr+1:mt]
    body_text = text[i+j+1:k]

    #if(l<0):
    #  code('inline'+text[i:k+2])
    #else:
    #  code(text[i:k+2])
    #print(text[i:k+2])
    #print(signature_text)
    #print(k)
    #print(j)
    kernel_params = [ var.strip() for var in signature_text.split(',')]
    #print(kernel_params)
    #print(body_text)
    code('')

    #for n in range (0, nargs):
    #  if arg_typ[n] == 'ops_arg_dat':
    #    if int(dims[n]) == 1:
    #      code('#undef OPS_ACC'+str(n))
    #code('')
    #for n in range (0, nargs):
    #  if arg_typ[n] == 'ops_arg_dat':
    #    if int(dims[n]) > 1:
    #      code('#undef OPS_ACC_MD'+str(n))
    #code('')
    #code('')

##########################################################################
#  generate C wrapper
##########################################################################
    code('void '+name+'_c_wrapper(')
    config.depth = config.depth+2
    for n in range (0, nargs):
      if arg_typ[n] == 'ops_arg_gbl' and accs[n] == OPS_READ and dims[n].isdigit() and int(dims[n])==1:
        code(typs[n]+' p_a'+str(n)+',')
	#code('int tot'+ str(n)+',')
      else:
   	code(typs[n]+' *p_a'+str(n)+',')
        if arg_typ[n] == 'ops_arg_gbl' and accs[n] == OPS_READ:
          code('int tot'+ str(n)+',')
        if compiler == "xl":
          if arg_typ[n] == 'ops_arg_dat':
	    #code('int base'+ str(n)+',')
            code('int tot'+ str(n)+','+'int base'+ str(n)+',')

    if arg_idx:
      if NDIM == 1:
        code('int arg_idx0,')
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

  #  for n in range (0,nargs):
   #   code('int tot'+ +str(n) + '= ')

    #code("int num_blocks = round(((double)x_size*(double)y_size)/OPS_block_size_x);");
    #code("int num_blocks = OPS_threads;");
    #code('printf("x_size=%d - y_size=%d \\n", x_size, y_size);')
    for n in range (0,nargs):
      if arg_typ[n] == 'ops_arg_gbl':
        if accs[n] <> OPS_READ:
          #if dims[n].isdigit() and int(dims[n]) == 1:
          #  code(typs[n]+' p_a'+str(n)+'_l = *p_a'+str(n)+';')
          #else:
          for d in range(0,int(dims[n])):
            code(typs[n]+' p_a'+str(n)+'_'+str(d)+' = p_a'+str(n)+'['+str(d)+'];')

    #line = '#pragma omp target enter data map(to:'
    #for n in range (0,nargs):
      #if arg_typ[n] == 'ops_arg_dat':
      #  line = line + 'p_a'+str(n)+'[0:tot'+str(n)+'],'
    #  if arg_typ[n] == 'ops_arg_gbl' :
    #    if accs[n] == OPS_READ and (not dims[n].isdigit() or int(dims[n])>1):
    #      line = line + 'p_a'+str(n)+'[0:tot'+str(n)+'],'

    #for nc in range (0,len(consts)):
    #  if re.search('[a-zA-Z]', consts[nc]['dim']) or  (int(consts[nc]['dim']) != 1) :
    #    num = str(consts[nc]['dim'])
    #    line = line + str(consts[nc]['name']).replace('"','')+'[0:'+num+'],'
    #line = line[:-1]+')'
    #code(line)
    
    #code('int num_team=(y_size*x_size)/OPS_block_size_x;\n')
    
    #line = '#pragma omp target teams \n'
    #line = '#pragma omp target teams distribute parallel for num_teams(OPS_threads)  thread_limit(OPS_threads_for_block)'
    #line = '#pragma omp target teams distribute parallel for num_teams(y_size)  thread_limit(x_size) schedule(static,1)'
    
    #line = '#pragma omp target teams map(to:'
    
    #line = '#pragma omp target teams distribute parallel for '
    #line = '#pragma omp target teams map(to:'
    #line = '#pragma omp target teams map(to:'
    
    if compiler == "clang":
     line = "#pragma omp target teams distribute parallel for "
     for n in range (0,nargs):
      if arg_typ[n] == 'ops_arg_gbl' :
        if accs[n] <> OPS_READ:
          line = line + ' map(tofrom :'
          line = line + ' p_a'+str(n)+'_'+str(d)+','
          line = line[:-1]+')'
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
         if accs[n] == OPS_WRITE: #this may not be correct
           for d in range(0,int(dims[n])):
             line = line + ' reduction(+:p_a'+str(n)+'_'+str(d)+')'



 
    if compiler == "xl":
     line = '#pragma omp target teams map(to:'
     for n in range (0,nargs):
       if arg_typ[n] == 'ops_arg_dat':
         line = line + 'p_a'+str(n)+'[0:tot'+str(n)+'],'
       if arg_typ[n] == 'ops_arg_gbl' :
         if accs[n] == OPS_READ and (not dims[n].isdigit() or int(dims[n])>1):
           line = line + 'p_a'+str(n)+'[0:tot'+str(n)+'],'
     line = line[:-1]+')'
     for n in range (0,nargs):
       if arg_typ[n] == 'ops_arg_gbl' :
         if accs[n] <> OPS_READ:
           line = line + ' map(tofrom :'
           line = line + ' p_a'+str(n)+'_'+str(d)+','
           line = line[:-1]+')'
     line = line + 'map(to:'
     for nc in range (0,len(consts)):
       if re.search('[a-zA-Z]', consts[nc]['dim']) or  (int(consts[nc]['dim']) != 1) : 
         num = str(consts[nc]['dim'])
         line = line + str(consts[nc]['name']).replace('"','')+'[0:'+num+'],'
     line = line[:-1]+')'
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
        if accs[n] == OPS_WRITE: #this may not be correct
          for d in range(0,int(dims[n])):
            line = line + ' reduction(+:p_a'+str(n)+'_'+str(d)+')'
     line = line + "\n#pragma omp distribute parallel for schedule(static, 1) "

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
         if accs[n] == OPS_WRITE: #this may not be correct
           for d in range(0,int(dims[n])):
             line = line + ' reduction(+:p_a'+str(n)+'_'+str(d)+')'


    #for n in range (0,nargs):
    #  if arg_typ[n] == 'ops_arg_dat':
    #    line = line + 'p_a'+str(n)+'[0:tot'+str(n)+'],'
    #  if arg_typ[n] == 'ops_arg_gbl' :
    #    if accs[n] == OPS_READ and (not dims[n].isdigit() or int(dims[n])>1):
    #      line = line + 'p_a'+str(n)+'[0:tot'+str(n)+'],'
    #line = line[:-1]+')'
    #line = line + 'map(to:'
    #for nc in range (0,len(consts)):
    #  if re.search('[a-zA-Z]', consts[nc]['dim']) or  (int(consts[nc]['dim']) != 1) :
    #   num = str(consts[nc]['dim'])
    #   line = line + str(consts[nc]['name']).replace('"','')+'[0:'+num+'],'
    #line = line[:-1]+')'

    #for n in range (0,nargs):
    #  if arg_typ[n] == 'ops_arg_gbl':
    #    if accs[n] == OPS_MIN:
    #      for d in range(0,int(dims[n])):
    #        line = line + ' reduction(min:p_a'+str(n)+'_'+str(d)+')'
    #    if accs[n] == OPS_MAX:
    #      for d in range(0,int(dims[n])):
    #        line = line + ' reduction(max:p_a'+str(n)+'_'+str(d)+')'
    #    if accs[n] == OPS_INC:
    #      for d in range(0,int(dims[n])):
    #        line = line + ' reduction(+:p_a'+str(n)+'_'+str(d)+')'
    #    if accs[n] == OPS_WRITE: #this may not be correct ..
    #      for d in range(0,int(dims[n])):
    #        line = line + ' reduction(+:p_a'+str(n)+'_'+str(d)+')'
    code('#ifdef OPS_GPU')
    

    #line = line + '\n #pragma omp target \n'
    #line = line + '\n#pragma omp teams num_teams(num_blocks)  thread_limit(OPS_threads_for_block)'
   
    #for n in range (0,nargs):
    #  if arg_typ[n] == 'ops_arg_gbl':
    #    if accs[n] == OPS_MIN:
    #      for d in range(0,int(dims[n])):
    #        line = line + ' reduction(min:p_a'+str(n)+'_'+str(d)+')'
    #    if accs[n] == OPS_MAX:
    #      for d in range(0,int(dims[n])):
    #        line = line + ' reduction(max:p_a'+str(n)+'_'+str(d)+')'
    #    if accs[n] == OPS_INC:
    #      for d in range(0,int(dims[n])):
    #        line = line + ' reduction(+:p_a'+str(n)+'_'+str(d)+')'
    #    if accs[n] == OPS_WRITE: #this may not be correct
    #      for d in range(0,int(dims[n])):
    #        line = line + ' reduction(+:p_a'+str(n)+'_'+str(d)+')'
   
    #code('\n')
    #code(line)
    #line = line + '\n'
    #line = line + '#pragma omp distribute parallel for simd collapse('+str(NDIM)+') schedule(static,1)'
    #line = line + '#pragma omp distribute parallel for simd schedule(static,1)'
    #code(line)

    #line = "\n#pragma omp target teams distribute parallel for "

    code('\n')
    code(line)
    code('#endif')
    if NDIM==3:
      #FOR('n_z','0','z_size')
      #code('#ifdef OPS_GPU')
      #code(line)
      #code('#endif')
      #FOR('n_y','0','y_size')
      #code('#ifdef OPS_GPU')
      #line = '#pragma omp target \n'
      #line = '#pragma omp teams \n'
      #line = line + '#pragma omp distribute parallel for simd \n'
      #code(line)
      #code('#endif')
      FOR('i','0','y_size*x_size*z_size')
    if NDIM==2:
      #FOR('n_y','0','y_size')
      FOR('i','0','y_size*x_size')
      #code('#ifdef OPS_GPU')
      #line = '#pragma omp target \n'
      #line = '#pragma omp teams \n'
      #line = line + '#pragma omp distribute parallel for simd \n'
      #code(line)
      #line = '\n #pragma omp target \n'
      #line =  '#pragma omp teams \n'
      #line = '#pragma omp parallel for '
      #for n in range (0,nargs):
      # if arg_typ[n] == 'ops_arg_gbl':
      #    if accs[n] == OPS_MIN:
      #      for d in range(0,int(dims[n])):
      #        line = line + ' reduction(min:p_a'+str(n)+'_'+str(d)+')'
      #    if accs[n] == OPS_MAX:
      #      for d in range(0,int(dims[n])):
      #        line = line + ' reduction(max:p_a'+str(n)+'_'+str(d)+')'
      #    if accs[n] == OPS_INC:
      #      for d in range(0,int(dims[n])):
      #        line = line + ' reduction(+:p_a'+str(n)+'_'+str(d)+')'
      #    if accs[n] == OPS_WRITE: #this may not be correct
      #      for d in range(0,int(dims[n])):
      #        line = line + ' reduction(+:p_a'+str(n)+'_'+str(d)+')'
      #code(line)
      #code('#endif')

    #FOR('n_x','0','x_size')

    
    #if arg_idx:
    #  if NDIM==1:
    #    code('int arg_idx[] = {arg_idx0+n_x};')
    #  elif NDIM==2:
    #    code('int arg_idx[] = {arg_idx0+i%x_size, arg_idx1+i/x_size};')
    #  elif NDIM==3:
    #    code('int arg_idx[] = {arg_idx0+i%x_size, arg_idx1+ (i / x_size) % y_size, arg_idx2+i/(x_size*y_size)};')

    for n in range (0,nargs):
      if arg_typ[n] == 'ops_arg_gbl':
        if accs[n] == OPS_MIN and int(dims[n])>1:
          code(typs[n]+' p_a'+str(n)+'_local['+str(dims[n])+'];')
          for d in range(0,int(dims[n])):
            code('p_a'+str(n)+'_local['+str(d)+'] = p_a'+str(n)+'['+str(d)+'];') #need +INFINITY_ change to
        if accs[n] == OPS_MAX and int(dims[n])>1:
          code(typs[n]+' p_a'+str(n)+'_local['+str(dims[n])+'];')
          for d in range(0,int(dims[n])):
            code('p_a'+str(n)+'_local['+str(d)+'] = p_a'+str(n)+'['+str(d)+'];') #need -INFINITY_ change to
        if accs[n] == OPS_INC and int(dims[n])>1:
          code(typs[n]+' p_a'+str(n)+'_local['+str(dims[n])+'];')
          for d in range(0,int(dims[n])):
            code('p_a'+str(n)+'_local['+str(d)+'] = ZERO_'+typs[n]+';')
        if accs[n] == OPS_WRITE and int(dims[n])>1: #this may not be correct
          code(typs[n]+' p_a'+str(n)+'_local['+str(dims[n])+'];')
          for d in range(0,int(dims[n])):
            code('p_a'+str(n)+'_local['+str(d)+'] = ZERO_'+typs[n]+';')
    #code('')


    #text = name+'( '
    text = ''

    #text = text +' int n_x= module(i,x_size);'
    #text = text+'const int blocksize =  omp_get_num_threads();\n'
    if compiler == "clang":
      text = text+'const int id = omp_get_num_threads() * omp_get_team_num() + omp_get_thread_num();\n'
    #text = text+' int n_y = 0;\n'
    #text = text+'int n_x = id % x_size;\n'
    #text = text+'while(id >= x_size){ id -= x_size; ++n_y;}\n'
    #text = text+'int n_x = id;\n'

    #text = text+'if(id < x_size*y_size) {'
    #text = text+'if( omp_get_team_num() < y_size && omp_get_thread_num()< x_size ) {'
    if compiler == "clang":
      text = text +'const int n_x= id%x_size;'
    if compiler == "xl":
      text = text +'const int n_x= i%x_size;'
    #text = text +'const int n_x= i%x_size;'
    #text = text +' int n_x= omp_get_thread_num();'
    #text = text + 'int n_x = i%x_size;'
    if NDIM == 2:
      if compiler == "clang":
        text = text +'const int n_x= id%x_size;'
      if compiler == "xl":
        text = text +'const int n_y= i/x_size;'
      #text = text +'const int n_y= i/x_size;'
      #text = text +' int n_y= omp_get_team_num();'
      #text = text +' int n_y= i/x_size;'# //omp_get_thread_num();'
    if NDIM == 3:
      if compiler == "clang":
        text = text +'const int n_y= (id/x_size)%y_size;'
        text = text +'const int n_z= id/(x_size*y_size);'
      if compiler == "xl":
        text = text +'const int n_y= (i/x_size)%y_size;'
        text = text +'const int n_z= i/(x_size*y_size);'
    if arg_idx:
      if NDIM==1:
        text = text +'int arg_idx[] = {arg_idx0+n_x};'
      elif NDIM==2:
        text = text +'int arg_idx[] = {arg_idx0+n_x, arg_idx1+n_y};'
      elif NDIM==3:
        text = text +'int arg_idx[] = {arg_idx0+n_x, arg_idx1+n_y, arg_idx2+n_z};'

    for n in range (0, nargs):
      text = text + kernel_params[n] + ' = '
      if arg_typ[n] == 'ops_arg_dat':
        if NDIM == 1:
          #text = text +' p_a'+str(n)+'+ base'+str(n)+' + n_x*'+str(stride[NDIM*n])+'*'+str(dims[n])
          text = text +' p_a'+str(n)+'+ n_x*'+str(stride[NDIM*n])+'*'+str(dims[n])
        elif NDIM == 2:
          #text = text +' p_a'+str(n)+'+ base'+str(n)+' + n_x*'+str(stride[NDIM*n])+'*'+str(dims[n])+\
          #text = text +' p_a'+str(n)+'+i'
          if compiler == "xl":
            text = text +' p_a'+str(n) +'+ base'+str(n)#+'+ n_x*'+str(stride[NDIM*n])+'*'+str(dims[n])+\
          if compiler == "clang":
            text = text +' p_a'+str(n)
          #' + n_y*xdim'+str(n)+'_'+name+'*'+str(stride[NDIM*n+1])+'*'+str(dims[n])
        elif NDIM == 3:
          #text = text +' p_a'+str(n)+'+ base'+str(n)+' + n_x*'+str(stride[NDIM*n])+'*'+str(dims[n])+' + n_y*xdim'+str(n)+'_'+name+'*'+str(stride[NDIM*n+1])+'*'+str(dims[n])
          if compiler == "xl":
            text = text +' p_a'+str(n) +'+ base'+str(n)#+'+ n_x*'+str(stride[NDIM*n])+'*'+str(dims[n])+\
          if compiler == "clang":
            text = text +' p_a'+str(n)
          #text = text +' p_a'+str(n) #+ 'n_x*'+str(stride[NDIM*n])+'*'+str(dims[n])+' + n_y*xdim'+str(n)+'_'+name+'*'+str(stride[NDIM*n+1])+'*'+str(dims[n])
          #text = text + ' + n_z*xdim'+str(n)+'_'+name+'*ydim'+str(n)+'_'+name+'*'+str(stride[NDIM*n+2])
      elif arg_typ[n] == 'ops_arg_gbl':
        if accs[n] == OPS_READ:
          if dims[n].isdigit() and int(dims[n])==1:
            text = text +' &p_a'+str(n)+''
          else:
            text = text +' p_a'+str(n)+''
        else:
          if dims[n].isdigit() and int(dims[n]) == 1:
            text = text +' &p_a'+str(n)+'_0'
          else:
            text = text +' p_a'+str(n)+'_local'
      elif arg_typ[n] == 'ops_arg_idx':
        text = text +'arg_idx'

      if nargs <> 1 and n != nargs-1:
        text = text + '; \n'
      else:
        text = text +' ;\n'
      if n%n_per_line == 0 and n <> nargs-1:
        text = text +'\n          '
    code(text)
    print(text)
    code(body_text)
    for n in range (0,nargs):
      if arg_typ[n] == 'ops_arg_gbl':
        if accs[n] == OPS_MIN and int(dims[n])>1:
          for d in range(0,int(dims[n])):
            code('p_a'+str(n)+'_'+str(d)+' = MIN(p_a'+str(n)+'_'+str(d)+',p_a'+str(n)+'_local['+str(d)+']);')
        if accs[n] == OPS_MAX and int(dims[n])>1:
          for d in range(0,int(dims[n])):
            code('p_a'+str(n)+'_'+str(d)+' = MAX(p_a'+str(n)+'_'+str(d)+'p_a'+str(n)+'_local['+str(d)+']);')
        if accs[n] == OPS_INC and int(dims[n])>1:
          for d in range(0,int(dims[n])):
            code('p_a'+str(n)+'_'+str(d)+' +=p_a'+str(n)+'_local['+str(d)+'];')
        if accs[n] == OPS_WRITE and int(dims[n])>1: #this may not be correct
          for d in range(0,int(dims[n])):
            code('p_a'+str(n)+'_'+str(d)+' +=p_a'+str(n)+'_local['+str(d)+'];')

    #code('}')
    #ENDFOR()
    if NDIM==2:
      ENDFOR()
    if NDIM==3:
     # ENDFOR()
     ENDFOR()

    for n in range (0, nargs):
      if arg_typ[n] == 'ops_arg_gbl':
        if accs[n] <> OPS_READ:
          #if dims[n].isdigit() and int(dims[n]) == 1:
          #  code('*p_a'+str(n)+' = p_a'+str(n)+'_l;')
          #else:
          for d in range(0,int(dims[n])):
            code('p_a'+str(n)+'['+str(d)+'] = p_a'+str(n)+'_'+str(d)+';')

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
    code('')
    code('')

##########################################################################
#  output individual kernel file
##########################################################################
    if not os.path.exists('./OpenMP4'):
      os.makedirs('./OpenMP4')
    fid = open('./OpenMP4/'+name+'_openmp4_kernel_c.c','w')
    date = datetime.datetime.now()
    fid.write('//\n// auto-generated by ops.py\n//\n')
    #print(config.file_text)
    fid.write(config.file_text)
    fid.close()
    config.file_text = ''
    #exit()
##########################################################################
#  now host stub
##########################################################################
    #code('extern "C" {')
    code('#include "./OpenMP4/'+master.split('.')[0]+'_common.h"')
    #code('extern "C" {')
    code('#include "./OpenMP4/'+name+'_openmp4_kernel_c.c"')
    #code('}')
    code('')
    if not (('calc_dt_kernel_print' in name)):
      if not (NDIM==3 and 'field_summary' in name):
        code('#define OPS_GPU')
        code('')
    for n in range (0, nargs):
      if arg_typ[n] == 'ops_arg_dat':
        code('int xdim'+str(n)+'_'+name+';')
        code('int xdim'+str(n)+'_'+name+'_h = -1;')
        if NDIM==3:
          code('int ydim'+str(n)+'_'+name+';')
          code('int ydim'+str(n)+'_'+name+'_h = -1;')
    code('')

    #code('#ifdef __cplusplus')
    #code('extern "C" {')
    #code('#endif')
    #code('void '+name+'_c_wrapper(')
    #config.depth = config.depth+2
    #for n in range (0, nargs):
    #  if arg_typ[n] == 'ops_arg_gbl' and accs[n] == OPS_READ and dims[n].isdigit() and int(dims[n])==1:
    #    code(typs[n]+' p_a'+str(n)+',')
    #  else:
    #    code(typs[n]+' *p_a'+str(n)+',')
    #if arg_idx:
    #  if NDIM == 1:
    #    code('int arg_idx0,')
    #  elif NDIM == 2:
    #    code('int arg_idx0, int arg_idx1,')
    #  elif NDIM == 3:
    #    code('int arg_idx0, int arg_idx1, int arg_idx2,')

    #if NDIM == 1:
    #  code('int x_size);')
    #elif NDIM == 2:
    #  code('int x_size, int y_size);')
    #elif NDIM == 3:
    #  code('int x_size, int y_size, int z_size);')
    #config.depth = config.depth-2
    #code('')
    #code('#ifdef __cplusplus')
    #code('}')
    #code('#endif')
    #code('')
    #comm(' host stub function')

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
    comm('Timing')
    code('double t1,t2,c1,c2;')

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

    IF('OPS_diags > 1')
    code('ops_timing_realloc('+str(nk)+',"'+name+'");')
    code('OPS_kernels['+str(nk)+'].count++;')
    code('ops_timers_core(&c1,&t1);')
    ENDIF()

    code('')
    comm('compute localy allocated range for the sub-block')
    comm('')
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

    code('')
    for n in range (0, nargs):
      if arg_typ[n] == 'ops_arg_dat':
        code('int tot'+str(n)+' = 1;')
        code('for (int i = 0; i < args['+str(n)+'].dat->block->dims; i++)')
        code('  tot'+str(n)+ '= tot'+str(n)+ ' * args['+str(n)+'].dat->size[i];')

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
    if GBL_READ == True and GBL_READ_MDIM == True:
      comm('Upload large globals')
      code('#ifdef OPS_GPU')
      code('int consts_bytes = 0;')
      for n in range (0, nargs):
        if arg_typ[n] == 'ops_arg_gbl':
          if accs[n] == OPS_READ and (not dims[n].isdigit() or int(dims[n])>1):
            code('consts_bytes += ROUND_UP('+str(dims[n])+'*sizeof('+typs[n]+'));')
      #code('OPS_gbl_changed = 0;')
      #code('if (OPS_gbl_prev != NULL)')
      #code('for (int i = 0; i < consts_bytes; i++) {')
      #code('if (OPS_consts_h[i] != OPS_gbl_prev[i])')
      #code('OPS_gbl_changed = 1;')
      #code('}')
      #code('else {')

      #code('OPS_gbl_changed = 0;')
      #code('OPS_gbl_prev = (char *)malloc(consts_bytes);')
      #if GBL_READ == True and GBL_READ_MDIM == True:
      #  code('int OPS_consts_bytes = 4 * consts_bytes;');
      #  code('if(OPS_consts_h == NULL)')
      #  code('OPS_consts_h = (char *)malloc(OPS_consts_bytes);');
      #  code('memset(OPS_consts_h, 0 , OPS_consts_bytes);');
        #code('reallocConstArrays(consts_bytes);')
      if GBL_READ == True and GBL_READ_MDIM == True:
        code('int OPS_consts_bytes = 4 * consts_bytes;');
        code('if(OPS_consts_h == NULL){')
        code('OPS_consts_h = (char *)malloc(OPS_consts_bytes);');
        #code('#pragma omp target enter data map(to: OPS_consts_h[0:consts_bytes]);')
        code('memset(OPS_consts_h, 0 , OPS_consts_bytes);}');
      for n in range (0, nargs):
        if arg_typ[n] == 'ops_arg_gbl':
          if accs[n] == OPS_READ and (not dims[n].isdigit() or int(dims[n])>1):
           code('consts_bytes = 0;')
           code('args['+str(n)+'].data = OPS_consts_h + consts_bytes;')
           code('args['+str(n)+'].data_d = OPS_consts_d + consts_bytes;')
           code('for (int d=0; d<'+str(dims[n])+'; d++) (('+typs[n]+' *)args['+str(n)+'].data)[d] = arg'+str(n)+'h[d];')
           code('consts_bytes += ROUND_UP('+str(dims[n])+'*sizeof(int));')
           #code('#pragma omp target enter data map(to: OPS_consts_h[0:consts_bytes]);')
           #code(' memcpy(OPS_gbl_prev, OPS_consts_h, consts_bytes);')
      #code('}')
      #code('if(OPS_gbl_changed) {')
    #if GBL_READ == True and GBL_READ_MDIM == True:
    #  code('int OPS_consts_bytes = 4 * consts_bytes;');
    #  code('if(OPS_consts_h == NULL)')
    #  code('OPS_consts_h = (char *)malloc(OPS_consts_bytes);');
    #  code('memset(OPS_consts_h, 0 , OPS_consts_bytes);');
     #code('reallocConstArrays(consts_bytes);')
    #for n in range (0, nargs):
    #  if arg_typ[n] == 'ops_arg_gbl':
    #    if accs[n] == OPS_READ and (not dims[n].isdigit() or int(dims[n])>1):
    #      code('consts_bytes = 0;')
    #      code('args['+str(n)+'].data = OPS_consts_h + consts_bytes;')
    #      code('args['+str(n)+'].data_d = OPS_consts_d + consts_bytes;')
    #      code('for (int d=0; d<'+str(dims[n])+'; d++) (('+typs[n]+' *)args['+str(n)+'].data)[d] = arg'+str(n)+'h[d];')
    #      code('consts_bytes += ROUND_UP('+str(dims[n])+'*sizeof(int));')
      #code('#endif //OPS_GPU')
    if GBL_READ == True and GBL_READ_MDIM == True:
      code('mvConstArraysToDevice(OPS_consts_bytes);')
      #code('mvConstArraysToDevice(consts_bytes);')
      #code(' memcpy(OPS_gbl_prev, OPS_consts_h, consts_bytes);')
      #code('}else{')
      #code('int OPS_consts_bytes = 4 * consts_bytes;')
      #code('memset(OPS_consts_h, 0, OPS_consts_bytes);')
      #code('int consts_bytes = 0;')
      #code('args['+str(n)+'].data = OPS_consts_h + consts_bytes;')
      #code('for (int d = 0; d < NUM_FIELDS; d++)')
      #code('((int *)args['+str(n)+'].data)[d] = arg'+str(n)+'h[d];')
      #code('consts_bytes += ROUND_UP('+str(dims[n])+'*sizeof(int));')
      #code('#pragma omp target update to( OPS_consts_h[0:consts_bytes]);')
      #code('}')
      code('#endif //OPS_GPU')


    comm('')
    comm('set up initial pointers')
    for n in range (0, nargs):
      if arg_typ[n] == 'ops_arg_dat':
        code('int base'+str(n)+' = args['+str(n)+'].dat->base_offset + args['+str(n)+'].dat->elem_size * start[0] * args['+str(n)+'].stencil->stride[0];')
        for d in range (1, NDIM):
          line = 'base'+str(n)+' = base'+str(n)+' + args['+str(n)+'].dat->elem_size *\n'
          for d2 in range (0,d):
            line = line + config.depth*' '+'  args['+str(n)+'].dat->size['+str(d2)+'] *\n'
          code(line[:-1])
          code('  start['+str(d)+'] * args['+str(n)+'].stencil->stride['+str(d)+'];')

        code('#ifdef OPS_GPU')
        #code(typs[n]+' *p_a'+str(n)+' = ('+typs[n]+' *)((char *)args['+str(n)+'].data_d + base'+str(n)+');')
        if compiler == "xl":
          code(typs[n]+' *p_a'+str(n)+' = ('+typs[n]+' *)((char *)args['+str(n)+'].data_d);')
        if compiler == "clang":
          code(typs[n]+' *p_a'+str(n)+' = ('+typs[n]+' *)((char *)args['+str(n)+'].data_d + base'+str(n)+' );')
        code('#else')
        code(typs[n]+' *p_a'+str(n)+' = ('+typs[n]+' *)((char *)args['+str(n)+'].data);')# + base'+str(n)+');')
        code('#endif')
        code('')
      elif arg_typ[n] == 'ops_arg_gbl':
        if accs[n] == OPS_READ:
          if dims[n].isdigit() and int(dims[n])==1:
            code(typs[n]+' *p_a'+str(n)+' = ('+typs[n]+' *)args['+str(n)+'].data;')
          else:
            code('#ifdef OPS_GPU')
            code(typs[n]+' *p_a'+str(n)+' = ('+typs[n]+' *)args['+str(n)+'].data;')
            code('#else')
            code(typs[n]+' *p_a'+str(n)+' = arg'+str(n)+'h;')
            code('#endif')
        else:
          code(typs[n]+' *p_a'+str(n)+' = arg'+str(n)+'h;')
      else:
        code(typs[n]+' *p_a'+str(n)+' = NULL;')
        code('')
#    for n in range (0, nargs):
#      if arg_typ[n] == 'ops_arg_dat':
#        code('int tot'+str(n)+' = 1;')
#        code('for (int i = 0; i < args['+str(n)+'].dat->block->dims; i++)')
#        code('  tot'+str(n)+ '= tot'+str(n)+ ' * args['+str(n)+'].dat->size[i];')

    code('')
    code('#ifdef OPS_GPU')
    code('for (int n = 0; n < '+str(nargs)+'; n++)')
    code('if (args[n].argtype == OPS_ARG_DAT && args[n].dat->dirty_hd == 1) {')
    code('int size = 1;')
    code('for (int i = 0; i < args[n].dat->block->dims; i++)')
    code('  size = size  * args[n].dat->size[i];')
    #code('  #pragma omp target update to( args[n].dat->data[0:size*args[n].dat->elem_size])')
    code('args[n].dat->dirty_hd = 0;')
    code('}')
    code('//ops_H_D_exchanges_device(args, '+str(nargs)+');')
    code('#else')
    code('for (int n = 0; n < '+str(nargs)+'; n++)')
    code('if (args[n].argtype == OPS_ARG_DAT && args[n].dat->dirty_hd == 2) {')
    code('int size = 1;')
    code('for (int i = 0; i < args[n].dat->block->dims; i++)')
    code('  size = size  * args[n].dat->size[i];')
    #code('#pragma omp target update from(args[n].dat->data[0:size*args[n].dat->elem_size])')
      #//ops_download_dat(args[n].dat);
      #// printf("halo exchanges on host\n");
    code('args[n].dat->dirty_hd = 0;')
    code('}')
    code('//ops_H_D_exchanges_host(args, '+str(nargs)+');')
    code('#endif')
    code('ops_halo_exchanges(args,'+str(nargs)+',range);')
    code('')
    code('#ifdef OPS_GPU')
    code('ops_H_D_exchanges_device(args, '+str(nargs)+');')
    code('#else')
    code('ops_H_D_exchanges_host(args, '+str(nargs)+');')
    code('#endif')

    IF('OPS_diags > 1')
    code('ops_timers_core(&c2,&t2);')
    code('OPS_kernels['+str(nk)+'].mpi_time += t2-t1;')
    ENDIF()
    code('')

    code(name+'_c_wrapper(')
    config.depth = config.depth+2
    for n in range (0, nargs):
      if arg_typ[n] == 'ops_arg_gbl' and accs[n] == OPS_READ and dims[n].isdigit() and int(dims[n])==1:
        code('*p_a'+str(n)+',')
      else:
        code('p_a'+str(n)+',')
        if arg_typ[n] == 'ops_arg_gbl' and accs[n] == OPS_READ:
	  code(str(dims[n])+',')
        if compiler == "xl": 
          if arg_typ[n] == 'ops_arg_dat':
            #code('base'+str(n)+'/args['+str(n)+'].dat->elem_size,')
	    code('tot'+str(n)+','+'base'+str(n)+'/args['+str(n)+'].dat->elem_size,')
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
    IF('OPS_diags > 1')
    code('ops_timers_core(&c1,&t1);')
    code('OPS_kernels['+str(nk)+'].time += t1-t2;')
    ENDIF()

    # if reduction == 1 :
    #   for n in range (0, nargs):
    #     if arg_typ[n] == 'ops_arg_gbl' and accs[n] != OPS_READ:
    #       #code('ops_mpi_reduce(&arg'+str(n)+',('+typs[n]+' *)args['+str(n)+'].data);')
    #   code('ops_timers_core(&c1,&t1);')
    #   code('OPS_kernels['+str(nk)+'].mpi_time += t1-t2;')

    code('#ifdef OPS_GPU')
    code('//for (int n = 0; n < '+str(nargs)+'; n++) {')
    code('//if ((args[n].argtype == OPS_ARG_DAT) &&')
    code('//(args[n].acc == OPS_INC || args[n].acc == OPS_WRITE ||')
    code('//args[n].acc == OPS_RW)) {')
    code('//args[n].dat->dirty_hd = 2;')
    code('//}')
    code('//}')
    code('ops_set_dirtybit_device(args, '+str(nargs)+');')
    code('#else')
    code('//for (int n = 0; n < '+str(nargs)+'; n++) {')
    code('//if ((args[n].argtype == OPS_ARG_DAT) &&')
    code('//(args[n].acc == OPS_INC || args[n].acc == OPS_WRITE ||')
    code('//args[n].acc == OPS_RW)) {')
    code('//args[n].dat->dirty_hd = 1;')
    code('//}')
    code('//}')
    code('ops_set_dirtybit_host(args, '+str(nargs)+');')
    code('#endif')
    for n in range (0, nargs):
      if arg_typ[n] == 'ops_arg_dat' and (accs[n] == OPS_WRITE or accs[n] == OPS_RW or accs[n] == OPS_INC):
        code('ops_set_halo_dirtybit3(&args['+str(n)+'],range);')

    code('')
    IF('OPS_diags > 1')
    comm('Update kernel record')
    code('ops_timers_core(&c2,&t2);')
    code('OPS_kernels['+str(nk)+'].mpi_time += t2-t1;')
    for n in range (0, nargs):
      if arg_typ[n] == 'ops_arg_dat':
        code('OPS_kernels['+str(nk)+'].transfer += ops_compute_transfer(dim, start, end, &arg'+str(n)+');')
    ENDIF()
    config.depth = config.depth - 2
    code('}')


##########################################################################
#  output individual kernel file
##########################################################################
    if not os.path.exists('./OpenMP4'):
      os.makedirs('./OpenMP4')
    fid = open('./OpenMP4/'+name+'_openmp4_kernel.cpp','w')
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
  #code('#ifdef OPS_GPU')
  code('#include "ops_cuda_rt_support.h"')
  code('#include "ops_openmp4_rt_support.h"')
  #code('#endif')
  if os.path.exists('./user_types.h'):
    code('#include "user_types.h"')
  code('')

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

  fid = open('./OpenMP4/'+master.split('.')[0]+'_common.h','w')
  fid.write('//\n// auto-generated by ops.py\n//\n')
  fid.write(config.file_text)
  fid.close()
  config.file_text =''
  code('#include "./OpenMP4/'+master.split('.')[0]+'_common.h"')
  code('')
  code('')
  code('')
  code('void ops_init_backend() {}')
  code('')
  code('void ops_decl_const_char(int dim, char const *type,')
  code('int size, char *dat, char const *name){')
  config.depth = config.depth + 2

  for nc in range(0,len(consts)):
    IF('!strcmp(name,"'+(str(consts[nc]['name']).replace('"','')).strip()+'")')
    if consts[nc]['dim'].isdigit() and int(consts[nc]['dim'])==1:
      code((str(consts[nc]['name']).replace('"','')).strip()+' = *('+consts[nc]['type']+'*)dat;')
      code('#pragma omp target enter data map(to:'+(str(consts[nc]['name']).replace('"','')).strip()+')')
    else:
      code((str(consts[nc]['name']).replace('"','')).strip()+' = ('+consts[nc]['type']+'*)dat;')
      code('#pragma omp target enter data map(to:'+(str(consts[nc]['name']).replace('"','')).strip()+'[0:'+(consts[nc]['dim'])+'])')

    ENDIF()
    code('else')

  code('{')
  config.depth = config.depth + 2
  code('printf("error: unknown const name\\n"); exit(1);')
  ENDIF()

  config.depth = config.depth - 2
  code('}')

  #for nc in range (0,len(consts)):
  #  if consts[nc]['dim'].isdigit() and int(consts[nc]['dim'])==1:
  #    code(consts[nc]['type']+' '+str(consts[nc]['name']).replace('"','')+'_ompkernel;')
  #  else:
  #    if consts[nc]['dim'] > 0:
  #      num = str(consts[nc]['dim'])
  #    else:
  #      num = 'MAX_CONST_SIZE'
  #    code(consts[nc]['type']+' '+str(consts[nc]['name']).replace('"','')+'_ompkernel['+num+'];')
  #code('')

  #code('void ops_decl_const_char(int dim, char const *type,')
  #code('  int size, char *dat, char const *name){')
  #indent = ' ' * ( 2+ config.depth)
  #line = '  '
  #for nc in range (0,len(consts)):
  #  varname = consts[nc]['name']
  #  if nc > 0:
  #      line += ' else '
  #  #IF('!strcmp(name,"'+(str(consts[nc]['name']).replace('"','')).strip()+'")')
  #  line += 'if(!strcmp(name,'+ (str(consts[nc]['name'])).strip()+'))'
  #  line += '{\n' + indent + 2*' '
  #  line +=(str(consts[nc]['name']).replace('"','')).strip()+' = *('+consts[nc]['type']+'*)dat;'
  #  if consts[nc]['dim'].isdigit() and int(consts[nc]['dim'])==1:
  #    line += 'memcpy(&'+str(consts[nc]['name']).replace('"','')+ '_ompkernel, dat, dim*size);\n' + indent + '#pragma omp target enter data map(to:'+str(consts[nc]['name']).replace('"','')+'_ompkernel'
  #  else:
  #    line += 'memcpy('+str(consts[nc]['name']).replace('"','')+ '_ompkernel, dat, dim*size);\n' + indent + '#pragma omp target enter data map(to:'+str(consts[nc]['name']).replace('"','')+'_ompkernel[:%s]' % str(consts[nc]['dim']) if consts[nc]['dim'] > 0 else 'MAX_CONST_SIZE'

  # line += ')\n'+indent + '}'
  #code(line)
  #code('}')

  #code('')
  #comm('user kernel files')

  #kernel_name_list = ['generate_chunk_kernel']

  #for nk in range(0,len(kernels)):
  #  if kernels[nk]['name'] not in kernel_name_list :
  #    code('#include "'+kernels[nk]['name']+'_openacc_kernel.cpp"')
  #    kernel_name_list.append(kernels[nk]['name'])

  fid = open('./OpenMP4/'+master.split('.')[0]+'_kernels.cpp','w')
  fid.write('//\n// auto-generated by ops.py//\n\n')
  fid.write(config.file_text)
  fid.close()

