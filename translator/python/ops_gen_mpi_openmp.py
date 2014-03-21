
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
OPS MPI_OpenMP code generator

This routine is called by ops.py which parses the input files

It produces a file xxx_omp_kernel.cpp for each kernel,
plus a master kernel file

"""

import re
import datetime

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

def mult(text, i, n):
  text = text + '1'
  for nn in range (0, i):
    text = text + '* args['+str(n)+'].dat->block_size['+str(nn)+']'

  return text

def ops_gen_mpi_openmp(master, date, consts, kernels):

  global dims, stens
  global g_m, file_text, depth

  OPS_ID   = 1;  OPS_GBL   = 2;  OPS_MAP = 3;

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

    #parse stencil to locate strided access
    stride = [1] * nargs * 2

    for n in range (0, nargs):
      if str(stens[n]).find('STRID2D_X') > 0:
        stride[2*n+1] = 0
      elif str(stens[n]).find('STRID2D_Y') > 0:
        stride[2*n] = 0


    reduction = 0
    for n in range (0, nargs):
      if arg_typ[n] == 'ops_arg_gbl' and accs[n] <> OPS_READ:
        reduction = 1



##########################################################################
#  start with omp kernel function
##########################################################################

    g_m = 0;
    file_text = ''
    depth = 0
    n_per_line = 4

    i = name.find('kernel')
    name2 = name[0:i-1]

    reduction = False
    ng_args = 0

    for n in range (0, nargs):
      if arg_typ[n] == 'ops_arg_gbl':
        reduction = True
      else:
        ng_args = ng_args + 1

    code('#ifdef _OPENMP')
    code('#include <omp.h>')
    code('#endif')

    comm('user function')
    code('#include "'+name2+'_kernel.h"')
    comm('')
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
    depth = 2

    code('');
    code('int  offs['+str(nargs)+'][2];')

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

    code('sub_block_list sb = OPS_sub_block_list[block->index];')

    comm('compute localy allocated range for the sub-block')
    code('')
    code('int start['+str(NDIM)+'];')
    code('int end['+str(NDIM)+'];')
    code('')

    FOR('n','0',str(NDIM))
    code('start[n] = sb->istart[n];end[n] = sb->iend[n]+1;')
    IF('start[n] >= range[2*n]')
    code('start[n] = 0;')
    ENDIF()
    ELSE()
    code('start[n] = range[2*n] - start[n];')
    ENDIF()

    IF('end[n] >= range[2*n+1]')
    code('end[n] = range[2*n+1] - sb->istart[n];')
    ENDIF()
    ELSE()
    code('end[n] = sb->sizes[n];')
    ENDIF()
    ENDFOR()
    code('')

    code('#ifdef OPS_DEBUG')
    code('ops_register_args(args, "'+name+'");')
    code('#endif')
    code('')

    for n in range (0, nargs):
      if arg_typ[n] == 'ops_arg_dat':
        code('offs['+str(n)+'][0] = args['+str(n)+'].stencil->stride[0]*1;  //unit step in x dimension')
        #original offset calculation via funcion call
        #FOR('n','1',str(NDIM))
        #code('offs['+str(n)+'][n] = off2('+str(NDIM)+', n, &start['+str(n)+'*'+str(NDIM)'+],')
        #code('&end['+str(n)+'*'+str(NDIM)+'],args['+str(n)+'].dat->block_size, args['+str(n)+'].stencil->stride);')
        #ENDFOR()
        for d in range (1, NDIM):
          code('offs['+str(n)+']['+str(d)+'] = off2D('+str(d)+', &start[0],')
          code('&end[0],args['+str(n)+'].dat->block_size, args['+str(n)+'].stencil->stride);')
          code('')

    code('')
    code('')

    for n in range (0, nargs):
      if arg_typ[n] == 'ops_arg_dat':
        code('int off'+str(n)+'_1 = offs['+str(n)+'][0];')
        code('int off'+str(n)+'_2 = offs['+str(n)+'][1];')
        code('int dat'+str(n)+' = args['+str(n)+'].dat->size;')

    code('')
    if reduction == True:
      for n in range (0, nargs):
        if arg_typ[n] == 'ops_arg_gbl':
          code((str(typs[n]).replace('"','')).strip()+'*arg'+str(n)+'h = ('+(str(typs[n]).replace('"','')).strip()+' *)arg'+str(n)+'.data;')

    code('')
    code('#ifdef _OPENMP')
    code('int nthreads = omp_get_max_threads( );')
    code('#else')
    code('int nthreads = 1;')
    code('#endif')

    #setup reduction variables
    if reduction == True:
      comm('allocate and initialise arrays for global reduction')
      comm('assumes a max of 64 threads with a cacche line size of 64 bytes')
      for n in range (0, nargs):
        if arg_typ[n] == 'ops_arg_gbl':
          code((str(typs[n]).replace('"','')).strip()+' arg_gbl'+str(n)+'['+dims[n]+' * 64 * 64];')

      FOR('thr','0','nthreads')
      for n in range (0, nargs):
        if arg_typ[n] == 'ops_arg_gbl' and accs[n] == OPS_INC:
          FOR('d', '0',dims[n])
          code('arg_gbl'+str(n)+'[d+64*thr] = ZERO_'+(str(typs[n]).replace('"','')).strip()+';')
          ENDFOR()
        elif arg_typ[n] == 'ops_arg_gbl' and accs[n] == OPS_WRITE:
          FOR('d', '0',dims[n])
          code('arg_gbl'+str(n)+'[d+64*thr] = ZERO_'+(str(typs[n]).replace('"','')).strip()+';')
          ENDFOR()
        elif arg_typ[n] == 'ops_arg_gbl' and accs[n] == OPS_MAX:
          FOR('d', '0',dims[n])
          code('arg_gbl'+str(n)+'[d+64*thr] = -INFINITY_'+(str(typs[n]).replace('"','')).strip()+';')
          ENDFOR()
        elif arg_typ[n] == 'ops_arg_gbl' and accs[n] == OPS_MIN:
          FOR('d', '0',dims[n])
          code('arg_gbl'+str(n)+'[d+64*thr] = INFINITY_'+(str(typs[n]).replace('"','')).strip()+';')
          ENDFOR()
        elif arg_typ[n] == 'ops_arg_gbl' and accs[n] == OPS_READ:
          FOR('d', '0',dims[n])
          code('arg_gbl'+str(n)+'[d+64*thr] = arg'+str(n)+'h[d];')
          ENDFOR()
      ENDFOR()

    for n in range (0, nargs):
      if arg_typ[n] == 'ops_arg_dat':
        code('xdim'+str(n)+' = args['+str(n)+'].dat->block_size[0]*args['+str(n)+'].dat->dim;')
    code('')

    for n in range (0, nargs):
      if arg_typ[n] == 'ops_arg_dat':
        #compute max halo depths using stencil
        code('int max'+str(n)+'['+str(NDIM)+']; int min'+str(n)+'['+str(NDIM)+'];')
        FOR('n','0',str(NDIM))
        code('max'+str(n)+'[n] = 0;min'+str(n)+'[n] = 0;')
        ENDFOR()
        FOR('p','0','args['+str(n)+'].stencil->points')
        FOR('n','0',str(NDIM))
        code('max'+str(n)+'[n] = MAX(max'+str(n)+'[n],args['+str(n)+'].stencil->stencil['+str(NDIM)+'*p + n]);')# * ((range[2*n+1]-range[2*n]) == 1 ? 0 : 1);');
        code('min'+str(n)+'[n] = MIN(min'+str(n)+'[n],args['+str(n)+'].stencil->stencil['+str(NDIM)+'*p + n]);')# * ((range[2*n+1]-range[2*n]) == 1 ? 0 : 1);');
        ENDFOR()
        ENDFOR()

    comm('Halo Exchanges')
    for n in range (0, nargs):
      if arg_typ[n] == 'ops_arg_dat' and (accs[n] == OPS_READ or accs[n] == OPS_RW ):
        code('ops_exchange_halo2(&args['+str(n)+'],max'+str(n)+',min'+str(n)+');')
        #code('ops_exchange_halo(&args['+str(n)+'],2);')
    code('')

    code('ops_H_D_exchanges(args, '+str(nargs)+');\n')

    code('')
    comm('Timing')
    code('double t1,t2,c1,c2;')
    code('ops_timing_realloc('+str(nk)+',"'+name+'");')
    code('ops_timers_core(&c1,&t1);')
    code('')


    code('')
    code('#pragma omp parallel for')
    FOR('thr','0','nthreads')

    code('')
    comm('start index for each dimension')
    for d in range (0, NDIM):
      code('int start'+str(d)+';')


    code('')
    code('int y_size = end[1]-start[1];')
    code('char *p_a['+str(nargs)+'];')
    code('')
    code('int start_i = start[1] + ((y_size-1)/nthreads+1)*thr;')
    code('int finish_i = start[1] + MIN(((y_size-1)/nthreads+1)*(thr+1),y_size);')
    code('')

    comm('get addresss per thread')
    code('start0 = start[0];')
    code('start1 = start[1] + ((y_size-1)/nthreads+1)*thr;')
    code('')



    for n in range (0, nargs):
      if arg_typ[n] == 'ops_arg_dat':

        comm('set up initial pointers ')

        code('int base'+str(n)+' = dat'+str(n)+' * 1 * ')
        code('(start0 * args['+str(n)+'].stencil->stride[0] - args['+str(n)+'].dat->offset[0]);')
        for d in range (1, NDIM):
          code('base'+str(n)+' = base'+str(n)+'  + dat'+str(n)+' * args['+str(n)+'].dat->block_size['+str(d-1)+'] * ')
          code('(start'+str(d)+' * args['+str(n)+'].stencil->stride['+str(d)+'] - args['+str(n)+'].dat->offset['+str(d)+']);')

        code('p_a['+str(n)+'] = (char *)args['+str(n)+'].data + base'+str(n)+';')
        #original address calculation via funcion call
        #code('+ address2('+str(NDIM)+', args['+str(n)+'].dat->size, &start0,')
        #code('args['+str(n)+'].dat->block_size, args['+str(n)+'].stencil->stride, args['+str(n)+'].dat->offset);')
      else:
        code('p_a['+str(n)+'] = (char *)args['+str(n)+'].data;')

      code('')
    code('')

    FOR('n_y','start_i','finish_i')
    FOR('n_x','start[0]','start[0]+(end[0]-start[0])/SIMD_VEC')
    #depth = depth+2

    comm('call kernel function, passing in pointers to data -vectorised')
    if reduction == 0:
      code('#pragma simd')
    FOR('i','0','SIMD_VEC')
    text = name+'( '
    for n in range (0, nargs):
      if arg_typ[n] == 'ops_arg_dat':
        text = text +' ('+(str(typs[n]).replace('"','')).strip()+' *)p_a['+str(n)+']+ i*'+str(stride[2*n])
      else:
        text = text +' &arg_gbl'+str(n)+'[64*thr]'

      if nargs <> 1 and n != nargs-1:
        text = text + ','
      else:
        text = text +' );\n'
      if n%n_per_line == 2 and n <> nargs-1:
        text = text +'\n          '
    code(text);
    ENDFOR()
    code('')


    comm('shift pointers to data x direction')
    for n in range (0, nargs):
      if arg_typ[n] == 'ops_arg_dat':
          code('p_a['+str(n)+']= p_a['+str(n)+'] + (dat'+str(n)+' * off'+str(n)+'_1)*SIMD_VEC;')

    ENDFOR()
    code('')

    FOR('n_x','start[0]+((end[0]-start[0])/SIMD_VEC)*SIMD_VEC','end[0]')
    #depth = depth+2
    comm('call kernel function, passing in pointers to data - remainder')
    text = name+'( '
    for n in range (0, nargs):
      if arg_typ[n] == 'ops_arg_dat':
        text = text +' ('+(str(typs[n]).replace('"','')).strip()+' *)p_a['+str(n)+']'
      else:
        text = text +' &arg_gbl'+str(n)+'[64*thr]'

      if nargs <> 1 and n != nargs-1:
        text = text + ','
      else:
        text = text +' );\n'
      if n%n_per_line == 2 and n <> nargs-1:
        text = text +'\n          '
    code(text);

    code('')


    comm('shift pointers to data x direction')
    for n in range (0, nargs):
      if arg_typ[n] == 'ops_arg_dat':
          code('p_a['+str(n)+']= p_a['+str(n)+'] + (dat'+str(n)+' * off'+str(n)+'_1);')

    ENDFOR()
    code('')


    comm('shift pointers to data y direction')
    for n in range (0, nargs):
      if arg_typ[n] == 'ops_arg_dat':
        code('p_a['+str(n)+']= p_a['+str(n)+'] + (dat'+str(n)+' * off'+str(n)+'_2);')
    ENDFOR()

    ENDFOR() #end of OMP parallel loop


    #generate code for combining the reductions
    if reduction == True:
      code('')
      comm(' combine reduction data')
      FOR('thr','0','nthreads')
      for n in range (0, nargs):
        if arg_typ[n] == 'ops_arg_gbl':
          FOR('d','0',dims[n])
          if accs[n] == OPS_INC:
            code('arg'+str(n)+'h[0] += arg_gbl'+str(n)+'[64*thr];')
          elif accs[n] == OPS_MIN:
            code('arg'+str(n)+'h[0] = MIN(arg'+str(n)+'h[0], arg_gbl'+str(n)+'[64*thr]);')
          elif accs[n] == OPS_MAX:
            code('arg'+str(n)+'h[0] = MAX(arg'+str(n)+'h[0], arg_gbl'+str(n)+'[64*thr]);')
          elif accs[n] == OPS_WRITE:
            code('if(arg_gbl'+str(n)+'[64*thr] != 0.0) arg'+str(n)+'h[0] += arg_gbl'+str(n)+'[64*thr];')
          ENDFOR()
      ENDFOR()


    for n in range (0, nargs):
      if arg_typ[n] == 'ops_arg_gbl' and accs[n] != OPS_READ:
        code('ops_mpi_reduce(&arg'+str(n)+',('+(str(typs[n]).replace('"','')).strip()+' *)arg'+str(n)+'h);')

    code('ops_set_dirtybit_host(args, '+str(nargs)+');\n')

    for n in range (0, nargs):
      if arg_typ[n] == 'ops_arg_dat' and (accs[n] == OPS_WRITE or accs[n] == OPS_RW or accs[n] == OPS_INC):
        code('ops_set_halo_dirtybit(&args['+str(n)+']);')



    code('')
    comm('Update kernel record')
    code('ops_timers_core(&c2,&t2);')
    code('OPS_kernels['+str(nk)+'].count++;')
    code('OPS_kernels['+str(nk)+'].time += t2-t1;')
    for n in range (0, nargs):
      if arg_typ[n] == 'ops_arg_dat':
        code('OPS_kernels['+str(nk)+'].transfer += ops_compute_transfer(dim, range, &arg'+str(n)+');')
    depth = depth - 2
    code('}')

##########################################################################
#  output individual kernel file
##########################################################################
    fid = open(name+'_omp_kernel.cpp','w')
    date = datetime.datetime.now()
    fid.write('//\n// auto-generated by ops.py on '+date.strftime("%Y-%m-%d %H:%M")+'\n//\n\n')
    fid.write(file_text)
    fid.close()

# end of main kernel call loop

##########################################################################
#  output one master kernel file
##########################################################################
  depth = 0
  file_text =''
  comm('header')
  code('#include "ops_lib_cpp.h"')
  code('#include "ops_lib_mpi.h"')
  code('')

  comm('global constants')

  for nc in range (0,len(consts)):
    if consts[nc]['dim']==1:
      code('extern '+consts[nc]['type'][1:-1]+' '+(str(consts[nc]['name']).replace('"','')).strip()+';')
    else:
      if consts[nc]['dim'] > 0:
        num = str(consts[nc]['dim'])
      else:
        num = 'MAX_CONST_SIZE'

      code('extern '+consts[nc]['type'][1:-1]+' '+(str(consts[nc]['name']).replace('"','')).strip()+'['+num+'];')
  code('')

  #constants for macros - #no need to code generate here as this is included in teh backend lib
  #for i in range(0,20):
  #  code('int xdim'+str(i)+';')
  #code('')

  comm('user kernel files')

  kernel_name_list = []

  for nk in range(0,len(kernels)):
    if kernels[nk]['name'] not in kernel_name_list :
      code('#include "'+kernels[nk]['name']+'_omp_kernel.cpp"')
      kernel_name_list.append(kernels[nk]['name'])

  master = master.split('.')[0]
  fid = open(master.split('.')[0]+'_omp_kernels.cpp','w')
  fid.write('//\n// auto-generated by op2.py on '+date.strftime("%Y-%m-%d %H:%M")+'\n//\n\n')
  fid.write(file_text)
  fid.close()
