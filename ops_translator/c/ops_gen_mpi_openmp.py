
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


def ops_gen_mpi_openmp(master, date, consts, kernels, soa_set):

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

    reduction = 0
    for n in range (0, nargs):
      if arg_typ[n] == 'ops_arg_gbl' and accs[n] != OPS_READ:
        reduction = 1

    arg_idx = 0
    for n in range (0, nargs):
      if arg_typ[n] == 'ops_arg_idx':
        arg_idx = 1

    config.file_text = ''
    config.depth = 0
    n_per_line = 4

    i = name.find('kernel')
    name2 = name[0:i-1]

    reduction = False
    ng_args = 0


##########################################################################
#  generate MACROS
##########################################################################
    for n in range (0, nargs):
      if arg_typ[n] == 'ops_arg_dat':
        if int(dims[n]) > 1:
          if NDIM==1:
            if soa_set:
              code('#define OPS_ACC_MD'+str(n)+'(d,x) ((x)+(d)*xdim'+str(n)+')')
            else:
              code('#define OPS_ACC_MD'+str(n)+'(d,x) ((x)*'+dims[n]+'+(d))')
          if NDIM==2:
            if soa_set:
              code('#define OPS_ACC_MD'+str(n)+'(d,x,y) ((x)+(xdim'+str(n)+'*(y))+(d)*xdim'+str(n)+'*ydim'+str(n)+')')
            else:
              code('#define OPS_ACC_MD'+str(n)+'(d,x,y) ((x)*'+dims[n]+'+(d)+(xdim'+str(n)+'*(y)*'+dims[n]+'))')
          if NDIM==3:
            if soa_set:
              code('#define OPS_ACC_MD'+str(n)+'(d,x,y,z) ((x)+(xdim'+str(n)+'*(y))+(xdim'+str(n)+'*ydim'+str(n)+'*(z))+(d)*xdim'+str(n)+'*ydim'+str(n)+'*zdim'+str(n)+')')
            else:
              code('#define OPS_ACC_MD'+str(n)+'(d,x,y,z) ((x)*'+dims[n]+'+(d)+(xdim'+str(n)+'*(y)*'+dims[n]+')+(xdim'+str(n)+'*ydim'+str(n)+'*(z)*'+dims[n]+'))')


##########################################################################
#  start with omp kernel function
##########################################################################

    for n in range (0, nargs):
      if arg_typ[n] == 'ops_arg_gbl' and accs[n] != OPS_READ:
        reduction = True
      else:
        ng_args = ng_args + 1

    code('#ifdef _OPENMP')
    code('#include <omp.h>')
    code('#endif')
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
      print("COUND NOT FIND KERNEL", name)

    fid = open(file_name, 'r')
    text = fid.read()

    fid.close()
    text = comment_remover(text)
    text = remove_trailing_w_space(text)

    p = re.compile('void\\s+\\b'+name+'\\b')

    i = p.search(text).start()



    if(i < 0):
      print("\n********")
      print("Error: cannot locate user kernel function: "+name+" - Aborting code generation")
      exit(2)
    i2 = i
    i = text[0:i].rfind('\n') #reverse find
    if i < 0:
      i = 0
    j = text[i:].find('{')
    k = para_parse(text, i+j, '{', '}')
    m = text.find(name)
    arg_list = parse_signature(text[i2+len(name):i+j])
    check_accs(name, arg_list, arg_typ, text[i+j:k])
    l = text[i:m].find('inline')
    if(l<0):
      code('inline '+text[i:k+2])
    else:
      code(text[i:k+2])
    code('')
    comm('')
    for n in range (0, nargs):
      if arg_typ[n] == 'ops_arg_dat':
        if int(dims[n]) > 1:
          code('#undef OPS_ACC_MD'+str(n))
    code('')
    code('')

##########################################################################
#  now host stub
##########################################################################
    comm(' host stub function')
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

    code('')
    comm('Timing')
    code('double t1,t2,c1,c2;')

    code('');
    code('int  offs['+str(nargs)+']['+str(NDIM)+'];')

    #code('ops_printf("In loop \%s\\n","'+name+'");')

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
    code('#ifdef OPS_MPI')
    code('sub_block_list sb = OPS_sub_block_list[block->index];')
    code('#endif')

    code('')
    comm('compute locally allocated range for the sub-block')
    code('')
    code('int start['+str(NDIM)+'];')
    code('int end['+str(NDIM)+'];')
    code('int arg_idx['+str(NDIM)+'];')
    code('')

    code('#ifdef OPS_MPI')
    code('if (compute_ranges(args, '+str(nargs)+',block, range, start, end, arg_idx) < 0) return;')
    code('#else')
    FOR('n','0',str(NDIM))
    code('start[n] = range[2*n];end[n] = range[2*n+1];')
    ENDFOR()
    code('#endif')

    code('#ifdef OPS_DEBUG')
    code('ops_register_args(args, "'+name+'");')
    code('#endif')
    code('')

    for n in range (0, nargs):
      if arg_typ[n] == 'ops_arg_dat':
        code('offs['+str(n)+'][0] = args['+str(n)+'].stencil->stride[0]*1;  //unit step in x dimension')
        for d in range (1, NDIM):
          code('offs['+str(n)+']['+str(d)+'] = off'+str(NDIM)+'D('+str(d)+', &start[0],')
          if d == 1:
            code('    &end[0],args['+str(n)+'].dat->size, args['+str(n)+'].stencil->stride) - offs['+str(n)+']['+str(d-1)+'];')
          if d == 2:
            code('    &end[0],args['+str(n)+'].dat->size, args['+str(n)+'].stencil->stride) - offs['+str(n)+']['+str(d-1)+'] - offs['+str(n)+']['+str(d-2)+'];')
        code('')

    code('')
    code('')

    for n in range (0, nargs):
      if arg_typ[n] == 'ops_arg_dat':
        for d in range (0, NDIM):
          code('int off'+str(n)+'_'+str(d)+' = offs['+str(n)+']['+str(d)+'];')
        code('int dat'+str(n)+' =  (OPS_soa ? args['+str(n)+'].dat->type_size : args['+str(n)+'].dat->elem_size);')

    code('')
    if reduction == True:
      for n in range (0, nargs):
        if arg_typ[n] == 'ops_arg_gbl':
          #code(typs[n]+'*arg'+str(n)+'h = ('+typs[n]+' *)arg'+str(n)+'.data;')
          if (accs[n] == OPS_READ):
            code(''+typs[n]+' *arg'+str(n)+'h = ('+typs[n]+' *)arg'+str(n)+'.data;')
          else:
            code('#ifdef OPS_MPI')
            code(typs[n]+' *arg'+str(n)+'h = ('+typs[n]+' *)(((ops_reduction)args['+str(n)+'].data)->data + ((ops_reduction)args['+str(n)+'].data)->size * block->index);')
            code('#else')
            code(typs[n]+' *arg'+str(n)+'h = ('+typs[n]+' *)(((ops_reduction)args['+str(n)+'].data)->data);')
            code('#endif')

    comm('Halo Exchanges')
    code('ops_H_D_exchanges_host(args, '+str(nargs)+');')
    code('ops_halo_exchanges(args,'+str(nargs)+',range);')
    code('ops_H_D_exchanges_host(args, '+str(nargs)+');')

    code('')
    code('#ifdef _OPENMP')
    code('int nthreads = omp_get_max_threads( );')
    code('#else')
    code('int nthreads = 1;')
    code('#endif')



    #setup reduction variables
    if reduction == True:
      comm('allocate and initialise arrays for global reduction')
      comm('assumes a max of MAX_REDUCT_THREADS threads with a cacche line size of 64 bytes')
      for n in range (0, nargs):
        if arg_typ[n] == 'ops_arg_gbl':
          code(typs[n]+' arg_gbl'+str(n)+'[MAX('+dims[n]+' , 64) * MAX_REDUCT_THREADS];')

      FOR('thr','0','nthreads')
      for n in range (0, nargs):
        if arg_typ[n] == 'ops_arg_gbl' and accs[n] == OPS_INC:
          FOR('d', '0',dims[n])
          code('arg_gbl'+str(n)+'[d+64*thr] = ZERO_'+typs[n]+';')
          ENDFOR()
        elif arg_typ[n] == 'ops_arg_gbl' and accs[n] == OPS_WRITE:
          FOR('d', '0',dims[n])
          code('arg_gbl'+str(n)+'[d+64*thr] = ZERO_'+typs[n]+';')
          ENDFOR()
        elif arg_typ[n] == 'ops_arg_gbl' and accs[n] == OPS_MAX:
          FOR('d', '0',dims[n])
          code('arg_gbl'+str(n)+'[d+64*thr] = -INFINITY_'+typs[n]+';')
          ENDFOR()
        elif arg_typ[n] == 'ops_arg_gbl' and accs[n] == OPS_MIN:
          FOR('d', '0',dims[n])
          code('arg_gbl'+str(n)+'[d+64*thr] = INFINITY_'+typs[n]+';')
          ENDFOR()
        elif arg_typ[n] == 'ops_arg_gbl' and accs[n] == OPS_READ:
          FOR('d', '0',dims[n])
          code('arg_gbl'+str(n)+'[d+64*thr] = arg'+str(n)+'h[d];')
          ENDFOR()
      ENDFOR()

    for n in range (0, nargs):
      if arg_typ[n] == 'ops_arg_dat':
        code('xdim'+str(n)+' = args['+str(n)+'].dat->size[0];')
        if NDIM>2 or (NDIM==2 and soa_set):
          code('ydim'+str(n)+' = args['+str(n)+'].dat->size[1];')
        if NDIM>3 or (NDIM==3 and soa_set):
          code('zdim'+str(n)+' = args['+str(n)+'].dat->size[1];')
    code('')

    # for n in range (0, nargs):
    #       if arg_typ[n] == 'ops_arg_dat':
    #         #compute max halo depths using stencil
    #         code('int max'+str(n)+'['+str(NDIM)+']; int min'+str(n)+'['+str(NDIM)+'];')
    #         FOR('n','0',str(NDIM))
    #         code('max'+str(n)+'[n] = 0;min'+str(n)+'[n] = 0;')
    #         ENDFOR()
    #         FOR('p','0','args['+str(n)+'].stencil->points')
    #         FOR('n','0',str(NDIM))
    #         code('max'+str(n)+'[n] = MAX(max'+str(n)+'[n],args['+str(n)+'].stencil->stencil['+str(NDIM)+'*p + n]);')# * ((range[2*n+1]-range[2*n]) == 1 ? 0 : 1);');
    #         code('min'+str(n)+'[n] = MIN(min'+str(n)+'[n],args['+str(n)+'].stencil->stencil['+str(NDIM)+'*p + n]);')# * ((range[2*n+1]-range[2*n]) == 1 ? 0 : 1);');
    #         ENDFOR()
    #         ENDFOR()



    #for n in range (0, nargs):
      #if arg_typ[n] == 'ops_arg_dat' and (accs[n] == OPS_READ or accs[n] == OPS_RW ):
        #code('ops_exchange_halo2(&args['+str(n)+'],max'+str(n)+',min'+str(n)+');')
        #code('ops_exchange_halo(&args['+str(n)+'],2);')

    code('')
    IF('OPS_diags > 1')
    code('ops_timers_core(&c2,&t2);')
    code('OPS_kernels['+str(nk)+'].mpi_time += t2-t1;')
    ENDIF()
    code('')

    code('')
    code('#pragma omp parallel for')
    FOR('thr','0','nthreads')

    code('')
    if NDIM==1:
      outer = 'x'
    if NDIM==2:
      outer = 'y'
    if NDIM==3:
      outer = 'z'
    code('int '+outer+'_size = end['+str(NDIM-1)+']-start['+str(NDIM-1)+'];')
    code('char *p_a['+str(nargs)+'];')
    code('')
    code('int start_i = start['+str(NDIM-1)+'] + (('+outer+'_size-1)/nthreads+1)*thr;')
    code('int finish_i = start['+str(NDIM-1)+'] + MIN((('+outer+'_size-1)/nthreads+1)*(thr+1),'+outer+'_size);')
    code('')

    comm('get address per thread')
    for d in range(0,NDIM-1):
      code('int start'+str(d)+' = start['+str(d)+'];')
    code('int start'+str(NDIM-1)+' = start_i;')
    code('')

    if arg_idx:
      code('int arg_idx['+str(NDIM)+'];')
      code('#ifdef OPS_MPI')
      for n in range (0,NDIM):
        code('arg_idx['+str(n)+'] = sb->decomp_disp['+str(n)+']+start'+str(n)+';')
      code('#else')
      for n in range (0,NDIM):
        code('arg_idx['+str(n)+'] = start'+str(n)+';')
      code('#endif')

    comm('set up initial pointers')
    code('int d_m[OPS_MAX_DIM];')
    for n in range (0, nargs):
      if arg_typ[n] == 'ops_arg_dat':
        code('#ifdef OPS_MPI')
        code('for (int d = 0; d < dim; d++) d_m[d] = args['+str(n)+'].dat->d_m[d] + OPS_sub_dat_list[args['+str(n)+'].dat->index]->d_im[d];')
        code('#else')
        code('for (int d = 0; d < dim; d++) d_m[d] = args['+str(n)+'].dat->d_m[d];')
        code('#endif')
        code('int base'+str(n)+' = dat'+str(n)+' * 1 *')
        code('(start0 * args['+str(n)+'].stencil->stride[0] - args['+str(n)+'].dat->base[0] - d_m[0]);')
        for d in range (1, NDIM):
          line = 'base'+str(n)+' = base'+str(n)+'+ dat'+str(n)+' *\n'
          for d2 in range (0,d):
            line = line + config.depth*' '+'  args['+str(n)+'].dat->size['+str(d2)+'] *\n'
          code(line[:-1])
          code('  (start'+str(d)+' * args['+str(n)+'].stencil->stride['+str(d)+'] - args['+str(n)+'].dat->base['+str(d)+'] - d_m['+str(d)+']);')

        code('p_a['+str(n)+'] = (char *)args['+str(n)+'].data + base'+str(n)+';')

        #original address calculation via funcion call
        #code('+ address2('+str(NDIM)+', args['+str(n)+'].dat->elem_size, &start0,')
        #code('args['+str(n)+'].dat->size, args['+str(n)+'].stencil->stride, args['+str(n)+'].dat->offset);')
      elif arg_typ[n] == 'ops_arg_gbl':
        if accs[n] == OPS_READ:
          code('p_a['+str(n)+'] = (char *)args['+str(n)+'].data;')
        else:
          code('p_a['+str(n)+'] = (char *)arg'+str(n)+'h;')
      elif arg_typ[n] == 'ops_arg_idx':
        code('p_a['+str(n)+'] = (char *)arg_idx;')

      code('')
    code('')

    if NDIM==3:
      FOR('n_z','start_i','finish_i')
      FOR('n_y','start[1]','end[1]')
    if NDIM==2:
      FOR('n_y','start_i','finish_i')
    if NDIM==1:
      FOR('n_x','start_i','finish_i')

    if NDIM > 1:
      FOR('n_x','start[0]','end[0]')


    comm('call kernel function, passing in pointers to data - remainder')
    text = name+'( '
    for n in range (0, nargs):
      if arg_typ[n] == 'ops_arg_dat':
        if accs[n] != OPS_READ:
          text = text +' ('+typs[n]+' * )p_a['+str(n)+']'
        else:
          text = text +' (const '+typs[n]+' * )p_a['+str(n)+']'
      elif arg_typ[n] == 'ops_arg_gbl':
        if accs[n] != OPS_READ:
          text = text +' &arg_gbl'+str(n)+'[64*thr]'
        else:
          text = text +' ('+typs[n]+' * )p_a['+str(n)+']'
      elif arg_typ[n] == 'ops_arg_idx':
        text = text +' arg_idx'

      if nargs != 1 and n != nargs-1:
        text = text + ','
      else:
        text = text +' );\n'
      if n%n_per_line == 2 and n != nargs-1:
        text = text +'\n          '
    code(text);

    code('')


    comm('shift pointers to data x direction')
    for n in range (0, nargs):
      if arg_typ[n] == 'ops_arg_dat':
          code('p_a['+str(n)+']= p_a['+str(n)+'] + (dat'+str(n)+' * off'+str(n)+'_0);')

    if arg_idx==1:
      code('arg_idx[0]++;')
    ENDFOR()
    code('')


    if NDIM > 1:
      comm('shift pointers to data y direction')
      for n in range (0, nargs):
        if arg_typ[n] == 'ops_arg_dat':
          code('p_a['+str(n)+']= p_a['+str(n)+'] + (dat'+str(n)+' * off'+str(n)+'_1);')

      if arg_idx:
        code('#ifdef OPS_MPI')
        for n in range (0,1):
          code('arg_idx['+str(n)+'] = sb->decomp_disp['+str(n)+']+start'+str(n)+';')
        code('#else')
        for n in range (0,1):
          code('arg_idx['+str(n)+'] = start'+str(n)+';')
        code('#endif')
        code('arg_idx[1]++;')
      ENDFOR()

    if NDIM==3:
      comm('shift pointers to data z direction')
      for n in range (0, nargs):
        if arg_typ[n] == 'ops_arg_dat':
            #code('p_a['+str(n)+']= p_a['+str(n)+'] + (dat'+str(n)+' * (off'+str(n)+'_2) - '+str(stride[NDIM*n])+');')
            code('p_a['+str(n)+']= p_a['+str(n)+'] + (dat'+str(n)+' * off'+str(n)+'_2);')
      if arg_idx:
        code('#ifdef OPS_MPI')
        for n in range (0,2):
          code('arg_idx['+str(n)+'] = sb->decomp_disp['+str(n)+']+start'+str(n)+';')
        code('#else')
        for n in range (0,2):
          code('arg_idx['+str(n)+'] = start'+str(n)+';')
        code('#endif')
        code('arg_idx[2]++;')
      ENDFOR()

    ENDFOR() #end of OMP parallel loop


    code('')
    IF('OPS_diags > 1')
    code('ops_timers_core(&c1,&t1);')
    code('OPS_kernels['+str(nk)+'].time += t1-t2;')
    ENDIF()
    code('')

    #generate code for combining the reductions
    if reduction == 1:
      code('')
      comm(' combine reduction data')
      FOR('thr','0','nthreads')
      for n in range (0, nargs):
        if arg_typ[n] == 'ops_arg_gbl':
          FOR('d','0',dims[n])
          if accs[n] == OPS_INC:
            code('arg'+str(n)+'h[d] += arg_gbl'+str(n)+'[64*thr+d];')
          elif accs[n] == OPS_MIN:
            code('arg'+str(n)+'h[d] = MIN(arg'+str(n)+'h[d], arg_gbl'+str(n)+'[64*thr+d]);')
          elif accs[n] == OPS_MAX:
            code('arg'+str(n)+'h[d] = MAX(arg'+str(n)+'h[d], arg_gbl'+str(n)+'[64*thr+d]);')
          elif accs[n] == OPS_WRITE:
            code('if(arg_gbl'+str(n)+'[64*thr+d] != 0.0) arg'+str(n)+'h[d] += arg_gbl'+str(n)+'[64*thr+d];')
          ENDFOR()
      ENDFOR()


    # for n in range (0, nargs):
    #   if arg_typ[n] == 'ops_arg_gbl' and accs[n] != OPS_READ:
    #     #code('ops_mpi_reduce(&arg'+str(n)+',('+typs[n]+' *)arg'+str(n)+'h);')

    code('ops_set_dirtybit_host(args, '+str(nargs)+');\n')

    for n in range (0, nargs):
      if arg_typ[n] == 'ops_arg_dat' and (accs[n] == OPS_WRITE or accs[n] == OPS_RW or accs[n] == OPS_INC):
        #code('ops_set_halo_dirtybit(&args['+str(n)+']);')
        code('ops_set_halo_dirtybit3(&args['+str(n)+'],range);')


    # code('')
    # code('#ifdef OPS_DEBUG')
    # for n in range (0,nargs):
    #   if arg_typ[n] == 'ops_arg_dat' and accs[n] <> OPS_READ:
    #     code('ops_dump3(arg'+str(n)+'.dat,"'+name+'");')
    # code('#endif')
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
    if not os.path.exists('./MPI_OpenMP'):
      os.makedirs('./MPI_OpenMP')
    fid = open('./MPI_OpenMP/'+name+'_omp_kernel.cpp','w')
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
  if soa_set:
    code('#define OPS_SOA')
  code('#define OPS_ACC_MD_MACROS')
  code('#include "ops_lib_cpp.h"')
  code('#ifdef OPS_MPI')
  code('#include "ops_mpi_core.h"')
  code('#endif')
  if os.path.exists('./user_types.h'):
    code('#include "user_types.h"')
  code('')

  comm('set max number of OMP threads for reductions')
  code('#ifndef MAX_REDUCT_THREADS')
  code('#define MAX_REDUCT_THREADS 64')
  code('#endif')

  comm('global constants')

  for nc in range (0,len(consts)):
    if consts[nc]['dim'].isdigit() and int(consts[nc]['dim'])==1:
      code('extern '+consts[nc]['type']+' '+(str(consts[nc]['name']).replace('"','')).strip()+';')
    else:
      if consts[nc]['dim'].isdigit() and int(consts[nc]['dim']) > 0:
        num = consts[nc]['dim']
        code('extern '+consts[nc]['type']+' '+(str(consts[nc]['name']).replace('"','')).strip()+'['+num+'];')
      else:
        code('extern '+consts[nc]['type']+' *'+(str(consts[nc]['name']).replace('"','')).strip()+';')

  code('')

  #constants for macros - #no need to code generate here as this is included in teh backend lib
  #for i in range(0,20):
  #  code('int xdim'+str(i)+';')
  #code('')

  code('')
  code('void ops_init_backend() {}')
  code('')
  comm('user kernel files')

  kernel_name_list = []

  for nk in range(0,len(kernels)):
    if kernels[nk]['name'] not in kernel_name_list :
      code('#include "'+kernels[nk]['name']+'_omp_kernel.cpp"')
      kernel_name_list.append(kernels[nk]['name'])

  master = master.split('.')[0]
  fid = open('./MPI_OpenMP/'+master.split('.')[0]+'_omp_kernels.cpp','w')
  fid.write('//\n// auto-generated by ops.py//\n\n')
  fid.write(config.file_text)
  fid.close()
