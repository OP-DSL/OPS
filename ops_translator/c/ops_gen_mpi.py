
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
OPS MPI_seq code generator (for C/C++ applications)

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


def ops_gen_mpi(master, date, consts, kernels, soa_set):

  OPS_GBL   = 2;

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
      if arg_typ[n] == 'ops_arg_gbl' and accs[n] <> OPS_READ:
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

##########################################################################
#  generate MACROS
##########################################################################
    for n in range (0, nargs):
      if arg_typ[n] == 'ops_arg_dat':
        md=False
        if dims[n].isdigit():
          if int(dims[n])>1:
             md=True
        else:
          # we assume that if a variable is in the call, the dat must be multi dim
          md=True
        if md:
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
    m = text.find(name)
    arg_list = parse_signature(text[i2+len(name):i+j])

    check_accs(name, arg_list, arg_typ, text[i+j:k])
    l = text[i:m].find('inline')
    if(l<0):
      code('inline '+text[i:k+2])
    else:
      code(text[i:k+2])
    code('')
    code('')
    for n in range (0, nargs):
      if arg_typ[n] == 'ops_arg_dat':
        if md :
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
      if nargs <> 1 and n != nargs-1:
        text = text +','
      else:
        text = text +') {'
      if n%n_per_line == 3 and n <> nargs-1:
         text = text +'\n'
    code(text);
    config.depth = 2

    code('')
    comm('Timing')
    code('double t1,t2,c1,c2;')

    code('');
    code('char *p_a['+str(nargs)+'];')
    code('int  offs['+str(nargs)+']['+dim+'];')

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

    code('#ifdef OPS_MPI')
    code('sub_block_list sb = OPS_sub_block_list[block->index];')
    code('#endif')

    code('#ifdef OPS_DEBUG')
    code('ops_register_args(args, "'+name+'");')
    code('#endif')
    code('')

    if arg_idx == 1:
      code('')
      code('int arg_idx['+str(NDIM)+'];')
      code('#ifdef OPS_MPI')
      for n in range (0, nargs):
        code('sub_dat *sd'+str(n)+' = OPS_sub_dat_list[args['+str(n)+'].dat->index];')
      for n in range (0,NDIM):
        code('arg_idx['+str(n)+'] = sb->decomp_disp['+str(n)+']+start['+str(n)+'];')
      code('#else')
      for n in range (0,NDIM):
        code('arg_idx['+str(n)+'] = start['+str(n)+'];')
      code('#endif')
    code('')

    code('')
    code('int arg_idx['+str(NDIM)+'];')
    code('int arg_idx_base['+str(NDIM)+'];')


    code('#ifdef OPS_MPI')
    code('if (compute_ranges(args, '+str(nargs)+',block, range, start, end, arg_idx) < 0) return;')
    code('#else //OPS_MPI')
    FOR('n','0',str(NDIM))
    code('start[n] = range[2*n];end[n] = range[2*n+1];')
    code('arg_idx[n] = start[n];')
    ENDFOR()
    code('#endif //OPS_MPI')
    FOR('n','0',str(NDIM))
    code('arg_idx_base[n] = arg_idx[n];')
    ENDFOR()

    if MULTI_GRID:
      code('int global_idx['+str(NDIM)+'];')
      code('#ifdef OPS_MPI')
      for n in range (0,NDIM):
        code('global_idx['+str(n)+'] = arg_idx['+str(n)+'];')
      code('#else //OPS_MPI')
      for n in range (0,NDIM):
        code('global_idx['+str(n)+'] = start['+str(n)+'];')
      code('#endif //OPS_MPI')
      code('')
    # elif arg_idx:
    #   code('#ifdef OPS_MPI')
    #   for n in range (0,NDIM):
    #     code('arg_idx['+str(n)+'] = sb->decomp_disp['+str(n)+']+start['+str(n)+'];')
    #   code('#else //OPS_MPI')
    #   for n in range (0,NDIM):
    #     code('arg_idx['+str(n)+'] = start['+str(n)+'];')
    #   code('#endif //OPS_MPI')

    if MULTI_GRID:
      for n in range (0, nargs):
        if restrict[n]  == 1 :
          code('int start_'+str(n)+'[2]; int end_'+str(n)+'[2]; int stride_'+str(n)+'[2];')
          FOR('n','0',str(NDIM))
          code('stride_'+str(n)+'[n] = args['+str(n)+'].stencil->mgrid_stride[n];')
          code('start_'+str(n)+'[n]  = start[n]*stride_'+str(n)+'[n];')
          code('end_'+str(n)+'[n]    = end[n];')
          ENDFOR()
        elif prolong[n] == 1:
          comm('This arg has a prolong stencil - so create different ranges')
          code('int start_'+str(n)+'[2]; int end_'+str(n)+'[2]; int stride_'+str(n)+'[2];int d_size_'+str(n)+'[2];')
          code('#ifdef OPS_MPI')
          FOR('n','0',str(NDIM))
          code('sub_dat *sd'+str(n)+' = OPS_sub_dat_list[args['+str(n)+'].dat->index];')
          code('stride_'+str(n)+'[n] = args['+str(n)+'].stencil->mgrid_stride[n];')
          code('d_size_'+str(n)+'[n] = args['+str(n)+'].dat->d_m[n] + sd'+str(n)+'->decomp_size[n] - args['+str(n)+'].dat->d_p[n];')
          code('start_'+str(n)+'[n] = global_idx[n]/stride_'+str(n)+'[n] - sd'+str(n)+'->decomp_disp[n] + args['+str(n)+'].dat->d_m[n];')
          code('end_'+str(n)+'[n] = start_'+str(n)+'[n] + d_size_'+str(n)+'[n];')
          ENDFOR()
          code('#else')
          FOR('n','0',str(NDIM))
          code('stride_'+str(n)+'[n] = args['+str(n)+'].stencil->mgrid_stride[n];')
          code('d_size_'+str(n)+'[n] = args['+str(n)+'].dat->d_m[n] + args['+str(n)+'].dat->size[n] - args['+str(n)+'].dat->d_p[n];')
          code('start_'+str(n)+'[n] = global_idx[n]/stride_'+str(n)+'[n];')
          code('end_'+str(n)+'[n] = start_'+str(n)+'[n] + d_size_'+str(n)+'[n];')
          ENDFOR()
          code('#endif')

    for n in range (0, nargs):
      if arg_typ[n] == 'ops_arg_dat':
        code('offs['+str(n)+'][0] = args['+str(n)+'].stencil->stride[0]*1;  //unit step in x dimension')
        for d in range (1, NDIM):
          if restrict[n]  == 1 or prolong[n] == 1:
            code('offs['+str(n)+']['+str(d)+'] = off'+str(NDIM)+'D('+str(d)+', &start_'+str(n)+'[0],')
            if d == 1:
              code('    &end_'+str(n)+'[0],args['+str(n)+'].dat->size, args['+str(n)+'].stencil->stride) - offs['+str(n)+']['+str(d-1)+'];')
            if d == 2:
              code('    &end_'+str(n)+'[0],args['+str(n)+'].dat->size, args['+str(n)+'].stencil->stride) - offs['+str(n)+']['+str(d-1)+'] - offs['+str(n)+']['+str(d-2)+'];')
          else:
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
        code('int dat'+str(n)+' = (OPS_soa ? args['+str(n)+'].dat->type_size : args['+str(n)+'].dat->elem_size);')


    code('')
    comm('set up initial pointers and exchange halos if necessary')
    for n in range (0, nargs):
      if arg_typ[n] == 'ops_arg_dat':
        if prolong[n] == 1 or restrict[n] == 1:
          starttext = 'start_'+str(n)
        else:
          starttext = 'start'
        code('int base'+str(n)+' = args['+str(n)+'].dat->base_offset + (OPS_soa ? args['+str(n)+'].dat->type_size : args['+str(n)+'].dat->elem_size) * '+starttext+'[0] * args['+str(n)+'].stencil->stride[0];')

        for d in range (1, NDIM):
          line = 'base'+str(n)+' = base'+str(n)+'+ (OPS_soa ? args['+str(n)+'].dat->type_size : args['+str(n)+'].dat->elem_size) *\n'
          for d2 in range (0,d):
            line = line + config.depth*' '+'  args['+str(n)+'].dat->size['+str(d2)+'] *\n'
          code(line[:-1])
          code('  '+starttext+'['+str(d)+'] * args['+str(n)+'].stencil->stride['+str(d)+'];')

        code('p_a['+str(n)+'] = (char *)args['+str(n)+'].data + base'+str(n)+';')

      elif arg_typ[n] == 'ops_arg_gbl':
        if accs[n] == OPS_READ:
          code('p_a['+str(n)+'] = args['+str(n)+'].data;')
        else:
          code('#ifdef OPS_MPI')
          code('p_a['+str(n)+'] = ((ops_reduction)args['+str(n)+'].data)->data + ((ops_reduction)args['+str(n)+'].data)->size * block->index;')
          code('#else')
          code('p_a['+str(n)+'] = ((ops_reduction)args['+str(n)+'].data)->data;')
          code('#endif')
        code('')
      elif arg_typ[n] == 'ops_arg_idx':
        code('p_a['+str(n)+'] = (char *)arg_idx;')
        code('')
      code('')
    code('')


    comm("initialize global variable with the dimension of dats")
    for n in range (0, nargs):
      if arg_typ[n] == 'ops_arg_dat':
        code('xdim'+str(n)+' = args['+str(n)+'].dat->size[0];')
        if NDIM>2 or (NDIM==2 and soa_set):
          code('ydim'+str(n)+' = args['+str(n)+'].dat->size[1];')
        if NDIM>3 or (NDIM==3 and soa_set):
          code('zdim'+str(n)+' = args['+str(n)+'].dat->size[1];')
    code('')

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

    code('int n_x;')
    if not(MULTI_GRID) :
    ###################### NON-MULTIGRID LOOP EXECUTION ########################
      if NDIM==3:
        FOR('n_z','start[2]','end[2]')

      if NDIM>1:
        FOR('n_y','start[1]','end[1]')
      #FOR('n_x','start[0]','start[0]+(end[0]-start[0])/SIMD_VEC')
      #FOR('n_x','start[0]','start[0]+(end[0]-start[0])/SIMD_VEC')
      #code('for( n_x=0; n_x<ROUND_DOWN((end[0]-start[0]),SIMD_VEC); n_x+=SIMD_VEC ) {')
      code('#pragma novector')
      code('for( n_x=start[0]; n_x<start[0]+((end[0]-start[0])/SIMD_VEC)*SIMD_VEC; n_x+=SIMD_VEC ) {')
      config.depth = config.depth+2

      comm('call kernel function, passing in pointers to data -vectorised')
      if reduction == 0 and arg_idx == 0:
        code('#pragma simd')
      FOR('i','0','SIMD_VEC')
      text = name+'( '
      for n in range (0, nargs):
        if arg_typ[n] == 'ops_arg_dat':
          if soa_set:
            text = text +' ('+typs[n]+' *)p_a['+str(n)+']+ i*'+str(stride[NDIM*n])
          else:
            text = text +' ('+typs[n]+' *)p_a['+str(n)+']+ i*'+str(stride[NDIM*n])+'*'+str(dims[n])
        else:
          text = text +' ('+typs[n]+' *)p_a['+str(n)+']'
        if nargs <> 1 and n != nargs-1:
          text = text + ','
        else:
          text = text +' );\n'
        if n%n_per_line == 2 and n <> nargs-1:
          text = text +'\n          '
      code(text);
      if arg_idx:
        code('arg_idx[0]++;')
      ENDFOR()
      code('')

      comm('shift pointers to data x direction')
      for n in range (0, nargs):
        if arg_typ[n] == 'ops_arg_dat':
            code('p_a['+str(n)+']= p_a['+str(n)+'] + (dat'+str(n)+' * off'+str(n)+'_0)*SIMD_VEC;')

      ENDFOR()
      code('')

      FOR('n_x','start[0]+((end[0]-start[0])/SIMD_VEC)*SIMD_VEC','end[0]')
      comm('call kernel function, passing in pointers to data - remainder')
      text = name+'( '
      for n in range (0, nargs):
        if arg_typ[n] == 'ops_arg_dat':
          text = text +' ('+typs[n]+' *)p_a['+str(n)+']'
        else:
          text = text +' ('+typs[n]+' *)p_a['+str(n)+']'
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
          code('p_a['+str(n)+']= p_a['+str(n)+'] + (dat'+str(n)+' * off'+str(n)+'_0);')

      if arg_idx:
        code('#ifdef OPS_MPI')
        for n in range (0,1):
          code('arg_idx['+str(n)+'] = sb->decomp_disp['+str(n)+']+start['+str(n)+'];')
        code('#else')
        for n in range (0,1):
          code('arg_idx['+str(n)+'] = start['+str(n)+'];')
        code('#endif')
        code('arg_idx[1]++;')
      ENDFOR()
      code('')

      if NDIM > 1:
        comm('shift pointers to data y direction')
        for n in range (0, nargs):
          if arg_typ[n] == 'ops_arg_dat':
            code('p_a['+str(n)+']= p_a['+str(n)+'] + (dat'+str(n)+' * off'+str(n)+'_1);')
        if arg_idx:
          for n in range (0,1):
            code('arg_idx['+str(n)+'] = arg_idx_base['+str(n)+'];')
          code('arg_idx[1]++;')
        ENDFOR()

      if NDIM==3:
        comm('shift pointers to data z direction')
        for n in range (0, nargs):
          if arg_typ[n] == 'ops_arg_dat':
            code('p_a['+str(n)+']= p_a['+str(n)+'] + (dat'+str(n)+' * off'+str(n)+'_2);')

        if arg_idx:
          for n in range (0,2):
            code('arg_idx['+str(n)+'] = arg_idx_base['+str(n)+'];')
          code('arg_idx[2]++;')
        ENDFOR()

    else:
    ######################### MULTIGRID LOOP EXECUTION #########################
      if NDIM==3:
        FOR('n_z','0','end[2]-start[2]')

      if NDIM>1:
        FOR('n_y','0','end[1]-start[1]')

      FOR('n_x','0','end[0]-start[0]')
      comm('call kernel function, passing in pointers to data')

      if arg_idx:
        if NDIM==1:
          code('int arg_idx_l[] = {arg_idx[0]+n_x};')
        elif NDIM==2:
          code('int arg_idx_l[] = {arg_idx[0]+n_x, arg_idx[1]+n_y};')
        elif NDIM==3:
          code('int arg_idx_l[] = {arg_idx[0]+n_x, arg_idx[1]+n_y, arg_idx[2]+n_z};')

      text = name+'( '
      for n in range (0, nargs):
        if arg_typ[n] == 'ops_arg_dat':
          if restrict[n] == 1:
            n_x = 'n_x*stride_'+str(n)+'[0]'
            n_y = 'n_y*stride_'+str(n)+'[1]'
            n_z = 'n_z*stride_'+str(n)+'[2]'
          elif prolong[n] == 1:
            n_x = '(n_x+global_idx[0]%stride_'+str(n)+'[0])/stride_'+str(n)+'[0]'
            n_y = '(n_y+global_idx[1]%stride_'+str(n)+'[1])/stride_'+str(n)+'[1]'
            n_z = '(n_z+global_idx[2]%stride_'+str(n)+'[2])/stride_'+str(n)+'[2]'
          else:
            n_x = 'n_x'
            n_y = 'n_y'
            n_z = 'n_z'

          text = text +' ('+typs[n]+' *)p_a['+str(n)+'] + '+n_x+'*'+str(stride[NDIM*n])+'*'+str(dims[n])
          if NDIM >= 2:
            text = text + ' + '+n_y+'*xdim'+str(n)+'*'+str(stride[NDIM*n+1])+'*'+str(dims[n])
          if NDIM >= 3:
            text = text + ' + '+n_z+'*xdim'+str(n)+'*ydim'+str(n)+'*'+str(stride[NDIM*n+2])
        elif arg_typ[n] == 'ops_arg_gbl':
          text = text +' ('+typs[n]+' *)p_a['+str(n)+']'
        elif arg_typ[n] == 'ops_arg_idx':
          text = text +'arg_idx_l'

        if nargs <> 1 and n != nargs-1:
          text = text + ','
        else:
          text = text +' );\n'
        if n%n_per_line == 0 and n <> nargs-1:
          text = text +'\n          '
      code(text);
      code('')

      comm('shift pointers to data x direction')
      for n in range (0, nargs):
        if arg_typ[n] == 'ops_arg_dat':
          if restrict[n] == 1:
            code('p_a['+str(n)+']= p_a['+str(n)+'] + (dat'+str(n)+' * off'+str(n)+'_0) * stride_'+str(n)+'[0];')
          elif prolong[n] == 1:
            code('p_a['+str(n)+']= p_a['+str(n)+'] + (dat'+str(n)+' * off'+str(n)+'_0) * (((global_idx[0]+1) % stride_'+str(n)+'[0] == 0)?1:0);')
          else:
            code('p_a['+str(n)+']= p_a['+str(n)+'] + (dat'+str(n)+' * off'+str(n)+'_0);')

      if arg_idx:
        code('#ifdef OPS_MPI')
        for n in range (0,2):
          code('arg_idx['+str(n)+'] = sb->decomp_disp['+str(n)+']+start['+str(n)+'];')
        code('#else')
        for n in range (0,2):
          code('arg_idx['+str(n)+'] = start['+str(n)+'];')
        code('#endif')
        code('arg_idx[2]++;')
      if MULTI_GRID:
        code('global_idx[0]++;')

      ENDFOR()
      if NDIM==2:
        ENDFOR()
      if NDIM==3:
        ENDFOR()
        ENDFOR()

      # text = name+'( '
      # for n in range (0, nargs):
      #   if arg_typ[n] == 'ops_arg_dat':
      #     text = text +' ('+typs[n]+' *)p_a['+str(n)+']'
      #   else:
      #     text = text +' ('+typs[n]+' *)p_a['+str(n)+']'
      #   if nargs <> 1 and n != nargs-1:
      #     text = text + ','
      #   else:
      #     text = text +' );\n'
      #   if n%n_per_line == 2 and n <> nargs-1:
      #     text = text +'\n          '
      # code(text);
      # code('')

      # comm('shift pointers to data x direction')
      # for n in range (0, nargs):
      #   if arg_typ[n] == 'ops_arg_dat':
      #     if restrict[n] == 1:
      #       code('p_a['+str(n)+']= p_a['+str(n)+'] + (dat'+str(n)+' * off'+str(n)+'_0) * stride_'+str(n)+'[0];')
      #     elif prolong[n] == 1:
      #       code('p_a['+str(n)+']= p_a['+str(n)+'] + (dat'+str(n)+' * off'+str(n)+'_0) * (((global_idx[0]+1) % stride_'+str(n)+'[0] == 0)?1:0);')
      #     else:
      #       code('p_a['+str(n)+']= p_a['+str(n)+'] + (dat'+str(n)+' * off'+str(n)+'_0);')

      # if arg_idx:
      #   code('arg_idx[0]++;')
      # if MULTI_GRID:
      #   code('global_idx[0]++;')

      # ENDFOR()
      # code('')

      # if NDIM > 1:
      #   comm('shift pointers to data y direction')
      #   for n in range (0, nargs):
      #     if arg_typ[n] == 'ops_arg_dat':
      #       if restrict[n] == 1:
      #         code('p_a['+str(n)+']= p_a['+str(n)+'] + (dat'+str(n)+' * off'+str(n)+'_1) * stride_'+str(n)+'[1];')
      #       elif prolong[n] == 1:
      #         IF('(global_idx[1]+1) % stride_'+str(n)+'[1] == 0')
      #         code('p_a['+str(n)+']= p_a['+str(n)+'] + (dat'+str(n)+' * off'+str(n)+'_1);')
      #         ENDIF()
      #         ELSE()
      #         code('p_a['+str(n)+']= p_a['+str(n)+'] - (dat'+str(n)+' * off'+str(n)+'_0) * (end_'+str(n)+'[0]-start_'+str(n)+'[0]);')
      #         ENDIF()
      #       else:
      #         code('p_a['+str(n)+']= p_a['+str(n)+'] + (dat'+str(n)+' * off'+str(n)+'_1);')

      #   if arg_idx:
      #     code('')
      #     for n in range (0,1):
      #       code('arg_idx['+str(n)+'] = arg_idx_base['+str(n)+'];')
      #     code('arg_idx[1]++;')

      #   if MULTI_GRID:
      #     code('')
      #     code('#ifdef OPS_MPI')
      #     for n in range (0,1):
      #       code('global_idx['+str(n)+'] = arg_idx['+str(n)+'];')
      #     code('#else //OPS_MPI')
      #     for n in range (0,1):
      #       code('global_idx['+str(n)+'] = start['+str(n)+'];')
      #     code('#endif //OPS_MPI')
      #     code('global_idx[1]++;')

      #   ENDFOR()

      # if NDIM==3:
      #   comm('shift pointers to data z direction')
      #   for n in range (0, nargs):
      #     if arg_typ[n] == 'ops_arg_dat':
      #       code('p_a['+str(n)+']= p_a['+str(n)+'] + (dat'+str(n)+' * off'+str(n)+'_2);')

      #   if arg_idx:
      #     for n in range (0,2):
      #       code('arg_idx['+str(n)+'] = arg_idx_base['+str(n)+'];')
      #     code('arg_idx[2]++;')
      #   ENDFOR()


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
      code('#include "'+kernels[nk]['name']+'_seq_kernel.cpp"')
      kernel_name_list.append(kernels[nk]['name'])

  master = master.split('.')[0]
  fid = open('./MPI/'+master.split('.')[0]+'_seq_kernels.cpp','w')
  fid.write('//\n// auto-generated by ops.py//\n\n')
  fid.write(config.file_text)
  fid.close()
