#!/usr/bin/env python

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
## @brief
#
#  OPS MPI_seq code generator for Fortran applications
#
#  This routine is called by ops_fortran.py which parses the input files
#
#  It produces a file xxx_seq_kernel.F90 for each kernel
#

"""
OPS MPI_seq code generator for Fortran applications

This routine is called by ops_fortran.py which parses the input files

It produces a file xxx_seq_kernel.F90 for each kernel

"""

import re
import errno
import os

import util_fortran
import config
from datetime import datetime

comment_remover = util_fortran.comment_remover
remove_trailing_w_space = util_fortran.remove_trailing_w_space
comm = util_fortran.comm
code = util_fortran.code
populate_stride = util_fortran.populate_stride
extract_intrinsic_functions = util_fortran.extract_intrinsic_functions

DO = util_fortran.DO
ENDDO = util_fortran.ENDDO
IF = util_fortran.IF
ENDIF = util_fortran.ENDIF

def ops_fortran_gen_mpi(master, consts, kernels, soa_set):

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
    stride = populate_stride(nargs,NDIM,stens)

#    print("MPI kernel name: " + name)

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

##########################################################################
#   Retrieve Kernel Function from file
##########################################################################

    fid = open(name2+'_kernel.inc', 'r')
    text = fid.read()
    fid.close()
    text = comment_remover(text)
    text = remove_trailing_w_space(text)
    i = text.find(name)
    if(i < 0):
      print("\n********")
      print(("Error: cannot locate user kernel function: "+name+" - Aborting code generation"))
      exit(2)

    # need to check accs here - under fortran the
    # parameter vars are declared inside the subroutine
    # for now no check is done
    req_kernel = util_fortran.find_kernel_routine(text, name)
    intrinsic_funcs = extract_intrinsic_functions(req_kernel)
    

##########################################################################
#  generate HEADER
##########################################################################

    code('MODULE '+name.upper()+'_MODULE')
    code('')
    config.depth = 4
    code('USE OPS_FORTRAN_DECLARATIONS')
    code('USE OPS_FORTRAN_RT_SUPPORT')
    code('')
    code('USE OPS_CONSTANTS')
    code('USE, INTRINSIC :: ISO_C_BINDING')
#    if intrinsic_funcs:
#        code(f'!USE, INTRINSIC :: {intrinsic_funcs}')
    code('')
    code('IMPLICIT NONE')
    code('')

##########################################################################
#  generate MACROS
##########################################################################
    for n in range (0, nargs):
      if arg_typ[n] == 'ops_arg_dat':
        config.depth = 4
        if int(dims[n]) == 1:
          code('INTEGER(KIND=4) :: xdim'+str(n+1)+'_'+name)
          if NDIM==1:
            config.depth = 0
            code('#define OPS_ACC'+str(n+1)+'(x) (x + 1)')
          if NDIM==2:
            code('INTEGER(KIND=4) :: ydim'+str(n+1)+'_'+name)
            config.depth = 0
            code('#define OPS_ACC'+str(n+1)+'(x,y) ((x) + (xdim'+str(n+1)+'_'+name+'*(y)) + 1)')
          if NDIM==3:
            code('INTEGER(KIND=4) :: ydim'+str(n+1)+'_'+name)
            code('INTEGER(KIND=4) :: zdim'+str(n+1)+'_'+name)
            config.depth = 0
            code('#define OPS_ACC'+str(n+1)+'(x,y,z) ((x) + (xdim'+str(n+1)+'_'+name+'*(y)) + (xdim'+str(n+1)+'_'+name+'*ydim'+str(n+1)+'_'+name+'*(z)) + 1)')
        code('')

    for n in range (0, nargs):
      if arg_typ[n] == 'ops_arg_dat':
        if dims[n] != '1':
          config.depth = 4
          code('INTEGER(KIND=4) :: multi_d'+str(n+1))
          code('INTEGER(KIND=4) :: xdim'+str(n+1)+'_'+name)
          if soa_set == 0:
            if NDIM==1:
              config.depth = 0
              code('#define OPS_ACC_MD'+str(n+1)+'(d,x) ((d) + ((x)*multi_d'+str(n+1)+'))')
            if NDIM==2:
              code('INTEGER(KIND=4) :: ydim'+str(n+1)+'_'+name)
              config.depth = 0
              code('#define OPS_ACC_MD'+str(n+1)+'(d,x,y) ((d) + ((x)*multi_d'+str(n+1)+') + (xdim'+str(n+1)+'_'+name+'*(y)*multi_d'+str(n+1)+'))')
            if NDIM==3:
              code('INTEGER(KIND=4) :: ydim'+str(n+1)+'_'+name)
              code('INTEGER(KIND=4) :: zdim'+str(n+1)+'_'+name)
              config.depth = 0
              code('#define OPS_ACC_MD'+str(n+1)+'(d,x,y,z) ((d) + ((x)*multi_d'+str(n+1)+') + (xdim'+str(n+1)+'_'+name+'*(y)*multi_d'+str(n+1)+') + (xdim'+str(n+1)+'_'+name+'*ydim'+str(n+1)+'_'+name+'*(z)*multi_d'+str(n+1)+'))')
          else:
            if NDIM==1:
              config.depth = 0
              code('#define OPS_ACC_MD'+str(n+1)+'(d,x) ((x) + (xdim'+str(n+1)+'_'+name+'*(d-1)) + 1)')
            if NDIM==2:
              code('INTEGER(KIND=4) :: ydim'+str(n+1)+'_'+name)
              config.depth = 0
              code('#define OPS_ACC_MD'+str(n+1)+'(d,x,y) ((x) + (xdim'+str(n+1)+'_'+name+'*(y)) + (xdim'+str(n+1)+'_'+name+'*ydim'+str(n+1)+'_'+name+'*(d-1)) + 1)')
            if NDIM==3:
              code('INTEGER(KIND=4) :: ydim'+str(n+1)+'_'+name)
              code('INTEGER(KIND=4) :: zdim'+str(n+1)+'_'+name)
              config.depth = 0
              code('#define OPS_ACC_MD'+str(n+1)+'(d,x,y,z) ((x) + (xdim'+str(n+1)+'_'+name+'*(y)) + (xdim'+str(n+1)+'_'+name+'*ydim'+str(n+1)+'_'+name+'*(z)) + (xdim'+str(n+1)+'_'+name+'*ydim'+str(n+1)+'_'+name+'*zdim'+str(n+1)+'_'+name+'*(d-1)) + 1)')
          code('')

    config.depth = 4
    code('CONTAINS')
    code('')

##########################################################################
#  user kernel subroutine
##########################################################################
    config.depth = 0
    comm('  =============')
    comm('  User function')
    comm('  =============')
    code('')
    code('!DEC$ ATTRIBUTES FORCEINLINE :: ' + name )
    if len(req_kernel) != 0:
      code(req_kernel.rstrip())

    code('')

    for n in range (0, nargs):
      if arg_typ[n] == 'ops_arg_dat':
        if dims[n] == '1':
          code('#undef OPS_ACC'+str(n+1))

    for n in range (0, nargs):
      if arg_typ[n] == 'ops_arg_dat':
        if dims[n] != '1':
          code('#undef OPS_ACC_MD'+str(n+1))
    code('')

##########################################################################
#  generate kernel wrapper subroutine
##########################################################################
    code('SUBROUTINE '+name+'_wrap( &')
    for n in range (0, nargs):
      if arg_typ[n] == 'ops_arg_idx':
        code('    idx, &')
      else:
        code('    opsDat'+str(n+1)+'Local, &')
    for n in range (0, nargs):
      if arg_typ[n] != 'ops_arg_idx':
        code('    dat'+str(n+1)+'_base, &')
    code('    start_indx, &')
    code('    end_indx )')

    code('')

    config.depth = 4
    for n in range (0, nargs):
      if arg_typ[n] == 'ops_arg_dat':
        if accs[n] == OPS_READ:
          code(typs[n].upper()+', DIMENSION(*), INTENT(in)    :: opsDat'+str(n+1)+'Local')
        elif accs[n] == OPS_WRITE:
          code(typs[n].upper()+', DIMENSION(*), INTENT(OUT) :: opsDat'+str(n+1)+'Local')
        elif accs[n] == OPS_RW or accs[n] == OPS_INC:
          code(typs[n].upper()+', DIMENSION(*), INTENT(INOUT) :: opsDat'+str(n+1)+'Local')
        code('INTEGER(KIND=4) :: dat' + str(n+1)+'_base')
        code('')

      elif arg_typ[n] == 'ops_arg_gbl':
        if accs[n] == OPS_READ:
          code(typs[n].upper()+', DIMENSION(*), INTENT(IN)    :: opsDat'+str(n+1)+'Local')
        elif accs[n] == OPS_INC or accs[n] == OPS_MIN or accs[n] == OPS_MAX:
          code(typs[n].upper()+', DIMENSION('+str(dims[n])+'), INTENT(INOUT)    :: opsDat'+str(n+1)+'Local')
        code('INTEGER(KIND=4) :: dat' + str(n+1)+'_base')
        code('')

      elif arg_typ[n] == 'ops_arg_idx':
        code('INTEGER(KIND=4), DIMENSION('+str(NDIM)+'), INTENT(IN) :: idx' )
        code('INTEGER(KIND=4), DIMENSION('+str(NDIM)+')             :: idx_local' )
        code('')

    code('INTEGER(KIND=4), DIMENSION('+str(NDIM)+'), INTENT(IN) :: start_indx, end_indx')
    code('')

    reduction = 0
    reduction_vars = ''
    reduct_op = ''
    for n in range (0, nargs):
      if arg_typ[n] == 'ops_arg_gbl' and accs[n] != OPS_READ:

        if accs[n] == OPS_INC:
          reduct_op = 'REDUCTION(+:'
        elif accs[n] == OPS_MIN:
          reduct_op = 'REDUCTION(MIN:'
        elif accs[n] == OPS_MAX:
          reduct_op = 'REDUCTION(MAX:'

        reduction_vars = reduction_vars + reduct_op +'opsDat'+str(n+1)+'Local) '
        reduction = 1

    if NDIM==1:
      code('INTEGER(KIND=4) :: n_x')
    elif NDIM==2:
      code('INTEGER(KIND=4) :: n_x, n_y')
    elif NDIM==3:
      code('INTEGER(KIND=4) :: n_x, n_y, n_z')
    code('')

    private_idx = ''
    if arg_idx == 1:
        private_idx = ',idx_local'

    if NDIM==1:
      code(f'!$OMP PARALLEL DO PRIVATE(n_x{private_idx}) {reduction_vars}')
      DO('n_x','1','end_indx(1)-start_indx(1)+1')

    elif NDIM==2:
      code(f'!$OMP PARALLEL DO PRIVATE(n_x,n_y{private_idx}) {reduction_vars}')
      DO('n_y','1','end_indx(2)-start_indx(2)+1')
      if reduction != 1:
        code('!$OMP SIMD')
      DO('n_x','1','end_indx(1)-start_indx(1)+1')

    elif NDIM==3:
      code(f'!$OMP PARALLEL DO PRIVATE(n_x,n_y,n_z{private_idx}) {reduction_vars}')
      DO('n_z','1','end_indx(3)-start_indx(3)+1')
      DO('n_y','1','end_indx(2)-start_indx(2)+1')
      if reduction != 1:
        code('!$OMP SIMD')
      DO('n_x','1','end_indx(1)-start_indx(1)+1')

    if arg_idx == 1:
      if NDIM==1:
        code('idx_local = [idx(1)+n_x-1]')
      elif NDIM==2:
        code('idx_local = [idx(1)+n_x-1,idx(2)+n_y-1]')
      elif NDIM==3:
        code('idx_local = [idx(1)+n_x-1,idx(2)+n_y-1,idx(3)+n_z-1]')

    code('')
    code('CALL '+name + '( &')
    indent = config.depth *' '
    line = ''
    for n in range (0, nargs):
      if arg_typ[n] == 'ops_arg_dat':
        if soa_set == 0:
          if NDIM==1:
            line = line + 'opsDat'+str(n+1)+'Local(dat'+str(n+1)+'_base + ((n_x-1)*'+str(stride[n][0])+'*'+str(dims[n])+'))'
          elif NDIM==2:
            line = line + 'opsDat'+str(n+1)+'Local(dat'+str(n+1)+'_base + ((n_x-1)*'+str(stride[n][0])+'*'+str(dims[n])+')'+\
               ' + ((n_y-1)*xdim'+str(n+1)+'_'+name+'*'+str(stride[n][1])+'*'+str(dims[n])+'))'
          elif NDIM==3:
            line = line + 'opsDat'+str(n+1)+'Local(dat'+str(n+1)+'_base + ((n_x-1)*'+str(stride[n][0])+'*'+str(dims[n])+')'+\
               ' + ((n_y-1)*xdim'+str(n+1)+'_'+name+'*'+str(stride[n][1])+'*'+str(dims[n])+')'+\
               ' + ((n_z-1)*ydim'+str(n+1)+'_'+name+'*xdim'+str(n+1)+'_'+name+'*'+str(stride[n][2])+'*'+str(dims[n])+'))'
        else:
          if NDIM==1:
            line = line + 'opsDat'+str(n+1)+'Local(dat'+str(n+1)+'_base + ((n_x-1)*'+str(stride[n][0])+'))'
          elif NDIM==2:
            line = line + 'opsDat'+str(n+1)+'Local(dat'+str(n+1)+'_base + ((n_x-1)*'+str(stride[n][0])+')'+\
                ' + ((n_y-1)*xdim'+str(n+1)+'_'+name+'*'+str(stride[n][1])+'))'
          elif NDIM==3:
            line = line + 'opsDat'+str(n+1)+'Local(dat'+str(n+1)+'_base + ((n_x-1)*'+str(stride[n][0])+')'+\
                ' + ((n_y-1)*xdim'+str(n+1)+'_'+name+'*'+str(stride[n][1])+')'+\
                ' + ((n_z-1)*ydim'+str(n+1)+'_'+name+'*xdim'+str(n+1)+'_'+name+'*'+str(stride[n][2])+'))'

      elif arg_typ[n] == 'ops_arg_gbl':
        line = line + 'opsDat'+str(n+1)+'Local(dat'+str(n+1)+'_base)'
      elif arg_typ[n] == 'ops_arg_idx':
        line = line + 'idx_local'

      if n == nargs-1:
        line = line + ' )'
      else:
        line = line + ', &\n'+indent

    code(line)
    code('')

    if NDIM==1:
      ENDDO()
      if reduction != 1 and arg_idx != 1:
        code('!$OMP END PARALLEL DO')
    elif NDIM==2:
      ENDDO()
      if reduction != 1:
        code('!$OMP END SIMD')
      ENDDO()
    elif NDIM==3:
      ENDDO()
      if reduction != 1:
        code('!$OMP END SIMD')
      ENDDO()
      ENDDO()
    code('')
    config.depth = 0
    code('END SUBROUTINE')

##########################################################################
#  host subroutine
##########################################################################

    code('')
    comm('  ===============')
    comm('  Host subroutine')
    comm('  ===============')
    code('#ifndef OPS_LAZY')
    code('SUBROUTINE '+name+'_host( userSubroutine, block, dim, range, &')
    for n in range (0, nargs):
      if n == nargs-1:
        code('    opsArg'+str(n+1)+')')
      else:
        code('    opsArg'+str(n+1)+', &')

    config.depth = 4
    code('')
    code('CHARACTER(KIND=C_CHAR,LEN=*), INTENT(IN) :: userSubroutine')
    code('TYPE(ops_block), INTENT(IN) :: block')
    code('INTEGER(KIND=4), INTENT(IN):: dim')
    code('INTEGER(KIND=4), DIMENSION(2*dim), INTENT(IN) :: range')
    code('')

    for n in range (0, nargs):
      if arg_typ[n] == 'ops_arg_idx' or arg_typ[n] == 'ops_arg_dat' or arg_typ[n] == 'ops_arg_gbl':
        code('TYPE(ops_arg), INTENT(IN) :: opsArg'+str(n+1))

    code('')
    code('TYPE(ops_arg), DIMENSION('+str(nargs)+') :: opsArgArray')
    code('')

    config.depth = 0
    code('#else')
    code('SUBROUTINE '+name+'_host_execute( descPtr )')
    config.depth = 4
    code('')
    code('TYPE(ops_kernel_descriptor), INTENT(IN) :: descPtr')
    code('TYPE(ops_block) :: block')
    code('INTEGER(KIND=C_INT) :: dim')
    code('INTEGER(KIND=C_INT), POINTER, DIMENSION(:) :: range')
    code('CHARACTER(KIND=C_CHAR), POINTER, DIMENSION(:) :: userSubroutine')
    code('TYPE(ops_arg), POINTER, DIMENSION(:) :: opsArgArray')
    code('')

    for n in range (0, nargs):
      if arg_typ[n] == 'ops_arg_idx' or arg_typ[n] == 'ops_arg_dat' or arg_typ[n] == 'ops_arg_gbl':
        code('TYPE(ops_arg) :: opsArg'+str(n+1))

    code('')
    config.depth = 0
    code('#endif')

    code('')
    config.depth = 4
    for n in range (0, nargs):
      if arg_typ[n] == 'ops_arg_dat':
        code(typs[n].upper()+', POINTER, DIMENSION(:) :: opsDat'+str(n+1)+'Local')
        code('INTEGER(KIND=4) :: opsDat'+str(n+1)+'Cardinality')
        code('INTEGER(KIND=4), POINTER, DIMENSION(:)  :: dat'+str(n+1)+'_size')
        code('INTEGER(KIND=4) :: dat'+str(n+1)+'_base')
        code('')
      elif arg_typ[n] == 'ops_arg_gbl':
        code(typs[n].upper()+', POINTER, DIMENSION(:) :: opsDat'+str(n+1)+'Local')
        code('INTEGER(KIND=4) :: dat'+str(n+1)+'_base')
        code('')

    code('REAL(KIND=8) :: t1__, t2__, t3__')
    code('REAL(KIND=4) :: transfer_total, transfer')
    code('')

    if arg_idx == 1:
      code('INTEGER(KIND=4), DIMENSION('+str(NDIM)+') :: idx' )
      code('')

    code('INTEGER(KIND=4), DIMENSION('+str(NDIM)+') :: start_indx, end_indx')
    code('INTEGER(KIND=4) :: n_indx')
    code('')

    config.depth = 0
    code('#ifdef OPS_LAZY')
    comm('  ==========================')
    comm('  Set from kernel descriptor')
    comm('  ==========================')
    config.depth = 4
    code('dim = descPtr%dim')
    code('CALL c_f_pointer(descPtr%range, range, (/2*dim/))')
    code('CALL c_f_pointer(descPtr%name, userSubroutine, (/descPtr%name_len/))')
    code('block%blockCptr = descPtr%block')
    code('CALL c_f_pointer(block%blockCptr, block%blockPtr)')
    code('CALL c_f_pointer(descPtr%args, opsArgArray, (/descPtr%nargs/))')
    code('')

    for n in range (0, nargs):
      code('opsArg'+str(n+1)+' = opsArgArray('+str(n+1)+')')
    
    config.depth = 0
    code('#else')
    config.depth = 4
    for n in range (0, nargs):
      code('opsArgArray('+str(n+1)+') = opsArg'+str(n+1))
    config.depth = 0
    code('#endif')

    code('')
    config.depth = 4
    code('CALL setKernelTime('+str(nk)+', "'+name+'", 0.0_8, 0.0_8, 0.0_4, 1)')
    code('CALL ops_timers_core(t1__)')
    code('')

    config.depth = 0
    code('#if defined(OPS_MPI) && !defined(OPS_LAZY)')
    config.depth = 4
    IF('getRange(block, start_indx, end_indx, range) < 0')
    code('RETURN')
    ENDIF()
    config.depth = 0
    code('#elif !defined(OPS_MPI)  && !defined(OPS_LAZY)')
    config.depth = 4
    DO('n_indx','1',str(NDIM))
    code('start_indx(n_indx) = range(2*n_indx-1)')
    code('end_indx  (n_indx) = range(2*n_indx)')
    ENDDO()
    config.depth = 0
    code('#else')
    config.depth = 4
    DO('n_indx','1',str(NDIM))
    code('start_indx(n_indx) = range(2*n_indx-1) + 1')
    code('end_indx  (n_indx) = range(2*n_indx) ')
    ENDDO()
    config.depth = 0
    code('#endif')
    
    code('')
    if arg_idx == 1:
      config.depth = 0
      code('#ifdef OPS_MPI')
      config.depth = 4
      code('CALL getIdx(block, start_indx, idx)')
      config.depth = 0
      code('#else')
      config.depth = 4
      for n in range (0, NDIM):
        code('idx('+str(n+1)+') = start_indx('+str(n+1)+')')
      config.depth = 0
      code('#endif')
      code('')

    config.depth = 4
    for n in range (0, nargs):
      if arg_typ[n] == 'ops_arg_dat':
        code('CALL c_f_pointer(getDatSizeFromOpsArg(opsArg'+str(n+1)+'), dat'+str(n+1)+'_size, (/dim/))')
        if NDIM==1:
          code('xdim'+str(n+1)+'_'+name+' = dat'+str(n+1)+'_size(1)')
          code('opsDat'+str(n+1)+'Cardinality = opsArg'+str(n+1)+'%dim * xdim'+str(n+1)+'_'+name)
        elif NDIM==2:
          code('xdim'+str(n+1)+'_'+name+' = dat'+str(n+1)+'_size(1)')
          code('ydim'+str(n+1)+'_'+name+' = dat'+str(n+1)+'_size(2)')
          code('opsDat'+str(n+1)+'Cardinality = opsArg'+str(n+1)+'%dim * xdim'+str(n+1)+'_'+name+' * ydim'+str(n+1)+'_'+name)
        elif NDIM==3:
          code('xdim'+str(n+1)+'_'+name+' = dat'+str(n+1)+'_size(1)')
          code('ydim'+str(n+1)+'_'+name+' = dat'+str(n+1)+'_size(2)')
          code('zdim'+str(n+1)+'_'+name+' = dat'+str(n+1)+'_size(3)')
          code('opsDat'+str(n+1)+'Cardinality = opsArg'+str(n+1)+'%dim * xdim'+str(n+1)+'_'+name+' * ydim'+str(n+1)+'_'+name+' * zdim'+str(n+1)+'_'+name)
        if dims[n] != '1':
          code('multi_d'+str(n+1)+' = getDatDimFromOpsArg(opsArg'+str(n+1)+') ! dimension of the dat')
          code('dat'+str(n+1)+'_base = getDatBaseFromOpsArg'+str(NDIM)+'D(opsArg'+str(n+1)+', start_indx, multi_d'+str(n+1)+')')
        else:
          code('dat'+str(n+1)+'_base = getDatBaseFromOpsArg'+str(NDIM)+'D(opsArg'+str(n+1)+', start_indx, 1)')
        code('CALL c_f_pointer(opsArg'+str(n+1)+'%data,opsDat'+str(n+1)+'Local, (/opsDat'+str(n+1)+'Cardinality/))')
        code('')
      elif arg_typ[n] == 'ops_arg_gbl':
        if accs[n] == OPS_READ:
          code('CALL c_f_pointer(getGblPtrFromOpsArg(opsArg'+str(n+1)+'), opsDat'+str(n+1)+'Local, (/opsArg'+str(n+1)+'%dim/))')
          code('dat'+str(n+1)+'_base = 1')
          code('')
        else:
          code('CALL c_f_pointer(getReductionPtrFromOpsArg(opsArg'+str(n+1)+',block), opsDat'+str(n+1)+'Local, (/opsArg'+str(n+1)+'%dim/))')
          code('dat'+str(n+1)+'_base = 1')
          code('')

    config.depth = 0
    comm('    ==============')
    comm('    Halo exchanges')
    comm('    ==============')
    code('#ifndef OPS_LAZY')
    config.depth = 4
    code('CALL ops_H_D_exchanges_host(opsArgArray, '+str(nargs)+')')
    code('CALL ops_halo_exchanges(opsArgArray, '+str(nargs)+', range)')
    code('CALL ops_H_D_exchanges_host(opsArgArray, '+str(nargs)+')')
    config.depth = 0
    code('#endif')
    code('')

    config.depth = 4
    code('CALL ops_timers_core(t2__)')
    code('')

    config.depth = 0
    #Call user kernel wrapper
    comm('  ==============================')
    comm('  Call kernel wrapper subroutine')
    comm('  ==============================')
    config.depth = 4
    code('CALL '+name+'_wrap( &')
    for n in range (0, nargs):
      if arg_typ[n] == 'ops_arg_idx':
        code('              idx, &')
      else:
        code('              opsDat'+str(n+1)+'Local, &')
    for n in range (0, nargs):
      if arg_typ[n] != 'ops_arg_idx':
        code('              dat'+str(n+1)+'_base, &')
    code('              start_indx, &')
    code('              end_indx )')
    code('')

    code('CALL ops_timers_core(t3__)')
    code('')

    config.depth = 0
    code('#ifndef OPS_LAZY')
    config.depth = 4
    code('CALL ops_set_dirtybit_host(opsArgArray, '+str(nargs)+')')
    for n in range (0, nargs):
      if arg_typ[n] == 'ops_arg_dat' and (accs[n] == OPS_WRITE or accs[n] == OPS_RW or accs[n] == OPS_INC):
        code('CALL ops_set_halo_dirtybit3(opsArg'+str(n+1)+', range)')
    config.depth = 0
    code('#endif')
    code('')

    comm('  ========================')
    comm('  Timing and data movement')
    comm('  ========================')
    config.depth = 4
    code('transfer_total = 0.0_4')
    for n in range (0, nargs):
      if arg_typ[n] == 'ops_arg_dat':
        code('CALL ops_compute_transfer('+str(NDIM)+', start_indx, end_indx, opsArg'+str(n+1)+', transfer)')
        code('transfer_total = transfer_total + transfer')
    code('')
    code('CALL setKernelTime('+str(nk)+', "'+name+'", t3__-t2__, t2__-t1__, transfer_total, 0)')

    config.depth = 0
    code('')
    code('END SUBROUTINE')

    code('')
    code('#ifdef OPS_LAZY')
    code('SUBROUTINE '+name+'_host( userSubroutine, block, dim, range, &')
    for n in range (0, nargs):
      if n == nargs-1:
        code('    opsArg'+str(n+1)+')')
      else:
        code('    opsArg'+str(n+1)+', &')

    config.depth = config.depth + 2
    code('')
    code('CHARACTER(KIND=C_CHAR,LEN=*), INTENT(IN), TARGET :: userSubroutine')
    code('TYPE(ops_block), INTENT(IN) :: block')
    code('INTEGER(KIND=4), INTENT(IN) :: dim')
    code('INTEGER(KIND=4), DIMENSION(2*dim), INTENT(INOUT), TARGET :: range')
    code('INTEGER(KIND=4), DIMENSION(2*dim), TARGET :: range_tmp')
    code('')

    for n in range (0, nargs):
      if arg_typ[n] == 'ops_arg_idx' or arg_typ[n] == 'ops_arg_dat' or 'ops_arg_gbl':
        code('TYPE(ops_arg), INTENT(IN) :: opsArg'+str(n+1))

    code('')
    code('TYPE(ops_arg), DIMENSION('+str(nargs)+'), TARGET :: opsArgArray')
    code('INTEGER(KIND=4) :: n_indx')
    code('CHARACTER(LEN=40) :: namelit')
    code('')

    code('namelit = "'+name+'"')

    code('')
    for n in range (0, nargs):
      code('opsArgArray('+str(n+1)+') = opsArg'+str(n+1))

    code('')
    DO('n_indx','1',str(NDIM))
    code('range_tmp(2*n_indx-1) = range(2*n_indx-1)-1')
    code('range_tmp(2*n_indx)   = range(2*n_indx)')
    ENDDO()

    code('')
    code('CALL create_kerneldesc_and_enque(namelit//c_null_char, c_loc(opsArgArray), &')
    code(f'                             {nargs}, {nk}, dim, 0, c_loc(range_tmp), &')
    code(f'                             block%blockCptr, c_funloc({name}_host_execute))')

    config.depth = 0
    code('')
    code('END SUBROUTINE')
    code('#endif')

    code('')
    code('END MODULE '+name.upper()+"_MODULE")

##########################################################################
#  output individual kernel file
##########################################################################
    try:
      os.makedirs('./mpi_openmp')
    except OSError as e:
      if e.errno != errno.EEXIST:
        raise
    fid = open('./mpi_openmp/'+name+'_seq_kernel.F90','w')
    header_text = f'! Auto-generated at {datetime.now()} by ops-translator legacy\n\n'
    fid.write(header_text)

    fid.write(config.file_text)
    fid.close()
