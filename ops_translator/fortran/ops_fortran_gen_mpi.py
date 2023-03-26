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
import datetime
import errno
import os

import util_fortran
import config

comment_remover = util_fortran.comment_remover
remove_trailing_w_space = util_fortran.remove_trailing_w_space
comm = util_fortran.comm
code = util_fortran.code

DO = util_fortran.DO
ENDDO = util_fortran.ENDDO
IF = util_fortran.IF
ENDIF = util_fortran.ENDIF

def ops_fortran_gen_mpi(master, date, consts, kernels, soa_set):

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
#  generate HEADER
##########################################################################

    code('MODULE '+name.upper()+'_MODULE')
    code('USE OPS_FORTRAN_DECLARATIONS')
    code('USE OPS_FORTRAN_RT_SUPPORT')
    code('')
    code('USE OPS_CONSTANTS')
    code('USE ISO_C_BINDING')
    code('')

##########################################################################
#  generate MACROS
##########################################################################
    for n in range (0, nargs):
      if arg_typ[n] == 'ops_arg_dat':
        if int(dims[n]) == 1:
          code('INTEGER(KIND=4) xdim'+str(n+1))
          if NDIM==1:
            code('#define OPS_ACC'+str(n+1)+'(x) (x+1)')
          if NDIM==2:
            code('#define OPS_ACC'+str(n+1)+'(x,y) (x+xdim'+str(n+1)+'*(y)+1)')
            code('INTEGER(KIND=4) ydim'+str(n+1))
          if NDIM==3:
            code('#define OPS_ACC'+str(n+1)+'(x,y,z) (x+xdim'+str(n+1)+'*(y)+xdim'+str(n+1)+'*ydim'+str(n+1)+'*(z)+1)')
            code('INTEGER(KIND=4) ydim'+str(n+1))
            code('INTEGER(KIND=4) zdim'+str(n+1))
    code('')

    for n in range (0, nargs):
      if arg_typ[n] == 'ops_arg_dat':
        if dims[n] != '1':
          code('INTEGER(KIND=4) multi_d'+str(n+1))
          code('INTEGER(KIND=4) xdim'+str(n+1))
          if soa_set == 0:
            if NDIM==1:
              code('#define OPS_ACC_MD'+str(n+1)+'(d,x) ((x)*multi_d'+str(n+1)+' + d)')
            if NDIM==2:
              code('#define OPS_ACC_MD'+str(n+1)+'(d,x,y) ((x)*multi_d'+str(n+1)+' + (d) + (xdim'+str(n+1)+'*(y)*multi_d'+str(n+1)+'))')
              code('INTEGER(KIND=4) ydim'+str(n+1))
            if NDIM==3:
              code('#define OPS_ACC_MD'+str(n+1)+'(d,x,y,z) ((x)*multi_d'+str(n+1)+' + (d) + (xdim'+str(n+1)+'*(y)*multi_d'+str(n+1)+') + (xdim'+str(n+1)+'*ydim'+str(n+1)+'*(z)*multi_d'+str(n+1)+'))')
              code('INTEGER(KIND=4) ydim'+str(n+1))
              code('INTEGER(KIND=4) zdim'+str(n+1))
          else:
            if NDIM==1:
              code('#define OPS_ACC_MD'+str(n+1)+'(d,x) ((x) + (xdim'+str(n+1)+'*(d-1) + 1))')
            if NDIM==2:
              code('#define OPS_ACC_MD'+str(n+1)+'(d,x,y) ((x) + (xdim'+str(n+1)+'*(y)) + (d-1)*xdim'+str(n+1)+'*ydim'+str(n+1)+' + 1)')
              code('INTEGER(KIND=4) ydim'+str(n+1))
            if NDIM==3:
              code('#define OPS_ACC_MD'+str(n+1)+'(d,x,y,z) ((x) + (xdim'+str(n+1)+'*(y)) + (xdim'+str(n+1)+'*ydim'+str(n+1)+'*(z)) + (d-1)*xdim'+str(n+1)+'*ydim'+str(n+1)+'*zdim'+str(n+1)+' + 1)')
              code('INTEGER(KIND=4) ydim'+str(n+1))
              code('INTEGER(KIND=4) zdim'+str(n+1))
    code('')
    code('contains')
    code('')

##########################################################################
#  user kernel subroutine
##########################################################################
    comm('user function')
    code('!DEC$ ATTRIBUTES FORCEINLINE :: ' + name )
    print(name2)
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
    if len(req_kernel) != 0:
      code(req_kernel)

    #code(text)
    code('')

    for n in range (0, nargs):
      if arg_typ[n] == 'ops_arg_dat':
        if dims[n] == '1':
          code('#undef OPS_ACC'+str(n+1))
    code('')
    for n in range (0, nargs):
      if arg_typ[n] == 'ops_arg_dat':
        if dims[n] != '1':
          code('#undef OPS_ACC_MD'+str(n+1))
    code('')
    code('')

##########################################################################
#  generate kernel wrapper subroutine
##########################################################################
    code('subroutine '+name+'_wrap( &')
    for n in range (0, nargs):
      if arg_typ[n] == 'ops_arg_idx':
        code('& idx, &')
      else:
        code('& opsDat'+str(n+1)+'Local, &')
    for n in range (0, nargs):
      if arg_typ[n] != 'ops_arg_idx':
        code('& dat'+str(n+1)+'_base, &')
    code('& start, &')
    code('& end )')

    config.depth = config.depth + 2
    code('IMPLICIT NONE')
    for n in range (0, nargs):
      if arg_typ[n] == 'ops_arg_dat' and accs[n] == OPS_READ:
        code(typs[n]+', INTENT(IN) :: opsDat'+str(n+1)+'Local(*)')
      elif arg_typ[n] == 'ops_arg_dat' and (accs[n] == OPS_WRITE or accs[n] == OPS_RW or accs[n] == OPS_INC):
        code(typs[n]+' opsDat'+str(n+1)+'Local(*)')
      elif arg_typ[n] == 'ops_arg_gbl':
        code(typs[n]+' opsDat'+str(n+1)+'Local(*)')
      elif arg_typ[n] == 'ops_arg_idx':
        code('integer(4) idx('+str(NDIM)+'),idx_local('+str(NDIM)+')' )

    for n in range (0, nargs):
      if arg_typ[n] != 'ops_arg_idx':
        code('integer dat' + str(n+1)+'_base')
    code('integer(4) start('+str(NDIM)+')')
    code('integer(4) end('+str(NDIM)+')')
    if NDIM==1:
      code('integer n_x')
    elif NDIM==2:
      code('integer n_x, n_y')
    elif NDIM==3:
      code('integer n_x, n_y, n_z')
    code('')

    if NDIM==1:
      if reduction != 1 and arg_idx != 1:
        code('!$OMP SIMD')
      DO('n_x','1','end(1)-start(1)+1')
      if arg_idx == 1:
        code('idx_local(1) = idx(1) + n_x - 1')
    elif NDIM==2:
      DO('n_y','1','end(2)-start(2)+1')
      if arg_idx == 1:
        code('idx_local(2) = idx(2) + n_y - 1')
      if reduction != 1:
        code('!$OMP SIMD')
      DO('n_x','1','end(1)-start(1)+1')
      if arg_idx == 1:
        code('idx_local(1) = idx(1) + n_x - 1')
    elif NDIM==3:
      DO('n_z','1','end(3)-start(3)+1')
      if arg_idx == 1:
        code('idx_local(3) = idx(3) + n_z - 1')
      DO('n_y','1','end(2)-start(2)+1')
      if arg_idx == 1:
        code('idx_local(2) = idx(2) + n_y - 1')
      if reduction != 1:
        code('!$OMP SIMD')
      DO('n_x','1','end(1)-start(1)+1')
      if arg_idx == 1:
        code('idx_local(1) = idx(1) + n_x - 1')

    code('call '+name + '( &')
    indent = config.depth *' '
    line = ''
    for n in range (0, nargs):
      if arg_typ[n] == 'ops_arg_dat':
        if soa_set == 0:
          if NDIM==1:
            line = line + '& opsDat'+str(n+1)+'Local(dat'+str(n+1)+'_base+(n_x-1)*'+str(dims[n])+')'
          elif NDIM==2:
            line = line + '& opsDat'+str(n+1)+'Local(dat'+str(n+1)+'_base+(n_x-1)*'+str(dims[n])+\
               ' + (n_y-1)*xdim'+str(n+1)+'*'+str(dims[n])+')'
          elif NDIM==3:
            line = line + '& opsDat'+str(n+1)+'Local(dat'+str(n+1)+'_base+(n_x-1)*'+str(dims[n])+\
               ' + (n_y-1)*xdim'+str(n+1)+'*'+str(dims[n])+''+\
               ' + (n_z-1)*ydim'+str(n+1)+'*xdim'+str(n+1)+'*'+str(dims[n])+')'
        else:
          if NDIM==1:
            line = line + '& opsDat'+str(n+1)+'Local(dat'+str(n+1)+'_base+(n_x-1)'+')'
          elif NDIM==2:
            line = line + '& opsDat'+str(n+1)+'Local(dat'+str(n+1)+'_base+(n_x-1)'+\
                ' + (n_y-1)*xdim'+str(n+1)+')'
          elif NDIM==3:
            line = line + '& opsDat'+str(n+1)+'Local(dat'+str(n+1)+'_base+(n_x-1)'+\
                ' + (n_y-1)*xdim'+str(n+1)+\
                ' + (n_z-1)*ydim'+str(n+1)+'*xdim'+str(n+1)+')'

      elif arg_typ[n] == 'ops_arg_gbl':
        line = line + '& opsDat'+str(n+1)+'Local(dat'+str(n+1)+'_base)'
      elif arg_typ[n] == 'ops_arg_idx':
        line = line + '& idx_local'

      if n == nargs-1:
        line = line + ' )'
      else:
        line = line + ', &\n'+indent

    code(line)

    if NDIM==1:
      ENDDO()
    elif NDIM==2:
      ENDDO()
      ENDDO()
    elif NDIM==3:
      ENDDO()
      ENDDO()
      ENDDO()
    config.depth = config.depth - 2
    code('end subroutine')

##########################################################################
#  host subroutine
##########################################################################

    code('')
    comm('host subroutine')
    code('#ifndef OPS_LAZY')
    code('subroutine '+name+'_host( userSubroutine, block, dim, range, &')
    for n in range (0, nargs):
      if n == nargs-1:
        code('& opsArg'+str(n+1)+')')
      else:
        code('& opsArg'+str(n+1)+', &')

    config.depth = config.depth + 2
    code('IMPLICIT NONE')
    code('character(kind=c_char,len=*), INTENT(IN) :: userSubroutine')
    code('type ( ops_block ), INTENT(IN) :: block')
    code('integer(kind=4), INTENT(IN):: dim')
    code('integer(kind=4)   , DIMENSION(2*dim), INTENT(IN) :: range')

    for n in range (0, nargs):
      if arg_typ[n] == 'ops_arg_idx':
        code('type ( ops_arg )  , INTENT(IN) :: opsArg'+str(n+1))
      if arg_typ[n] == 'ops_arg_dat':
        code('type ( ops_arg )  , INTENT(IN) :: opsArg'+str(n+1))
      elif arg_typ[n] == 'ops_arg_gbl':
        code('type ( ops_arg )  , INTENT(IN) :: opsArg'+str(n+1))
    code('')
    code('type ( ops_arg ) , DIMENSION('+str(nargs)+') :: opsArgArray')

    config.depth = config.depth - 2
    code('#else')
    code('subroutine '+name+'_host_execute( descPtr )')
    config.depth = config.depth + 2
    code('use iso_c_binding')
    code('IMPLICIT NONE')
    code('type (ops_kernel_descriptor), intent(in) :: descPtr')
    code('type (ops_block) :: block')
    code('integer(kind=c_int) :: dim')
    code('integer(kind=c_int), pointer, DIMENSION(:) :: range')
    code('character(kind=c_char), pointer, DIMENSION(:) :: userSubroutine')
    code('type ( ops_arg ) , pointer, DIMENSION(:) :: opsArgArray')

    for n in range (0, nargs):
      if arg_typ[n] == 'ops_arg_idx':
        code('type ( ops_arg ) :: opsArg'+str(n+1))
      if arg_typ[n] == 'ops_arg_dat':
        code('type ( ops_arg ) :: opsArg'+str(n+1))
      elif arg_typ[n] == 'ops_arg_gbl':
        code('type ( ops_arg ) :: opsArg'+str(n+1))

    config.depth = config.depth - 2
    code('#endif')

    code('')
    config.depth = config.depth + 2
    for n in range (0, nargs):
      if arg_typ[n] == 'ops_arg_dat':
        code(typs[n]+', POINTER, DIMENSION(:) :: opsDat'+str(n+1)+'Local')
        code('integer(kind=4) :: opsDat'+str(n+1)+'Cardinality')
        code('integer(kind=4) , POINTER, DIMENSION(:)  :: dat'+str(n+1)+'_size')
        code('integer(kind=4) :: dat'+str(n+1)+'_base')
        code('')
      elif arg_typ[n] == 'ops_arg_gbl':
        code(typs[n]+', POINTER, DIMENSION(:) :: opsDat'+str(n+1)+'Local')
        code('integer(kind=4) :: dat'+str(n+1)+'_base')
        code('')

    code('real(kind=8) t1,t2,t3')
    code('real(kind=4) transfer_total, transfer')
    code('integer start('+str(NDIM)+')')
    code('integer end('+str(NDIM)+')')
    if arg_idx == 1:
      code('integer idx('+str(NDIM)+')')
    code('integer(kind=4) :: n')
    code('')
  

    code('')

    config.depth = config.depth - 2
    code('#ifdef OPS_LAZY')
    config.depth = config.depth + 2
    comm('Set from kernel descriptor')
    code('dim = descPtr%dim')
    code('call c_f_pointer(descPtr%range, range, (/2*dim/))')
    code('call c_f_pointer(descPtr%name, userSubroutine, (/descPtr%name_len/))')
    code('block%blockCptr = descPtr%block')
    code('call c_f_pointer(block%blockCptr, block%blockPtr)')
    code('call c_f_pointer(descPtr%args, opsArgArray, (/descPtr%nargs/))')
    for n in range (0, nargs):
      code('opsArg'+str(n+1)+' = opsArgArray('+str(n+1)+')')
    code('')
    
    config.depth = config.depth - 2
    code('#else')
    config.depth = config.depth + 2
    for n in range (0, nargs):
      code('opsArgArray('+str(n+1)+') = opsArg'+str(n+1))
    config.depth = config.depth - 2
    code('#endif')

    code('')
    config.depth = config.depth + 2
    code('call setKernelTime('+str(nk)+',userSubroutine,0.0_8,0.0_8,0.0_4,0)')
    code('call ops_timers_core(t1)')
    code('')

    config.depth = config.depth - 2
    code('#if defined(OPS_MPI) && !defined(OPS_LAZY)')
    config.depth = config.depth + 2
    IF('getRange(block, start, end, range) < 0')
    code('return')
    ENDIF()
    config.depth = config.depth - 2
    code('#elif !defined(OPS_MPI)  && !defined(OPS_LAZY)')
    config.depth = config.depth + 2
    DO('n','1',str(NDIM))
    code('start(n) = range(2*n-1)')
    code('end(n) = range(2*n)')
    ENDDO()
    config.depth = config.depth - 2
    code('#else')
    config.depth = config.depth + 2
    DO('n','1',str(NDIM))
    code('start(n) = range(2*n-1) + 1')
    code('end(n) = range(2*n) ')
    ENDDO()
    config.depth = config.depth - 2
    code('#endif')
    
    config.depth = config.depth + 2
    code('')
    if arg_idx == 1:
      config.depth = config.depth - 2
      code('#ifdef OPS_MPI')
      config.depth = config.depth + 2
      code('call getIdx(block,start,idx)')
      config.depth = config.depth - 2
      code('#else')
      config.depth = config.depth + 2
      for n in range (0, NDIM):
        code('idx('+str(n+1)+') = start('+str(n+1)+')')
      config.depth = config.depth - 2
      code('#endif')
      config.depth = config.depth + 2
      code('')
    #if arg_idx == 1:
    #  for n in range (0, NDIM):
    #    code('idx('+str(n+1)+') = start('+str(n+1)+')')
    #  code('')

    for n in range (0, nargs):
      if arg_typ[n] == 'ops_arg_dat':
        code('call c_f_pointer(getDatSizeFromOpsArg(opsArg'+str(n+1)+'),dat'+str(n+1)+'_size,(/dim/))')
        if NDIM==1:
          code('xdim'+str(n+1)+' = dat'+str(n+1)+'_size(1)')
          code('opsDat'+str(n+1)+'Cardinality = opsArg'+str(n+1)+'%dim * xdim'+str(n+1))
        elif NDIM==2:
          code('xdim'+str(n+1)+' = dat'+str(n+1)+'_size(1)')
          code('ydim'+str(n+1)+' = dat'+str(n+1)+'_size(2)')
          code('opsDat'+str(n+1)+'Cardinality = opsArg'+str(n+1)+'%dim * xdim'+str(n+1)+' * ydim'+str(n+1))
        elif NDIM==3:
          code('xdim'+str(n+1)+' = dat'+str(n+1)+'_size(1)')
          code('ydim'+str(n+1)+' = dat'+str(n+1)+'_size(2)')
          code('zdim'+str(n+1)+' = dat'+str(n+1)+'_size(3)')
          code('opsDat'+str(n+1)+'Cardinality = opsArg'+str(n+1)+'%dim * xdim'+str(n+1)+' * ydim'+str(n+1)+' * zdim'+str(n+1))
        if dims[n] != '1':
          code('multi_d'+str(n+1)+' = getDatDimFromOpsArg(opsArg'+str(n+1)+') ! dimension of the dat')
          code('dat'+str(n+1)+'_base = getDatBaseFromOpsArg'+str(NDIM)+'D(opsArg'+str(n+1)+',start,multi_d'+str(n+1)+')')
        else:
          code('dat'+str(n+1)+'_base = getDatBaseFromOpsArg'+str(NDIM)+'D(opsArg'+str(n+1)+',start,1)')
        code('call c_f_pointer(opsArg'+str(n+1)+'%data,opsDat'+str(n+1)+'Local,(/opsDat'+str(n+1)+'Cardinality/))')
        code('')
      elif arg_typ[n] == 'ops_arg_gbl':
        if accs[n] == OPS_READ:
          code('call c_f_pointer(getGblPtrFromOpsArg(opsArg'+str(n+1)+'),opsDat'+str(n+1)+'Local, (/opsArg'+str(n+1)+'%dim/))')
          code('dat'+str(n+1)+'_base = 1')
          code('')
        else:
          code('call c_f_pointer(getReductionPtrFromOpsArg(opsArg'+str(n+1)+',block),opsDat'+str(n+1)+'Local, (/opsArg'+str(n+1)+'%dim/))')
          code('dat'+str(n+1)+'_base = 1')
          code('')

    config.depth = config.depth - 2
    code('#ifndef OPS_LAZY')
    config.depth = config.depth + 2
    code('call ops_H_D_exchanges_host(opsArgArray,'+str(nargs)+')')
    code('call ops_halo_exchanges(opsArgArray,'+str(nargs)+',range)')
    code('call ops_H_D_exchanges_host(opsArgArray,'+str(nargs)+')')
    config.depth = config.depth - 2
    code('#endif')
    config.depth = config.depth + 2
    code('')
    code('call ops_timers_core(t2)')
    code('')

    #Call user kernel wrapper
    code('call '+name+'_wrap( &')
    for n in range (0, nargs):
      if arg_typ[n] == 'ops_arg_idx':
        code('& idx, &')
      else:
        code('& opsDat'+str(n+1)+'Local, &')
    for n in range (0, nargs):
      if arg_typ[n] != 'ops_arg_idx':
        code('& dat'+str(n+1)+'_base, &')
    code('& start, &')
    code('& end )')
    code('')

    code('call ops_timers_core(t3)')

    config.depth = config.depth - 2
    code('#ifndef OPS_LAZY')
    config.depth = config.depth + 2
    code('call ops_set_dirtybit_host(opsArgArray, '+str(nargs)+')')
    for n in range (0, nargs):
      if arg_typ[n] == 'ops_arg_dat' and (accs[n] == OPS_WRITE or accs[n] == OPS_RW or accs[n] == OPS_INC):
        code('call ops_set_halo_dirtybit3(opsArg'+str(n+1)+',range)')
    config.depth = config.depth - 2
    code('#endif')
    config.depth = config.depth + 2

    code('')
    comm('Timing and data movement')
    code('transfer_total = 0.0_4')
    for n in range (0, nargs):
      if arg_typ[n] == 'ops_arg_dat':
        code('call ops_compute_transfer('+str(NDIM)+', start, end, opsArg'+str(n+1)+',transfer)')
        code('transfer_total = transfer_total + transfer')
    code('call setKernelTime('+str(nk)+',userSubroutine,t3-t2,t2-t1,transfer_total,1)') 

    config.depth = config.depth - 2
    code('end subroutine')

    code('')
    code('#ifdef OPS_LAZY')
    code('subroutine '+name+'_host( userSubroutine, block, dim, range, &')
    for n in range (0, nargs):
      if n == nargs-1:
        code('& opsArg'+str(n+1)+')')
      else:
        code('& opsArg'+str(n+1)+', &')

    config.depth = config.depth + 2
    code('IMPLICIT NONE')
    code('character(kind=c_char,len=*), INTENT(IN), TARGET :: userSubroutine')
    code('type ( ops_block ), INTENT(IN) :: block')
    code('integer(kind=4), INTENT(IN):: dim')
    code('integer(kind=4), DIMENSION(2*dim), INTENT(INOUT), TARGET :: range')
    code('integer(kind=4), DIMENSION(2*dim), TARGET :: range_tmp')
	
    for n in range (0, nargs):
      if arg_typ[n] == 'ops_arg_idx':
        code('type ( ops_arg ), INTENT(IN) :: opsArg'+str(n+1))
      if arg_typ[n] == 'ops_arg_dat':
        code('type ( ops_arg ), INTENT(IN) :: opsArg'+str(n+1))
      elif arg_typ[n] == 'ops_arg_gbl':
        code('type ( ops_arg ), INTENT(IN) :: opsArg'+str(n+1))
    code('type ( ops_arg ), DIMENSION('+str(nargs)+'), TARGET :: opsArgArray')
    code('integer(kind=4) :: n')
    
    code('')
    for n in range (0, nargs):
      code('opsArgArray('+str(n+1)+') = opsArg'+str(n+1))

    code('')
    DO('n','1',str(NDIM))
    code('range_tmp(2*n-1) = range(2*n-1)-1')
    code('range_tmp(2*n) = range(2*n)')
    ENDDO()

    code('')
    text = 'call create_kerneldesc_and_enque(userSubroutine//c_null_char, c_loc(opsArgArray), '
    text = text + f'{nargs}, '
    text = text + f'{nk}, '
    text = text + 'dim, 0, c_loc(range_tmp), block%blockCptr, '
    text = text + f'c_funloc({name}_host_execute))'
    code(text)

    config.depth = config.depth - 2
    code('end subroutine')
    code('#endif')

    code('')
    code('END MODULE')

##########################################################################
#  output individual kernel file
##########################################################################
    try:
      os.makedirs('./MPI')
    except OSError as e:
      if e.errno != errno.EEXIST:
        raise
    fid = open('./MPI/'+name+'_seq_kernel.F90','w')
    date = datetime.datetime.now()
    fid.write('!\n! auto-generated by ops_fortran.py\n!\n')

    fid.write(config.file_text)
    fid.close()
