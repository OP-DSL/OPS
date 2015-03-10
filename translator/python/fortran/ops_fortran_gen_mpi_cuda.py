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

"""
OPS MPI_CUDA code generator for Fortran applications

This routine is called by ops_fortran.py which parses the input files

It produces a file xxx_cuda_kernel.CUF for each kernel

"""

import re
import datetime
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

def ops_fortran_gen_mpi_cuda(master, date, consts, kernels):

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
    reduction_vars = ''
    reduct_op = ''
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
#  generate HEADER
##########################################################################

    code('MODULE '+name.upper()+'_MODULE')
    code('USE OPS_FORTRAN_DECLARATIONS')
    code('USE OPS_FORTRAN_RT_SUPPORT')
    code('')
    code('USE OPS_CONSTANTS')
    code('USE ISO_C_BINDING')
    code('USE CUDAFOR')
    code('')

##########################################################################
#  generate Vars for reductions
##########################################################################
    for n in range(0,nargs):
      if arg_typ[n] == 'ops_arg_gbl':
        comm('Vars for reductions')
        if (accs[n]== OPS_INC or accs[n]== OPS_MIN or accs[n]== OPS_MAX):
          code(typs[n]+', DIMENSION(:), DEVICE, ALLOCATABLE :: reductionArrayDevice_'+str(n+1)+name)
        if ((accs[n]==OPS_READ and dims[n] > 1) or accs[n]==OPS_WRITE):
          code(typs[n]+', DIMENSION(:), DEVICE, ALLOCATABLE :: opGblDat'+str(n+1)+'Device_'+name)

##########################################################################
#  generate MACROS
##########################################################################
    for n in range (0, nargs):
      if arg_typ[n] == 'ops_arg_dat':
        if int(dims[n]) == 1:
          comm('single-dim macros')
          code('INTEGER(KIND=4), constant :: xdim'+str(n+1)+'_'+name)
          code('INTEGER(KIND=4):: xdim'+str(n+1)+'_'+name+'_h  = -1')
          if NDIM==1:
            code('#define OPS_ACC'+str(n+1)+'(x) (x+1)')
          if NDIM==2:
            code('#define OPS_ACC'+str(n+1)+'(x,y) (x+xdim'+str(n+1)+'_'+name+'_'+name+'*(y)+1)')
          if NDIM==3:
            code('#define OPS_ACC'+str(n+1)+'(x,y,z) (x+xdim'+str(n+1)+'_'+name+'_'+name+'*(y)+xdim'+str(n+1)+'_'+name+'_'+name+'*ydim'+str(n+1)+'_'+name+'*(z)+1)')
    code('')

    for n in range (0, nargs):
      if arg_typ[n] == 'ops_arg_dat':
        if int(dims[n]) > 1:
          comm('multi-dim macros')
          code('INTEGER(KIND=4), constant :: xdim'+str(n+1)+'_'+name)
          code('INTEGER(KIND=4):: xdim'+str(n+1)+'_'+name+'_h  = -1')
          if NDIM==1:
            code('#define OPS_ACC_MD'+str(n+1)+'(d,x) ((x)*'+str(dims[n])+'+(d))')
          if NDIM==2:
            code('#define OPS_ACC_MD'+str(n+1)+'(d,x,y) ((x)*'+str(dims[n])+'+(d)+(xdim'+str(n+1)+'_'+name+'*(y)*'+str(dims[n])+'))')
          if NDIM==3:
            code('#define OPS_ACC_MD'+str(n+1)+'(d,x,y,z) ((x)*'+str(dims[n])+'+(d)+(xdim'+str(n+1)+'_'+name+'*(y)*'+str(dims[n])+')+(xdim'+str(n+1)+'_'+name+'*ydim'+str(n+1)+'*(z)*'+str(dims[n])+'))')

    code('')
    code('contains')
    code('')

##########################################################################
#  user kernel subroutine
##########################################################################
    comm('user function')
    fid = open(name2+'_kernel.inc', 'r')
    text = fid.read()
    fid.close()
    text = comment_remover(text)
    text = remove_trailing_w_space(text)
    i = text.find(name)
    if(i < 0):
      print "\n********"
      print "Error: cannot locate user kernel function: "+name+" - Aborting code generation"
      exit(2)

    # need to check accs here - under fortran the
    # parameter vars are declared inside the subroutine
    # for now no check is done

    code('attributes (device) '+text)
    code('')

    for n in range (0, nargs):
      if arg_typ[n] == 'ops_arg_dat':
        if int(dims[n]) == 1:
          code('#undef OPS_ACC'+str(n+1))
    code('')
    for n in range (0, nargs):
      if arg_typ[n] == 'ops_arg_dat':
        if int(dims[n]) > 1:
          code('#undef OPS_ACC_MD'+str(n+1))
    code('')
    code('')


##########################################################################
#  generate kernel wrapper subroutine
##########################################################################
    comm('CUDA kernel function -- wrapper calling user kernel')
    code('attributes (global) subroutine '+name+'_wrap( &')
    for n in range (0, nargs):
      if arg_typ[n] == 'ops_arg_idx':
        code('& idx, &')
      elif arg_typ[n] == 'ops_arg_dat':
        code('& opsDat'+str(n+1)+'Local, &')
      elif arg_typ[n] == 'ops_arg_gbl' and accs[n] <> OPS_READ:
        code('& reductionArrayDevice'+str(n+1)+',   &')
      elif accs[n] == OPS_READ and dims[n]==1:
        code('& opsGblDat'+str(n+1)+'Device_'+name+',   &')

    for n in range (0, nargs):
      if arg_typ[n] == 'ops_arg_dat':
        code('& dat'+str(n+1)+'_base, &')
    if NDIM==1:
      code('& size1, &')
    elif NDIM==2:
      code('& size1, size2, &')
    elif NDIM==3:
      code('& size1, size2, size3, &')
    code('& start, &')
    code('& end )')

    config.depth = config.depth + 2
    code('IMPLICIT NONE')
    for n in range (0, nargs):
      if arg_typ[n] == 'ops_arg_dat' and accs[n] == OPS_READ:
        code(typs[n]+', DEVICE, INTENT(IN) :: opsDat'+str(n+1)+'Local(*)')
        code('integer(4) arg'+str(n+1))
      elif arg_typ[n] == 'ops_arg_dat' and (accs[n] == OPS_WRITE or accs[n] == OPS_RW or accs[n] == OPS_INC):
        code(typs[n]+', DEVICE :: opsDat'+str(n+1)+'Local(*)')
        code('integer(4) arg'+str(n+1))
      elif arg_typ[n] == 'ops_arg_gbl':
        code(typs[n]+' opsDat'+str(n+1)+'Local('+str(dims[n])+')')
      elif arg_typ[n] == 'ops_arg_idx':
        code('integer(4) idx('+str(NDIM)+'),idx_local('+str(NDIM)+')' )

    #vars for reductions
    for n in range (0, nargs):
      if accs[n] == OPS_INC or accs[n] == OPS_MIN or accs[n] == OPS_MAX:
          #if it's a global reduction, then we pass in a reductionArrayDevice
          code(typs[n]+', DIMENSION(:), DEVICE :: reductionArrayDevice'+str(n+1))
          #and additionally we need registers to store contributions, depending on dim:
          if dims[n] == 1:
            code(typs[n]+' :: opsGblDat'+str(n+1)+'Device_'+name)
          else:
            code(typs[n]+', DIMENSION(0:'+dims[n]+'-1) :: opsGblDat'+str(n+1)+'Device_'+name)
      else:
        #if it's not  a global reduction, and multidimensional then we pass in a device array
        if dims[n] == 1:
          if accs[n] == OPS_READ: #if OPS_READ and dim 1, we can pass in by value
            code(typs[n]+', VALUE :: opsGblDat'+str(n+1)+'Device_'+name)

    #vars for arg_idx
    for n in range (0, nargs):
      if arg_typ[n] <> 'ops_arg_idx':
        code('integer dat' + str(n+1)+'_base')
    code('integer(4) start('+str(NDIM)+')')
    code('integer(4) end('+str(NDIM)+')')
    code('integer(4) d')

    if NDIM==1:
      code('integer n_x, size1')
    elif NDIM==2:
      code('integer n_x, n_y, size1, size2')
    elif NDIM==3:
      code('integer n_x, n_y, n_z, size1, size2, size3')
    code('')


    code('')
    if NDIM==3:
      code('n_z = blockDim%z * (blockIdx%z-1) + threadIdx%z')
      code('n_y = blockDim%y * (blockIdx%y-1) + threadIdx%y')
    if NDIM==2:
      code('n_y = blockDim%y * (blockIdx%y-1) + threadIdx%y')
    code('n_x = blockDim%x * (blockIdx%x-1) + threadIdx%x')
    code('')
    if arg_idx:
      code('idx_local(1) = idx(1)+ n_x-1')
      if NDIM==2:
        code('idx_local(2) = idx(2)+ n_y-1')
      if NDIM==3:
        code('idx_local(2) = idx(2}+ n_y-1')
        code('idx_local(3) = idx(3)+ n_z-1')

    for n in range (0, nargs):
      if arg_typ[n] == 'ops_arg_dat':
        if NDIM == 1:
          code('arg'+str(n+1)+' = (n_x-1) * '+str(stride[NDIM*n])+'*'+str(dims[n]))
        elif NDIM == 2:
          code('arg'+str(n+1)+' = (n_x-1) * '+str(stride[NDIM*n])+'*'+str(dims[n])+' + (n_y-1) * '+str(stride[NDIM*n+1])+'*'+str(dims[n])+' * xdim'+str(n+1)+'_'+name)
        elif NDIM==3:
          code('arg'+str(n+1)+' = (n_x-1) * '+str(stride[NDIM*n])+'*'+str(dims[n])+' + (n_y-1) * '+str(stride[NDIM*n+1])+'*'+str(dims[n])+' * xdim'+str(n+1)+'_'+name+' + (n_z-1) * '+str(stride[NDIM*n+2])+'*'+str(dims[n])+' * xdim'+str(n)+'_'+name+' * ydim'+str(n))

    if NDIM==1:
      IF('(n_x_1) < size1')
    if NDIM==2:
      IF('(n_x-1) < size1 .AND. (n_y-1) < size2')
    elif NDIM==3:
      IF('(n_x-1) < size1 .AND. (n_y-1) < size2 .AND. (n_z-1) < size3')

    code('call '+name + '( &')
    indent = config.depth *' '
    line = ''
    for n in range (0, nargs):
      if arg_typ[n] == 'ops_arg_dat':
        if NDIM==1:
          line = line + '& opsDat'+str(n+1)+'Local(dat'+str(n+1)+'_base+arg'+str(n+1)+')'
        elif NDIM==2:
          line = line + '& opsDat'+str(n+1)+'Local(dat'+str(n+1)+'_base+arg'+str(n+1)+')'
        elif NDIM==3:
          line = line + '& opsDat'+str(n+1)+'Local(dat'+str(n+1)+'_base+arg'+str(n+1)+')'
      elif arg_typ[n] == 'ops_arg_gbl':
        line = line + '& opsDat'+str(n+1)+'Local'
      elif arg_typ[n] == 'ops_arg_idx':
        line = line + '& idx_local'

      if n == nargs-1:
        line = line + ' )'
      else:
        line = line + ', &\n'+indent

    code(line)
    ENDIF()

    code('')
    #reduction across blocks
    for n in range(0,nargs):
      if arg_typ[n] == 'ops_arg_gbl' and (accs[n] == OPS_INC or accs[n] == OPS_MIN or accs[n] == OPS_MAX):
        if 'real' in typs[n].lower():
          if dims[n].isdigit() and int(dims[n])==1:
            code('!call ReductionFloat8(reductionArrayDevice'+str(n+1)+'(blockIdx%x - 1 + 1:),opsGblDat'+str(n+1)+'Device_'+name+',0)')
          else:
            code('!call ReductionFloat8Mdim(reductionArrayDevice'+str(n+1)+'((blockIdx%x - 1)*('+dims[n]+') + 1:),opsGblDat'+str(n+1)+'Device_'+name+',0,'+dims[n]+')')
        elif 'integer' in typs[n].lower():
          if dims[n].isdigit() and int(dims[n])==1:
            code('!call ReductionInt4(reductionArrayDevice'+str(n+1)+'(blockIdx%x - 1 + 1:),opsGblDat'+str(n+1)+'Device_'+name+',0)')
          else:
            code('!call ReductionInt4Mdim(reductionArrayDevice'+str(n+1)+'((blockIdx%x - 1)*('+dims[n]+') + 1:),opsGblDat'+str(n+1)+'Device_'+name+',0,'+dims[n]+')')
    code('')


    config.depth = config.depth - 2
    code('end subroutine')

#########################################################################
#  host subroutine
##########################################################################

    code('')
    comm('host subroutine')
    code('attributes (host) subroutine '+name+'_host( userSubroutine, block, dim, range, &')
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
    code('integer(kind=4)   , DIMENSION(dim), INTENT(IN) :: range')
    code('')
    for n in range (0, nargs):
      if arg_typ[n] == 'ops_arg_idx':
        code('type ( ops_arg )  , INTENT(IN) :: opsArg'+str(n+1))
        code('')
      if arg_typ[n] == 'ops_arg_dat':
        code('type ( ops_arg )  , INTENT(IN) :: opsArg'+str(n+1))
        code(typs[n]+', DIMENSION(:), DEVICE, ALLOCATABLE  :: opsDat'+str(n+1)+'Local')
        code('integer(kind=4) :: opsDat'+str(n+1)+'Cardinality')
        code('integer(kind=4), POINTER, DIMENSION(:)  :: dat'+str(n+1)+'_size')
        code('integer(kind=4), DEVICE  :: dat'+str(n+1)+'_base')
        code('INTEGER(KIND=4) :: xdim'+str(n+1))
        if int(dims[n]) > 1:
          code('INTEGER(KIND=4) :: multi_d'+str(n+1))
        if NDIM==2:
          code('integer ydim'+str(n+1))
        elif NDIM==2:
          code('integer ydim'+str(n+1)+', zdim'+str(n+1))
        code('')

    for n in range(0,nargs):
      if arg_typ[n] == 'ops_arg_gbl':
        code('type ( ops_arg )  , INTENT(IN) :: opsArg'+str(n+1))
        code('integer(kind=4) :: opsDat'+str(n+1)+'Cardinality')
        if accs[n] == OPS_WRITE or dims[n] > 1:
          code(typs[n]+', DIMENSION(:), POINTER :: opsDat'+str(n+1)+'Host')
        else:
          code(typs[n]+', POINTER :: opsDat'+str(n+1)+'Host')
        if (accs[n] == OPS_INC or accs[n] == OPS_MAX or accs[n] == OPS_MIN):
          code(typs[n]+', DIMENSION(:), ALLOCATABLE :: reductionArrayHost'+str(n+1))
          code('INTEGER(kind=4) :: reductionCardinality'+str(n+1))
    code('')


    if NDIM==1:
      code('integer, DEVICE :: x_size')
    elif NDIM==2:
      code('integer, DEVICE :: x_size, y_size')
    elif NDIM==2:
      code('integer, DEVICE :: x_size, y_size, z_size')
    code('integer start('+str(NDIM)+')')
    code('integer, DEVICE :: start_d('+str(NDIM)+')')
    code('integer end('+str(NDIM)+')')
    code('integer, DEVICE :: end_d('+str(NDIM)+')')
    if arg_idx == 1:
      code('integer, DEVICE :: idx('+str(NDIM)+')')
    code('integer(kind=4) :: n')
    code('integer(kind=4) :: i10')
    code('integer(kind=4) :: i20')

    code('integer(kind=4) :: blocksPerGrid')
    code('integer(kind=4) :: dynamicSharedMemorySize')


    code('')
    comm('cuda grid and thread block sizes')
    code('type(dim3) :: grid, tblock')
    code('')
    code('type ( ops_arg ) , DIMENSION('+str(nargs)+') :: opsArgArray')
    code('')

    for n in range (0, nargs):
      code('opsArgArray('+str(n+1)+') = opsArg'+str(n+1))
    code('')

    config.depth = config.depth - 2
    code('#ifdef OPS_MPI')
    config.depth = config.depth + 2
    code('call getRange(block, start, end, range)')
    DO('n','1',str(NDIM))
    code('start_d(n) = start(n)')
    code('end_d(n) = end(n)')
    ENDDO()
    config.depth = config.depth - 2
    code('#else')
    config.depth = config.depth + 2
    DO('n','1',str(NDIM))
    code('start(n) = range(2*n-1)')
    code('end(n) = range(2*n)')
    code('start_d(n) = range(2*n-1)')
    code('end_d(n) = range(2*n)')
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


    code('')
    code('x_size = MAX(0,end(1)-start(1)+1)')
    if NDIM==2:
      code('y_size = MAX(0,end(2)-start(2)+1)')
    if NDIM==3:
      code('y_size = MAX(0,end(2)-start(2)+1)')
      code('z_size = MAX(0,end(3)-start(3)+1)')
    code('')

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
        if int(dims[n]) <> 1:
          code('multi_d'+str(n+1)+' = getDatDimFromOpsArg(opsArg'+str(n+1)+') ! dimension of the dat')
          code('dat'+str(n+1)+'_base = getDatBaseFromOpsArg'+str(NDIM)+'D(opsArg'+str(n+1)+',start,multi_d'+str(n+1)+')')
        else:
          code('dat'+str(n+1)+'_base = getDatBaseFromOpsArg'+str(NDIM)+'D(opsArg'+str(n+1)+',start,1)')
        code('call c_f_pointer(opsArg'+str(n+1)+'%data_d,opsDat'+str(n+1)+'Local,(/opsDat'+str(n+1)+'Cardinality/))')

      if arg_typ[n] == 'ops_arg_gbl':
        if accs[n] == OPS_WRITE or int(dims[n])>1:
          code('call c_f_pointer(opsArg'+str(n+1)+'%data,opsDat'+str(n+1)+'Host,(/opsDat'+str(n+1)+'Cardinality/))')
        else:
          code('call c_f_pointer(opsArg'+str(n+1)+'%data,opsDat'+str(n+1)+'Host)')
      code('')

    #NEED TO COPY CONSTANTS TO Symbol
    condition = ''
    for n in range (0, nargs):
      if arg_typ[n] == 'ops_arg_dat':
        condition = condition + '(xdim'+str(n+1)+' .NE. xdim'+str(n+1)+'_'+name+'_h) .OR. '
        if NDIM==3:
          condition = condition + '(ydim'+str(n+1)+' .NE. ydim'+str(n+1)+'_'+name+'_h) .OR. '
    condition = condition[:-5]

    IF(condition)
    for n in range (0, nargs):
      if arg_typ[n] == 'ops_arg_dat':
        code('xdim'+str(n+1)+'_'+name+' = xdim'+str(n+1))
        code('xdim'+str(n+1)+'_'+name+'_h = xdim'+str(n+1))
        if NDIM==3:
          code('ydim'+str(n+1)+'_'+name+' = ydim'+str(n+1))
          code('ydim'+str(n+1)+'_'+name+'_h = ydim'+str(n+1))
    ENDIF()


    #set up CUDA grid and thread blocks for kernel call
    code('')
    if NDIM==1:
      code('grid = dim3( (x_size-1)/getOPS_block_size_x()+ 1, 1, 1)')
    if NDIM==2:
      code('grid = dim3( (x_size-1)/getOPS_block_size_x()+ 1, (y_size-1)/getOPS_block_size_y() + 1, 1)')
    if NDIM==3:
      code('grid = dim3( (x_size-1)/getOPS_block_size_x()+ 1, (y_size-1)/getOPS_block_size_y() + 1, z_size)')

    if NDIM>1:
      code('tblock = dim3(getOPS_block_size_x(),getOPS_block_size_y(),1)')
    else:
      code('tblock = dim3(getOPS_block_size_x(),1,1)')
    code('')

    #setup reduction variables and shared memory for reduction
    code('blocksPerGrid = 200')
    code('dynamicSharedMemorySize = 100*8')
    for n in range(0,nargs):
      if arg_typ[n] == 'ops_arg_gbl' and (accs[n] == OPS_INC or accs[n] == OPS_MAX or accs[n] == OPS_MIN):
        code('reductionCardinality'+str(n+1)+' = blocksPerGrid * 1')
        code('allocate( reductionArrayHost'+str(n+1)+'(reductionCardinality'+str(n+1)+'* ('+dims[n]+')) )')
        IF ('.not. allocated(reductionArrayDevice_'+str(n+1)+name+')')
        code('allocate( reductionArrayDevice_'+str(n+1)+name+'(reductionCardinality'+str(n+1)+'* ('+dims[n]+')) )')
        ENDIF()
        code('')
        DO('i10','0','reductionCardinality'+str(n+1)+'')
        if int(dims[n]) == 1:
          code('reductionArrayHost'+str(n+1)+'(i10+1) = 0.0')
        else:
          code('reductionArrayHost'+str(n+1)+'(i10 * ('+dims[n]+') + 1 : i10 * ('+dims[n]+') + ('+dims[n]+')) = 0.0')
        ENDDO()
        code('')
        code('reductionArrayDevice_'+str(n+1)+name+' = reductionArrayHost'+str(n+1)+'')


    #halo exchange
    code('')
    code('call ops_H_D_exchanges_device(opsArgArray,'+str(nargs)+')')
    code('call ops_halo_exchanges(opsArgArray,'+str(nargs)+',range)')
    code('')

    #Call cuda kernel  - i.e. the wrapper calling the user kernel
    if reduction:
      code('call '+name+'_wrap <<<grid,tblock,dynamicSharedMemorySize>>> (&')
    else:
      code('call '+name+'_wrap <<<grid,tblock>>> (&')

    for n in range (0, nargs):
      if arg_typ[n] == 'ops_arg_idx':
        code('& idx, &')
      elif arg_typ[n] == 'ops_arg_dat':
        code('& opsDat'+str(n+1)+'Local, &')
      elif arg_typ[n] == 'ops_arg_gbl':
        if accs[n] == OPS_INC or accs[n] == OPS_MIN or accs[n] == OPS_MAX:
          code('& reductionArrayDevice_'+str(n+1)+name+', &')
        elif accs[n] == OPS_READ and dims[n]==1:
          code('& opsDat'+str(n+1)+'Host, &')

    for n in range (0, nargs):
      if arg_typ[n] == 'ops_arg_dat' :
        code('& dat'+str(n+1)+'_base, &')
    if NDIM==1:
      code('& x_size, &')
    elif NDIM==2:
      code('& x_size, y_size, &')
    elif NDIM==3:
      code('& x_size, y_size, z_size, &')
    code('& start_d, &')
    code('& end_d )')
    code('')

    #
    # Complete Reduction Operation by moving data onto host
    # and reducing over blocks
    #
    for n in range(0,nargs):
      if arg_typ[n] == 'ops_arg_gbl' and accs[n] == OPS_WRITE:
        code('opsDat'+str(n+1)+'Host(1:opsArg'+str(n+1)+'%dim) = opGblDat'+str(n+1)+'Device_'+name+'(1:opsArg'+str(n+1)+'%dim)')

    if reduction:
      #reductions
      for n in range(0,nargs):
        if arg_typ[n] == 'ops_arg_gbl' and accs[n] == OPS_INC:
          code('reductionArrayHost'+str(n+1)+' = reductionArrayDevice_'+str(n+1)+name+'')
          code('')
          DO('i10','0','reductionCardinality'+str(n+1)+'')
          if dims[n].isdigit() and int(dims[n]) == 1:
            code('opsDat'+str(n+1)+'Host = opsDat'+str(n+1)+'Host + reductionArrayHost'+str(n+1)+'(i10+1)')
          else:
            code('opsDat'+str(n+1)+'Host(1:'+dims[n]+') = opsDat'+str(n+1)+'Host(1:'+dims[n]+') + reductionArrayHost'+str(n+1)+'(i10 * ('+dims[n]+') + 1 : i10 * ('+dims[n]+') + ('+dims[n]+'))')
          ENDDO()
          code('')
          code('deallocate( reductionArrayHost'+str(n+1)+' )')



    code('call ops_set_dirtybit_device(opsArgArray, '+str(nargs)+')')
    for n in range (0, nargs):
      if arg_typ[n] == 'ops_arg_dat' and (accs[n] == OPS_WRITE or accs[n] == OPS_RW or accs[n] == OPS_INC):
        code('call ops_set_halo_dirtybit3(opsArg'+str(n+1)+',range)')
    code('')


    config.depth = config.depth - 2
    code('end subroutine')
    code('END MODULE')

##########################################################################
#  output individual kernel file
##########################################################################
    if not os.path.exists('./CUDA'):
      os.makedirs('./CUDA')
    fid = open('./CUDA/'+name+'_cuda_kernel.CUF','w')
    date = datetime.datetime.now()
    fid.write('!\n! auto-generated by ops_fortran.py\n!\n')

    fid.write(config.file_text)
    fid.close()