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
#  OPS MPI_CUDA code generator for Fortran applications
#
#  This routine is called by ops_fortran.py which parses the input files
#
#  It produces a file xxx_cuda_kernel.CUF for each kernel
#

"""
OPS MPI_CUDA code generator for Fortran applications

This routine is called by ops_fortran.py which parses the input files

It produces a file xxx_cuda_kernel.CUF for each kernel

"""

import re
import errno
import os

import util_fortran
import config
from datetime import datetime

import threading

comment_remover = util_fortran.comment_remover
remove_trailing_w_space = util_fortran.remove_trailing_w_space
comm = util_fortran.comm
code = util_fortran.code

DO = util_fortran.DO
DOWHILE = util_fortran.DOWHILE
ENDDO = util_fortran.ENDDO
IF = util_fortran.IF
ENDIF = util_fortran.ENDIF


def replace_consts(text):
  if not os.path.isfile("constants_list.txt"):
    return text

  with open("constants_list.txt", 'r') as f:
    words_list = f.read().splitlines()

  regex_pattern = r'\b(' + '|'.join(words_list) + r')\b'
  replacement_pattern = r'\g<1>_opsconstant'
  text = re.sub(regex_pattern, replacement_pattern, text)

  return text

def ops_fortran_gen_mpi_cuda(master, consts, kernels, soa_set):

    threads_list = []

    for nk, cur_kernel in enumerate(kernels):
        ops_fortran_gen_mpi_cuda_process(consts, cur_kernel, soa_set, nk)
#        thread = threading.Thread(target=ops_fortran_gen_mpi_cuda_process, args=(date, consts, cur_kernel, soa_set, nk))
#        threads_list.append(thread)
#        thread.start()

#    for thread in threads_list:
#        thread.join()


def ops_fortran_gen_mpi_cuda_process(consts, cur_kernel, soa_set, nk):
    global funlist
    OPS_GBL   = 2;

    OPS_READ = 1;  OPS_WRITE = 2;  OPS_RW  = 3;
    OPS_INC  = 4;  OPS_MAX   = 5;  OPS_MIN = 6;

    accsstring = ['OPS_READ','OPS_WRITE','OPS_RW','OPS_INC','OPS_MAX','OPS_MIN' ]


##########################################################################
#  create new kernel file
##########################################################################

    arg_typ  = cur_kernel['arg_type']
    name  = cur_kernel['name']
    nargs = cur_kernel['nargs']
    dim   = cur_kernel['dim']
    dims  = cur_kernel['dims']
    stens = cur_kernel['stens']
    var   = cur_kernel['var']
    accs  = cur_kernel['accs']
    typs  = cur_kernel['typs']
    NDIM = int(dim)
    #parse stencil to locate strided access
    stride = [1] * nargs * NDIM

#    print("kernel name:"+name)

    reduction = 0
    reduction_vars = ''
    reduct_op = ''
    reduct_mdim = 0
    reduct_1dim = 0
    gbls_mdim = 0
    for n in range (0, nargs):
      if arg_typ[n] == 'ops_arg_gbl' and accs[n] != OPS_READ:
        reduction = 1
        if (not dims[n].isdigit()) or dims[n] != '1':
          reduct_mdim = 1
        else:
          reduct_1dim = 1
      if arg_typ[n] == 'ops_arg_gbl' and accs[n] == OPS_READ and dims[n] != '1':
        gbls_mdim = 1

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
        #comm('Vars for reductions')
        if (accs[n]== OPS_INC or accs[n]== OPS_MIN or accs[n]== OPS_MAX):
          code(typs[n]+', dimension(:), device, allocatable :: reductionArrayDevice'+str(n+1)+'_'+name)
        if ((accs[n]==OPS_READ and ((not dims[n].isdigit()) or dims[n] != '1')) or accs[n]==OPS_WRITE):
          code(typs[n]+', dimension(:), device, allocatable :: opGblDat'+str(n+1)+'Device_'+name)

##########################################################################
#  generate MACROS
##########################################################################
    for n in range (0, nargs):
      if arg_typ[n] == 'ops_arg_dat':
        if dims[n] == '1':
          #comm('single-dim macros')
          code('integer(4), constant :: xdim'+str(n+1)+'_'+name)
          code('integer(4):: xdim'+str(n+1)+'_'+name+'_h  = -1')
          if NDIM==1:
            code('#define OPS_ACC'+str(n+1)+'(x) (x+1)')
          if NDIM==2:
            code('#define OPS_ACC'+str(n+1)+'(x,y) (x+xdim'+str(n+1)+'_'+name+'*(y)+1)')
          if NDIM==3:
            code('integer(4), constant :: ydim'+str(n+1)+'_'+name)
            code('integer(4):: ydim'+str(n+1)+'_'+name+'_h  = -1')
            code('#define OPS_ACC'+str(n+1)+'(x,y,z) (x+xdim'+str(n+1)+'_'+name+'*(y)+xdim'+str(n+1)+'_'+name+'*ydim'+str(n+1)+'_'+name+'*(z)+1)')
        code('')

    for n in range (0, nargs):
      if arg_typ[n] == 'ops_arg_dat':
        if dims[n] != '1':
          #comm('multi-dim macros')
          code('integer(4), constant :: xdim'+str(n+1)+'_'+name)
          code('integer(4):: xdim'+str(n+1)+'_'+name+'_h  = -1')
          if NDIM==1:
            code('#define OPS_ACC_MD'+str(n+1)+'(d,x) ((x)*'+str(dims[n])+'+(d))')
          if NDIM==2:
            code('#define OPS_ACC_MD'+str(n+1)+'(d,x,y) ((x)*'+str(dims[n])+'+(d)+(xdim'+str(n+1)+'_'+name+'*(y)*'+str(dims[n])+'))')
          if NDIM==3:
            code('integer(KIND=4), constant :: ydim'+str(n+1)+'_'+name)
            code('integer(KIND=4):: ydim'+str(n+1)+'_'+name+'_h  = -1')
            code('#define OPS_ACC_MD'+str(n+1)+'(d,x,y,z) ((x)*'+str(dims[n])+'+(d)+(xdim'+str(n+1)+'_'+name+'*(y)*'+str(dims[n])+')+(xdim'+str(n+1)+'_'+name+'*ydim'+str(n+1)+'_'+name+'*(z)*'+str(dims[n])+'))')
        code('')

    code('contains')
    code('')

##########################################################################
#  Reduction kernel function - if an OP_GBL exists
##########################################################################
    if reduct_1dim:
      comm('Reduction cuda kernel')
      code('attributes (device) SUBROUTINE ReductionFloat8(sharedDouble8, reductionResult,inputValue,reductionOperation)')
      config.depth = config.depth +2;
      code('REAL(8), dimension(:), DEVICE :: reductionResult')
      code('REAL(8) :: inputValue')
      code('integer(4), VALUE :: reductionOperation')
      code('REAL(8), dimension(0:*) :: sharedDouble8')
      code('integer(4) :: i1')
      code('integer(4) :: threadID')
      code('threadID = (threadIdx%y-1)*blockDim%x + (threadIdx%x - 1)')
      code('i1 = ishft(blockDim%x*blockDim%y,-1)')
      code('CALL syncthreads()')
      code('sharedDouble8(threadID) = inputValue')

      DOWHILE('i1 > 0')
      code('CALL syncthreads()')
      IF('threadID < i1')
      code('SELECT CASE(reductionOperation)')
      code('CASE (0)')
      code('sharedDouble8(threadID) = sharedDouble8(threadID) + sharedDouble8(threadID + i1)')
      code('CASE (1)')
      IF('sharedDouble8(threadID + i1) < sharedDouble8(threadID)')
      code('sharedDouble8(threadID) = sharedDouble8(threadID + i1)')
      ENDIF()
      code('CASE (2)')
      IF('sharedDouble8(threadID + i1) > sharedDouble8(threadID)')
      code('sharedDouble8(threadID) = sharedDouble8(threadID + i1)')
      ENDIF()
      code('END SELECT')
      ENDIF()
      code('i1 = ishft(i1,-1)')
      ENDDO()

      code('CALL syncthreads()')

      IF('threadID .EQ. 0')
      code('SELECT CASE(reductionOperation)')
      code('CASE (0)')
      code('reductionResult(1) = reductionResult(1) + sharedDouble8(0)')
      code('CASE (1)')
      IF('sharedDouble8(0) < reductionResult(1)')
      code('reductionResult(1) = sharedDouble8(0)')
      ENDIF()
      code('CASE (2)')
      IF('sharedDouble8(0) > reductionResult(1)')
      code('reductionResult(1) = sharedDouble8(0)')
      ENDIF()
      code('END SELECT')
      ENDIF()

      code('CALL syncthreads()')
      code('END SUBROUTINE')
      code('')

      code('attributes (device) SUBROUTINE ReductionInt4(sharedInt4, reductionResult,inputValue,reductionOperation)')
      code('integer(4), dimension(:), DEVICE :: reductionResult')
      code('integer(4) :: inputValue')
      code('integer(4), VALUE :: reductionOperation')
      code('integer(4), dimension(0:*) :: sharedInt4')
      code('integer(4) :: i1')
      code('integer(4) :: threadID')
      code('threadID = (threadIdx%y-1)*blockDim%x + (threadIdx%x - 1)')
      code('i1 = ishft(blockDim%x*blockDim%y,-1)')
      code('CALL syncthreads()')
      code('sharedInt4(threadID) = inputValue')

      DOWHILE('i1 > 0')
      code('CALL syncthreads()')
      IF('threadID < i1')
      code('SELECT CASE(reductionOperation)')
      code('CASE (0)')
      code('sharedInt4(threadID) = sharedInt4(threadID) + sharedInt4(threadID + i1)')
      code('CASE (1)')
      IF('sharedInt4(threadID + i1) < sharedInt4(threadID)')
      code('sharedInt4(threadID) = sharedInt4(threadID + i1)')
      ENDIF()
      code('CASE (2)')
      IF('sharedInt4(threadID + i1) > sharedInt4(threadID)')
      code('sharedInt4(threadID) = sharedInt4(threadID + i1)')
      ENDIF()
      code('END SELECT')
      ENDIF()
      code('i1 = ishft(i1,-1)')
      ENDDO()

      code('CALL syncthreads()')

      IF('threadID .EQ. 0')
      code('SELECT CASE(reductionOperation)')
      code('CASE (0)')
      code('reductionResult(1) = reductionResult(1) + sharedInt4(0)')
      code('CASE (1)')
      IF('sharedInt4(0) < reductionResult(1)')
      code('reductionResult(1) = sharedInt4(0)')
      ENDIF()
      code('CASE (2)')
      IF('sharedInt4(0) > reductionResult(1)')
      code('reductionResult(1) = sharedInt4(0)')
      ENDIF()
      code('END SELECT')
      ENDIF()

      code('CALL syncthreads()')
      config.depth = config.depth - 2;
      code('END SUBROUTINE')
      code('')

    if reduct_mdim:
      comm('Multidimensional reduction cuda kernel')
      code('attributes (device) SUBROUTINE ReductionFloat8Mdim(sharedDouble8, reductionResult,inputValue,reductionOperation,dim)')
      config.depth = config.depth +2;
      code('REAL(8), dimension(:), DEVICE :: reductionResult')
      code('REAL(8), dimension(:) :: inputValue')
      code('integer(4), VALUE :: reductionOperation')
      code('integer(4), VALUE :: dim')
      code('REAL(8), dimension(0:*) :: sharedDouble8')
      code('integer(4) :: i1')
      code('integer(4) :: d')
      code('integer(4) :: threadID')
      code('threadID = (threadIdx%y-1)*blockDim%x + (threadIdx%x - 1)')
      code('i1 = ishft(blockDim%x*blockDim%y,-1)')
      code('CALL syncthreads()')
      code('sharedDouble8(threadID*dim:threadID*dim+dim-1) = inputValue(1:dim)')

      DOWHILE('i1 > 0')
      code('CALL syncthreads()')
      IF('threadID < i1')
      code('SELECT CASE(reductionOperation)')
      code('CASE (0)') #inc
      DO('i2','0','dim-1')
      code('sharedDouble8(threadID*dim + i2) = sharedDouble8(threadID*dim + i2) + sharedDouble8((threadID + i1)*dim + i2)')
      ENDDO()
      code('CASE (1)')#max
      DO('i2','0','dim-1')
      IF('sharedDouble8(threadID*dim + i2) < sharedDouble8((threadID + i1)*dim + i2)')
      code('sharedDouble8(threadID*dim + i2) = sharedDouble8((threadID + i1)*dim + i2)')
      ENDIF()
      ENDDO()
      code('CASE (2)')#min
      DO('i2','0','dim-1')
      IF('sharedDouble8(threadID*dim + i2) < sharedDouble8((threadID + i1)*dim + i2)')
      code('sharedDouble8(threadID*dim + i2) = sharedDouble8((threadID + i1)*dim + i2)')
      ENDIF()
      ENDDO()
      code('END SELECT')
      ENDIF()
      code('i1 = ishft(i1,-1)')
      ENDDO()

      code('CALL syncthreads()')
      IF('threadID .EQ. 0')
      code('SELECT CASE(reductionOperation)')
      code('CASE (0)')#inc
      code('reductionResult(1:dim) = reductionResult(1:dim) + sharedDouble8(0:dim-1)')
      code('CASE (1)')#max
      DO('i2','1','dim')
      IF('reductionResult(i2) < sharedDouble8(i2-1)')
      code('reductionResult(i2) = sharedDouble8(i2-1)')
      ENDIF()
      ENDDO()
      code('CASE (2)')#min
      DO('i2','1','dim')
      IF('reductionResult(i2) > sharedDouble8(i2-1)')
      code('reductionResult(i2) = sharedDouble8(i2-1)')
      ENDIF()
      ENDDO()
      code('END SELECT')
      ENDIF()

      code('CALL syncthreads()')
      config.depth = config.depth - 2;
      code('END SUBROUTINE')
      code('')

      comm('Multidimensional reduction cuda kernel')
      code('attributes (device) SUBROUTINE ReductionInt4Mdim(sharedInt4, reductionResult,inputValue,reductionOperation,dim)')
      config.depth = config.depth +2;
      code('integer(4), dimension(:), DEVICE :: reductionResult')
      code('integer(4), dimension(:) :: inputValue')
      code('integer(4), VALUE :: reductionOperation')
      code('integer(4), VALUE :: dim')
      code('integer(4), dimension(0:*) :: sharedInt4')
      code('integer(4) :: i1')
      code('integer(4) :: d')
      code('integer(4) :: threadID')
      code('threadID = (threadIdx%y-1)*blockDim%x + (threadIdx%x - 1)')
      code('i1 = ishft(blockDim%x*blockDim%y,-1)')
      code('CALL syncthreads()')
      code('sharedInt4(threadID*dim:threadID*dim+dim-1) = inputValue(1:dim)')

      DOWHILE('i1 > 0')
      code('CALL syncthreads()')
      IF('threadID < i1')
      code('SELECT CASE(reductionOperation)')
      code('CASE (0)') #inc
      DO('i2','0','dim-1')
      code('sharedInt4(threadID*dim + i2) = sharedInt4(threadID*dim + i2) + sharedInt4((threadID + i1)*dim + i2)')
      ENDDO()
      code('CASE (1)')#max
      DO('i2','0','dim-1')
      IF('sharedInt4(threadID*dim + i2) < sharedInt4((threadID + i1)*dim + i2)')
      code('sharedInt4(threadID*dim + i2) = sharedInt4((threadID + i1)*dim + i2)')
      ENDIF()
      ENDDO()
      code('CASE (2)')#min
      DO('i2','0','dim-1')
      IF('sharedInt4(threadID*dim + i2) < sharedInt4((threadID + i1)*dim + i2)')
      code('sharedInt4(threadID*dim + i2) = sharedInt4((threadID + i1)*dim + i2)')
      ENDIF()
      ENDDO()
      code('END SELECT')
      ENDIF()
      code('i1 = ishft(i1,-1)')
      ENDDO()

      code('CALL syncthreads()')
      IF('threadID .EQ. 0')
      code('SELECT CASE(reductionOperation)')
      code('CASE (0)')#inc
      code('reductionResult(1:dim) = reductionResult(1:dim) + sharedInt4(0:dim-1)')
      code('CASE (1)')#max
      DO('i2','1','dim')
      IF('reductionResult(i2) < sharedInt4(i2-1)')
      code('reductionResult(i2) = sharedInt4(i2-1)')
      ENDIF()
      ENDDO()
      code('CASE (2)')#min
      DO('i2','1','dim')
      IF('reductionResult(i2) > sharedInt4(i2-1)')
      code('reductionResult(i2) = sharedInt4(i2-1)')
      ENDIF()
      ENDDO()
      code('END SELECT')
      ENDIF()

      code('CALL syncthreads()')
      config.depth = config.depth - 2;
      code('END SUBROUTINE')
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
      print("\n********")
      print(("Error: cannot locate user kernel function: "+name+" - Aborting code generation"))
      exit(2)

    # need to check accs here - under fortran the
    # parameter vars are declared inside the subroutine
    # for now no check is done

    text = replace_consts(text)
    
    #find subroutine calls
    funlist = [name.lower()]
    req_kernel = util_fortran.find_kernel_routine(text, name)
    if len(req_kernel) != 0:
      fun = name.lower()
      regex = re.compile('\\b'+fun+'\\b',re.I)
      req_kernel = regex.sub(fun+'_gpu',req_kernel)
      code('attributes (device) '+req_kernel.strip())

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
      elif arg_typ[n] == 'ops_arg_gbl' and accs[n] != OPS_READ:
        code('& reductionArrayDevice'+str(n+1)+',   &')
      elif accs[n] == OPS_READ:
        code('& opsGblDat'+str(n+1)+'Device,   &')

    for n in range (0, nargs):
      if arg_typ[n] == 'ops_arg_dat':
        code('& dat'+str(n+1)+'_base, &')
    if NDIM==1:
      code('& size1 )')
    elif NDIM==2:
      code('& size1, size2 )')
    elif NDIM==3:
      code('& size1, size2, size3 )')

    config.depth = config.depth + 2
    code('')
    code('IMPLICIT NONE')
    code('')

    for n in range (0, nargs):
      if arg_typ[n] == 'ops_arg_dat' and accs[n] == OPS_READ:
        code(typs[n]+', device, dimension(*), intent(in)    :: opsDat'+str(n+1)+'Local')
        code('integer(4) :: arg'+str(n+1))
      elif arg_typ[n] == 'ops_arg_dat' and (accs[n] == OPS_WRITE or accs[n] == OPS_RW or accs[n] == OPS_INC):
        code(typs[n]+', device, dimension(*), intent(inout) :: opsDat'+str(n+1)+'Local(*)')
        code('integer(4) :: arg'+str(n+1))
      elif arg_typ[n] == 'ops_arg_idx':
        code('integer(4), dimension('+str(NDIM)+'), intent(in) :: idx')
        code('integer(4), dimension('+str(NDIM)+')             :: idx_local')

    #vars for reductions
    for n in range (0, nargs):
      if arg_typ[n] == 'ops_arg_gbl':
        if accs[n] == OPS_INC or accs[n] == OPS_MIN or accs[n] == OPS_MAX:
          #if it's a global reduction, then we pass in a reductionArrayDevice
          code(typs[n]+', dimension(:), device :: reductionArrayDevice'+str(n+1))
          #and additionally we need registers to store contributions, depending on dim:
          if dims[n].isdigit() and dims[n] == '1':
            code(typs[n]+' :: opsGblDat'+str(n+1)+'Device')
          else:
            code(typs[n]+', dimension('+dims[n]+') :: opsGblDat'+str(n+1)+'Device')

          code(typs[n]+', dimension(0:*), shared :: sharedMem')
        else:
          #if it's not  a global reduction, and multidimensional then we pass in a device array
          if accs[n] == OPS_READ: #if OPS_READ and dim 1, we can pass in by value
            if dims[n] == '1':
              code(typs[n]+', value :: opsGblDat'+str(n+1)+'Device')
            else:
              code(typs[n]+', device :: opsGblDat'+str(n+1)+'Device(:)')
#              code(typs[n]+' :: opsGblDat'+str(n+1)+'Device')

    #vars for arg_idx
    for n in range (0, nargs):
      if arg_typ[n] == 'ops_arg_dat':
        code('integer(4), value :: dat' + str(n+1)+'_base')

    if NDIM==1:
      code('integer(4), value :: size1')
      code('integer(4)        :: n_x')
    elif NDIM==2:
      code('integer(4), value :: size1, size2')
      code('integer(4)        :: n_x, n_y')
    elif NDIM==3:
      code('integer(4), value :: size1, size2, size3')
      code('integer(4)        :: n_x, n_y, n_z')
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
        code('idx_local(2) = idx(2)+ n_y-1')
        code('idx_local(3) = idx(3)+ n_z-1')

    for n in range (0, nargs):
      if arg_typ[n] == 'ops_arg_dat':
        if NDIM == 1:
          code('arg'+str(n+1)+' = (n_x-1) * '+str(stride[NDIM*n])+'*'+str(dims[n]))
        elif NDIM == 2:
          code('arg'+str(n+1)+' = (n_x-1) * '+str(stride[NDIM*n])+'*'+str(dims[n])+' + (n_y-1) * '+str(stride[NDIM*n+1])+'*'+str(dims[n])+' * xdim'+str(n+1)+'_'+name)
        elif NDIM==3:
          code('arg'+str(n+1)+' = (n_x-1) * '+str(stride[NDIM*n])+'*'+str(dims[n])+' + (n_y-1) * '+str(stride[NDIM*n+1])+'*'+str(dims[n])+' * xdim'+str(n+1)+'_'+name+' + (n_z-1) * '+str(stride[NDIM*n+2])+'*'+str(dims[n])+' * xdim'+str(n+1)+'_'+name+' * ydim'+str(n+1)+'_'+name)

    #initialize local reduction variables depending on the operation
    for n in range (0, nargs):
      if arg_typ[n] == 'ops_arg_gbl':
        if accs[n] == OPS_INC:
          code('opsGblDat'+str(n+1)+'Device = 0.0_8')
        if accs[n] == OPS_MIN:
          code('opsGblDat'+str(n+1)+'Device = HUGE(opsGblDat'+str(n+1)+'Device)')
        if accs[n] == OPS_MAX:
          code('opsGblDat'+str(n+1)+'Device = -1.0_8*HUGE(opsGblDat'+str(n+1)+'Device)')

    if NDIM==1:
      IF('(n_x-1) < size1')
    if NDIM==2:
      IF('(n_x-1) < size1 .AND. (n_y-1) < size2')
    elif NDIM==3:
      IF('(n_x-1) < size1 .AND. (n_y-1) < size2 .AND. (n_z-1) < size3')

    code('')
    code('call '+name + '_gpu( &')
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
        if dims[n] != '1':# and not 'reduction' in var[n]):
          line =line + '& opsGblDat'+str(n+1)+'Device(1)'
        else:
          line =line + '& opsGblDat'+str(n+1)+'Device'
      elif arg_typ[n] == 'ops_arg_idx':
        line = line + '& idx_local'

      if n == nargs-1:
        line = line + ' )'
      else:
        line = line + ', &\n'+indent

    code(line)

    code('')
    ENDIF()
    code('')
    #reduction across blocks
    for n in range(0,nargs):
      if arg_typ[n] == 'ops_arg_gbl' and (accs[n] == OPS_INC or accs[n] == OPS_MIN or accs[n] == OPS_MAX):
        if (accs[n]==OPS_INC):
          operation = '0'
        if (accs[n]==OPS_MIN):
          operation = '1'
        if (accs[n]==OPS_MAX):
          operation = '2'
        if 'real' in typs[n].lower():
          if dims[n].isdigit() and dims[n] == '1':
            code('call ReductionFloat8(sharedMem, reductionArrayDevice'+str(n+1)+
                 '((blockIdx%z - 1)*gridDim%y*gridDim%x + (blockIdx%y - 1)*gridDim%x + (blockIdx%x-1) + 1:),opsGblDat'+str(n+1)+'Device,'+operation+')')
          else:
            code('call ReductionFloat8Mdim(sharedMem, reductionArrayDevice'+str(n+1)+ '(' +
                 '((blockIdx%z - 1)*gridDim%y*gridDim%x + (blockIdx%y - 1)*gridDim%x + (blockIdx%x-1))*('+dims[n]+') + 1:),opsGblDat'+str(n+1)+'Device,'+operation+','+dims[n]+')')
        elif 'integer' in typs[n].lower():
          if dims[n].isdigit() and dims[n] == '1':
            code('call ReductionInt4(sharedMem, reductionArrayDevice'+str(n+1)+
                 '((blockIdx%z - 1)*gridDim%y*gridDim%x + (blockIdx%y - 1)*gridDim%x + (blockIdx%x-1) + 1:),opsGblDat'+str(n+1)+'Device,'+operation+')')
          else:
            code('call ReductionInt4Mdim(sharedMem, reductionArrayDevice'+str(n+1)+ '(' +
                 '((blockIdx%z - 1)*gridDim%y*gridDim%x + (blockIdx%y - 1)*gridDim%x + (blockIdx%x-1))*('+dims[n]+') + 1:),opsGblDat'+str(n+1)+'Device,'+operation+','+dims[n]+')')
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
    code('')
    code('USE CUDAFOR')
    code('IMPLICIT NONE')
    code('')
    code('character(kind=c_char,len=*), intent(in) :: userSubroutine')
    code('type(ops_block), intent(in) :: block')
    code('integer(4), intent(in):: dim')
    code('integer(4), dimension(2*dim), intent(in) :: range')
    code('real(8) :: t1,t2,t3')
    code('real(4) :: transfer_total, transfer')
    code('integer(4) :: istat')
    code('')
    for n in range (0, nargs):
      if arg_typ[n] == 'ops_arg_idx':
        code('type(ops_arg), intent(in) :: opsArg'+str(n+1))
        code('')
      if arg_typ[n] == 'ops_arg_dat':
        code('type(ops_arg), intent(in) :: opsArg'+str(n+1))
        code(typs[n]+', dimension(:), device, pointer  :: opsDat'+str(n+1)+'Local')
        code('integer(4) :: opsDat'+str(n+1)+'Cardinality')
        code('integer(4), pointer, dimension(:) :: dat'+str(n+1)+'_size')
        code('integer(4) :: dat'+str(n+1)+'_base')
        code('integer(4) :: xdim'+str(n+1))
        if dims[n] != '1':
          code('integer(4) :: multi_d'+str(n+1))
        if NDIM==2:
          code('integer(4) :: ydim'+str(n+1))
        elif NDIM==3:
          code('integer(4) :: ydim'+str(n+1)+', zdim'+str(n+1))
        code('')

    for n in range(0,nargs):
      if arg_typ[n] == 'ops_arg_gbl':
        code('type(ops_arg), intent(in) :: opsArg'+str(n+1))
        code('integer(4) :: opsDat'+str(n+1)+'Cardinality')
        if accs[n] == OPS_READ and dims[n] != '1':
          code(typs[n]+', dimension(:), device, pointer :: opsDat'+str(n+1)+'Host')
        else:
          code(typs[n]+', dimension(:), pointer :: opsDat'+str(n+1)+'Host')
        if (accs[n] == OPS_INC or accs[n] == OPS_MAX or accs[n] == OPS_MIN):
          code(typs[n]+', dimension(:), allocatable :: reductionArrayHost'+str(n+1))
          code('integer(4) :: reductionCardinality'+str(n+1))
    code('')


    if NDIM==1:
      code('integer(4) :: x_size')
    elif NDIM==2:
      code('integer(4) :: x_size, y_size')
    elif NDIM==3:
      code('integer(4) :: x_size, y_size, z_size')
    code('integer(4), dimension('+str(NDIM)+') :: start_indx, end_indx')

    if arg_idx == 1:
      code('integer(4), dimension('+str(NDIM)+'), device :: idx')
      code('integer(4), dimension('+str(NDIM)+')         :: idx_h')

    code('integer(4) :: n')
    code('integer(4) :: i10')
    code('integer(4) :: i20')

    code('integer(4) :: blocksPerGrid')
    code('integer(4) :: nshared')
    code('integer(4) :: nthread')


    code('')
    comm('cuda grid and thread block sizes')
    code('type(dim3) :: grid, tblock')
    code('')
    code('type(ops_arg), dimension('+str(nargs)+') :: opsArgArray')
    code('')

    for n in range (0, nargs):
      code('opsArgArray('+str(n+1)+') = opsArg'+str(n+1))
    code('')
    code('call setKernelTime('+str(nk)+',userSubroutine//char(0),0.0_8,0.0_8,0.0_4,1)')
    code('call ops_timers_core(t1)')
    code('')

    config.depth = config.depth - 2
    code('#ifdef OPS_MPI')
    config.depth = config.depth + 2
    IF('getRange(block, start_indx, end_indx, range) < 0')
    code('return')
    ENDIF()
    config.depth = config.depth - 2
    code('#else')
    config.depth = config.depth + 2
    DO('n','1',str(NDIM))
    code('start_indx(n) = range(2*n-1)')
    code('end_indx(n)   = range(2*n)')
    ENDDO()
    config.depth = config.depth - 2
    code('#endif')
    config.depth = config.depth + 2
    code('')
    if arg_idx == 1:
      config.depth = config.depth - 2
      code('#ifdef OPS_MPI')
      config.depth = config.depth + 2
      code('call getIdx(block,start_indx,idx_h)')
      code('idx = idx_h')
      config.depth = config.depth - 2
      code('#else')
      config.depth = config.depth + 2
      for n in range (0, NDIM):
        code('idx('+str(n+1)+') = start_indx('+str(n+1)+')')
      config.depth = config.depth - 2
      code('#endif')
      config.depth = config.depth + 2
      code('')


    code('')
    code('x_size = MAX(0,end_indx(1)-start_indx(1)+1)')
    if NDIM==2:
      code('y_size = MAX(0,end_indx(2)-start_indx(2)+1)')
    if NDIM==3:
      code('y_size = MAX(0,end_indx(2)-start_indx(2)+1)')
      code('z_size = MAX(0,end_indx(3)-start_indx(3)+1)')
    code('')
    if gbls_mdim > 0:
      code('call ops_upload_gbls(opsArgArray,'+str(nargs)+')')

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
          code('dat'+str(n+1)+'_base = getDatBaseFromOpsArg'+str(NDIM)+'D(opsArg'+str(n+1)+',start_indx,multi_d'+str(n+1)+')')
        else:
          code('dat'+str(n+1)+'_base = getDatBaseFromOpsArg'+str(NDIM)+'D(opsArg'+str(n+1)+',start_indx,1)')
        code('call c_f_pointer(opsArg'+str(n+1)+'%data_d,opsDat'+str(n+1)+'Local,(/opsDat'+str(n+1)+'Cardinality/))')

      if arg_typ[n] == 'ops_arg_gbl':
        code('opsDat'+str(n+1)+'Cardinality = opsArg'+str(n+1)+'%dim')
        if accs[n] == OPS_INC or accs[n] == OPS_MAX or accs[n] == OPS_MIN:
          code('call c_f_pointer(getReductionPtrFromOpsArg(opsArg'+str(n+1)+',block),opsDat'+str(n+1)+'Host,(/opsDat'+str(n+1)+'Cardinality/))')
        else:
          if dims[n] != '1':
            code('call c_f_pointer(opsArgArray('+str(n+1)+')%data_d,opsDat'+str(n+1)+'Host,(/opsDat'+str(n+1)+'Cardinality/))')
          else:
            code('call c_f_pointer(opsArg'+str(n+1)+'%data,opsDat'+str(n+1)+'Host,(/opsDat'+str(n+1)+'Cardinality/))')
          
      code('')

    #NEED TO COPY CONSTANTS TO Symbol
    condition = ''
    for n in range (0, nargs):
      if arg_typ[n] == 'ops_arg_dat':
        condition = condition + '(xdim'+str(n+1)+' .NE. xdim'+str(n+1)+'_'+name+'_h) .OR. &\n  '
        if NDIM==3:
          condition = condition + '(ydim'+str(n+1)+' .NE. ydim'+str(n+1)+'_'+name+'_h) .OR. &\n  '
    condition = condition[:-9]

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
    if reduction:
      comm('Reduction vars and shared memory for reductions')
      code('nshared = 0')
      if NDIM==1:
         code('nthread = getOPS_block_size_x()')
         code('blocksPerGrid = ((x_size-1)/getOPS_block_size_x()+ 1)* 1* 1')
      elif NDIM==2:
         code('nthread = getOPS_block_size_x()*getOPS_block_size_y()')
         code('blocksPerGrid = ((x_size-1)/getOPS_block_size_x()+ 1)*((y_size-1)/getOPS_block_size_y() + 1)* 1')
      elif NDIM==3:
         code('nthread = getOPS_block_size_x()*getOPS_block_size_y()')
         code('blocksPerGrid = ((x_size-1)/getOPS_block_size_x()+ 1)*((y_size-1)/getOPS_block_size_y() + 1)* z_size')
      code('')
      for n in range (0, nargs):
        if arg_typ[n] == 'ops_arg_gbl' and accs[n] != OPS_READ:
          code('nshared = MAX(nshared,8*'+str(dims[n])+'*nthread)') #hardcoded to real(8)
      code('')


    for n in range(0,nargs):
      if arg_typ[n] == 'ops_arg_gbl' and (accs[n] == OPS_INC or accs[n] == OPS_MAX or accs[n] == OPS_MIN):
        code('reductionCardinality'+str(n+1)+' = blocksPerGrid * 1')
        code('allocate( reductionArrayHost'+str(n+1)+'(reductionCardinality'+str(n+1)+'* ('+dims[n]+')) )')
        IF ('.not. allocated(reductionArrayDevice'+str(n+1)+'_'+name+')')
        code('allocate( reductionArrayDevice'+str(n+1)+'_'+name+'(reductionCardinality'+str(n+1)+'* ('+dims[n]+')) )')
        ENDIF()
        code('')
        DO('i10','0','reductionCardinality'+str(n+1)+'-1')
        if dims[n].isdigit() and dims[n] == '1':
          if accs[n] == OPS_INC:
            code('reductionArrayHost'+str(n+1)+'(i10+1) = 0.0')
          if accs[n] == OPS_MIN:
            code('reductionArrayHost'+str(n+1)+'(i10+1) = HUGE(reductionArrayHost'+str(n+1)+'(1))')
          if accs[n] == OPS_MAX:
            code('reductionArrayHost'+str(n+1)+'(i10+1) = -1.0*HUGE(reductionArrayHost'+str(n+1)+'(1))')
        else:
          if accs[n] == OPS_INC:
            code('reductionArrayHost'+str(n+1)+'(i10 * ('+dims[n]+') + 1 : i10 * ('+dims[n]+') + ('+dims[n]+')) = 0.0')
          if accs[n] == OPS_MIN:
            code('reductionArrayHost'+str(n+1)+'(i10 * ('+dims[n]+') + 1 : i10 * ('+dims[n]+') + ('+dims[n]+')) = HUGE(reductionArrayHost'+str(n+1)+'(1))')
          if accs[n] == OPS_MAX:
            code('reductionArrayHost'+str(n+1)+'(i10 * ('+dims[n]+') + 1 : i10 * ('+dims[n]+') + ('+dims[n]+')) = -1.0*HUGE(reductionArrayHost'+str(n+1)+'(1))')
        ENDDO()
        code('')
        code('reductionArrayDevice'+str(n+1)+'_'+name+' = reductionArrayHost'+str(n+1)+'')


    #halo exchange
    code('')
    comm('halo exchanges')
    code('call ops_H_D_exchanges_device(opsArgArray,'+str(nargs)+')')
    code('call ops_halo_exchanges(opsArgArray,'+str(nargs)+',range)')
    code('call ops_H_D_exchanges_device(opsArgArray,'+str(nargs)+')')
    code('')
    code('call ops_timers_core(t2)')
    code('')

    #Call cuda kernel  - i.e. the wrapper calling the user kernel
    if reduction:
      code('call '+name+'_wrap <<<grid,tblock,nshared>>> (&')
    else:
      code('call '+name+'_wrap <<<grid,tblock>>> (&')

    for n in range (0, nargs):
      if arg_typ[n] == 'ops_arg_idx':
        code('& idx, &')
      elif arg_typ[n] == 'ops_arg_dat':
        code('& opsDat'+str(n+1)+'Local, &')
      elif arg_typ[n] == 'ops_arg_gbl':
        if accs[n] == OPS_INC or accs[n] == OPS_MIN or accs[n] == OPS_MAX:
          code('& reductionArrayDevice'+str(n+1)+'_'+name+', &')
        elif accs[n] == OPS_READ and dims[n].isdigit() and dims[n]=='1':
          code('& opsDat'+str(n+1)+'Host(1), &')
        elif accs[n] == OPS_READ:
          code('& opsDat'+str(n+1)+'Host, &')
        #need to support multi-dim OPS_READS here



    for n in range (0, nargs):
      if arg_typ[n] == 'ops_arg_dat' :
        code('& dat'+str(n+1)+'_base, &')
    if NDIM==1:
      code('& x_size )')
    elif NDIM==2:
      code('& x_size, y_size )')
    elif NDIM==3:
      code('& x_size, y_size, z_size )')
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
        if arg_typ[n] == 'ops_arg_gbl' and accs[n] != OPS_READ:
          code('reductionArrayHost'+str(n+1)+' = reductionArrayDevice'+str(n+1)+'_'+name+'')
          code('')
          DO('i10','0','reductionCardinality'+str(n+1)+'-1')
          if dims[n].isdigit() and dims[n] == '1':
            if accs[n] == OPS_INC:
              code('opsDat'+str(n+1)+'Host = opsDat'+str(n+1)+'Host + reductionArrayHost'+str(n+1)+'(i10+1)')
            if accs[n] == OPS_MIN:
              code('opsDat'+str(n+1)+'Host = min(opsDat'+str(n+1)+'Host, reductionArrayHost'+str(n+1)+'(i10+1))')
            if accs[n] == OPS_MAX:
              code('opsDat'+str(n+1)+'Host = max(opsDat'+str(n+1)+'Host, reductionArrayHost'+str(n+1)+'(i10+1))')
          else:
            if accs[n] == OPS_INC:
              code('opsDat'+str(n+1)+'Host(1:'+dims[n]+') = opsDat'+str(n+1)+'Host(1:'+dims[n]+') + reductionArrayHost'+str(n+1)+'(i10 * ('+dims[n]+') + 1 : i10 * ('+dims[n]+') + ('+dims[n]+'))')
            else:
              DO('i20','1',dims[n])
              if accs[n] == OPS_MIN:
                code('opsDat'+str(n+1)+'Host(i20) = min(opsDat'+str(n+1)+'Host(i20), reductionArrayHost'+str(n+1)+'(i10 * ('+dims[n]+') + i20))')
              if accs[n] == OPS_MAX:
                code('opsDat'+str(n+1)+'Host(i20) = max(opsDat'+str(n+1)+'Host(i20), reductionArrayHost'+str(n+1)+'(i10 * ('+dims[n]+') + i20))')
              ENDDO()


          ENDDO()
          
          code('')
          code('deallocate( reductionArrayHost'+str(n+1)+' )')


    code('istat = cudaDeviceSynchronize()')
    code('call ops_timers_core(t3)')

    code('call ops_set_dirtybit_device(opsArgArray, '+str(nargs)+')')
    for n in range (0, nargs):
      if arg_typ[n] == 'ops_arg_dat' and (accs[n] == OPS_WRITE or accs[n] == OPS_RW or accs[n] == OPS_INC):
        code('call ops_set_halo_dirtybit3(opsArg'+str(n+1)+',range)')
    code('')

    comm('Timing and data movement')
    code('transfer_total = 0.0_4')
    for n in range (0, nargs):
      if arg_typ[n] == 'ops_arg_dat':
        code('call ops_compute_transfer('+str(NDIM)+', start_indx, end_indx, opsArg'+str(n+1)+',transfer)')
        code('transfer_total = transfer_total + transfer')
    code('call setKernelTime('+str(nk)+',userSubroutine,t3-t2,t2-t1,transfer_total,0)') 



    config.depth = config.depth - 2
    code('')
    code('end subroutine')
    code('')
    code('END MODULE')

##########################################################################
#  output individual kernel file
##########################################################################
    try:
      os.makedirs('./CUDA')
    except OSError as e:
      if e.errno != errno.EEXIST:
        raise
    fid = open('./CUDA/'+name+'_cuda_kernel.CUF','w')
    header_text = f"! Auto-generated at {datetime.now()} by ops-translator legacy\n\n"
    fid.write(header_text)

    fid.write(config.file_text)
    fid.close()
