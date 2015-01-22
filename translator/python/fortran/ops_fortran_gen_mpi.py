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
OPS MPI_seq code generator for Fortran applications

This routine is called by ops_fortran.py which parses the input files

It produces a file xxx_seq_kernel.F90 for each kernel

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

def ops_fortran_gen_mpi(master, date, consts, kernels):

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
    code('USE ISO_C_BINDING')
    code('USE OPS_CONSTANTS')
    code('')

##########################################################################
#  generate MACROS
##########################################################################
    for n in range (0, nargs):
      if arg_typ[n] == 'ops_arg_dat':
        if int(dims[n]) == 1:
          code('INTEGER(KIND=4) xdim'+str(n+1))
          if NDIM==1:
            code('#define OPS_ACC'+str(n+1)+'(x) (x)')
          if NDIM==2:
            code('#define OPS_ACC'+str(n+1)+'(x,y) (x+xdim'+str(n+1)+'_'+name+'*(y))')
          if NDIM==3:
            code('#define OPS_ACC'+str(n+1)+'(x,y,z) (x+xdim'+str(n+1)+'_'+name+'*(y)+xdim'+str(n+1)+'_'+name+'*ydim'+str(n+1)+'_'+name+'*(z))')
    code('')

    for n in range (0, nargs):
      if arg_typ[n] == 'ops_arg_dat':
        if int(dims[n]) > 1:
          code('INTEGER(KIND=4) multi_d'+str(n+1))
          code('INTEGER(KIND=4) xdim'+str(n+1))
          if NDIM==1:
            code('#define OPS_ACC_MD'+str(nn+1)+'(d,x) ((x)*'+str(dims[n])+'+(d))')
          if NDIM==2:
            code('#define OPS_ACC_MD'+str(n+1)+'(d,x,y) ((x)*'+str(dims[n])+'+(d)+(xdim'+str(n)+'*(y)*'+str(dims[n])+'))')
          if NDIM==3:
            code('#define OPS_ACC_MD'+str(n+1)+'(d,x,y,z) ((x)*'+str(dims[n])+'+(d)+(xdim'+str(n)+'*(y)*'+str(dims[n])+')+(xdim'+str(n)+'*ydim'+str(n)+'*(z)*'+str(dims[n])+'))')

    code('')
    code('contains')
    code('')

##########################################################################
#  user kernel subroutine
##########################################################################
    comm('user function')
    code('!DEC$ ATTRIBUTES FORCEINLINE :: ' + name )
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

    code(text)
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
    code('subroutine '+name+'_wrap( &')
    for n in range (0, nargs):
      if arg_typ[n] == 'ops_arg_dat':
        code('& opsDat'+str(n+1)+'Local, &')
    for n in range (0, nargs):
      if arg_typ[n] == 'ops_arg_dat':
        code('& dat'+str(n+1)+'_base, &')
    code('& start, &')
    code('& end )')

    config.depth = config.depth + 2
    code('IMPLICIT NONE')
    for n in range (0, nargs):
      if arg_typ[n] == 'ops_arg_dat'and accs[n] == OPS_READ:
        code(typs[n]+', INTENT(IN) :: opsDat'+str(n+1)+'Local(*)')
      elif arg_typ[n] == 'ops_arg_dat'and (accs[n] == OPS_WRITE or accs[n] == OPS_RW or accs[n] == OPS_INC):
        code(typs[n]+'opsDat'+str(n+1)+'Local(*)')
    for n in range (0, nargs):
      if arg_typ[n] == 'ops_arg_dat':
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
      DO('n_x','start(1)','end(1)')
    elif NDIM==2:
      DO('n_y','start(2)','end(2)')
      DO('n_x','start(1)','end(1)')
    elif NDIM==3:
      DO('n_z','start(3)','end(3)')
      DO('n_y','start(2)','end(2)')
      DO('n_x','start(1)','end(1)')

    code('call '+name + '( &')
    indent = config.depth *' '
    line = ''
    for n in range (0, nargs):
      print arg_typ[n]
      if arg_typ[n] == 'ops_arg_dat':
        if NDIM==1:
          line = line + '& opsDat'+str(n+1)+'(dat'+str(n+1)+'_base+(n_x-1)*'+str(dims[n])+')'
        elif NDIM==2:
          line = line + '& opsDat'+str(n+1)+'(dat'+str(n+1)+'_base+(n_x-1)*'+str(dims[n])+\
             ' + (n_y-1)*xdim'+str(n+1)+'*'+str(dims[n])+')'
        elif NDIM==3:
          line = line + '& opsDat'+str(n+1)+'(dat'+str(n+1)+'_base+(n_x-1)*'+str(dims[n])+\
             ' + (n_y-1)*xdim'+str(n+1)+'*'+str(dims[n])+'), '+\
             ' + (n_z-1)*ydim'+str(n+1)+'*'+str(dims[n])+')'
      elif arg_typ[n] == 'ops_arg_gbl':
        line = line + '& opsDat'+str(n+1)+'(dat'+str(n+1)+'_base)'

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


##########################################################################
#  output individual kernel file
##########################################################################
    if not os.path.exists('./MPI'):
      os.makedirs('./MPI')
    fid = open('./MPI/'+name+'_seq_kernel.F90','w')
    date = datetime.datetime.now()
    fid.write('!\n! auto-generated by ops_fortran.py\n!\n')

    fid.write(config.file_text)
    fid.close()