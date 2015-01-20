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
OPS source code transformation tool (for the Fortran API)

This tool parses the user's original source code to produce
target-specific code to execute the user's kernel functions.

This prototype is written in Python

usage: ./ops_fortran.py file1, file2 ,...

This takes as input

file1.F90, file2.F90, ... (can be files with any fortran suffix .f, .f90, .F90, .F95)

and produces as output modified versions

file1_ops.F90, file2_ops.F90, ...

then calls a number of target-specific code generators
to produce individual kernel files of the form

xxx_seq_kernel.F90 -- for single threaded x86 execution (also used for MPI)
xxx_omp_kernel.F90 -- for OpenMP x86 execution
xxx_kernel.CUF -- for CUDA execution with CUDA Fortran

"""

import sys
import re
import datetime

"""import SEQ/MPI, OpenMP, CUDA, OpenACC and OpenCL code generation functions"""

import util_fortran

comment_remover = util_fortran.comment_remover
remove_trailing_w_space = util_fortran.remove_trailing_w_space

def ops_parse_calls(text):
    """Parsing for ops_init/ops_exit"""

    # remove comments just for this call
    text = comment_remover(text)

    inits = len(re.findall('ops_init', text))
    exits = len(re.findall('ops_exit', text))

    return (inits, exits)

def main():

  # declare constants

  ninit = 0
  nexit = 0
  nkernels = 0
  nconsts = 0
  consts = []
  kernels = []
  kernels_in_files = []

  OPS_GBL = 2

  OPS_READ = 1
  OPS_WRITE = 2
  OPS_RW = 3
  OPS_INC = 4
  OPS_MAX = 5
  OPS_MIN = 6

  OPS_accs_labels = ['OPS_READ', 'OPS_WRITE', 'OPS_RW', 'OPS_INC',
                    'OPS_MAX', 'OPS_MIN']


  #
  # loop over all input source files
  #

  kernels_in_files = [[] for _ in range(len(sys.argv) - 1)]
  for a in range(1, len(sys.argv)):
      print 'processing file ' + str(a) + ' of ' + str(len(sys.argv) - 1) + \
            ' ' + str(sys.argv[a])
      src_file = str(sys.argv[a])
      f = open(src_file, 'r')
      text = f.read()

      #get rid of all comments
      text = remove_trailing_w_space(comment_remover(text))


      #
      # check for ops_init, ops_exit calls
      #

      inits, exits = ops_parse_calls(text)

      if inits + exits > 0:
        print ' '
      if inits > 0:
        print'contains ops_init call'
      if exits > 0:
        print'contains ops_exit call'

      ninit = ninit + inits
      nexit = nexit + exits


      #
      # parse and process constants
      #


      #
      # parse and process ops_par_loop calls
      #

      #
      # output new source file
      #


  #
  # errors and warnings
  #

  if ninit == 0:
      print' '
      print'-----------------------------'
      print'  ERROR: no call to ops_init  '
      print'-----------------------------'

  if nexit == 0:
      print' '
      print'-------------------------------'
      print'  WARNING: no call to ops_exit  '
      print'-------------------------------'


  #
  # finally, generate target-specific kernel files
  #
  #ops_fortran_gen_mpi(str(sys.argv[1]), date, consts, kernels)

if __name__ == '__main__':
    if len(sys.argv) > 1:
        main()
    #Print usage message if no arguments given
    else:
        print __doc__
        sys.exit(1)