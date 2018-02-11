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
#  OPS source code transformation tool (for the Fortran API)
#
#  This tool parses the user's original source code to produce
#  target-specific code to execute the user's kernel functions.
#
#  This prototype is written in Python
#
#  usage: ./ops_fortran.py file1, file2 ,...
#
#  This takes as input
#
#  file1.F90, file2.F90, ... (can be files with any fortran suffix .f, .f90, .F90, .F95)
#
#  and produces as output modified versions
#
#  file1_ops.F90, file2_ops.F90, ...
#
#  then calls a number of target-specific code generators
#  to produce individual kernel files of the form
#
#  xxx_seq_kernel.F90 -- for single threaded x86 execution (also used for MPI)
#  xxx_omp_kernel.F90 -- for OpenMP x86 execution
#  xxx_kernel.CUF -- for CUDA execution with CUDA Fortran
#  xxx_openacc_kernel.F90 -- for OpenACC execution
#

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
xxx_openacc_kernel.F90 -- for OpenACC execution

"""

import sys
import re
import datetime
import os

"""import SEQ/MPI, OpenMP, CUDA, OpenACC and OpenCL code generation functions"""
from ops_fortran_gen_mpi import ops_fortran_gen_mpi
from ops_fortran_gen_mpi_openmp import ops_fortran_gen_mpi_openmp
from ops_fortran_gen_mpi_cuda import ops_fortran_gen_mpi_cuda
from ops_fortran_gen_mpi_openacc import ops_fortran_gen_mpi_openacc

"""import fortran code generation function"""
import util_fortran

comment_remover = util_fortran.comment_remover
remove_trailing_w_space = util_fortran.remove_trailing_w_space

no_master_gen = 0

def ops_parse_calls(text):
    """Parsing for ops_init/ops_exit"""

    # remove comments just for this call
    text = comment_remover(text)

    inits = len(re.findall('ops_init', text))
    exits = len(re.findall('ops_exit', text))

    return (inits, exits)

def ops_decl_const_parse(text):
  """Parsing for ops_decl_const calls"""

  consts = []
  for m in re.finditer('(.*)call(.+)ops_decl_const(.*)\((.*)\)', text):
    args = m.group(4).split(',')
    print m.group(4)
    # check for syntax errors
    if len(args) != 4:
      print 'Error in ops_decl_const : must have four arguments'
      return

    consts.append({
          'loc': m.start(),
          'name': args[0].strip(),
          'dim': args[1].strip(),
          'type': (args[2].replace('"','')).strip(),
          'name2': args[3].strip()
    })

  return consts

def arg_parse(text, j):
    """Parsing arguments in op_par_loop to find the correct closing brace"""

    depth = 0
    loc2 = j
    while 1:
        if text[loc2] == '(':
            depth = depth + 1

        elif text[loc2] == ')':
            depth = depth - 1
            if depth == 0:
                return loc2

        loc2 = loc2 + 1

def arg_parse_list(text, j):
    """Parsing arguments in op_par_loop to find the correct closing brace"""

    depth = 0
    loc2 = j
    args = []
    arg_begin = j
    while 1:
        if text[loc2] == '(':
            depth = depth + 1
            if depth == 1:
                arg_begin = loc2 + 1

        elif text[loc2] == ')':
            depth = depth - 1
            if depth == 0:
                args.append(text[arg_begin:loc2].strip())
                return loc2, args

        elif text[loc2] == ',' and depth == 1:
            args.append(text[arg_begin:loc2].strip())
            arg_begin = loc2 + 1

        loc2 = loc2 + 1

def get_arg_dat(arg_string, j):
    loc = arg_parse(arg_string, j + 1)
    dat_type = arg_string[j:j+arg_string[j:].find('(')].strip()
    dat_args_string = arg_string[arg_string.find('(', j):loc+1]

    # remove comments
    dat_args_string = comment_remover(dat_args_string)
    loc, argsl = arg_parse_list(dat_args_string,0)
    #print dat_args_string
    #num = len(dat_args_string.split(','))
    #print num

    if dat_type == 'ops_arg_dat':
      if len(argsl) <> 5:
        print 'Error parsing op_arg_dat(%s): must have five arguments' % dat_args_string
        return
      # split the dat_args_string into  5 and create a struct with the elements
      # and type as op_arg_dat
      temp_dat = {'type': 'ops_arg_dat',
                  'dat': argsl[0].strip(),
                  'idx': '-1',
                  'dim': argsl[1].strip(),
                  'sten': argsl[2].strip(),
                  'typ': (argsl[3].replace('"','')).strip(),
                  'acc': argsl[4].strip()}
      print temp_dat
    if dat_type == 'ops_arg_dat_opt':
      if len(argsl) <> 6:
        print 'Error parsing op_arg_dat_opt(%s): must have six arguments' % dat_args_string
        return
      # split the dat_args_string into  6 and create a struct with the elements
      # and type as op_arg_dat
      temp_dat = {'type': 'ops_arg_dat_opt',
                  'dat':  argsl[0].strip(),
                  'idx':  '-1',
                  'dim':  argsl[1].strip(),
                  'sten': argsl[2].strip(),
                  'typ': (argsl[3].replace('"','')).strip(),
                  'acc':  argsl[4].strip(),
                  'opt':  argsl[5].strip()}
    if dat_type == 'ops_arg_dptr':
      if len(argsl) <> 5 and len (argsl) <> 7:
        print 'Error parsing op_arg_dptr(%s): must have five or seven arguments' % dat_args_string
        return
      # split the dat_args_string into  5 and create a struct with the elements
      # and type as op_arg_dat
      if len(argsl) == 5:
        temp_dat = {'type': 'ops_arg_dat',
                    'dat': argsl[0].strip(),
                    'idx': '-1',
                    'dim': '1',
                    'sten': argsl[2].strip(),
                    'typ': (argsl[3].replace('"','')).strip(),
                    'acc': argsl[4].strip()}
      else:
        temp_dat = {'type': 'ops_arg_dat',
                    'dat': argsl[0].strip(),
                    'idx': argsl[2].strip(),
                    'dim': argsl[3].strip(),
                    'sten': argsl[4].strip(),
                    'typ': (argsl[5].replace('"','')).strip(),
                    'acc': argsl[6].strip()}

    return temp_dat

def get_arg_gbl(arg_string, k):
    loc = arg_parse(arg_string, k + 1)
    gbl_args_string = arg_string[arg_string.find('(', k) + 1:loc]

    # remove comments
    gbl_args_string = comment_remover(gbl_args_string)

    # check for syntax errors
    if len(gbl_args_string.split(',')) != 4:
        print 'Error parsing op_arg_gbl(%s): must have four arguments' \
              % gbl_args_string
        return

    # split the gbl_args_string into  4 and create a struct with the elements
    # and type as op_arg_gbl
    temp_gbl = {'type': 'ops_arg_gbl',
                'data': gbl_args_string.split(',')[0].strip(),
                'dim': gbl_args_string.split(',')[1].strip(),
                'typ': (gbl_args_string.split(',')[2].replace('"','')).strip(),
                'acc': gbl_args_string.split(',')[3].strip()}

    return temp_gbl

def get_arg_idx(arg_string, l):
    loc = arg_parse(arg_string, l + 1)

    temp_idx = {'type': 'ops_arg_idx'}
    return temp_idx

def ops_par_loop_parse(text):
  global no_master_gen
  """Parsing for op_par_loop calls"""

  loop_args = []

  #text = comment_remover(text)
  search = "ops_par_loop"
  i = text.find(search)
  while i > -1:
      arg_string = text[text.find('(',i):arg_parse(text,i+11)+1]
      arg_string = arg_string.replace('&','')
      arg_string = arg_string.replace(' ','')
      arg_string = arg_string.replace('\n','')
      loc, args = arg_parse_list(arg_string, 0)

      parloop_suffix = text[i+12:text.find('(',i)].strip()
      if len(parloop_suffix)>0:
        if (no_master_gen == -1):
          print 'Error: cannot have generic ops_par_loop calls in a single file and ops_par_loop_* calls'
          sys.exit(-1) 
        no_master_gen = 1
      else:
        if (no_master_gen == 1):
          print 'Error: cannot have generic ops_par_loop calls in a single file and ops_par_loop_* calls'
          sys.exit(-1) 
        no_master_gen = -1


      
      j = 0
      while j < len(args):
        if "ops_arg" in args[j]:
          break
        j = j+1

      # parse arguments in par loop
      temp_args = []
      num_args = 0

      for arg in range(j, len(args)):
        if 'ops_arg_d' in args[arg]:
          temp_dat =  get_arg_dat(args[arg],0)
          temp_args.append(temp_dat)
          num_args = num_args+1
        elif 'ops_arg_gbl' in args[arg] or 'ops_arg_reduce' in args[arg]:
          temp_gbl = get_arg_gbl(args[arg],0)
          temp_args.append(temp_gbl)
          num_args = num_args + 1
        elif 'ops_arg_idx' in args[arg]:
          temp_idx = get_arg_idx(args[arg],0)
          temp_args.append(temp_idx)
          num_args = num_args + 1
        else:
          print 'Error: unrecognised argument to ops_par_loop: ' + args[arg]
          sys.exit(-1)
      print parloop_suffix
      if len(parloop_suffix)>0:
        temp = {'loc': i,
              'name1': (args[0].replace('"','')).strip(),
              'name2': (args[0].replace('"','')).strip(),
              'block': args[1],
              'dim'  : args[2],
              'range': args[3],
              'args': temp_args,
              'nargs': num_args}
      else:
        temp = {'loc': i,
              'name1': args[0],
              'name2': args[1],
              'block': args[2],
              'dim'  : args[3],
              'range': args[4],
              'args': temp_args,
              'nargs': num_args}

      loop_args.append(temp)

      i = text.find(search, i + 15)
  print '\n\n'
  return (loop_args)

def main(source_files):

  if not source_files:
    raise ValueError("No source files specified.")

  amr=os.getenv('OPS_AMR','0')
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

  no_master_gen = 0
  #
  # loop over all input source files
  #

  kernels_in_files = [[] for _ in range(len(source_files))]
  for a in range(0, len(source_files)):
      print 'processing file ' + str(a) + ' of ' + str(len(source_files)) + \
            ' ' + str(source_files[a])

      src_file = str(source_files[a])
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

      const_args = ops_decl_const_parse(text)


      #
      # parse and process ops_par_loop calls
      #
      loop_args = ops_par_loop_parse(text)

      for i in range(0, len(loop_args)):
        name = loop_args[i]['name1']
        nargs = loop_args[i]['nargs']
        dim   = loop_args[i]['dim']
        block = loop_args[i]['block']
        _range   = loop_args[i]['range']
        print '\nprocessing kernel ' + name + ' with ' + str(nargs) + ' arguments'
        print 'dim: '+dim
        print 'range: '+str(_range)

        #
        # process arguments
        #
        typ = [''] * nargs
        var = [''] * nargs
        stens = [0] * nargs
        accs = [0] * nargs
        dims = [''] * nargs #only for globals
        typs = [''] * nargs

        for m in range(0, nargs):
          arg_type = loop_args[i]['args'][m]['type']
          args = loop_args[i]['args'][m]

          if arg_type.strip() == 'ops_arg_dat' or arg_type.strip() == 'ops_arg_dat_opt':
            var[m] = args['dat']
            dims[m] = args['dim']
            stens[m] = args['sten']
            typs[m] = args['typ']
            typ[m] = 'ops_arg_dat'

            l = -1
            for l in range(0, len(OPS_accs_labels)):
                if args['acc'].strip() == OPS_accs_labels[l].strip():
                  break

            if l == -1:
                print 'unknown access type for argument ' + str(m)
            else:
                accs[m] = l + 1

            print var[m]+' '+str(dims[m]) +' '+str(stens[m])+' '+str(accs[m])


          if arg_type.strip() == 'ops_arg_gbl':
            var[m] = args['data']
            dims[m] = args['dim']
            typs[m] = args['typ']
            typ[m] = 'ops_arg_gbl'

            l = -1
            for l in range(0, len(OPS_accs_labels)):
                if args['acc'].strip() == OPS_accs_labels[l].strip():
                    break
            if l == -1:
                print 'unknown access type for argument ' + str(m)
            else:
                accs[m] = l + 1

            print var[m]+' '+ str(dims[m]) +' '+str(accs[m])

          if arg_type.strip() == 'ops_arg_idx':
            var[m] = ''
            dims[m] = 0
            typs[m] = 'int'
            typ[m] = 'ops_arg_idx'
            print 'arg_idx'


        #
        # check for repeats
        #
        repeat = False
        rep1 = False
        rep2 = False
        which_file = -1
        for nk in range(0, nkernels):
          rep1 = kernels[nk]['name'] == name and \
            kernels[nk]['nargs'] == nargs and \
            kernels[nk]['dim'] == dim and \
            kernels[nk]['range'] == _range
          if rep1:
            rep2 = True
            for arg in range(0, nargs):
                rep2 = rep2 and \
                    kernels[nk]['stens'][arg] == stens[arg] and \
                    kernels[nk]['dims'][arg] == dims[arg] and \
                    kernels[nk]['typs'][arg] == typs[arg] and \
                    kernels[nk]['accs'][arg] == accs[arg]
            if rep2:
              print 'repeated kernel with compatible arguments: ' + \
                    kernels[nk]['name'],
              repeat = True
              which_file = nk
            else:
              print 'repeated kernel with incompatible arguments: ERROR'
              break

        #
        # output various diagnostics
        #
        ##
        ##todo -- not sure what will be interesting here
        ##

        #
        # store away in master list
        #
        if not repeat:
              nkernels = nkernels + 1
              temp = { 'arg_type':typ,
                       'name': name,
                      'nargs': nargs,
                      'dim': dim,
                      'dims': dims,
                      'stens': stens,
                      'var': var,
                      'accs': accs,
                      'typs': typs,
                      'range': _range
              }
              kernels.append(temp)
              (kernels_in_files[a - 1]).append(nkernels - 1)
        else:
              append = 1
              for in_file in range(0, len(kernels_in_files[a - 1])):
                  if kernels_in_files[a - 1][in_file] == which_file:
                      append = 0
              if append == 1:
                  (kernels_in_files[a - 1]).append(which_file)


      #
      # output new source file
      #

      if no_master_gen == 1: #if the source file is not using generic ops_par_loop calls
        continue

      #gen_fortran_source_file(str(source_files[0]), consts, kernels, src_file, text, loop_args)
      fid = open(src_file.split('.')[0] + '_ops.F90', 'w')
      date = datetime.datetime.now()
      fid.write('!\n! auto-generated by ops_fortran.py\n!\n')

      loc_old = 0

      # read original file and locate header location
      header_len = 25
      loc_header = [text.find("use OPS_Fortran_Reference")]

      # get locations of all ops_decl_consts
      n_consts = len(const_args)
      loc_consts = [0] * n_consts
      for n in range(0, n_consts):
          loc_consts[n] = const_args[n]['loc']

      # get locations of all ops_par_loops
      n_loops = len(loop_args)
      loc_loops = [0] * n_loops
      for n in range(0, n_loops):
          loc_loops[n] = loop_args[n]['loc']

      #get locations of all kernel.inc headder file declarations
      loc_kernel_headers = []

      p = re.compile('#include *kernel.inc')
      iterator = p.finditer(text)
      for match in iterator:
        loc_kernel_headers.append(match.start());

      locs = sorted(loc_header + loc_consts + loc_loops + loc_kernel_headers)

      # process header and loops
      for loc in range(0, len(locs)):
        if locs[loc] != -1:
            fid.write(text[loc_old:locs[loc] - 1])
            loc_old = locs[loc] - 1

        indent = ''
        ind = 0
        while 1:
            if text[locs[loc] - ind] == '\n':
                break
            indent = indent + ' '
            ind = ind + 1

        if locs[loc] in loc_header:
          line = ''
          line = line +'\n'+'  use OPS_Fortran_Declarations'
          line = line +'\n'+'  use OPS_Fortran_RT_Support'+'\n'

          for nk in range (0,len(kernels)):
            line = line +'  use ' + kernels[nk]['name'].upper()+'_MODULE'+'\n'

          fid.write(line[2:len(line)]);
          loc_old = locs[loc] + header_len + 1
          continue


        if (locs[loc] in loc_kernel_headers) and (locs[loc] != -1):
            fid.write('\n//')
            endofcall = text.find('kernel.h', locs[loc])
            loc_old = locs[loc] #endofcall + 1
            continue

        if locs[loc] in loc_loops:
          indent = indent + ' ' * len('ops_par_loop')
          endofcall = arg_parse(text,locs[loc]+11)
          curr_loop = loc_loops.index(locs[loc])
          name = loop_args[curr_loop]['name1']
          line = str(' ops_par_loop_'+loop_args[curr_loop]['name1'] + '(' +
                     loop_args[curr_loop]['name2'] + ', ' +
                     loop_args[curr_loop]['block'] + ', ' +
                     loop_args[curr_loop]['dim'] + ', ' +
                     loop_args[curr_loop]['range'] + ', &\n' + indent)

          for arguments in range(0, loop_args[curr_loop]['nargs']):
              elem = loop_args[curr_loop]['args'][arguments]
              if elem['type'] == 'ops_arg_dat':
                  line = line + '& '+elem['type'] + '(' + elem['dat'] + \
                      ', ' + elem['dim'] + ', ' + elem['sten'] + ', "' + elem['typ'] + \
                      '", ' + elem['acc'] + '), &\n' + indent
              if elem['type'] == 'ops_arg_dat_opt':
                  line = line + '& '+elem['type'] + '(' + elem['dat'] + \
                      ', ' + elem['dim'] + ', ' + elem['sten'] + ', "' + elem['typ'] + \
                      '", ' + elem['acc'] + \
                      ', ' + elem['opt'] +'), &\n' + indent

              elif elem['type'] == 'ops_arg_gbl':
                if elem['acc'] == 'OPS_READ':
                  line = line + '& '+elem['type'] + '(' + elem['data'] + \
                      ', ' + elem['dim'] + ', "' +  elem['typ'] + \
                      '", ' +  elem['acc'] + '), &\n' + indent
                else:
                  line = line + '& ops_arg_reduce(' + elem['data'] + \
                        ', ' + elem['dim'] + ', "' +  elem['typ'] + \
                        '", ' +  elem['acc'] + '), &\n' + indent
              elif elem['type'] == 'ops_arg_idx':
                  line = line + '& '+elem['type'] + '(), &\n' + indent

          fid.write(line[0:-len(indent) - 4] + ')')

          loc_old = endofcall + 1
          continue

        # stripping the ops_decl_consts -- as there is no implentation required
        if locs[loc] in loc_consts:
          line = ''
          fid.write(line);
          endofcall = text.find('\n', locs[loc])
          loc_old = endofcall+1
          continue

      fid.write(text[loc_old:])
      fid.close()
      f.close()

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

  ops_fortran_gen_mpi(str(source_files[0]), date, consts, kernels, amr)
  ops_fortran_gen_mpi_openmp(str(source_files[0]), date, consts, kernels, amr)
#  ops_fortran_gen_mpi_cuda(str(source_files[0]), date, consts, kernels)
#  ops_fortran_gen_mpi_openacc(str(source_files[0]), date, consts, kernels)

if __name__ == '__main__':
    if len(sys.argv) > 1:
        main(source_files=sys.argv[1:]) # [1:] ignores the ops.py file itself.
    # Print usage message if no arguments given
    else:
        print __doc__
        sys.exit(1)
