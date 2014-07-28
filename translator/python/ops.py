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
OPS source code transformation tool

This tool parses the user's original source code to produce
target-specific code to execute the user's kernel functions.

This prototype is written in Python

usage: ./ops 'file1','file2',...

This takes as input

file1.cpp, file2.cpp, ...

and produces as output modified versions

file1_ops.cpp, file2_ops.cpp, ...

then calls a number of target-specific code generators
to produce individual kernel files of the form

xxx_seq_kernel.cpp -- for single threaded x86 execution
xxx_omp_kernel.cpp -- for OpenMP x86 execution
xxx_kernel.cu -- for CUDA execution

"""

import sys
import re
import datetime

# import SEQ, OpenMP and CUDA code generation functions

#from ops_gen_seq_macro import ops_gen_seq_macro # deprecated .. use ops_gen_mpi
#from ops_gen_openmp_macro import ops_gen_openmp_macro # deprecated .. use ops_gen_mpi_openmp
#from ops_gen_cuda import ops_gen_cuda # deprecated .. use ops_gen_mpi_cuda

from ops_gen_mpi import ops_gen_mpi
from ops_gen_mpi_openmp import ops_gen_mpi_openmp
from ops_gen_mpi_cuda import ops_gen_mpi_cuda
from ops_gen_mpi_openacc import ops_gen_mpi_openacc
from ops_gen_mpi_opencl import ops_gen_mpi_opencl


# from http://stackoverflow.com/a/241506/396967
def comment_remover(text):
    """Remove comments from text"""

    def replacer(match):
        s = match.group(0)
        if s.startswith('/'):
            return ""
        else:
            return s
    pattern = re.compile(
        r'//.*?$|/\*.*?\*/|\'(?:\\.|[^\\\'])*\'|"(?:\\.|[^\\"])*"',
        re.DOTALL | re.MULTILINE
    )
    return re.sub(pattern, replacer, text)

def remove_trailing_w_space(text):
  line_start = 0
  line = ""
  line_end = 0
  striped_test = ''
  count = 0
  while 1:
    line_end =  text.find("\n",line_start+1)
    line = text[line_start:line_end]
    line = line.rstrip()
    striped_test = striped_test + line +'\n'
    line_start = line_end + 1
    line = ""
    if line_end < 0:
      return striped_test[:-1]

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
  for m in re.finditer('ops_decl_const\((.*)\)', text):
    args = m.group(1).split(',')

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

def get_arg_dat(arg_string, j):
    loc = arg_parse(arg_string, j + 1)
    dat_args_string = arg_string[arg_string.find('(', j) + 1:loc]

    # remove comments
    dat_args_string = comment_remover(dat_args_string)
    #print dat_args_string
    #num = len(dat_args_string.split(','))
    #print num

    # check for syntax errors
    if not(len(dat_args_string.split(',')) == 4 or len(dat_args_string.split(',')) == 5 ):
      print 'Error parsing op_arg_dat(%s): must have four or five arguments' % dat_args_string
      return

    if len(dat_args_string.split(',')) == 4:
      # split the dat_args_string into  6 and create a struct with the elements
      # and type as op_arg_dat
      temp_dat = {'type': 'ops_arg_dat',
                  'dat': dat_args_string.split(',')[0].strip(),
                  'sten': dat_args_string.split(',')[1].strip(),
                  'typ': (dat_args_string.split(',')[2].replace('"','')).strip(),
                  'acc': dat_args_string.split(',')[3].strip()}
    elif len(dat_args_string.split(',')) == 5:
      # split the dat_args_string into  6 and create a struct with the elements
      # and type as op_arg_dat
      temp_dat = {'type': 'ops_arg_dat_opt',
                  'dat': dat_args_string.split(',')[0].strip(),
                  'sten': dat_args_string.split(',')[1].strip(),
                  'typ': (dat_args_string.split(',')[2].replace('"','')).strip(),
                  'acc': dat_args_string.split(',')[3].strip(),
                  'opt': dat_args_string.split(',')[4].strip()}


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

def ops_par_loop_parse(text):
  """Parsing for op_par_loop calls"""

  loop_args = []

  #text = comment_remover(text)
  search = "ops_par_loop"
  i = text.find(search)
  while i > -1:
      arg_string = text[text.find('(', i) + 1:text.find(';', i + 12)]

      # parse arguments in par loop
      temp_args = []
      num_args = 0

      # parse each op_arg_dat
      search2 = "ops_arg_dat"
      search3 = "ops_arg_gbl"
      j = arg_string.find(search2)
      k = arg_string.find(search3)

      while j > -1 or k > -1:
        if k <= -1:
            temp_dat = get_arg_dat(arg_string, j)
            # append this struct to a temporary list/array
            temp_args.append(temp_dat)
            num_args = num_args + 1
            j = arg_string.find(search2, j + 12)

        elif j <= -1:
            temp_gbl = get_arg_gbl(arg_string, k)
            # append this struct to a temporary list/array
            temp_args.append(temp_gbl)
            num_args = num_args + 1
            k = arg_string.find(search3, k + 12)

        elif j < k:
            temp_dat = get_arg_dat(arg_string, j)
            # append this struct to a temporary list/array
            temp_args.append(temp_dat)
            num_args = num_args + 1
            j = arg_string.find(search2, j + 12)

        else:
            temp_gbl = get_arg_gbl(arg_string, k)
            # append this struct to a temporary list/array
            temp_args.append(temp_gbl)
            num_args = num_args + 1
            k = arg_string.find(search3, k + 12)

      temp = {'loc': i,
            'name1': arg_string.split(',')[0].strip(),
            'name2': arg_string.split(',')[1].strip(),
            'block': arg_string.split(',')[2].strip(),
            'dim': arg_string.split(',')[3].strip(),
            'range': arg_string.split(',')[4].strip(),
            'args': temp_args,
            'nargs': num_args}
      #print temp
      loop_args.append(temp)

      i = text.find(search, i + 15)
  print '\n\n'
  return (loop_args)

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
      #text = comment_remover(text)

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

      # parse and process constants

      const_args = ops_decl_const_parse(text)
      print str(len(const_args))

      # cleanup '&' symbols from name and convert dim to integer
      if const_args:
        for i in range(0, len(const_args)):
            #if const_args[i]['name2'][0] == '&':
            const_args[i]['name2'] = const_args[i]['name2']
            const_args[i]['dim'] = const_args[i]['dim']
            const_args[i]['name'] = const_args[i]['name']

      # check for repeats
      nconsts = 0
      if const_args:
        for i in range(0, len(const_args)):
            repeat = 0
            name = const_args[i]['name']
            for c in range(0, nconsts):
                if const_args[i]['name'] == consts[c]['name']:
                    repeat = 1
                    if const_args[i]['type'] != consts[c]['type']:
                        print 'type mismatch in repeated ops_decl_const'
                    if const_args[i]['dim'] != consts[c]['dim']:
                        print 'size mismatch in repeated ops_decl_const'

            if repeat > 0:
                print 'repeated global constant ' + const_args[i]['name']
            else:
                print '\nglobal constant (' + const_args[i]['name'].strip() \
                      + ') of size ' + str(const_args[i]['dim'])

            # store away in master list
            if repeat == 0:
                nconsts = nconsts + 1
                temp = {'dim': const_args[i]['dim'].strip(),
                        'type': const_args[i]['type'].strip(),
                        'name': const_args[i]['name'].strip()}
                consts.append(temp)

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

            print var[m]+' '+ str(stens[m]) +' '+str(accs[m])


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
#                    kernels[nk]['var'][arg] == var[arg] and \
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
        ##todo -- not sure what are interesting here
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

      fid = open(src_file.split('.')[0] + '_ops.cpp', 'w')
      date = datetime.datetime.now()
      fid.write('//\n// auto-generated by ops.py on ' +
                date.strftime("%Y-%m-%d %H:%M") + '\n//\n\n')

      loc_old = 0

      # read original file and locate header location
      header_len = 9
      loc_header = [text.find("ops_seq.h")]
      if loc_header[0] == -1:
        header_len = 13
        loc_header = [text.find("ops_lib_cpp.h")]

      # get locations of all op_decl_consts
      n_consts = len(const_args)
      loc_consts = [0] * n_consts
      for n in range(0, n_consts):
          loc_consts[n] = const_args[n]['loc']

      # get locations of all ops_par_loops
      n_loops = len(loop_args)
      loc_loops = [0] * n_loops
      for n in range(0, n_loops):
          loc_loops[n] = loop_args[n]['loc']

      #get locations of all kernel.h headder file declarations
      loc_kernel_headers = []

      p = re.compile('#include .*kernel.h')
      iterator = p.finditer(text)
      for match in iterator:
        #print match.start()
        loc_kernel_headers.append(match.start());
        #loc_kernel_headers.append(match.end());


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

        if (locs[loc] in loc_header) and (locs[loc] != -1):
            fid.write(' "ops_lib_cpp.h"\n\n')
            if len(kernels_in_files[a - 1]) > 0:
              fid.write('//\n// ops_par_loop declarations\n//\n')
            for k_iter in range(0, len(kernels_in_files[a - 1])):
                k = kernels_in_files[a - 1][k_iter]
                line = '\nvoid ops_par_loop_' + \
                    kernels[k]['name'] + '(char const *, ops_block, int , int*,\n'
                for n in range(1, kernels[k]['nargs']):
                    line = line + '  ops_arg,\n'
                line = line + '  ops_arg );\n'
                fid.write(line)

            fid.write('\n')
            loc_old = locs[loc] + header_len+1
            continue


        if (locs[loc] in loc_kernel_headers) and (locs[loc] != -1):
            fid.write('\n//')
            endofcall = text.find('kernel.h', locs[loc])
            loc_old = locs[loc] #endofcall + 1
            continue

        if locs[loc] in loc_loops:
          indent = indent + ' ' * len('ops_par_loop')
          endofcall = text.find(';', locs[loc])
          curr_loop = loc_loops.index(locs[loc])
          name = loop_args[curr_loop]['name1']
          line = str(' ops_par_loop_' + name + '(' +
                     loop_args[curr_loop]['name2'] + ', ' +
                     loop_args[curr_loop]['block'] + ', ' +
                     loop_args[curr_loop]['dim'] + ', ' +
                     loop_args[curr_loop]['range'] + ',\n' + indent)

          for arguments in range(0, loop_args[curr_loop]['nargs']):
              elem = loop_args[curr_loop]['args'][arguments]
              if elem['type'] == 'ops_arg_dat':
                  line = line + elem['type'] + '(' + elem['dat'] + \
                      ', ' + elem['sten'] + ', "' + elem['typ'] + \
                      '", ' + elem['acc'] + '),\n' + indent
              if elem['type'] == 'ops_arg_dat_opt':
                  line = line + elem['type'] + '(' + elem['dat'] + \
                      ', ' + elem['sten'] + ', "' + elem['typ'] + \
                      '", ' + elem['acc'] + \
                      ', ' + elem['opt'] +'),\n' + indent
                  #loop_args[curr_loop]['args'][arguments]['type'] = 'ops_arg_dat' # make opt arg a normal arg
              elif elem['type'] == 'ops_arg_gbl':
                  line = line + elem['type'] + '(' + elem['data'] + \
                      ', ' + elem['dim'] + ', "' +  elem['typ'] + \
                      '", ' +  elem['acc'] + '),\n' + indent

          fid.write(line[0:-len(indent) - 2] + ');')

          loc_old = endofcall + 1
          continue

        if locs[loc] in loc_consts:
          curr_const = loc_consts.index(locs[loc])
          endofcall = text.find(';', locs[loc])
          name = const_args[curr_const]['name']
          fid.write(indent[0:-2] + 'ops_decl_const2( '+ name.strip() +
            ',' + str(const_args[curr_const]['dim']) + ', "' +
            const_args[curr_const]['type'] + '",' +
            const_args[curr_const]['name2'].strip() + ');')
          loc_old = endofcall + 1
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


  #ops_gen_seq_macro(str(sys.argv[1]), date, consts, kernels) # deprecated .. use ops_gen_mpi
  #ops_gen_openmp_macro(str(sys.argv[1]), date, consts, kernels) # deprecated .. use ops_gen_mpi_openmp
  #ops_gen_cuda(str(sys.argv[1]), date, consts, kernels) # deprecated .. use ops_gen_mpi_cuda

  #ops_gen_mpi(str(sys.argv[1]), date, consts, kernels)
  #ops_gen_mpi_openmp(str(sys.argv[1]), date, consts, kernels)
  #ops_gen_mpi_cuda(str(sys.argv[1]), date, consts, kernels)
  #ops_gen_mpi_openacc(str(sys.argv[1]), date, consts, kernels)
  ops_gen_mpi_opencl(str(sys.argv[1]), date, consts, kernels)


if __name__ == '__main__':
    if len(sys.argv) > 1:
        main()
    # Print usage message if no arguments given
    else:
        print __doc__
        sys.exit(1)
