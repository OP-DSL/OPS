#!/usr/bin/env python3

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
#  OPS source code transformation tool (for the C/C++ API)
#
#  This tool parses the user's original source code to produce
#  target-specific code to execute the user's kernel functions.
#
#  This prototype is written in Python
#
#  usage: ./ops.py file1, file2 ,...
#
#  This takes as input
#
#  file1.cpp, file2.cpp, ...
#
#  and produces as output modified versions
#
#  file1_ops.cpp, file2_ops.cpp, ...
#
#  then calls a number of target-specific code generators
#  to produce individual kernel files of the form
#
#  xxx_seq_kernel.cpp -- for single threaded x86 execution (also used for MPI)
#  xxx_omp_kernel.cpp -- for OpenMP x86 execution
#  xxx_kernel.cu -- for CUDA execution
#

"""
OPS source code transformation tool (for the C/C++ API)

This tool parses the user's original source code to produce
target-specific code to execute the user's kernel functions.

This prototype is written in Python

usage: ./ops.py file1, file2 ,...

This takes as input

file1.cpp, file2.cpp, ...

and produces as output modified versions

file1_ops.cpp, file2_ops.cpp, ...

then calls a number of target-specific code generators
to produce individual kernel files of the form

xxx_seq_kernel.cpp -- for single threaded x86 execution (also used for MPI)
xxx_omp_kernel.cpp -- for OpenMP x86 execution
xxx_kernel.cu -- for CUDA execution

"""

import sys
from os import path
import re
import datetime

"""import SEQ/MPI, OpenMP, CUDA, OpenACC and OpenCL code generation functions"""
from ops_gen_mpi_inline import ops_gen_mpi_inline
from ops_gen_mpi_lazy import ops_gen_mpi_lazy
from ops_gen_mpi_cuda import ops_gen_mpi_cuda
from ops_gen_mpi_hip import ops_gen_mpi_hip
from ops_gen_mpi_openacc import ops_gen_mpi_openacc
from ops_gen_mpi_opencl import ops_gen_mpi_opencl

import util
import config
from config import OPS_READ, OPS_WRITE, OPS_RW, OPS_INC, OPS_MAX, OPS_MIN, OPS_accs_labels

arithmetic_regex_pattern = r'^[ \(\)\+\-\*\\\.\%0-9]+$'

comment_remover = util.comment_remover
remove_trailing_w_space = util.remove_trailing_w_space
verbose = config.verbose

def ops_parse_calls(text):
    """Parsing for ops_init/ops_exit"""

    # remove comments just for this call
    text = comment_remover(text)

    inits = len(re.findall('ops_init', text))
    exits = len(re.findall('ops_exit', text))

    return (inits, exits)

def ops_parse_macro_defs(text):
    """Parsing for C macro definitions"""

    defs = {}
    macro_def_pattern = r'(\n|^)[ ]*(#define[ ]+)([A-Za-z0-9\_]+)[ ]+([0-9A-Za-z\_\.\+\-\*\/\(\) ]+)'
    for match in re.findall(macro_def_pattern, text):
        if len(match) < 4:
            continue
        elif len(match) > 4:
            print("Unexpected format for macro definition: " + str(match))
            continue
        key = match[2]
        value = match[3]
        defs[key] = value
    return defs

def self_evaluate_macro_defs(macro_defs):
    """Recursively evaluate C macro definitions that refer to other detected macros"""

    substitutions_performed = True
    while substitutions_performed:
        substitutions_performed = False
        for k in list(macro_defs.keys()):
            k_val = macro_defs[k]
            m = re.search(arithmetic_regex_pattern, k_val)
            if m != None:
                ## This macro definition is numeric
                continue

            ## If value of key 'k' depends on value of other
            ## keys, then substitute in value:
            for k2 in list(macro_defs.keys()):
                pattern = r'' + '(^|[^a-zA-Z0-9_])' + k2 + '($|[^a-zA-Z0-9_])'
                m = re.search(pattern, k_val)

                if m != None:
                    ## The macro "k" refers to macro "k2"
                    k2_val = macro_defs[k2]
                    macro_defs[k] = re.sub(pattern, "\\g<1>"+k2_val+"\\g<2>", k_val)
                    substitutions_performed = True

    ## Evaluate any mathematical expressions:
    for k in list(macro_defs.keys()):
        val = macro_defs[k]
        m = re.search(arithmetic_regex_pattern, val)
        if m != None:
            res = ""
            try:
                res = eval(val)
            except:
                pass
            if type(res) != type(""):
                if str(res) != val:
                    macro_defs[k] = str(res)

def evaluate_macro_defs_in_string(macro_defs, string):
    """Recursively evaluate C macro definitions in 'string' """

    resolved_string = string

    substitutions_performed = True
    while substitutions_performed:
        substitutions_performed = False
        for k in list(macro_defs.keys()):
            k_val = macro_defs[k]

            k_pattern = r'' + r'' + '(^|[^a-zA-Z0-9_])' + k + '($|[^a-zA-Z0-9_])'
            m = re.search(k_pattern, resolved_string)
            if m != None:
                ## "string" contains a reference to macro "k", so substitute
                ## in its definition:
                resolved_string_new = re.sub(k_pattern, "\\g<1>"+k_val+"\\g<2>", resolved_string)
                resolved_string = resolved_string_new
                substitutions_performed = True


    if re.search(arithmetic_regex_pattern, resolved_string) != None:
        res = ""
        try:
            res = eval(resolved_string)
        except:
            return resolved_string
        else:
            if type(res) != type(""):
                resolved_string = str(res)

    return resolved_string

def ops_decl_const_parse(text, macro_defs):
  """Parsing for ops_decl_const calls"""

  consts = []
  for m in re.finditer('(ops_|\.|->)decl_const\((.*)\)', text):
    args = m.group(2).split(',')

    # check for syntax errors
    if len(args) != 4:
      print('Error in ops_decl_const : must have four arguments')
      return
    args[1] = evaluate_macro_defs_in_string(macro_defs, args[1])

    if args[0].count('"') != 2:
      print('Error in ops_decl_const : name must be a string literal')
      return

    if args[2].count('"') != 2:
      print('Error in ops_decl_const : type must be a string literal')
      return

    consts.append({
          'loc': m.start(),
          'name': args[0].strip(),
          'dim': evaluate_macro_defs_in_string(macro_defs, args[1].strip()),
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

def get_arg_dat(arg_string, j, macro_defs):
    loc = arg_parse(arg_string, j + 1)
    dat_args_string = arg_string[arg_string.find('(', j) + 1:loc]

    # remove comments
    dat_args_string = comment_remover(dat_args_string)

    # check for syntax errors
    if not(len(dat_args_string.split(',')) == 5 or len(dat_args_string.split(',')) == 6 ):
      print('Error parsing op_arg_dat(%s): must have five or six arguments' % dat_args_string)
      return

    if len(dat_args_string.split(',')) == 5:
      # split the dat_args_string into  5 and create a struct with the elements
      # and type as op_arg_dat
      temp_dat = {'type': 'ops_arg_dat',
                  'dat': dat_args_string.split(',')[0].strip(),
                  'dim': evaluate_macro_defs_in_string(macro_defs, dat_args_string.split(',')[1].strip()),
                  'sten': dat_args_string.split(',')[2].strip(),
                  'typ': (dat_args_string.split(',')[3].replace('"','')).strip(),
                  'acc': dat_args_string.split(',')[4].strip()}
    elif len(dat_args_string.split(',')) == 6:
      # split the dat_args_string into  6 and create a struct with the elements
      # and type as op_arg_dat
      temp_dat = {'type': 'ops_arg_dat_opt',
                  'dat': dat_args_string.split(',')[0].strip(),
                  'dim': evaluate_macro_defs_in_string(macro_defs, dat_args_string.split(',')[1].strip()),
                  'sten': dat_args_string.split(',')[2].strip(),
                  'typ': (dat_args_string.split(',')[3].replace('"','')).strip(),
                  'acc': dat_args_string.split(',')[4].strip(),
                  'opt': dat_args_string.split(',')[5].strip()}


    return temp_dat

def get_arg_gbl(arg_string, k, macro_defs):
    loc = arg_parse(arg_string, k + 1)
    gbl_args_string = arg_string[arg_string.find('(', k) + 1:loc]

    # remove comments
    gbl_args_string = comment_remover(gbl_args_string)

    # check for syntax errors
    if len(gbl_args_string.split(',')) != 4:
        print('Error parsing op_arg_gbl(%s): must have four arguments' \
              % gbl_args_string)
        return

    # split the gbl_args_string into  4 and create a struct with the elements
    # and type as op_arg_gbl
    temp_gbl = {'type': 'ops_arg_gbl',
                'data': gbl_args_string.split(',')[0].strip(),
                'dim': evaluate_macro_defs_in_string(macro_defs, gbl_args_string.split(',')[1].strip()),
                'typ': (gbl_args_string.split(',')[2].replace('"','')).strip(),
                'acc': gbl_args_string.split(',')[3].strip()}

    return temp_gbl

def get_arg_idx(arg_string, l):
    loc = arg_parse(arg_string, l + 1)

    temp_idx = {'type': 'ops_arg_idx'}

    return temp_idx

def ops_par_loop_parse(text, macro_defs):
  """Parsing for op_par_loop calls"""

  loop_args = []

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
      search4 = "ops_arg_idx"
      search5 = "ops_arg_reduce"
      j = arg_string.find(search2)
      k = arg_string.find(search3)
      l = arg_string.find(search4)
      m = arg_string.find(search5)

      while j > -1 or k > -1 or l > -1 or m>-1:
        if j>=0 and (j < k or k<=-1) and (j < l or l <=-1) and (j < m or m <=-1):
            temp_dat = get_arg_dat(arg_string, j, macro_defs)
            # append this struct to a temporary list/array
            temp_args.append(temp_dat)
            num_args = num_args + 1
            j = arg_string.find(search2, j + 12)

        elif k>=0 and (k < j or j<=-1) and (k < l or l <=-1) and (k < m or m <=-1):
            temp_gbl = get_arg_gbl(arg_string, k, macro_defs)
            # append this struct to a temporary list/array
            temp_args.append(temp_gbl)
            num_args = num_args + 1
            k = arg_string.find(search3, k + 12)

        elif l>=0 and (l < j or j<=-1) and (l < k or k <=-1) and (l < m or m <=-1):
            temp_idx = get_arg_idx(arg_string, l)
            # append this struct to a temporary list/array
            temp_args.append(temp_idx)
            num_args = num_args + 1
            l = arg_string.find(search4, l + 12)

        elif m>=0 and (m < j or j<=-1) and (m < l or l <=-1) and (m < k or k <=-1):
            temp_gbl = get_arg_gbl(arg_string, m,  macro_defs)
            # append this struct to a temporary list/array
            temp_args.append(temp_gbl)
            num_args = num_args + 1
            m = arg_string.find(search5, m + 15)

      temp = {'loc': i,
            'name1': arg_string.split(',')[0].strip(),
            'name2': arg_string.split(',')[1].strip(),
            'block': arg_string.split(',')[2].strip(),
            'dim': evaluate_macro_defs_in_string(macro_defs, arg_string.split(',')[3].strip()),
            'range': arg_string.split(',')[4].strip(),
            'args': temp_args,
            'nargs': num_args}
      loop_args.append(temp)

      i = text.find(search, i + 15)
  if verbose:
      print('\n\n')
  return (loop_args)


def parse_source_files(source_files):
    # declare constants
    ninit = 0
    nexit = 0
    soa_set = 0
    consts = []
    kernels = []
    kernels_in_files = [[] for _ in source_files]
    loop_args_in_files = []
    texts = []
    macro_defs = {}

    #
    # loop over all input source files
    #

    # Find the macros defined in the source files
    for src_file in source_files:
        with open(src_file, 'r') as f:
            text = f.read()

        defs = ops_parse_macro_defs(text)
        for k in list(defs.keys()):
            if (k in macro_defs) and (defs[k] != macro_defs[k]):
                raise ValueError("Multiple macros with same same %s", k)
            else:
                macro_defs[k] = defs[k]
        defs = {}
    self_evaluate_macro_defs(macro_defs)

    for a, (src_file,
            kernels_in_file) in enumerate(zip(source_files, kernels_in_files)):
        if verbose:
            print(f'processing file {a} of {len(source_files)} {src_file}')

        with open(src_file, 'r') as f:
            text = f.read()

        #get rid of all comments
        text = remove_trailing_w_space(comment_remover(text))

        text = re.sub('ops_init(.*);', 'ops_init\\1;\n  ops_init_backend();',
                      text)
        if text.find('ops_init') > -1:
            text = re.sub('#include', 'void ops_init_backend();\n#include',
                          text, 1)
        texts.append(text)
        #
        # check for ops_init, ops_exit calls
        #

        inits, exits = ops_parse_calls(text)

        if verbose:
            if inits + exits > 0:
                print(' ')
            if inits > 0:
                print('contains ops_init call')
            if exits > 0:
                print('contains ops_exit call')

        ninit = ninit + inits
        nexit = nexit + exits

        #
        # check for SoA
        #
        file_soa = len(re.findall('#define OPS_SOA', text))
        if a > 0 and soa_set == 1 and file_soa == 0:
            print('Error: all or no source files must include #define OPS_SOA')
            sys.exit(1)
        if file_soa != 0:
            soa_set = file_soa
        if inits > 0 and file_soa and len(
                re.findall(r'\bOPS_soa\b\s*=\s*1', text)) == 0:
            print(
                'Error: the source file with ops_init, must include the line OPS_soa = 1 immediately after ops_init'
            )
            sys.exit(1)

        #
        # parse and process constants
        #

        const_args = ops_decl_const_parse(text, macro_defs)
        if verbose:
            print(str(len(const_args)))

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
                            print('type mismatch in repeated ops_decl_const')
                        if const_args[i]['dim'] != consts[c]['dim']:
                            print('size mismatch in repeated ops_decl_const')

                if repeat > 0:
                    if verbose:
                        print('repeated global constant ' +
                               const_args[i]['name'])
                else:
                    if verbose:
                        print('\nglobal constant (' + const_args[i]['name'].strip() \
                              + ') of size ' + str(const_args[i]['dim']))

                # store away in master list
                if repeat == 0:
                    nconsts = nconsts + 1
                    temp = {
                        'dim': const_args[i]['dim'].strip(),
                        'type': const_args[i]['type'].strip(),
                        'name': const_args[i]['name'].strip()
                    }
                    consts.append(temp)

        #
        # parse and process ops_par_loop calls
        #

        loop_args = ops_par_loop_parse(text, macro_defs)
        loop_args_in_files.append(loop_args)

        for i in range(0, len(loop_args)):
            name = loop_args[i]['name1']
            nargs = loop_args[i]['nargs']
            dim = loop_args[i]['dim']
            block = loop_args[i]['block']
            _range = loop_args[i]['range']
            if verbose:
                print(f'\nprocessing kernel {name} with {nargs} arguments')
                print('dim: ' + dim)
                print('range: ' + str(_range))

            #
            # process arguments
            #
            typ = [''] * nargs
            var = [''] * nargs
            stens = [0] * nargs
            accs = [0] * nargs
            dims = [''] * nargs  #only for globals
            typs = [''] * nargs

            for m in range(0, nargs):
                arg_type = loop_args[i]['args'][m]['type']
                args = loop_args[i]['args'][m]

                if arg_type == 'ops_arg_dat' or arg_type == 'ops_arg_dat_opt':
                    var[m] = args['dat']
                    dims[m] = args['dim']
                    stens[m] = args['sten']
                    typs[m] = args['typ']
                    typ[m] = 'ops_arg_dat'

                    try:
                        accs[m] = OPS_accs_labels.index(
                            args['acc'].strip()) + 1
                    except ValueError:
                        print('unknown access type for argument ' + str(m))
                    if verbose:
                        print(var[m] + ' ' + str(dims[m]) + ' ' +
                              str(stens[m]) + ' ' + str(accs[m]))

                elif arg_type == 'ops_arg_gbl':
                    var[m] = args['data']
                    dims[m] = args['dim']
                    typs[m] = args['typ']
                    typ[m] = 'ops_arg_gbl'

                    try:
                        accs[m] = OPS_accs_labels.index(
                            args['acc'].strip()) + 1
                    except ValueError:
                        print('unknown access type for argument ' + str(m))
                    if verbose:
                        print(
                            (var[m] + ' ' + str(dims[m]) + ' ' + str(accs[m])))

                elif arg_type == 'ops_arg_idx':
                    var[m] = ''
                    dims[m] = 0
                    typs[m] = 'int'
                    typ[m] = 'ops_arg_idx'

            #
            # check for repeats
            #
            repeat = False
            which_file = -1
            for nk, kernel in enumerate(kernels):
                rep1 = kernel['name'] == name and \
                  kernel['nargs'] == nargs and \
                  kernel['dim'] == dim and \
                  kernel['range'] == _range
                if rep1:
                    rep2 = True
                    for arg in range(0, nargs):
                        rep2 = rep2 and \
                            kernel['stens'][arg] == stens[arg] and \
                            kernel['dims'][arg] == dims[arg] and \
                            kernel['typs'][arg] == typs[arg] and \
                            kernel['accs'][arg] == accs[arg]
                        #kernel['var'][arg] == var[arg] and \
                    if rep2:
                        if verbose:
                            print('repeated kernel with compatible arguments: ' + \
                                kernel['name'])
                        repeat = True
                        which_file = nk
                    else:
                        print(
                            'repeated kernel with incompatible arguments: ERROR'
                            + kernel['name'])
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
                temp = {
                    'arg_type': typ,
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
                kernels_in_file.append(len(kernels) - 1)
            else:
                if which_file not in kernels_in_file:
                    kernels_in_file.append(which_file)

    #
    # errors and warnings
    #
    if ninit == 0 and verbose:
        print(' ')
        print('-----------------------------')
        print('  ERROR: no call to ops_init  ')
        print('-----------------------------')
    if nexit == 0 and verbose:
        print(' ')
        print('-------------------------------')
        print('  WARNING: no call to ops_exit  ')
        print('-------------------------------')

    return texts, loop_args_in_files, kernels_in_files, kernels, consts, soa_set

def generate_ops_files(sources, source_texts, loop_args_in_files,
                       kernels_in_files, kernels):
    for src_file, text, loop_args, kernels_in_file in zip(
            sources, source_texts, loop_args_in_files, kernels_in_files):
        (src_file_name, src_file_ext) = path.splitext(path.basename(src_file))
        tgt_file_name = src_file_name + "_ops" + src_file_ext
        with open(tgt_file_name, 'w') as fid:
            fid.write('//\n// auto-generated by ops.py\n//\n')

            loc_old = 0

            # read original file and locate header location
            header_len = 9
            loc_header = [text.find("ops_seq.h")]
            if loc_header[0] == -1:
                header_len = 13
                loc_header = [text.find("ops_lib_core.h")]

            if loc_header[0] == -1:
                header_len = 12
                loc_header = [text.find("ops_seq_v2.h")]

            if loc_header[0] != -1: # make sure loc points to whitespace
                loc_header[0] -= 1
                header_len += 1

            # get locations of all ops_par_loops
            loc_loops = [loop['loc'] for loop in loop_args]

            #get locations of all kernel.h header file declarations
            loc_kernel_headers = [
                m.start()
                for m in re.compile('#include .*kernel.h').finditer(text)
            ]

            locs = sorted(loc_header + loc_loops + loc_kernel_headers)

            # process header and loops
            for loc in locs:
                if loc != -1:
                    fid.write(text[loc_old:loc])
                    loc_old = loc

                indent = ' ' * (loc - text[:loc].rindex('\n'))

                if (loc in loc_header) and (loc != -1):
                    fid.write(' "ops_lib_core.h"\n\n')
                    if len(kernels_in_file) > 0:
                        fid.write('//\n// ops_par_loop declarations\n//\n')
                    for k in kernels_in_file:
                        line = '\nvoid ops_par_loop_' + \
                            kernels[k]['name'] + '(char const *, ops_block, int , int*,\n'
                        line += ',\n'.join(['  ops_arg'] * kernels[k]['nargs']) + ' );\n'
                        fid.write(line)

                    fid.write('\n')
                    loc_old = loc + header_len + 1
                    continue

                if (loc in loc_kernel_headers) and (loc != -1):
                    fid.write('//')
                    endofcall = text.find('kernel.h', loc)
                    loc_old = loc  #endofcall + 1
                    continue

                if loc in loc_loops:
                    indent = indent + ' ' * len('ops_par_loop')
                    endofcall = text.find(';', loc)
                    curr_loop = loc_loops.index(loc)
                    name = loop_args[curr_loop]['name1']
                    line = str('ops_par_loop_' + name + '(' +
                               loop_args[curr_loop]['name2'] + ', ' +
                               loop_args[curr_loop]['block'] + ', ' +
                               loop_args[curr_loop]['dim'] + ', ' +
                               loop_args[curr_loop]['range'] + ',\n' + indent)

                    for elem in loop_args[curr_loop]['args']:
                        if elem['type'] == 'ops_arg_dat':
                            line += elem['type'] + '(' + elem['dat'] + \
                                ', ' + elem['dim'] + ', ' + elem['sten'] + ', "' + elem['typ'] + \
                                '", ' + elem['acc'] + '),\n' + indent
                        if elem['type'] == 'ops_arg_dat_opt':
                            line += elem['type'] + '(' + elem['dat'] + \
                                ', ' + elem['dim'] + ', ' + elem['sten'] + ', "' + elem['typ'] + \
                                '", ' + elem['acc'] + ', ' + elem['opt'] +'),\n' + indent
                        elif elem['type'] == 'ops_arg_gbl':
                            if elem['acc'] == 'OPS_READ':
                                line += elem['type'] + '(' + elem['data'] + \
                                    ', ' + elem['dim'] + ', "' +  elem['typ'] + \
                                    '", ' +  elem['acc'] + '),\n' + indent
                            else:
                                line += 'ops_arg_reduce(' + elem['data'] + \
                                      ', ' + elem['dim'] + ', "' +  elem['typ'] + \
                                      '", ' +  elem['acc'] + '),\n' + indent
                        elif elem['type'] == 'ops_arg_idx':
                            line += elem['type'] + '(),\n' + indent

                    fid.write(line[0:-len(indent) - 2] + ');')

                    loc_old = endofcall + 1
                    continue

            fid.write(text[loc_old:])


def generate_kernel_files(app_name, consts, kernels, soa_set):
    date = datetime.datetime.now()
    ops_gen_mpi_inline(app_name, date, consts, kernels, soa_set)
    ops_gen_mpi_lazy(app_name, date, consts, kernels, soa_set)
    ops_gen_mpi_cuda(app_name, date, consts, kernels, soa_set)
    ops_gen_mpi_hip(app_name, date, consts, kernels, soa_set)
    ops_gen_mpi_openacc(app_name, date, consts, kernels, soa_set)
    ops_gen_mpi_opencl(app_name, date, consts, kernels, soa_set)

    import subprocess
    retcode = subprocess.call("which clang-format 1> /dev/null 2>&1",
                              shell=True)
    if retcode == 0:
        pass
        #subprocess.call("$OPS_INSTALL_PATH/../ops_translator/c/format.sh", shell=True)
    else:
        if verbose:
            print('Cannot find clang-format in PATH')
            print(
                'Install and add clang-format to PATH to format generated code to conform to code formatting guidelines'
            )

def main(source_files):
    if not source_files:
        raise ValueError("No source files specified.")

    texts, loop_args_in_files, kernels_in_files, kernels, consts, soa_set = parse_source_files(
        source_files)
    #
    # output new source file
    #
    generate_ops_files(source_files, texts, loop_args_in_files,
                       kernels_in_files, kernels)
    #
    # finally, generate target-specific kernel files
    #
    generate_kernel_files(str(source_files[0]), consts, kernels, soa_set)


if __name__ == '__main__':
    if len(sys.argv) > 1:
        main(source_files=sys.argv[1:])  # [1:] ignores the ops.py file itself.
    # Print usage message if no arguments given
    else:
        print(__doc__)
        sys.exit(1)
