#!/usr/bin/env python

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

# import OpenMP and CUDA code generation functions
#from op2_gen_seq import op2_gen_seq
#from op2_gen_openmp import op2_gen_openmp
#from op2_gen_cuda import op2_gen_cuda


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


def ops_parse_calls(text):
    """Parsing for ops_init/ops_exit"""

    # remove comments just for this call
    text = comment_remover(text)

    inits = len(re.findall('ops_init', text))
    exits = len(re.findall('ops_exit', text))

    return (inits, exits)

def ops_par_loop_parse(text):
  """Parsing for op_par_loop calls"""

  search = "ops_par_loop_opt"
  i = text.find(search)
  while i > -1:
      arg_string = text[text.find('(', i) + 1:text.find(';', i + 16)]
      print arg_string
      i = text.find(search, i + 16)
  print '\n\n'
  #return (loop_args)

def main():

  # declare constants

  ninit = 0
  nexit = 0
  nkernels = 0
  consts = []
  kernels = []
  kernels_in_files = []

  OP_GBL = 2

  OP_READ = 1
  OP_WRITE = 2
  OP_RW = 3
  OP_INC = 4
  OP_MAX = 5
  OP_MIN = 6

  OP_accs_labels = ['OP_READ', 'OP_WRITE', 'OP_RW', 'OP_INC',
                    'OP_MAX', 'OP_MIN']

  # loop over all input source files

  kernels_in_files = [[] for _ in range(len(sys.argv) - 1)]
  for a in range(1, len(sys.argv)):
      print 'processing file ' + str(a) + ' of ' + str(len(sys.argv) - 1) + \
            ' ' + str(sys.argv[a])

      src_file = str(sys.argv[a])
      f = open(src_file, 'r')
      text = f.read()

      # check for ops_init, ops_exit calls

      inits, exits = ops_parse_calls(text)

      if inits + exits > 0:
        print ' '
      if inits > 0:
        print'contains ops_init call'
      if exits > 0:
        print'contains ops_exit call'

      ninit = ninit + inits
      nexit = nexit + exits

      loop_args = ops_par_loop_parse(text)

      loop_args = []




if __name__ == '__main__':
    if len(sys.argv) > 1:
        main()
    # Print usage message if no arguments given
    else:
        print __doc__
        sys.exit(1)
