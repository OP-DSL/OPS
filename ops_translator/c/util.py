
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
#  utility functions for code generator
#
"""
utility functions for code generator

"""

import re
import config

def comm(line):
  prefix = ' '*config.depth
  if len(line) == 0:
    config.file_text +='\n'
  else:
    config.file_text +=prefix+'//'+line+'\n'

def code(text):
  prefix = ''
  if len(text) != 0:
    prefix = ' '*config.depth

  config.file_text += prefix+text+'\n'

def FOR(i,start,finish):
  code('for ( int '+i+'='+start+'; '+i+'<'+finish+'; '+i+'++ ){')
  config.depth += 2

def FOR2(i,start,finish,increment):
  code('for ( int '+i+'='+start+'; '+i+'<'+finish+'; '+i+'+='+increment+' ){')
  config.depth += 2

def WHILE(line):
  code('while ( '+ line+ ' ){')
  config.depth += 2

def ENDWHILE():
  config.depth -= 2
  code('}')

def ENDFOR():
  config.depth -= 2
  code('}')

def IF(line):
  code('if ('+ line + ') {')
  config.depth += 2

def ELSEIF(line):
  code('else if ('+ line + ') {')
  config.depth += 2

def ELSE():
  code('else {')
  config.depth += 2

def ENDIF():
  config.depth -= 2
  code('}')


def mult(text, i, n):
  text = text + '1'
  for nn in range (0, i):
    text = text + '* args['+str(n)+'].dat->size['+str(nn)+']'

  return text

def para_parse(text, j, op_b, cl_b):
    """Parsing code block, i.e. text to find the correct closing brace"""

    depth = 0
    loc2 = j

    while 1:
      if text[loc2] == op_b:
            depth = depth + 1

      elif text[loc2] == cl_b:
            depth = depth - 1
            if depth == 0:
                return loc2
      loc2 = loc2 + 1

def comment_remover(text):
    """Remove comments from text"""

    def replacer(match):
        s = match.group(0)
        if s.startswith('/'):
            return ''
        else:
            return s
    pattern = re.compile(
        r'//.*?$|/\*.*?\*/|\'(?:\\.|[^\\\'])*\'|"(?:\\.|[^\\"])*"',
        re.DOTALL | re.MULTILINE
    )
    return re.sub(pattern, replacer, text)

def remove_trailing_w_space(text):
  text = text+' '
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

def arg_parse_list(text, j):
    """Parsing arguments in function to find the correct closing brace"""

    depth = 0
    loc2 = j
    arglist = []
    prev_start = j
    while 1:
        if text[loc2] == '(':
            if depth == 0:
                prev_start = loc2+1
            depth = depth + 1

        elif text[loc2] == ')':
            depth = depth - 1
            if depth == 0:
                arglist.append(text[prev_start:loc2].strip())
                return arglist

        elif text[loc2] == ',':
            if depth == 1:
                arglist.append(text[prev_start:loc2].strip())
                prev_start = loc2+1
        elif text[loc2] == '{':
            depth = depth + 1
        elif text[loc2] == '}':
            depth = depth - 1
        loc2 = loc2 + 1

def parse_replace_ACC_signature(text, arg_typ, dims, opencl=0, accs=[], typs=[]):
  for i in range(0,len(dims)):
    if arg_typ[i] == 'ops_arg_dat':
      if not dims[i].isdigit() or int(dims[i])>1:
        text = re.sub(r'ACC<([a-zA-Z0-9]*)>\s*&', r'ptrm_\1 ',text, 1)
      else:
        text = re.sub(r'ACC<([a-zA-Z0-9]*)>\s*&', r'ptr_\1 ',text, 1)
    elif opencl == 1 and arg_typ[i] == 'ops_arg_gbl' and accs[i] == 1 and (not dims[i].isdigit() or int(dims[i])>1):
        #if multidim global read, then it is passed in as a global pointer, otherwise it's local
        args = text.split(',')
        text = ''
        for j in range(0,len(args)):
            if j == i:
              text = text + args[j].replace(typs[j],'__global '+typs[j]).replace('*', '* restrict ')+', '
            else:    
              text = text + args[j]+', '
        text = text[:-2]

  return text

def convert_ACC_signature(text, arg_typ):
  arg_list = arg_parse_list(text,0)
  for i in range(0,len(arg_list)):
      if arg_typ[i] == 'ops_arg_dat' and not ('ACC' in arg_list[i]):
          arg_list[i] = arg_list[i].replace('int','ACC<int>')
          arg_list[i] = arg_list[i].replace('float','ACC<float>')
          arg_list[i] = arg_list[i].replace('double','ACC<double>')
          arg_list[i] = arg_list[i].replace('*','&')
  signature = ''
  for i in range(0,len(arg_list)):
      signature = signature + arg_list[i] + ',\n  '
  return signature[:-4]

def convert_ACC_body(text):
  text = re.sub('\[OPS_ACC_MD[0-9]+(\([ -A-Za-z0-9,+]*\))\]', r'\1', text)
  text = re.sub('\[OPS_ACC[0-9]+(\([ -A-Za-z0-9,+]*\))\]', r'\1', text)
  return text

def convert_ACC(text, arg_typ):
  openb = text.find('(')
  closeb = text[0:text.find('{')].rfind(')')+1
  text = text[0:openb]+'('+convert_ACC_signature(text[openb:closeb], arg_typ)+')'+text[closeb:]
  body_start = text.find('{')
  text = text[0:body_start]+convert_ACC_body(text[body_start:])
  return text

def parse_signature(text):
  text2 = text.replace('const','')
  text2 = text2.replace('ACC<','')
  text2 = text2.replace('>','')
  text2 = text2.replace('int','')
  text2 = text2.replace('long','')
  text2 = text2.replace('float','')
  text2 = text2.replace('double','')
  text2 = text2.replace('*','')
  text2 = text2.replace('&','')
  text2 = text2.replace(')','')
  text2 = text2.replace('(','')
  text2 = text2.replace('\n','')
  text2 = re.sub('\[[0-9]*\]','',text2)
  arg_list = []
  args = text2.split(',')
  for n in range(0,len(args)):
    arg_list.append(args[n].strip())
  return arg_list

def find_consts(text, consts):
  found_consts = []

  for cn in range(0,len(consts)):
    pattern = consts[cn]['name'][1:-1]
    if re.search('\\b'+pattern+'\\b', text):
      print(("found " + consts[cn]['name'][1:-1]))
      found_consts.append(cn)

  return found_consts


def parse_signature_opencl(text2):

  #text2 = text2.replace('const','')
  text2 = text2.replace('*','* restrict ')
  text2 = text2.replace('int','__global int')
  #text2 = re.sub('[\s]int','__global int',text2)
  text2 = text2.replace('float','__global float')
  text2 = text2.replace('double','__global double')
  #text2 = re.sub('double','__global double',text2)
  return text2

def complex_numbers_cuda(text):
    """ Handle complex numbers, and translate to the relevant CUDA function in cuComplex.h """

    # Complex number assignment
    p = re.compile("([a-zA-Z_][a-zA-Z0-9]+)(\s+\_\_complex\_\_\s+)([a-zA-Z_][a-zA-Z0-9]*)\s*=\s*(.+)\s*;")
    result = p.finditer(text)
    new_code = text
    complex_variable_names = []
    for match in result:
        complex_variable_names.append(match.group(3))
        rhs = match.group(4)
        if rhs in complex_variable_names:
            # Assignment of another complex variable already defined.
            if match.group(1) == "double":
                new_statement = "cuDoubleComplex %s = %s;" % (match.group(3), rhs)
            elif match.group(1) == "float":
                new_statement = "cuFloatComplex %s = %s;" % (match.group(3), rhs)
            else:
                continue
        else:
            # Assignment of a complex number in real and imaginary parts.
            p = re.compile("(\S+I?)\s*([+-]?)\s*(\S*I?)?")
            complex_number = p.search(rhs)
            if(complex_number.group(1)[-1] == "I"):  # Real after imaginary part
                imag = complex_number.group(1)[:-1]
                if(complex_number.group(3)):  # If real part specified
                    real = complex_number.group(3)
                else:
                    real = "0.0"
            elif(complex_number.group(3)[-1] == "I"):  # Imaginary after real part
                if(complex_number.group(2) == "-"):
                    imag = "-" + complex_number.group(3)[:-1]
                else:
                    imag = complex_number.group(3)[:-1]
                if(complex_number.group(1)):  # If real part specified
                    real = complex_number.group(1)
                else:
                    real = "0.0"
            else:  # No imaginary part
                real = complex_number.group(0)
                imag = "0.0"
            if match.group(1) == "double":
                new_statement = "cuDoubleComplex %s = make_cuDoubleComplex(%s, %s);" % (match.group(3), real, imag)
            elif match.group(1) == "float":
                new_statement = "cuFloatComplex %s = make_cuFloatComplex(%s, %s);" % (match.group(3), real, imag)
            else:
                continue

        # Perform replacement.
        new_code = new_code.replace(match.group(0), new_statement)

    # Complex number __real__ and __imag__
    p = re.compile("(\_\_real\_\_)\s+([a-zA-Z_][a-zA-Z0-9]*)")
    result = p.finditer(new_code)
    for match in result:
        new_code = new_code.replace(match.group(0), "cuCreal(%s)" % (match.group(2)))
    p = re.compile("(\_\_imag\_\_)\s+([a-zA-Z_][a-zA-Z0-9]*)")
    result = p.finditer(new_code)
    for match in result:
        new_code = new_code.replace(match.group(0), "cuCimag(%s)" % (match.group(2)))

    # Multiplication of two complex numbers
    p = re.compile("([a-zA-Z_][a-zA-Z0-9]*)\s*\*\s*([a-zA-Z_][a-zA-Z0-9]*)")
    result = p.finditer(new_code)
    for match in result:
        if(match.group(1) in complex_variable_names or match.group(2) in complex_variable_names):
            new_code = new_code.replace(match.group(0), "cuCmul(%s, %s)" % (match.group(1), match.group(2)))

    return new_code

def arg_parse(text, j):
    """Parsing arguments in ops_par_loop to find the correct closing brace"""

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

def check_accs(name, arg_list, arg_typ, text):
  for n in range(0,len(arg_list)):
    if arg_typ[n] == 'ops_arg_dat':
      pos = 0
      while 1:
        match = re.search('\\b'+arg_list[n]+'\\b',text[pos:])
        if match == None:
          break
        pos = pos + match.start(0)
        if pos < 0:
          break
        pos = pos + len(arg_list[n])

        match0 = re.search('OPS_ACC_MD\d',text[pos:])
        match1 = re.search('OPS_ACC\d',text[pos:])

        if match0 != None :
          if match1 != None:
            if match0.start(0) > match1.start(0):
              match = re.search('OPS_ACC\d',text[pos:])
              pos = pos + match.start(0)
              pos2 = text[pos+7:].find('(')
              num = int(text[pos+7:pos+7+pos2])
              if num != n:
                print(('Access mismatch in '+name+', arg '+str(n)+'('+arg_list[n]+') with OPS_ACC'+str(num)))
              pos = pos+7+pos2
            elif match0.start(0) < match1.start(0):
              match = re.search('OPS_ACC_MD\d',text[pos:])
              pos = pos + match.start(0)
              pos2 = text[pos+10:].find('(')
              num = int(text[pos+10:pos+10+pos2])
              if num != n:
                print(('Access mismatch in '+name+', arg '+str(n)+'('+arg_list[n]+') with OPS_ACC_MD'+str(num)))
              pos = pos+10+pos2
          else:
            match = re.search('OPS_ACC_MD\d',text[pos:])
            pos = pos + match.start(0)
            pos2 = text[pos+10:].find('(')
            num = int(text[pos+10:pos+10+pos2])
            if num != n:
              print(('Access mismatch in '+name+', arg '+str(n)+'('+arg_list[n]+') with OPS_ACC_MD'+str(num)))
            pos = pos+10+pos2
        else:
          if match1 != None:
            match = re.search('OPS_ACC\d',text[pos:])
            pos = pos + match.start(0)
            pos2 = text[pos+7:].find('(')
            num = int(text[pos+7:pos+7+pos2])
            if num != n:
              print(('Access mismatch in '+name+', arg '+str(n)+'('+arg_list[n]+') with OPS_ACC'+str(num)))
            pos = pos+7+pos2
          else:
            break


        #if match0 <> None and match1 <> None:
        #  if match0.start(0) > match1.start(0):
        #    match = re.search('OPS_ACC\d',text[pos:])
        #   pos = pos + match.start(0)
        #    pos2 = text[pos+7:].find('(')
        #    num = int(text[pos+7:pos+7+pos2])
        #    if num <> n:
        #      print 'Access mismatch in '+name+', arg '+str(n)+'('+arg_list[n]+') with OPS_ACC'+str(num)
        #    pos = pos+7+pos2
        #  elif match0.start(0) < match1.start(0):
        #    match = re.search('OPS_ACC_MD\d',text[pos:])
        #    pos = pos + match.start(0)
        #    pos2 = text[pos+10:].find('(')
        #    num = int(text[pos+10:pos+10+pos2])
        #    if num <> n:
        #      print 'Access mismatch in '+name+', arg '+str(n)+'('+arg_list[n]+') with OPS_ACC_MD'+str(num)
        #    pos = pos+10+pos2

def replace_ACC_kernel_body(kernel_text, arg_list, arg_typ, nargs, opencl=0, dims=[]):
    # replace all data args with macros
    for n in range(0,nargs):
      if arg_typ[n] == 'ops_arg_dat':
        pattern = re.compile(r'\b'+arg_list[n]+r'\b')
        match = pattern.search(kernel_text,0)
        while match:
          closeb = para_parse(kernel_text,match.start(),'(',')')+1
          openb = kernel_text.find('(',match.start())
          if opencl == 1:
            if not dims[n].isdigit() or int(dims[n])>1:
              acc = 'OPS_ACCM('+arg_list[n]+', '+kernel_text[openb+1:closeb-1]+')'
            else:
              acc = 'OPS_ACCS('+arg_list[n]+', '+kernel_text[openb+1:closeb-1]+')'
          else:
            acc = 'OPS_ACC('+arg_list[n]+', '+kernel_text[openb+1:closeb-1]+')'
          kernel_text = kernel_text[0:match.start()] + acc + kernel_text[closeb:]
          match = pattern.search(kernel_text,match.start()+10)
    return kernel_text
