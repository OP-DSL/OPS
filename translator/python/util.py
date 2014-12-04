
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
utility functions for code generator

"""

import re

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
      return striped_test

def parse_signature(text):
  text2 = text.replace('const','')
  text2 = text2.replace('int','')
  text2 = text2.replace('float','')
  text2 = text2.replace('double','')
  text2 = text2.replace('*','')
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
      print "found " + consts[cn]['name'][1:-1]
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

def parse_signature_openacc(text):
  text2 = text.replace('const','')
  text2 = text2.replace('int','')
  text2 = text2.replace('float','')
  text2 = text2.replace('double','')
  text2 = text2.replace('*','')
  text2 = text2.replace(')','')
  text2 = text2.replace('(','')
  text2 = text2.replace('\n','')
  text2 = re.sub('\[[0-9]*\]','',text2)
  arg_list = []
  args = text2.split(',')
  for n in range(0,len(args)):
    arg_list.append(args[n].strip())
  return arg_list

def parse_signature_cuda(text):
  text2 = text.replace('const','')
  text2 = text2.replace('int','')
  text2 = text2.replace('float','')
  text2 = text2.replace('double','')
  text2 = text2.replace('*','')
  text2 = text2.replace(')','')
  text2 = text2.replace('(','')
  text2 = text2.replace('\n','')
  text2 = re.sub('\[[0-9]*\]','',text2)
  arg_list = []
  args = text2.split(',')
  for n in range(0,len(args)):
    arg_list.append(args[n].strip())
  return arg_list

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

        if match0 <> None :
          if match1 <> None:
            if match0.start(0) > match1.start(0):
              match = re.search('OPS_ACC\d',text[pos:])
              pos = pos + match.start(0)
              pos2 = text[pos+7:].find('(')
              num = int(text[pos+7:pos+7+pos2])
              if num <> n:
                print 'Access mismatch in '+name+', arg '+str(n)+'('+arg_list[n]+') with OPS_ACC'+str(num)
              pos = pos+7+pos2
            elif match0.start(0) < match1.start(0):
              match = re.search('OPS_ACC_MD\d',text[pos:])
              pos = pos + match.start(0)
              pos2 = text[pos+10:].find('(')
              num = int(text[pos+10:pos+10+pos2])
              if num <> n:
                print 'Access mismatch in '+name+', arg '+str(n)+'('+arg_list[n]+') with OPS_ACC_MD'+str(num)
              pos = pos+10+pos2
          else:
            match = re.search('OPS_ACC_MD\d',text[pos:])
            pos = pos + match.start(0)
            pos2 = text[pos+10:].find('(')
            num = int(text[pos+10:pos+10+pos2])
            if num <> n:
              print 'Access mismatch in '+name+', arg '+str(n)+'('+arg_list[n]+') with OPS_ACC_MD'+str(num)
            pos = pos+10+pos2
        else:
          if match1 <> None:
            match = re.search('OPS_ACC\d',text[pos:])
            pos = pos + match.start(0)
            pos2 = text[pos+7:].find('(')
            num = int(text[pos+7:pos+7+pos2])
            if num <> n:
              print 'Access mismatch in '+name+', arg '+str(n)+'('+arg_list[n]+') with OPS_ACC'+str(num)
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

