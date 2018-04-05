
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
import os

def comm(line):
  prefix = ' '*config.depth
  if len(line) == 0:
    config.file_text +='\n'
  else:
    config.file_text +=prefix+'!'+line+'\n'

def code(text):
  prefix = ''
  if len(text) != 0:
    prefix = ' '*config.depth

  config.file_text += prefix+text+'\n'

def DO(i,start,finish):
  code('DO '+i+' = '+start+', '+finish)
  config.depth += 2

def ENDDO():
  config.depth -= 2
  code('END DO')

def IF(line):
  code('IF ('+ line + ') THEN')
  config.depth += 2

def ELSEIF(line):
  code('ELSEIF ('+ line + ') THEN')
  config.depth += 2

def ELSE():
  code('ELSE')
  config.depth += 2

def ENDIF():
  config.depth -= 2
  code('ENDIF')

def DOWHILE(line):
  code('DO WHILE ('+line+' )')
  config.depth += 2


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


def comment_remover(text):
    """Remove comments from text"""

    def replacer(match):
        s = match.group(0)
        if s.startswith('!'):  #only F90 type comments
            return ''
        else:
            return s
    pattern = re.compile(
        r'!.*?$',
        re.DOTALL | re.MULTILINE
    )
    return re.sub(pattern, replacer, text)

def find_subroutine(fun_name):
    subr_file =  os.popen('grep -Rilw --include "*.F95" --include "*.F90" --include "*.F" --include "*.inc" --exclude "*kernel.*" "subroutine '+fun_name+'" . | head -n1').read().strip()
    if (len(subr_file) == 0) or (not os.path.exists(subr_file)):
      print 'Error, subroutine '+fun_name+' implementation not found in files, check parser!'
      exit(1)
    #read the file and find the implementation
    subr_fileh = open(subr_file,'r')
    subr_fileh_text = subr_fileh.read()
    subr_fileh_text = re.sub('\n*!.*\n','\n',subr_fileh_text)
    subr_fileh_text = re.sub('!.*\n','\n',subr_fileh_text)
    subr_begin = subr_fileh_text.lower().find('subroutine '+fun_name.lower())
    #function name as spelled in the file
    fun_name = subr_fileh_text[subr_begin+11:subr_begin+11+len(fun_name)]
    subr_end = subr_fileh_text[subr_begin:].lower().find('end subroutine')
    if subr_end<0:
      print 'Error, could not find string "end subroutine" for implemenatation of '+fun_name+' in '+subr_file
      exit(-1)
    subr_end= subr_begin+subr_end
    subr_text =  subr_fileh_text[subr_begin:subr_end+14]
    if subr_text[10:len(subr_text)-20].lower().find('subroutine')>=0:
      print 'Error, could not properly parse subroutine, more than one encompassed '+fun_name+' in '+subr_file
      #print subr_text
      exit(-1)
    return subr_text

def convert_freeform(text):
    text = comment_remover(text)
    loc = 0
    textl = text.splitlines()
    for i in range(0,len(textl)):
        if i>0 and len(textl[i].strip()) > 0 and \
                textl[i].strip()[0] == '&' and len(textl[i-1].strip()) > 0 and \
                textl[i-1].strip()[-1] <> '&':
            textl[i-1] = textl[i-1] + ' &'
    text = textl[0]
    for i in range(1,len(textl)):
        text = text + '\n' + textl[i]
    return text

