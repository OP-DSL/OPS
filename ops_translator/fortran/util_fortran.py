
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
