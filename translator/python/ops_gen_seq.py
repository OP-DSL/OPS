"""
OPS Sequential code generator

This routine is called by ops.py which parses the input files

It produces a file xxx_seq_kernel.cpp for each kernel,

"""

import re
import datetime

def comm(line):
  global file_text, FORTRAN, CPP
  global depth
  prefix = ' '*depth
  if len(line) == 0:
    file_text +='\n'
  else:
    file_text +=prefix+'//'+line+'\n'

def code(text):
  global file_text, g_m
  global depth
  prefix = ' '*depth
  #file_text += prefix+rep(text,g_m)+'\n'
  file_text += prefix+text+'\n'

def FOR(i,start,finish):
  global file_text
  global depth
  code('for ( int '+i+'='+start+'; '+i+'<'+finish+'; '+i+'++ ){')
  depth += 2

def WHILE(line):
  global file_text
  global depth
  code('while ( '+ line+ ' ){')
  depth += 2

def ENDWHILE():
  global file_text
  global depth
  depth -= 2
  code('}')

def ENDFOR():
  global file_text
  global depth
  depth -= 2
  code('}')

def IF(line):
  global file_text
  global depth
  code('if ('+ line + ') {')
  depth += 2

def ELSEIF(line):
  global file_text
  global depth
  code('else if ('+ line + ') {')
  depth += 2

def ELSE(line):
  global file_text
  global depth
  code('else {')
  depth += 2

def ENDIF():
  global file_text
  global depth
  depth -= 2
  code('}')

def ops_gen_seq(master, date, kernels):

  global dims, stens
  global g_m, file_text, depth

  OP_ID   = 1;  OP_GBL   = 2;  OP_MAP = 3;

  OP_READ = 1;  OP_WRITE = 2;  OP_RW  = 3;
  OP_INC  = 4;  OP_MAX   = 5;  OP_MIN = 6;

  accsstring = ['OP_READ','OP_WRITE','OP_RW','OP_INC','OP_MAX','OP_MIN' ]


##########################################################################
#  create new kernel file
##########################################################################

  for nk in range (0,len(kernels)):

    name  = kernels[nk]['name']
    nargs = kernels[nk]['nargs']
    dim   = kernels[nk]['dim']
    dims  = kernels[nk]['dims']
    stens = kernels[nk]['stens']
    var   = kernels[nk]['var']
    accs  = kernels[nk]['accs']

##########################################################################
#  start with seq kernel function
##########################################################################

    g_m = 0;
    file_text = ''
    depth = 0
    n_per_line = 4

    comm('user function')
    code('#include "'+name+'.h"')
    comm('')
    comm(' host stub function          ')
    code('void ops_par_loop_'+name+'(char const *name, int dim, int* range,')
    text = ''
    for n in range (0, nargs):

      text = text +' ops_arg arg'+str(n)
      if nargs <> 1 and n != nargs-1:
        text = text +','
      else:
        text = text +') {'
      if n%n_per_line == 3 and n <> nargs-1:
         text = text +'\n'
    code(text);
    depth = 2
    code('');
    code('char **p_a['+str(nargs)+'];')
    code('int  offs['+str(nargs)+'][2];')
    code('int  count[dim];\n')

    text ='ops_arg args['+str(nargs)+'] = {'
    for n in range (0, nargs):
      text = text +' arg'+str(n)
      if nargs <> 1 and n != nargs-1:
        text = text +','
      else:
        text = text +'};\n\n'
      if n%n_per_line == 5 and n <> nargs-1:
        text = text +'\n                    '
    code(text);

    FOR('i','0',str(nargs))
    IF('args[i].stencil!=NULL')
    code('offs[i][0] = 1;  //unit step in x dimension')
    code('offs[i][1] = ops_offs_set(range[0],range[2]+1, args[i]) - ops_offs_set(range[1],range[2], args[i]) +1;')

    comm('stride in y as x stride is 0')
    IF('args[i].stencil->stride[0] == 0')
    code('offs[i][0] = 0;')
    code('offs[i][1] = args[i].dat->block_size[0];')
    ENDIF();
    comm('stride in x as y stride is 0')
    ELSEIF('args[i].stencil->stride[1] == 0')
    code('offs[i][0] = 1;')
    code('offs[i][1] = -( range[1] - range[0] ) //+1;')
    ENDIF()
    ENDIF()
    ENDFOR()

    comm('store index of non_gbl args\n')
    text = 'int non_gbl['+str(nargs)+'] = {'
    for n in range (0, nargs):
        text = text + '0'
        if nargs <> 1 and n != nargs-1:
          text = text +', '
        else:
          text = text +'};\n'
        if n%n_per_line == 5 and n <> nargs-1:
          text = text+'\n'
    text = text+'  int g = 0;\n'
    code(text);

    FOR('i','0',str(nargs))
    IF('args[i].argtype == OPS_ARG_DAT')
    code('p_a[i] = (char **)malloc(args[i].stencil->points * sizeof(char *));')
    code('non_gbl[g++] = i;')
    ENDIF()
    ELSEIF('args[i].argtype == OPS_ARG_GBL)')
    code('p_a[i] = (char **)malloc(args[i].dim * sizeof(char *));')
    ENDIF()
    ENDFOR()

    code('int total_range = 1;')
    FOR('m','0','dim')
    comm('number in each dimension')
    code('count[m] = range[2*m+1]-range[2*m];')
    code('total_range *= count[m];')
    ENDFOR()
    comm('extra in last to ensure correct termination')
    code('count[dim-1]++;\n\n')

    comm('set up initial pointers')

    code('ops_args_set(range[0], range[2], '+str(nargs)+', args,p_a); //set up the initial possition \n\n')

    FOR('nt','0','total_range')

    comm('call kernel function, passing in pointers to data')
    text = '\n    '+name+'( '
    for n in range (0, nargs):
        text = text +' (double **)p_a['+str(n)+']'
        if nargs <> 1 and n != nargs-1:
          text = text + ','
        else:
          text = text +' );\n'
        if n%n_per_line == 2 and n <> nargs-1:
          text = text +'\n          '
    code(text);
    comm('decrement counter')
    code('count[0]--;\n')
    comm('max dimension with changed index')
    code('int m = 0;')

    WHILE('(count[m]==0)')
    code('count[m] = range[2*m+1]-range[2*m]; // reset counter')
    code('m++;                                // next dimension')
    code('count[m]--;                         // decrement counter')
    ENDWHILE()
    code('')

    code('int a = 0;')
    comm('shift pointers to data')
    FOR('i','0','g')
    code('a = non_gbl[i];')
    FOR('np','0','args[a].stencil->points')
    code('p_a[a][np] = p_a[a][np] + (args[a].dat->size * offs[a][m]);')
    ENDFOR()
    ENDFOR()
    ENDFOR()
    code('')

    FOR('i','0',str(nargs))
    code('free(p_a[i]);')
    ENDFOR()

    depth = depth - 2
    code('}')

##########################################################################
#  output individual kernel file
##########################################################################
    fid = open(name+'_seq_kernel.cpp','w')
    date = datetime.datetime.now()
    fid.write('//\n// auto-generated by ops.py on '+date.strftime("%Y-%m-%d %H:%M")+'\n//\n\n')
    fid.write(file_text)
    fid.close()

# end of main kernel call loop
