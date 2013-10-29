"""
OPS OpenMP code generator

This routine is called by ops.py which parses the input files

It produces a file xxx_omp_kernel.cpp for each kernel,
plus a master kernel file

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
  prefix = ''
  if len(text) != 0:
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

def ops_gen_openmp(master, date, kernels):

  global dims, stens
  global g_m, file_text, depth

  OPS_ID   = 1;  OPS_GBL   = 2;

  OPS_READ = 1;  OPS_WRITE = 2;  OPS_RW  = 3;
  OPS_INC  = 4;  OPS_MAX   = 5;  OPS_MIN = 6;

  accsstring = ['OPS_READ','OPS_WRITE','OPS_RW','OPS_INC','OPS_MAX','OPS_MIN' ]


##########################################################################
#  create new kernel file
##########################################################################

  for nk in range (0,len(kernels)):
    arg_typ  = kernels[nk]['arg_type']
    name  = kernels[nk]['name']
    nargs = kernels[nk]['nargs']
    dim   = kernels[nk]['dim']
    dims  = kernels[nk]['dims']
    stens = kernels[nk]['stens']
    var   = kernels[nk]['var']
    accs  = kernels[nk]['accs']
    typs  = kernels[nk]['typs']


##########################################################################
#  start with seq kernel function
##########################################################################

    g_m = 0;
    file_text = ''
    depth = 0
    n_per_line = 4

    i = name.find('kernel')
    name2 = name[0:i-1]
    #print name2

    reduction = False
    ng_args = 0

    for n in range (0, nargs):
      if arg_typ[n] == 'ops_arg_gbl':
        reduction = True
      else:
        ng_args = ng_args + 1


    #backend functions that should go to the sequential backend lib
    code('#include "ops_lib_openmp.h"')

    code('#ifdef _OPENMP')
    code('#include <omp.h>')
    code('#endif')

    comm('user function')
    code('#include "'+name2+'_kernel.h"')
    comm('')
    comm(' host stub function')

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

    text ='ops_arg args['+str(nargs)+'] = {'
    for n in range (0, nargs):
      text = text +' arg'+str(n)
      if nargs <> 1 and n != nargs-1:
        text = text +','
      else:
        text = text +'};\n'
      if n%n_per_line == 5 and n <> nargs-1:
        text = text +'\n                    '
    code(text);

    if reduction == True:
      for n in range (0, nargs):
        if arg_typ[n] == 'ops_arg_gbl':
          code('double*arg'+str(n)+'h = (double *)arg'+str(n)+'.data;')

    code('')
    comm('setup offsets');
    code('int  offs['+str(nargs)+'][2];')
    code('')
    FOR('i','0',str(nargs))
    IF('args[i].stencil!=NULL')
    code('offs[i][0] = 1;  //unit step in x dimension')
    code('offs[i][1] = ops_offs_set(range[0],range[2]+1, args[i]) - ops_offs_set(range[1],range[2], args[i]);')

    comm('stride in y as x stride is 0')
    IF('args[i].stencil->stride[0] == 0')
    code('offs[i][0] = 0;')
    code('offs[i][1] = args[i].dat->block_size[0];')
    ENDIF();
    comm('stride in x as y stride is 0')
    ELSEIF('args[i].stencil->stride[1] == 0')
    code('offs[i][0] = 1;')
    code('offs[i][1] = -( range[1] - range[0] );')
    ENDIF()
    ENDIF()
    ENDFOR()

    code('')
    code('#ifdef _OPENMP')
    code('int nthreads = omp_get_max_threads( );')
    code('#else')
    code('int nthreads = 1;')
    code('#endif')
    code('')

    #setup reduction variables
    if reduction == True:
      comm('allocate and initialise arrays for global reduction')
      comm('assumes a max of 64 threads with a cacche line size of 64 bytes')
      for n in range (0, nargs):
        if arg_typ[n] == 'ops_arg_gbl':
          code((str(typs[n]).replace('"','')).strip()+' arg_gbl'+str(n)+'['+dims[n]+' * 64 * 64];')

      FOR('thr','0','nthreads')
      for n in range (0, nargs):
        if arg_typ[n] == 'ops_arg_gbl':
          FOR('d', '0',dims[n])
          code('arg_gbl'+str(n)+'[64*thr] = *arg'+str(n)+'h;')
          ENDFOR()
      ENDFOR()

    code('')
    code('int y_size = range[3]-range[2];')
    code('')

    code('#pragma omp parallel for')
    FOR('thr','0','nthreads')
    code('')
    code('char **p_a['+str(nargs)+'];')
    code('')
    comm('store index of non_gbl args')
    code('int g = '+str(ng_args)+';')
    text = 'int non_gbl['+str(ng_args)+'] = {'
    for n in range (0, ng_args):
      if arg_typ[n] == 'ops_arg_dat':
        text = text + str(n)

      if nargs <> 1 and n != ng_args-1:
        text = text +', '
      else:
        text = text +'};\n'
      if n%n_per_line == 5 and n <> ng_args-1:
        text = text+'\n'
    code(text);

    for n in range (0, nargs):
      if arg_typ[n] == 'ops_arg_dat':
        code('p_a['+str(n)+'] = (char **)malloc(args['+str(n)+'].stencil->points * sizeof(char *));')

    code('')

    code('int start = range[2] + ((y_size-1)/nthreads+1)*thr;')
    code('int finish = range[2] +  MIN(((y_size-1)/nthreads+1)*(thr+1),y_size);')

    comm('')
    comm('set up initial pointers')

    for n in range (0, nargs):
      if arg_typ[n] == 'ops_arg_dat':
        code('ops_arg_set(range[0], start, args['+str(n)+'], p_a['+str(n)+']);')
    code('')

    FOR('n_y','start','finish')
    FOR('n_x','range[0]','range[1]')

    comm('call kernel function, passing in pointers to data')
    code('')
    n_per_line = 2
    text = name+'( '
    for n in range (0, nargs):
      if arg_typ[n] == 'ops_arg_dat':
        text = text +' ('+(str(typs[n]).replace('"','')).strip()+' **)p_a['+str(n)+']'
      else:
        text = text +' &arg_gbl'+str(n)+'[64*thr]'

      if nargs <> 1 and n != nargs-1:
        text = text + ','
      else:
        text = text +' );\n'
      if n%n_per_line == 1 and n <> nargs-1:
        text = text +'\n          '
    code(text);

    code('int a = 0;')
    comm('shift pointers to data x direction')
    FOR('i','0','g')
    code('a = non_gbl[i];')
    FOR('np','0','args[a].stencil->points')
    code('p_a[a][np] = p_a[a][np] + (args[a].dat->size * offs[a][0]);')
    ENDFOR()
    ENDFOR()
    ENDFOR()

    code('')
    code('int a = 0;')
    comm('shift pointers to data y direction')
    FOR('i','0','g')
    code('a = non_gbl[i];')
    FOR('np','0','args[a].stencil->points')
    code('p_a[a][np] = p_a[a][np] + (args[a].dat->size * offs[a][1]);')
    ENDFOR()
    ENDFOR()


    ENDFOR()
    code('')

    for n in range (0, nargs):
      if arg_typ[n] == 'ops_arg_dat':
        code('free(p_a['+str(n)+']);')

    ENDFOR()

    #generate code for combining the reductions
    if reduction == True:
      code('')
      comm(' combine reduction data')
      FOR('thr','0','nthreads')
      for n in range (0, nargs):
        if arg_typ[n] == 'ops_arg_gbl':
          FOR('d','0',dims[n])
          if accs[n] == OPS_INC:
            code('arg'+str(n)+'h[0] += arg_gbl'+str(n)+'[64*thr];')
          elif accs[n] == OPS_MIN:
            code('arg'+str(n)+'h[0] = MIN(arg'+str(n)+'h[0], arg_gbl'+str(n)+'[64*thr]);')
          elif accs[n] == OPS_MAX:
            code('arg'+str(n)+'h[0] = MAX(arg'+str(n)+'h[0], arg_gbl'+str(n)+'[64*thr]);')
          elif accs[n] == OPS_WRITE:
            code('if(arg_gbl'+str(n)+'[64*thr] != 0.0) arg'+str(n)+'h[0] = arg_gbl'+str(n)+'[64*thr];')
          ENDFOR()
      ENDFOR()

    depth = depth - 2
    code('}')


##########################################################################
#  output individual kernel file
##########################################################################
    fid = open(name+'_omp_kernel.cpp','w')
    date = datetime.datetime.now()
    fid.write('//\n// auto-generated by ops.py on '+date.strftime("%Y-%m-%d %H:%M")+'\n//\n\n')
    fid.write(file_text)
    fid.close()

# end of main kernel call loop

##########################################################################
#  output one master kernel file
##########################################################################

  file_text =''
  comm('header')
  code('#include "ops_lib_cpp.h"')
  code('')

  comm('user kernel files')

  kernel_name_list = []

  for nk in range(0,len(kernels)):
    if kernels[nk]['name'] not in kernel_name_list :
      code('#include "'+kernels[nk]['name']+'_omp_kernel.cpp"')
      kernel_name_list.append(kernels[nk]['name'])

  master = master.split('.')[0]
  fid = open(master.split('.')[0]+'_omp_kernels.cpp','w')
  fid.write('//\n// auto-generated by op2.py on '+date.strftime("%Y-%m-%d %H:%M")+'\n//\n\n')
  fid.write(file_text)
  fid.close()
