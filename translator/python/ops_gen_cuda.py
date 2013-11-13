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

def ops_gen_cuda(master, date, kernels):

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

    #parse stencil to locate strided access
    stride = [1] * nargs * 2
    #print stride
    for n in range (0, nargs):
      if str(stens[n]).find('STRID2D_X') > 0:
        stride[2*n+1] = 0
      elif str(stens[n]).find('STRID2D_Y') > 0:
        stride[2*n] = 0


    reduct = 0
    for n in range (0, nargs):
      if arg_typ[n] == 'ops_arg_gbl' and accs[n] <> OPS_READ:
        reduct = 1


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



##########################################################################
#  generate constants and MACROS
##########################################################################

    #for n in range (0, nargs):
    #  code('__constant__ int xdim'+str(n)+'_accel;')
    #code('__constant__ double dt_device;')
    #code('')

    #code('#define OPS_ACC_MACROS')
    #for n in range (0, nargs):
    #  code('#define OPS_ACC'+str(n)+'(x,y) (x+xdim'+str(n)+'_accel*(y))')
    #code('')

##########################################################################
#  generate headder
##########################################################################

    comm('user function')
    code('#include "'+name2+'_kernel.h"')
    comm('')

##########################################################################
#  generate cuda kernel wrapper function
##########################################################################

    code('__global__ void ops_'+name+'(')
    for n in range (0, nargs):
      if arg_typ[n] == 'ops_arg_dat'and accs[n] == OPS_READ:
        code('const '+(str(typs[n]).replace('"','')).strip()+'* __restrict arg'+str(n)+',')
      elif arg_typ[n] == 'ops_arg_dat'and (accs[n] == OPS_WRITE or accs[n] == OPS_RW) :
        code((str(typs[n]).replace('"','')).strip()+'* __restrict arg'+str(n)+',')

    code('int* fields_device,')
    code('int size0,')
    code('int size1 ){')
    depth = depth + 2
    code('')
    code('int idx_y = blockDim.y * blockIdx.y + threadIdx.y;')
    code('int idx_x = blockDim.x * blockIdx.x + threadIdx.x;')
    code('')
    for n in range (0, nargs):
      code('arg'+str(n)+' += idx_x * '+str(stride[2*n])+' + idx_y * '+str(stride[2*n+1])+' * xdim'+str(n)+'_device;')

    n_per_line = 5
    IF('idx_x < size0 && idx_y < size1')
    text = name+'('
    for n in range (0, nargs):
      text = text +'arg'+str(n)+' '
      if nargs <> 1 and n != nargs-1:
        text = text +','
      else:
        text = text +', fields_device);'
      if n%n_per_line == 3 and n <> nargs-1:
         text = text +'\n'
    code(text)
    ENDIF()
    depth = depth - 2
    code('}')


##########################################################################
#  now host stub
##########################################################################
    code('')
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


    code('')
    if reduction == True:
      for n in range (0, nargs):
        if arg_typ[n] == 'ops_arg_gbl':
          code('double*arg'+str(n)+'h = (double *)arg'+str(n)+'.data;')

    #setup reduction variables
    if reduction == True:
      comm('allocate and initialise arrays for global reduction')
      comm('assumes a max of 64 threads with a cacche line size of 64 bytes')
      for n in range (0, nargs):
        if arg_typ[n] == 'ops_arg_gbl':
          code((str(typs[n]).replace('"','')).strip()+' arg_gbl'+str(n)+'['+dims[n]+' * 64 * 64];')


    code('')
    code('int x_size = range[1]-range[0];')
    code('int y_size = range[3]-range[2];')
    code('')

    for n in range (0, nargs):
      if arg_typ[n] == 'ops_arg_dat':
        code('int xdim'+str(n)+' = args['+str(n)+'].dat->block_size[0];')
    code('')

    for n in range (0, nargs):
      if arg_typ[n] == 'ops_arg_dat':
        code('cudaMemcpyToSymbol( xdim'+str(n)+'_device, &xdim'+str(n)+', sizeof(int) );')

    #these constant copy needs to be stripped out to the headder file
    code('cudaMemcpyToSymbol( dt_device,  &dt, sizeof(double) );')
    #code('cudaMemcpyToSymbol( fields_device, fields , sizeof(int)*NUM_FIELDS, cudaMemcpyHostToDevice);')

    code('cudaMalloc((void **)&fields_device, sizeof(int)*NUM_FIELDS);')
    code('cudaMemcpy(fields_device, fields , sizeof(int)*NUM_FIELDS, cudaMemcpyHostToDevice);')

    code('')

    code('char *p_a['+str(nargs)+'];')
    code('')


    comm('')
    comm('set up initial pointers')
    for n in range (0, nargs):
      if arg_typ[n] == 'ops_arg_dat':
        code('p_a['+str(n)+'] = &args['+str(n)+'].data_d[')
        code('+ args['+str(n)+'].dat->size * args['+str(n)+'].dat->block_size[0] * ( range[2] * '+str(stride[2*n+1])+' - args['+str(n)+'].dat->offset[1] )')
        code('+ args['+str(n)+'].dat->size * ( range[0] * '+str(stride[2*n])+' - args['+str(n)+'].dat->offset[0] ) ];')
        code('')
      else:
        code('p_a['+str(n)+'] = (char *)args['+str(n)+'].data;')
        code('')

    code('')
    code('ops_halo_exchanges_cuda(args, '+str(nargs)+');')
    code('')
    code('int block_size = 16;')
    code('dim3 grid( (x_size-1)/block_size+ 1, (y_size-1)/block_size + 1, 1);')
    code('dim3 block(16,16,1);')
    code('')

    #for n in range (0, nargs):
    #  code('arg'+str(n)+'.dat->dirty_hd = 1;')

    comm('call kernel wrapper function, passing in pointers to data')
    n_per_line = 2
    text = 'ops_'+name+'<<<grid, block >>> ( '
    for n in range (0, nargs):
      if arg_typ[n] == 'ops_arg_dat':
        text = text +' ('+(str(typs[n]).replace('"','')).strip()+' *)p_a['+str(n)+'],'
      else:
        text = text +' &arg_gbl'+str(n)+'[64*thr]'

      if n%n_per_line == 1 and n <> nargs-1:
        text = text +'\n          '
    text = text +'fields_device, x_size, y_size);'
    code(text);

    code('')

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
            code('if(arg_gbl'+str(n)+'[64*thr] != 0.0) arg'+str(n)+'h[0] += arg_gbl'+str(n)+'[64*thr];')
          ENDFOR()
      ENDFOR()

    code('cudaDeviceSynchronize();')
    code('ops_set_dirtybit_cuda(args, '+str(nargs)+');')
    #code('ops_halo_exchanges(args, '+str(nargs)+');')

    depth = depth - 2
    code('}')


##########################################################################
#  output individual kernel file
##########################################################################
    fid = open(name+'_cuda_kernel.cu','w')
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

  #constants for macros
  for i in range(0,20):
    code('int xdim'+str(i)+';')
  code('')

  comm('user kernel files')

  kernel_name_list = []

  for nk in range(0,len(kernels)):
    if kernels[nk]['name'] not in kernel_name_list :
      code('#include "'+kernels[nk]['name']+'_cuda_kernel.cu"')
      kernel_name_list.append(kernels[nk]['name'])

  master = master.split('.')[0]
  fid = open(master.split('.')[0]+'_cuda_kernels.cu','w')
  fid.write('//\n// auto-generated by op2.py on '+date.strftime("%Y-%m-%d %H:%M")+'\n//\n\n')
  fid.write(file_text)
  fid.close()
