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
      elif arg_typ[n] == 'ops_arg_gbl':
        if accs[n] == OPS_READ:
          code('const '+(str(typs[n]).replace('"','')).strip()+'* __restrict arg'+str(n)+',')
        else:
          code((str(typs[n]).replace('"','')).strip()+'* __restrict arg'+str(n)+',')

    code('int size0,')
    code('int size1 ){')
    depth = depth + 2

    #local variable to hold reductions on GPU
    code('')
    for n in range (0, nargs):
      if arg_typ[n] == 'ops_arg_gbl' and accs[n] != OPS_READ:
        code((str(typs[n]).replace('"','')).strip()+' arg'+str(n)+'_l['+str(dims[n])+'];')

    # set local variables to 0 if OPS_INC, INF if OPS_MIN, -INF
    for n in range (0, nargs):
      if arg_typ[n] == 'ops_arg_gbl' and accs[n] == OPS_INC:
        code('for (int d=0; d<'+str(dims[n])+'; d++) arg'+str(n)+'_l[d] = ZERO_'+(str(typs[n]).replace('"','')).strip()+';')
      if arg_typ[n] == 'ops_arg_gbl' and accs[n] == OPS_MIN:
        code('for (int d=0; d<'+str(dims[n])+'; d++) arg'+str(n)+'_l[d] = INFINITY_'+(str(typs[n]).replace('"','')).strip()+';')
      if arg_typ[n] == 'ops_arg_gbl' and accs[n] == OPS_MAX:
        code('for (int d=0; d<'+str(dims[n])+'; d++) arg'+str(n)+'_l[d] = -INFINITY_'+(str(typs[n]).replace('"','')).strip()+';')


    code('')
    code('int idx_y = blockDim.y * blockIdx.y + threadIdx.y;')
    code('int idx_x = blockDim.x * blockIdx.x + threadIdx.x;')
    code('')
    for n in range (0, nargs):
      if arg_typ[n] == 'ops_arg_dat':
        code('arg'+str(n)+' += idx_x * '+str(stride[2*n])+' + idx_y * '+str(stride[2*n+1])+' * xdim'+str(n)+'_device;')

    code('')
    n_per_line = 5
    IF('idx_x < size0 && idx_y < size1')
    text = name+'('
    for n in range (0, nargs):
      if arg_typ[n] == 'ops_arg_dat':
        text = text +'arg'+str(n)
      elif arg_typ[n] == 'ops_arg_gbl' and accs[n] == OPS_READ:
        text = text +'arg'+str(n)
      elif arg_typ[n] == 'ops_arg_gbl' and accs[n] != OPS_READ:
        text = text +'arg'+str(n)+'_l'

      if nargs <> 1 and n <> nargs-1:
        if n%n_per_line <> 3:
          text = text +', '
        else:
          text = text +','
      else:
        text = text +');'
      if n%n_per_line == 3 and n <> nargs-1:
         text = text +'\n                   '
    code(text)
    ENDIF()

    #reduction accross blocks
    for n in range (0, nargs):
      if arg_typ[n] == 'ops_arg_gbl' and accs[n] == OPS_INC:
        code('for (int d=0; d<'+str(dims[n])+'; d++)')
        code('  ops_reduction<OPS_INC>(&arg'+str(n)+'[d+blockIdx.x + blockIdx.y*gridDim.x],arg'+str(n)+'_l[d]);')
      if arg_typ[n] == 'ops_arg_gbl' and accs[n] == OPS_MIN:
        code('for (int d=0; d<'+str(dims[n])+'; d++)')
        code('  ops_reduction<OPS_MIN>(&arg'+str(n)+'[d+blockIdx.x + blockIdx.y*gridDim.x],arg'+str(n)+'_l[d]);')
      if arg_typ[n] == 'ops_arg_gbl' and accs[n] == OPS_MAX:
        code('for (int d=0; d<'+str(dims[n])+'; d++)')
        code('  ops_reduction<OPS_MAX>(&arg'+str(n)+'[d+blockIdx.x + blockIdx.y*gridDim.x],arg'+str(n)+'_l[d]);')


    code('')
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
    #code('cudaMemcpyToSymbol( dt_device,  &dt, sizeof(double) );')
    code('cudaMemcpyToSymbol( g_small_device,  &g_small, sizeof(double) );')
    code('cudaMemcpyToSymbol( g_big_device,  &g_big, sizeof(double) );')
    code('cudaMemcpyToSymbol( dtc_safe_device,  &dtc_safe, sizeof(double) );')
    code('cudaMemcpyToSymbol( dtu_safe_device,  &dtu_safe, sizeof(double) );')
    code('cudaMemcpyToSymbol( dtv_safe_device,  &dtv_safe, sizeof(double) );')
    code('cudaMemcpyToSymbol( dtdiv_safe_device,  &dtdiv_safe, sizeof(double) );')

    code('')

    #setup reduction variables
    code('')
    for n in range (0, nargs):
        if arg_typ[n] == 'ops_arg_gbl':
          code(''+(str(typs[n]).replace('"','')).strip()+' *arg'+str(n)+'h = ('+(str(typs[n]).replace('"','')).strip()+' *)arg'+str(n)+'.data;')
    code('')

    #set up CUDA grid and thread blocks
    code('int block_size = 16;')
    code('dim3 grid( (x_size-1)/block_size+ 1, (y_size-1)/block_size + 1, 1);')
    code('dim3 block(block_size,block_size,1);')
    code('')

    GBL_READ = False
    GBL_INC = False
    GBL_MAX = False
    GBL_MIN = False

    for n in range (0, nargs):
      if arg_typ[n] == 'ops_arg_gbl':
        if accs[n] == OPS_READ:
          GBL_READ = True
        if accs[n] == OPS_INC:
          GBL_INC = True
        if accs[n] == OPS_MAX:
          GBL_MAX = True
        if accs[n] == OPS_MIN:
          GBL_MIN = True

    if GBL_INC == True or GBL_MIN == True or GBL_MAX == True:
      code('int nblocks = ((x_size-1)/block_size+ 1)*((y_size-1)/block_size + 1);')
      code('int maxblocks = nblocks;')
      code('int reduct_bytes = 0;')
      code('int reduct_size = 0;')
      code('')

    if GBL_READ == True:
      code('int consts_bytes = 0;')
      code('')

    for n in range (0, nargs):
      if arg_typ[n] == 'ops_arg_gbl':
        if accs[n] == OPS_READ:
          code('consts_bytes += ROUND_UP('+str(dims[n])+'*sizeof('+(str(typs[n]).replace('"','')).strip()+'));')
        else:
          code('reduct_bytes += ROUND_UP(maxblocks*'+str(dims[n])+'*sizeof('+(str(typs[n]).replace('"','')).strip()+'));')
          code('reduct_size = MAX(reduct_size,sizeof('+(str(typs[n]).replace('"','')).strip()+')*'+str(dims[n])+');')
    code('')

    if GBL_READ == True:
      code('reallocConstArrays(consts_bytes);')
    if GBL_INC == True or GBL_MIN == True or GBL_MAX == True:
      code('reallocReductArrays(reduct_bytes);')
      code('reduct_bytes = 0;')
      code('')

    for n in range (0, nargs):
      if arg_typ[n] == 'ops_arg_gbl' and accs[n] != OPS_READ:
        code('arg'+str(n)+'.data = OPS_reduct_h + reduct_bytes;')
        code('arg'+str(n)+'.data_d = OPS_reduct_d + reduct_bytes;')
        code('for (int b=0; b<maxblocks; b++)')
        if accs[n] == OPS_INC:
          code('for (int d=0; d<'+str(dims[n])+'; d++) (('+(str(typs[n]).replace('"','')).strip()+' *)arg'+str(n)+'.data)[d+b*'+str(dims[n])+'] = ZERO_'+(str(typs[n]).replace('"','')).strip()+';')
        if accs[n] == OPS_MAX:
          code('for (int d=0; d<'+str(dims[n])+'; d++) (('+(str(typs[n]).replace('"','')).strip()+' *)arg'+str(n)+'.data)[d+b*'+str(dims[n])+'] = -INFINITY_'+(str(typs[n]).replace('"','')).strip()+';')
        if accs[n] == OPS_MIN:
          code('for (int d=0; d<'+str(dims[n])+'; d++) (('+(str(typs[n]).replace('"','')).strip()+' *)arg'+str(n)+'.data)[d+b*'+str(dims[n])+'] = INFINITY_'+(str(typs[n]).replace('"','')).strip()+';')
        code('reduct_bytes += ROUND_UP(maxblocks*'+str(dims[n])+'*sizeof('+(str(typs[n]).replace('"','')).strip()+'));')
        code('')

    code('')

    for n in range (0, nargs):
      if arg_typ[n] == 'ops_arg_gbl':
        if accs[n] == OPS_READ:
          code('consts_bytes = 0;')
          code('arg'+str(n)+'.data = OPS_consts_h + consts_bytes;')
          code('arg'+str(n)+'.data_d = OPS_consts_d + consts_bytes;')
          code('for (int d=0; d<'+str(dims[n])+'; d++) (('+(str(typs[n]).replace('"','')).strip()+' *)arg'+str(n)+'.data)[d] = arg'+str(n)+'h[d];')
          code('consts_bytes += ROUND_UP('+str(dims[n])+'*sizeof(int));')
          code('mvConstArraysToDevice(consts_bytes);')

    if GBL_INC == True or GBL_MIN == True or GBL_MAX == True:
      code('mvReductArraysToDevice(reduct_bytes);')


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


    code('')
    code('ops_halo_exchanges_cuda(args, '+str(nargs)+');')
    code('')


    #set up shared memory for reduction
    if GBL_INC == True or GBL_MIN == True or GBL_MAX == True:
       code('int nshared = 0;')
       code('int nthread = block_size*block_size;')
       code('')
    for n in range (0, nargs):
      if arg_typ[n] == 'ops_arg_gbl' and accs[n] != OPS_READ:
        code('nshared = MAX(nshared,sizeof('+(str(typs[n]).replace('"','')).strip()+')*'+str(dims[n])+');')
    code('')
    if GBL_INC == True or GBL_MIN == True or GBL_MAX == True:
      code('nshared = MAX(nshared*nthread,reduct_size*nthread);')
      code('')


   #kernel call
    comm('call kernel wrapper function, passing in pointers to data')
    n_per_line = 2
    if GBL_INC == True or GBL_MIN == True or GBL_MAX == True:
      text = 'ops_'+name+'<<<grid, block, nshared >>> ( '
    else:
      text = 'ops_'+name+'<<<grid, block >>> ( '
    for n in range (0, nargs):
      if arg_typ[n] == 'ops_arg_dat':
        text = text +' ('+(str(typs[n]).replace('"','')).strip()+' *)p_a['+str(n)+'],'
      else:
        text = text +' ('+(str(typs[n]).replace('"','')).strip()+' *)arg'+str(n)+'.data_d, '

      if n%n_per_line == 1 and n <> nargs-1:
        text = text +'\n          '
    text = text +'x_size, y_size);'
    code(text);

    code('')

    if GBL_INC == True or GBL_MIN == True or GBL_MAX == True:
      code('mvReductArraysToHost(reduct_bytes);')

    for n in range (0, nargs):
      if arg_typ[n] == 'ops_arg_gbl' and accs[n] != OPS_READ:
        FOR('b','0','maxblocks')
        FOR('d','0',str(dims[n]))
        if accs[n] == OPS_INC:
          code('arg'+str(n)+'h[d] = arg'+str(n)+'h[d] + ((double *)arg'+str(n)+'.data)[d+b*'+str(dims[n])+'];')
        elif accs[n] == OPS_MAX:
          code('arg'+str(n)+'h[d] = MAX(arg'+str(n)+'h[d],((double *)arg'+str(n)+'.data)[d+b*'+str(dims[n])+']);')
        elif accs[n] == OPS_MIN:
          code('arg'+str(n)+'h[d] = MIN(arg'+str(n)+'h[d],((double *)arg'+str(n)+'.data)[d+b*'+str(dims[n])+']);')
        ENDFOR()
        ENDFOR()
        code('arg'+str(n)+'.data = (char *)arg'+str(n)+'h;')
        code('')

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
