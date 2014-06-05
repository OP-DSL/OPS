
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
OPS OpenCL code generator

This routine is called by ops.py which parses the input files

It produces a file xxx_opencl_kernel.cpp for each kernel,
plus a master kernel file

"""

import re
import datetime
import os

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

  file_text += prefix+text+'\n'

def FOR(i,start,finish):
  global file_text
  global depth
  code('for ( int '+i+'='+start+'; '+i+'<'+finish+'; '+i+'++ ){')
  depth += 2

def FOR2(i,start,finish,increment):
  global file_text
  global depth
  code('for ( int '+i+'='+start+'; '+i+'<'+finish+'; '+i+'+='+increment+' ){')
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

def ELSE():
  global file_text
  global depth
  code('else {')
  depth += 2

def ENDIF():
  global file_text
  global depth
  depth -= 2
  code('}')

def mult(text, i, n):
  text = text + '1'
  for nn in range (0, i):
    text = text + '* args['+str(n)+'].dat->block_size['+str(nn)+']'

  return text

def ops_gen_mpi_opencl(master, date, consts, kernels):

  global dims, stens
  global g_m, file_text, depth

  OPS_ID   = 1;  OPS_GBL   = 2;  OPS_MAP = 3;

  OPS_READ = 1;  OPS_WRITE = 2;  OPS_RW  = 3;
  OPS_INC  = 4;  OPS_MAX   = 5;  OPS_MIN = 6;

  accsstring = ['OPS_READ','OPS_WRITE','OPS_RW','OPS_INC','OPS_MAX','OPS_MIN' ]

  NDIM = 2 #the dimension of the application is hardcoded here .. need to get this dynamically

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

    for n in range (0, nargs):
      if str(stens[n]).find('STRID2D_X') > 0:
        stride[2*n+1] = 0
      elif str(stens[n]).find('STRID2D_Y') > 0:
        stride[2*n] = 0


    reduction = 0
    for n in range (0, nargs):
      if arg_typ[n] == 'ops_arg_gbl' and accs[n] <> OPS_READ:
        reduction = 1



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

    comm('user function')
    code('#include "'+name2+'_kernel.h"')
    comm('')
    comm(' host stub function')
    code('void ops_par_loop_'+name+'(char const *name, ops_block block, int dim, int* range,')
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
    code('char *p_a['+str(nargs)+'];')
    code('int  offs['+str(nargs)+'][2];')

    #code('ops_printf("In loop \%s\\n","'+name+'");')

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

    code('sub_block_list sb = OPS_sub_block_list[block->index];')

    comm('compute localy allocated range for the sub-block')
    code('int* start = (int *)xmalloc(sizeof(int)*'+str(NDIM)+');')
    code('int* end = (int *)xmalloc(sizeof(int)*'+str(NDIM)+');')

    FOR('n','0',str(NDIM))
    code('start[n] = sb->istart[n];end[n] = sb->iend[n]+1;')
    IF('start[n] >= range[2*n]')
    code('start[n] = 0;')
    ENDIF()
    ELSE()
    code('start[n] = range[2*n] - start[n];')
    ENDIF()

    IF('end[n] >= range[2*n+1]')
    code('end[n] = range[2*n+1] - sb->istart[n];')
    ENDIF()
    ELSE()
    code('end[n] = sb->sizes[n];')
    ENDIF()
    ENDFOR()
    code('')

    code('#ifdef OPS_DEBUG')
    code('ops_register_args(args, "'+name+'");')
    code('#endif')
    code('')

    for n in range (0, nargs):
      if arg_typ[n] == 'ops_arg_dat':
        code('offs['+str(n)+'][0] = args['+str(n)+'].stencil->stride[0]*1;  //unit step in x dimension')
        #FOR('n','1',str(NDIM))
        #code('offs['+str(n)+'][n] = off2('+str(NDIM)+', n, &start[0],')
        #code('&end[0],args['+str(n)+'].dat->block_size, args['+str(n)+'].stencil->stride);')
        #ENDFOR()
        for d in range (1, NDIM):
          code('offs['+str(n)+']['+str(d)+'] = off2D('+str(d)+', &start[0],')
          code('&end[0],args['+str(n)+'].dat->block_size, args['+str(n)+'].stencil->stride);')
          code('')

    code('')
    code('')

    comm('Timing')
    code('double t1,t2,c1,c2;')
    code('ops_timing_realloc('+str(nk)+',"'+name+'");')
    code('ops_timers_core(&c2,&t2);')
    code('')

    for n in range (0, nargs):
      if arg_typ[n] == 'ops_arg_dat':
        code('int off'+str(n)+'_1 = offs['+str(n)+'][0];')
        code('int off'+str(n)+'_2 = offs['+str(n)+'][1];')
        code('int dat'+str(n)+' = args['+str(n)+'].dat->size;')

    code('')
    for n in range (0, nargs):
      if arg_typ[n] == 'ops_arg_dat':

        #compute max halo depths using stencil
        #code('int max'+str(n)+'['+str(NDIM)+']; int min'+str(n)+'['+str(NDIM)+'];')
        #FOR('n','0',str(NDIM))
        #code('max'+str(n)+'[n] = 0;min'+str(n)+'[n] = 0;')
        #ENDFOR()
        #FOR('p','0','args['+str(n)+'].stencil->points')
        #FOR('n','0',str(NDIM))
        #code('max'+str(n)+'[n] = MAX(max'+str(n)+'[n],args['+str(n)+'].stencil->stencil['+str(NDIM)+'*p + n]);')# * ((range[2*n+1]-range[2*n]) == 1 ? 0 : 1);');
        #code('min'+str(n)+'[n] = MIN(min'+str(n)+'[n],args['+str(n)+'].stencil->stencil['+str(NDIM)+'*p + n]);')# * ((range[2*n+1]-range[2*n]) == 1 ? 0 : 1);');
        #ENDFOR()
        #ENDFOR()

        comm('set up initial pointers and exchange halos if nessasary')

        code('int base'+str(n)+' = dat'+str(n)+' * 1 * ')
        code('(start[0] * args['+str(n)+'].stencil->stride[0] - args['+str(n)+'].dat->offset[0]);')
        for d in range (1, NDIM):
          code('base'+str(n)+' = base'+str(n)+'  + dat'+str(n)+' * args['+str(n)+'].dat->block_size['+str(d-1)+'] * ')
          code('(start['+str(d)+'] * args['+str(n)+'].stencil->stride['+str(d)+'] - args['+str(n)+'].dat->offset['+str(d)+']);')

        code('p_a['+str(n)+'] = (char *)args['+str(n)+'].data + base'+str(n)+';')

        #original address calculation via funcion call
        #code('p_a['+str(n)+'] = (char *)args['+str(n)+'].data')
        #code('+ address2('+str(NDIM)+', args['+str(n)+'].dat->size, &start['+str(n)+'*'+str(NDIM)+'],')
        #code('args['+str(n)+'].dat->block_size, args['+str(n)+'].stencil->stride, args['+str(n)+'].dat->offset);')

      else:
        code('p_a['+str(n)+'] = (char *)args['+str(n)+'].data;')
        code('')

      #if arg_typ[n] == 'ops_arg_dat' and (accs[n] == OPS_READ or accs[n] == OPS_RW ):# or accs[n] == OPS_INC):
        #code('ops_exchange_halo2(&args['+str(n)+'],max'+str(n)+',min'+str(n)+');')
        #code('ops_exchange_halo3(&args['+str(n)+'],max'+str(n)+',min'+str(n)+',range);')
        #code('ops_exchange_halo(&args['+str(n)+'],2);')
      code('')
    code('')
    
    code('ops_halo_exchanges(args,'+str(nargs)+',range);')
    code('')
    code('ops_H_D_exchanges(args, '+str(nargs)+');')
    code('')
    code('ops_timers_core(&c1,&t1);')
    code('OPS_kernels['+str(nk)+'].mpi_time += t1-t2;')
    code('')

    #code('ops_halo_exchanges(args, '+str(nargs)+');\n')


    for n in range (0, nargs):
      if arg_typ[n] == 'ops_arg_dat':
        code('xdim'+str(n)+' = args['+str(n)+'].dat->block_size[0]*args['+str(n)+'].dat->dim;')
    code('')

    code('int n_x;')

    FOR('n_y','start[1]','end[1]')
    #FOR('n_x','start[0]','start[0]+(end[0]-start[0])/SIMD_VEC')
    #FOR('n_x','start[0]','start[0]+(end[0]-start[0])/SIMD_VEC')
    #code('for( n_x=0; n_x<ROUND_DOWN((end[0]-start[0]),SIMD_VEC); n_x+=SIMD_VEC ) {')
    code('for( n_x=start[0]; n_x<start[0]+((end[0]-start[0])/SIMD_VEC)*SIMD_VEC; n_x+=SIMD_VEC ) {')
    depth = depth+2

    comm('call kernel function, passing in pointers to data -vectorised')
    if reduction == 0:
      code('#pragma simd')
    FOR('i','0','SIMD_VEC')
    text = name+'( '
    for n in range (0, nargs):
      if arg_typ[n] == 'ops_arg_dat':
        text = text +' ('+(str(typs[n]).replace('"','')).strip()+' *)p_a['+str(n)+']+ i*'+str(stride[2*n])
      else:
        text = text +' ('+(str(typs[n]).replace('"','')).strip()+' *)p_a['+str(n)+']'
      if nargs <> 1 and n != nargs-1:
        text = text + ','
      else:
        text = text +' );\n'
      if n%n_per_line == 2 and n <> nargs-1:
        text = text +'\n          '
    code(text);
    ENDFOR()
    code('')


    comm('shift pointers to data x direction')
    for n in range (0, nargs):
      if arg_typ[n] == 'ops_arg_dat':
          code('p_a['+str(n)+']= p_a['+str(n)+'] + (dat'+str(n)+' * off'+str(n)+'_1)*SIMD_VEC;')

    ENDFOR()
    code('')


    FOR('n_x','start[0]+((end[0]-start[0])/SIMD_VEC)*SIMD_VEC','end[0]')
    #code('for(;n_x<(end[0]-start[0]);n_x++) {')
    depth = depth+2
    comm('call kernel function, passing in pointers to data - remainder')
    text = name+'( '
    for n in range (0, nargs):
      if arg_typ[n] == 'ops_arg_dat':
        text = text +' ('+(str(typs[n]).replace('"','')).strip()+' *)p_a['+str(n)+']'
      else:
        text = text +' ('+(str(typs[n]).replace('"','')).strip()+' *)p_a['+str(n)+']'
      if nargs <> 1 and n != nargs-1:
        text = text + ','
      else:
        text = text +' );\n'
      if n%n_per_line == 2 and n <> nargs-1:
        text = text +'\n          '
    code(text);

    code('')


    comm('shift pointers to data x direction')
    for n in range (0, nargs):
      if arg_typ[n] == 'ops_arg_dat':
          code('p_a['+str(n)+']= p_a['+str(n)+'] + (dat'+str(n)+' * off'+str(n)+'_1);')

    ENDFOR()
    code('')


    comm('shift pointers to data y direction')
    for n in range (0, nargs):
      if arg_typ[n] == 'ops_arg_dat':
          #code('p_a['+str(n)+']= p_a['+str(n)+'] + (dat'+str(n)+' * (off'+str(n)+'_2) - '+str(stride[2*n])+');')
          code('p_a['+str(n)+']= p_a['+str(n)+'] + (dat'+str(n)+' * off'+str(n)+'_2);')
    ENDFOR()

    
    code('ops_set_dirtybit_host(args, '+str(nargs)+');')
    
    code('ops_timers_core(&c2,&t2);')
    code('OPS_kernels['+str(nk)+'].time += t2-t1;')

    if reduction == 1 :

      for n in range (0, nargs):
        if arg_typ[n] == 'ops_arg_gbl' and accs[n] != OPS_READ:
          code('ops_mpi_reduce(&arg'+str(n)+',('+(str(typs[n]).replace('"','')).strip()+' *)p_a['+str(n)+']);')

      code('ops_timers_core(&c1,&t1);')
      code('OPS_kernels['+str(nk)+'].mpi_time += t1-t2;')

    for n in range (0, nargs):
      if arg_typ[n] == 'ops_arg_dat' and (accs[n] == OPS_WRITE or accs[n] == OPS_RW or accs[n] == OPS_INC):
        #code('ops_set_halo_dirtybit(&args['+str(n)+']);')
        code('ops_set_halo_dirtybit3(&args['+str(n)+'],range);')

    code('free(start);free(end);')

    code('')
    comm('Update kernel record')
    code('OPS_kernels['+str(nk)+'].count++;')
    for n in range (0, nargs):
      if arg_typ[n] == 'ops_arg_dat':
        code('OPS_kernels['+str(nk)+'].transfer += ops_compute_transfer(dim, range, &arg'+str(n)+');')
    depth = depth - 2
    code('}')

##########################################################################
#  output individual kernel file
##########################################################################
    if not os.path.exists('./OpenCL_test'):
      os.makedirs('./OpenCL_test')
    fid = open('./OpenCL_test/'+name+'_opencl_kernel.cpp','w')
    date = datetime.datetime.now()
    fid.write('//\n// auto-generated by ops.py on '+date.strftime("%Y-%m-%d %H:%M")+'\n//\n\n')
    fid.write(file_text)
    fid.close()

# end of main kernel call loop

##########################################################################
#  output one master kernel file
##########################################################################
  depth = 0
  file_text =''
  comm('header')
  code('#include "ops_lib_cpp.h"')
  code('#include "stdlib.h"')
  code('#include "stdio.h"')
  code('#include "ops_lib_cpp.h"')
  code('#include "ops_opencl_rt_support.h"')
  
  kernel_name_list = []
  kernel_list_text = ''
  kernel_list__build_text = ''
  indent = 10*' '
  for nk in range(0,len(kernels)):
    if kernels[nk]['name'] not in kernel_name_list :
      kernel_name_list.append(kernels[nk]['name'])
      kernel_list_text = kernel_list_text + '"./OpenCL/'+kernel_name_list[nk]+'.cl"'
      if nk != len(kernels)-1:
        kernel_list_text = kernel_list_text+',\n'+indent
      kernel_list__build_text = kernel_list__build_text + \
      'OPS_opencl_core.kernel['+str(nk)+'] = clCreateKernel(OPS_opencl_core.program, "'+kernel_name_list[nk]+'", &ret);\n      '+\
      'clSafeCall( ret );\n      '  
  

  opencl_build = """
  extern ops_opencl_core OPS_opencl_core;

  #define MAX_SOURCE_SIZE ("""+str(len(kernels))+"""*0x1000000)
  
  void buildOpenCLKernels() {
    static bool isbuilt = false;
  
    if(!isbuilt) {
      clSafeCall( clUnloadCompiler() );
  
      OPS_opencl_core.n_kernels = """+str(len(kernels))+""";
      OPS_opencl_core.kernel = (cl_kernel*) malloc("""+str(len(kernels))+"""*sizeof(cl_kernel));  
      
      cl_int ret;
      char* source_filename["""+str(len(kernels))+"""] = {
          """+kernel_list_text+"""
      };
  
      // Load the kernel source code into the array source_str
      FILE *fid;
      char *source_str["""+str(len(kernels))+"""];
      size_t source_size["""+str(len(kernels))+"""];
  
      for(int i=0; i<"""+str(len(kernels))+"""; i++) {
        fid = fopen(source_filename[i], "r");
        if (!fid) {
          fprintf(stderr, "Can't open the kernel source file!\\n");
          exit(1);
        }
        
        source_str[i] = (char*)malloc(MAX_SOURCE_SIZE);
        source_size[i] = fread(source_str[i], 1, MAX_SOURCE_SIZE, fid);
        if(source_size[i] != MAX_SOURCE_SIZE) {
          if (ferror(fid)) {
            printf ("Error while reading kernel source file %s\\n", source_filename[i]);
            exit(-1);
          }
          if (feof(fid))
            printf ("Kernel source file %s succesfuly read.\\n", source_filename[i]);
            //printf("%s\\n",source_str[i]);
        }
        fclose(fid);
      }
      
      printf(" compiling sources \\n");
  
        // Create a program from the source
        OPS_opencl_core.program = clCreateProgramWithSource(OPS_opencl_core.context, """+str(len(kernels))+""", (const char **) &source_str, (const size_t *) &source_size, &ret);
        clSafeCall( ret );
  
        // Build the program
        char buildOpts[255];
        sprintf(buildOpts,"-cl-mad-enable -DOPS_WARPSIZE=%d", 32);
        ret = clBuildProgram(OPS_opencl_core.program, 1, &OPS_opencl_core.device_id, buildOpts, NULL, NULL);
  
        if(ret != CL_SUCCESS) {
          char* build_log;
          size_t log_size;
          clSafeCall( clGetProgramBuildInfo(OPS_opencl_core.program, OPS_opencl_core.device_id, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size) );
          build_log = (char*) malloc(log_size+1);
          clSafeCall( clGetProgramBuildInfo(OPS_opencl_core.program, OPS_opencl_core.device_id, CL_PROGRAM_BUILD_LOG, log_size, build_log, NULL) );
          build_log[log_size] = '\0';
          fprintf(stderr, "=============== OpenCL Program Build Info ================\\n\\n%s", build_log);
          fprintf(stderr, "\\n========================================================= \\n");
          free(build_log);
          exit(EXIT_FAILURE);
        }
        printf(" compiling done\\n");
  
      // Create the OpenCL kernel
      """+kernel_list__build_text+"""      
      isbuilt = true;
    }
    
  }
  
//this needs to be a platform specific copy symbol to device function
void ops_decl_const_char( int dim, char const * type, int typeSize, char * data, char const * name )
{
  (void)dim;
  (void)type;
  (void)typeSize;
  (void)data;
  (void)name;
}

  """
  code(opencl_build)
  
  
  comm('user kernel files')

  for nk in range(0,len(kernel_name_list)):
    code('#include "'+kernel_name_list[nk]+'_opencl_kernel.cpp"')
  
   
  master = master.split('.')[0]
  fid = open('./OpenCL_test/'+master.split('.')[0]+'_opencl_kernels.cpp','w')
  fid.write('//\n// auto-generated by op2.py on '+date.strftime("%Y-%m-%d %H:%M")+'\n//\n\n')
  fid.write(file_text)
  fid.close()
