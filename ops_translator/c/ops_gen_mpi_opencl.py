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
#  OPS OpenCL code generator
#
#  This routine is called by ops.py which parses the input files
#
#  It produces a file xxx_opencl_kernel.cpp and a XX_kernel.cl for each kernel,
#  plus a master kernel file
#

"""
OPS OpenCL code generator

This routine is called by ops.py which parses the input files

It produces a file xxx_opencl_kernel.cpp and a XX_kernel.cl for each kernel,
plus a master kernel file

"""

import re
import errno
import os

import config
from config import OPS_READ, OPS_WRITE, OPS_RW, OPS_INC, OPS_MAX, OPS_MIN

import util
from util import (
    para_parse,
    parse_signature,
    find_consts,
    replace_ACC_kernel_body,
    parse_replace_ACC_signature,
)
from util import comm, code, FOR, ENDFOR, IF, ELSE, ENDIF


def ops_gen_mpi_opencl(master, consts, kernels, soa_set):
  NDIM = 2 #the dimension of the application, set to 2 by default. Will be updated later from loops

  src_dir = os.path.dirname(master) or '.'
  master_basename = os.path.splitext(os.path.basename(master))

  ##########################################################################
  #  create new kernel files **_kernel.cl
  ##########################################################################
  try:
    os.makedirs('./OpenCL')
  except OSError as e:
    if e.errno != errno.EEXIST:
      raise
  for nk in range (0,len(kernels)):
    assert config.file_text == '' and config.depth == 0
    arg_typ  = kernels[nk]['arg_type']
    name  = kernels[nk]['name']
    nargs = kernels[nk]['nargs']
    dim   = kernels[nk]['dim']
    dims  = kernels[nk]['dims']
    stens = kernels[nk]['stens']
    accs  = kernels[nk]['accs']
    typs  = kernels[nk]['typs']

    if ('initialise' in name) or ('generate' in name):
      print(f'WARNING: skipping kernel {name} due to OpenCL compiler bugs: this kernel will run sequentially on the host')
      continue

    #reset dimension of the application
    NDIM = int(dim)

    #parse stencil to locate strided access
    stride = [1] * nargs * NDIM

    if NDIM == 2:
      for n in range (0, nargs):
        if str(stens[n]).find('STRID2D_X') > 0:
          stride[NDIM*n+1] = 0
        elif str(stens[n]).find('STRID2D_Y') > 0:
          stride[NDIM*n] = 0

    if NDIM == 3:
      for n in range (0, nargs):
        if str(stens[n]).find('STRID3D_XY') > 0:
          stride[NDIM*n+2] = 0
        elif str(stens[n]).find('STRID3D_YZ') > 0:
          stride[NDIM*n] = 0
        elif str(stens[n]).find('STRID3D_XZ') > 0:
          stride[NDIM*n+1] = 0
        elif str(stens[n]).find('STRID3D_X') > 0:
          stride[NDIM*n+1] = 0
          stride[NDIM*n+2] = 0
        elif str(stens[n]).find('STRID3D_Y') > 0:
          stride[NDIM*n] = 0
          stride[NDIM*n+2] = 0
        elif str(stens[n]).find('STRID3D_Z') > 0:
          stride[NDIM*n] = 0
          stride[NDIM*n+1] = 0

    reduction = 0
    for n in range (0, nargs):
      if arg_typ[n] == 'ops_arg_gbl' and accs[n] != OPS_READ:
        reduction = 1


    arg_idx = 0
    for n in range (0, nargs):
      if arg_typ[n] == 'ops_arg_idx':
        arg_idx = 1


    ##########################################################################
    #  start with opencl kernel function
    ##########################################################################

    n_per_line = 4

    i = name.find('kernel')
    name2 = name[0:i-1]

    code('')

    code('#ifdef OCL_FMA')
    code('#pragma OPENCL FP_CONTRACT ON')
    code('#else')
    code('#pragma OPENCL FP_CONTRACT OFF')
    code('#endif')
    code('#pragma OPENCL EXTENSION cl_khr_fp64:enable')
    code('')
    if os.path.exists(os.path.join(src_dir,'user_types.h')):
      code('#include "user_types.h"')
    code(f'#define OPS_{NDIM}D')
    code('#define OPS_API 2')
    code('#define OPS_NO_GLOBALS')
    code('#include "ops_macros.h"')
    code('#include "ops_opencl_reduction.h"')
    #generate MACROS
    comm('')
    code('#ifndef MIN')
    code('#define MIN(a,b) ((a<b) ? (a) : (b))')
    code('#endif')
    code('#ifndef MAX')
    code('#define MAX(a,b) ((a>b) ? (a) : (b))')
    code('#endif')
    code('#ifndef SIGN')
    code('#define SIGN(a,b) ((b<0.0) ? (a*(-1)) : (a))')
    code('#endif')

    code('#define OPS_READ 0')
    code('#define OPS_WRITE 1')
    code('#define OPS_RW 2')
    code('#define OPS_INC 3')
    code('#define OPS_MIN 4')
    code('#define OPS_MAX 5')

    code('')
    comm('user function')
    text = util.get_file_text_for_kernel(name, src_dir)

    p = re.compile(f'void\\s+\\b{name}\\b')

    i = p.search(text).start()
    if(i < 0):
      print("\n********")
      print(f"Error: cannot locate user kernel function: {name} - Aborting code generation")
      exit(2)


    i = text[0:i].rfind('\n') #reverse find

    text = text[i:]
    j = text.find('{')
    k = para_parse(text, j, '{', '}')
    text = text[0:k+1]
    #convert to new API if in old
    text = util.convert_ACC(text,arg_typ)
    j = text.find('{')
    k = para_parse(text, j, '{', '}')
    m = text.find(name)
    arg_list = parse_signature(text[m+len(name):j])


    part_name = text[0:m+len(name)]
    part_args = text[m+len(name):j]
    part_body = text[j:]
    found_consts = find_consts(part_body,consts)
    if len(found_consts) != 0:
      text = part_args[0:part_args.rfind(')')]
      for c in range(0, len(found_consts)):
        if (consts[found_consts[c]]['dim']).isdigit() and int(consts[found_consts[c]]['dim'])==1:
          text += f", const {consts[found_consts[c]]['type']} {consts[found_consts[c]]['name'][1:-1]}"
        else:
          text += f", __constant const{consts[found_consts[c]]['type']} *{consts[found_consts[c]]['name'][1:-1]}"
        if c == len(found_consts)-1:
          text = text + ')\n'
      part_args = text


    text = part_name + \
            parse_replace_ACC_signature(part_args, arg_typ, dims, 1, accs, typs) + \
            replace_ACC_kernel_body(part_body, arg_list, arg_typ, nargs, 1, dims)


    code(text)
    code('')
    code('')



    ##########################################################################
    #  generate opencl kernel wrapper function
    ##########################################################################
    code(f'__kernel void ops_{name}(')
    #currently the read only vars have not been generated differently
    for n in range (0, nargs):
      if arg_typ[n] == 'ops_arg_dat' and accs[n] == OPS_READ:
        code('__global const '+(str(typs[n]).replace('"','')).strip()+f'* restrict arg{n},')
      elif arg_typ[n] == 'ops_arg_dat'and (accs[n] == OPS_WRITE or accs[n] == OPS_RW or accs[n] == OPS_INC) :
        code('__global '+(str(typs[n]).replace('"','')).strip()+f'* restrict arg{n},')
      elif arg_typ[n] == 'ops_arg_gbl':
        if accs[n] == OPS_READ:
          if dims[n].isdigit() and int(dims[n]) == 1:
            code('const '+(str(typs[n]).replace('"','')).strip()+f' arg{n},')
          else:
            code('__global const '+(str(typs[n]).replace('"','')).strip()+f'* restrict arg{n},')
        else:
          code('__global '+(str(typs[n]).replace('"','')).strip()+f'* restrict arg{n},')
          code('__local '+(str(typs[n]).replace('"','')).strip()+f'* scratch{n},')
          code(f'int r_bytes{n},')

    for c in range(0, len(found_consts)):
      if consts[found_consts[c]]['type']=='int' or consts[found_consts[c]]['type']=='double' or consts[found_consts[c]]['type']=='float':
        code(f"const {consts[found_consts[c]]['type']} {consts[found_consts[c]]['name'][1:-1]},")
      else:
        code(f"__constant const struct {consts[found_consts[c]]['type']} * restrict {consts[found_consts[c]]['name'][1:-1]},")


    for n in range (0, nargs):
      if arg_typ[n] == 'ops_arg_dat':
        code(f'const int base{n},')


    if arg_idx:
      if NDIM==1:
          code('int arg_idx0,')
      elif NDIM==2:
        code('int arg_idx0, int arg_idx1,')
      elif NDIM==3:
        code('int arg_idx0, int arg_idx1, int arg_idx2,')
    if NDIM==1:
      code('const int size0 ){')
    elif NDIM==2:
      code('const int size0,')
      code('const int size1 ){')
    elif NDIM==3:
      code('const int size0,')
      code('const int size1,')
      code('const int size2 ){')


    config.depth = config.depth + 2

    #local variable to hold reductions on GPU
    code('')
    for n in range (0, nargs):
      if arg_typ[n] == 'ops_arg_gbl' and accs[n] != OPS_READ:
        code(f'arg{n} += r_bytes{n};')
        code((str(typs[n]).replace('"','')).strip()+f' arg{n}_l[{dims[n]}];')

    # set local variables to 0 if OPS_INC, INF if OPS_MIN, -INF
    for n in range (0, nargs):
      if arg_typ[n] == 'ops_arg_gbl' and accs[n] == OPS_INC:
        code(f'for (int d=0; d<{dims[n]}; d++) arg{n}_l[d] = ZERO_'+(str(typs[n]).replace('"','')).strip()+';')
      if arg_typ[n] == 'ops_arg_gbl' and accs[n] == OPS_MIN:
        code(f'for (int d=0; d<{dims[n]}; d++) arg{n}_l[d] = INFINITY_'+(str(typs[n]).replace('"','')).strip()+';')
      if arg_typ[n] == 'ops_arg_gbl' and accs[n] == OPS_MAX:
        code(f'for (int d=0; d<{dims[n]}; d++) arg{n}_l[d] = -INFINITY_'+(str(typs[n]).replace('"','')).strip()+';')


    code('')
    if NDIM==3:
      code('int idx_y = get_global_id(1);')
      code('int idx_z = get_global_id(2);')
    elif NDIM==2:
        code('int idx_y = get_global_id(1);')
    code('int idx_x = get_global_id(0);')
    code('')
    if arg_idx:
      code(f'int arg_idx[{NDIM}];')
      code('arg_idx[0] = arg_idx0+idx_x;')
      if NDIM==2:
        code('arg_idx[1] = arg_idx1+idx_y;')
      elif NDIM==3:
        code('arg_idx[1] = arg_idx1+idx_y;')
        code('arg_idx[2] = arg_idx2+idx_z;')


    indent = (len(name2)+config.depth+8)*' '
    n_per_line = 5
    if NDIM==1:
      IF('idx_x < size0')
    elif NDIM==2:
      IF('idx_x < size0 && idx_y < size1')
    elif NDIM==3:
      IF('idx_x < size0 && idx_y < size1 && idx_z < size2')
    for n in range (0, nargs):
      text = ''
      if arg_typ[n] == 'ops_arg_dat':
        if NDIM==1:
          if soa_set:
            text += f'&arg{n}[base{n} + idx_x * {stride[NDIM*n]}]'
          else:
            text += f'&arg{n}[base{n} + idx_x * {stride[NDIM*n]}*{dims[n]}]'
        elif NDIM==2:
          if soa_set:
            text += f'&arg{n}[base{n} + idx_x * {stride[NDIM*n]} + idx_y * {stride[NDIM*n+1]} * xdim{n}_{name}]'
          else:
            text += f'&arg{n}[base{n} + idx_x * {stride[NDIM*n]}*{dims[n]} + idx_y * {stride[NDIM*n+1]}*{dims[n]} * xdim{n}_{name}]'
        elif NDIM==3:
          if soa_set:
            text += f'&arg{n}[base{n} + idx_x * {stride[NDIM*n]} + idx_y * {stride[NDIM*n+1]} * xdim{n}_{name} + idx_z * {stride[NDIM*n+2]} * xdim{n}_{name} * ydim{n}_{name}]'
          else:
            text += f'&arg{n}[base{n} + idx_x * {stride[NDIM*n]}*{dims[n]} + idx_y * {stride[NDIM*n+1]}*{dims[n]} * xdim{n}_{name} + idx_z * {stride[NDIM*n+2]}*{dims[n]} * xdim{n}_{name} * ydim{n}_{name}]'
        pre = ''
        if accs[n] == OPS_READ:
          pre = 'const '
        dim = ''
        sizelist = ''
        extradim = 0
        if dims[n].isdigit() and int(dims[n])>1:
            dim = dims[n]
            extradim = 1
        elif not dims[n].isdigit():
            dim = f'arg{n}.dim'
            extradim = 1
        dimlabels = 'xyzuv'
        for i in range(1,NDIM):
          sizelist +=  f'{dimlabels[i-1]}dim{n}_{name}, '
        extradim = f'{dimlabels[NDIM+extradim-2]}dim{n}_{name}'
        if dim == '':
          if NDIM==1:
            code(f'{pre}ptr_{typs[n]} ptr{n} = {{ {text} }};')
          else:
            code(f'{pre}ptr_{typs[n]} ptr{n} = {{ {text}, {sizelist[:-2]}}};')
        else:
          code('#ifdef OPS_SOA')
          code(f'{pre}ptrm_{typs[n]} ptr{n} = {{ {text}, {sizelist + extradim}}};')
          code('#else')
          code(f'{pre}ptrm_{typs[n]} ptr{n} = {{ {text}, {sizelist+dim}}};')
          code('#endif')


    text = name+'('
    for n in range (0, nargs):
      if arg_typ[n] == 'ops_arg_dat':
          text += f'ptr{n}'
      elif arg_typ[n] == 'ops_arg_gbl' and accs[n] == OPS_READ:
        if dims[n].isdigit() and int(dims[n]) == 1:
          text += f'&arg{n}'
        else:
          text += f'arg{n}'
      elif arg_typ[n] == 'ops_arg_gbl' and accs[n] != OPS_READ:
        text += f'arg{n}_l'
      elif arg_typ[n] == 'ops_arg_idx':
        text += 'arg_idx'

      if n != nargs-1 :
        text += ',\n  '+indent
      elif len(found_consts) > 0:
        text += ',\n  '+indent



    for c in range(0, len(found_consts)):
      if consts[found_consts[c]]['type'] == 'int' or consts[found_consts[c]]['type'] == 'double' or consts[found_consts[c]]['type'] == 'float':
        text += consts[found_consts[c]]['name'][1:-1]
      else:
        text += '*'+consts[found_consts[c]]['name'][1:-1]
      if c != len(found_consts)-1:
        text += ',\n  '+indent

    code(text+');')

    ENDIF()

    #reduction across blocks
    if reduction:
      code('int group_index = get_group_id(0) + get_group_id(1)*get_num_groups(0)+ get_group_id(2)*get_num_groups(0)*get_num_groups(1);')
      for n in range (0, nargs):
        if arg_typ[n] == 'ops_arg_gbl' and accs[n] == OPS_INC:
          code(f'for (int d=0; d<{dims[n]}; d++)')
          code('  reduce_'+(str(typs[n]).replace('"','')).strip()+f'(arg{n}_l[d], scratch{n}, &arg{n}[group_index*{dims[n]}+d], OPS_INC);')
        if arg_typ[n] == 'ops_arg_gbl' and accs[n] == OPS_MIN:
          code(f'for (int d=0; d<{dims[n]}; d++)')
          code('  reduce_'+(str(typs[n]).replace('"','')).strip()+f'(arg{n}_l[d], scratch{n}, &arg{n}[group_index*{dims[n]}+d], OPS_MIN);')
        if arg_typ[n] == 'ops_arg_gbl' and accs[n] == OPS_MAX:
          code(f'for (int d=0; d<{dims[n]}; d++)')
          code('  reduce_'+(str(typs[n]).replace('"','')).strip()+f'(arg{n}_l[d], scratch{n}, &arg{n}[group_index*{dims[n]}+d], OPS_MAX);')


    code('')
    config.depth = config.depth - 2
    code('}')


    try:
      os.makedirs('./OpenCL')
    except OSError as e:
      if e.errno != errno.EEXIST:
        raise
    util.write_text_to_file(f"./OpenCL/{name}.cl")

    ##########################################################################
    #  generate opencl kernel build function
    ##########################################################################

    kernel_list_text = f'"./OpenCL/{name}.cl"'
    arg_text = ''
    compile_line = ''
    arg_values = ''
    for n in range (0, nargs):
      if arg_typ[n] == 'ops_arg_dat':
        arg_text += f'int_xdim{n} '
        compile_line += f' -Dxdim{n}_{name}=%d '
        arg_values += f'xdim{n} '
        if NDIM>2 or (NDIM==2 and soa_set):
          arg_text += f'int_ydim{n} '
          compile_line += f' -Dydim{n}_{name}=%d '
          arg_values += f'ydim{n} '
        if NDIM>3 or (NDIM==3 and soa_set):
          arg_text += f'int_zdim{n} '
          compile_line += f' -Dzdim{n}_{name}=%d '
          arg_values += f'zdim{n} '

    ' '.join(arg_values.split())
    arg_values = arg_values.replace(' ',',')
    arg_values = arg_values[:-1]

    ' '.join(arg_text.split())
    arg_text = arg_text.replace(' ',', ')
    arg_text = arg_text[:-2]
    arg_text = arg_text.replace('_',' ')

    compile_line = compile_line + '"'


    opencl_build_kernel = """
    #ifdef OCL_FMA_SWITCH_ON
    #define OCL_FMA 1
    #else
    #define OCL_FMA 0
    #endif


    static bool isbuilt_"""+name+""" = false;

    void buildOpenCLKernels_"""+name+"""(OPS_instance *instance, """+arg_text+""") {

      //int ocl_fma = OCL_FMA;
      if(!isbuilt_"""+name+""") {
        buildOpenCLKernels(instance);
        //clSafeCall( clUnloadCompiler() );
        cl_int ret;
        char* source_filename[1] = {(char*)"""+kernel_list_text+"""};

        // Load the kernel source code into the array source_str
        FILE *fid;
        char *source_str[1] = {NULL};
        size_t source_size[1];

        for(int i=0; i<1; i++) {
          fid = fopen(source_filename[i], "r");
          if (!fid) {
            OPSException e(OPS_RUNTIME_ERROR, "Can't open the kernel source file: ");
            e << source_filename[i] << "\\n";
            throw e;
          }

          source_str[i] = (char*)malloc(4*0x1000000);
          source_size[i] = fread(source_str[i], 1, 4*0x1000000, fid);
          if(source_size[i] != 4*0x1000000) {
            if (ferror(fid)) {
              OPSException e(OPS_RUNTIME_ERROR, "Error while reading kernel source file ");
              e << source_filename[i] << "\\n";
              throw e;
            }
            if (feof(fid))
              instance->ostream() << "Kernel source file "<< source_filename[i] <<" succesfully read.\\n";
          }
          fclose(fid);
        }

        instance->ostream() <<"Compiling """+name+""" "<<OCL_FMA<<" source -- start \\n";

          // Create a program from the source
          instance->opencl_instance->OPS_opencl_core.program = clCreateProgramWithSource(instance->opencl_instance->OPS_opencl_core.context, 1, (const char **) &source_str, (const size_t *) &source_size, &ret);
          clSafeCall( ret );

          // Build the program
          char buildOpts[255*"""+str(nargs)+"""];
          char* pPath = NULL;
          pPath = getenv ("OPS_INSTALL_PATH");
          if (pPath!=NULL)
            if(OCL_FMA)
              sprintf(buildOpts,"-cl-mad-enable -DOCL_FMA -I%s/include -DOPS_WARPSIZE=%d """+compile_line+""", pPath, 32,"""+arg_values+""");
            else
              sprintf(buildOpts,"-cl-mad-enable -I%s/include -DOPS_WARPSIZE=%d """+compile_line+""", pPath, 32,"""+arg_values+""");
          else {
            sprintf((char*)"Incorrect OPS_INSTALL_PATH %s\\n",pPath);
            exit(EXIT_FAILURE);
          }

          #ifdef OPS_SOA
          sprintf(buildOpts, "%s -DOPS_SOA", buildOpts);
          #endif
          sprintf(buildOpts, "%s -I%s/c/include", buildOpts, pPath);
          ret = clBuildProgram(instance->opencl_instance->OPS_opencl_core.program, 1, &instance->opencl_instance->OPS_opencl_core.device_id, buildOpts, NULL, NULL);

          if(ret != CL_SUCCESS) {
            char* build_log;
            size_t log_size;
            clSafeCall( clGetProgramBuildInfo(instance->opencl_instance->OPS_opencl_core.program, instance->opencl_instance->OPS_opencl_core.device_id, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size) );
            build_log = (char*) malloc(log_size+1);
            clSafeCall( clGetProgramBuildInfo(instance->opencl_instance->OPS_opencl_core.program, instance->opencl_instance->OPS_opencl_core.device_id, CL_PROGRAM_BUILD_LOG, log_size, build_log, NULL) );
            build_log[log_size] = '\\0';
            instance->ostream() << "=============== OpenCL Program Build Info ================\\n\\n" << build_log;
            instance->ostream() << "\\n========================================================= \\n";
            free(build_log);
            exit(EXIT_FAILURE);
          }
          instance->ostream() << "compiling """+name+""" -- done\\n";

        // Create the OpenCL kernel
        instance->opencl_instance->OPS_opencl_core.kernel["""+str(nk)+"""] = clCreateKernel(instance->opencl_instance->OPS_opencl_core.program, "ops_"""+name+"""", &ret);
        clSafeCall( ret );\n
        isbuilt_"""+name+""" = true;
        free(source_str[0]);
      }

    }

    """



    ##########################################################################
    #  generate opencl host stub function
    ##########################################################################

    config.file_text = opencl_build_kernel
    config.depth = 0
    code('')
    comm(' host stub function')

    code(f'void ops_par_loop_{name}(char const *name, ops_block block, int dim, int* range,')
    text = ''
    for n in range (0, nargs):

      text += f' ops_arg arg{n}'
      if nargs != 1 and n != nargs-1:
        text += ','
      else:
        text += ') {'
      if n%n_per_line == 3 and n != nargs-1:
         text += '\n'
    code(text);
    config.depth = 2

    #timing structs
    code('')
    comm('Timing')
    code('double t1,t2,c1,c2;')
    code('')

    text =f'ops_arg args[{nargs}] = {{'
    for n in range (0, nargs):
      text += ' arg'+str(n)
      if nargs != 1 and n != nargs-1:
        text += ','
      else:
        text += '};\n'
      if n%n_per_line == 5 and n != nargs-1:
        text += '\n                    '
    code(text);
    code('')
    code('#ifdef CHECKPOINTING')
    code(f'if (!ops_checkpointing_before(args,{nargs},range,{nk})) return;')
    code('#endif')
    code('')

    IF('block->instance->OPS_diags > 1')
    code(f'ops_timing_realloc(block->instance,{nk},"{name}");')
    code(f'block->instance->OPS_kernels[{nk}].count++;')
    code('ops_timers_core(&c1,&t1);')
    ENDIF()
    code('')

    comm('compute locally allocated range for the sub-block')

    code(f'int start[{NDIM}];')
    code(f'int end[{NDIM}];')


    code('#ifdef OPS_MPI')
    code('sub_block_list sb = OPS_sub_block_list[block->index];')
    code('if (!sb->owned) return;')
    FOR('n','0',str(NDIM))
    code('start[n] = sb->decomp_disp[n];end[n] = sb->decomp_disp[n]+sb->decomp_size[n];')
    IF('start[n] >= range[2*n]')
    code('start[n] = 0;')
    ENDIF()
    ELSE()
    code('start[n] = range[2*n] - start[n];')
    ENDIF()
    code('if (sb->id_m[n]==MPI_PROC_NULL && range[2*n] < 0) start[n] = range[2*n];')
    IF('end[n] >= range[2*n+1]')
    code('end[n] = range[2*n+1] - sb->decomp_disp[n];')
    ENDIF()
    ELSE()
    code('end[n] = sb->decomp_size[n];')
    ENDIF()
    code('if (sb->id_p[n]==MPI_PROC_NULL && (range[2*n+1] > sb->decomp_disp[n]+sb->decomp_size[n]))')
    code('  end[n] += (range[2*n+1]-sb->decomp_disp[n]-sb->decomp_size[n]);')
    ENDFOR()
    code('#else')
    FOR('n','0',str(NDIM))
    code('start[n] = range[2*n];end[n] = range[2*n+1];')
    ENDFOR()
    code('#endif')

    code('')
    code('int x_size = MAX(0,end[0]-start[0]);')
    if NDIM==2:
      code('int y_size = MAX(0,end[1]-start[1]);')
    if NDIM==3:
      code('int y_size = MAX(0,end[1]-start[1]);')
      code('int z_size = MAX(0,end[2]-start[2]);')
    code('')

    if arg_idx:
      code(f'int arg_idx[{NDIM}];')
      code('#ifdef OPS_MPI')
      for n in range (0,NDIM):
        code(f'arg_idx[{n}] = sb->decomp_disp[{n}]+start[{n}];')
      code('#else')
      for n in range (0,NDIM):
        code(f'arg_idx[{n}] = start[{n}];')
      code('#endif')
    code('')


    for n in range (0, nargs):
      if arg_typ[n] == 'ops_arg_dat':
        code(f'int xdim{n} = args[{n}].dat->size[0];')
        if NDIM>2 or (NDIM==2 and soa_set):
          code(f'int ydim{n} = args[{n}].dat->size[1];')
        if NDIM>3 or (NDIM==3 and soa_set):
          code(f'int zdim{n} = args[{n}].dat->size[2];')
    code('')

    comm('build opencl kernel if not already built')
    code('');
    code(f'buildOpenCLKernels_{name}(block->instance,')
    arg_text = ''
    for n in range (0, nargs):
      if arg_typ[n] == 'ops_arg_dat':
        arg_text += f'xdim{n} '
        if NDIM>2 or (NDIM==2 and soa_set):
          arg_text += f'ydim{n} '
        if NDIM>3 or (NDIM==3 and soa_set):
          arg_text += f'zdim{n} '

    ' '.join(arg_text.split())
    arg_text = arg_text.replace(' ',',')
    arg_text = arg_text[:-1]


    code(arg_text+');')
    code('')

    #set up OpenCL grid and thread blocks
    comm('set up OpenCL thread blocks')
    if NDIM==1:
      code('size_t globalWorkSize[3] = {((x_size-1)/block->instance->OPS_block_size_x+ 1)*block->instance->OPS_block_size_x, 1, 1};')
    if NDIM==2:
      code('size_t globalWorkSize[3] = {((x_size-1)/block->instance->OPS_block_size_x+ 1)*block->instance->OPS_block_size_x, ((y_size-1)/block->instance->OPS_block_size_y + 1)*block->instance->OPS_block_size_y, 1};')
    if NDIM==3:
      code('size_t globalWorkSize[3] = {((x_size-1)/block->instance->OPS_block_size_x+ 1)*block->instance->OPS_block_size_x, ((y_size-1)/block->instance->OPS_block_size_y + 1)*block->instance->OPS_block_size_y, ((z_size-1)/block->instance->OPS_block_size_z+ 1)*block->instance->OPS_block_size_z};')

    if NDIM>1:
      code('size_t localWorkSize[3] =  {block->instance->OPS_block_size_x,block->instance->OPS_block_size_y,block->instance->OPS_block_size_z};')
    else:
      code('size_t localWorkSize[3] =  {block->instance->OPS_block_size_x,1,1};')
    code('')

    #setup reduction variables
    code('')
    for n in range (0, nargs):
        if arg_typ[n] == 'ops_arg_gbl' and (accs[n] != OPS_READ or (accs[n] == OPS_READ and (not dims[n].isdigit() or int(dims[n])>1))):
          if (accs[n] == OPS_READ):
            code(f'{typs[n]} *arg{n}h = ({typs[n]} *)arg{n}.data;')
          else:
            code('#ifdef OPS_MPI')
            code(f'{typs[n]} *arg{n}h = ({typs[n]} *)(((ops_reduction)args[{n}].data)->data + ((ops_reduction)args[{n}].data)->size * block->index);')
            code('#else')
            code(f'{typs[n]} *arg{n}h = ({typs[n]} *)(((ops_reduction)args[{n}].data)->data);')
            code('#endif')

    code('')


    GBL_READ = False
    GBL_READ_MDIM = False
    GBL_INC = False
    GBL_MAX = False
    GBL_MIN = False
    GBL_WRITE = False

    for n in range (0, nargs):
      if arg_typ[n] == 'ops_arg_gbl':
        if accs[n] == OPS_READ:
          GBL_READ = True
          if not dims[n].isdigit() or int(dims[n])>1:
            GBL_READ_MDIM = True
        if accs[n] == OPS_INC:
          GBL_INC = True
        if accs[n] == OPS_MAX:
          GBL_MAX = True
        if accs[n] == OPS_MIN:
          GBL_MIN = True
        if accs[n] == OPS_WRITE:
          GBL_WRITE = True

    if GBL_INC == True or GBL_MIN == True or GBL_MAX == True or GBL_WRITE == True:
      if NDIM==1:
        code('int nblocks = ((x_size-1)/block->instance->OPS_block_size_x+ 1);')
      elif NDIM==2:
        code('int nblocks = ((x_size-1)/block->instance->OPS_block_size_x+ 1)*((y_size-1)/block->instance->OPS_block_size_y + 1);')
      elif NDIM==3:
        code('int nblocks = ((x_size-1)/block->instance->OPS_block_size_x+ 1)*((y_size-1)/block->instance->OPS_block_size_y + 1)*((z_size-1)/block->instance->OPS_block_size_z + 1);')
      code('int maxblocks = nblocks;')
      code('int reduct_bytes = 0;')
      code('')

    if GBL_READ == True and GBL_READ_MDIM == True:
      code('int consts_bytes = 0;')
      code('')

    for n in range (0, nargs):
      if arg_typ[n] == 'ops_arg_gbl':
        if accs[n] == OPS_READ and (not dims[n].isdigit() or int(dims[n])>1):
            code(f'consts_bytes += ROUND_UP({dims[n]}*sizeof('+(str(typs[n]).replace('"','')).strip()+'));')
        elif accs[n] != OPS_READ:
          code(f'reduct_bytes += ROUND_UP(maxblocks*{dims[n]}*sizeof({typs[n]}));')
    code('')

    if GBL_READ == True and GBL_READ_MDIM == True:
      code('reallocConstArrays(block->instance,consts_bytes);')
    if GBL_INC == True or GBL_MIN == True or GBL_MAX == True or GBL_WRITE == True:
      code('reallocReductArrays(block->instance,reduct_bytes);')
      code('reduct_bytes = 0;')
      code('')

    for n in range (0, nargs):
      if arg_typ[n] == 'ops_arg_gbl' and accs[n] != OPS_READ:
        code(f'int r_bytes{n} = reduct_bytes/sizeof('+(str(typs[n]).replace('"','')).strip()+');')
        code(f'arg{n}.data = block->instance->OPS_reduct_h + reduct_bytes;')
        code(f'arg{n}.data_d = block->instance->OPS_reduct_d;// + reduct_bytes;')
        code('for (int b=0; b<maxblocks; b++)')
        if accs[n] == OPS_INC:
          code(f'for (int d=0; d<{dims[n]}; d++) (('+(str(typs[n]).replace('"','')).strip()+f' *)arg{n}.data)[d+b*{dims[n]}] = ZERO_'+(str(typs[n]).replace('"','')).strip()+';')
        if accs[n] == OPS_MAX:
          code(f'for (int d=0; d<{dims[n]}; d++) (('+(str(typs[n]).replace('"','')).strip()+f' *)arg{n}.data)[d+b*{dims[n]}] = -INFINITY_'+(str(typs[n]).replace('"','')).strip()+';')
        if accs[n] == OPS_MIN:
          code(f'for (int d=0; d<{dims[n]}; d++) (('+(str(typs[n]).replace('"','')).strip()+f' *)arg{n}.data)[d+b*{dims[n]}] = INFINITY_'+(str(typs[n]).replace('"','')).strip()+';')
        code(f'reduct_bytes += ROUND_UP(maxblocks*{dims[n]}*sizeof('+(str(typs[n]).replace('"','')).strip()+'));')
        code('')

    code('')

    for n in range (0, nargs):
      if arg_typ[n] == 'ops_arg_gbl':
        if accs[n] == OPS_READ and (not dims[n].isdigit() or int(dims[n])>1):
          code('consts_bytes = 0;')
          code(f'arg{n}.data = block->instance->OPS_consts_h + consts_bytes;')
          code(f'arg{n}.data_d = block->instance->OPS_consts_d + consts_bytes;')
          code(f'for (int d=0; d<{dims[n]}; d++) (('+(str(typs[n]).replace('"','')).strip()+f' *)arg{n}.data)[d] = arg{n}h[d];')
          code(f'consts_bytes += ROUND_UP({dims[n]}*sizeof(int));')
    if GBL_READ == True and GBL_READ_MDIM == True:
      code('mvConstArraysToDevice(block->instance,consts_bytes);')

    if GBL_INC == True or GBL_MIN == True or GBL_MAX == True or GBL_WRITE == True:
      code('mvReductArraysToDevice(block->instance,reduct_bytes);')


    comm('')
    comm('set up initial pointers')
    code('int d_m[OPS_MAX_DIM];')
    for n in range (0, nargs):
      if arg_typ[n] == 'ops_arg_dat':
        code('#ifdef OPS_MPI')
        code(f'for (int d = 0; d < dim; d++) d_m[d] = args[{n}].dat->d_m[d] + OPS_sub_dat_list[args[{n}].dat->index]->d_im[d];')
        code('#else')
        code(f'for (int d = 0; d < dim; d++) d_m[d] = args[{n}].dat->d_m[d];')
        code('#endif')
        if soa_set:
          code(f'int base{n} = 1 *')
        else:
          code(f'int base{n} = 1 *{dims[n]}*')
        code(f'(start[0] * args[{n}].stencil->stride[0] - args[{n}].dat->base[0] - d_m[0]);')
        for d in range (1, NDIM):
          line = f'base{n} = base{n} +'
          for d2 in range (0,d):
            if soa_set:
              line += f' args[{n}].dat->size[{d2}] * '
            else:
              line += f' args[{n}].dat->size[{d2}] *{dims[n]}* '
          code(line[:-1])
          code(f'(start[{d}] * args[{n}].stencil->stride[{d}] - args[{n}].dat->base[{d}] - d_m[{d}]);')
        code('')


    code('')
    code(f'ops_H_D_exchanges_device(args, {nargs});')
    code(f'ops_halo_exchanges(args,{nargs},range);')
    code(f'ops_H_D_exchanges_device(args, {nargs});')
    code('')
    IF('block->instance->OPS_diags > 1')
    code('ops_timers_core(&c2,&t2);')
    code(f'block->instance->OPS_kernels[{nk}].mpi_time += t2-t1;')
    ENDIF()
    code('')


    #set up shared memory for reduction
    if GBL_INC == True or GBL_MIN == True or GBL_MAX == True or GBL_WRITE == True:
       code('int nthread = block->instance->OPS_block_size_x*block->instance->OPS_block_size_y*block->instance->OPS_block_size_z;')
       code('')

    IF('globalWorkSize[0]>0 && globalWorkSize[1]>0 && globalWorkSize[2]>0')
    #upload global constants to device
    for c in range(0, len(found_consts)):
      const_type = consts[found_consts[c]]['type']
      const_dim = consts[found_consts[c]]['dim']
      if not (consts[found_consts[c]]['type'] == 'int' or consts[found_consts[c]]['type'] == 'double' or consts[found_consts[c]]['type'] == 'float'):
        code(f"clSafeCall( clEnqueueWriteBuffer(block->instance->opencl_instance->OPS_opencl_core.command_queue, block->instance->opencl_instance->OPS_opencl_core.constant[{found_consts[c]}], CL_TRUE, 0, sizeof({const_type})*{const_dim}, (void*) &{consts[found_consts[c]]['name'][1:-1]}, 0, NULL, NULL) );")
        code('clSafeCall( clFlush(block->instance->opencl_instance->OPS_opencl_core.command_queue) );')
    code('')

    #set up arguments in order to do the kernel call
    nkernel_args = 0
    for n in range (0, nargs):
      if arg_typ[n] == 'ops_arg_dat':
        code(f'clSafeCall( clSetKernelArg(block->instance->opencl_instance->OPS_opencl_core.kernel[{nk}], {nkernel_args}, sizeof(cl_mem), (void*) &arg{n}.data_d ));')
        nkernel_args = nkernel_args+1
      if arg_typ[n] == 'ops_arg_gbl' and accs[n] == OPS_READ:
        if dims[n].isdigit() and int(dims[n]) == 1:
          if typs[n] == 'long long' or typs[n] == 'll':
            print('OpenCL codegen error: long long is not supported by OpenCL')
            code(f'clSafeCall( clSetKernelArg(block->instance->opencl_instance->OPS_opencl_core.kernel[{nk}], {nkernel_args}, sizeof(cl_long), (void*) arg{n}.data ));')
          else:
            code(f'clSafeCall( clSetKernelArg(block->instance->opencl_instance->OPS_opencl_core.kernel[{nk}], {nkernel_args}, sizeof(cl_{typs[n]}), (void*) arg{n}.data ));')
          nkernel_args = nkernel_args+1
        else:
          code(f'clSafeCall( clSetKernelArg(block->instance->opencl_instance->OPS_opencl_core.kernel[{nk}], {nkernel_args}, sizeof(cl_mem), (void*) &arg{n}.data_d ));')
          nkernel_args = nkernel_args+1
      elif arg_typ[n] == 'ops_arg_gbl':
        code(f'clSafeCall( clSetKernelArg(block->instance->opencl_instance->OPS_opencl_core.kernel[{nk}], {nkernel_args}, sizeof(cl_mem), (void*) &arg{n}.data_d ));')
        nkernel_args = nkernel_args+1
        code(f'clSafeCall( clSetKernelArg(block->instance->opencl_instance->OPS_opencl_core.kernel[{nk}], {nkernel_args}, nthread*sizeof('+(str(typs[n]).replace('"','')).strip()+'), NULL));')
        nkernel_args = nkernel_args+1
        code(f'clSafeCall( clSetKernelArg(block->instance->opencl_instance->OPS_opencl_core.kernel[{nk}], {nkernel_args}, sizeof(cl_int), (void*) &r_bytes{n} ));')
        nkernel_args = nkernel_args+1

    for c in range(0, len(found_consts)):
      if consts[found_consts[c]]['type'] == 'int' or consts[found_consts[c]]['type'] == 'double' or consts[found_consts[c]]['type'] == 'float':
        code(f"clSafeCall( clSetKernelArg(block->instance->opencl_instance->OPS_opencl_core.kernel[{nk}], {nkernel_args}, sizeof(cl_{consts[found_consts[c]]['type']}), (void*) &{consts[found_consts[c]]['name'][1:-1]} ));")
      else:
        code(f'clSafeCall( clSetKernelArg(block->instance->opencl_instance->OPS_opencl_core.kernel[{nk}], {nkernel_args}, sizeof(cl_mem), (void*) &block->instance->opencl_instance->OPS_opencl_core.constant[{found_consts[c]}]) );')

      nkernel_args = nkernel_args+1

    for n in range (0, nargs):
      if arg_typ[n] == 'ops_arg_dat':
        code(f'clSafeCall( clSetKernelArg(block->instance->opencl_instance->OPS_opencl_core.kernel[{nk}], {nkernel_args}, sizeof(cl_int), (void*) &base{n} ));')
        nkernel_args = nkernel_args+1

    if arg_idx:
      code(f'clSafeCall( clSetKernelArg(block->instance->opencl_instance->OPS_opencl_core.kernel[{nk}], {nkernel_args}, sizeof(cl_int), (void*) &arg_idx[0] ));')
      nkernel_args = nkernel_args+1
      if NDIM==2:
        code(f'clSafeCall( clSetKernelArg(block->instance->opencl_instance->OPS_opencl_core.kernel[{nk}], {nkernel_args}, sizeof(cl_int), (void*) &arg_idx[1] ));')
        nkernel_args = nkernel_args+1
      if NDIM==3:
        code(f'clSafeCall( clSetKernelArg(block->instance->opencl_instance->OPS_opencl_core.kernel[{nk}], {nkernel_args}, sizeof(cl_int), (void*) &arg_idx[1] ));')
        nkernel_args = nkernel_args+1
        code(f'clSafeCall( clSetKernelArg(block->instance->opencl_instance->OPS_opencl_core.kernel[{nk}], {nkernel_args}, sizeof(cl_int), (void*) &arg_idx[2] ));')
        nkernel_args = nkernel_args+1

    code(f'clSafeCall( clSetKernelArg(block->instance->opencl_instance->OPS_opencl_core.kernel[{nk}], {nkernel_args}, sizeof(cl_int), (void*) &x_size ));')
    nkernel_args = nkernel_args+1
    if NDIM==2:
      code(f'clSafeCall( clSetKernelArg(block->instance->opencl_instance->OPS_opencl_core.kernel[{nk}], {nkernel_args}, sizeof(cl_int), (void*) &y_size ));')
      nkernel_args = nkernel_args+1
    if NDIM==3:
      code(f'clSafeCall( clSetKernelArg(block->instance->opencl_instance->OPS_opencl_core.kernel[{nk}], {nkernel_args}, sizeof(cl_int), (void*) &y_size ));')
      nkernel_args = nkernel_args+1
      code(f'clSafeCall( clSetKernelArg(block->instance->opencl_instance->OPS_opencl_core.kernel[{nk}], {nkernel_args}, sizeof(cl_int), (void*) &z_size ));')
      nkernel_args = nkernel_args+1

    #kernel call
    code('')
    comm('call/enqueue opencl kernel wrapper function')
    code(f'clSafeCall( clEnqueueNDRangeKernel(block->instance->opencl_instance->OPS_opencl_core.command_queue, block->instance->opencl_instance->OPS_opencl_core.kernel[{nk}], 3, NULL, globalWorkSize, localWorkSize, 0, NULL, NULL) );')
    ENDIF()
    IF('block->instance->OPS_diags>1')
    code('clSafeCall( clFinish(block->instance->opencl_instance->OPS_opencl_core.command_queue) );')
    ENDIF()
    code('')

    IF('block->instance->OPS_diags > 1')
    code('ops_timers_core(&c1,&t1);')
    code(f'block->instance->OPS_kernels[{nk}].time += t1-t2;')
    ENDIF()
    code('')

    if GBL_INC == True or GBL_MIN == True or GBL_MAX == True or GBL_WRITE == True:
      code('mvReductArraysToHost(block->instance,reduct_bytes);')

    for n in range (0, nargs):
      if arg_typ[n] == 'ops_arg_gbl' and accs[n] != OPS_READ:
        FOR('b','0','maxblocks')
        FOR('d','0',str(dims[n]))
        if accs[n] == OPS_INC:
          code(f'arg{n}h[d] = arg{n}h[d] + (({typs[n]} *)arg{n}.data)[d+b*{dims[n]}];')
        elif accs[n] == OPS_MAX:
          code(f'arg{n}h[d] = MAX(arg{n}h[d],(({typs[n]} *)arg{n}.data)[d+b*{dims[n]}]);')
        elif accs[n] == OPS_MIN:
          code(f'arg{n}h[d] = MIN(arg{n}h[d],(({typs[n]} *)arg{n}.data)[d+b*{dims[n]}]);')
        ENDFOR()
        ENDFOR()
        code(f'arg{n}.data = (char *)arg{n}h;')
        code('')

    code(f'ops_set_dirtybit_device(args, {nargs});')
    for n in range (0, nargs):
      if arg_typ[n] == 'ops_arg_dat' and (accs[n] == OPS_WRITE or accs[n] == OPS_RW or accs[n] == OPS_INC):
        code(f'ops_set_halo_dirtybit3(&args[{n}],range);')

    code('')
    IF('block->instance->OPS_diags > 1')
    comm('Update kernel record')
    code('ops_timers_core(&c2,&t2);')
    code(f'block->instance->OPS_kernels[{nk}].mpi_time += t2-t1;')
    for n in range (0, nargs):
      if arg_typ[n] == 'ops_arg_dat':
        code(f'block->instance->OPS_kernels[{nk}].transfer += ops_compute_transfer(dim, start, end, &arg{n});')
    ENDIF()
    config.depth = config.depth - 2
    code('}')

    ##########################################################################
    #  output individual kernel file
    ##########################################################################
    util.write_text_to_file(f"./OpenCL/{name}_opencl_kernel.cpp")

  # end of main kernel call loop

  ##########################################################################
  #  output one master kernel file
  ##########################################################################
  comm('header')
  code('#define OPS_API 2')
  if NDIM==1:
    code('#define OPS_1D')
  if NDIM==2:
    code('#define OPS_2D')
  if NDIM==3:
    code('#define OPS_3D')
  if soa_set:
    code('#define OPS_SOA')
  code('#include "stdlib.h"')
  code('#include "stdio.h"')
  code('#include "ops_lib_core.h"')
  code('#include "ops_opencl_rt_support.h"')
  if os.path.exists('./user_types.h'):
    code('#include "user_types.h"')
  code('#ifdef OPS_MPI')
  code('#include "ops_mpi_core.h"')
  code('#endif')

  comm('global constants')
  for nc in range (0,len(consts)):
    if consts[nc]['dim'].isdigit() and int(consts[nc]['dim'])==1:
      code(f"extern {consts[nc]['type']} "+(str(consts[nc]['name']).replace('"','')).strip()+';')
    else:
      if consts[nc]['dim'].isdigit() and int(consts[nc]['dim']) > 0:
        num = consts[nc]['dim']
        code(f"extern {consts[nc]['type']} "+(str(consts[nc]['name']).replace('"','')).strip()+f'[{num}];')
      else:
        code(f"extern {consts[nc]['type']} *"+(str(consts[nc]['name']).replace('"','')).strip()+';')

  code('')

  code('')
  code('void ops_init_backend() {}')
  code('')
  comm('this needs to be a platform specific copy symbol to device function')
  code('void ops_decl_const_char(OPS_instance *instance, int dim, char const * type, int typeSize, char * dat, char const * name ) {')
  config.depth =config.depth + 2
  code('ops_execute(instance);')
  code('cl_int ret = 0;')
  IF('instance->opencl_instance->OPS_opencl_core.constant == NULL')
  code('instance->opencl_instance->OPS_opencl_core.constant = (cl_mem*) malloc(('+str(len(consts))+')*sizeof(cl_mem));')
  FOR('i','0',str(len(consts)))
  code('instance->opencl_instance->OPS_opencl_core.constant[i] = NULL;')
  ENDFOR()
  ENDIF()

  for nc in range(0,len(consts)):
    IF('!strcmp(name,"'+(str(consts[nc]['name']).replace('"','')).strip()+'")')
    IF(f'instance->opencl_instance->OPS_opencl_core.constant[{nc}] == NULL')
    code(f'instance->opencl_instance->OPS_opencl_core.constant[{nc}] = clCreateBuffer(instance->opencl_instance->OPS_opencl_core.context, CL_MEM_READ_ONLY, dim*typeSize, NULL, &ret);')
    code('clSafeCall( ret );')
    ENDIF()
    comm('Write the new constant to the memory of the device')
    code(f'clSafeCall( clEnqueueWriteBuffer(instance->opencl_instance->OPS_opencl_core.command_queue, instance->opencl_instance->OPS_opencl_core.constant[{nc}], CL_TRUE, 0, dim*typeSize, (void*) dat, 0, NULL, NULL) );')
    code('clSafeCall( clFlush(instance->opencl_instance->OPS_opencl_core.command_queue) );')
    code('clSafeCall( clFinish(instance->opencl_instance->OPS_opencl_core.command_queue) );')
    ENDIF()
    code('else')

  code('{')
  config.depth = config.depth + 2
  code('throw OPSException(OPS_RUNTIME_ERROR, "error: unknown const name");')
  ENDIF()
  config.depth = config.depth - 2
  code('}')


  opencl_build = """


  void buildOpenCLKernels(OPS_instance *instance) {
    static bool isbuilt = false;

    if(!isbuilt) {
      //clSafeCall( clUnloadCompiler() );

      instance->opencl_instance->OPS_opencl_core.n_kernels = """+str(len(kernels))+""";
      instance->opencl_instance->OPS_opencl_core.kernel = (cl_kernel*) malloc("""+str(len(kernels))+"""*sizeof(cl_kernel));
    }
    isbuilt = true;
  }
  """


  config.depth = -2
  code(opencl_build)


  comm('user kernel files')

  for kernel_name in sorted(map(lambda kernel: kernel['name'], kernels)):
    if not (('initialise' in kernel_name) or ('generate' in kernel_name)):
      code(f"#include \"{kernel_name}_opencl_kernel.cpp\"")
    else:
       code(f"#include \"../MPI_OpenMP/{kernel_name}_cpu_kernel.cpp\"")


  util.write_text_to_file(
      f"./OpenCL/{master_basename[0]}_opencl_kernels.cpp",
      "//\n// auto-generated by ops.py//\n\n",
  )
