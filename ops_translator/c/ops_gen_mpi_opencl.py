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
import datetime
import os
import glob

import util
import config

para_parse = util.para_parse
comment_remover = util.comment_remover
remove_trailing_w_space = util.remove_trailing_w_space
parse_signature = util.parse_signature
check_accs = util.check_accs
arg_parse = util.arg_parse
find_consts = util.find_consts
mult = util.mult
replace_ACC_kernel_body = util.replace_ACC_kernel_body
parse_replace_ACC_signature = util.parse_replace_ACC_signature

comm = util.comm
code = util.code
FOR = util.FOR
FOR2 = util.FOR2
WHILE = util.WHILE
ENDWHILE = util.ENDWHILE
ENDFOR = util.ENDFOR
IF = util.IF
ELSEIF = util.ELSEIF
ELSE = util.ELSE
ENDIF = util.ENDIF


def ops_gen_mpi_opencl(master, date, consts, kernels, soa_set):

  OPS_ID   = 1;  OPS_GBL   = 2;  OPS_MAP = 3;

  OPS_READ = 1;  OPS_WRITE = 2;  OPS_RW  = 3;
  OPS_INC  = 4;  OPS_MAX   = 5;  OPS_MIN = 6;

  accsstring = ['OPS_READ','OPS_WRITE','OPS_RW','OPS_INC','OPS_MAX','OPS_MIN' ]

  NDIM = 2 #the dimension of the application, set to 2 by default. Will be updated later from loops

  src_dir = os.path.dirname(master) or '.'
  master_basename = os.path.splitext(os.path.basename(master))

##########################################################################
#  create new kernel files **_kernel.cl
##########################################################################

  #kernel_name_list = []
  #kernel_list_text = ''
  #kernel_list__build_text = ''
  #indent = 10*' '
  #for nk in range(0,len(kernels)):
  #  if kernels[nk]['name'] not in kernel_name_list :
  #    kernel_name_list.append(kernels[nk]['name'])

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

    if ('initialise' in name) or ('generate' in name):
      print(('WARNING: skipping kernel '+name+' due to OpenCL compiler bugs: this kernel will run sequentially on the host'))
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

    config.file_text = ''
    config.depth = 0
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
    code('#define OPS_'+str(NDIM)+'D')
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

    code('#define ZERO_double 0.0;')
    code('#define INFINITY_double INFINITY;')
    code('#define ZERO_float 0.0f;')
    code('#define INFINITY_float INFINITY;')
    code('#define ZERO_int 0;')
    code('#define INFINITY_int INFINITY;')
    code('#define ZERO_uint 0;')
    code('#define INFINITY_uint INFINITY;')
    code('#define ZERO_ll 0;')
    code('#define INFINITY_ll INFINITY;')
    code('#define ZERO_ull 0;')
    code('#define INFINITY_ull INFINITY;')
    code('#define ZERO_bool 0;')

    code('')
    comm('user function')
    found = 0
    for files in glob.glob( os.path.join(src_dir,"*.h") ):
      f = open( files, 'r' )
      for line in f:
        if name in line:
          file_name = f.name
          found = 1;
          break
      if found == 1:
        break;

    if found == 0:
      print(("COUND NOT FIND KERNEL", name))

    fid = open(file_name, 'r')
    text = fid.read()

    fid.close()
    text = comment_remover(text)
    text = remove_trailing_w_space(text)

    p = re.compile('void\\s+\\b'+name+'\\b')

    i = p.search(text).start()
    if(i < 0):
      print("\n********")
      print(("Error: cannot locate user kernel function: "+name+" - Aborting code generation"))
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
          text = text + ', const '+consts[found_consts[c]]['type']+' '+consts[found_consts[c]]['name'][1:-1]
        else:
          text = text + ', __constant const'+consts[found_consts[c]]['type']+' *'+consts[found_consts[c]]['name'][1:-1]
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
    code('__kernel void ops_'+name+'(')
    #currently the read only vars have not been generated differently
    for n in range (0, nargs):
      if arg_typ[n] == 'ops_arg_dat' and accs[n] == OPS_READ:
        code('__global const '+(str(typs[n]).replace('"','')).strip()+'* restrict arg'+str(n)+',')
      elif arg_typ[n] == 'ops_arg_dat'and (accs[n] == OPS_WRITE or accs[n] == OPS_RW or accs[n] == OPS_INC) :
        code('__global '+(str(typs[n]).replace('"','')).strip()+'* restrict arg'+str(n)+',')
      elif arg_typ[n] == 'ops_arg_gbl':
        if accs[n] == OPS_READ:
          if dims[n].isdigit() and int(dims[n]) == 1:
            code('const '+(str(typs[n]).replace('"','')).strip()+' arg'+str(n)+',')
          else:
            code('__global const '+(str(typs[n]).replace('"','')).strip()+'* restrict arg'+str(n)+',')
        else:
          code('__global '+(str(typs[n]).replace('"','')).strip()+'* restrict arg'+str(n)+',')
          code('__local '+(str(typs[n]).replace('"','')).strip()+'* scratch'+str(n)+',')
          code('int r_bytes'+str(n)+',')

    for c in range(0, len(found_consts)):
      if consts[found_consts[c]]['type']=='int' or consts[found_consts[c]]['type']=='double' or consts[found_consts[c]]['type']=='float':
        code('const '+consts[found_consts[c]]['type']+' '+consts[found_consts[c]]['name'][1:-1]+',')
      else:
        code('__constant const struct '+consts[found_consts[c]]['type']+' * restrict '+consts[found_consts[c]]['name'][1:-1]+',')


    for n in range (0, nargs):
      if arg_typ[n] == 'ops_arg_dat':
        code('const int base'+str(n)+',')


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
        code('arg'+str(n)+' += r_bytes'+str(n)+';')
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
    if NDIM==3:
      code('int idx_y = get_global_id(1);')
      code('int idx_z = get_global_id(2);')
    elif NDIM==2:
        code('int idx_y = get_global_id(1);')
    code('int idx_x = get_global_id(0);')
    code('')
    if arg_idx:
      code('int arg_idx['+str(NDIM)+'];')
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
            text = text +'&arg'+str(n)+'[base'+str(n)+\
                ' + idx_x * '+str(stride[NDIM*n])+']'
          else:
            text = text +'&arg'+str(n)+'[base'+str(n)+\
                ' + idx_x * '+str(stride[NDIM*n])+'*'+str(dims[n])+']'
        elif NDIM==2:
          if soa_set:
            text = text +'&arg'+str(n)+'[base'+str(n)+\
                ' + idx_x * '+str(stride[NDIM*n])+ \
                ' + idx_y * '+str(stride[NDIM*n+1])+' * xdim'+str(n)+'_'+name+']'
          else:
            text = text +'&arg'+str(n)+'[base'+str(n)+\
                ' + idx_x * '+str(stride[NDIM*n])+'*'+str(dims[n])+ \
                ' + idx_y * '+str(stride[NDIM*n+1])+'*'+str(dims[n])+' * xdim'+str(n)+'_'+name+']'
        elif NDIM==3:
          if soa_set:
            text = text + '&arg'+str(n)+'[base'+str(n)+\
                ' + idx_x * '+str(stride[NDIM*n])+ \
                ' + idx_y * '+str(stride[NDIM*n+1])+' * xdim'+str(n)+'_'+name+ \
                ' + idx_z * '+str(stride[NDIM*n+2])+' * xdim'+str(n)+'_'+name+' * ydim'+str(n)+'_'+name+']'
          else:
            text = text + '&arg'+str(n)+'[base'+str(n)+\
                ' + idx_x * '+str(stride[NDIM*n])+'*'+str(dims[n])+ \
                ' + idx_y * '+str(stride[NDIM*n+1])+'*'+str(dims[n])+' * xdim'+str(n)+'_'+name+ \
                ' + idx_z * '+str(stride[NDIM*n+2])+'*'+str(dims[n])+' * xdim'+str(n)+'_'+name+' * ydim'+str(n)+'_'+name+']'
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
            dim = 'arg'+str(n)+'.dim'
            extradim = 1
        dimlabels = 'xyzuv'
        for i in range(1,NDIM):
          sizelist = sizelist + dimlabels[i-1]+'dim'+str(n)+'_'+name+', '
        extradim = dimlabels[NDIM+extradim-2]+'dim'+str(n)+'_'+name
        if dim == '':
          if NDIM==1:
            code(pre+'ptr_'+typs[n]+' ptr'+str(n)+' = { '+text+' };')
          else:
            code(pre+'ptr_'+typs[n]+' ptr'+str(n)+' = { '+text+', '+sizelist[:-2]+'};')
        else:
          code('#ifdef OPS_SOA')
          code(pre+'ptrm_'+typs[n]+' ptr'+str(n)+' = { '+text+', '+sizelist + extradim+'};')
          code('#else')
          code(pre+'ptrm_'+typs[n]+' ptr'+str(n)+' = { '+text+', '+sizelist+dim+'};')
          code('#endif')


    text = name+'('
    for n in range (0, nargs):
      if arg_typ[n] == 'ops_arg_dat':
          text = text + 'ptr'+str(n)
      elif arg_typ[n] == 'ops_arg_gbl' and accs[n] == OPS_READ:
        if dims[n].isdigit() and int(dims[n]) == 1:
          text = text +'&arg'+str(n)
        else:
          text = text +'arg'+str(n)
      elif arg_typ[n] == 'ops_arg_gbl' and accs[n] != OPS_READ:
        text = text +'arg'+str(n)+'_l'
      elif arg_typ[n] == 'ops_arg_idx':
        text = text +'arg_idx'

      if n != nargs-1 :
        text = text+',\n  '+indent
      elif len(found_consts) > 0:
        text = text +',\n  '+indent



    for c in range(0, len(found_consts)):
      #if (consts[found_consts[c]]['dim']).isdigit() and int(consts[found_consts[c]]['dim'])==1:
      if consts[found_consts[c]]['type'] == 'int' or consts[found_consts[c]]['type'] == 'double' or consts[found_consts[c]]['type'] == 'float':
        text = text + consts[found_consts[c]]['name'][1:-1]
      else:
        text = text + '*'+consts[found_consts[c]]['name'][1:-1]
      if c != len(found_consts)-1:
        text = text+',\n  '+indent

    code(text+');')

    ENDIF()

    #reduction accross blocks
    if reduction:
      code('int group_index = get_group_id(0) + get_group_id(1)*get_num_groups(0)+ get_group_id(2)*get_num_groups(0)*get_num_groups(1);')
      for n in range (0, nargs):
        if arg_typ[n] == 'ops_arg_gbl' and accs[n] == OPS_INC:
          code('for (int d=0; d<'+str(dims[n])+'; d++)')
          code('  reduce_'+(str(typs[n]).replace('"','')).strip()+'(arg'+str(n)+'_l[d], scratch'+str(n)+', &arg'+str(n)+'[group_index*'+str(dims[n])+'+d], OPS_INC);')
        if arg_typ[n] == 'ops_arg_gbl' and accs[n] == OPS_MIN:
          code('for (int d=0; d<'+str(dims[n])+'; d++)')
          code('  reduce_'+(str(typs[n]).replace('"','')).strip()+'(arg'+str(n)+'_l[d], scratch'+str(n)+', &arg'+str(n)+'[group_index*'+str(dims[n])+'+d], OPS_MIN);')
        if arg_typ[n] == 'ops_arg_gbl' and accs[n] == OPS_MAX:
          code('for (int d=0; d<'+str(dims[n])+'; d++)')
          code('  reduce_'+(str(typs[n]).replace('"','')).strip()+'(arg'+str(n)+'_l[d], scratch'+str(n)+', &arg'+str(n)+'[group_index*'+str(dims[n])+'+d], OPS_MAX);')


    code('')
    config.depth = config.depth - 2
    code('}')


    if not os.path.exists('./OpenCL'):
      os.makedirs('./OpenCL')
    fid = open('./OpenCL/'+name+'.cl','w')
    date = datetime.datetime.now()
    fid.write('//\n// auto-generated by ops.py\n//\n')
    fid.write(config.file_text)
    fid.close()

##########################################################################
#  generate opencl kernel build function
##########################################################################

    kernel_list_text = '"./OpenCL/'+name+'.cl"'
    arg_text = ''
    compile_line = ''
    arg_values = ''
    for n in range (0, nargs):
      if arg_typ[n] == 'ops_arg_dat':
        arg_text = arg_text +'int_xdim'+str(n)+' '
        compile_line = compile_line + ' -Dxdim'+str(n)+'_'+name+'=%d'+' '
        arg_values = arg_values + 'xdim'+str(n)+' '
        if NDIM>2 or (NDIM==2 and soa_set):
          arg_text = arg_text +'int_ydim'+str(n)+' '
          compile_line = compile_line + ' -Dydim'+str(n)+'_'+name+'=%d'+' '
          arg_values = arg_values + 'ydim'+str(n)+' '
        if NDIM>3 or (NDIM==3 and soa_set):
          arg_text = arg_text +'int_zdim'+str(n)+' '
          compile_line = compile_line + ' -Dzdim'+str(n)+'_'+name+'=%d'+' '
          arg_values = arg_values + 'zdim'+str(n)+' '

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

void buildOpenCLKernels_"""+name+"""("""+arg_text+""") {

  //int ocl_fma = OCL_FMA;
  if(!isbuilt_"""+name+""") {
    buildOpenCLKernels();
    //clSafeCall( clUnloadCompiler() );
    cl_int ret;
    char* source_filename[1] = {(char*)"""+kernel_list_text+"""};

    // Load the kernel source code into the array source_str
    FILE *fid;
    char *source_str[1];
    size_t source_size[1];

    for(int i=0; i<1; i++) {
      fid = fopen(source_filename[i], "r");
      if (!fid) {
        fprintf(stderr, "Can't open the kernel source file!\\n");
        exit(1);
      }

      source_str[i] = (char*)malloc(4*0x1000000);
      source_size[i] = fread(source_str[i], 1, 4*0x1000000, fid);
      if(source_size[i] != 4*0x1000000) {
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

    printf("Compiling """+name+""" %d source -- start \\n",OCL_FMA);

      // Create a program from the source
      OPS_instance::getOPSInstance()->opencl_instance->OPS_opencl_core.program = clCreateProgramWithSource(OPS_instance::getOPSInstance()->opencl_instance->OPS_opencl_core.context, 1, (const char **) &source_str, (const size_t *) &source_size, &ret);
      clSafeCall( ret );

      // Build the program
      char buildOpts[255*"""+str(nargs)+"""];
      char* pPath = NULL;
      pPath = getenv ("OPS_INSTALL_PATH");
      if (pPath!=NULL)
        if(OCL_FMA)
          sprintf(buildOpts,"-cl-mad-enable -DOCL_FMA -I%s/c/include -DOPS_WARPSIZE=%d """+compile_line+""", pPath, 32,"""+arg_values+""");
        else
          sprintf(buildOpts,"-cl-mad-enable -I%s/c/include -DOPS_WARPSIZE=%d """+compile_line+""", pPath, 32,"""+arg_values+""");
      else {
        sprintf((char*)"Incorrect OPS_INSTALL_PATH %s\\n",pPath);
        exit(EXIT_FAILURE);
      }

      #ifdef OPS_SOA
      sprintf(buildOpts, "%s -DOPS_SOA", buildOpts);
      #endif

      ret = clBuildProgram(OPS_instance::getOPSInstance()->opencl_instance->OPS_opencl_core.program, 1, &OPS_instance::getOPSInstance()->opencl_instance->OPS_opencl_core.device_id, buildOpts, NULL, NULL);

      if(ret != CL_SUCCESS) {
        char* build_log;
        size_t log_size;
        clSafeCall( clGetProgramBuildInfo(OPS_instance::getOPSInstance()->opencl_instance->OPS_opencl_core.program, OPS_instance::getOPSInstance()->opencl_instance->OPS_opencl_core.device_id, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size) );
        build_log = (char*) malloc(log_size+1);
        clSafeCall( clGetProgramBuildInfo(OPS_instance::getOPSInstance()->opencl_instance->OPS_opencl_core.program, OPS_instance::getOPSInstance()->opencl_instance->OPS_opencl_core.device_id, CL_PROGRAM_BUILD_LOG, log_size, build_log, NULL) );
        build_log[log_size] = '\\0';
        fprintf(stderr, "=============== OpenCL Program Build Info ================\\n\\n%s", build_log);
        fprintf(stderr, "\\n========================================================= \\n");
        free(build_log);
        exit(EXIT_FAILURE);
      }
      printf("compiling """+name+""" -- done\\n");

    // Create the OpenCL kernel
    OPS_instance::getOPSInstance()->opencl_instance->OPS_opencl_core.kernel["""+str(nk)+"""] = clCreateKernel(OPS_instance::getOPSInstance()->opencl_instance->OPS_opencl_core.program, "ops_"""+name+"""", &ret);
    clSafeCall( ret );\n
    isbuilt_"""+name+""" = true;
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

    code('void ops_par_loop_'+name+'(char const *name, ops_block block, int dim, int* range,')
    text = ''
    for n in range (0, nargs):

      text = text +' ops_arg arg'+str(n)
      if nargs != 1 and n != nargs-1:
        text = text +','
      else:
        text = text +') {'
      if n%n_per_line == 3 and n != nargs-1:
         text = text +'\n'
    code(text);
    config.depth = 2

    #timing structs
    code('')
    comm('Timing')
    code('double t1,t2,c1,c2;')
    code('')

    text ='ops_arg args['+str(nargs)+'] = {'
    for n in range (0, nargs):
      text = text +' arg'+str(n)
      if nargs != 1 and n != nargs-1:
        text = text +','
      else:
        text = text +'};\n'
      if n%n_per_line == 5 and n != nargs-1:
        text = text +'\n                    '
    code(text);
    code('')
    code('#ifdef CHECKPOINTING')
    code('if (!ops_checkpointing_before(args,'+str(nargs)+',range,'+str(nk)+')) return;')
    code('#endif')
    code('')

    IF('OPS_instance::getOPSInstance()->OPS_diags > 1')
    code('ops_timing_realloc('+str(nk)+',"'+name+'");')
    code('OPS_instance::getOPSInstance()->OPS_kernels['+str(nk)+'].count++;')
    code('ops_timers_core(&c1,&t1);')
    ENDIF()
    code('')

    comm('compute locally allocated range for the sub-block')

    code('int start['+str(NDIM)+'];')
    code('int end['+str(NDIM)+'];')


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
      code('int arg_idx['+str(NDIM)+'];')
      code('#ifdef OPS_MPI')
      for n in range (0,NDIM):
        code('arg_idx['+str(n)+'] = sb->decomp_disp['+str(n)+']+start['+str(n)+'];')
      code('#else')
      for n in range (0,NDIM):
        code('arg_idx['+str(n)+'] = start['+str(n)+'];')
      code('#endif')
    code('')


    for n in range (0, nargs):
      if arg_typ[n] == 'ops_arg_dat':
        code('int xdim'+str(n)+' = args['+str(n)+'].dat->size[0];')
        if NDIM>2 or (NDIM==2 and soa_set):
          code('int ydim'+str(n)+' = args['+str(n)+'].dat->size[1];')
        if NDIM>3 or (NDIM==3 and soa_set):
          code('int zdim'+str(n)+' = args['+str(n)+'].dat->size[2];')
    code('')

    comm('build opencl kernel if not already built')
    code('');
    code('buildOpenCLKernels_'+name+'(')
    arg_text = ''
    for n in range (0, nargs):
      if arg_typ[n] == 'ops_arg_dat':
        arg_text = arg_text +'xdim'+str(n)+' '
        if NDIM>2 or (NDIM==2 and soa_set):
          arg_text = arg_text +'ydim'+str(n)+' '
        if NDIM>3 or (NDIM==3 and soa_set):
          arg_text = arg_text +'zdim'+str(n)+' '

    ' '.join(arg_text.split())
    arg_text = arg_text.replace(' ',',')
    arg_text = arg_text[:-1]


    code(arg_text+');')
    code('')

    #set up OpenCL grid and thread blocks
    comm('set up OpenCL thread blocks')
    if NDIM==1:
      code('size_t globalWorkSize[3] = {((x_size-1)/OPS_instance::getOPSInstance()->OPS_block_size_x+ 1)*OPS_instance::getOPSInstance()->OPS_block_size_x, 1, 1};')
    if NDIM==2:
      code('size_t globalWorkSize[3] = {((x_size-1)/OPS_instance::getOPSInstance()->OPS_block_size_x+ 1)*OPS_instance::getOPSInstance()->OPS_block_size_x, ((y_size-1)/OPS_instance::getOPSInstance()->OPS_block_size_y + 1)*OPS_instance::getOPSInstance()->OPS_block_size_y, 1};')
    if NDIM==3:
      code('size_t globalWorkSize[3] = {((x_size-1)/OPS_instance::getOPSInstance()->OPS_block_size_x+ 1)*OPS_instance::getOPSInstance()->OPS_block_size_x, ((y_size-1)/OPS_instance::getOPSInstance()->OPS_block_size_y + 1)*OPS_instance::getOPSInstance()->OPS_block_size_y, ((z_size-1)/OPS_instance::getOPSInstance()->OPS_block_size_z+ 1)*OPS_instance::getOPSInstance()->OPS_block_size_z};')

    if NDIM>1:
      code('size_t localWorkSize[3] =  {OPS_instance::getOPSInstance()->OPS_block_size_x,OPS_instance::getOPSInstance()->OPS_block_size_y,OPS_instance::getOPSInstance()->OPS_block_size_z};')
    else:
      code('size_t localWorkSize[3] =  {OPS_instance::getOPSInstance()->OPS_block_size_x,1,1};')
    code('')

    #setup reduction variables
    code('')
    for n in range (0, nargs):
        if arg_typ[n] == 'ops_arg_gbl' and (accs[n] != OPS_READ or (accs[n] == OPS_READ and (not dims[n].isdigit() or int(dims[n])>1))):
          if (accs[n] == OPS_READ):
            code(''+typs[n]+' *arg'+str(n)+'h = ('+typs[n]+' *)arg'+str(n)+'.data;')
          else:
            code('#ifdef OPS_MPI')
            code(typs[n]+' *arg'+str(n)+'h = ('+typs[n]+' *)(((ops_reduction)args['+str(n)+'].data)->data + ((ops_reduction)args['+str(n)+'].data)->size * block->index);')
            code('#else')
            code(typs[n]+' *arg'+str(n)+'h = ('+typs[n]+' *)(((ops_reduction)args['+str(n)+'].data)->data);')
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
        code('int nblocks = ((x_size-1)/OPS_instance::getOPSInstance()->OPS_block_size_x+ 1);')
      elif NDIM==2:
        code('int nblocks = ((x_size-1)/OPS_instance::getOPSInstance()->OPS_block_size_x+ 1)*((y_size-1)/OPS_instance::getOPSInstance()->OPS_block_size_y + 1);')
      elif NDIM==3:
        code('int nblocks = ((x_size-1)/OPS_instance::getOPSInstance()->OPS_block_size_x+ 1)*((y_size-1)/OPS_instance::getOPSInstance()->OPS_block_size_y + 1)*((z_size-1)/OPS_instance::getOPSInstance()->OPS_block_size_z + 1);')
      code('int maxblocks = nblocks;')
      code('int reduct_bytes = 0;')
      code('')

    if GBL_READ == True and GBL_READ_MDIM == True:
      code('int consts_bytes = 0;')
      code('')

    for n in range (0, nargs):
      if arg_typ[n] == 'ops_arg_gbl':
        if accs[n] == OPS_READ and (not dims[n].isdigit() or int(dims[n])>1):
            code('consts_bytes += ROUND_UP('+str(dims[n])+'*sizeof('+(str(typs[n]).replace('"','')).strip()+'));')
        elif accs[n] != OPS_READ:
          #code('reduct_bytes += ROUND_UP(maxblocks*'+str(dims[n])+'*sizeof('+(str(typs[n]).replace('"','')).strip()+')*64);')
          code('reduct_bytes += ROUND_UP(maxblocks*'+str(dims[n])+'*sizeof('+typs[n]+'));')
    code('')

    if GBL_READ == True and GBL_READ_MDIM == True:
      code('reallocConstArrays(consts_bytes);')
    if GBL_INC == True or GBL_MIN == True or GBL_MAX == True or GBL_WRITE == True:
      code('reallocReductArrays(reduct_bytes);')
      code('reduct_bytes = 0;')
      code('')

    for n in range (0, nargs):
      if arg_typ[n] == 'ops_arg_gbl' and accs[n] != OPS_READ:
        code('int r_bytes'+str(n)+' = reduct_bytes/sizeof('+(str(typs[n]).replace('"','')).strip()+');')
        code('arg'+str(n)+'.data = OPS_instance::getOPSInstance()->OPS_reduct_h + reduct_bytes;')
        code('arg'+str(n)+'.data_d = OPS_instance::getOPSInstance()->OPS_reduct_d;// + reduct_bytes;')
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
        if accs[n] == OPS_READ and (not dims[n].isdigit() or int(dims[n])>1):
          code('consts_bytes = 0;')
          code('arg'+str(n)+'.data = OPS_consts_h + consts_bytes;')
          code('arg'+str(n)+'.data_d = OPS_consts_d + consts_bytes;')
          code('for (int d=0; d<'+str(dims[n])+'; d++) (('+(str(typs[n]).replace('"','')).strip()+' *)arg'+str(n)+'.data)[d] = arg'+str(n)+'h[d];')
          code('consts_bytes += ROUND_UP('+str(dims[n])+'*sizeof(int));')
    if GBL_READ == True and GBL_READ_MDIM == True:
      code('mvConstArraysToDevice(consts_bytes);')

    if GBL_INC == True or GBL_MIN == True or GBL_MAX == True or GBL_WRITE == True:
      code('mvReductArraysToDevice(reduct_bytes);')


    #set up initial pointers
    #for n in range (0, nargs):
    #  if arg_typ[n] == 'ops_arg_dat':
    #    code('int dat'+str(n)+' = args['+str(n)+'].dat->elem_size;')


    comm('')
    comm('set up initial pointers')
    code('int d_m[OPS_MAX_DIM];')
    for n in range (0, nargs):
      if arg_typ[n] == 'ops_arg_dat':
        code('#ifdef OPS_MPI')
        code('for (int d = 0; d < dim; d++) d_m[d] = args['+str(n)+'].dat->d_m[d] + OPS_sub_dat_list[args['+str(n)+'].dat->index]->d_im[d];')
        code('#else')
        code('for (int d = 0; d < dim; d++) d_m[d] = args['+str(n)+'].dat->d_m[d];')
        code('#endif')
        if soa_set:
          code('int base'+str(n)+' = 1 *')
        else:
          code('int base'+str(n)+' = 1 *'+str(dims[n])+'*')
        code('(start[0] * args['+str(n)+'].stencil->stride[0] - args['+str(n)+'].dat->base[0] - d_m[0]);')
        for d in range (1, NDIM):
          line = 'base'+str(n)+' = base'+str(n)+' +'
          for d2 in range (0,d):
            if soa_set:
              line = line + ' args['+str(n)+'].dat->size['+str(d2)+'] * '
            else:
              line = line + ' args['+str(n)+'].dat->size['+str(d2)+'] *'+str(dims[n])+'* '
          code(line[:-1])
          code('(start['+str(d)+'] * args['+str(n)+'].stencil->stride['+str(d)+'] - args['+str(n)+'].dat->base['+str(d)+'] - d_m['+str(d)+']);')
        #code('printf("base'+str(n)+' = %d, d_m[0] = %d\\n",base'+str(n)+',d_m[0]);')
        code('')


    code('')
    code('ops_H_D_exchanges_device(args, '+str(nargs)+');')
    code('ops_halo_exchanges(args,'+str(nargs)+',range);')
    code('ops_H_D_exchanges_device(args, '+str(nargs)+');')
    code('')
    IF('OPS_instance::getOPSInstance()->OPS_diags > 1')
    code('ops_timers_core(&c2,&t2);')
    code('OPS_instance::getOPSInstance()->OPS_kernels['+str(nk)+'].mpi_time += t2-t1;')
    ENDIF()
    code('')


    #set up shared memory for reduction
    if GBL_INC == True or GBL_MIN == True or GBL_MAX == True or GBL_WRITE == True:
       code('int nthread = OPS_instance::getOPSInstance()->OPS_block_size_x*OPS_instance::getOPSInstance()->OPS_block_size_y*OPS_instance::getOPSInstance()->OPS_block_size_z;')
       code('')

    IF('globalWorkSize[0]>0 && globalWorkSize[1]>0 && globalWorkSize[2]>0')
    #upload gloabal constants to device
    for c in range(0, len(found_consts)):
      const_type = consts[found_consts[c]]['type']
      const_dim = consts[found_consts[c]]['dim']
      #if const_dim.isdigit() and int(const_dim)==1:
        #code('clSafeCall( clEnqueueWriteBuffer(OPS_instance::getOPSInstance()->opencl_instance->OPS_opencl_core.command_queue, OPS_instance::getOPSInstance()->opencl_instance->OPS_opencl_core.constant['+str(found_consts[c])+'], CL_TRUE, 0, sizeof('+const_type+')*'+const_dim+', (void*) &'+consts[found_consts[c]]['name'][1:-1]+', 0, NULL, NULL) );')
      #  code('')
      #else:
      if not (consts[found_consts[c]]['type'] == 'int' or consts[found_consts[c]]['type'] == 'double' or consts[found_consts[c]]['type'] == 'float'):
        code('clSafeCall( clEnqueueWriteBuffer(OPS_instance::getOPSInstance()->opencl_instance->OPS_opencl_core.command_queue, OPS_instance::getOPSInstance()->opencl_instance->OPS_opencl_core.constant['+str(found_consts[c])+'], CL_TRUE, 0, sizeof('+const_type+')*'+const_dim+', (void*) &'+consts[found_consts[c]]['name'][1:-1]+', 0, NULL, NULL) );')
        code('clSafeCall( clFlush(OPS_instance::getOPSInstance()->opencl_instance->OPS_opencl_core.command_queue) );')
    code('')

    #set up arguments in order to do the kernel call
    nkernel_args = 0
    for n in range (0, nargs):
      if arg_typ[n] == 'ops_arg_dat':
        code('clSafeCall( clSetKernelArg(OPS_instance::getOPSInstance()->opencl_instance->OPS_opencl_core.kernel['+str(nk)+'], '+str(nkernel_args)+', sizeof(cl_mem), (void*) &arg'+str(n)+'.data_d ));')
        nkernel_args = nkernel_args+1
      if arg_typ[n] == 'ops_arg_gbl' and accs[n] == OPS_READ:
        if dims[n].isdigit() and int(dims[n]) == 1:
          code('clSafeCall( clSetKernelArg(OPS_instance::getOPSInstance()->opencl_instance->OPS_opencl_core.kernel['+str(nk)+'], '+str(nkernel_args)+', sizeof(cl_'+typs[n]+'), (void*) arg'+str(n)+'.data ));')
          nkernel_args = nkernel_args+1
        else:
          code('clSafeCall( clSetKernelArg(OPS_instance::getOPSInstance()->opencl_instance->OPS_opencl_core.kernel['+str(nk)+'], '+str(nkernel_args)+', sizeof(cl_mem), (void*) &arg'+str(n)+'.data_d ));')
          nkernel_args = nkernel_args+1
      elif arg_typ[n] == 'ops_arg_gbl':
        code('clSafeCall( clSetKernelArg(OPS_instance::getOPSInstance()->opencl_instance->OPS_opencl_core.kernel['+str(nk)+'], '+str(nkernel_args)+', sizeof(cl_mem), (void*) &arg'+str(n)+'.data_d ));')
        nkernel_args = nkernel_args+1
        code('clSafeCall( clSetKernelArg(OPS_instance::getOPSInstance()->opencl_instance->OPS_opencl_core.kernel['+str(nk)+'], '+str(nkernel_args)+', nthread*sizeof('+(str(typs[n]).replace('"','')).strip()+'), NULL));')
        nkernel_args = nkernel_args+1
        code('clSafeCall( clSetKernelArg(OPS_instance::getOPSInstance()->opencl_instance->OPS_opencl_core.kernel['+str(nk)+'], '+str(nkernel_args)+', sizeof(cl_int), (void*) &r_bytes'+str(n)+' ));')
        nkernel_args = nkernel_args+1

    for c in range(0, len(found_consts)):
      if consts[found_consts[c]]['type'] == 'int' or consts[found_consts[c]]['type'] == 'double' or consts[found_consts[c]]['type'] == 'float':
        #code('clSafeCall( clSetKernelArg(OPS_instance::getOPSInstance()->opencl_instance->OPS_opencl_core.kernel['+str(nk)+'], '+str(nkernel_args)+', sizeof(cl_mem'+\
        #     consts[found_consts[c]]['type']+'), (void*) &'+consts[found_consts[c]]['name'][1:-1]+' ));')
        code('clSafeCall( clSetKernelArg(OPS_instance::getOPSInstance()->opencl_instance->OPS_opencl_core.kernel['+str(nk)+'], '+str(nkernel_args)+', sizeof(cl_'+consts[found_consts[c]]['type']+\
             '), (void*) &'+consts[found_consts[c]]['name'][1:-1]+' ));')
      else:
        code('clSafeCall( clSetKernelArg(OPS_instance::getOPSInstance()->opencl_instance->OPS_opencl_core.kernel['+str(nk)+'], '+str(nkernel_args)+', sizeof(cl_mem), (void*) &OPS_instance::getOPSInstance()->opencl_instance->OPS_opencl_core.constant['+str(found_consts[c])+']) );')
        #code(consts[found_consts[c]]['type']+'clSafeCall( clSetKernelArg(OPS_instance::getOPSInstance()->opencl_instance->OPS_opencl_core.kernel['+str(nk)+'], '+str(nkernel_args)+', sizeof(cl_mem), (void*) &'+consts[found_consts[c]]['name'][1:-1]+') );')

      nkernel_args = nkernel_args+1

    for n in range (0, nargs):
      if arg_typ[n] == 'ops_arg_dat':
        code('clSafeCall( clSetKernelArg(OPS_instance::getOPSInstance()->opencl_instance->OPS_opencl_core.kernel['+str(nk)+'], '+str(nkernel_args)+', sizeof(cl_int), (void*) &base'+str(n)+' ));')
        nkernel_args = nkernel_args+1

    if arg_idx:
      code('clSafeCall( clSetKernelArg(OPS_instance::getOPSInstance()->opencl_instance->OPS_opencl_core.kernel['+str(nk)+'], '+str(nkernel_args)+', sizeof(cl_int), (void*) &arg_idx[0] ));')
      nkernel_args = nkernel_args+1
      if NDIM==2:
        code('clSafeCall( clSetKernelArg(OPS_instance::getOPSInstance()->opencl_instance->OPS_opencl_core.kernel['+str(nk)+'], '+str(nkernel_args)+', sizeof(cl_int), (void*) &arg_idx[1] ));')
        nkernel_args = nkernel_args+1
      if NDIM==3:
        code('clSafeCall( clSetKernelArg(OPS_instance::getOPSInstance()->opencl_instance->OPS_opencl_core.kernel['+str(nk)+'], '+str(nkernel_args)+', sizeof(cl_int), (void*) &arg_idx[1] ));')
        nkernel_args = nkernel_args+1
        code('clSafeCall( clSetKernelArg(OPS_instance::getOPSInstance()->opencl_instance->OPS_opencl_core.kernel['+str(nk)+'], '+str(nkernel_args)+', sizeof(cl_int), (void*) &arg_idx[2] ));')
        nkernel_args = nkernel_args+1

    code('clSafeCall( clSetKernelArg(OPS_instance::getOPSInstance()->opencl_instance->OPS_opencl_core.kernel['+str(nk)+'], '+str(nkernel_args)+', sizeof(cl_int), (void*) &x_size ));')
    nkernel_args = nkernel_args+1
    if NDIM==2:
      code('clSafeCall( clSetKernelArg(OPS_instance::getOPSInstance()->opencl_instance->OPS_opencl_core.kernel['+str(nk)+'], '+str(nkernel_args)+', sizeof(cl_int), (void*) &y_size ));')
      nkernel_args = nkernel_args+1
    if NDIM==3:
      code('clSafeCall( clSetKernelArg(OPS_instance::getOPSInstance()->opencl_instance->OPS_opencl_core.kernel['+str(nk)+'], '+str(nkernel_args)+', sizeof(cl_int), (void*) &y_size ));')
      nkernel_args = nkernel_args+1
      code('clSafeCall( clSetKernelArg(OPS_instance::getOPSInstance()->opencl_instance->OPS_opencl_core.kernel['+str(nk)+'], '+str(nkernel_args)+', sizeof(cl_int), (void*) &z_size ));')
      nkernel_args = nkernel_args+1

    #kernel call
    code('')
    comm('call/enque opencl kernel wrapper function')
    code('clSafeCall( clEnqueueNDRangeKernel(OPS_instance::getOPSInstance()->opencl_instance->OPS_opencl_core.command_queue, OPS_instance::getOPSInstance()->opencl_instance->OPS_opencl_core.kernel['+str(nk)+'], 3, NULL, globalWorkSize, localWorkSize, 0, NULL, NULL) );')
    ENDIF()
    IF('OPS_instance::getOPSInstance()->OPS_diags>1')
    code('clSafeCall( clFinish(OPS_instance::getOPSInstance()->opencl_instance->OPS_opencl_core.command_queue) );')
    ENDIF()
    code('')

    IF('OPS_instance::getOPSInstance()->OPS_diags > 1')
    code('ops_timers_core(&c1,&t1);')
    code('OPS_instance::getOPSInstance()->OPS_kernels['+str(nk)+'].time += t1-t2;')
    ENDIF()
    code('')

    if GBL_INC == True or GBL_MIN == True or GBL_MAX == True or GBL_WRITE == True:
      code('mvReductArraysToHost(reduct_bytes);')

    for n in range (0, nargs):
      if arg_typ[n] == 'ops_arg_gbl' and accs[n] != OPS_READ:
        FOR('b','0','maxblocks')
        FOR('d','0',str(dims[n]))
        if accs[n] == OPS_INC:
          code('arg'+str(n)+'h[d] = arg'+str(n)+'h[d] + (('+typs[n]+' *)arg'+str(n)+'.data)[d+b*'+str(dims[n])+'];')
        elif accs[n] == OPS_MAX:
          code('arg'+str(n)+'h[d] = MAX(arg'+str(n)+'h[d],(('+typs[n]+' *)arg'+str(n)+'.data)[d+b*'+str(dims[n])+']);')
        elif accs[n] == OPS_MIN:
          code('arg'+str(n)+'h[d] = MIN(arg'+str(n)+'h[d],(('+typs[n]+' *)arg'+str(n)+'.data)[d+b*'+str(dims[n])+']);')
        ENDFOR()
        ENDFOR()
        code('arg'+str(n)+'.data = (char *)arg'+str(n)+'h;')
        code('')

    code('ops_set_dirtybit_device(args, '+str(nargs)+');')
    for n in range (0, nargs):
      if arg_typ[n] == 'ops_arg_dat' and (accs[n] == OPS_WRITE or accs[n] == OPS_RW or accs[n] == OPS_INC):
        code('ops_set_halo_dirtybit3(&args['+str(n)+'],range);')

    code('')
    IF('OPS_instance::getOPSInstance()->OPS_diags > 1')
    comm('Update kernel record')
    code('ops_timers_core(&c2,&t2);')
    code('OPS_instance::getOPSInstance()->OPS_kernels['+str(nk)+'].mpi_time += t2-t1;')
    for n in range (0, nargs):
      if arg_typ[n] == 'ops_arg_dat':
        code('OPS_instance::getOPSInstance()->OPS_kernels['+str(nk)+'].transfer += ops_compute_transfer(dim, start, end, &arg'+str(n)+');')
    ENDIF()
    config.depth = config.depth - 2
    code('}')

##########################################################################
#  output individual kernel file
##########################################################################
    if not os.path.exists('./OpenCL'):
      os.makedirs('./OpenCL')
    fid = open('./OpenCL/'+name+'_opencl_kernel.cpp','w')
    date = datetime.datetime.now()
    fid.write('//\n// auto-generated by ops.py\n//\n')
    fid.write(config.file_text)
    fid.close()

# end of main kernel call loop

##########################################################################
#  output one master kernel file
##########################################################################
  config.depth = 0
  config.file_text =''
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
  code('#include "ops_lib_cpp.h"')
  code('#include "ops_opencl_rt_support.h"')
  if os.path.exists('./user_types.h'):
    code('#include "user_types.h"')
  code('#ifdef OPS_MPI')
  code('#include "ops_mpi_core.h"')
  code('#endif')

  comm('global constants')
  for nc in range (0,len(consts)):
    if consts[nc]['dim'].isdigit() and int(consts[nc]['dim'])==1:
      code('extern '+consts[nc]['type']+' '+(str(consts[nc]['name']).replace('"','')).strip()+';')
    else:
      if consts[nc]['dim'].isdigit() and int(consts[nc]['dim']) > 0:
        num = consts[nc]['dim']
        code('extern '+consts[nc]['type']+' '+(str(consts[nc]['name']).replace('"','')).strip()+'['+num+'];')
      else:
        code('extern '+consts[nc]['type']+' *'+(str(consts[nc]['name']).replace('"','')).strip()+';')

  code('')

  code('')
  code('void ops_init_backend() {}')
  code('')
  comm('this needs to be a platform specific copy symbol to device function')
  code('void ops_decl_const_char( int dim, char const * type, int typeSize, char * dat, char const * name ) {')
  config.depth =config.depth + 2
  code('cl_int ret = 0;')
  IF('OPS_instance::getOPSInstance()->opencl_instance->OPS_opencl_core.constant == NULL')
  code('OPS_instance::getOPSInstance()->opencl_instance->OPS_opencl_core.constant = (cl_mem*) malloc(('+str(len(consts))+')*sizeof(cl_mem));')
  FOR('i','0',str(len(consts)))
  code('OPS_instance::getOPSInstance()->opencl_instance->OPS_opencl_core.constant[i] = NULL;')
  ENDFOR()
  ENDIF()

  for nc in range(0,len(consts)):
    IF('!strcmp(name,"'+(str(consts[nc]['name']).replace('"','')).strip()+'")')
    IF('OPS_instance::getOPSInstance()->opencl_instance->OPS_opencl_core.constant['+str(nc)+'] == NULL')
    code('OPS_instance::getOPSInstance()->opencl_instance->OPS_opencl_core.constant['+str(nc)+'] = clCreateBuffer(OPS_instance::getOPSInstance()->opencl_instance->OPS_opencl_core.context, CL_MEM_READ_ONLY, dim*typeSize, NULL, &ret);')
    code('clSafeCall( ret );')
    ENDIF()
    comm('Write the new constant to the memory of the device')
    code('clSafeCall( clEnqueueWriteBuffer(OPS_instance::getOPSInstance()->opencl_instance->OPS_opencl_core.command_queue, OPS_instance::getOPSInstance()->opencl_instance->OPS_opencl_core.constant['+str(nc)+'], CL_TRUE, 0, dim*typeSize, (void*) dat, 0, NULL, NULL) );')
    code('clSafeCall( clFlush(OPS_instance::getOPSInstance()->opencl_instance->OPS_opencl_core.command_queue) );')
    code('clSafeCall( clFinish(OPS_instance::getOPSInstance()->opencl_instance->OPS_opencl_core.command_queue) );')
    ENDIF()
    code('else')

  code('{')
  config.depth = config.depth + 2
  code('printf("error: unknown const name\\n"); exit(1);')
  ENDIF()
  config.depth = config.depth - 2
  code('}')






  kernel_name_list = []
  kernel_list_text = ''
  kernel_list__build_text = ''
  indent = 10*' '
  for nk in range(0,len(kernels)):
    #if kernels[nk]['name'] not in kernel_name_list : -- is this nessasary ??
    kernel_name_list.append(kernels[nk]['name'])
    if not (('initialise' in kernels[nk]['name']) or ('generate' in kernels[nk]['name'])):
      kernel_list_text = kernel_list_text + '"./OpenCL/'+kernel_name_list[nk]+'.cl"'
      if nk != len(kernels)-1:
        kernel_list_text = kernel_list_text+',\n'+indent
      kernel_list__build_text = kernel_list__build_text + \
      'OPS_instance::getOPSInstance()->opencl_instance->OPS_opencl_core.kernel['+str(nk)+'] = clCreateKernel(OPS_instance::getOPSInstance()->opencl_instance->OPS_opencl_core.program, "ops_'+kernel_name_list[nk]+'", &ret);\n      '+\
      'clSafeCall( ret );\n      '


  opencl_build = """


void buildOpenCLKernels() {
  static bool isbuilt = false;

  if(!isbuilt) {
    //clSafeCall( clUnloadCompiler() );

    OPS_instance::getOPSInstance()->opencl_instance->OPS_opencl_core.n_kernels = """+str(len(kernels))+""";
    OPS_instance::getOPSInstance()->opencl_instance->OPS_opencl_core.kernel = (cl_kernel*) malloc("""+str(len(kernels))+"""*sizeof(cl_kernel));
  }
  isbuilt = true;
}
"""


  config.depth = -2
  code(opencl_build)




  comm('user kernel files')

  #create unique set of kernel names list
  unique = list(set(kernel_name_list))

  for nk in range(0,len(unique)):
    if not (('initialise' in unique[nk]) or ('generate' in unique[nk])):
      code('#include "'+unique[nk]+'_opencl_kernel.cpp"')


  fid = open('./OpenCL/'+master_basename[0]+'_opencl_kernels.cpp','w')
  fid.write('//\n// auto-generated by ops.py//\n\n')
  fid.write(config.file_text)
  fid.close()
