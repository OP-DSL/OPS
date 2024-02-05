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
## @brief TODO
#  This routine is called by ops.py which parses the input files
#
#  It produces a file TODO
#  plus a master TODO
#

"""
OPS MPI_seq mixed precision code generator

This routine is called by ops.py which parses the input files

It produces a file TODO
plus a master TODO

"""



import config
from config import OPS_READ, OPS_WRITE, OPS_RW, OPS_INC, OPS_MAX, OPS_MIN

import util
from util import comm, code, FOR, ENDFOR, IF, ENDIF, ELSE

from ops_gen_mpi_lazy import ops_gen_mpi_lazy

import os, re

def ops_gen_mixedprec(master, consts, kernels, soa_set, offload=0):
    
    #list of global constants names to modify
    global_constants = [const['name'].replace('"', '') for const in consts if const['type'] in ['float', 'double']]
    
    #create output directory if it does not exist
    output_dir = 'mixed_kernels'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    user_kernels=[]

    for nk in range(0,len(kernels)):
        (
            arg_typ,
            name,
            nargs,
            dims,
            accs,
            typs,
            NDIM,
            stride,
            restrict,
            prolong,
            MULTI_GRID,
            _,
            _,
            has_reduction,
            arg_idx,
            _,
        ) = util.create_kernel_info(kernels[nk])

        user_kernels.append('{}/{}_mixed.h'.format(output_dir,name))
        #TODO also need to do with reduction
        generate_mixed_prec_kernel(master,name,global_constants,output_dir,arg_typ)


    suffix='_float'
    #gen_seq -- float
    ops_gen_mpi_lazy(master, consts, kernels, soa_set, offload, suffix, './{}'.format(output_dir))
    #gen_float_consts
        
    #for all kernels, replace floats to doubles
        
    suffix='_double'
    for i in range(len(user_kernels)):
      user_kernels[i] = user_kernels[i].replace('float', 'double')

    #for all kernels, replace doubles to floats
    for i in range(len(kernels)):
        kernels[i]['typs'] = [typ.replace('float', 'double') for typ in  kernels[i]['typs']]
        

    #gen_seq -- double
    ops_gen_mpi_lazy(master, consts, kernels, soa_set, offload, suffix, './{}'.format(output_dir))
      
    #genereate umbrella kernel to switch between float and double
    generate_umbrella_kernels(master,kernels,output_dir,offload)

    #generate master kernel files
    generate_master_kernels(master,consts,kernels,output_dir,soa_set,offload)


def generate_consts(consts):

    code("")
    for nc in range(0, len(consts)):
        if consts[nc]['type'] in ['float', 'double']:
            name = (str(consts[nc]["name"]).replace('"', "")).strip()
            code('float ' + name + '_f;')
            code('double ' + name + '_d;')


    code("")
    code("void ops_decl_const_char(OPS_instance *instance, int dim, char const *type,")
    code("int size, char *dat, char const *name){")
    config.depth = config.depth + 2
    code("ops_execute(instance);")

    for nc in range(0, len(consts)):
        if consts[nc]['type'] in ['float', 'double']:
            name = (str(consts[nc]["name"]).replace('"', "")).strip()
            print('generating const: ', name)
            IF('!strcmp(name,"' + name + '")')
        
            code(name+'_f='+name+';')
            code(name+'_d=(double)'+name+';')
            ENDIF()
            code("else")
        else:
            name = (str(consts[nc]["name"]).replace('"', "")).strip()
            IF('!strcmp(name,"' + name + '")')
            comm("Not floating type, no need for conversion.")
            ENDIF()
            code("else")

    code("{")
    config.depth = config.depth + 2
    code('throw OPSException(OPS_RUNTIME_ERROR, "error: unknown const name");')
    ENDIF()

    config.depth = config.depth - 2
    code("}")
    code("")


    

def generate_master_kernels(master,consts,kernels,output_dir,soa_set,offload=0):
    generate_master_kernels_mixd(master,kernels,"_float",offload)
    generate_master_kernels_mixd(master,kernels,"_double",offload)
    generate_main_master_kernel(master,consts,kernels,output_dir,soa_set,offload)
    
def generate_main_master_kernel(master,consts,kernels,output_dir,soa_set,offload=0):
    NDIM = 2
    NDIM=kernels[-1]['dim']
    src_dir = os.path.dirname(master) or "."
    master_basename = os.path.splitext(os.path.basename(master))

    comm("header")
    code(f"#define OPS_{NDIM}D")
    if soa_set:
        code("#define OPS_SOA")
    code("#define OPS_API 2")
    code('#include "ops_lib_core.h"')
    code("#ifdef OPS_MPI")
    code('#include "ops_mpi_core.h"')
    code("#endif")
    if os.path.exists(os.path.join(src_dir, "user_types.h")):
        code('#include "user_types.h"')
    code("")

    util.generate_extern_global_consts_declarations(consts)

    
    generate_consts(consts)

    code(f"#include \"{master_basename[0]}_float_{'cpu' if offload==0 else 'ompoffload'}_kernels.cpp\"")
    code(f"#include \"{master_basename[0]}_double_{'cpu' if offload==0 else 'ompoffload'}_kernels.cpp\"")


    code("")
    code("void ops_init_backend() {}")
    code("")
    

    if offload:
        code("void ops_decl_const_char(OPS_instance *instance, int dim, char const *type,")
        code("int size, char *dat, char const *name){")
        config.depth = config.depth + 2
        code("ops_execute(instance);")

        for nc in range(0, len(consts)):
            IF('!strcmp(name,"' + (str(consts[nc]["name"]).replace('"', "")).strip() + '")')
            if consts[nc]["dim"].isdigit() and int(consts[nc]["dim"]) == 1:
                code(f"#pragma omp target enter data map(to:{consts[nc]['name'][1:-1]})")
            else:
                code(f"#pragma omp target enter data map(to:{consts[nc]['name'][1:-1]}[0:dim])")
            ENDIF()
            code("else")

        code("{")
        config.depth = config.depth + 2
        code('throw OPSException(OPS_RUNTIME_ERROR, "error: unknown const name");')
        ENDIF()

        config.depth = config.depth - 2
        code("}")
        code("")
    comm("user kernel files")

    for kernel_name in map(lambda kernel: kernel["name"], kernels):
        code(f'#include "{kernel_name}_{"cpu" if offload==0 else "ompoffload"}_kernel.cpp"')

    util.write_text_to_file(f"./{'MPI_OpenMP' if offload==0 else 'OpenMP_offload'}/{master_basename[0]}_{'cpu' if offload==0 else 'ompoffload'}_kernels.cpp")

    pass

def generate_master_kernels_mixd(master,kernels,suffix,offload=0):

#    util.generate_extern_global_consts_declarations(consts)

    #for kernel_name in map(lambda kernel: kernel["name"], kernels):
    for nk in range(len(kernels)):
        kernel_name=kernels[nk]['name']
        #code(f'#include "{kernels[nk]['name']}_{"cpu" if offload==0 else "ompoffload"}_kernel.cpp"')
        code(f'#include "{kernel_name}{suffix}_{"cpu" if offload==0 else "ompoffload"}_kernel.cpp"')

    master_basename = os.path.splitext(os.path.basename(master))
    util.write_text_to_file(f"./{'MPI_OpenMP' if offload==0 else 'OpenMP_offload'}/{master_basename[0]}{suffix}_{'cpu' if offload==0 else 'ompoffload'}_kernels.cpp")


def generate_umbrella_kernels(master,kernels,output_dir,offload=0):
    for nk in range(0,len(kernels)):
        name=kernels[nk]['name']
        nargs=kernels[nk]['nargs']
        code(
            f"void ops_par_loop_{name}(char const *name, ops_block block, int dim, int* range,"
        )        
        code(util.group_n_per_line([f" ops_arg arg{n}" for n in range(nargs)]) + ") {")
        IF("OPS_instance::getOPSInstance()->OPS_precision==0")
        code(
            f"ops_par_loop_{name}_float(name,  block, dim, range,"
        )        
        code(util.group_n_per_line([f" arg{n}" for n in range(nargs)]) + ");")
        ENDIF()
        ELSE()
        code(
            f"ops_par_loop_{name}_double(name,  block, dim, range,"
        )        
        code(util.group_n_per_line([f" arg{n}" for n in range(nargs)]) + ");")
        ENDIF()

        code("}")

        if offload:
            util.write_text_to_file(f"./OpenMP_offload/{name}_ompoffload_kernel.cpp")
        else:
            util.write_text_to_file(f"./MPI_OpenMP/{name}_cpu_kernel.cpp")


            
    master_basename = os.path.splitext(os.path.basename(master))

    pass


def generate_mixed_prec_kernel(master,name,global_constants,output_dir,arg_typ):

  print('generating kernel: ',name)
    
  src_dir = os.path.dirname(master) or "."
  kernel_text = util.get_kernel_func_text(name, src_dir, arg_typ)

  with open('{}/{}/{}_mixed.h'.format(src_dir,output_dir,name), 'w') as f:
      for const in global_constants:
          kernel_text = kernel_text.replace(const, const+'_f')
      f.write(generate_float(kernel_text, name, global_constants))
      f.write('\n\n')
      
      for const in global_constants:
          kernel_text = kernel_text.replace(const+'_f', const+'_d')
      f.write(generate_double(kernel_text, name, global_constants))
  
  #with open('{}/{}_float.h'.format(output_dir,name), 'w') as f:
  #  f.write(generate_float(name, global_constants))
  #
  #with open('{}/{}_double.h'.format(output_dir,name), 'w') as f:
  #  f.write(generate_double(name, global_constants))

def generate_float(kernel_text,name, global_constants):
    # Read the contents of the input file into a variable
#    with open('{}.h'.format(name), 'r') as input_file:
#        input_contents = input_file.read()

    # Replace occurrences of the function name with the new name and type (float)    
    output_contents = kernel_text
    #output_contents = input_contents.replace('#ifndef op2_mf_{}'.format(name), '#ifndef op2_mf_{}_f'.format(name))
    #output_contents = output_contents.replace('#define op2_mf_{}'.format(name), '#define op2_mf_{}_f'.format(name))
    output_contents = output_contents.replace('void {}'.format(name), 'void {}_float'.format(name))

    # Rename global constants with '_f' suffix in the first copy
    #for var_name in global_constants:
    #    output_contents = re.sub(r'\b{}\b'.format(var_name), '{}_f'.format(var_name), output_contents)

    # Return the modified contents as float version
    return output_contents

def generate_double(kernel_text,name, global_constants):
    # Read the contents of the input file into a variable
    #with open('{}.h'.format(name), 'r') as input_file:
    #    input_contents = input_file.read()

    # Replace occurrences of the function name with the new name and type (double)
    output_contents = kernel_text
    #output_contents = input_contents.replace('#ifndef op2_mf_{}'.format(name), '#ifndef op2_mf_{}_d'.format(name))
    #output_contents = output_contents.replace('#define op2_mf_{}'.format(name), '#define op2_mf_{}_d'.format(name))
    output_contents = output_contents.replace('void {}'.format(name), 'void {}_double'.format(name))
    output_contents = output_contents.replace('float', 'double')

    # Rename global constants with '_d' suffix in the double copy
    #for var_name in global_constants:
    #    output_contents = re.sub(r'\b{}\b'.format(var_name), '{}_d'.format(var_name), output_contents)

    # Replace all literal constants with their double counterparts
    output_contents = re.sub(r'(?<!\w)(\d+(\.\d*)?|\.\d+)([eE][-+]?\d+)?[fF]?', r'\g<1>', output_contents)

    # Return the modified contents as double version
    return output_contents