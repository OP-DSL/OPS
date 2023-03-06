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
## @brief OPS CUDA code generator
#
#  This routine is called by ops.py which parses the input files
#
#  It produces a file xxx_cuda_kernel.cpp for each kernel,
#  plus a master kernel file
#

"""
OPS CUDA code generator

This routine is called by ops.py which parses the input files

It produces a file xxx_cuda_kernel.cpp for each kernel,
plus a master kernel file

"""

import re
import os

import config
from config import OPS_READ, OPS_WRITE, OPS_RW, OPS_INC, OPS_MAX, OPS_MIN

import util
from util import complex_numbers_cuda, get_kernel_func_text
from util import comm, code, FOR, ENDFOR, IF, ENDIF


def ops_gen_mpi_cuda(master, consts, kernels, soa_set, hip=0):
    NDIM = 2  # the dimension of the application is hardcoded here .. need to get this dynamically

    src_dir = os.path.dirname(master) or "."
    master_basename = os.path.splitext(os.path.basename(master))

    if hip == 1:
        cuda = "hip"
        cutil = "hip"
        dir_name = "HIP"
        file_ext = "cpp"
    else:
        cuda = "cuda"
        cutil = "cutil"
        dir_name = "CUDA"
        file_ext = "cu"

    ##########################################################################
    #  create new kernel file
    ##########################################################################
    if not os.path.exists(f"./{dir_name}"):
        os.makedirs(f"./{dir_name}")

    for nk in range(0, len(kernels)):
        assert config.file_text == "" and config.depth == 0
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
            GBL_READ,
            GBL_READ_MDIM,
            has_reduction,
            arg_idx,
            needDimList,
        ) = util.create_kernel_info(kernels[nk])
        any_prolong = any(prolong)

        ##########################################################################
        # generate constants and MACROS
        ##########################################################################

        num_dims = max(1, NDIM - 1)
        if NDIM > 1 and soa_set:
            num_dims += 1
        code(f"__constant__ int dims_{name} [{nargs}][{num_dims}];")
        code(f"static int dims_{name}_h [{nargs}][{num_dims}] = {{{{0}}}};")
        code("")

        ##########################################################################
        #  generate header
        ##########################################################################

        comm("user function")
        code("__device__")
        text = get_kernel_func_text(name, src_dir, arg_typ)
        text = re.sub(f"void\\s+\\b{name}\\b", f"void {name}_gpu", text)
        code(complex_numbers_cuda(text))
        code("")
        code("")

        ##########################################################################
        #  generate cuda kernel wrapper function
        ##########################################################################

        code(f"__global__ void ops_{name}(")
        for n in range(0, nargs):
            if arg_typ[n] == "ops_arg_dat":
                code(f"{typs[n]}* __restrict arg{n},")
                if n in needDimList:
                    code(f"int arg{n}dim,")
            elif arg_typ[n] == "ops_arg_gbl":
                if accs[n] == OPS_READ:
                    if dims[n].isdigit() and int(dims[n]) == 1:
                        code(f"const {typs[n]} arg{n},")
                    else:
                        code(f"const {typs[n]}* __restrict arg{n},")
                        if n in needDimList:
                            code(f"int arg{n}dim,")
                else:
                    code(f"{typs[n]}* __restrict arg{n},")
            if restrict[n] or prolong[n]:
                if NDIM == 1:
                    code(f"int stride_{n}0,")
                if NDIM == 2:
                    code(f"int stride_{n}0, int stride_{n}1,")
                if NDIM == 3:
                    code(f"int stride_{n}0, int stride_{n}1, int stride_{n}2,")

            elif arg_typ[n] == "ops_arg_idx":
                if NDIM == 1:
                    code("int arg_idx0,")
                elif NDIM == 2:
                    code("int arg_idx0, int arg_idx1,")
                elif NDIM == 3:
                    code("int arg_idx0, int arg_idx1, int arg_idx2,")

        if any_prolong:
            if NDIM == 1:
                code("int global_idx0,")
            elif NDIM == 2:
                code("int global_idx0, int global_idx1,")
            elif NDIM == 3:
                code("int global_idx0, int global_idx1, int global_idx2,")
        if NDIM == 1:
            code("int size0 ){")
        elif NDIM == 2:
            code("int size0,")
            code("int size1 ){")
        elif NDIM == 3:
            code("int size0,")
            code("int size1,")
            code("int size2 ){")

        config.depth = config.depth + 2

        # local variable to hold reductions on GPU
        code("")
        for n in range(0, nargs):
            if arg_typ[n] == "ops_arg_gbl" and accs[n] != OPS_READ:
                code(f"{typs[n]} arg{n}_l[{dims[n]}];")

        # set local variables to 0 if OPS_INC, INF if OPS_MIN, -INF
        for n in range(0, nargs):
            if arg_typ[n] == "ops_arg_gbl" and accs[n] == OPS_INC:
                code(f"for (int d=0; d<{dims[n]}; d++) arg{n}_l[d] = ZERO_{typs[n]};")
            if arg_typ[n] == "ops_arg_gbl" and accs[n] == OPS_MIN:
                code(
                    f"for (int d=0; d<{dims[n]}; d++) arg{n}_l[d] = INFINITY_{typs[n]};"
                )
            if arg_typ[n] == "ops_arg_gbl" and accs[n] == OPS_MAX:
                code(
                    f"for (int d=0; d<{dims[n]}; d++) arg{n}_l[d] = -INFINITY_{typs[n]};"
                )

        code("")
        if NDIM == 3:
            code("int idx_z = blockDim.z * blockIdx.z + threadIdx.z;")
            code("int idx_y = blockDim.y * blockIdx.y + threadIdx.y;")
        if NDIM == 2:
            code("int idx_y = blockDim.y * blockIdx.y + threadIdx.y;")
        code("int idx_x = blockDim.x * blockIdx.x + threadIdx.x;")
        code("")
        if arg_idx != -1:
            code(f"int arg_idx[{NDIM}];")
            code("arg_idx[0] = arg_idx0+idx_x;")
            if NDIM == 2:
                code("arg_idx[1] = arg_idx1+idx_y;")
            if NDIM == 3:
                code("arg_idx[1] = arg_idx1+idx_y;")
                code("arg_idx[2] = arg_idx2+idx_z;")

        for n in range(0, nargs):
            if arg_typ[n] == "ops_arg_dat":
                if restrict[n] == 1:
                    n_x = f"idx_x*stride_{n}0"
                    n_y = f"idx_y*stride_{n}1"
                    n_z = f"idx_z*stride_{n}2"
                elif prolong[n] == 1:
                    n_x = f"(idx_x+global_idx0%stride_{n}0)/stride_{n}0"
                    n_y = f"(idx_y+global_idx1%stride_{n}1)/stride_{n}1"
                    n_z = f"(idx_z+global_idx2%stride_{n}2)/stride_{n}2"
                else:
                    n_x = "idx_x"
                    n_y = "idx_y"
                    n_z = "idx_z"

                argdim = str(dims[n])
                if n in needDimList:
                    argdim = f"arg{n}dim"
                if NDIM == 1:
                    if soa_set:
                        code(f"arg{n} += {n_x} * {stride[n][0]};")
                    else:
                        code(f"arg{n} += {n_x} * {stride[n][0]}*{argdim};")
                elif NDIM == 2:
                    if soa_set:
                        code(
                            f"arg{n} += {n_x} * {stride[n][0]} + {n_y} * {stride[n][1]} * dims_{name}[{n}][0]"
                            + ";"
                        )
                    else:
                        code(
                            f"arg{n} += {n_x} * {stride[n][0]}*{argdim} + {n_y} * {stride[n][1]}*{argdim} * dims_{name}[{n}][0]"
                            + ";"
                        )
                elif NDIM == 3:
                    if soa_set:
                        code(
                            f"arg{n} += {n_x} * {stride[n][0]}+ {n_y} * {stride[n][1]}* dims_{name}[{n}][0] + {n_z} * {stride[n][2]} * dims_{name}[{n}][0] * dims_{name}[{n}][1];"
                        )
                    else:
                        code(
                            f"arg{n} += {n_x} * {stride[n][0]}*{argdim} + {n_y} * {stride[n][1]}*{argdim} * dims_{name}[{n}][0] + {n_z} * {stride[n][2]}*{argdim} * dims_{name}[{n}][0] * dims_{name}[{n}][1];"
                        )

        code("")
        if NDIM == 1:
            IF("idx_x < size0")
        if NDIM == 2:
            IF("idx_x < size0 && idx_y < size1")
        elif NDIM == 3:
            IF("idx_x < size0 && idx_y < size1 && idx_z < size2")
        for n in range(0, nargs):
            if arg_typ[n] == "ops_arg_dat":
                dim = ""
                sizelist = ""
                pre = ""
                extradim = 0
                if dims[n].isdigit() and int(dims[n]) > 1:
                    dim = dims[n] + ", "
                    extradim = 1
                elif not dims[n].isdigit():
                    dim = f"arg{n}dim, "
                    extradim = 1
                for i in range(1, NDIM):
                    sizelist += f"dims_{name}[{n}][{i-1}], "
                if extradim:
                    if soa_set:
                        sizelist += f"dims_{name}[{n}][{NDIM-1}], "
                    else:
                        sizelist += "0, "

                if accs[n] == OPS_READ:
                    pre = "const "

                code(f"{pre}ACC<{typs[n]}> argp{n}({dim+sizelist}arg{n});")
        code(f"{name}_gpu(")
        param_strings = []
        for n in range(0, nargs):
            if arg_typ[n] == "ops_arg_dat":
                param_strings.append(f" argp{n}")
            elif arg_typ[n] == "ops_arg_gbl" and accs[n] == OPS_READ:
                if dims[n].isdigit() and int(dims[n]) == 1:
                    param_strings.append(f" &arg{n}")
                else:
                    param_strings.append(f" arg{n}")
            elif arg_typ[n] == "ops_arg_gbl" and accs[n] != OPS_READ:
                param_strings.append(f" arg{n}_l")
            elif arg_typ[n] == "ops_arg_idx":
                param_strings.append(" arg_idx")
        code(
            util.group_n_per_line(
                param_strings, n_per_line=5, group_sep="\n" + " " * config.depth
            )
            + ");"
        )
        ENDIF()

        # reduction across blocks
        cont = "(blockIdx.x + blockIdx.y*gridDim.x)*"
        if NDIM == 2:
            cont = "(blockIdx.x + blockIdx.y*gridDim.x)*"
        elif NDIM == 3:
            cont = (
                "(blockIdx.x + blockIdx.y*gridDim.x + blockIdx.z*gridDim.x*gridDim.y)*"
            )
        for n in range(0, nargs):
            if arg_typ[n] == "ops_arg_gbl" and accs[n] == OPS_INC:
                code(f"for (int d=0; d<{dims[n]}; d++)")
                code(
                    f"  ops_reduction_{cuda}<OPS_INC>(&arg{n}[d+{cont}{dims[n]}],arg{n}_l[d]);"
                )
            if arg_typ[n] == "ops_arg_gbl" and accs[n] == OPS_MIN:
                code(f"for (int d=0; d<{dims[n]}; d++)")
                code(
                    f"  ops_reduction_{cuda}<OPS_MIN>(&arg{n}[d+{cont}{dims[n]}],arg{n}_l[d]);"
                )
            if arg_typ[n] == "ops_arg_gbl" and accs[n] == OPS_MAX:
                code(f"for (int d=0; d<{dims[n]}; d++)")
                code(
                    f"  ops_reduction_{cuda}<OPS_MAX>(&arg{n}[d+{cont}{dims[n]}],arg{n}_l[d]);"
                )

        code("")
        config.depth = config.depth - 2
        code("}")

        ##########################################################################
        #  now host stub
        ##########################################################################
        code("")
        comm(" host stub function")
        code("#ifndef OPS_LAZY")
        code(
            f"void ops_par_loop_{name}(char const *name, ops_block block, int dim, int* range,"
        )
        code(util.group_n_per_line([f" ops_arg arg{n}" for n in range(nargs)]) + ") {")
        code("#else")
        code(f"void ops_par_loop_{name}_execute(ops_kernel_descriptor *desc) {{")
        config.depth = 2
        code("int dim = desc->dim;")
        code("#if OPS_MPI")
        code("ops_block block = desc->block;")
        code("#endif")
        code("int *range = desc->range;")

        for n in range(0, nargs):
            code(f"ops_arg arg{n} = desc->args[{n}];")
        code("#endif")

        code("")
        comm("Timing")
        code("double t1,t2,c1,c2;")
        code("")

        code(
            f"ops_arg args[{nargs}] = {{"
            + ",".join([f" arg{n}" for n in range(nargs)])
            + "};\n"
        )
        code("")
        code("#if CHECKPOINTING && !OPS_LAZY")
        code(f"if (!ops_checkpointing_before(args,{nargs},range,{nk})) return;")
        code("#endif")
        code("")

        IF("block->instance->OPS_diags > 1")
        code(f'ops_timing_realloc(block->instance,{nk},"{name}");')
        code(f"block->instance->OPS_kernels[{nk}].count++;")
        code("ops_timers_core(&c1,&t1);")
        ENDIF()

        code("")
        comm("compute locally allocated range for the sub-block")

        code(f"int start[{NDIM}];")
        code(f"int end[{NDIM}];")

        code("")
        if arg_idx == -1:
            code("#ifdef OPS_MPI")
        code(f"int arg_idx[{NDIM}];")
        if arg_idx == -1:
            code("#endif")

        code("#if defined(OPS_LAZY) || !defined(OPS_MPI)")
        FOR("n", "0", str(NDIM))
        code("start[n] = range[2*n];end[n] = range[2*n+1];")
        ENDFOR()
        code("#else")
        code(
            f"if (compute_ranges(args, {nargs},block, range, start, end, arg_idx) < 0) return;"
        )
        code("#endif")

        code("")
        if arg_idx != -1 or MULTI_GRID:
            code("#if defined(OPS_MPI)")
            code("#if defined(OPS_LAZY)")
            code("sub_block_list sb = OPS_sub_block_list[block->index];")
            for n in range(0, NDIM):
                code(f"arg_idx[{n}] = sb->decomp_disp[{n}]+start[{n}];")
            code("#endif")
            code("#else //OPS_MPI")
            for n in range(0, NDIM):
                code(f"arg_idx[{n}] = start[{n}];")
            code("#endif //OPS_MPI")

        if MULTI_GRID:
            code(f"int global_idx[{NDIM}];")
            code("#ifdef OPS_MPI")
            for n in range(0, NDIM):
                code(f"global_idx[{n}] = arg_idx[{n}];")
            code("#else")
            for n in range(0, NDIM):
                code(f"global_idx[{n}] = start[{n}];")
            code("#endif")
            code("")

        for n in range(0, nargs):
            if arg_typ[n] == "ops_arg_dat":
                code(f"int xdim{n} = args[{n}].dat->size[0];")
                if NDIM > 2 or (NDIM == 2 and soa_set):
                    code(f"int ydim{n} = args[{n}].dat->size[1];")
                if NDIM > 3 or (NDIM == 3 and soa_set):
                    code(f"int zdim{n} = args[{n}].dat->size[2];")
        code("")

        condition = ""
        for n in range(0, nargs):
            if arg_typ[n] == "ops_arg_dat":
                condition += f"xdim{n} != dims_{name}_h[{n}][0] || "
                if NDIM > 2 or (NDIM == 2 and soa_set):
                    condition += f"ydim{n} != dims_{name}_h[{n}][1] || "
                if NDIM > 3 or (NDIM == 3 and soa_set):
                    condition += f"zdim{n} != dims_{name}_h[{n}][2] || "
        condition = condition[:-4]
        IF(condition)

        for n in range(0, nargs):
            if arg_typ[n] == "ops_arg_dat":
                code(f"dims_{name}_h[{n}][0] = xdim{n};")
                if NDIM > 2 or (NDIM == 2 and soa_set):
                    code(f"dims_{name}_h[{n}][1] = ydim{n};")
                if NDIM > 3 or (NDIM == 3 and soa_set):
                    code(f"dims_{name}_h[{n}][2] = zdim{n};")
        code(
            f"{cutil}SafeCall(block->instance->ostream(), {cuda}MemcpyToSymbol( dims_{name}, dims_{name}_h, sizeof(dims_{name})));"
        )
        ENDIF()

        code("")

        # setup reduction variables
        code("")
        if has_reduction and arg_idx == -1:
            code("#if defined(OPS_LAZY) && !defined(OPS_MPI)")
            code("ops_block block = desc->block;")
            code("#endif")
        for n in range(0, nargs):
            if arg_typ[n] == "ops_arg_gbl" and (
                accs[n] != OPS_READ
                or (accs[n] == OPS_READ and (not dims[n].isdigit() or int(dims[n]) > 1))
            ):
                if accs[n] == OPS_READ:
                    code(f"{typs[n]} *arg{n}h = ({typs[n]} *)arg{n}.data;")
                else:
                    code("#ifdef OPS_MPI")
                    code(
                        f"{typs[n]} *arg{n}h = ({typs[n]} *)(((ops_reduction)args[{n}].data)->data + ((ops_reduction)args[{n}].data)->size * block->index);"
                    )
                    code("#else")
                    code(
                        f"{typs[n]} *arg{n}h = ({typs[n]} *)(((ops_reduction)args[{n}].data)->data);"
                    )
                    code("#endif")

        code("")
        code("int x_size = MAX(0,end[0]-start[0]);")
        if NDIM == 2:
            code("int y_size = MAX(0,end[1]-start[1]);")
        if NDIM == 3:
            code("int y_size = MAX(0,end[1]-start[1]);")
            code("int z_size = MAX(0,end[2]-start[2]);")
        code("")

        # set up CUDA grid and thread blocks for kernel call
        if NDIM == 1:
            code("dim3 grid( (x_size-1)/block->instance->OPS_block_size_x+ 1, 1, 1);")
        if NDIM == 2:
            code(
                "dim3 grid( (x_size-1)/block->instance->OPS_block_size_x+ 1, (y_size-1)/block->instance->OPS_block_size_y + 1, 1);"
            )
        if NDIM == 3:
            code(
                "dim3 grid( (x_size-1)/block->instance->OPS_block_size_x+ 1, (y_size-1)/block->instance->OPS_block_size_y + 1, (z_size-1)/block->instance->OPS_block_size_z +1);"
            )

        if NDIM > 1:
            code(
                "dim3 tblock(block->instance->OPS_block_size_x,block->instance->OPS_block_size_y,block->instance->OPS_block_size_z);"
            )
        else:
            code("dim3 tblock(block->instance->OPS_block_size_x,1,1);")

        code("")

        if has_reduction:
            if NDIM == 1:
                code("int nblocks = ((x_size-1)/block->instance->OPS_block_size_x+ 1);")
            elif NDIM == 2:
                code(
                    "int nblocks = ((x_size-1)/block->instance->OPS_block_size_x+ 1)*((y_size-1)/block->instance->OPS_block_size_y + 1);"
                )
            elif NDIM == 3:
                code(
                    "int nblocks = ((x_size-1)/block->instance->OPS_block_size_x+ 1)*((y_size-1)/block->instance->OPS_block_size_y + 1)*((z_size-1)/block->instance->OPS_block_size_z +1);"
                )
            code("int maxblocks = nblocks;")
            code("int reduct_bytes = 0;")
            code("size_t reduct_size = 0;")
            code("")

        if GBL_READ == True and GBL_READ_MDIM == True:
            code("int consts_bytes = 0;")
            code("")

        for n in range(0, nargs):
            if arg_typ[n] == "ops_arg_gbl":
                if accs[n] == OPS_READ and (not dims[n].isdigit() or int(dims[n]) > 1):
                    code(f"consts_bytes += ROUND_UP(arg{n}.dim*sizeof({typs[n]}));")
                elif accs[n] != OPS_READ:
                    code(
                        f"reduct_bytes += ROUND_UP(maxblocks*{dims[n]}*sizeof({typs[n]}));"
                    )
                    code(f"reduct_size = MAX(reduct_size,sizeof({typs[n]})*{dims[n]});")
        code("")

        if GBL_READ == True and GBL_READ_MDIM == True:
            code("reallocConstArrays(block->instance,consts_bytes);")
            code("consts_bytes = 0;")
        if has_reduction:
            code("reallocReductArrays(block->instance,reduct_bytes);")
            code("reduct_bytes = 0;")
            code("")

        for n in range(0, nargs):
            if arg_typ[n] == "ops_arg_gbl" and accs[n] != OPS_READ:
                code(f"arg{n}.data = block->instance->OPS_reduct_h + reduct_bytes;")
                code(f"arg{n}.data_d = block->instance->OPS_reduct_d + reduct_bytes;")
                code("for (int b=0; b<maxblocks; b++)")
                if accs[n] == OPS_INC:
                    code(
                        f"for (int d=0; d<{dims[n]}; d++) (({typs[n]} *)arg{n}.data)[d+b*{dims[n]}] = ZERO_{typs[n]};"
                    )
                if accs[n] == OPS_MAX:
                    code(
                        f"for (int d=0; d<{dims[n]}; d++) (({typs[n]} *)arg{n}.data)[d+b*{dims[n]}] = -INFINITY_{typs[n]};"
                    )
                if accs[n] == OPS_MIN:
                    code(
                        f"for (int d=0; d<{dims[n]}; d++) (({typs[n]} *)arg{n}.data)[d+b*{dims[n]}] = INFINITY_{typs[n]};"
                    )
                code(
                    f"reduct_bytes += ROUND_UP(maxblocks*{dims[n]}*sizeof({typs[n]}));"
                )
                code("")

        code("")

        for n in range(0, nargs):
            if arg_typ[n] == "ops_arg_gbl":
                if accs[n] == OPS_READ and (not dims[n].isdigit() or int(dims[n]) > 1):
                    code(f"arg{n}.data = block->instance->OPS_consts_h + consts_bytes;")
                    code(
                        f"arg{n}.data_d = block->instance->OPS_consts_d + consts_bytes;"
                    )
                    code(
                        f"for (int d=0; d<arg{n}.dim; d++) (({typs[n]} *)arg{n}.data)[d] = arg{n}h[d];"
                    )
                    code(f"consts_bytes += ROUND_UP(arg{n}.dim*sizeof({typs[n]}));")
        if GBL_READ == True and GBL_READ_MDIM == True:
            code("mvConstArraysToDevice(block->instance,consts_bytes);")

        if has_reduction:
            code("mvReductArraysToDevice(block->instance,reduct_bytes);")

        for n in range(0, nargs):
            if arg_typ[n] == "ops_arg_dat":
                code(
                    f"long long int dat{n} = (block->instance->OPS_soa ? args[{n}].dat->type_size : args[{n}].dat->elem_size);"
                )

        code("")
        code(f"char *p_a[{nargs}];")

        # some custom logic for multigrid
        if MULTI_GRID:
            for n in range(0, nargs):
                if prolong[n] == 1 or restrict[n] == 1:
                    comm("This arg has a prolong stencil - so create different ranges")
                    code(
                        f"int start_{n}[{NDIM}]; int end_{n}[{NDIM}]; int stride_{n}[{NDIM}];int d_size_{n}[{NDIM}];"
                    )
                    code("#ifdef OPS_MPI")
                    FOR("n", "0", str(NDIM))
                    code(f"sub_dat *sd{n} = OPS_sub_dat_list[args[{n}].dat->index];")
                    code(f"stride_{n}[n] = args[{n}].stencil->mgrid_stride[n];")
                    code(
                        f"d_size_{n}[n] = args[{n}].dat->d_m[n] + sd{n}->decomp_size[n] - args[{n}].dat->d_p[n];"
                    )
                    if restrict[n] == 1:
                        code(
                            f"start_{n}[n] = global_idx[n]*stride_{n}[n] - sd{n}->decomp_disp[n] + args[{n}].dat->d_m[n];"
                        )
                    else:
                        code(
                            f"start_{n}[n] = global_idx[n]/stride_{n}[n] - sd{n}->decomp_disp[n] + args[{n}].dat->d_m[n];"
                        )
                    code(f"end_{n}[n] = start_{n}[n] + d_size_{n}[n];")
                    ENDFOR()
                    code("#else")
                    FOR("n", "0", str(NDIM))
                    code(f"stride_{n}[n] = args[{n}].stencil->mgrid_stride[n];")
                    code(
                        f"d_size_{n}[n] = args[{n}].dat->d_m[n] + args[{n}].dat->size[n] - args[{n}].dat->d_p[n];"
                    )
                    if restrict[n] == 1:
                        code(f"start_{n}[n] = global_idx[n]*stride_{n}[n];")
                    else:
                        code(f"start_{n}[n] = global_idx[n]/stride_{n}[n];")
                    code(f"end_{n}[n] = start_{n}[n] + d_size_{n}[n];")
                    ENDFOR()
                    code("#endif")

        comm("")
        comm("set up initial pointers")
        for n in range(0, nargs):
            if arg_typ[n] == "ops_arg_dat":
                if prolong[n] == 1 or restrict[n] == 1:
                    starttext = "start_" + str(n)
                else:
                    starttext = "start"
                code(f"long long int base{n} = args[{n}].dat->base_offset + ")
                code(
                    f"         dat{n} * 1 * ({starttext}[0] * args[{n}].stencil->stride[0]);"
                )
                for d in range(1, NDIM):
                    line = f"base{n} = base{n}+ dat{n} *\n"
                    for d2 in range(0, d):
                        line += config.depth * " " + f"  args[{n}].dat->size[{d2}] *\n"
                    code(line[:-1])
                    code(f"  ({starttext}[{d}] * args[{n}].stencil->stride[{d}]);")

                code(f"p_a[{n}] = (char *)args[{n}].data_d + base{n};")
                code("")

        # halo exchange
        code("")
        code("#ifndef OPS_LAZY")
        code(f"ops_H_D_exchanges_device(args, {nargs});")
        code(f"ops_halo_exchanges(args,{nargs},range);")
        code("#endif")
        code("")
        IF("block->instance->OPS_diags > 1")
        code("ops_timers_core(&c2,&t2);")
        code(f"block->instance->OPS_kernels[{nk}].mpi_time += t2-t1;")
        ENDIF()
        code("")

        # set up shared memory for reduction
        if has_reduction:
            code("size_t nshared = 0;")
            code(
                "int nthread = block->instance->OPS_block_size_x*block->instance->OPS_block_size_y*block->instance->OPS_block_size_z;"
            )
            code("")
        for n in range(0, nargs):
            if arg_typ[n] == "ops_arg_gbl" and accs[n] != OPS_READ:
                code(f"nshared = MAX(nshared,sizeof({typs[n]})*{dims[n]});")
        code("")
        if has_reduction:
            code("nshared = MAX(nshared*nthread,reduct_size*nthread);")
            code("")

        # kernel call
        comm("call kernel wrapper function, passing in pointers to data")
        if NDIM == 1:
            code("if (x_size > 0)")
        if NDIM == 2:
            code("if (x_size > 0 && y_size > 0)")
        if NDIM == 3:
            code("if (x_size > 0 && y_size > 0 && z_size > 0)")
        config.depth = config.depth + 2
        if has_reduction:
            code(f"ops_" + name + "<<<grid, tblock, nshared >>> ( ")
        else:
            code(f"ops_{name}<<<grid, tblock >>> ( ")
        param_strings = []
        for n in range(0, nargs):
            if arg_typ[n] == "ops_arg_dat":
                param_strings.append(f" ({typs[n]} *)p_a[{n}],")
                if n in needDimList:
                    param_strings.append(f" arg{n}.dim,")
            elif arg_typ[n] == "ops_arg_gbl":
                if dims[n].isdigit() and int(dims[n]) == 1 and accs[n] == OPS_READ:
                    param_strings.append(f" *({typs[n]} *)arg{n}.data,")
                else:
                    param_strings.append(f" ({typs[n]} *)arg{n}.data_d,")
                    if n in needDimList and accs[n] != OPS_READ:
                        param_strings.append(" arg{n}.dim,")
            elif arg_typ[n] == "ops_arg_idx":
                if NDIM == 1:
                    param_strings.append(" arg_idx[0],")
                if NDIM == 2:
                    param_strings.append(" arg_idx[0], arg_idx[1],")
                elif NDIM == 3:
                    param_strings.append(" arg_idx[0], arg_idx[1], arg_idx[2],")
            if restrict[n] or prolong[n]:
                if NDIM == 1:
                    param_strings.append(f"stride_{n}[0],")
                if NDIM == 2:
                    param_strings.append(f"stride_{n}[0],stride_{n}[1],")
                if NDIM == 3:
                    param_strings.append(f"stride_{n}[0],stride_{n}[1],stride_{n}[2],")
        code(
            util.group_n_per_line(
                param_strings, n_per_line=2, sep="", group_sep="\n" + " " * config.depth
            )
        )
        if any_prolong:
            if NDIM == 1:
                code("global_idx[0],")
            elif NDIM == 2:
                code("global_idx[0], global_idx[1],")
            elif NDIM == 3:
                code("global_idx[0], global_idx[1], global_idx[2],")

        if NDIM == 1:
            code("x_size);")
        if NDIM == 2:
            code("x_size, y_size);")
        elif NDIM == 3:
            code("x_size, y_size, z_size);")
        config.depth = config.depth - 2

        code("")
        code(f"{cutil}SafeCall(block->instance->ostream(), {cuda}GetLastError());")
        code("")

        #
        # Complete Reduction Operation by moving data onto host
        # and reducing over blocks
        #
        if has_reduction:
            code("mvReductArraysToHost(block->instance,reduct_bytes);")

        for n in range(0, nargs):
            if arg_typ[n] == "ops_arg_gbl" and accs[n] != OPS_READ:
                FOR("b", "0", "maxblocks")
                FOR("d", "0", str(dims[n]))
                if accs[n] == OPS_INC:
                    code(
                        f"arg{n}h[d] = arg{n}h[d] + (({typs[n]} *)arg{n}.data)[d+b*{dims[n]}];"
                    )
                elif accs[n] == OPS_MAX:
                    code(
                        f"arg{n}h[d] = MAX(arg{n}h[d],(({typs[n]} *)arg{n}.data)[d+b*{dims[n]}]);"
                    )
                elif accs[n] == OPS_MIN:
                    code(
                        f"arg{n}h[d] = MIN(arg{n}h[d],(({typs[n]} *)arg{n}.data)[d+b*{dims[n]}]);"
                    )
                ENDFOR()
                ENDFOR()
                code(f"arg{n}.data = (char *)arg{n}h;")
                code("")

        IF("block->instance->OPS_diags>1")
        code(f"{cutil}SafeCall(block->instance->ostream(), {cuda}DeviceSynchronize());")
        code("ops_timers_core(&c1,&t1);")
        code(f"block->instance->OPS_kernels[{nk}].time += t1-t2;")
        ENDIF()
        code("")

        code("#ifndef OPS_LAZY")
        code(f"ops_set_dirtybit_device(args, {nargs});")
        for n in range(0, nargs):
            if arg_typ[n] == "ops_arg_dat" and (
                accs[n] == OPS_WRITE or accs[n] == OPS_RW or accs[n] == OPS_INC
            ):
                code(f"ops_set_halo_dirtybit3(&args[{n}],range);")
        code("#endif")

        code("")
        IF("block->instance->OPS_diags > 1")
        comm("Update kernel record")
        code("ops_timers_core(&c2,&t2);")
        code(f"block->instance->OPS_kernels[{nk}].mpi_time += t2-t1;")
        for n in range(0, nargs):
            if arg_typ[n] == "ops_arg_dat":
                code(
                    f"block->instance->OPS_kernels[{nk}].transfer += ops_compute_transfer(dim, start, end, &arg{n});"
                )
        ENDIF()
        config.depth = config.depth - 2
        code("}")
        code("")
        code("#ifdef OPS_LAZY")
        code(
            f"void ops_par_loop_{name}(char const *name, ops_block block, int dim, int* range,"
        )
        code(util.group_n_per_line([f" ops_arg arg{n}" for n in range(nargs)]) + ") {")
        config.depth = 2
        code(
            "ops_kernel_descriptor *desc = (ops_kernel_descriptor *)calloc(1,sizeof(ops_kernel_descriptor));"
        )
        code("desc->name = name;")
        code("desc->block = block;")
        code("desc->dim = dim;")
        code("desc->device = 1;")
        code(f"desc->index = {nk};")
        code("desc->hash = 5381;")
        code(f"desc->hash = ((desc->hash << 5) + desc->hash) + {nk};")
        FOR("i", "0", str(2 * NDIM))
        code("desc->range[i] = range[i];")
        code("desc->orig_range[i] = range[i];")
        code("desc->hash = ((desc->hash << 5) + desc->hash) + range[i];")
        ENDFOR()

        code(f"desc->nargs = {nargs};")
        code(f"desc->args = (ops_arg*)ops_malloc({nargs}*sizeof(ops_arg));")
        declared = 0
        for n in range(0, nargs):
            code(f"desc->args[{n}] = arg{n};")
            if arg_typ[n] == "ops_arg_dat":
                code(
                    f"desc->hash = ((desc->hash << 5) + desc->hash) + arg{n}.dat->index;"
                )
            if arg_typ[n] == "ops_arg_gbl" and accs[n] == OPS_READ:
                if declared == 0:
                    code(
                        f"char *tmp = (char*)ops_malloc(arg{n}.dim*sizeof({typs[n]}));"
                    )
                    declared = 1
                else:
                    code(f"tmp = (char*)ops_malloc(arg{n}.dim*sizeof({typs[n]}));")
                code(f"memcpy(tmp, arg{n}.data,arg{n}.dim*sizeof({typs[n]}));")
                code(f"desc->args[{n}].data = tmp;")
        code(f"desc->function = ops_par_loop_{name}_execute;")
        IF("block->instance->OPS_diags > 1")
        code(f'ops_timing_realloc(block->instance,{nk},"{name}");')
        ENDIF()
        code("ops_enqueue_kernel(desc);")
        config.depth = 0
        code("}")
        code("#endif")

        ##########################################################################
        #  output individual kernel file
        ##########################################################################
        util.write_text_to_file(f"./{dir_name}/{name}_{cuda}_kernel.{file_ext}")

    # end of main kernel call loop

    ##########################################################################
    #  output one master kernel file
    ##########################################################################
    comm("header")
    code("#define OPS_API 2")
    code(f"#define OPS_{NDIM}D")
    if soa_set:
        code("#define OPS_SOA")
    code('#include "ops_lib_core.h"')
    code("")
    code(f'#include "ops_{cuda}_rt_support.h"')
    code(f'#include "ops_{cuda}_reduction.h"')
    code("")
    code(
        "#include <cuComplex.h>"
    )  # Include the CUDA complex numbers library, in case complex numbers are used anywhere.
    code("")
    if os.path.exists(os.path.join(src_dir, "user_types.h")):
        code("#define OPS_FUN_PREFIX __device__ __host__")
        code('#include "user_types.h"')
    code("#ifdef OPS_MPI")
    code('#include "ops_mpi_core.h"')
    code("#endif")

    util.generate_extern_global_consts_declarations(consts, for_cuda=True)

    code("")
    code("void ops_init_backend() {}")
    code("")
    code("void ops_decl_const_char(OPS_instance *instance, int dim, char const *type,")
    code("int size, char *dat, char const *name){")
    config.depth = config.depth + 2
    code("ops_execute(instance);")

    for nc in range(0, len(consts)):
        IF('!strcmp(name,"' + (str(consts[nc]["name"]).replace('"', "")).strip() + '")')
        if consts[nc]["dim"].isdigit():
            code(
                f"{cutil}SafeCall(instance->ostream(),{cuda}MemcpyToSymbol("
                + (str(consts[nc]["name"]).replace('"', "")).strip()
                + ", dat, dim*size));"
            )
        else:
            code(
                f"char *temp; {cutil}SafeCall(instance->ostream(),{cuda}Malloc((void**)&temp,dim*size));"
            )
            code(
                f"{cutil}SafeCall(instance->ostream(),{cuda}Memcpy(temp,dat,dim*size,{cuda}MemcpyHostToDevice));"
            )
            code(
                f"{cutil}SafeCall(instance->ostream(),{cuda}MemcpyToSymbol("
                + (str(consts[nc]["name"]).replace('"', "")).strip()
                + ", &temp, sizeof(char *)));"
            )
        ENDIF()
        code("else")

    code("{")
    config.depth = config.depth + 2
    code('throw OPSException(OPS_RUNTIME_ERROR, "error: unknown const name");')
    ENDIF()

    config.depth = config.depth - 2
    code("}")
    code("")

    code("")
    comm("user kernel files")

    for kernel_name in map(lambda kernel: kernel["name"], kernels):
        code(f'#include "{kernel_name}_{cuda}_kernel.{file_ext}"')

    util.write_text_to_file(f"./{dir_name}/{master_basename[0]}_kernels.{file_ext}")
