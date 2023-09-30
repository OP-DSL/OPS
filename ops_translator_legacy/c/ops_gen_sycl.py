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
## @brief OPS SYCL code generator
#
#  This routine is called by ops.py which parses the input files
#
#  It produces a file xxx_sycl_kernel.cpp for each kernel,
#  plus a master kernel file
#
"""
OPS MPI_seq code generator

This routine is called by ops.py which parses the input files

It produces a file xxx_sycl_kernel.cpp for each kernel,
plus a master kernel file

"""

import re
import os

import util
import config
from config import OPS_READ, OPS_WRITE, OPS_RW, OPS_INC, OPS_MAX, OPS_MIN

para_parse = util.para_parse
comment_remover = util.comment_remover
remove_trailing_w_space = util.remove_trailing_w_space
parse_signature = util.parse_signature
check_accs = util.check_accs
mult = util.mult
convert_ACC_body = util.convert_ACC_body

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


def clean_type(arg):
    for qual in ["__restrict__", "RESTRICT", "__volatile__"]:
        arg = arg.replace(qual, "")
    return arg


def ops_gen_sycl(master, consts, kernels, soa_set):
    gen_oneapi = True
    sycl_guarded_namespace = "cl::sycl::"
    if gen_oneapi:
        sycl_guarded_namespace = "cl::sycl::"

    NDIM = 2  # the dimension of the application is hardcoded here .. need to get this dynamically

    src_dir = os.path.dirname(master) or "."
    master_basename = os.path.splitext(os.path.basename(master))

    ##########################################################################
    #  create new kernel file
    ##########################################################################
    if not os.path.exists("./sycl"):
        os.makedirs("./sycl")

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
            _,
        ) = util.create_kernel_info(kernels[nk])

        builtin_reduction = False
        red_arg_idxs = []
        for n in range(0, nargs):
            if arg_typ[n] == "ops_arg_gbl" and accs[n] != OPS_READ:
                red_arg_idxs.append(n)
                if not dims[n].isdigit():
                    builtin_reduction = False
        if os.getenv("OPS_FLAT"):
            flat_parallel = True
            ops_cpu = False
        else:
            flat_parallel = False
            ops_cpu = os.getenv("OPS_CPU")
            if ops_cpu and ops_cpu.isdigit():
                ops_cpu = int(ops_cpu)
            elif ops_cpu:
                ops_cpu = 1
        if (has_reduction and not builtin_reduction) or (gen_oneapi and has_reduction):
            # if flat_parallel and reduction: #and not builtin_reduction:
            flat_parallel = False
            ops_cpu = False

        ##########################################################################
        #  start with seq kernel function
        ##########################################################################

        kernel_text, arg_list = util.get_kernel_body_and_arg_list(
            name, src_dir, arg_typ
        )

        global_consts = []
        for c in consts:
            const = c["name"].replace('"', "")
            if re.search(r"\b" + const + r"\b", kernel_text):
                global_consts = global_consts + [
                    "auto "
                    + const
                    + "_sycl = (*"
                    + const
                    + "_p).template get_access<cl::sycl::access::mode::read>(cgh);"
                ]
                if c["dim"].isdigit() and int(c["dim"]) == 1:
                    kernel_text = re.sub(
                        r"\b" + const + r"\b", const + "_sycl[0]", kernel_text
                    )
                else:
                    kernel_text = re.sub(
                        r"\b" + const + r"\b", const + "_sycl", kernel_text
                    )

        kernel_text = re.sub(r"\bsqrt\b", "cl::sycl::sqrt", kernel_text)
        kernel_text = re.sub(r"\bcbrt\b", "cl::sycl::cbrt", kernel_text)
        kernel_text = re.sub(r"\bfabs\b", "cl::sycl::fabs", kernel_text)
        kernel_text = re.sub(r"\bfmin\b", "cl::sycl::fmin", kernel_text)
        kernel_text = re.sub(r"\bfmax\b", "cl::sycl::fmax", kernel_text)
        kernel_text = re.sub(r"\bisnan\b", "cl::sycl::isnan", kernel_text)
        kernel_text = re.sub(r"\bisinf\b", "cl::sycl::isinf", kernel_text)
        kernel_text = re.sub(r"\bsin\b", "cl::sycl::sin", kernel_text)
        kernel_text = re.sub(r"\bcos\b", "cl::sycl::cos", kernel_text)
        kernel_text = re.sub(r"\bexp\b", "cl::sycl::exp", kernel_text)

        comm("")
        comm("  ==================")
        comm("  Host stub function")
        comm("  ==================")
        code("#ifndef OPS_LAZY")
        code(f"void ops_par_loop_{name}(")
        code("    const char *name,")
        code("    ops_block block,")
        code("    int dim,")
        code("    int* range,")
        for n in range(nargs):
            if n == nargs-1:
                code(f"    ops_arg arg{n}")
            else:
                code(f"    ops_arg arg{n}, ")
        code(")\n{")
        code("#else")
        code(f"void ops_par_loop_{name}_execute(ops_kernel_descriptor *desc)")
        code("{")
        config.depth = 4
        code("ops_block block = desc->block;")
        code("int dim = desc->dim;")
        code("int *range = desc->range;")

        for n in range(0, nargs):
            code(f"ops_arg arg{n} = desc->args[{n}];")

        config.depth = 0
        code("#endif")
        code("")
        
        comm("  ======")
        comm("  Timing")
        comm("  ======")
        config.depth = 4
        code("double __t1,__t2,__c1,__c2;")
        code("")

        code(f"ops_arg args[{nargs}];")
        code("")

        for n in range(nargs):
            code(f"args[{n}] = arg{n};")

        code("")
        config.depth = 0
        code("#if defined(CHECKPOINTING) && !defined(OPS_LAZY)")
        config.depth = 4
        code(f"if (!ops_checkpointing_before(args,{nargs}, range, {nk})) return;")
        config.depth = 0
        code("#endif")
        code("")

        config.depth = 4
        IF("block->instance->OPS_diags > 1")
        code(f'ops_timing_realloc(block->instance, {nk}, "{name}");')
        code(f"block->instance->OPS_kernels[{nk}].count++;")
        code("ops_timers_core(&__c1, &__t1);")
        ENDIF()
        code("")

        config.depth = 0
        code("#ifdef OPS_DEBUG")
        config.depth = 4
        code(f'ops_register_args(block->instance, args, "{name}");')
        config.depth = 0
        code("#endif")

        code("")

        comm("  =================================================")
        comm("  compute locally allocated range for the sub-block")
        comm("  =================================================")
        config.depth = 4
        code(f"int start[{NDIM}];")
        code(f"int end[{NDIM}];")
        if not (arg_idx != -1) and not MULTI_GRID:
            config.depth = 0
            code("#if defined(OPS_MPI) && !defined(OPS_LAZY)")
        code(f"int arg_idx[{NDIM}];")
        if not (arg_idx != -1) and not MULTI_GRID:
            config.depth = 0
            code("#endif")

        config.depth = 0
        code("#if defined(OPS_LAZY) || !defined(OPS_MPI)")
        config.depth = 4
        FOR("n", "0", str(NDIM))
        code("start[n] = range[2*n];")
        code("end[n]   = range[2*n+1];")
        ENDFOR()
        config.depth = 0
        code("#else")
        config.depth = 4
        code(
            f"if (compute_ranges(args, {nargs}, block, range, start, end, arg_idx) < 0) return;"
        )
        config.depth = 0
        code("#endif")

        code("")
        if arg_idx != -1 or MULTI_GRID:
            config.depth = 0
            code("#if defined(OPS_MPI)")
            code("#if defined(OPS_LAZY)")
            config.depth = 4
            code("sub_block_list sb = OPS_sub_block_list[block->index];")
            for n in range(0, NDIM):
                code(f"arg_idx[{n}] = sb->decomp_disp[{n}];")
            config.depth = 0
            code("#else")
            config.depth = 4
            for n in range(0, NDIM):
                code(f"arg_idx[{n}] -= start[{n}];")
            config.depth = 0
            code("#endif")
            code("#else //OPS_MPI")
            config.depth = 4
            for n in range(0, NDIM):
                code(f"arg_idx[{n}] = 0;")
            config.depth = 0
            code("#endif //OPS_MPI")

        code("")
        comm("  =====================================================")
        comm("  Initialize global variable with the dimension of dats")
        comm("  =====================================================")
        config.depth = 4
        for n in range(0, nargs):
            if arg_typ[n] == "ops_arg_dat":
                if NDIM > 1 or (
                    NDIM == 1 and (not dims[n].isdigit() or int(dims[n]) > 1)
                ):
                    code(
                        f"int xdim{n}_{name} = args[{n}].dat->size[0];"
                    )  # *args[{n}].dat->dim;')
                if NDIM > 2 or (
                    NDIM == 2 and (not dims[n].isdigit() or int(dims[n]) > 1)
                ):
                    code(f"int ydim{n}_{name} = args[{n}].dat->size[1];")
                if NDIM > 3 or (
                    NDIM == 3 and (not dims[n].isdigit() or int(dims[n]) > 1)
                ):
                    code(f"int zdim{n}_{name} = args[{n}].dat->size[2];")

        code("")
        config.depth = 0
        comm("  =======================================================")
        comm("  Set up initial pointers and exchange halos if necessary")
        comm("  =======================================================")
        config.depth = 4
        for n in range(0, nargs):
            if arg_typ[n] == "ops_arg_dat":
                code(f"int base{n} = args[{n}].dat->base_offset/sizeof({typs[n]});")
                code(
                    f"{typs[n]}* {clean_type(arg_list[n])}_p = ({typs[n]}*)args[{n}].data_d;"
                )
                if restrict[n] == 1 or prolong[n] == 1:
                    config.depth = 0
                    code("#ifdef OPS_MPI")
                    config.depth = 4
                    code(
                        f"sub_dat_list sd{n} = OPS_sub_dat_list[args[{n}].dat->index];"
                    )
                if restrict[n] == 1:
                    code(
                        f"{clean_type(arg_list[n])}_p += arg_idx[0]*args[{n}].stencil->mgrid_stride[0] - sd{n}->decomp_disp[0] + args[{n}].dat->d_m[0];"
                    )
                    if NDIM > 1:
                        code(
                            f"{clean_type(arg_list[n])}_p += (arg_idx[1]*args[{n}].stencil->mgrid_stride[1] - sd{n}->decomp_disp[1] + args[{n}].dat->d_m[1])*xdim{n}_{name};"
                        )
                    if NDIM > 2:
                        code(
                            f"{clean_type(arg_list[n])}_p += (arg_idx[2]*args[{n}].stencil->mgrid_stride[2] - sd{n}->decomp_disp[2] + args[{n}].dat->d_m[2])*xdim{n}_{name} * ydim{n}_{name};"
                        )
                if prolong[n] == 1:
                    code(
                        f"{clean_type(arg_list[n])}_p += arg_idx[0]/args[{n}].stencil->mgrid_stride[0] - sd{n}->decomp_disp[0] + args[{n}].dat->d_m[0];"
                    )
                    if NDIM > 1:
                        code(
                            f"{clean_type(arg_list[n])}_p += (arg_idx[1]/args[{n}].stencil->mgrid_stride[1] - sd{n}->decomp_disp[1] + args[{n}].dat->d_m[1])*xdim{n}_{name};"
                        )
                    if NDIM > 2:
                        code(
                            f"{clean_type(arg_list[n])}_p += (arg_idx[2]/args[{n}].stencil->mgrid_stride[2] - sd{n}->decomp_disp[2] + args[{n}].dat->d_m[2])*xdim{n}_{name} * ydim{n}_{name};"
                        )

                if restrict[n] == 1 or prolong[n] == 1:
                    config.depth = 0
                    code("#endif")
                    config.depth = 4
            elif arg_typ[n] == "ops_arg_gbl":
                if accs[n] == OPS_READ:
                    if dims[n].isdigit() and int(dims[n]) == 1:
                        code(
                            f"{typs[n]} {clean_type(arg_list[n])}_val = *({typs[n]} *)args[{n}].data;"
                        )
                    else:
                        code(f"{typs[n]} *arg{n}h = ({typs[n]} *)args[{n}].data;")
                else:
                    config.depth = 0
                    code("#ifdef OPS_MPI")
                    config.depth = 4
                    code(
                        f"{typs[n]} * __restrict__ p_a{n} = ({typs[n]} *)(((ops_reduction)args[{n}].data)->data + ((ops_reduction)args[{n}].data)->size * block->index);"
                    )
                    config.depth = 0
                    code("#else //OPS_MPI")
                    config.depth = 4
                    code(
                        f"{typs[n]} * __restrict__ p_a{n} = ({typs[n]} *)((ops_reduction)args[{n}].data)->data;"
                    )
                    config.depth = 0
                    code("#endif //OPS_MPI")
                    config.depth = 4
                code("")

        if has_reduction and not builtin_reduction:
            if ops_cpu and ops_cpu >= 1:
                code("int maxblocks = 1;")
            else:
                code(
                    "int maxblocks = (end[0]-start[0]-1)/block->instance->OPS_block_size_x+1;"
                )

            if NDIM > 1:
                if not ops_cpu or ops_cpu < 2:
                    code(
                        "maxblocks *= (end[1]-start[1]-1)/block->instance->OPS_block_size_y+1;"
                    )
            if NDIM > 2:
                code(
                    "maxblocks *= (end[2]-start[2]-1)/block->instance->OPS_block_size_z+1;"
                )
            code("int reduct_bytes = 0;")
            code("size_t reduct_size = 0;")
            code("")
        if GBL_READ and GBL_READ_MDIM:
            code("int consts_bytes = 0;")
            code("")

        for n in range(0, nargs):
            if arg_typ[n] == "ops_arg_gbl":
                if accs[n] == OPS_READ and (not dims[n].isdigit() or int(dims[n]) > 1):
                    code(f"consts_bytes += ROUND_UP({dims[n]}*sizeof({typs[n]}));")
                elif accs[n] != OPS_READ and not builtin_reduction:
                    code(
                        f"reduct_bytes += ROUND_UP(maxblocks*{dims[n]}*sizeof({typs[n]}));"
                    )
                    code(f"reduct_size = MAX(reduct_size,sizeof({typs[n]}));")
                    code("")

        if GBL_READ and GBL_READ_MDIM:
            code("reallocConstArrays(block->instance,consts_bytes);")
            code("consts_bytes = 0;")
        if has_reduction and not builtin_reduction:
            code("reallocReductArrays(block->instance,reduct_bytes);")
            code("reduct_bytes = 0;")
            code("")
            for n in red_arg_idxs:
                code(f"arg{n}.data = block->instance->OPS_reduct_h + reduct_bytes;")
                code(
                    f"{typs[n]} *arg{n}_data_d = ({typs[n]}*)(block->instance->OPS_reduct_d + reduct_bytes);"
                )

                FOR("b", "0", "maxblocks")
                FOR("d", "0", str(dims[n]))
                if accs[n] == OPS_INC:
                    code(f"(({typs[n]} *)arg{n}.data)[d+b*{dims[n]}] = ZERO_{typs[n]};")
                elif accs[n] == OPS_MAX:
                    code(
                        f"(({typs[n]} *)arg{n}.data)[d+b*{dims[n]}] = -INFINITY_{typs[n]};"
                    )
                elif accs[n] == OPS_MIN:
                    code(
                        f"(({typs[n]} *)arg{n}.data)[d+b*{dims[n]}] = INFINITY_{typs[n]};"
                    )
                ENDFOR()
                ENDFOR()
                code(
                    f"reduct_bytes += ROUND_UP(maxblocks*{dims[n]}*sizeof({typs[n]}));"
                )
                code("")
        else:
            for n in red_arg_idxs:
                assert dims[n].isdigit() and "dinamic extent in SYCL"
                if int(dims[n]) == 1:
                    code(
                        f"cl::sycl::buffer<{typs[n]}, 1> reduct_p_a{n}(p_a{n}, {dims[n]});"
                    )
                else:
                    for i in range(int(dims[n])):
                        code(
                            f"cl::sycl::buffer<{typs[n]},1> reduct_p_a{n}_{i}(p_a{n} + {i}, 1);"
                        )

        for n in range(0, nargs):
            if arg_typ[n] == "ops_arg_gbl":
                if accs[n] == OPS_READ and (not dims[n].isdigit() or int(dims[n]) > 1):
                    code("consts_bytes = 0;")
                    code(f"arg{n}.data = block->instance->OPS_consts_h + consts_bytes;")
                    code(
                        f"{typs[n]}* arg{n}_data_d = ({typs[n]}*)( block->instance->OPS_consts_d + consts_bytes);"
                    )
                    code(
                        f"for (int d=0; d<{dims[n]}; d++) (({typs[n]} *)arg{n}.data)[d] = arg{n}h[d];"
                    )
                    code(f"consts_bytes += ROUND_UP({dims[n]}*sizeof({typs[n]}));")
        if GBL_READ and GBL_READ_MDIM:
            code("mvConstArraysToDevice(block->instance,consts_bytes);")

        if has_reduction and not builtin_reduction:
            code("mvReductArraysToDevice(block->instance,reduct_bytes);")

        code("")

        config.depth = 0
        code("#ifndef OPS_LAZY")
        comm("  ==============")
        comm("  Halo Exchanges")
        comm("  ==============")
        config.depth = 4
        code(f"ops_H_D_exchanges_device(args, {nargs});")
        code(f"ops_halo_exchanges(args,{nargs},range);")
        config.depth = 0
        code("#endif")
        code("")
        
        config.depth = 4
        IF("block->instance->OPS_diags > 1")
        code("ops_timers_core(&__c2, &__t2);")
        code(f"block->instance->OPS_kernels[{nk}].mpi_time += __t2 - __t1;")
        ENDIF()
        code("")

        for d in range(0, NDIM):
            code(f"int start_{d} = start[{d}];")
            code(f"int end_{d} = end[{d}];")
            if arg_idx != -1:
                code(f"int arg_idx_{d} = arg_idx[{d}];")

        condition = "(end[0]-start[0])>0"
        if NDIM > 1:
            condition = condition + " && (end[1]-start[1])>0"
        if NDIM > 2:
            condition = condition + " && (end[2]-start[2])>0"
        IF(condition)
        code(
            "block->instance->sycl_instance->queue->submit([&](cl::sycl::handler &cgh) {"
        )
        config.depth += 2

        if has_reduction:
            if not builtin_reduction:
                local_mem_size = (
                    "reduct_size * cl::sycl::range<1>("
                    + "*".join(
                        [
                            "block->instance->OPS_block_size_" + ["x", "y", "z"][i]
                            for i in range(NDIM)
                        ]
                    )
                    + ")"
                )
                code(
                    "cl::sycl::accessor<char, 1, cl::sycl::access::mode::read_write, "
                    "cl::sycl::access::target::local> local_mem("
                    + local_mem_size
                    + ",cgh);"
                )
            else:
                for n in red_arg_idxs:
                    assert dims[n].isdigit()
                    red_operation = (
                        "plus"
                        if accs[n] == OPS_INC
                        else "maximum"
                        if accs[n] == OPS_MAX
                        else "minimum"
                    )
                    red_identity = (
                        f"({typs[n]})0"
                        if accs[n] == OPS_INC
                        else f"std::numeric_limits<{typs[n]}>::min()"
                        if accs[n] == OPS_MAX
                        else f"std::numeric_limits<{typs[n]}>::max()"
                    )
                    if int(dims[n]) == 1:
                        code(
                            f"auto reduction_acc_p_a{n} = reduct_p_a{n}.get_access(cgh);"
                        )
                        code(
                            f"auto reduction_handler_p_a{n} = {sycl_guarded_namespace}reduction(reduction_acc_p_a{n}, {red_identity}, {sycl_guarded_namespace}{red_operation}<{typs[n]}>());"
                        )
                    else:
                        for i in range(int(dims[n])):
                            code(
                                f"auto reduction_acc_p_a{n}_{i} = reduct_p_a{n}_{i}.get_access(cgh);"
                            )
                            code(
                                f"auto reduction_handler_p_a{n}_{i} = {sycl_guarded_namespace}reduction(reduction_acc_p_a{n}_{i}, {red_identity}, {sycl_guarded_namespace}{red_operation}<{typs[n]}>());"
                            )
        code("")
        for c in global_consts:
            code(c)
        code("")

        if flat_parallel:
            code(f"cgh.parallel_for<class {name}_kernel>(cl::sycl::range<{NDIM}>(")
            if NDIM > 2:
                code("     end[2]-start[2],")
            if NDIM > 1:
                code("     end[1]-start[1],")
            code("     end[0]-start[0])")
        else:
            code(
                f"cgh.parallel_for<class {name}_kernel>(cl::sycl::nd_range<{NDIM}>(cl::sycl::range<{NDIM}>("
            )
            if NDIM > 2:
                code(
                    "     ((end[2]-start[2]-1)/block->instance->OPS_block_size_z+1)*block->instance->OPS_block_size_z,"
                )
            if NDIM > 1:
                if ops_cpu and ops_cpu >= 2:
                    code("      end[1]-start[1],")
                else:
                    code(
                        "     ((end[1]-start[1]-1)/block->instance->OPS_block_size_y+1)*block->instance->OPS_block_size_y,"
                    )
            if ops_cpu and ops_cpu >= 1:
                code("      end[0]-start[0]")
            else:
                code(
                    "      ((end[0]-start[0]-1)/block->instance->OPS_block_size_x+1)*block->instance->OPS_block_size_x"
                )
            code(f"       ),cl::sycl::range<{NDIM}>(")
            if NDIM > 2:
                code("       block->instance->OPS_block_size_z,")
            if NDIM > 1:
                if ops_cpu and ops_cpu >= 2:
                    code("      end[1]-start[1],")
                else:
                    code("       block->instance->OPS_block_size_y,")
            if ops_cpu and ops_cpu >= 1:
                code("      end[0]-start[0]")
            else:
                code("block->instance->OPS_block_size_x")

            code("       ))")
        if has_reduction and builtin_reduction:
            for n in red_arg_idxs:
                if int(dims[n]) == 1:
                    code(f", reduction_handler_p_a{n}")
                else:
                    code(
                        ","
                        + ",".join(
                            [
                                f"reduction_handler_p_a{n}_{i}"
                                for i in range(int(dims[n]))
                            ]
                        )
                    )
        if flat_parallel:
            code(f", [=](cl::sycl::item<{NDIM}> item")
        else:
            code(f", [=](cl::sycl::nd_item<{NDIM}> item")
        if has_reduction and builtin_reduction:
            for n in red_arg_idxs:
                if int(dims[n]) == 1:
                    code(f", auto &reduction_h_p_a{n}")
                else:
                    code(
                        ","
                        + ",".join(
                            [
                                f" auto &reduction_h_p_a{n}_{i}"
                                for i in range(int(dims[n]))
                            ]
                        )
                    )
        code(") [[intel::kernel_args_restrict]] {")
        code("")
        config.depth += 4
        line3 = ""
        for n in range(0, nargs):
            if arg_typ[n] == "ops_arg_dat":
                line3 = line3 + arg_list[n] + ","
        if NDIM > 2:
            if flat_parallel:
                code("int n_z = item.get_id(0)+start_2;")
            else:
                code("int n_z = item.get_global_id()[0]+start_2;")
        if NDIM > 1:
            if flat_parallel:
                code(f"int n_y = item.get_id({NDIM - 2})+start_1;")
            else:
                code(f"int n_y = item.get_global_id()[{NDIM - 2}]+start_1;")
        if flat_parallel:
            code(f"int n_x = item.get_id({NDIM - 1})+start_0;")
        else:
            code(f"int n_x = item.get_global_id()[{NDIM - 1}]+start_0;")

        code("")
        if arg_idx != -1:
            if NDIM == 1:
                code("int " + clean_type(arg_list[arg_idx]) + "[] = {arg_idx_0+n_x};")
            elif NDIM == 2:
                code(
                    "int "
                    + clean_type(arg_list[arg_idx])
                    + "[] = {arg_idx_0+n_x, arg_idx_1+n_y};"
                )
            elif NDIM == 3:
                code(
                    "int "
                    + clean_type(arg_list[arg_idx])
                    + "[] = {arg_idx_0+n_x, arg_idx_1+n_y, arg_idx_2+n_z};"
                )
            code("")

        for n in range(0, nargs):
            if arg_typ[n] == "ops_arg_dat":
                pre = ""
                if accs[n] == OPS_READ:
                    pre = "const "
                offset = ""
                dim = ""
                sizelist = ""
                extradim = 0
                if dims[n].isdigit() and int(dims[n]) > 1:
                    dim = dims[n] + ", "
                    extradim = 1
                elif not dims[n].isdigit():
                    dim = f"arg{n}.dim, "
                    extradim = 1
                if restrict[n] == 1:
                    n_x = f"n_x*args[{n}].stencil->mgrid_stride[0]"
                    n_y = f"n_y*args[{n}].stencil->mgrid_stride[1]"
                    n_z = f"n_z*args[{n}].stencil->mgrid_stride[2]"
                elif prolong[n] == 1:
                    n_x = f"(n_x+arg_idx[0]%args[{n}].stencil->mgrid_stride[0])/args[{n}].stencil->mgrid_stride[0]"
                    n_y = f"(n_y+arg_idx[1]%args[{n}].stencil->mgrid_stride[1])/args[{n}].stencil->mgrid_stride[1]"
                    n_z = f"(n_z+arg_idx[2]%args[{n}].stencil->mgrid_stride[2])/args[{n}].stencil->mgrid_stride[2]"
                else:
                    n_x = "n_x"
                    n_y = "n_y"
                    n_z = "n_z"

                if NDIM > 0:
                    offset += f"{n_x}*{stride[n][0]}"
                if NDIM > 1:
                    offset += f" + {n_y} * xdim{n}_{name}*{stride[n][1]}"
                if NDIM > 2:
                    offset += (
                        f" + {n_z} * xdim{n}_{name} * ydim{n}_{name}*{stride[n][2]}"
                    )
                dimlabels = "xyzuv"
                for i in range(1, NDIM + extradim):
                    sizelist += f"{dimlabels[i-1]}dim{n}_{name}, "

                if not dims[n].isdigit() or int(dims[n]) > 1:
                    code("#ifdef OPS_SOA")
                code(
                    f"{pre}ACC<{typs[n]}> {arg_list[n]}({dim}{sizelist}&{arg_list[n]}_p[0] + base{n} + {offset});"
                )
                if not dims[n].isdigit() or int(dims[n]) > 1:
                    code("#else")
                    code(
                        f"{pre}ACC<{typs[n]}> {arg_list[n]}({dim}{sizelist}&{arg_list[n]}_p[0] + {dim[:-2]}*({offset}));"
                    )
                    code("#endif")
        for n in range(0, nargs):
            if arg_typ[n] == "ops_arg_gbl":
                if accs[n] == OPS_MIN:
                    code(f"{typs[n]} {arg_list[n]}[{dims[n]}];")
                    for d in range(0, int(dims[n])):
                        code(
                            f"{arg_list[n]}[{d}] = +INFINITY_{typs[n]};"
                        )  # need +INFINITY_ change to
                if accs[n] == OPS_MAX:
                    code(f"{typs[n]} {arg_list[n]}[{dims[n]}];")
                    for d in range(0, int(dims[n])):
                        code(
                            f"{arg_list[n]}[{d}] = -INFINITY_{typs[n]};"
                        )  # need -INFINITY_ change to
                if accs[n] == OPS_INC:
                    code(f"{typs[n]} {arg_list[n]}[{dims[n]}];")
                    for d in range(0, int(dims[n])):
                        code(f"{arg_list[n]}[{d}] = ZERO_{typs[n]};")
                if accs[n] == OPS_WRITE:  # this may not be correct
                    code(f"{typs[n]} {arg_list[n]}[{dims[n]}];")
                    for d in range(0, int(dims[n])):
                        code(f"{arg_list[n]}[{d}] = ZERO_{typs[n]};")
                if accs[n] == OPS_READ:
                    if dims[n].isdigit() and int(dims[n]) == 1:
                        code(
                            f"const {typs[n]} *{clean_type(arg_list[n])} = &{clean_type(arg_list[n])}_val;"
                        )
                    else:
                        code(
                            f"const {typs[n]} *{clean_type(arg_list[n])} = arg{n}_data_d;"
                        )

        # insert user kernel
        config.depth = 0
        comm("  =========")
        comm("  User code")
        comm("  =========")
        config.depth = 16
        cond = "n_x < end_0"
        if NDIM > 1:
            cond = cond + " && n_y < end_1"
        if NDIM > 2:
            cond = cond + " && n_z < end_2"
        IF(cond)
        code(kernel_text)
        ENDIF()

        if has_reduction:
            if builtin_reduction:
                for n in red_arg_idxs:
                    assert dims[n].isdigit()
                    if int(dims[n]) == 1:
                        code(f"reduction_h_p_a{n}.combine({arg_list[n]}[0]);")
                    else:
                        for i in range(int(dims[n])):
                            code(f"reduction_h_p_a{n}_{i}.combine({arg_list[n]}[{i}]);")
            else:
                code("int group_size = item.get_local_range(0);")
                if NDIM > 1:
                    code("group_size *= item.get_local_range(1);")
                if NDIM > 2:
                    code("group_size *= item.get_local_range(2);")
                for n in red_arg_idxs:
                    FOR("d", "0", dims[n])
                    if accs[n] == OPS_MIN:
                        code(
                            f"ops_reduction_sycl<OPS_MIN>(arg{n}_data_d + d+item.get_group_linear_id()*{dims[n]}, {arg_list[n]}[d], ({typs[n]}*)&local_mem[0], item, group_size);"
                        )
                    if accs[n] == OPS_MAX:
                        code(
                            f"ops_reduction_sycl<OPS_MAX>(arg{n}_data_d + d+item.get_group_linear_id()*{dims[n]}, {arg_list[n]}[d], ({typs[n]}*)&local_mem[0], item, group_size);"
                        )
                    if accs[n] == OPS_INC:
                        code(
                            f"ops_reduction_sycl<OPS_INC>(arg{n}_data_d + d+item.get_group_linear_id()*{dims[n]}, {arg_list[n]}[d], ({typs[n]}*)&local_mem[0], item, group_size);"
                        )
                    if accs[n] == OPS_WRITE:  # this may not be correct
                        code(
                            f"ops_reduction_sycl<OPS_MIN>(arg{n}_data_d + d+item.get_group_linear_id()*{dims[n]}, {arg_list[n]}[d], ({typs[n]}*)&local_mem[0], item, group_size);"
                        )
                    ENDFOR()

        config.depth -= 4
        code("});")
        config.depth -= 4
        code("});")
        ENDIF()

        #
        # Complete Reduction Operation by moving data onto host
        # and reducing over blocks
        #
        if has_reduction and not builtin_reduction:
            code("mvReductArraysToHost(block->instance,reduct_bytes);")
            for n in red_arg_idxs:
                FOR("b", "0", "maxblocks")
                FOR("d", "0", str(dims[n]))
                if accs[n] == OPS_INC:
                    code(
                        f"p_a{n}[d] = p_a{n}[d] + (({typs[n]} *)arg{n}.data)[d+b*{dims[n]}];"
                    )
                elif accs[n] == OPS_MAX:
                    code(
                        f"p_a{n}[d] = MAX(p_a{n}[d],(({typs[n]} *)arg{n}.data)[d+b*{dims[n]}]);"
                    )
                elif accs[n] == OPS_MIN:
                    code(
                        f"p_a{n}[d] = MIN(p_a{n}[d],(({typs[n]} *)arg{n}.data)[d+b*{dims[n]}]);"
                    )
                ENDFOR()
                ENDFOR()

        code("")
        IF("block->instance->OPS_diags > 1")
        code("block->instance->sycl_instance->queue->wait();")
        code("ops_timers_core(&__c1, &__t1);")
        code(f"block->instance->OPS_kernels[{nk}].time += __t1 -__t2;")
        ENDIF()

        code("")
        config.depth = 0
        code("#ifndef OPS_LAZY")
        config.depth = 4
        code(f"ops_set_dirtybit_device(args, {nargs});")
        for n in range(0, nargs):
            if arg_typ[n] == "ops_arg_dat" and (
                accs[n] == OPS_WRITE or accs[n] == OPS_RW or accs[n] == OPS_INC
            ):
                code(f"ops_set_halo_dirtybit3(&args[{n}], range);")
        config.depth = 0
        code("#endif")

        code("")
        config.depth = 4
        IF("block->instance->OPS_diags > 1")
        config.depth = 0
        comm("  ====================")
        comm("  Update kernel record")
        comm("  ====================")
        config.depth = 4
        code("ops_timers_core(&__c2, &__t2);")
        code(f"block->instance->OPS_kernels[{nk}].mpi_time += __t2 -__t1;")
        for n in range(0, nargs):
            if arg_typ[n] == "ops_arg_dat":
                code(
                    f"block->instance->OPS_kernels[{nk}].transfer += ops_compute_transfer(dim, start, end, &arg{n});"
                )
        ENDIF()
        config.depth = 0
        code("}")
        code("")

        code("#ifdef OPS_LAZY")
        code(f"void ops_par_loop_{name}(")
        code("    const char *name,")
        code("    ops_block block,")
        code("    int dim,")
        code("    int* range,")
        for n in range(nargs):
            if n == nargs-1:
                code(f"    ops_arg arg{n}")
            else:
                code(f"    ops_arg arg{n},")
        code("  )\n{")
        config.depth = 4
        code(f"ops_arg args[{nargs}];")
        code("")
        for n in range (0, nargs):
          code(f"args[{n}] = arg{n};")

        code("")
        text = 'create_kerneldesc_and_enque(name, '
        text = text + f'"{name}", '
        text = text + f'args, {nargs}, '
        text = text + f'{nk}, '
        text = text + 'dim, 1, range, block, '
        text = text + f'ops_par_loop_{name}_execute'
        text = text + ');'
        code(text)
    
        config.depth = 0
        code("}")
        code("#endif")

        ##########################################################################
        #  output individual kernel file
        ##########################################################################
        util.write_text_to_file(f"./sycl/{name}_kernel.cpp")

    # end of main kernel call loop

    ##########################################################################
    #  output one master kernel file
    ##########################################################################
    comm("header")
    code(f"#define OPS_{NDIM}D")
    if soa_set:
        code("#define OPS_SOA")
    code("#define OPS_API 2")
    code('#include "ops_lib_core.h"')
    code("#ifdef OPS_MPI")
    code('#include "ops_mpi_core.h"')
    code("#include <limits>")
    code("#endif")
    if os.path.exists(os.path.join(src_dir, "user_types.h")):
        code('#include "user_types.h"')
    code("")

    code('#include "ops_sycl_rt_support.h"')
    code('#include "ops_sycl_reduction.h"')

    comm(" global constants")
    for nc in range(0, len(consts)):
        code(
            "cl::sycl::buffer<"
            + consts[nc]["type"]
            + ",1> *"
            + consts[nc]["name"].replace('"', "")
            + "_p=nullptr;"
        )
        if (not consts[nc]["dim"].isdigit()) or int(consts[nc]["dim"]) > 1:
            code(
                "extern "
                + consts[nc]["type"]
                + " *"
                + consts[nc]["name"].replace('"', "")
                + ";"
            )
        else:
            code(
                "extern "
                + consts[nc]["type"]
                + " "
                + consts[nc]["name"].replace('"', "")
                + ";"
            )
    code("")

    code("void ops_init_backend() {}")
    code("")

    code(
        "void ops_decl_const_char(OPS_instance *instance, int dim, char const * type, int size, char * dat, char const * name ) {"
    )
    config.depth = config.depth + 4
    for nc in range(0, len(consts)):
        IF('!strcmp(name,"' + (consts[nc]["name"].replace('"', "")).strip() + '")')
        code(
            "if ("
            + consts[nc]["name"].replace('"', "")
            + "_p == nullptr) "
            + consts[nc]["name"].replace('"', "")
            + "_p = new cl::sycl::buffer<"
            + consts[nc]["type"]
            + ",1>(cl::sycl::range<1>(dim));"
        )
        code(
            "auto accessor = (*"
            + consts[nc]["name"].replace('"', "")
            + "_p).get_access<cl::sycl::access::mode::write>();"
        )
        FOR("d", "0", "dim")
        code(f"accessor[d] = ((" + consts[nc]["type"] + "*)dat)[d];")
        ENDFOR()
        ENDIF()
        code("else")
    code("{")
    config.depth = config.depth + 4
    code('throw OPSException(OPS_RUNTIME_ERROR, "error: unknown const name");')
    ENDIF()
    config.depth = config.depth - 4
    code("}")
    code("")
    comm("user kernel files")

    for kernel_name in map(lambda kernel: kernel["name"], kernels):
        if kernel_name == "calc_dt_kernel_print":
            code(f'#include "../mpi_openmp/{kernel_name}_kernel.cpp"')
        else:
            code(f'#include "{kernel_name}_kernel.cpp"')

    util.write_text_to_file(f"./sycl/sycl_kernels.cpp")
