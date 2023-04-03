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
## @brief OPS MPI_seq code generator
#
#  This routine is called by ops.py which parses the input files
#
#  It produces a file xxx_cpu_kernel.cpp for each kernel,
#  plus a master kernel file
#

"""
OPS MPI_seq code generator

This routine is called by ops.py which parses the input files

It produces a file xxx_cpu_kernel.cpp for each kernel,
plus a master kernel file

"""

import os

import config
from config import OPS_READ, OPS_WRITE, OPS_RW, OPS_INC, OPS_MAX, OPS_MIN

import util
from util import comm, code, FOR, ENDFOR, IF, ENDIF


def clean_type(arg):
    for qual in ["__restrict__", "RESTRICT", "__volatile__"]:
        arg = arg.replace(qual, "")
    return arg


def ops_gen_mpi_lazy(master, consts, kernels, soa_set, offload=0):
    NDIM = 2  # the dimension of the application is hardcoded here .. need to get this dynamically

    gen_full_code = 1

    src_dir = os.path.dirname(master) or "."
    master_basename = os.path.splitext(os.path.basename(master))

    ##########################################################################
    #  create new kernel file
    ##########################################################################
    if offload:
        if not os.path.exists("./OpenMP_offload"):
            os.makedirs("./OpenMP_offload")
    else:
        if not os.path.exists("./MPI_OpenMP"):
            os.makedirs("./MPI_OpenMP")

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
            _,
            _,
            has_reduction,
            arg_idx,
            _,
        ) = util.create_kernel_info(kernels[nk])

        ##########################################################################
        #  start with seq kernel function
        ##########################################################################

        code("")
        comm("user function")
        kernel_text, arg_list = util.get_kernel_body_and_arg_list(
            name, src_dir, arg_typ
        )

        comm("")
        comm(" host stub function")
        code("#ifndef OPS_LAZY")
        code(
            f"void ops_par_loop_{name}(char const *name, ops_block block, int dim, int* range,"
        )
        code(util.group_n_per_line([f" ops_arg arg{n}" for n in range(nargs)]) + ") {")
        code("#else")
        code(f"void ops_par_loop_{name}_execute(ops_kernel_descriptor *desc) {{")
        config.depth = 2
        code("ops_block block = desc->block;")
        code("int dim = desc->dim;")
        code("int *range = desc->range;")

        for n in range(0, nargs):
            code(f"ops_arg arg{n} = desc->args[{n}];")

        code("#endif")

        code("")
        comm("Timing")
        code("double __t1,__t2,__c1,__c2;")
        code("")

        code(
            f"ops_arg args[{nargs}] = {{"
            + ",".join([f" arg{n}" for n in range(nargs)])
            + "};\n\n"
        )
        code("")
        code("#if defined(CHECKPOINTING) && !defined(OPS_LAZY)")
        code(f"if (!ops_checkpointing_before(args,{nargs},range,{nk})) return;")
        code("#endif")
        code("")

        if gen_full_code:
            IF("block->instance->OPS_diags > 1")
            code(f'ops_timing_realloc(block->instance,{nk},"{name}");')
            code(f"block->instance->OPS_kernels[{nk}].count++;")
            code("ops_timers_core(&__c2,&__t2);")
            ENDIF()
            code("")

        code("#ifdef OPS_DEBUG")
        code(f'ops_register_args(block->instance, args, "{name}");')
        code("#endif")
        code("")

        code("")

        comm("compute locally allocated range for the sub-block")
        code(f"int start[{NDIM}];")
        code(f"int end[{NDIM}];")
        if not (arg_idx != -1) and not MULTI_GRID:
            code("#if defined(OPS_MPI) && !defined(OPS_LAZY)")
        code(f"int arg_idx[{NDIM}];")
        if not (arg_idx != -1) and not MULTI_GRID:
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
                code(f"arg_idx[{n}] = sb->decomp_disp[{n}];")
            code("#else")
            for n in range(0, NDIM):
                code(f"arg_idx[{n}] -= start[{n}];")
            code("#endif")
            code("#else //OPS_MPI")
            for n in range(0, NDIM):
                code(f"arg_idx[{n}] = 0;")
            code("#endif //OPS_MPI")

        code("")
        comm("initialize global variable with the dimension of dats")
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
        comm("set up initial pointers and exchange halos if necessary")
        ptr_suffix = ""
        if offload:
            ptr_suffix = "_d"
        for n in range(0, nargs):
            if arg_typ[n] == "ops_arg_dat":
                code(f"int base{n} = args[{n}].dat->base_offset;")
                code(
                    f"{typs[n]} * __restrict__ {clean_type(arg_list[n])}_p = ({typs[n]} *)(args[{n}].data{ptr_suffix} + base{n});"
                )
                if restrict[n] == 1 or prolong[n] == 1:
                    code("#ifdef OPS_MPI")
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
                    code("#endif")
                if offload:
                    code(f"size_t arg{n}_size = (args[{n}].dat->mem - ((char*){clean_type(arg_list[n])}_p-args[{n}].data{ptr_suffix}))/sizeof({typs[n]});")
            elif arg_typ[n] == "ops_arg_gbl":
                if accs[n] == OPS_READ:
                    code(
                        f"{typs[n]} * __restrict__ {clean_type(arg_list[n])} = ({typs[n]} *)args[{n}].data;"
                    )
                else:
                    code("#ifdef OPS_MPI")
                    code(
                        f"{typs[n]} * __restrict__ p_a{n} = ({typs[n]} *)(((ops_reduction)args[{n}].data)->data + ((ops_reduction)args[{n}].data)->size * block->index);"
                    )
                    code("#else //OPS_MPI")
                    code(
                        f"{typs[n]} * __restrict__ p_a{n} = ({typs[n]} *)((ops_reduction)args[{n}].data)->data;"
                    )
                    code("#endif //OPS_MPI")
                code("")
            code("")
        code("")

        code("")

        code("#ifndef OPS_LAZY")
        comm("Halo Exchanges")
        exec_space = "host"
        if offload:
            exec_space = "device"
        code(f"ops_H_D_exchanges_{exec_space}(args, {nargs});")
        code(f"ops_halo_exchanges(args,{nargs},range);")
        code(f"ops_H_D_exchanges_{exec_space}(args, {nargs});")
        code("#endif")
        code("")
        if gen_full_code == 1:
            IF("block->instance->OPS_diags > 1")
            code("ops_timers_core(&__c1,&__t1);")
            code(f"block->instance->OPS_kernels[{nk}].mpi_time += __t1-__t2;")
            ENDIF()
            code("")

        for n in range(0, nargs):
            if arg_typ[n] == "ops_arg_gbl":
                if accs[n] != OPS_READ:
                    for d in range(0, int(dims[n])):
                        code(f"{typs[n]} p_a{n}_{d} = p_a{n}[{d}];")

        line = ""
        for n in range(0, nargs):
            if arg_typ[n] == "ops_arg_gbl":
                if accs[n] == OPS_MIN:
                    for d in range(0, int(dims[n])):
                        line += f" reduction(min:p_a{n}_{d})"
                if accs[n] == OPS_MAX:
                    for d in range(0, int(dims[n])):
                        line += f" reduction(max:p_a{n}_{d})"
                if accs[n] == OPS_INC:
                    for d in range(0, int(dims[n])):
                        line += f" reduction(+:p_a{n}_{d})"
                if accs[n] == OPS_WRITE:  # this may not be correct ..
                    for d in range(0, int(dims[n])):
                        line += f" reduction(+:p_a{n}_{d})"
        if NDIM == 3 and has_reduction == 0 and offload == 0:
            line2 = " collapse(2)"
        else:
            line2 = line
        if offload == 0:
            code("#pragma omp parallel for" + line2)
        else:
            for g_m in range(0,nargs):
                if arg_typ[g_m] == "ops_arg_dat":
                    line2 += f" map({'to' if accs[g_m] == OPS_READ else 'tofrom'}:{arg_list[g_m]}_p[0:arg{g_m}_size])"
                elif arg_typ[g_m] == "ops_arg_gbl" and accs[g_m] == OPS_READ:
                    line2 += f" map(to:{clean_type(arg_list[g_m])}[0:{dims[g_m]}])"
            code(f"#pragma omp target teams distribute parallel for collapse({NDIM})" + line2)
        if NDIM > 2:
            FOR("n_z", "start[2]", "end[2]")
        if NDIM > 1:
            FOR("n_y", "start[1]", "end[1]")

        line3 = ""
        for n in range(0, nargs):
            if arg_typ[n] == "ops_arg_dat":
                line3 += arg_list[n] + ","
        if NDIM > 1:
            if offload == 0:
                code("#ifdef __INTEL_COMPILER")
                code("#pragma loop_count(10000)")
                code("#pragma omp simd" + line)  # +' aligned('+clean_type(line3[:-1])+')')
                code("#elif defined(__clang__)")
                code("#pragma clang loop vectorize(assume_safety)")
                code("#elif defined(__GNUC__)")
                code("#pragma GCC ivdep")
                code("#else")
                code("#pragma simd")
                code("#endif")
        FOR("n_x", "start[0]", "end[0]")
        if arg_idx != -1:
            if NDIM == 1:
                code(f"int {clean_type(arg_list[arg_idx])}[] = {{arg_idx[0]+n_x}};")
            elif NDIM == 2:
                code(
                    f"int {clean_type(arg_list[arg_idx])}[] = {{arg_idx[0]+n_x, arg_idx[1]+n_y}};"
                )
            elif NDIM == 3:
                code(
                    f"int {clean_type(arg_list[arg_idx])}[] = {{arg_idx[0]+n_x, arg_idx[1]+n_y, arg_idx[2]+n_z}};"
                )

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
                    f"{pre}ACC<{typs[n]}> {arg_list[n]}({dim}{sizelist}{arg_list[n]}_p + {offset});"
                )
                if not dims[n].isdigit() or int(dims[n]) > 1:
                    code("#else")
                    code(
                        f"{pre}ACC<{typs[n]}> {arg_list[n]}({dim}{sizelist}{arg_list[n]}_p + {dim[:-2]}*({offset}));"
                    )
                    code("#endif")
        for n in range(0, nargs):
            if arg_typ[n] == "ops_arg_gbl":
                if accs[n] == OPS_MIN:
                    code(f"{typs[n]} {arg_list[n]}[{dims[n]}];")
                    for d in range(0, int(dims[n])):
                        code(
                            f"{arg_list[n]}[{d}] = p_a{n}[{d}];"
                        )  # need +INFINITY_ change to
                if accs[n] == OPS_MAX:
                    code(f"{typs[n]} {arg_list[n]}[{dims[n]}];")
                    for d in range(0, int(dims[n])):
                        code(
                            f"{arg_list[n]}[{d}] = p_a{n}[{d}];"
                        )  # need -INFINITY_ change to
                if accs[n] == OPS_INC:
                    code(f"{typs[n]} {arg_list[n]}[{dims[n]}];")
                    for d in range(0, int(dims[n])):
                        code(f"{arg_list[n]}[{d}] = ZERO_{typs[n]};")
                if accs[n] == OPS_WRITE:  # this may not be correct
                    code(f"{typs[n]} {arg_list[n]}[{dims[n]}];")
                    for d in range(0, int(dims[n])):
                        code(f"{arg_list[n]}[{d}] = ZERO_{typs[n]};")

        # insert user kernel
        code(kernel_text)

        for n in range(0, nargs):
            if arg_typ[n] == "ops_arg_gbl":
                if accs[n] == OPS_MIN:
                    for d in range(0, int(dims[n])):
                        code(f"p_a{n}_{d} = MIN(p_a{n}_{d},{arg_list[n]}[{d}]);")
                if accs[n] == OPS_MAX:
                    for d in range(0, int(dims[n])):
                        code(f"p_a{n}_{d} = MAX(p_a{n}_{d},{arg_list[n]}[{d}]);")
                if accs[n] == OPS_INC:
                    for d in range(0, int(dims[n])):
                        code(f"p_a{n}_{d} +={arg_list[n]}[{d}];")
                if accs[n] == OPS_WRITE:  # this may not be correct
                    for d in range(0, int(dims[n])):
                        code(f"p_a{n}_{d} +={arg_list[n]}[{d}];")

        ENDFOR()
        if NDIM > 1:
            ENDFOR()
        if NDIM > 2:
            ENDFOR()

        for n in range(0, nargs):
            if arg_typ[n] == "ops_arg_gbl":
                if accs[n] != OPS_READ:
                    for d in range(0, int(dims[n])):
                        code(f"p_a{n}[{d}] = p_a{n}_{d};")

        if gen_full_code == 1:
            IF("block->instance->OPS_diags > 1")
            code("ops_timers_core(&__c2,&__t2);")
            code(f"block->instance->OPS_kernels[{nk}].time += __t2-__t1;")
            ENDIF()

        code("#ifndef OPS_LAZY")
        code(f"ops_set_dirtybit_{exec_space}(args, {nargs});")
        for n in range(0, nargs):
            if arg_typ[n] == "ops_arg_dat" and (
                accs[n] == OPS_WRITE or accs[n] == OPS_RW or accs[n] == OPS_INC
            ):
                code(f"ops_set_halo_dirtybit3(&args[{n}],range);")
        code("#endif")

        if gen_full_code == 1:
            code("")
            IF("block->instance->OPS_diags > 1")
            comm("Update kernel record")
            code("ops_timers_core(&__c1,&__t1);")
            code(f"block->instance->OPS_kernels[{nk}].mpi_time += __t1-__t2;")
            for n in range(0, nargs):
                if arg_typ[n] == "ops_arg_dat":
                    code(
                        f"block->instance->OPS_kernels[{nk}].transfer += ops_compute_transfer(dim, start, end, &arg{n});"
                    )
            ENDIF()
        config.depth = config.depth - 2
        code("}")
        code("")

        code("")
        code("#ifdef OPS_LAZY")
        code(
            f"void ops_par_loop_{name}(char const *name, ops_block block, int dim, int* range,"
        )
        code(util.group_n_per_line([f" ops_arg arg{n}" for n in range(nargs)]) + ") {")
        config.depth = 2
        n_per_line=5
        text = f'ops_arg args[{nargs}] = {{'
        for n in range (0, nargs):
          text += f' arg{n}'
          if nargs != 1 and n != nargs-1:
            text += ','
          else:
            text += ' };\n'
          if n%n_per_line == 5 and n != nargs-1:
            text +='\n                    '
        code(text)

        comm('create kernel descriptor and pass it to ops_enqueue_kernel')
        text = 'create_kerneldesc_and_enque(name, args, '
        text = text + f'{nargs}, '
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
        if offload:
            util.write_text_to_file(f"./OpenMP_offload/{name}_ompoffload_kernel.cpp")
        else:
            util.write_text_to_file(f"./MPI_OpenMP/{name}_cpu_kernel.cpp")

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
    code("#endif")
    if os.path.exists(os.path.join(src_dir, "user_types.h")):
        code('#include "user_types.h"')
    code("")

    util.generate_extern_global_consts_declarations(consts)

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
