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
OPS CUDA ajdoint code generator

This routine is called by ops.py which parses the input files

It produces a file xxx_cuda_kernel.cpp for each kernel,
plus a master kernel file

"""

import datetime
import glob
import os
import re
from functools import reduce
from operator import mul

import config
import util

verbose = util.verbose

para_parse = util.para_parse
comment_remover = util.comment_remover
remove_trailing_w_space = util.remove_trailing_w_space
parse_signature = util.parse_signature
complex_numbers_cuda = util.complex_numbers_cuda
check_accs = util.check_accs
mult = util.mult
convert_ACC = util.convert_ACC
find_kernel = util.find_kernel
find_kernel_with_retry = util.find_kernel_with_retry

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

def has_ldim_reduction(stride):
    return stride[0] == 0 or stride[1] == 0

def inclusive_scan(strings, join_str = '*'):
    result = []
    for i in range(1, len(strings)):
        result.append(join_str.join(strings[:i]))
    return result


def get_stride_n(n, stride, NDIM):
    return stride[n * NDIM : (n + 1) * NDIM]


def  get_bounding_rectangle_for_stencil(points, stride, NDIM):
    # returns the length of each dimension and the position of 0 in each dimension
    reg_stride = [1] * NDIM 
    zero_pos = [0] * NDIM
    for i, s in enumerate(stride):
        if s == 0:
            continue
        idxs = [ p[i] for p in points]
        left_end = min(idxs)
        right_end = max(idxs)
        zero_pos[i] = -left_end
        reg_stride[i] = right_end - left_end + 1

    return reg_stride, zero_pos


def take_leading_zeros(text):
    if text.startswith('0') and len(text) > 1:
        return ['0'] + take_leading_zeros(text[1:])
    return [text]


def get_point_from_chunk(chunk, strides):
    # split along pm
    parts = list(filter(None, re.split(r'([pm])', chunk)))
    for i in range(len(parts)-1):
        if parts[i] == 'm':
            parts[i+1] = '-' + parts[i+1] 
    parts = list(filter(lambda x : x not in 'mp', parts))
    # take leading zeros as 0
    parts = sum([take_leading_zeros(p) for p in parts], [])
    # get zeros from end to match stride
    if len(parts) != len(strides):
        for i in range(-1, -len(strides)-1, -1):
            if strides[i] == 0 and parts[i] != '0':
                assert len(parts[i]) > 1 and parts[i].endswith('0')
                end = parts[i+1:] if i + 1 < 0 else []
                parts = parts[:i] + [parts[i][:-1], '0'] + end
    return [int(p) for p in parts]


def get_points_for_stencil(stencil, strides):
    # accepted format: ..._(([mp]d)+_)*STRID{NDIM}_...
    # NDIM = [match.group(1) for match in re.finditer(r'STRID(\d)D_.', stencil)][0]
    parts = list(filter(lambda x: re.fullmatch(r'[mp\d]+', x), stencil.lower().split('_')))
    if len(parts):
        return [get_point_from_chunk(part, strides) for part in parts]
    else:
        return [[0] * len(strides)]


def clean_type(arg):
    for qual in ['__restrict__', 'RESTRICT', '__volatile__']:
        arg = arg.replace(qual, '')
    return arg


def get_idxs(arg_list_with_adjoints, orig_arg_list):
    idxs = [0] * len(arg_list_with_adjoints)
    for i in range(0, len(idxs)):
        if arg_list_with_adjoints[i][-4:] == "_a1s":
            idxs[i] = -orig_arg_list.index(arg_list_with_adjoints[i][0:-4]) - 1
        else:
            idxs[i] = orig_arg_list.index(arg_list_with_adjoints[i])
    return idxs


def get_ad_idx(n, arg_idxs_with_adjoints):
    n_ad = -n - 1
    try:
        return arg_idxs_with_adjoints.index(n_ad)
    except ValueError:
        return -1


def group_n_per_line(vals, n_per_line=4, sep=','):
    return (sep + '\n').join([
        sep.join([vals[i] for i in range(s, min(len(vals), s + n_per_line))])
        for s in range(0, len(vals), n_per_line)
    ])


def generate_cuda_kernel(arg_typ, name, nargs, accs, typs, dims, NDIM, stride,
                         restrict, any_prolong, prolong, MULTI_GRID, arg_idx,
                         n_per_line, nk, soa_set, src_dir):  # reduction,
    OPS_READ = 1
    OPS_WRITE = 2
    OPS_RW = 3
    OPS_INC = 4
    OPS_MAX = 5
    OPS_MIN = 6

    accsstring = [
        'OPS_READ', 'OPS_WRITE', 'OPS_RW', 'OPS_INC', 'OPS_MAX', 'OPS_MIN'
    ]
    use_atomic_reduction = True  #FIXME move param to some other place

    kernel_text, arg_list = find_kernel(src_dir, name)
    kernel_text = kernel_text[kernel_text.find('{') + 1:-1]

    comm("forward kernel")
    code('__global__ void ops_' + name + '(')
    for n, (arg_type, typ, dim,
            acc) in enumerate(zip(arg_typ, typs, dims, accs)):
        if arg_type == 'ops_arg_dat':
            code(f'{typ}* __restrict arg{n},')
            if acc != OPS_READ:
                code(f'{typ}* __restrict arg{n}_ops_cp,')
        elif arg_type == 'ops_arg_gbl':
            if acc == OPS_READ:
                if dim.isdigit() and int(dim) == 1:
                    code(f'const {typ} arg{n},')
                else:
                    code(f'const {typ}* __restrict arg{n},')
            else:
                code(f'{typ} * __restrict arg{n},')
        elif arg_type == 'ops_arg_scalar':
            code(f'const {typ}* __restrict arg{n},')

        if restrict[n] or prolong[n]:
            code(''.join([f'int stride_{n}{i},' for i in range(NDIM)]))

        elif arg_type == 'ops_arg_idx':
            code(''.join([f'int arg_idx{i},' for i in range(NDIM)]))

    if any_prolong:
        code(''.join([f'int global_idx{i},' for i in range(NDIM)]))
    # iteration range sizes
    code(','.join([f'int size{i}' for i in range(NDIM)]) + ') {')

    config.depth = config.depth + 2

    #local variable to hold reductions on GPU
    code('')
    for n in range(0, nargs):
        if arg_typ[n] == 'ops_arg_gbl' and accs[n] != OPS_READ:
            code(typs[n] + ' ' + arg_list[n] + '[' + str(dims[n]) + '];')

    # set local variables to 0 if OPS_INC, INF if OPS_MIN, -INF
    for n in range(0, nargs):
        if arg_typ[n] == 'ops_arg_gbl' and accs[n] == OPS_INC:
            code('for (int d=0; d<' + str(dims[n]) + '; d++) ' + arg_list[n] +
                 '[d] = ZERO_' + typs[n] + ';')
        if arg_typ[n] == 'ops_arg_gbl' and accs[n] == OPS_MIN:
            code('for (int d=0; d<' + str(dims[n]) + '; d++) ' + arg_list[n] +
                 '[d] = INFINITY_' + typs[n] + ';')
        if arg_typ[n] == 'ops_arg_gbl' and accs[n] == OPS_MAX:
            code('for (int d=0; d<' + str(dims[n]) + '; d++) ' + arg_list[n] +
                 '[d] = -INFINITY_' + typs[n] + ';')

    code('')
    if NDIM == 3:
        code('int idx_z = blockDim.z * blockIdx.z + threadIdx.z;')
        code('int idx_y = blockDim.y * blockIdx.y + threadIdx.y;')
    if NDIM == 2:
        code('int idx_y = blockDim.y * blockIdx.y + threadIdx.y;')
    code('int idx_x = blockDim.x * blockIdx.x + threadIdx.x;')
    code('')
    if arg_idx:
        code('int arg_idx[' + str(NDIM) + '];')
        code('arg_idx[0] = arg_idx0+idx_x;')
        if NDIM == 2:
            code('arg_idx[1] = arg_idx1+idx_y;')
        if NDIM == 3:
            code('arg_idx[1] = arg_idx1+idx_y;')
            code('arg_idx[2] = arg_idx2+idx_z;')

    for n in range(0, nargs):
        if arg_typ[n] == 'ops_arg_dat':
            if restrict[n] == 1:
                n_x = 'idx_x*stride_' + str(n) + '0'
                n_y = 'idx_y*stride_' + str(n) + '1'
                n_z = 'idx_z*stride_' + str(n) + '2'
            elif prolong[n] == 1:
                n_x = '(idx_x+global_idx0%stride_' + str(
                    n) + '0)/stride_' + str(n) + '0'
                n_y = '(idx_y+global_idx1%stride_' + str(
                    n) + '1)/stride_' + str(n) + '1'
                n_z = '(idx_z+global_idx2%stride_' + str(
                    n) + '2)/stride_' + str(n) + '2'
            else:
                n_x = 'idx_x'
                n_y = 'idx_y'
                n_z = 'idx_z'

            if NDIM == 1:
                if soa_set:
                    code('arg' + str(n) + ' += ' + n_x + ' * ' +
                         str(stride[NDIM * n]) + ';')
                else:
                    code('arg' + str(n) + ' += ' + n_x + ' * ' +
                         str(stride[NDIM * n]) + '*' + str(dims[n]) + ';')
            elif NDIM == 2:
                if soa_set:
                    code('arg' + str(n) + ' += ' + n_x + ' * ' +
                         str(stride[NDIM * n]) + ' + ' + n_y + ' * ' +
                         str(stride[NDIM * n + 1]) + ' * dims_' + name + '[' +
                         str(n) + '][0]' + ';')
                else:
                    code('arg' + str(n) + ' += ' + n_x + ' * ' +
                         str(stride[NDIM * n]) + '*' + str(dims[n]) + ' + ' +
                         n_y + ' * ' + str(stride[NDIM * n + 1]) + '*' +
                         str(dims[n]) + ' * dims_' + name + '[' + str(n) +
                         '][0]' + ';')
            elif NDIM == 3:
                if soa_set:
                    code('arg' + str(n) + ' += ' + n_x + ' * ' +
                         str(stride[NDIM * n]) + '+ ' + n_y + ' * ' +
                         str(stride[NDIM * n + 1]) + '* dims_' + name + '[' +
                         str(n) + '][0]' + ' + ' + n_z + ' * ' +
                         str(stride[NDIM * n + 2]) + ' * dims_' + name + '[' +
                         str(n) + '][0]' + ' * dims_' + name + '[' + str(n) +
                         '][1]' + ';')
                else:
                    code('arg' + str(n) + ' += ' + n_x + ' * ' +
                         str(stride[NDIM * n]) + '*' + str(dims[n]) + ' + ' +
                         n_y + ' * ' + str(stride[NDIM * n + 1]) + '*' +
                         str(dims[n]) + ' * dims_' + name + '[' + str(n) +
                         '][0]' + ' + ' + n_z + ' * ' +
                         str(stride[NDIM * n + 2]) + '*' + str(dims[n]) +
                         ' * dims_' + name + '[' + str(n) + '][0]' +
                         ' * dims_' + name + '[' + str(n) + '][1]' + ';')
            # Shift Checkpoints
            if accs[n] != OPS_READ:
                IF('arg{}_ops_cp != nullptr'.format(n))
                if NDIM == 1:
                    if soa_set:
                        code('arg' + str(n) + '_ops_cp += ' + n_x + ' * ' +
                             str(stride[NDIM * n]) + ';')
                    else:
                        code('arg' + str(n) + '_ops_cp += ' + n_x + ' * ' +
                             str(stride[NDIM * n]) + '*' + str(dims[n]) + ';')
                elif NDIM == 2:
                    if soa_set:
                        code('arg' + str(n) + '_ops_cp += ' + n_x + ' * ' +
                             str(stride[NDIM * n]) + ' + ' + n_y + ' * ' +
                             str(stride[NDIM * n + 1]) + ' * size0;')
                    else:
                        code('arg' + str(n) + '_ops_cp += ' + n_x + ' * ' +
                             str(stride[NDIM * n]) + '*' + str(dims[n]) +
                             ' + ' + n_y + ' * ' + str(stride[NDIM * n + 1]) +
                             '*' + str(dims[n]) + ' * size0;')
                elif NDIM == 3:
                    if soa_set:
                        code('arg' + str(n) + '_ops_cp += ' + n_x + ' * ' +
                             str(stride[NDIM * n]) + '+ ' + n_y + ' * ' +
                             str(stride[NDIM * n + 1]) + '* size0 + ' + n_z +
                             ' * ' + str(stride[NDIM * n + 2]) +
                             ' * size0 * size1;')
                    else:
                        code('arg' + str(n) + '_ops_cp += ' + n_x + ' * ' +
                             str(stride[NDIM * n]) + '*' + str(dims[n]) +
                             ' + ' + n_y + ' * ' + str(stride[NDIM * n + 1]) +
                             '*' + str(dims[n]) + ' * size0 + ' + n_z + ' * ' +
                             str(stride[NDIM * n + 2]) + '*' + str(dims[n]) +
                             ' * size0 * size1;')
                ENDIF()

    code('')
    n_per_line = 5
    if NDIM == 1:
        IF('idx_x < size0')
    if NDIM == 2:
        IF('idx_x < size0 && idx_y < size1')
    elif NDIM == 3:
        IF('idx_x < size0 && idx_y < size1 && idx_z < size2')
    for n in range(0, nargs):
        if arg_typ[n] == 'ops_arg_dat':
            dim = ''
            sizelist = ''
            pre = ''
            extradim = 0
            if dims[n].isdigit() and int(dims[n]) > 1:
                dim = dims[n] + ', '
                extradim = 1
            elif not dims[n].isdigit():
                dim = 'arg' + str(n) + '.dim, '
                extradim = 1
            for i in range(1, NDIM + extradim):
                sizelist = sizelist + 'dims_' + name + '[' + str(
                    n) + '][' + str(i - 1) + '], '
            if accs[n] == OPS_READ:
                pre = 'const '

            code(pre + 'ACC<' + typs[n] + '> ' + arg_list[n] + '(' + dim +
                 sizelist + 'arg' + str(n) + ');')
            if accs[n] != OPS_READ:
                cp_sizelist = ','.join(
                    ['size' + str(i)
                     for i in range(0, NDIM + extradim - 1)]) + ','
                code(pre + 'ACC<' + typs[n] + '> argp' + str(n) + '_ops_cp(' +
                     dim + cp_sizelist + 'arg' + str(n) + '_ops_cp);')

    # create checkpoints
    for n in range(0, nargs):
        if arg_typ[n] == 'ops_arg_dat' and accs[n] != OPS_READ:
            IF('arg{}_ops_cp != nullptr'.format(n))
            if dims[n].isdigit() and int(dims[n]) <= 1:
                code('argp' + str(n) + '_ops_cp( 0' + (',0' * (NDIM - 1)) +
                     ') = ' + arg_list[n] + '(0' + (',0' * (NDIM - 1)) + ');')
            else:
                FOR(
                    'd', '0', 'arg' + str(n) +
                    '.dim' if not dims[n].isdigit() else dims[n])
                code('argp' + str(n) + '_ops_cp( d, 0' + (',0' * (NDIM - 1)) +
                     ') = ' + arg_list[n] + '(d, 0' + (',0' *
                                                       (NDIM - 1)) + ');')
                ENDFOR()
            ENDIF()

    #function inline params
    for n, (arg_type, typ, dim, acc,
            arg) in enumerate(zip(arg_typ, typs, dims, accs, arg_list)):
        if arg_type == 'ops_arg_gbl':
            if acc == OPS_READ:
                if dim.isdigit() and int(dim) == 1:
                    code(
                        f'const {typ} * __restrict__ {clean_type(arg)} = &arg{n};'
                    )
                else:
                    code(
                        f'const {typ} * __restrict__ {clean_type(arg)} = arg{n};'
                    )
        elif arg_type == 'ops_arg_scalar':
            code(f'const {typ} *__restrict {clean_type(arg)} = arg{n};')
        elif arg_type == 'ops_arg_idx':
            code(f'int *{clean_type(arg)} = arg_idx;')

    #insert user kernel
    code(kernel_text)

    ENDIF()

    #reduction across blocks
    if NDIM == 1:
        cont = '(blockIdx.x + blockIdx.y*gridDim.x)*'
    if NDIM == 2:
        cont = '(blockIdx.x + blockIdx.y*gridDim.x)*'
    elif NDIM == 3:
        cont = '(blockIdx.x + blockIdx.y*gridDim.x + blockIdx.z*gridDim.x*gridDim.y)*'
    for n, (arg_type, dim, acc) in enumerate(zip(arg_typ, dims, accs)):
        if arg_type == 'ops_arg_gbl' and acc != OPS_READ:
            FOR('ops_arg_d', '0', str(dim))
            if use_atomic_reduction:
                code(
                    f'ops_reduction_cuda_atomic<{accsstring[acc-1]}>(arg{n} + ops_arg_d,{arg_list[n]}[ops_arg_d]);'
                )
            else:
                code(
                    f'ops_reduction_cuda<{accsstring[acc-1]}>(arg{n} + ops_arg_d+{cont}{dim},{arg_list[n]}[ops_arg_d]);'
                )
            ENDFOR()

    code('')
    config.depth = config.depth - 2
    code('}')

    ##########################################################################
    #  now host stub
    ##########################################################################
    code('')
    comm(' host stub function')
    code('void ops_par_loop_' + name +
         '_execute(ops_kernel_descriptor *desc) {')
    config.depth = 2
    code('int dim = desc->dim;')
    code('int *range = desc->range;')
    code('ops_block block = desc->block;')

    for n in range(0, nargs):
        code('ops_arg arg' + str(n) + ' = desc->args[' + str(n) + '];')

    code('')
    comm('Timing')
    code('double t1,t2,c1,c2;')
    code('')

    text = 'ops_arg args[' + str(nargs) + '] = {'
    text += ','.join(['arg' + str(i) for i in range(0, nargs)]) + "};\n"
    code(text)
    code('')

    IF('block->instance->OPS_diags > 1')
    code('ops_timing_realloc(block->instance,' + str(nk) + ',"' + name + '");')
    code('block->instance->OPS_kernels[' + str(nk) + '].count++;')
    code('ops_timers_core(&c1,&t1);')
    ENDIF()

    code('')
    comm('compute locally allocated range for the sub-block')
    code('int start[' + str(NDIM) + '];')
    code('int end[' + str(NDIM) + '];')

    code('')
    if arg_idx:
        code('int arg_idx[' + str(NDIM) + '];')

    FOR('n', '0', str(NDIM))
    code('start[n] = range[2*n];end[n] = range[2*n+1];')
    if arg_idx:
        code('arg_idx[n] = start[n];')
    ENDFOR()

    #  if MULTI_GRID:
    #    code('int global_idx['+str(NDIM)+'];')
    #    code('#ifdef OPS_MPI')
    #    for n in range (0,NDIM):
    #      code('global_idx['+str(n)+'] = arg_idx['+str(n)+'];')
    #    code('#else')
    #    for n in range (0,NDIM):
    #      code('global_idx['+str(n)+'] = start['+str(n)+'];')
    #    code('#endif')
    #    code('')

    for n in range(0, nargs):
        if arg_typ[n] == 'ops_arg_dat':
            code('int xdim' + str(n) + ' = args[' + str(n) + '].dat->size[0];')
            if NDIM > 2 or (NDIM == 2 and soa_set):
                code('int ydim' + str(n) + ' = args[' + str(n) +
                     '].dat->size[1];')
            if NDIM > 3 or (NDIM == 3 and soa_set):
                code('int zdim' + str(n) + ' = args[' + str(n) +
                     '].dat->size[2];')
    code('')

    condition = ''
    for n in range(0, nargs):
        if arg_typ[n] == 'ops_arg_dat':
            condition = condition + 'xdim' + str(
                n) + ' != dims_' + name + '_h[' + str(n) + '][0] || '
            if NDIM > 2 or (NDIM == 2 and soa_set):
                condition = condition + 'ydim' + str(
                    n) + ' != dims_' + name + '_h[' + str(n) + '][1] || '
            if NDIM > 3 or (NDIM == 3 and soa_set):
                condition = condition + 'zdim' + str(
                    n) + ' != dims_' + name + '_h[' + str(n) + '][2] || '
    condition = condition[:-4]
    IF(condition)

    #    for n in range (0, nargs):
    #      if arg_typ[n] == 'ops_arg_dat':
    #        code('cudaMemcpyToSymbol( dims_'+name+'['+str(n)+'][0]'+', &xdim'+str(n)+', sizeof(int) );')
    #        code('dims_'+name+'_h['+str(n)+'][0] = xdim'+str(n)+';')
    #        if NDIM>2 or (NDIM==2 and soa_set):
    #          code('cudaMemcpyToSymbol( dims_'+name+'['+str(n)+'][1]'+', &ydim'+str(n)+', sizeof(int) );')
    #          code('dims_'+name+'_h['+str(n)+'][1] = ydim'+str(n)+';')
    #        if NDIM>3 or (NDIM==3 and soa_set):
    #          code('cudaMemcpyToSymbol( dims_'+name+'['+str(n)+'][2]'+', &zdim'+str(n)+', sizeof(int) );')
    #          code('dims_'+name+'_h['+str(n)+'][2] = zdim'+str(n)+';')
    for n in range(0, nargs):
        if arg_typ[n] == 'ops_arg_dat':
            code('dims_' + name + '_h[' + str(n) + '][0] = xdim' + str(n) +
                 ';')
            if NDIM > 2 or (NDIM == 2 and soa_set):
                code('dims_' + name + '_h[' + str(n) + '][1] = ydim' + str(n) +
                     ';')
            if NDIM > 3 or (NDIM == 3 and soa_set):
                code('dims_' + name + '_h[' + str(n) + '][2] = zdim' + str(n) +
                     ';')
    code('cutilSafeCall(block->instance->ostream(),cudaMemcpyToSymbol( dims_' +
         name + ', dims_' + name + '_h, sizeof(dims_' + name + ')));')
    ENDIF()

    code('')

    #setup reduction variables
    for n, (arg_type, typ, acc,
            dim) in enumerate(zip(arg_typ, typs, accs, dims)):
        if arg_type == 'ops_arg_gbl':
            if acc == OPS_READ and (not dim.isdigit() or int(dim) > 1):
                code(f'{typ} *arg{n}h = ({typ} *)arg{n}.data;')
            elif acc != OPS_READ:
                #  code('#ifdef OPS_MPI')
                #  code(f'{typ} *arg{n}h = ({typ} *)(((ops_reduction)args[{n}].data)->data + ((ops_reduction)args[{n}].data)->size * block->index);')
                #  code('#else')
                code(
                    f'{typ} *arg{n}h = ({typ} *)(((ops_reduction)args[{n}].data)->data);'
                )
                #  code('#endif')

    code('')
    code('int x_size = MAX(0,end[0]-start[0]);')
    if NDIM == 2:
        code('int y_size = MAX(0,end[1]-start[1]);')
    if NDIM == 3:
        code('int y_size = MAX(0,end[1]-start[1]);')
        code('int z_size = MAX(0,end[2]-start[2]);')
    code('')

    #set up CUDA grid and thread blocks for kernel call
    if NDIM == 1:
        code(
            'dim3 grid( (x_size-1)/block->instance->OPS_block_size_x+ 1, 1, 1);'
        )
    if NDIM == 2:
        code(
            'dim3 grid( (x_size-1)/block->instance->OPS_block_size_x+ 1, (y_size-1)/block->instance->OPS_block_size_y + 1, 1);'
        )
    if NDIM == 3:
        code(
            'dim3 grid( (x_size-1)/block->instance->OPS_block_size_x+ 1, (y_size-1)/block->instance->OPS_block_size_y + 1, (z_size-1)/block->instance->OPS_block_size_z +1);'
        )

    if NDIM > 1:
        code(
            'dim3 tblock(block->instance->OPS_block_size_x,block->instance->OPS_block_size_y,block->instance->OPS_block_size_z);'
        )
    else:
        code('dim3 tblock(block->instance->OPS_block_size_x,1,1);')

    code('')

    GBL_READ = False
    GBL_READ_MDIM = False
    GBL_INC = False
    GBL_MAX = False
    GBL_MIN = False
    GBL_WRITE = False

    #set up reduction variables
    for n in range(0, nargs):
        if arg_typ[n] == 'ops_arg_gbl':
            if accs[n] == OPS_READ:
                GBL_READ = True
                if not dims[n].isdigit() or int(dims[n]) > 1:
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
        if not use_atomic_reduction:
            if NDIM == 1:
                code(
                    'int nblocks = ((x_size-1)/block->instance->OPS_block_size_x+ 1);'
                )
            elif NDIM == 2:
                code(
                    'int nblocks = ((x_size-1)/block->instance->OPS_block_size_x+ 1)*((y_size-1)/block->instance->OPS_block_size_y + 1);'
                )
            elif NDIM == 3:
                code(
                    'int nblocks = ((x_size-1)/block->instance->OPS_block_size_x+ 1)*((y_size-1)/block->instance->OPS_block_size_y + 1)*((z_size-1)/block->instance->OPS_block_size_z +1);'
                )
            code('int maxblocks = nblocks;')
        code('int reduct_bytes = 0;')
        code('int reduct_size = 0;')
        code('')

    if GBL_READ == True and GBL_READ_MDIM == True:
        code('int consts_bytes = 0;')
        code('')

    for arg_type, typ, acc, dim in zip(arg_typ, typs, accs, dims):
        if arg_type == 'ops_arg_gbl' and acc == OPS_READ and (
                not dim.isdigit() or int(dim) > 1):
            code(f'consts_bytes += ROUND_UP({dim}*sizeof({typ}));')

    if use_atomic_reduction:
        for arg_type, typ, acc, dim in zip(arg_typ, typs, accs, dims):
            if arg_type == 'ops_arg_gbl' and acc != OPS_READ:
                code(f'reduct_bytes += ROUND_UP({dim}*sizeof({typ}));')
                code(f'reduct_size = MAX(reduct_size,sizeof({typ})*{dim});')
    else:
        for arg_type, typ, acc, dim in zip(arg_typ, typs, accs, dims):
            if arg_type == 'ops_arg_gbl' and acc != OPS_READ:
                code(
                    f'reduct_bytes += ROUND_UP(maxblocks*{dim}*sizeof({typ}));'
                )
                code(f'reduct_size = MAX(reduct_size,sizeof({typ})*{dim});')

    code('')

    if GBL_READ == True and GBL_READ_MDIM == True:
        code('reallocConstArrays(block->instance, consts_bytes);')
        code('consts_bytes = 0;')
    if GBL_INC == True or GBL_MIN == True or GBL_MAX == True or GBL_WRITE == True:
        code('reallocReductArrays(block->instance, reduct_bytes);')
        code('reduct_bytes = 0;')
        code('')

    for n, (arg_type, typ, acc,
            dim) in enumerate(zip(arg_typ, typs, accs, dims)):
        if arg_type == 'ops_arg_gbl' and acc != OPS_READ:
            code(
                f'arg{n}.data = block->instance->OPS_reduct_h + reduct_bytes;')
            code(
                f'arg{n}.data_d = block->instance->OPS_reduct_d + reduct_bytes;'
            )
            if use_atomic_reduction:
                FOR('d', '0', str(dim))
                if acc == OPS_INC:
                    code(f'(({typ} *)arg{n}.data)[d] = ZERO_{typ};')
                if acc == OPS_MAX:
                    code(f'(({typ} *)arg{n}.data)[d] = -INFINITY_{typ};')
                if acc == OPS_MIN:
                    code(f'(({typ} *)arg{n}.data)[d] = INFINITY_{typ};')
                ENDFOR()
                code(f'reduct_bytes += ROUND_UP({dim}*sizeof({typ}));')
            else:
                code('for (int b=0; b<maxblocks; b++)')
                FOR('d', '0', str(dim))
                if acc == OPS_INC:
                    code(f'(({typ} *)arg{n}.data)[d+b*{dim}] = ZERO_{typ};')
                if acc == OPS_MAX:
                    code(
                        f'(({typ} *)arg{n}.data)[d+b*{dim}] = -INFINITY_{typ};'
                    )
                if acc == OPS_MIN:
                    code(
                        f'(({typ} *)arg{n}.data)[d+b*{dim}] = INFINITY_{typ};')
                ENDFOR()
                code(
                    f'reduct_bytes += ROUND_UP(maxblocks*{dim}*sizeof({typ}));'
                )
            code('')

    code('')

    for n, (arg_type, typ, acc,
            dim) in enumerate(zip(arg_typ, typs, accs, dims)):
        if arg_type == 'ops_arg_gbl' and acc == OPS_READ and (
                not dim.isdigit() or int(dim) > 1):
            code(
                f'arg{n}.data = block->instance->OPS_consts_h + consts_bytes;')
            code(
                f'arg{n}.data_d = block->instance->OPS_consts_d + consts_bytes;'
            )
            FOR('d', '0', str(dim))
            code(f'(({typ} *)arg{n}.data)[d] = arg{n}h[d];')
            ENDFOR()
            code(f'consts_bytes += ROUND_UP({dim}*sizeof({typ}));')

    if GBL_READ == True and GBL_READ_MDIM == True:
        code('mvConstArraysToDevice(block->instance, consts_bytes);')

    if GBL_INC == True or GBL_MIN == True or GBL_MAX == True or GBL_WRITE == True:
        code('mvReductArraysToDevice(block->instance, reduct_bytes);')

    for n in range(0, nargs):
        if arg_typ[n] == 'ops_arg_dat':
            code('int dat' + str(n) +
                 ' = (block->instance->OPS_soa ? sizeof(' + typs[n] +
                 ') : args[' + str(n) + '].dat->elem_size);')

    code('')
    code('char *p_a[' + str(nargs) + '];')

    #some custom logic for multigrid
    if MULTI_GRID:
        for n, (prol, restr) in enumerate(zip(prolong, restrict)):
            if prol == 1 or restr == 1:
                comm(
                    'This arg has a prolong stencil - so create different ranges'
                )
                code(
                    f'int start_{n}[{NDIM}]; int end_{n}[{NDIM}]; int stride_{n}[{NDIM}];int d_size_{n}[{NDIM}];'
                )
                code('#ifdef OPS_MPI')
                FOR('n', '0', str(NDIM))
                code(
                    f'sub_dat *sd{n} = OPS_sub_dat_list[args[{n}].dat->index];'
                )
                code(f'stride_{n}[n] = args[{n}].stencil->mgrid_stride[n];')
                code(
                    f'd_size_{n}[n] = args[{n}].dat->d_m[n] + sd{n}->decomp_size[n] - args[{n}].dat->d_p[n];'
                )
                if restr == 1:
                    code(
                        f'start_{n}[n] = global_idx[n]*stride_{n}[n] - sd{n}->decomp_disp[n] + args[{n}].dat->d_m[n];'
                    )
                else:
                    code(
                        f'start_{n}[n] = global_idx[n]/stride_{n}[n] - sd{n}->decomp_disp[n] + args[{n}].dat->d_m[n];'
                    )
                code(f'end_{n}[n] = start_{n}[n] + d_size_{n}[n];')
                ENDFOR()
                code('#else')
                FOR('n', '0', str(NDIM))
                code(f'stride_{n}[n] = args[{n}].stencil->mgrid_stride[n];')
                code(
                    f'd_size_{n}[n] = args[{n}].dat->d_m[n] + args[{n}].dat->size[n] - args[{n}].dat->d_p[n];'
                )
                if restr == 1:
                    code(f'start_{n}[n] = global_idx[n]*stride_{n}[n];')
                else:
                    code(f'start_{n}[n] = global_idx[n]/stride_{n}[n];')
                code(f'end_{n}[n] = start_{n}[n] + d_size_{n}[n];')
                ENDFOR()
                code('#endif')

    comm('')
    comm('set up initial pointers')
    for n in range(0, nargs):
        if arg_typ[n] == 'ops_arg_dat':
            if prolong[n] == 1 or restrict[n] == 1:
                starttext = 'start_' + str(n)
            else:
                starttext = 'start'
            code('int base' + str(n) + ' = args[' + str(n) +
                 '].dat->base_offset + ')
            code('         dat' + str(n) + ' * 1 * (' + starttext +
                 '[0] * args[' + str(n) + '].stencil->stride[0]);')
            for d in range(1, NDIM):
                line = 'base' + str(n) + ' = base' + str(n) + '+ dat' + str(
                    n) + ' *\n'
                for d2 in range(0, d):
                    line = line + config.depth * ' ' + '  args[' + str(
                        n) + '].dat->size[' + str(d2) + '] *\n'
                code(line[:-1])
                code('  (' + starttext + '[' + str(d) + '] * args[' + str(n) +
                     '].stencil->stride[' + str(d) + ']);')

            code('p_a[' + str(n) + '] = (char *)args[' + str(n) +
                 '].data_d + base' + str(n) + ';')
            code('')
            if accs[n] != OPS_READ:
                code("%s *arg%d_cp = nullptr;" % (typs[n], n))
                IF("block->instance->ad_instance && block->instance->ad_instance->get_current_tape()->dagState == OPSDagState::FORWARD_SAVE"
                   )
                cp_size = '*'.join(
                    ['%s_size' % (['x', 'y', 'z'][i]) for i in range(NDIM)])
                code(
                    "arg{0}_cp = ({1} *) ops_alloc_cp(arg{0}.dat, {2} * arg{0}.dat->elem_size);"
                    .format(n, typs[n], cp_size))
                ENDIF()

    #halo exchange
    code('')
    code('#ifndef OPS_LAZY')
    code('ops_H_D_exchanges_device(args, ' + str(nargs) + ');')
    code('ops_halo_exchanges(args,' + str(nargs) + ',range);')
    code('#endif')
    code('')
    IF('block->instance->OPS_diags > 1')
    code('ops_timers_core(&c2,&t2);')
    code('block->instance->OPS_kernels[' + str(nk) + '].mpi_time += t2-t1;')
    ENDIF()
    code('')

    #set up shared memory for reduction
    if GBL_INC == True or GBL_MIN == True or GBL_MAX == True or GBL_WRITE == True:
        code('int nshared = 0;')
        code(
            'int nthread = block->instance->OPS_block_size_x*block->instance->OPS_block_size_y*block->instance->OPS_block_size_z;'
        )
        code('')
    for n, (arg_type, typ, dim,
            acc) in enumerate(zip(arg_typ, typs, dims, accs)):
        if arg_type == 'ops_arg_gbl' and acc != OPS_READ:
            code(f'nshared = MAX(nshared,sizeof({typ})*{dim});')
    code('')
    if GBL_INC == True or GBL_MIN == True or GBL_MAX == True or GBL_WRITE == True:
        code('nshared = MAX(nshared*nthread,reduct_size*nthread);')
        code('')


    #kernel call
    comm('call kernel wrapper function, passing in pointers to data')
    if NDIM == 1:
        code('if (x_size > 0)')
    if NDIM == 2:
        code('if (x_size > 0 && y_size > 0)')
    if NDIM == 3:
        code('if (x_size > 0 && y_size > 0 && z_size > 0)')
    config.depth = config.depth + 2
    n_per_line = 2
    if GBL_INC == True or GBL_MIN == True or GBL_MAX == True or GBL_WRITE == True:
        text = 'ops_' + name + '<<<grid, tblock, nshared >>> ( '
    else:
        text = 'ops_' + name + '<<<grid, tblock >>> ( '

    param_vals = []
    for n, (arg_type, typ, dim,
            acc) in enumerate(zip(arg_typ, typs, dims, accs)):
        if arg_type == 'ops_arg_dat':
            param_vals.append(f' ({typ} *)p_a[{n}]')
            if acc != OPS_READ:
                param_vals.append(f' arg{n}_cp')
        elif arg_type == 'ops_arg_gbl':
            if dim.isdigit() and int(dim) == 1 and acc == OPS_READ:
                param_vals.append(f' *({typ} *)arg{n}.data')
            else:
                param_vals.append(f' ({typ} *)arg{n}.data_d')
        elif arg_type == 'ops_arg_idx':
            if NDIM == 1:
                param_vals.append(' arg_idx[0]')
            if NDIM == 2:
                param_vals.append(' arg_idx[0], arg_idx[1]')
            elif NDIM == 3:
                param_vals.append(' arg_idx[0], arg_idx[1], arg_idx[2]')
        elif arg_type == 'ops_arg_scalar':
            param_vals.append(f' ({typ} *)args[{n}].data_d')
        if restrict[n] or prolong[n]:
            if NDIM == 1:
                param_vals.append(f'stride_{n}[0]')
            if NDIM == 2:
                param_vals.append(f'stride_{n}[0],stride_{n}[1]')
            if NDIM == 3:
                param_vals.append(f'stride_{n}[0],stride_{n}[1],stride_{n}[2]')

    text += group_n_per_line(param_vals, n_per_line) + ','
    if any_prolong:
        if NDIM == 1:
            text = text + 'global_idx[0],'
        elif NDIM == 2:
            text = text + 'global_idx[0], global_idx[1],'
        elif NDIM == 3:
            text = text + 'global_idx[0], global_idx[1], global_idx[2],'

    if NDIM == 1:
        text = text + 'x_size);'
    if NDIM == 2:
        text = text + 'x_size, y_size);'
    elif NDIM == 3:
        text = text + 'x_size, y_size, z_size);'
    code(text)
    config.depth = config.depth - 2

    code('')
    code('cutilSafeCall(block->instance->ostream(),cudaGetLastError());')
    code('')

    #
    # Complete Reduction Operation by moving data onto host
    # and reducing over blocks
    #
    if GBL_INC == True or GBL_MIN == True or GBL_MAX == True or GBL_WRITE == True:
        code('mvReductArraysToHost(block->instance, reduct_bytes);')

    for n, (arg_type, typ, dim,
            acc) in enumerate(zip(arg_typ, typs, dims, accs)):
        if arg_type == 'ops_arg_gbl' and acc != OPS_READ:
            if not use_atomic_reduction:
                code('for(int b = 0; b < maxblocks; ++b)')
            FOR('d', '0', str(dim))
            if use_atomic_reduction:
                if acc == OPS_INC:
                    code(f'arg{n}h[d] += (({typ} *)arg{n}.data)[d];')
                elif acc == OPS_MAX:
                    code(
                        f'arg{n}h[d] = MAX(arg{n}h[d],(({typ} *)arg{n}.data)[d]);'
                    )
                else:
                    code(
                        f'arg{n}h[d] = MIN(arg{n}h[d],(({typ} *)arg{n}.data)[d]);'
                    )
            else:
                if acc == OPS_INC:
                    code(f'arg{n}h[d] += (({typ} *)arg{n}.data)[d+b*{dim}];')
                elif acc == OPS_MAX:
                    code(
                        f'arg{n}h[d] = MAX(arg{n}h[d],(({typ} *)arg{n}.data)[d+b*{dim}]);'
                    )
                elif acc == OPS_MIN:
                    code(
                        f'arg{n}h[d] = MIN(arg{n}h[d],(({typ} *)arg{n}.data)[d+b*{dim}]);'
                    )
            ENDFOR()
            code(f'((ops_reduction)args[{n}].data)->data = (char *)arg{n}h;')
        if (
            arg_type == "ops_arg_gbl"
            and acc == OPS_READ
            and (not dim.isdigit() or int(dim) > 1)
        ):
            code(f"arg{n}.data = (char *)arg{n}h;")

    IF('block->instance->OPS_diags>1')
    code('cutilSafeCall(block->instance->ostream(),cudaDeviceSynchronize());')
    code('ops_timers_core(&c1,&t1);')
    code('block->instance->OPS_kernels[' + str(nk) + '].time += t1-t2;')
    ENDIF()
    code('')

    code('ops_set_dirtybit_device(args, ' + str(nargs) + ');')
    for n in range(0, nargs):
        if arg_typ[n] == 'ops_arg_dat' and (accs[n] == OPS_WRITE or accs[n]
                                            == OPS_RW or accs[n] == OPS_INC):
            code('ops_set_halo_dirtybit3(&args[' + str(n) + '],range);')

    code('')
    IF('block->instance->OPS_diags > 1')
    comm('Update kernel record')
    code('ops_timers_core(&c2,&t2);')
    code('block->instance->OPS_kernels[' + str(nk) + '].mpi_time += t2-t1;')
    for n in range(0, nargs):
        if arg_typ[n] == 'ops_arg_dat':
            code(
                'block->instance->OPS_kernels[{}].transfer += ops_compute_transfer_2(block->instance, dim, start, end, &arg{});'
                .format(nk, n))
    ENDIF()
    config.depth = config.depth - 2
    code('}')
    code('')


def generate_adjoint_cuda_kernel(arg_typ, name, nargs, accs, typs, dims, NDIM,
                                 stride, stens, restrict, any_prolong, prolong,
                                 MULTI_GRID, arg_idx, n_per_line, nk, soa_set,
                                 src_dir, consts):  # reduction,
    OPS_READ = 1
    OPS_WRITE = 2
    OPS_RW = 3
    OPS_INC = 4
    OPS_MAX = 5
    OPS_MIN = 6

    accsstring = [
        'OPS_READ', 'OPS_WRITE', 'OPS_RW', 'OPS_INC', 'OPS_MAX', 'OPS_MIN'
    ]

    dim_labels = ['x', 'y', 'z']

    adjoint_name = name + "_adjoint"
    kernel_text, arg_list_with_adjoints = find_kernel_with_retry(
        src_dir, adjoint_name)
    orig_kernel_func, arg_list = find_kernel(src_dir, name)
    if kernel_text is None:
        if verbose:
            print(
                f"Couldn't find adjoint for kernel for {name}. Generate as passive loop."
            )
        return None
    kernel_text = kernel_text[kernel_text.find('{') + 1:-1]
    arg_idxs_with_adjoints = get_idxs(
        arg_list_with_adjoints, arg_list
    )  # an array can be indexed with the adjoint arg idx value: indexes from -n-1 to n. negative value -i stands for adjoint of the i-1th argument

    comm("adjoint kernel")
    code('__global__ void ops_' + adjoint_name + '(')
    for n, (arg_type, typ, dim,
            acc) in enumerate(zip(arg_typ, typs, dims, accs)):
        n_ad = get_ad_idx(n, arg_idxs_with_adjoints)
        if arg_type == 'ops_arg_dat':
            code(f'{typ}* __restrict arg{n},')
            if acc != OPS_READ:
                code(f'const {typ}* __restrict arg{n}_ops_cp,')
            if n_ad != -1:
                code(f'{typ} * __restrict arg{n}_a1s,')
        elif arg_type == 'ops_arg_gbl':
            if acc == OPS_READ:
                if dim.isdigit() and int(dim) == 1:
                    code(f'const {typ} arg{n},')
                else:
                    code(f'const {typ}* __restrict arg{n},')
            else:
                # reductions are skipped in adjoints
                if n_ad != -1:
                    if dim.isdigit() and int(dim) == 1:
                        code(f'const {typ} arg{n}_a1s,')
                    else:
                        code(f'const {typ} * __restrict arg{n}_a1s,')
        elif arg_type == 'ops_arg_scalar':
            code(f'const {typ}* __restrict arg{n},')
            if n_ad != -1:
                code(f'{typ} * __restrict arg{n}_a1s,')

        if restrict[n] or prolong[n]:
            code(''.join([f'int stride_{n}{i},' for i in range(NDIM)]))

        elif arg_type == 'ops_arg_idx':
            code(''.join([f'int arg_idx{i},' for i in range(NDIM)]))

    if any_prolong:
        code(''.join([f'int global_idx{i},' for i in range(NDIM)]))
    # iteration range sizes
    code(','.join([f'int size{i}' for i in range(NDIM)]) + ') {')

    config.depth = config.depth + 2

    #local variable to hold reductions on GPU
    code('')
    for n, (arg_type, typ, dim, acc,
            arg) in enumerate(zip(arg_typ, typs, dims, accs, arg_list)):
        if arg_type == 'ops_arg_gbl' and acc != OPS_READ:
            code(f'{typ} {arg}[{dim}];')
            # set local variables to 0 if OPS_INC, INF if OPS_MIN, -INF
            if acc == OPS_INC:
                code(f'for (int d=0; d<{dim}; d++) {arg}[d] = ZERO_{typ};')
            if acc == OPS_MIN:
                code(f'for (int d=0; d<{dim}; d++) {arg}[d] = INFINITY_{typ};')
            if acc == OPS_MAX:
                code(
                    f'for (int d=0; d<{dim}; d++) {arg}[d] = -INFINITY_{typ};')
        elif arg_type == 'ops_arg_scalar' and acc == OPS_READ:
            n_ad = get_ad_idx(n, arg_idxs_with_adjoints)
            if n_ad != -1:
                code(f'{typ} {arg_list_with_adjoints[n_ad]}[{dim}];')
                code(
                    f'for (int d=0; d<{dim}; d++) {arg_list_with_adjoints[n_ad]}[d] = 0.0;'
                )
        elif arg_type == 'ops_arg_dat' and acc != OPS_READ:
            assert (dim.isdigit())
            code(f'{typ} arg{n}_local[{dim}] = {{}};')

    code('')
    if NDIM == 3:
        code('int idx_z = blockDim.z * blockIdx.z + threadIdx.z;')
        code('int idx_y = blockDim.y * blockIdx.y + threadIdx.y;')
    if NDIM == 2:
        code('int idx_y = blockDim.y * blockIdx.y + threadIdx.y;')
    code('int idx_x = blockDim.x * blockIdx.x + threadIdx.x;')
    code('')
    if arg_idx:
        code('int arg_idx[' + str(NDIM) + '];')
        code('arg_idx[0] = arg_idx0+idx_x;')
        if NDIM == 2:
            code('arg_idx[1] = arg_idx1+idx_y;')
        if NDIM == 3:
            code('arg_idx[1] = arg_idx1+idx_y;')
            code('arg_idx[2] = arg_idx2+idx_z;')

    for n in range(0, nargs):
        n_ad = get_ad_idx(n, arg_idxs_with_adjoints)
        if arg_typ[n] == 'ops_arg_dat':
            if restrict[n] == 1:
                n_x = 'idx_x*stride_' + str(n) + '0'
                n_y = 'idx_y*stride_' + str(n) + '1'
                n_z = 'idx_z*stride_' + str(n) + '2'
            elif prolong[n] == 1:
                n_x = '(idx_x+global_idx0%stride_' + str(
                    n) + '0)/stride_' + str(n) + '0'
                n_y = '(idx_y+global_idx1%stride_' + str(
                    n) + '1)/stride_' + str(n) + '1'
                n_z = '(idx_z+global_idx2%stride_' + str(
                    n) + '2)/stride_' + str(n) + '2'
            else:
                n_x = 'idx_x'
                n_y = 'idx_y'
                n_z = 'idx_z'

            shift_idx = ""
            dims_fmt_str = "dims_%s[%d][{}]" % (name, n)
            shift_idx = '+'.join([
                "{} * {} {}".format([n_x, n_y, n_z][dim],
                                    stride[NDIM * n + dim], ''.join([
                                        '* ' + dims_fmt_str.format(d)
                                        for d in range(dim)
                                    ])) for dim in range(NDIM)
            ])
            if not soa_set:
                shift_idx = '({})*{}'.format(shift_idx, dims[n])
            code('arg{} += {};'.format(n, shift_idx))
            if n_ad != -1:
                code('arg{}_a1s += {};'.format(n, shift_idx))

            # Shift Checkpoints
            if accs[n] != OPS_READ:
                cp_shift_idx = ""
                cp_shift_idx = '+'.join([
                    "{} * {} * {} {}".format(
                        [n_x, n_y, n_z][dim], stride[NDIM * n + dim], dims[n],
                        ''.join(['* size{}'.format(d) for d in range(dim)]))
                    for dim in range(NDIM)
                ])
                if not soa_set:
                    cp_shift_idx = '({})*{}'.format(cp_shift_idx, dims[n])
                code('arg{}_ops_cp += {};'.format(n, cp_shift_idx))

    code('')
    # generate registers for lowdim reductions
    # and set up pointers to sencil center
    for n, (arg_type, typ, dim, acc, arg, sten) in enumerate(
        zip(arg_typ, typs, dims, accs, arg_list, stens)
    ):
        n_ad = get_ad_idx(n, arg_idxs_with_adjoints)
        curr_stride = get_stride_n(n, stride, NDIM)
        if (
            n_ad == -1
            or arg_type != "ops_arg_dat"
            or not has_ldim_reduction(curr_stride)
        ):
            continue
        # at this point we found an active lowdim dataset
        points = get_points_for_stencil(sten, curr_stride)
        reg_dims, zero_pos = get_bounding_rectangle_for_stencil(
            points, curr_stride, NDIM
        )
        regnum = reduce(mul, reg_dims)
        if (dim.isdigit() and int(dim) > 1) or not dim.isdigit():
            assert (dim.isdigit())  # FIXME support variables as dim
            regnum *= int(dim)
        # Shift pointer to the position of 0,0 (to zero_pos * cum_stride
        cum_stride = int(dim)
        offset_text = ""
        for s, pos in zip(reg_dims, zero_pos):
            offset_text += f"+ {pos}*{cum_stride} "
            cum_stride *= s
        code(f"{typ} arg{n}_a1s_local[{regnum}] = {{}};")
        code(f"{typ}* arg{n}_a1s_local_ptr = arg{n}_a1s_local {offset_text};")

    n_per_line = 5
    IF('&&'.join([f'idx_{dim_labels[d]} < size{d}' for d in range(NDIM)]))
    for n, (arg_type, typ, dim, acc,
            arg, sten) in enumerate(zip(arg_typ, typs, dims, accs, arg_list, stens)):
        n_ad = get_ad_idx(n, arg_idxs_with_adjoints)
        if arg_type == 'ops_arg_dat':
            sizelist = ', '.join(
                [f'dims_{name}[{n}][{i}]' for i in range(0, NDIM - 1)])
            cp_sizelist = ', '.join([f'size{i}' for i in range(0, NDIM - 1)])
            zeros = ('0,' * (NDIM))[0:-1]
            dummy_size_list = ('0,' * (NDIM - 1))[0:-1]
            pre = ''
            if (dim.isdigit() and int(dim) > 1) or not dim.isdigit():
                assert (dim.isdigit())  # FIXME support variables as dim
                sizelist = f'{dim}, ' + sizelist
                cp_sizelist = f'{dim}, ' + cp_sizelist
                dummy_size_list = f'{dim}, ' + dummy_size_list
                if soa_set:
                    sizelist += f', dims_{name}[{n}][{NDIM - 1}]'
                    cp_sizelist += f', size{NDIM - 1}'
                    dummy_size_list += ', 0'
                else:
                    sizelist += ', 0'
                    cp_sizelist += ', 0'
                    dummy_size_list += ', 0'

            if acc == OPS_READ:
                pre = 'const '
            if n_ad != -1:
                code(
                    f'ACC_A1S<{typ}> {arg_list_with_adjoints[n_ad]}({sizelist}, arg{n}_a1s);'
                )
            if acc == OPS_READ:
                code(f'{pre}ACC<{typ}> {arg}({sizelist}, arg{n});')
            else:
                code(f'{pre}ACC<{typ}> arg{n}_primal({sizelist}, arg{n});')
                code(
                    f'{pre}ACC<{typ}> {arg}({dummy_size_list}, arg{n}_local);')
                code(
                    f'const ACC<const {typ}> argp{n}_ops_cp({cp_sizelist}, arg{n}_ops_cp);'
                )
                # restore checkpoints
                FOR('ops_arg_d', '0', dim)
                if acc != OPS_WRITE:
                    code(f'{arg}(ops_arg_d, {zeros}) = argp{n}_ops_cp(ops_arg_d, {zeros});')
                code(
                    f'arg{n}_primal(ops_arg_d, {zeros}) = argp{n}_ops_cp(ops_arg_d, {zeros});')
                ENDFOR()

    #function inline params
    for n, (arg_type, typ, dim, acc,
            arg) in enumerate(zip(arg_typ, typs, dims, accs, arg_list)):
        n_ad = get_ad_idx(n, arg_idxs_with_adjoints)
        if arg_type == 'ops_arg_gbl':
            if acc == OPS_READ:
                if dim.isdigit() and int(dim) == 1:
                    code(
                        f'const {typ} * __restrict {clean_type(arg)} = &arg{n};'
                    )
                else:
                    code(
                        f'const {typ} * __restrict {clean_type(arg)} = arg{n};'
                    )
            elif n_ad != -1:
                if dim.isdigit() and int(dim) == 1:
                    code(
                        f'const {typ} * __restrict {arg_list_with_adjoints[n_ad]} = &arg{n}_a1s;'
                    )
                else:
                    code(
                        f'const {typ} * __restrict {arg_list_with_adjoints[n_ad]}= arg{n}_a1s;'
                    )
        elif arg_type == 'ops_arg_scalar':
            code(f'const {typ} *__restrict {clean_type(arg)} = arg{n};')
        elif arg_type == 'ops_arg_idx':
            code(f'int *{clean_type(arg)} = arg_idx;')

    #insert user kernel
    code(kernel_text)

    ENDIF()

    # reduction of scalar adjoints across blocks
    for n, (arg_type, typ, dim, acc, sten) in enumerate(
        zip(arg_typ, typs, dims, accs, stens)
    ):
        n_ad = get_ad_idx(n, arg_idxs_with_adjoints)
        if n_ad == -1:
            continue
        if arg_type == "ops_arg_scalar" and acc == OPS_READ:
            FOR("d", "0", str(dim))
            code(
                f"ops_reduction_cuda_atomic<OPS_INC>(arg{n}_a1s + d, {arg_list_with_adjoints[n_ad]}[d]);"
            )
            ENDFOR()

    code('')
    config.depth = config.depth - 2
    code('}')

    ##########################################################################
    #  now host stub
    ##########################################################################
    code('')
    comm(' host stub function')
    code('void ops_par_loop_' + adjoint_name +
         '_execute(ops_kernel_descriptor *desc) {')
    config.depth = 2
    code('int dim = desc->dim;')
    code('int *range = desc->range;')
    code('ops_block block = desc->block;')

    code('')
    comm('Timing')
    code('double t1,t2,c1,c2;')
    code('')

    code('ops_arg *args = desc->args;')
    code('')

    IF('block->instance->OPS_diags > 1')
    code('ops_timing_realloc(block->instance,' + str(nk) + ',"' + name + '");')
    code('block->instance->OPS_kernels[' + str(nk) + '].ad_count++;')
    code('ops_timers_core(&c1,&t1);')
    ENDIF()

    code('')
    comm('compute locally allocated range for the sub-block')
    code('int start[' + str(NDIM) + '];')
    code('int end[' + str(NDIM) + '];')
    if arg_idx:
        code('int arg_idx[' + str(NDIM) + '];')

    FOR('n', '0', str(NDIM))
    code('start[n] = range[2*n];end[n] = range[2*n+1];')
    if arg_idx:
        code('arg_idx[n] = start[n];')
    ENDFOR()

    #  if MULTI_GRID:
    #    code('int global_idx['+str(NDIM)+'];')
    #    code('#ifdef OPS_MPI')
    #    for n in range (0,NDIM):
    #      code('global_idx['+str(n)+'] = arg_idx['+str(n)+'];')
    #    code('#else')
    #    for n in range (0,NDIM):
    #      code('global_idx['+str(n)+'] = start['+str(n)+'];')
    #    code('#endif')
    #    code('')

    for n in range(0, nargs):
        if arg_typ[n] == 'ops_arg_dat':
            code('int xdim' + str(n) + ' = args[' + str(n) + '].dat->size[0];')
            if NDIM > 2 or (NDIM == 2 and soa_set):
                code('int ydim' + str(n) + ' = args[' + str(n) +
                     '].dat->size[1];')
            if NDIM > 3 or (NDIM == 3 and soa_set):
                code('int zdim' + str(n) + ' = args[' + str(n) +
                     '].dat->size[2];')
    code('')

    condition = ''
    for n in range(0, nargs):
        if arg_typ[n] == 'ops_arg_dat':
            condition = condition + 'xdim' + str(
                n) + ' != dims_' + name + '_h[' + str(n) + '][0] || '
            if NDIM > 2 or (NDIM == 2 and soa_set):
                condition = condition + 'ydim' + str(
                    n) + ' != dims_' + name + '_h[' + str(n) + '][1] || '
            if NDIM > 3 or (NDIM == 3 and soa_set):
                condition = condition + 'zdim' + str(
                    n) + ' != dims_' + name + '_h[' + str(n) + '][2] || '
    condition = condition[:-4]
    IF(condition)

    #    for n in range (0, nargs):
    #      if arg_typ[n] == 'ops_arg_dat':
    #        code('cudaMemcpyToSymbol( dims_'+name+'['+str(n)+'][0]'+', &xdim'+str(n)+', sizeof(int) );')
    #        code('dims_'+name+'_h['+str(n)+'][0] = xdim'+str(n)+';')
    #        if NDIM>2 or (NDIM==2 and soa_set):
    #          code('cudaMemcpyToSymbol( dims_'+name+'['+str(n)+'][1]'+', &ydim'+str(n)+', sizeof(int) );')
    #          code('dims_'+name+'_h['+str(n)+'][1] = ydim'+str(n)+';')
    #        if NDIM>3 or (NDIM==3 and soa_set):
    #          code('cudaMemcpyToSymbol( dims_'+name+'['+str(n)+'][2]'+', &zdim'+str(n)+', sizeof(int) );')
    #          code('dims_'+name+'_h['+str(n)+'][2] = zdim'+str(n)+';')
    for n in range(0, nargs):
        if arg_typ[n] == 'ops_arg_dat':
            code('dims_' + name + '_h[' + str(n) + '][0] = xdim' + str(n) +
                 ';')
            if NDIM > 2 or (NDIM == 2 and soa_set):
                code('dims_' + name + '_h[' + str(n) + '][1] = ydim' + str(n) +
                     ';')
            if NDIM > 3 or (NDIM == 3 and soa_set):
                code('dims_' + name + '_h[' + str(n) + '][2] = zdim' + str(n) +
                     ';')
    code('cutilSafeCall(block->instance->ostream(),cudaMemcpyToSymbol( dims_' +
         name + ', dims_' + name + '_h, sizeof(dims_' + name + ')));')
    ENDIF()

    code('')

    #setup const variables
    for n, (arg_type, typ, acc,
            dim) in enumerate(zip(arg_typ, typs, accs, dims)):
        if arg_type == 'ops_arg_gbl' and acc == OPS_READ and (
                not dim.isdigit() or int(dim) > 1):
            code(f'{typ} *arg{n}h = ({typ} *)args[{n}].data;')

    code('')
    code('int x_size = MAX(0,end[0]-start[0]);')
    if NDIM == 2:
        code('int y_size = MAX(0,end[1]-start[1]);')
    if NDIM == 3:
        code('int y_size = MAX(0,end[1]-start[1]);')
        code('int z_size = MAX(0,end[2]-start[2]);')
    code('')

    # set up CUDA grid and thread blocks for kernel call
    # If the kernel has lowdim datasets we will have hard coded kernel size 
    # and reductions
    thr_blk_x = 'block->instance->OPS_block_size_x'
    thr_blk_y = 'block->instance->OPS_block_size_y'
    thr_blk_z = 'block->instance->OPS_block_size_z'
    if not all([x == 1 for x in stride]): # 1D lowdim in 1D loop? -> blk_y =4 low perf
        thr_blk_x = '32'
        thr_blk_y = '4'
        thr_blk_z = '1'
        
    if NDIM == 1:
        code(
            f'dim3 grid( (x_size-1)/{thr_blk_x}+ 1, 1, 1);'
        )
    if NDIM == 2:
        code(
            f'dim3 grid( (x_size-1)/{thr_blk_x}+ 1, (y_size-1)/{thr_blk_y} + 1, 1);'
        )
    if NDIM == 3:
        code(
            f'dim3 grid( (x_size-1)/{thr_blk_x}+ 1, (y_size-1)/{thr_blk_y} + 1, (z_size-1)/{thr_blk_z} +1);'
        )

    if NDIM > 1:
        code(f'dim3 tblock({thr_blk_x},{thr_blk_y},{thr_blk_z});')
    else:
        code(f'dim3 tblock({thr_blk_x},1,1);')

    code('')

    GBL_READ = False
    GBL_READ_MDIM = False
    GBL_INC = False  # reductions on scalar adjoints

    #set up reduction variables
    for n in range(0, nargs):
        n_ad = get_ad_idx(n, arg_idxs_with_adjoints)
        if arg_typ[n] == 'ops_arg_gbl':
            if accs[n] == OPS_READ or n_ad != -1:
                GBL_READ = True
                if not dims[n].isdigit() or int(dims[n]) > 1:
                    GBL_READ_MDIM = True
        elif arg_typ[n] == 'ops_arg_scalar':
            if accs[n] == OPS_READ:
                if n_ad != -1:
                    GBL_INC = True

    if GBL_INC == True:
        code('int reduct_size = 0;')
        code('')

    if GBL_READ == True and GBL_READ_MDIM == True:
        code('int consts_bytes = 0;')
        code('')

    for arg_type, typ, acc, dim in zip(arg_typ, typs, accs, dims):
        if arg_type == 'ops_arg_gbl' and acc == OPS_READ and (
                not dim.isdigit() or int(dim) > 1):
            code(f'consts_bytes += ROUND_UP({dim}*sizeof({typ}));')

    for arg_type, typ, acc, dim in zip(arg_typ, typs, accs, dims):
        n_ad = get_ad_idx(n, arg_idxs_with_adjoints)
        if arg_type == 'ops_arg_gbl' and acc != OPS_READ:
            if n_ad != -1 and (not dims[n].isdigit() or int(dims[n]) > 1):
                code(f'consts_bytes += ROUND_UP({dim}*sizeof({typ}));')
        elif arg_type == 'ops_arg_scalar':
            if acc == OPS_READ and n_ad != -1:
                code(f'reduct_size = MAX(reduct_size,sizeof({typ})*{dim});')

    code('')

    if GBL_READ == True and GBL_READ_MDIM == True:
        code('reallocConstArrays(block->instance, consts_bytes);')

    code('')

    for n in range(0, nargs):
        if arg_typ[n] == 'ops_arg_gbl':
            if accs[n] == OPS_READ and (not dims[n].isdigit()
                                        or int(dims[n]) > 1):
                code('consts_bytes = 0;')
                code('args[' + str(n) +
                     '].data = block->instance->OPS_consts_h + consts_bytes;')
                code(
                    'args[' + str(n) +
                    '].data_d = block->instance->OPS_consts_d + consts_bytes;')
                code('for (int d=0; d<' + str(dims[n]) + '; d++) ((' +
                     typs[n] + ' *)args[' + str(n) + '].data)[d] = arg' +
                     str(n) + 'h[d];')
                code('consts_bytes += ROUND_UP(' + str(dims[n]) + '*sizeof(' +
                     typs[n] + '));')
            elif accs[n] != OPS_READ and -1 != get_ad_idx(
                    n, arg_idxs_with_adjoints):
                #reduction adjoints as global constants
                if not dims[n].isdigit() or int(dims[n]) > 1:
                    code(
                        'const {1} *arg{0}_a1s = ({1} *)(block->instance->OPS_consts_d + consts_bytes);'
                        .format(n, typs[n]))
                    code(
                        '{1} *arg{0}_a1s_h = ({1} *)(block->instance->OPS_consts_h + consts_bytes);'
                        .format(n, typs[n]))
                    code('for (int d=0; d<' + str(dims[n]) + '; d++) arg' +
                         str(n) + '_a1s_h[d] = ((' + typs[n] + ' *)arg' +
                         str(n) + '.derivative)[d];')
                    code('consts_bytes += ROUND_UP(' + str(dims[n]) +
                         '*sizeof(' + typs[n] + '));')
                else:
                    code(
                        'const {1} arg{0}_a1s = *(({1} *)args[{0}].derivative);'
                        .format(n, typs[n]))
    if GBL_READ == True and GBL_READ_MDIM == True:
        code('mvConstArraysToDevice(block->instance, consts_bytes);')

    for n in range(0, nargs):
        if arg_typ[n] == 'ops_arg_dat':
            code('int dat' + str(n) +
                 ' = (block->instance->OPS_soa ? sizeof(' + typs[n] +
                 ') : args[' + str(n) + '].dat->elem_size);')

    code('')
    code('char *p_a[' + str(nargs) + '];')

    #some custom logic for multigrid
    if MULTI_GRID:
        for n in range(0, nargs):
            if prolong[n] == 1 or restrict[n] == 1:
                comm(
                    'This arg has a prolong stencil - so create different ranges'
                )
                code('int start_' + str(n) + '[' + str(NDIM) + ']; int end_' +
                     str(n) + '[' + str(NDIM) + ']; int stride_' + str(n) +
                     '[' + str(NDIM) + '];int d_size_' + str(n) + '[' +
                     str(NDIM) + '];')
                code('#ifdef OPS_MPI')
                FOR('n', '0', str(NDIM))
                code('sub_dat *sd' + str(n) + ' = OPS_sub_dat_list[args[' +
                     str(n) + '].dat->index];')
                code('stride_' + str(n) + '[n] = args[' + str(n) +
                     '].stencil->mgrid_stride[n];')
                code('d_size_' + str(n) + '[n] = args[' + str(n) +
                     '].dat->d_m[n] + sd' + str(n) +
                     '->decomp_size[n] - args[' + str(n) + '].dat->d_p[n];')
                if restrict[n] == 1:
                    code('start_' + str(n) + '[n] = global_idx[n]*stride_' +
                         str(n) + '[n] - sd' + str(n) +
                         '->decomp_disp[n] + args[' + str(n) +
                         '].dat->d_m[n];')
                else:
                    code('start_' + str(n) + '[n] = global_idx[n]/stride_' +
                         str(n) + '[n] - sd' + str(n) +
                         '->decomp_disp[n] + args[' + str(n) +
                         '].dat->d_m[n];')
                code('end_' + str(n) + '[n] = start_' + str(n) +
                     '[n] + d_size_' + str(n) + '[n];')
                ENDFOR()
                code('#else')
                FOR('n', '0', str(NDIM))
                code('stride_' + str(n) + '[n] = args[' + str(n) +
                     '].stencil->mgrid_stride[n];')
                code('d_size_' + str(n) + '[n] = args[' + str(n) +
                     '].dat->d_m[n] + args[' + str(n) +
                     '].dat->size[n] - args[' + str(n) + '].dat->d_p[n];')
                if restrict[n] == 1:
                    code('start_' + str(n) + '[n] = global_idx[n]*stride_' +
                         str(n) + '[n];')
                else:
                    code('start_' + str(n) + '[n] = global_idx[n]/stride_' +
                         str(n) + '[n];')
                code('end_' + str(n) + '[n] = start_' + str(n) +
                     '[n] + d_size_' + str(n) + '[n];')
                ENDFOR()
                code('#endif')

    comm('')
    comm('set up initial pointers')
    for n, (arg_type, typ, dim,
            acc) in enumerate(zip(arg_typ, typs, dims, accs)):
        n_ad = get_ad_idx(n, arg_idxs_with_adjoints)
        if arg_type == 'ops_arg_dat':
            if prolong[n] == 1 or restrict[n] == 1:
                starttext = 'start_' + str(n)
            else:
                starttext = 'start'
            code(
                f'int base{n} = args[{n}].dat->base_offset + dat{n} * 1 * ({starttext}[0] * args[{n}].stencil->stride[0]);'
            )
            for d in range(1, NDIM):
                line = f'base{n} = base{n}+ dat{n} *\n'
                for d2 in range(0, d):
                    line = line + config.depth * ' ' + f'  args[{n}].dat->size[{d2}] *\n'
                code(line[:-1])
                code(f'  ({starttext}[{d}] * args[{n}].stencil->stride[{d}]);')

            code(f'p_a[{n}] = (char *)args[{n}].data_d + base{n};')
            code('')

            if acc != OPS_READ:
                code(
                    f'{typ} *__restrict__ arg{n}_cp = ({typ} *)ops_get_cp(args[{n}].dat);'
                )
            if n_ad != -1:
                code(f'int base{n}_t = base{n} / args[{n}].dat->type_size;')
                code(
                    f'{typ} *__restrict__ arg{n}_a1s = ({typ} *)args[{n}].derivative_d + base{n}_t;'
                )
        elif arg_type == 'ops_arg_scalar':
            code(f'const {typ} *arg{n}d = ({typ} *)args[{n}].data_d;')
            code(f'{typ} *arg{n}_a1s = ({typ} *)args[{n}].derivative_d;')

    #halo exchange
    code('')
    code('#ifndef OPS_LAZY')
    code('ops_derivative_H_D_exchanges_device(args, ' + str(nargs) + ');')
    code('ops_H_D_exchanges_device(args, ' + str(nargs) + ');')
    code('ops_halo_exchanges(args,' + str(nargs) + ',range);')
    code('#endif')
    code('')
    IF('block->instance->OPS_diags > 1')
    code('ops_timers_core(&c2,&t2);')
    code('block->instance->OPS_kernels[' + str(nk) + '].mpi_time += t2-t1;')
    ENDIF()
    code('')

    #set up shared memory for reduction
    if GBL_INC == True:
        code(f'int nthread = {thr_blk_x}*{thr_blk_y}*{thr_blk_z};')
        code('')
        code('int nshared = reduct_size*nthread;')
        code('')

    #kernel call
    comm('call kernel wrapper function, passing in pointers to data')
    IF('&&'.join(['{}_size > 0'.format(dim_labels[d]) for d in range(NDIM)]))
    if GBL_INC == True:
        text = 'ops_' + adjoint_name + '<<<grid, tblock, nshared >>> ( '
    else:
        text = 'ops_' + adjoint_name + '<<<grid, tblock >>> ( '
    for n in range(0, nargs):
        n_ad = get_ad_idx(n, arg_idxs_with_adjoints)
        if arg_typ[n] == 'ops_arg_dat':
            text = text + ' (' + typs[n] + ' *)p_a[' + str(n) + '],'
            if accs[n] != OPS_READ:
                text = text + ' arg' + str(n) + '_cp,'
        elif arg_typ[n] == 'ops_arg_gbl':
            if accs[n] == OPS_READ:
                if dims[n].isdigit() and int(dims[n]) == 1:
                    text = text + ' *(' + typs[n] + ' *)args[' + str(
                        n) + '].data,'
                else:
                    text = text + ' (' + typs[n] + ' *)args[' + str(
                        n) + '].data_d,'
        elif arg_typ[n] == 'ops_arg_idx':
            text += ','.join(['arg_idx[{}]'.format(d)
                              for d in range(NDIM)]) + ','
        elif arg_typ[n] == 'ops_arg_scalar':
            text = text + ' arg' + str(n) + 'd,'
        if n_ad != -1:
            text += " arg{}_a1s,".format(n)
        if restrict[n] or prolong[n]:
            text += ','.join(
                ['stride_{}[{}]'.format(n, d) for d in range(NDIM)]) + ','
    if any_prolong:
        text += ','.join(['global_idx[{}]'.format(d)
                          for d in range(NDIM)]) + ','

    text += ','.join(['{}_size'.format(dim_labels[d])
                      for d in range(NDIM)]) + ');'
    code(text)
    ENDIF()

    code('')
    code('cutilSafeCall(block->instance->ostream(),cudaGetLastError());')
    code('')

    for n, (arg_type, typ, dim,
            acc) in enumerate(zip(arg_typ, typs, dims, accs)):
        if arg_type == 'ops_arg_gbl' and acc == OPS_READ and (
                not dim.isdigit() or int(dim) > 1):
            code(f'args[{n}].data = (char *)arg{n}h;')
            code('')

    IF('block->instance->OPS_diags>1')
    code('cutilSafeCall(block->instance->ostream(),cudaDeviceSynchronize());')
    code('ops_timers_core(&c1,&t1);')
    code('block->instance->OPS_kernels[' + str(nk) +
         '].adjoint_time += t1-t2;')
    ENDIF()
    code('')

    code('ops_set_dirtybit_device(args, ' + str(nargs) + ');')
    code('ops_ad_set_dirtybit_device(args, ' + str(nargs) + ');')
    for n in range(0, nargs):
        if arg_typ[n] == 'ops_arg_dat' and (accs[n] == OPS_WRITE or accs[n]
                                            == OPS_RW or accs[n] == OPS_INC):
            code('ops_set_halo_dirtybit3(&args[' + str(n) + '],range);')

    code('')
    IF('block->instance->OPS_diags > 1')
    comm('Update kernel record')
    code('ops_timers_core(&c2,&t2);')
    code('block->instance->OPS_kernels[' + str(nk) + '].mpi_time += t2-t1;')
    for n, typ in enumerate(arg_typ):
        if typ == 'ops_arg_dat':
            code(
                f'block->instance->OPS_kernels[{nk}].ad_transfer += ops_compute_transfer_adjoint(dim, start, end, &args[{n}]);'
            )
    ENDIF()
    config.depth = config.depth - 2
    code('}')
    code('')

    return arg_idxs_with_adjoints


def ops_gen_mpi_adjoint_cuda(master, date, consts, kernels, soa_set):
    OPS_ID = 1
    OPS_GBL = 2

    OPS_READ = 1
    OPS_WRITE = 2
    OPS_RW = 3
    OPS_INC = 4
    OPS_MAX = 5
    OPS_MIN = 6

    accsstring = [
        'OPS_READ', 'OPS_WRITE', 'OPS_RW', 'OPS_INC', 'OPS_MAX', 'OPS_MIN'
    ]

    NDIM = 2  #the dimension of the application is hardcoded here .. need to get this dynamically

    src_dir = os.path.dirname(master) or '.'
    master_basename = os.path.splitext(os.path.basename(master))

    ##########################################################################
    #  create new kernel file
    ##########################################################################

    for nk in range(0, len(kernels)):
        arg_typ = kernels[nk]['arg_type']
        name = kernels[nk]['name']
        nargs = kernels[nk]['nargs']
        dim = kernels[nk]['dim']
        dims = kernels[nk]['dims']
        stens = kernels[nk]['stens']
        var = kernels[nk]['var']
        accs = kernels[nk]['accs']
        typs = kernels[nk]['typs']
        NDIM = int(dim)
        #parse stencil to locate strided access
        stride = [1] * nargs * NDIM
        restrict = [1] * nargs
        prolong = [1] * nargs

        if NDIM == 2:
            for n in range(0, nargs):
                if str(stens[n]).find('STRID2D_X') > 0:
                    stride[NDIM * n + 1] = 0
                elif str(stens[n]).find('STRID2D_Y') > 0:
                    stride[NDIM * n] = 0

        if NDIM == 3:
            for n in range(0, nargs):
                if str(stens[n]).find('STRID3D_XY') > 0:
                    stride[NDIM * n + 2] = 0
                elif str(stens[n]).find('STRID3D_YZ') > 0:
                    stride[NDIM * n] = 0
                elif str(stens[n]).find('STRID3D_XZ') > 0:
                    stride[NDIM * n + 1] = 0
                elif str(stens[n]).find('STRID3D_X') > 0:
                    stride[NDIM * n + 1] = 0
                    stride[NDIM * n + 2] = 0
                elif str(stens[n]).find('STRID3D_Y') > 0:
                    stride[NDIM * n] = 0
                    stride[NDIM * n + 2] = 0
                elif str(stens[n]).find('STRID3D_Z') > 0:
                    stride[NDIM * n] = 0
                    stride[NDIM * n + 1] = 0

        ### Determine if this is a MULTI_GRID LOOP with
        ### either restrict or prolong
        MULTI_GRID = 0
        any_prolong = 0
        for n in range(0, nargs):
            restrict[n] = 0
            prolong[n] = 0
            if str(stens[n]).find('RESTRICT') > 0:
                restrict[n] = 1
                MULTI_GRID = 1
            if str(stens[n]).find('PROLONG') > 0:
                prolong[n] = 1
                MULTI_GRID = 1
                any_prolong = 1

        reduct = 0
        for n in range(0, nargs):
            if arg_typ[n] == 'ops_arg_gbl' and accs[n] != OPS_READ:
                reduct = 1

        config.file_text = ''
        config.depth = 0
        n_per_line = 4

        i = name.find('kernel')
        name2 = name[0:i - 1]
        #print name2

        reduction = False
        ng_args = 0

        for n in range(0, nargs):
            if arg_typ[n] == 'ops_arg_gbl' and accs[n] != OPS_READ:
                reduction = True
            else:
                ng_args = ng_args + 1

        arg_idx = 0
        for n in range(0, nargs):
            if arg_typ[n] == 'ops_arg_idx':
                arg_idx = 1

    ##########################################################################
    #  generate constants and MACROS
    ##########################################################################

        num_dims = max(1, NDIM - 1)
        if NDIM > 1 and soa_set:
            num_dims += 1
        code('__constant__ int dims_' + name + ' [' + str(nargs) + '][' +
             str(num_dims) + '];')
        code('static int dims_' + name + '_h [' + str(nargs) + '][' +
             str(num_dims) + '] = {0};')
        code('')

        ##########################################################################
        #  generate cuda kernel wrapper function
        ##########################################################################

        generate_cuda_kernel(arg_typ, name, nargs, accs, typs, dims, NDIM,
                             stride, restrict, any_prolong, prolong,
                             MULTI_GRID, arg_idx, n_per_line, nk, soa_set,
                             src_dir)  # reduction,
        arg_idxs_with_adjoints = generate_adjoint_cuda_kernel(
            arg_typ, name, nargs, accs, typs, dims, NDIM, stride, stens, restrict,
            any_prolong, prolong, MULTI_GRID, arg_idx, n_per_line, nk, soa_set,
            src_dir, consts)  # reduction,

        ####################################################
        ##  generate desc creation and loop register
        ####################################################
        code(
            'void ops_par_loop_' + name +
            '_impl(char const *name, ops_block block, int dim, int* range, bool is_passive,'
        )
        code(
            group_n_per_line(
                [' ops_arg arg{}'.format(n)
                 for n in range(nargs)], n_per_line) + ') {')
        config.depth = 2
        if arg_idxs_with_adjoints:
            IF('!is_passive')
            # if ith arg is passive and ad has ad idx throw error
            for i in range(nargs):
                if arg_typ[i] == 'ops_arg_dat' and get_ad_idx(
                        i, arg_idxs_with_adjoints) != -1:
                    IF('arg{}.dat->is_passive'.format(i))
                    code('OPSException ex(OPS_INVALID_ARGUMENT);')
                    code(
                        'ex << "Error: passing passive dat to active loop argument loop: " << name << " arg{0}: " << arg{0}.dat->name;'
                        .format(i))
                    code('throw ex;')
                    ENDIF()
            ENDIF()
        code(
            'ops_kernel_descriptor *desc = (ops_kernel_descriptor *)malloc(sizeof(ops_kernel_descriptor));'
        )
        code('desc->name = name;')
        code('desc->block = block;')
        code('desc->dim = dim;')
        code('desc->device = 1;')
        code('desc->index = ' + str(nk) + ';')
        code('desc->hash = 5381;')
        code('desc->hash = ((desc->hash << 5) + desc->hash) + ' + str(nk) +
             ';')
        FOR('i', '0', str(2 * NDIM))
        code('desc->range[i] = range[i];')
        code('desc->orig_range[i] = range[i];')
        code('desc->hash = ((desc->hash << 5) + desc->hash) + range[i];')
        ENDFOR()

        code('desc->nargs = ' + str(nargs) + ';')
        code('desc->args = (ops_arg*)malloc(' + str(nargs) +
             '*sizeof(ops_arg));')
        declared = 0
        for n in range(0, nargs):
            code('desc->args[' + str(n) + '] = arg' + str(n) + ';')
            if arg_typ[n] == 'ops_arg_dat':
                code('desc->hash = ((desc->hash << 5) + desc->hash) + arg' +
                     str(n) + '.dat->index;')
            if arg_typ[n] == 'ops_arg_gbl' and accs[n] == OPS_READ:
                if declared == 0:
                    code('char *tmp = (char*)malloc(' + dims[n] + '*sizeof(' +
                         typs[n] + '));')
                    declared = 1
                else:
                    code('tmp = (char*)malloc(' + dims[n] + '*sizeof(' +
                         typs[n] + '));')
                code('memcpy(tmp, arg' + str(n) + '.data,' + dims[n] +
                     '*sizeof(' + typs[n] + '));')
                code('desc->args[' + str(n) + '].data = tmp;')
        code('desc->function = ops_par_loop_' + name + '_execute;')
        if not arg_idxs_with_adjoints:
            code('desc->adjoint_function = nullptr;')
        else:
            IF('is_passive')
            code('desc->adjoint_function = nullptr;')
            ENDIF()
            ELSE()
            code('desc->adjoint_function = ops_par_loop_' + name +
                 '_adjoint_execute;')
            ENDIF()
        IF('block->instance->OPS_diags > 1')
        code('ops_timing_realloc(block->instance,' + str(nk) + ',"' + name +
             '");')
        ENDIF()
        code('ops_add_to_dag(desc);')
        config.depth = 0
        code('}')

        ##########################
        ##  generate active code
        ##########################
        code('void ops_par_loop_' + name +
             '(char const *name, ops_block block, int dim, int* range,')
        code(
            group_n_per_line(
                [' ops_arg arg{}'.format(n)
                 for n in range(nargs)], n_per_line) + ') {')
        config.depth = 2
        code('ops_par_loop_' + name + '_impl(name, block, dim, range, false,')
        code(
            group_n_per_line([' arg{}'.format(n)
                              for n in range(nargs)], n_per_line) + ');')
        config.depth = 0
        code('}')
        ##########################
        ##  generate passive code
        ##########################
        code('')
        code('void ops_par_loop_passive_' + name +
             '(char const *name, ops_block block, int dim, int* range,')
        code(
            group_n_per_line(
                [' ops_arg arg{}'.format(n)
                 for n in range(nargs)], n_per_line) + ') {')
        config.depth = 2
        code('ops_par_loop_' + name + '_impl(name, block, dim, range, true,')
        code(
            group_n_per_line([' arg{}'.format(n)
                              for n in range(nargs)], n_per_line) + ');')
        config.depth = 0
        code('}')

        ##########################################################################
        #  output individual kernel file
        ##########################################################################
        if not os.path.exists('./CUDA_adjoint'):
            os.makedirs('./CUDA_adjoint')
        fid = open('./CUDA_adjoint/' + name + '_cuda_kernel.cu', 'w')
        date = datetime.datetime.now()  #unused
        fid.write('//\n// auto-generated by ops.py\n//\n')
        fid.write(config.file_text)
        fid.close()


    # end of main kernel call loop
    
    ##########################################################################
    #  output one master kernel file
    ##########################################################################

    config.file_text = ''
    config.depth = 0
    comm('header')
    code('#define OPS_API 2')
    if NDIM == 1:
        code('#define OPS_1D')
    if NDIM == 2:
        code('#define OPS_2D')
    if NDIM == 3:
        code('#define OPS_3D')
    if soa_set:
        code('#define OPS_SOA')
    code('#include "ops_lib_core.h"')
    code('#include "ops_algodiff.hpp"')
    code('')
    code('#include "ops_cuda_rt_support.h"')
    code('#include "ops_cuda_reduction.h"')
    code('')
    code(
        '#include <cuComplex.h>'
    )  # Include the CUDA complex numbers library, in case complex numbers are used anywhere.
    code('')
    if os.path.exists(os.path.join(src_dir, 'user_types.h')):
        code('#define OPS_FUN_PREFIX __device__ __host__')
        code('#include "user_types.h"')
    code('#ifdef OPS_MPI')
    code('#include "ops_mpi_core.h"')
    code('#endif')

    code(util.generate_extern_global_consts_declarations(consts,
                                                         for_cuda=True))

    code('')
    code('void ops_init_backend() {}')
    code('')
    code(
        'void ops_decl_const_char(OPS_instance *instance, int dim, char const *type,'
    )
    code('int size, char *dat, char const *name){')
    config.depth = config.depth + 2
    code('ops_execute(instance);')

    for nc in range(0, len(consts)):
        IF('!strcmp(name,"' +
           (str(consts[nc]['name']).replace('"', '')).strip() + '")')
        if consts[nc]['dim'].isdigit():
            code('cutilSafeCall(instance->ostream(),cudaMemcpyToSymbol(' +
                 (str(consts[nc]['name']).replace('"', '')).strip() +
                 ', dat, dim*size));')
        else:
            code(
                'char *temp; cutilSafeCall(instance->ostream(),cudaMalloc((void**)&temp,dim*size));'
            )
            code(
                'cutilSafeCall(instance->ostream(),cudaMemcpy(temp,dat,dim*size,cudaMemcpyHostToDevice));'
            )
            code('cutilSafeCall(instance->ostream(),cudaMemcpyToSymbol(' +
                 (str(consts[nc]['name']).replace('"', '')).strip() +
                 ', &temp, sizeof(char *)));')
        ENDIF()
        code('else')

    code('{')
    config.depth = config.depth + 2
    code('printf("error: unknown const name\\n"); exit(1);')
    ENDIF()

    config.depth = config.depth - 2
    code('}')
    code('')

    code('')
    comm('user kernel files')

    kernel_name_list = []

    for nk in range(0, len(kernels)):
        if kernels[nk]['name'] not in kernel_name_list:
            code('#include "' + kernels[nk]['name'] + '_cuda_kernel.cu"')
            kernel_name_list.append(kernels[nk]['name'])

    fid = open('./CUDA_adjoint/' + master_basename[0] + '_kernels.cu', 'w')
    fid.write('//\n// auto-generated by ops.py\n//\n\n')
    fid.write(config.file_text)
    fid.close()
