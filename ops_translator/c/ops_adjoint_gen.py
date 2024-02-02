#!/usr/bin/env python3
"""
OPS derivative kernel generator tool (for the C/C++ API)

This tool parses the user's original source code to produce
derivative functionf for the user's kernel functions.

This prototype is written in Python and uses Tapenade to generate
derivative functions

usage: ./ops_adjoint_gen.py file1, file2 ,...

This takes as input

file1.cpp, file2.cpp, ...

and produces as output 

xxx_adjoint_kernel.h
"""

import re, random, tempfile, shutil
import os, sys, subprocess
import util
import ops
from config import OPS_READ, OPS_WRITE, OPS_RW, OPS_INC, OPS_MAX, OPS_MIN, OPS_accs_labels

verbose = util.verbose

DERIVATIVE_SUFFIX = "_a1s"
DERIVATIVE_FILE_SUFFIX = "_adjoint"


def get_tapenade_compatible_kernel(arg_list, kernel, arg_typs):
    param_list_start = kernel.find('(')
    param_list_end = util.para_parse(kernel, param_list_start, '(', ')')
    kernel_text = kernel[kernel.find('{'):]
    for i in range(len(arg_typs)):
        if arg_typs[i] == "ops_arg_dat":
            p = re.compile(arg_list[i] + '\s?\(')
            match = re.search(p, kernel_text)
            while match:
                closing_brace = util.para_parse(kernel_text,
                                                match.end() - 1, '(', ')')
                # create params for indexing. +0 forces tapenade to keep the parentheses
                mid = '*'.join([
                    '(' + d.strip() + '+0)'
                    for d in kernel_text[match.end():closing_brace].split(',')
                ])
                mid = ' '.join(mid.split())
                kernel_text = kernel_text[:match.start(
                ) + len(arg_list[i]
                        )] + "[" + mid + "]" + kernel_text[closing_brace + 1:]
                match = re.search(p, kernel_text)
    tapenade_kernel_signature = ','.join([
        'double *' + arg for arg in arg_list
    ])  # every param should be some sort of pointer or array..
    return kernel[:param_list_start +
                  1] + tapenade_kernel_signature + ")" + kernel_text


def run_tapenade(tapenade_kernel, function_name, arg_list, arg_typs, accs,
                 consts, src_dir):

    tapenade_dir = ""
    if 0 != subprocess.call("which tapenade 1> /dev/null 2>&1", shell=True):
        tapenade_dir = os.getenv("TAPENADE_INSTALL_PATH")
        if tapenade_dir is None:
            raise Exception(
                "Couldn't find tapenade. Please add it to PATH or specify TAPENADE_INSTALL_PATH"
            )
        tapenade_dir = os.path.join(tapenade_dir, "bin")
    tapenade_exec = os.path.join(tapenade_dir, 'tapenade')
    output = []
    active_input = []
    for arg_typ, acc, arg in zip(arg_typs, accs, arg_list):
        if arg_typ == "ops_arg_gbl":
            #  Only INC reductions has proper derivatives that does not depend on order
            if acc == OPS_INC:
                output.append(arg)
        elif arg_typ == "ops_arg_scalar" or arg_typ == "ops_arg_dat":
            if acc == OPS_READ:
                active_input.append(arg)
            else:
                output.append(arg)
    all_active = ' '.join(output + active_input)
    head = f'{function_name}({all_active})/({all_active})'

    temp_dir = tempfile.TemporaryDirectory()
    fname = os.path.join(temp_dir.name, f"{function_name}.c")

    with open(fname, 'w') as fout:
        if os.path.exists(os.path.join(src_dir, 'user_types.h')):
            fout.write('#include "user_types.h"')
            shutil.copy2(os.path.join(src_dir, 'user_types.h'), temp_dir.name)
        fout.write(util.generate_extern_global_consts_declarations(consts))
        fout.write(tapenade_kernel)
    tapenade_params = [
        '-head', head, '-fixinterface', '-b', '-adjfuncname',
        DERIVATIVE_FILE_SUFFIX, '-adjvarname', DERIVATIVE_SUFFIX, '-O',
        temp_dir.name, fname
    ]
    subprocess.run([tapenade_exec] + tapenade_params)
    output_fname = os.path.join(temp_dir.name,
                                f"{function_name}{DERIVATIVE_FILE_SUFFIX}.c")
    if os.path.isfile(output_fname):
        with open(output_fname, 'r') as fin:
            tapenade_kernel = fin.read()
        return util.get_kernel_func_from_text(
            function_name + DERIVATIVE_FILE_SUFFIX, tapenade_kernel)
    raise Exception("Something went wrong during adjoint kernel generation")
    return ""


def transform_derivative_back(derivative_kernel, orig_param_list, arg_list,
                              arg_typs, accs, typs):
    orig_param_list = orig_param_list.split(',')
    param_list_start = derivative_kernel.find('(')
    param_list_end = util.para_parse(derivative_kernel, param_list_start, '(',
                                     ')')
    tapenade_param_list = []
    tapenade_kernel_text = derivative_kernel[derivative_kernel.find('{'):]
    for arg_typ, typ, acc, arg, orig_param in zip(arg_typs, typs, accs,
                                                  arg_list, orig_param_list):
        tapenade_param_list.append(orig_param)
        if arg_typ == "ops_arg_dat":
            tapenade_param_list.append("ACC<" + typ + "> &" + arg +
                                       DERIVATIVE_SUFFIX)
            for var in [arg, arg + DERIVATIVE_SUFFIX]:
                p = re.compile(var + '\s*\[')
                match = re.search(p, tapenade_kernel_text)
                while match:
                    closing_brace = util.para_parse(tapenade_kernel_text,
                                                    match.end() - 1, '[', ']')
                    assert closing_brace > match.end()
                    mid = []
                    indices = tapenade_kernel_text[match.end():closing_brace]
                    begin_p =  indices.find('(')
                    while begin_p != -1:
                        end_p = util.para_parse(indices, begin_p, '(', ')')
                        last = indices[:end_p].rfind(
                            "+")  # there is a plus +0 at the end of each arg.
                        mid.append(indices[begin_p + 1:last])
                        indices = indices[end_p:]
                        begin_p = indices.find('(')

                    mid = ','.join(mid)
                    tapenade_kernel_text = tapenade_kernel_text[:match.start(
                    ) + len(var)] + "(" + mid + ")" + tapenade_kernel_text[
                        closing_brace + 1:]
                    match = re.search(p, tapenade_kernel_text)
            tapenade_kernel_text = re.sub(f'({arg+DERIVATIVE_SUFFIX}\([^)]+\))\s*=\s*\\1\s*([\+-])','\\1 += \\2', tapenade_kernel_text)
        elif arg_typ == "ops_arg_scalar":
            tapenade_param_list.append(typ + " *" + arg + DERIVATIVE_SUFFIX)
        elif arg_typ == "ops_arg_gbl" and acc != OPS_READ:
            tapenade_param_list.append(typ + " *" + arg + DERIVATIVE_SUFFIX)

    return derivative_kernel[:param_list_start] + "(" + ','.join(
        tapenade_param_list) + ')' + tapenade_kernel_text


def get_derivative_function(function_text, function_name, arg_typs, accs, typs,
                            consts, src_dir):
    arg_list, kernel = util.get_kernel_func_from_text(function_name,
                                                      function_text)
    param_list_start = kernel.find('(')
    param_list_end = util.para_parse(kernel, param_list_start, '(', ')')
    orig_param_list = ' '.join(kernel[param_list_start +
                                      1:param_list_end].strip().split())
    tapenade_kernel = get_tapenade_compatible_kernel(arg_list, kernel,
                                                     arg_typs)
    derivative_arg_list, derivative_kernel = run_tapenade(
        tapenade_kernel, function_name, arg_list, arg_typs, accs, consts,
        src_dir)
    return transform_derivative_back(derivative_kernel, orig_param_list,
                                     arg_list, arg_typs, accs, typs)


def generate_derivative_functions(master, consts, kernels):
    src_dir = os.path.dirname(master) or '.'
    master_basename = os.path.splitext(os.path.basename(master))
    with open(f'{master_basename[0]}_ops_adjoint_kernel.h', 'w') as fid:
        fid.write('//\n// auto-generated by ops_adjoin_gen.py\n//\n')
        for kernel in kernels:
            name = kernel['name']
            accs = kernel['accs']
            arg_typ = kernel['arg_type']
            typs = kernel['typs']
            adjoint_name = f'{name}_adjoint'
            kernel_text, _ = util.find_kernel_with_retry(src_dir, adjoint_name)
            if kernel_text is None:
                orig_kernel_func, arg_list = util.find_kernel(src_dir, name)
                kernel_text = get_derivative_function(orig_kernel_func, name,
                                                      arg_typ, accs, typs,
                                                      consts, src_dir)
                fid.write(kernel_text + "\n\n")


def main(source_files):
    assert len(source_files) >= 1

    _, _, _, kernels, consts, _ = ops.parse_source_files(source_files)

    #
    # generate adjoint kernel headers
    #
    generate_derivative_functions(str(source_files[0]), consts, kernels)


if __name__ == '__main__':
    if len(sys.argv) > 1:
        main(source_files=sys.argv[1:])
    # Print usage message if no arguments given
    else:
        print(__doc__)
        sys.exit(1)
