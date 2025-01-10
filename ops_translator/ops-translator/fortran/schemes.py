import os
import re
import copy
from pathlib import Path
from typing import Dict, Any

import fortran.translator.kernels as ftk
import fortran.translator.kernels_c as ftk_c
from fortran.parser import getChild, parseIdentifier

import ops as OPS
from language import Lang
from scheme import Scheme
from store import Application, ParseError, Program
from target import Target
from util import find, extract_arglist_fortran, KernelProcess

import fparser.two.Fortran2003 as f2003
from fparser.common.readfortran import FortranStringReader
from fparser.two.parser import ParserFactory


def generate_cpp_Kernel(loop: OPS.Loop,
                    kernel_args: dict,
                    local_vars: dict,
                    var_init: str,
                    kernel_body: str,
                    f90_src: str) -> str :

    param_list = extract_arglist_fortran(f90_src)

    # Create CPP side function argument list
    arg_str = ""
    for i, arg in enumerate(loop.args):
        var_name = param_list[i].lower()        # variable name used in fortran subroutine declaration

        if var_name in kernel_args:
            param_type_and_size = kernel_args.get(var_name)
        else:
            raise ParseError(f"Unable to find file {var_name} for subroutine in argument list: {subroutine_name}")
        c_type = param_type_and_size[0]
        arr_sizes = param_type_and_size[1]

        loop.f2c_type.insert(i, c_type)

        if isinstance(arg, OPS.ArgDat):
            if arg.access_type == OPS.AccessType.OPS_READ:
                arg_str += f"const ACC<{c_type}> &{var_name}, "
            else:
                arg_str += f"ACC<{c_type}> &{var_name}, "
        elif isinstance(arg, OPS.ArgGbl):
            arg_str += f"const {c_type} *{var_name}, "
        elif isinstance(arg, OPS.ArgReduce):
            arg_str += f"{c_type} *{var_name}, "
        elif isinstance(arg, OPS.ArgIdx):
            arg_str += f"const int *{var_name}, "

    cpp_kernel = ""
    cpp_kernel = f"void {loop.kernel}({arg_str[:-2]}) " + "{\n\n"

    kp_obj = KernelProcess()

    sorted_args = []  # storing ops_gbl in descending order of sizes
    # Update kernel body
    for i, arg in enumerate(loop.args):
        var_name = param_list[i].lower()        # variable name used in fortran subroutine declaration

        if var_name in kernel_args:
            param_type_and_size = kernel_args.get(var_name)
        else:
            raise ParseError(f"Unable to find file {var_name} for subroutine in argument list: {subroutine_name}")
        arr_sizes = param_type_and_size[1]

        if isinstance(arg, OPS.ArgDat):
            if arg.dim > 1:
                kernel_body = kp_obj.convert_muldim_dat_indexing(kernel_body, var_name)
        elif isinstance(arg, OPS.ArgGbl): # convert all multi-dim vectors to single dim
            sorted_args.append((var_name, arr_sizes))
        elif isinstance(arg, OPS.ArgReduce):
            if arg.dim > 1:
                kernel_body = kp_obj.convert_1d_indexing(kernel_body, var_name)
            else:
                kernel_body = kp_obj.replace_array_with_pointer(kernel_body, var_name)
        elif isinstance(arg, OPS.ArgIdx):   # converting idx from fortran to c style
            kernel_body = kp_obj.replace_fixed_indexing(kernel_body, var_name)

    # Sort them in descending order so that first 3D index will be converted, then 2D and lastly 1D
    sorted_args.sort(key=lambda x: len(x[1]), reverse=True)

    # This will convert from multi-dim array to single-dim array format.
    # Here each these routine subtract 1 from index position assume array start at 1 base index in Fortran and C++ side will be 0
    for var_name, arr_sizes in sorted_args:
        # print(var_name + " : " + str(arr_sizes))
        if (len(arr_sizes) == 1):
            if arr_sizes[0].isdigit() and int(arr_sizes[0]) == 0:
                kernel_body = kp_obj.replace_array_with_first_element(kernel_body, var_name)
            else:
                kernel_body = kp_obj.convert_1d_indexing(kernel_body, var_name)
        elif len(arr_sizes) == 2:
            kernel_body = kp_obj.convert_2d_to_1d_indexing(kernel_body, var_name, arr_sizes[0])
        elif len(arr_sizes) == 3:
            kernel_body = kp_obj.convert_3d_to_1d_indexing(kernel_body, var_name, arr_sizes[0], arr_sizes[1])

    sorted_local_args = []

    for value in local_vars.values():
        var_name = value[0]
        arr_sizes = value[1]
        sorted_local_args.append((var_name, arr_sizes))

    sorted_local_args.sort(key=lambda x: len(x[1]), reverse=True)

    for var_name, arr_sizes in sorted_local_args:
        if (len(arr_sizes) == 1 and
                    (
                        (arr_sizes[0].isdigit() and int(arr_sizes[0]) != 0) or
                        (not arr_sizes[0].isdigit())
                    )
            ):
            patterm = rf"{var_name}\s*\(\s*0\s*:\s*[^)]+\)"
            match = re.search(patterm, f90_src)
            if match:   # Fortran using 0 base index declaration, just replace var_name(index) -> var_name[index]
                kernel_body = kp_obj.convert_zerobase_1d_indexing(kernel_body, var_name)
            else:
                kernel_body = kp_obj.convert_1d_indexing(kernel_body, var_name)
        elif len(arr_sizes) == 2:
            kernel_body = kp_obj.convert_2d_to_1d_indexing(kernel_body, var_name, arr_sizes[0])
        elif len(arr_sizes) == 3:
            kernel_body = kp_obj.convert_3d_to_1d_indexing(kernel_body, var_name, arr_sizes[0], arr_sizes[1])

#    print("============================================================================")
#    print(kernel_body)
#    print("============================================================================")

    # Add declarations of local variables
    for value in local_vars.keys():
        cpp_kernel += f"    {value};" + "\n"

    cpp_kernel += "\n"
    if len(var_init) > 0:
        cpp_kernel += var_init
        cpp_kernel += "\n\n"
    cpp_kernel += kernel_body
    cpp_kernel += "\n}"
    return cpp_kernel


def retrieve_subroutine_ast(file_path, subroutine_name):
    if not os.path.exists(file_path):
        raise ParseError(f"Unable to find file {file_path} for subroutine: {subroutine_name}")

    source = retrieve_subroutine_by_name_regex(file_path, subroutine_name)

    # Replace OPS_ACC<digit> and OPS_ACC_MD<digit>
    # converting to normal array shape fortran uses before generating AST
    # and passing it to kernels_c.py
    #pattern = r'\s*\(\s*\b(?:OPS_ACC|OPS_ACC_MD)[0-9]+\s*\(\s*([\s0-9,+-]+)\s*\)\s*\)'
    pattern = r'\s*\(\s*\b(?:OPS_ACC|OPS_ACC_MD)[0-9]+\s*\(\s*([a-zA-Z0-9_,+\-\s]+)\s*\)\s*\)'

    # Replace function
    def replace_function(match):
        # Remove extra spaces from the digits, commas, and symbols
        digits = re.sub(r'\s+', '', match.group(1))
        return f"({digits})"

    # Perform substitution with case-insensitive flag
    result_src = re.sub(pattern, replace_function, source, flags=re.IGNORECASE)

    # Replace kind=8 used in intrinsic functions
    pattern = r",\s*kind\s*=\s*8\s*\)"
    replacement = ")"
    result_src = re.sub(pattern, replacement, result_src, flags=re.IGNORECASE)

#    print("============================================================================")
#    print(result_src)
#    print("============================================================================")

    reader = FortranStringReader(result_src, ignore_comments=True)
    parser = ParserFactory().create(std="f2003")
    ast =  parser(reader)
    for child in ast.children:
        if child is None:
            continue

        if isinstance(child, f2003.Subroutine_Subprogram):
            return result_src, child
    return None


def retrieve_subroutine_by_name(file_path, subroutine_name):
# TODO : uses fparser to find and read the subroutine from file, but the existing formatting from file is lost.
# This could results in single statement broken into multiple lines in original file to very long single line statement
# need to break that again to multi-line statement
    if not os.path.exists(file_path):
        raise ParseError(f"Unable to find file {file_path} for subroutine: {subroutine_name}")

    path = Path(file_path)
    source = path.read_text()
    reader = FortranStringReader(source, ignore_comments=True)
    parser = ParserFactory().create(std="f2003")
    ast =  parser(reader)

    for child in ast.children:
        if child is None:
            continue

        if isinstance(child, f2003.Subroutine_Subprogram):
            definition_statement = getChild(child, f2003.Subroutine_Stmt)
            name_node = getChild(definition_statement, f2003.Name)
            name = parseIdentifier(name_node, None)
            if name.lower() == subroutine_name.lower():
                req_kernel = str(child)

                # replacing OPS_ACC and OPS_ACC_MD to uppercase if any lowecase occurence found
                pattern = re.compile(r'ops_acc(?:|_md)\d+', re.IGNORECASE)
                new_kernel = pattern.sub(lambda x: x.group(0).upper(), req_kernel)

                return new_kernel

    return None


def retrieve_subroutine_by_name_regex(file_path, subroutine_name):
    if not os.path.exists(file_path):
        raise ParseError(f"Unable to find file {file_path} for subroutine: {subroutine_name}")

    with open(file_path, 'r') as f:
        fortran_code = f.read()

    beg = re.search(r'\s*\bsubroutine\s*'+subroutine_name+r'\b\s*\(', fortran_code, re.IGNORECASE)
    if beg == None:
        raise ParseError(f"Unable to find subroutine: {subroutine_name}")
        exit(1)
    beg_pos = beg.start()
    end = re.search(r'\s*end\s*subroutine\b', fortran_code[beg_pos:], re.IGNORECASE)
    if end == None:
        raise ParseError(f"'Could not find matching end subroutine for {subroutine_name}")
        exit(1)

    req_kernel = fortran_code[beg_pos:beg_pos+end.end()]
    return req_kernel+'\n'


class FortranMPIOpenMP(Scheme):
    lang = Lang.find("F90")
    target = Target.find("mpi_openmp")

    fallback = None

    loop_host_template = Path("fortran/mpi_openmp/loop_host.F90.j2")
    loop_host_f2c_template = None
    master_kernel_template = None    

    loop_kernel_extension = "F90"

    def translateKernel(
        self,
        loop: OPS.Loop,
        program: Program,
        app: Application,
        kernel_idx: int
    ) -> str:

        filename = loop.kernel[:loop.kernel.find("kernel")]+"kernel.inc"

        #kernel_entities = retrieve_subroutine_by_name(filename, loop.kernel)
        kernel_entities = retrieve_subroutine_by_name_regex(filename, loop.kernel)

        if kernel_entities is None or (kernel_entities is not None and len(kernel_entities) == 0):
            raise ParseError(f"unable to find kernel function: {loop.kernel}")

        return kernel_entities.strip()

Scheme.register(FortranMPIOpenMP)


class F2CMPIOpenMP(Scheme):
    lang = Lang.find("F90")
    target = Target.find("f2c_mpi_openmp")

    fallback = None

    loop_host_template = Path("fortran/f2c_mpi_openmp/loop_host.F90.j2")
    loop_host_f2c_template = Path("fortran/f2c_mpi_openmp/loop_f2c_host.cpp.j2")
    master_kernel_template = Path("fortran/f2c_mpi_openmp/master_kernel.cpp.j2")

    loop_kernel_extension = "F90"
    loop_kernel_f2c_extension = "cpp"
    master_kernel_extension = "cpp"

    def translateKernel(
        self,
        loop: OPS.Loop,
        program: Program,
        app: Application,
        kernel_idx: int
    ) -> str:

        filename = loop.kernel[:loop.kernel.find("kernel")]+"kernel.inc"
        f90_src, entity_ast = retrieve_subroutine_ast(filename, loop.kernel)

        info = ftk_c.parseInfo(entity_ast, app, loop)
        kernel_args, local_vars, c_var_init, c_kernel_body = ftk_c.translate(info)

        cpp_kernel = generate_cpp_Kernel(loop, kernel_args, local_vars, c_var_init, c_kernel_body, f90_src)

        return cpp_kernel

Scheme.register(F2CMPIOpenMP)


class FortranCuda(Scheme):
    lang = Lang.find("F90")
    target = Target.find("cuda")

    fallback = None

    loop_host_template = Path("fortran/cuda/loop_host.F90.j2")
    loop_host_f2c_template = None
    master_kernel_template = None

    loop_kernel_extension = "CUF"

    def translateKernel(
        self,
        loop: OPS.Loop,
        program: Program,
        app: Application,
        kernel_idx: int
    ) -> str:

        filename = loop.kernel[:loop.kernel.find("kernel")]+"kernel.inc"

        #kernel_entities = retrieve_subroutine_by_name(filename, loop.kernel)
        kernel_entities = retrieve_subroutine_by_name_regex(filename, loop.kernel)

        if kernel_entities is None or (kernel_entities is not None and len(kernel_entities) == 0):
            raise ParseError(f"unable to find kernel function: {loop.kernel}")

        # Replace KernelName with KernelName_gpu
        replacement_string = loop.kernel+"_gpu"
        output_string = re.sub(re.escape(loop.kernel), replacement_string, kernel_entities, flags=re.IGNORECASE)

        # Replace all constants:   constname-> constname_opsconstant
        def replace_consts(text):
            if not os.path.isfile("constants_list.txt"):
                return text

            with open("constants_list.txt", 'r') as f:
                words_list = f.read().splitlines()

            if not words_list:
                return text

            regex_pattern = r'\b(' + '|'.join(words_list) + r')\b'
            replacement_pattern = r'\g<1>_opsconstant'
            text = re.sub(regex_pattern, replacement_pattern, text)

            return text

        output_string = replace_consts(output_string)

        return output_string.strip()

Scheme.register(FortranCuda)


class FortranOpenMPOffload(Scheme):
    lang = Lang.find("F90")
    target = Target.find("openmp_offload")

    fallback = None

    loop_host_template = Path("fortran/openmp_offload/loop_host.F90.j2")
    loop_host_f2c_template = None
    master_kernel_template = None

    loop_kernel_extension = "F90"

    def translateKernel(
        self,
        loop: OPS.Loop,
        program: Program,
        app: Application,
        kernel_idx: int
    ) -> str:

        filename = loop.kernel[:loop.kernel.find("kernel")]+"kernel.inc"

        #kernel_entities = retrieve_subroutine_by_name(filename, loop.kernel)
        kernel_entities = retrieve_subroutine_by_name_regex(filename, loop.kernel)

        if kernel_entities is None or (kernel_entities is not None and len(kernel_entities) == 0):
            raise ParseError(f"unable to find kernel function: {loop.kernel}")

        return kernel_entities.strip()

Scheme.register(FortranOpenMPOffload)
