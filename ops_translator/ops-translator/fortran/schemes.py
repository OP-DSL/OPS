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

def extract_values(data, var):
    result = None
    # Regex pattern to match exact variable name and capture value
    pattern = rf"\b{var}\b\s*=\s*(\d+)"
    match = re.search(pattern, data, re.IGNORECASE)
    if match:
        result = match.group(1)
    return result


def replace_variable(data, var, value):
    # Regex pattern to match the exact variable name with word boundaries
    pattern = rf"\b{var}\b"
    # Replace all occurrences with (value)
    return re.sub(pattern, f"{value}", data)


def generate_cpp_Kernel(loop: OPS.Loop,
                        kernel_args: dict,
                        local_vars: dict,
                        var_init: str,
                        kernel_body: str,
                        f90_src: str) -> str:

    # ============================================================
    # Extract argument list from the original Fortran subroutine
    # ============================================================
    param_list = extract_arglist_fortran(f90_src)

    # ============================================================
    # Generate C++ kernel function argument list
    # ============================================================
    cpp_argument_list = ""

    for arg_index, arg in enumerate(loop.args):
        variable_name = param_list[arg_index].lower()

        if variable_name in kernel_args:
            parameters_type_and_size = kernel_args.get(variable_name)
        else:
            raise ParseError(
                f"Unable to find file {variable_name} for subroutine in argument list: {subroutine_name}"
            )

        cpp_type = parameters_type_and_size[0]
        array_sizes = parameters_type_and_size[1]

        loop.f2c_type.insert(arg_index, cpp_type)

        if isinstance(arg, OPS.ArgDat):
            if arg.access_type == OPS.AccessType.OPS_READ:
                cpp_argument_list += f"const ACC<{cpp_type}> &{variable_name}, "
            else:
                cpp_argument_list += f"ACC<{cpp_type}> &{variable_name}, "

        elif isinstance(arg, OPS.ArgGbl):
            cpp_argument_list += f"const {cpp_type} *{variable_name}, "

        elif isinstance(arg, OPS.ArgReduce):
            cpp_argument_list += f"{cpp_type} *{variable_name}, "

        elif isinstance(arg, OPS.ArgIdx):
            cpp_argument_list += f"const int *{variable_name}, "

    cpp_kernel = f"void {loop.kernel}({cpp_argument_list[:-2]}) " + "{\n\n"

    # Helper object used for all Fortran-to-C++ indexing conversions
    kernel_process = KernelProcess()

    # ============================================================
    # Convert Fortran array indexing into equivalent C++ indexing
    # for OPS arguments used inside the kernel body
    # ============================================================
    sorted_ops_gbl_args = []

    for arg_index, arg in enumerate(loop.args):
        variable_name = param_list[arg_index].lower()

        if variable_name in kernel_args:
            parameters_type_and_size = kernel_args.get(variable_name)
        else:
            raise ParseError(
                f"Unable to find file {variable_name} for subroutine in argument list: {subroutine_name}"
            )

        array_sizes = parameters_type_and_size[1]

        if isinstance(arg, OPS.ArgDat):

            is_numeric_dimension = (
                isinstance(arg.dim, str) and arg.dim.isdigit()
            )

            is_multidimensional_dat = (
                (is_numeric_dimension and int(arg.dim) > 1)
                or not is_numeric_dimension
            )

            if is_multidimensional_dat:
                kernel_body = kernel_process.convert_muldim_dat_indexing(
                    kernel_body, variable_name
                )

        elif isinstance(arg, OPS.ArgGbl):
            sorted_ops_gbl_args.append((variable_name, array_sizes))

        elif isinstance(arg, OPS.ArgReduce):
            if arg.dim > 1:
                kernel_body = kernel_process.convert_1d_indexing(
                    kernel_body, variable_name
                )
            else:
                kernel_body = kernel_process.replace_array_with_pointer(
                    kernel_body, variable_name
                )

        elif isinstance(arg, OPS.ArgIdx):
            kernel_body = kernel_process.replace_fixed_indexing(
                kernel_body, variable_name
            )

    # ============================================================
    # Collect dimension-variable names used in local array
    # declarations. These symbolic dimensions will later be
    # resolved from constants.F90.
    # ============================================================
    sorted_local_args = []

    for local_var_info in local_vars.values():
        variable_name = local_var_info[0]
        array_sizes = local_var_info[1]
        sorted_local_args.append((variable_name, array_sizes))

    sorted_local_args.sort(key=lambda x: len(x[1]), reverse=True)

    local_var_sizes = {}
    local_arr_dimvar_names = []

    for variable_name, array_sizes in sorted_local_args:
        for array_size in array_sizes:
            if not array_size.isdigit():
                if array_size not in local_arr_dimvar_names:
                    local_arr_dimvar_names.append(array_size)

    # ============================================================
    # Resolve dimension-variable values from constants.F90
    #
    # Example:
    #   real :: tmp(nx, ny)
    #
    # Here nx and ny are looked up from constants.F90 and replaced
    # by their literal values because CUDA/HIP code generation
    # requires compile-time array dimensions.
    # ============================================================
    filename = "constants.F90"

    if not os.path.exists(filename):
        raise ParseError(f"Unable to find file {filename}")

    with open(filename, "r") as f:
        constants_f90source = f.read()

    for arr_dimvar_name in local_arr_dimvar_names:

        arr_dimvar_value = extract_values(
            constants_f90source,
            arr_dimvar_name
        )

        if arr_dimvar_value is not None and arr_dimvar_value.isdigit():
            local_var_sizes[arr_dimvar_name] = int(arr_dimvar_value)
        else:
            raise ParseError(
                f"Unable to find varible {arr_dimvar_name}'s literal value in constants.F90, "
                f"please declare this variable with parameter in constants.F90"
            )

    # ============================================================
    # Convert indexing of local arrays declared inside the kernel
    # ============================================================
    for variable_name, array_sizes in sorted_local_args:

        if (
            len(array_sizes) == 1
            and (
                (array_sizes[0].isdigit() and int(array_sizes[0]) != 0)
                or (not array_sizes[0].isdigit())
            )
        ):

            pattern = rf"{variable_name}\s*\(\s*0\s*:\s*[^)]+\)"
            match = re.search(pattern, f90_src)

            if match:
                # Fortran array declared with 0-based indexing
                kernel_body = kernel_process.convert_zerobase_1d_indexing(
                    kernel_body,
                    variable_name,
                )
            else:
                kernel_body = kernel_process.convert_1d_indexing(
                    kernel_body,
                    variable_name,
                )

        elif len(array_sizes) == 2:
            kernel_body = kernel_process.convert_2d_to_1d_indexing(
                kernel_body,
                variable_name,
                array_sizes[0],
            )

        elif len(array_sizes) == 3:
            kernel_body = kernel_process.convert_3d_to_1d_indexing(
                kernel_body,
                variable_name,
                array_sizes[0],
                array_sizes[1],
            )

    # ============================================================
    # Convert OPS_GBL arrays after local arrays have been processed.
    # Multi-dimensional arrays are handled first (3D -> 2D -> 1D)
    # to avoid incorrect partial replacements.
    # ============================================================
    sorted_ops_gbl_args.sort(
        key=lambda x: len(x[1]),
        reverse=True
    )

    for variable_name, array_sizes in sorted_ops_gbl_args:

        if len(array_sizes) == 1:

            if array_sizes[0].isdigit() and int(array_sizes[0]) == 0:
                kernel_body = kernel_process.replace_array_with_first_element(
                    kernel_body,
                    variable_name,
                )
            else:
                kernel_body = kernel_process.convert_1d_indexing(
                    kernel_body,
                    variable_name,
                )

        elif len(array_sizes) == 2:
            kernel_body = kernel_process.convert_2d_to_1d_indexing(
                kernel_body,
                variable_name,
                array_sizes[0],
            )

        elif len(array_sizes) == 3:
            kernel_body = kernel_process.convert_3d_to_1d_indexing(
                kernel_body,
                variable_name,
                array_sizes[0],
                array_sizes[1],
            )

    # ============================================================
    # Emit local variable declarations into generated C++ kernel
    # ============================================================
    for variable_declaration_with_type, name_and_sizes in local_vars.items():

        for array_size in name_and_sizes[1]:

            if not array_size.isdigit():

                if array_size in local_var_sizes.keys():

                    array_size_literal_value = local_var_sizes[array_size]

                    # When a kernel is not used in a particular test case,
                    # some dimension parameters may evaluate to zero.
                    # CUDA/HIP do not allow zero-sized arrays, therefore
                    # use size 1 as a safe fallback.
                    if array_size_literal_value == 0:
                        array_size_literal_value = 1

                    variable_declaration_with_type = replace_variable(
                        variable_declaration_with_type,
                        array_size,
                        array_size_literal_value,
                    )

                else:
                    raise ParseError(
                        f"Unable to find varible {array_size}'s literal value, "
                        f"please check if declared in constants.F90"
                    )

        cpp_kernel += (
            f"    {variable_declaration_with_type};\n"
        )

    cpp_kernel += "\n"

    if len(var_init) > 0:
        cpp_kernel += var_init
        cpp_kernel += "\n\n"

    # ============================================================
    # Assemble final generated C++ kernel source
    # ============================================================
    cpp_kernel += kernel_body
    cpp_kernel += "\n}"

    return cpp_kernel


def retrieve_subroutine_ast(file_path, subroutine_name):
    if not os.path.exists(file_path):
        raise ParseError(f"Unable to find file {file_path} for subroutine: {subroutine_name}")

    ftn_source = retrieve_subroutine_by_name_regex(file_path, subroutine_name)
    if ftn_source is None or (ftn_source is not None and len(ftn_source) == 0):
        raise ParseError(f"unable to find kernel function: {subroutine_name}")

#    # find if there is any nested subroutine/function calls inside elemental kernel and retrieve those as well
#    pattern = r"\bCALL\s+([a-zA-Z0-9_]+)\s*\(([^)]*)\)"
#    # Find all matches
#    subroutine_calls = re.findall(pattern, ftn_source.strip(), re.IGNORECASE)
#
#    for match in subroutine_calls:
#        subroutine_call, args = match
#        # Determine the filename and retrieve the corresponding subroutine code
#        filename = subroutine_call[:subroutine_call.find("kernel")]+"kernel.inc"
#        # Retrieve the subroutine code from the file or other sources
#        sub_kernel = retrieve_subroutine_by_name_regex(filename, subroutine_call)
#        if sub_kernel is None or (sub_kernel is not None and len(sub_kernel) == 0):
#            raise ParseError(f"unable to find kernel function: {sub_kernel}")
#        ftn_source = sub_kernel + "\n" + ftn_source

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
    result_src = re.sub(pattern, replace_function, ftn_source, flags=re.IGNORECASE)

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


def retrieve_subroutine_and_nestedsubroutines(loop_kernel):

    filename = loop_kernel[:loop_kernel.find("kernel")]+"kernel.inc"

    #kernel_entities = retrieve_subroutine_by_name(filename, loop_kernel)
    kernel_entities = retrieve_subroutine_by_name_regex(filename, loop_kernel)

    if kernel_entities is None or (kernel_entities is not None and len(kernel_entities) == 0):
        raise ParseError(f"unable to find kernel function: {loop_kernel}")

    # find if there is any nested subroutine/function calls inside elemental kernel and retrieve those as well
    pattern = r"\bCALL\s+([a-zA-Z0-9_]+)\s*\(([^)]*)\)"
    # Find all matches
    subroutine_calls = re.findall(pattern, kernel_entities.strip(), re.IGNORECASE)

    modified_kernel = kernel_entities.strip()

    kernel_founds = []

    sub_kernels = []
    for match in subroutine_calls:

        subroutine_call, args = match

        if subroutine_call in kernel_founds:
            continue

        # Modify the subroutine call in the original kernel code
        modified_call = f"CALL {loop_kernel}_{subroutine_call}({args})"

        # modified_kernel = re.sub(rf"\bCALL\s+{re.escape(subroutine_call)}\s*\({re.escape(args)}\)", modified_call, modified_kernel, flags=re.IGNORECASE)
        modified_kernel = re.sub(rf"\b{re.escape(subroutine_call)}\b", f"{loop_kernel}_{subroutine_call}", modified_kernel)

        # Determine the filename and retrieve the corresponding subroutine code
        filename = subroutine_call[:subroutine_call.find("kernel")]+"kernel.inc"

        # Retrieve the subroutine code from the file or other sources
        sub_kernel = retrieve_subroutine_by_name_regex(filename, subroutine_call)

        if sub_kernel is None or (sub_kernel is not None and len(sub_kernel) == 0):
            raise ParseError(f"unable to find kernel function: {sub_kernel}")

        # Replace the original subroutine name in the sub_kernel with the new modified name
        modified_sub_kernel = re.sub(rf"\b{re.escape(subroutine_call)}\b", f"{loop_kernel}_{subroutine_call}", sub_kernel.strip())

        sub_kernels.append([f"{loop_kernel}_{subroutine_call}",modified_sub_kernel])
        kernel_founds.append(f"{subroutine_call}")

    return modified_kernel, sub_kernels


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

        #kernel_entities, sub_kernels = retrieve_subroutine_and_nestedsubroutines(loop.kernel)
        #return kernel_entities, sub_kernels

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

#        print("=========================")
#        print(cpp_kernel)

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

        #kernel_entities, sub_kernels = retrieve_subroutine_and_nestedsubroutines(loop.kernel)

        # Replace KernelName with KernelName_gpu
        replacement_string = loop.kernel + "_gpu"

        # Pattern: match 'subroutine' (any case) + spaces + kernel name
        pattern = r'(\bsubroutine\s+)' + re.escape(loop.kernel) + r'\b'

        # Replace with 'SUBROUTINE ' + kernel_gpu
        output_string = re.sub(pattern, r'SUBROUTINE ' + replacement_string, kernel_entities, flags=re.IGNORECASE)

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

        return output_string.strip()#, sub_kernels

Scheme.register(FortranCuda)


class F2CCuda(Scheme):
    lang = Lang.find("F90")
    target = Target.find("f2c_cuda")

    fallback = None

    loop_host_template = Path("fortran/f2c_cuda/loop_host.F90.j2")
    loop_host_f2c_template = Path("fortran/f2c_cuda/loop_f2c_host.cpp.j2")
    master_kernel_template = Path("fortran/f2c_cuda/master_kernel.cpp.j2")

    loop_kernel_extension = "F90"
    loop_kernel_f2c_extension = "cu"
    master_kernel_extension = "cu"

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

Scheme.register(F2CCuda)


class F2CHip(Scheme):
    lang = Lang.find("F90")
    target = Target.find("f2c_hip")

    fallback = None

    loop_host_template = Path("fortran/f2c_cuda/loop_host.F90.j2")
    loop_host_f2c_template = Path("fortran/f2c_cuda/loop_f2c_host.cpp.j2")
    master_kernel_template = Path("fortran/f2c_cuda/master_kernel.cpp.j2")

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

Scheme.register(F2CHip)


class F2CSycl(Scheme):
    lang = Lang.find("F90")
    target = Target.find("f2c_sycl")

    fallback = None

    loop_host_template = Path("fortran/f2c_sycl/loop_host.F90.j2")
    loop_host_f2c_template = Path("fortran/f2c_sycl/loop_f2c_host.cpp.j2")
    master_kernel_template = Path("fortran/f2c_sycl/master_kernel.cpp.j2")

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

Scheme.register(F2CSycl)


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

        #kernel_entities, sub_kernels = retrieve_subroutine_and_nestedsubroutines(loop.kernel)
        #return kernel_entities.strip(), sub_kernels

Scheme.register(FortranOpenMPOffload)
