import os
import re
import sys

from typing import List

import fparser.two.Fortran2003 as f2003
import fparser.two.utils as fpu

import ops as OPS
from store import Program

def translateProgram(program: Program, force_soa: bool, offload_pragma_flag_dict: dict) -> str:
    ast = program.ast
    req_module = {}
    locations = []
    const_list = []
    const_list_dim = []

    # 1. comment const calls
    for call in fpu.walk(ast, f2003.Call_Stmt):
        name = fpu.get_child(call, f2003.Name)
        if name is None or name.string.lower() != "ops_decl_const":
            continue

        args = fpu.get_child(call, f2003.Actual_Arg_Spec_List)
        const_list_dim.append(str(list(args.items)[1]))
        const_list.append(str(list(args.items)[3])) 

    #print(const_list)
    # Write all constants to file, required to replace the constantname with constantname_opsconstant for CUDA kernels generated
    with open('constants_list.txt', 'a') as file:
        file.writelines([item + '\n' for item in const_list])

    # 2. Update loop calls
    for call in fpu.walk(ast, f2003.Call_Stmt):
        name = fpu.get_child(call, f2003.Name)

        if name is None or name.string.lower() != "ops_par_loop":
            continue

        args = fpu.get_child(call, f2003.Actual_Arg_Spec_List)
        arg_list = list(args.items)

        kernel_name = arg_list[0].string.lower()

        # Update the argument list to point from 1st index onwards emitting 0th index which contain name of par loop
        #args.items = tuple(arg_list[1:])

        argument_string = ", ".join(arg.tostr() for arg in arg_list[1:5]) + ", &\n"
        argument_string += "".join(arg.tostr() + ", &\n" for arg in arg_list[5:])[:-4]
        new_arg_list = [argument_string]
        args.items = tuple(new_arg_list)

        # if kernel name not in dictionary, add it to dictionary along with arguments
        if kernel_name not in req_module:
            req_module[kernel_name] =  arg_list[3:]
        else:
            # if kernel name in dictionary, check if args matching for dim, stencil, type and access
            repeat = True
            cur_arg_list = arg_list[3:]
            from_dict = req_module[kernel_name]
            if len(from_dict) != len(cur_arg_list):
                repeat = False
            else:
                # ops_par_loop dims parameter mismatch
                if cur_arg_list[0] != from_dict[0]:
                    repeat = False
                    break
                for indx in range(2,len(cur_arg_list)):
                    arg_name1 = list(cur_arg_list[indx].items)[0]
                    arg_name2 = list(from_dict[indx].items)[0]    

                    # if argument is ops_arg_dat, ops_arg_dat_opt, ops_arg_gbl or ops_arg_reduce
                    # check if values match for n-species, stencil, type and access
                    if ( arg_name1.string.lower() == arg_name2.string.lower() 
                        and arg_name1.string.lower() != "ops_arg_idx" ) :
                        args1 = (list(cur_arg_list[indx].items)[1]).items[1:]
                        args2 = (list(from_dict[indx].items)[1]).items[1:]
                        if args1 != args2:
                            repeat = False

            if not repeat:
                print("Error!!! Repeated kernel with incompatible arguments for kernel: "+kernel_name)
                sys.exit()

        # Update subroutine name by replacing ops_par_loop with kernelname_host
        name.string = f"{kernel_name}_host"

    # 3. Update headers
    #    Add use module statement for ops_par_loops referenced in current file
    for main_program in fpu.walk(ast, (f2003.Main_Program,
                                       f2003.Function_Subprogram,
                                       f2003.Subroutine_Subprogram,
                                       f2003.Module)):
        spec = fpu.get_child(main_program, f2003.Specification_Part)
        new_content = []

        for node in spec.content:
            if ( isinstance(node, f2003.Use_Stmt) 
                and fpu.get_child(node, f2003.Name).string.lower() == "ops_fortran_reference" 
               ):
                new_content.append(f2003.Use_Stmt("use ops_fortran_declarations"))
                new_content.append(f2003.Use_Stmt("use ops_fortran_rt_support"))
                for key in req_module.keys():
                    new_statement = "use "+key+"_module"
                    new_content.append(f2003.Use_Stmt(new_statement))
            else:
                new_content.append(node)

        spec.content = new_content

    temp_source = str(ast)
    # Comment the call to ops_decl_const, no implementation needed in Fortran
    pattern = r"(?i)(call|CALL)\sops_decl_const\(.*?\)"
    new_source = re.sub(pattern, r"!\g<0>", temp_source)

    # 5. add the omp target directives for constants
    content_to_append = ""
    if len(const_list): # Contain call to ops_decl_const
        content_to_append += "\n#ifdef OPS_WITH_OMPOFFLOADFOR\n"
        for dim,name in zip(const_list_dim,const_list):
            if len(offload_pragma_flag_dict) and offload_pragma_flag_dict.get(name):
                if dim.isdigit() and int(dim) == 1:
                    content_to_append += f"!$OMP TARGET UPDATE TO({name})\n"
                else:
                    content_to_append += f"!$OMP TARGET UPDATE TO({name}(1:{dim}))\n"
        content_to_append += "#endif\n"

        # Find the last occurance of ops_decl_const in the file and append this contents
        pattern = re.compile(r'call\s+ops_decl_const\(', re.IGNORECASE)
        matches = list(pattern.finditer(new_source))

        if matches:
            last_occurrence = matches[-1]
            next_line_start = new_source.find('\n', last_occurrence.end())
            modified_new_source = (
                new_source[:next_line_start] + content_to_append + new_source[next_line_start:]
            )
            return unindent_cpp_directives(modified_new_source)

    return unindent_cpp_directives(new_source)


def unindent_cpp_directives(s: str) -> str:
    directives = [
        "if",
        "ifdef",
        "ifndef",
        "elif",
        "else",
        "endif",
        "include",
        "define",
        "undef",
        "line",
        "error",
        "warning",
    ]

    return re.sub(rf"^\s*#({'|'.join(directives)})(\s+|\s*$)", r"#\1\2", s, flags=re.MULTILINE)


def add_offload_directives(app_consts: List[OPS.Const], offload_pragma_flag_dict: dict):
    file_path = 'constants.F90'
    if os.path.exists(file_path):
        with open('constants.F90', 'r') as file:
            file_content = file.read()

        contents_to_append = ""
        contents_to_append += "\n#ifdef OPS_WITH_OMPOFFLOADFOR\n"
        # For each const variable in app_consts, add the required pragma for ompoffload
        for const in app_consts:
            dim = const.dim
            ptr = const.ptr
            if len(offload_pragma_flag_dict) and offload_pragma_flag_dict.get(ptr):
                if dim.isdigit() and int(dim) == 1:
                    contents_to_append += f"!$OMP DECLARE TARGET({ptr})\n"
                else:
                    contents_to_append += f"!$OMP DECLARE TARGET({ptr}(1:{dim}))\n"
        contents_to_append += "#endif\n"

        # Find the last occurrence of #endif in file_content
        last_endif_index = file_content.rfind("#endif")

        # Insert contents_to_append before the last #endif
        updated_content = file_content[:last_endif_index] + contents_to_append + file_content[last_endif_index:]

        # Write the updated content to a new file (constants_offload.F90)
        output_file_path = 'constants_offload.F90'
        with open(output_file_path, 'w') as output_file:
            output_file.write(updated_content)
    else:
        return


def check_offload_pragma_required(app_consts: List[OPS.Const]):
    file_path = 'constants.F90'
    if os.path.exists(file_path):
        with open('constants.F90', 'r') as file:
            file_content = file.read()

    offload_pragma_flag = {}

    for const in app_consts:
        ptr = const.ptr
#       pattern = r'\b{}\s*=\s*[^,\n]*'.format(re.escape(ptr))
        pattern = r'parameter.*\b{}\s*=\s*[^,\n]*'.format(re.escape(ptr))

        matches = re.findall(pattern, file_content, flags=re.IGNORECASE)

        if len(matches) > 0:
            offload_pragma_flag[ptr] = False
        else:
            offload_pragma_flag[ptr] = True
    return offload_pragma_flag
