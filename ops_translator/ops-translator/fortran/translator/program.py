import re
import sys

import fparser.two.Fortran2003 as f2003
import fparser.two.utils as fpu

from store import Program

#use_regex_translator = true
#def translateProgram2(program: Program, force_soa: bool) -> str:
#

def translateProgram(program: Program, force_soa: bool) -> str:
    ast = program.ast
    req_module = {}
    locations = []
    const_list = []

    # 1. comment const calls
    for call in fpu.walk(ast, f2003.Call_Stmt):
        name = fpu.get_child(call, f2003.Name)
        if name is None or name.string.lower() != "ops_decl_const":
            continue

        args = fpu.get_child(call, f2003.Actual_Arg_Spec_List)
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
        args.items = tuple(arg_list[1:])

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

        # Update par loop name by adding kernel name to ops_par_loop
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
