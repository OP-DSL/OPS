import re

from typing import List

from ops import Const, OpsError, ArgDat, ArgGbl, ArgIdx, ArgReduce
from store import Program
from util import SourceBuffer, Rewriter, findIdx
import logging

# Augment source program to use generated kernel hosts
def translateProgram(source: str, program: Program, app_consts: List[Const], force_soa: bool = False) -> str:
    buffer = SourceBuffer(source)

    # 1. Update const calls
    for const in program.consts:
        buffer.apply(
            const.loc.line -1,
            lambda line: re.sub(r"ops_decl_const\s*\(", f"ops_decl_const2(", line)
        )

    # 2. Update loop calls
    for loop in program.loops:
        before, after = buffer.get(loop.loc.line - 1).split("ops_par_loop", 1)
        after = re.sub(
            rf"{loop.kernel}\s*,\s*", "", after, count=1
        ) #TODO: This assumes that the kernel argument is on the same line as the call
        buffer.update(loop.loc.line -1, before + f"ops_par_loop_{loop.kernel}" + after)

    # 3. Update headers
    index = buffer.search(r'\s*#include\s+("|<)\s*ops_seq(_v2)?\.h\s*("|>)') + 2

    buffer.insert(index, '/* ops_par_loop declarations */\n')

    existingLoopProtypes = []

    for loop in program.loops:
        existingIdx = findIdx(existingLoopProtypes, lambda l: l.kernel == loop.kernel and len(l.args) == len(loop.args))
        if existingIdx is None:
            existingLoopProtypes.append(loop)
            prototype = f'void ops_par_loop_{loop.kernel}(char const *, ops_block, int, int*{", ops_arg" * len(loop.args)});\n'
            buffer.insert(index, prototype)

    # 4. Update ops_init
    buffer.insert(0, '\nvoid ops_init_backend();\n')

    if buffer.search(r'\s* ops_init\('):
        index = buffer.search(r'\s* ops_init\(') + 1
        buffer.insert(index, '\tops_init_backend();\n')

    # 5. Find the Global declarations of constant and add relevant OpenACC pragmas
    # Add #pragma acc declare create() near main declarations of variable (no need to place near extern declarations)
    # Add #pragma acc update device() near initialization of those variables
    dynamic_const = []
    for const in app_consts:
        name, typ = const.name, repr(const.typ)
        pattern1 = rf'\s*\b{typ}\b(?:(?!;).)*\b{name}\b'
        pattern2 = rf'\s*extern \s*\b{typ}\b(?:(?!;).)*\b{name}\b'
        # Check if main declaration of global is present in file, skip for extern declarations
        if buffer.search2(pattern1) is not None and buffer.search2(pattern2) is None:
            index = buffer.search2(pattern1)
            #print(f"found {name}")
            if const.dim.isdigit() and int(const.dim)==1:
                buffer.insert(index+1, f'#pragma acc declare create({const.ptr})')
            else:
                rawline = buffer.get(index)
                match = re.search(r'.*{name}\s+\[.*\]', rawline)
                if match:
                    # if static allocation
                    buffer.insert(index+1, f'#pragma acc declare create({const.ptr}[0:{const.dim}])')
                else:
                    # if global pointer
                    #print(f"Please make sure ops_decl_const call for {const.ptr} is in same file where it is declared as pointer globally (not where it is extern)")
                    buffer.insert(index+1, f'#pragma acc declare create({const.ptr})')
                    dynamic_const.append(name)


    # Find all ops_decl_const calls and add update directives
    pattern3 = r'ops_decl_const\s*\('
    indexes = buffer.search_all(pattern3)
    for index in indexes:
        rawline = buffer.get(index)
        pattern = r'ops_decl_const\s*\(\s*"([^"]+)"'
        match = re.search(pattern, rawline)
        if match:
            name = match.group(1)
            for const in app_consts:
                if name.strip().lower() == const.name.strip().lower():
                    dim, ptr = const.dim, const.ptr
                    if dim.isdigit() and int(dim) == 1:
                        buffer.insert(index+1, f'#pragma acc update device({ptr})')
                    else:
                        # TODO : Assuming main global declaration and ops_decl_const call is in same file
                        if name in dynamic_const:
                            buffer.insert(index+1, f'#pragma acc enter data create({ptr}[0:{const.dim}])')
                        buffer.insert(index+1, f'#pragma acc update device({ptr}[0:{dim}])')
        else:
            raise OpsError(f"Unable to extract name from ops_decl_const line: {rawline}")

    # 6. Translation
    new_source = buffer.translate()

    # 7. Substitude the ops_seq.h/ops_seq_v2.h with ops_lib_core.h
    new_source = re.sub(r'#include\s+("|<)\s*ops_seq(_v2)?\.h\s*("|>)', '#include "ops_lib_core.h"', new_source)

    # 8. check if SOA is set
    def replacer(match):
        s = match.group(0)
        if s.startswith("/"):
            return ""
        else:
            return s

    pattern_comment = re.compile(r'//.*?$|/\*.*?\*/|\'(?:\\.|[^\\\'])*\'|"(?:\\.|[^\\"])*"',
                                 re.DOTALL | re.MULTILINE,
                                )

    pattern = r'(#define\s*OPS_SOA|OPS_soa\s*=\s*1\s*;)'
    matches = re.findall(pattern, re.sub(pattern_comment, replacer, new_source), re.IGNORECASE)

    if len(matches) == 2 and not program.soa_val:
        program.soa_val = True

    # Return new updated source
    return new_source


# Augment source program to use generated kernel hosts
def translateProgramHLS(source: str, program: Program, app_consts: List[Const], force_soa: bool = False) -> str:
    buffer = SourceBuffer(source)

    # 1. Update const calls
    for const in program.consts:
        # buffer.apply(
        #     const.loc.line -1,
        #     lambda line: re.sub(r"ops_decl_const\s*\(", f"ops_decl_const2(", line)
        # )
        buffer.remove(const.loc.line -1)

    # 2. Update loop calls
    
    for iterloop in program.outerloops:
        startLoc = iterloop.scope[0]
        
        before, after = buffer.get(startLoc.line - 1).split("ops_iter_par_loop", 1)
        loop_indices = [startLoc.line - 1]
        
        if (after.find(";") == -1):
            index = startLoc.line
            loop_indices.append(index)
            while(True):
                line = buffer.get(index)
                after += line
                buffer.remove(index)
                if line.find(";") != -1:
                    break
                index += 1
        [split_after, split_retain] = after.split(";", 1)
        split_after = split_after.split(",")
        
        new_iter_loop_call = f"{iterloop.unique_name}({split_after[1]}, {iterloop.ops_range}"
        
        for arg in iterloop.joint_args:
            new_iter_loop_call += f", {iterloop.dats[arg.dat_id][0].ptr}"
        
        new_iter_loop_call = before + new_iter_loop_call + ");" + split_retain 
        

        buffer.remove(startLoc.line - 1)
        
        buffer.update(index, new_iter_loop_call)
        
        
    for loop in program.loops:
        if loop.iterativeLoopId != -1:
            continue
        
        before, after = buffer.get(loop.loc.line - 1).split("ops_par_loop", 1)
        loop_indices = [loop.loc.line - 1]
        
        if (after.find(";") == -1):
            index = loop.loc.line
            loop_indices.append(index)
            while(True):
                line = buffer.get(index)
                after += line
                buffer.remove(index)
                if line.find(";") != -1:
                    break
                index += 1
                
        # print ("After: ", after)
        split_after = after.split(",")
        new_loop_call = before + f"ops_par_loop_{loop.kernel}({split_after[2]}, {split_after[3]} , {split_after[4]}"
        
        for arg in loop.args:
            if isinstance(arg, ArgDat):
                dat = loop.dats[arg.dat_id]
                new_loop_call = f"{new_loop_call}, {dat.ptr}"
        new_loop_call = f'{new_loop_call});' 
        
        for index in loop_indices:
            buffer.remove(index)
        
        buffer.update(index, new_loop_call)

    # 3. Update headers
    index = buffer.search(r'\s*#include\s*("|<)\s*ops_seq(_v2)?\.h\s*("|>)') + 2

    buffer.insert(index, '/* ops_par_loop declarations */\n')

    existingLoopProtypes = []

    for loop in program.loops:
        existingIdx = findIdx(existingLoopProtypes, lambda l: l.kernel == loop.kernel and len(l.args) == len(loop.args))
        
        if (existingIdx is None) and (loop.iterativeLoopId == -1):
            existingLoopProtypes.append(loop)
            prototype = f'void ops_par_loop_{loop.kernel}(ops::hls::Block, int, int*'
            isArgIdx = loop.arg_idx == 1
            
            for arg in loop.args:
                if isinstance(arg, ArgDat):
                    dat = loop.dats[arg.dat_id]
                    prototype = f'{prototype}, ops::hls::Grid<{dat.typ}>&'
                    
                #elif isinstance(arg, ArgGbl): TODO: ArgGbl 
                #TODO: ArgReduce
            
            prototype = f'{prototype});\n'        
            # {", ops::hls::Grid" * len(loop.args)});\n'
            buffer.insert(index, prototype)

    # 4. Update ops_init

    if buffer.search(r'\s* ops_init\s*\('):
        index = buffer.search(r'\s* ops_init\s*\(')
        buffer.update(index, '\tops_init_backend(argc, argv);\n')

    # 5. Update ops_exit
    if buffer.search(r'\s* ops_exit\s*\('):
        index = buffer.search(r'\s* ops_exit\s*\(')
        buffer.update(index, '\tops_exit_backend();\n')
    
    # 6. Find the Global declarations of constant and check extern or not
    for const in app_consts:
        name, typ = const.name, repr(const.typ)
        pattern1 = rf'\s*\b{typ}\b(?:(?!;).)*\b{name}\b'
        pattern2 = rf'\s*extern \s*\b{typ}\b(?:(?!;).)*\b{name}\b'
        # Check if main declaration of global is present in file, skip for extern declarations
        
        # TODO: Have to find a mechanism to do this
        # if buffer.search2(pattern1) is not None and buffer.search2(pattern2) is None:
        #     const.setExtern(False)
        # elif buffer.search2(pattern2) is not None:
        #     const.setExtern(True) 
    
    
    # 7. Removing stencil declaration
    if buffer.search(r'\s*ops_stencil.*'):
        line_indices = buffer.search_all(r'\s*ops_stencil.*')
    
        # print(f"Found ops_stencils ({len(line_indices)})")  
        for start_index in line_indices:
            index = start_index
            while(True):
                line = buffer.get(index)
                buffer.remove(index)
                if line.find(";") != -1:
                    break
                index += 1
                
    if buffer.search(r'.*ops_decl_stencil.*'):
        line_indices = buffer.search_all(r'.*ops_decl_stencil.*')
    
        # print(f"Found ops_stencils ({len(line_indices)})")  
        for start_index in line_indices:
            index = start_index
            while(True):
                line = buffer.get(index)
                buffer.remove(index)
                if line.find(";") != -1:
                    break
                index += 1
        
    # 8. Removing ops_partition        
    if (buffer.search(r'.*ops_partition.*')):
        index = buffer.search(r'.*ops_partition.*')
        buffer.remove(index)
    
    # 9. Add kernel_wrapp_master_kernels include
    if (buffer.search(r'#include\s+("|<)\s*ops_seq(_v2)?\.h\s*("|>)')):
        index = buffer.search(r'#include\s+("|<)\s*ops_seq(_v2)?\.h\s*("|>)')
        buffer.insert(index+1, "#include <hls_kernels.hpp>")
    else:
        raise OpsError(f"OPS program failed to include core header file, ops_seq.h or ops_seq_V2.h")

    # 10. get_raw_pointer update
    found_indices = buffer.search_all(r'.*ops_dat_get_raw_pointer.*')
    print (f"found_indices: {found_indices}")
    
    for index in found_indices:
        before, after = buffer.get(index).split("ops_dat_get_raw_pointer", 1)
        # print (f"before: {before}, after: {after}")
        loop_indices = [index]
        
        if (after.find(";") == -1):
            index = startLoc.line
            loop_indices.append(index)
            while(True):
                line = buffer.get(index)
                after += line
                buffer.remove(index)
                if line.find(";") != -1:
                    break
                index += 1
        [split_after, split_retain] = after.split(";", 1)
        
        dat_name = split_after.split(",", 1)[0].replace('(', '')
        
        new_call_line = before + f"{dat_name}.get_raw_pointer()" + ";" + split_retain 
        buffer.update(index, new_call_line)
    
    found_indices = buffer.search_all(r'.*->.*get_raw_pointer.*')
    print (f"found_indices: {found_indices}")
    
    for index in found_indices:
        before, after = buffer.get(index).split("get_raw_pointer", 1)
        print (f"before: {before}, after: {after}")
        loop_indices = [index]
        
        if (after.find(";") == -1):
            index = startLoc.line
            loop_indices.append(index)
            while(True):
                line = buffer.get(index)
                after += line
                buffer.remove(index)
                if line.find(";") != -1:
                    break
                index += 1
        [split_after, split_retain] = after.split(";", 1)
        
        new_call_line = before.replace("->", ".") + f"get_raw_pointer()" + ";" + split_retain 
        buffer.update(index, new_call_line)
    new_source = buffer.translate()
    
    # 11. Replace ops_block to ops::hls::Block and replace ops_decl_block to ops_hls_decl_block
    new_source = new_source.replace("ops_block", "ops::hls::Block").replace("ops_decl_block", "ops_hls_decl_block")
    
    # 12. Replace ops_dat to auto and ops_decl_dat to ops_hls_decl_dat
    # TODO: Make the ops_dat to ops::hls::Grid without type defined. to support predefined ops_dat without declaration. 
    #       right now simply translating ops_dat to ops::hls::Grid<stencil_type>
    new_source = new_source.replace("ops_dat", "ops::hls::Grid<stencil_type>").replace("ops_decl_dat", "ops_hls_decl_dat")
    
    # 13. Replace ops_printf
    new_source = new_source.replace("ops_printf", "printf")
    
    # 14. Substitude the ops_seq.h/ops_seq_v2.h with ops_lib_core.h
    new_source = re.sub(r'#include\s+("|<)\s*ops_seq(_v2)?\.h\s*("|>)', '#include <ops_hls_rt_support.h>', new_source)

    # 15. check if SOA is set
    def replacer(match):
        s = match.group(0)
        if s.startswith("/"):
            return ""
        else:
            return s

    pattern_comment = re.compile(r'//.*?$|/\*.*?\*/|\'(?:\\.|[^\\\'])*\'|"(?:\\.|[^\\"])*"',
                                 re.DOTALL | re.MULTILINE,
                                )

    pattern = r'(#define\s*OPS_SOA|OPS_soa\s*=\s*1\s*;)'
    matches = re.findall(pattern, re.sub(pattern_comment, replacer, new_source), re.IGNORECASE)

    if len(matches) == 2 and not program.soa_val:
        program.soa_val = True

    # Return new updated source
    return new_source
