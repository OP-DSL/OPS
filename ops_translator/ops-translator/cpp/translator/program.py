import re

from typing import List

from ops import Const, OpsError
from store import Program
from util import SourceBuffer, Rewriter, findIdx

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
