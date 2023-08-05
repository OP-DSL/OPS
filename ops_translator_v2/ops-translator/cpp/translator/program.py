import re

from store import Program
from util import SourceBuffer, Rewriter, findIdx

# Augment source program to use generated kernel hosts
def translateProgram(source: str, program: Program, force_soa: bool = False) -> str:
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

    buffer.insert(index, '/*\n** ops_par_loop declarations\n*/\n')     
    buffer.insert(index, '#ifdef OPENACC\n#ifdef __cplusplus\nextern "C" {\n#endif\n#endif\n')

    existingLoopProtypes = []

    for loop in program.loops:
        existingIdx = findIdx(existingLoopProtypes, lambda l: l.kernel == loop.kernel and len(l.args) == len(loop.args))
        if existingIdx is None:
            existingLoopProtypes.append(loop)
            prototype = f'void ops_par_loop_{loop.kernel}(char const *, ops_block, int, int*{", ops_arg" * len(loop.args)});\n'
            buffer.insert(index, prototype)

    buffer.insert(index, '#ifdef OPENACC\n#ifdef __cplusplus\n}\n#endif\n#endif\n')

    # 4. Update ops_init
    buffer.insert(0, '\nvoid ops_init_backend();\n')

    if buffer.search(r'\s* ops_init\('):
        index = buffer.search(r'\s* ops_init\(') + 1
        buffer.insert(index, '\tops_init_backend();\n')

    # 5. Translation
    new_source = buffer.translate()

    # 6. Substitude the ops_seq.h/ops_seq_v2.h with ops_lib_core.h
    new_source = re.sub(r'#include\s+("|<)\s*ops_seq(_v2)?\.h\s*("|>)', '#include "ops_lib_core.h"', new_source)

    # 7. check if SOA is set
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

    return new_source

def translateProgram2(source: str, program: Program, force_soa: bool = False) -> str:
    rewriter = Rewriter(source)

    # 1. Update const calls
