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
## @brief
#
#  utility functions for code generator
#
"""
utility functions for code generator

"""

import re
import glob
import os
import config
from config import OPS_READ
from datetime import datetime

def comm(line):
    prefix = " " * config.depth
    if len(line) == 0:
        config.file_text += "\n"
    else:
        config.file_text += f"{prefix}//{line}\n"


def code(text):
    prefix = ""
    if len(text) != 0:
        prefix = " " * config.depth

    config.file_text += f"{prefix}{text}\n"


def FOR(i, start, finish):
    code(f"for (int {i} = {start}; {i} < {finish}; {i}++) {{")
    config.depth += 4


def FOR2(i, start, finish, increment):
    code(f"for (int {i} = {start}; {i} < {finish}; {i}+={increment}) {{")
    config.depth += 4


def WHILE(line):
    code(f"while ({line}){{")
    config.depth += 4


def ENDWHILE():
    config.depth -= 4
    code("}")


def ENDFOR():
    config.depth -= 4
    code("}")


def IF(line):
    code(f"if ({line}) {{")
    config.depth += 4


def ELSEIF(line):
    code(f"else if ({line}) {{")
    config.depth += 4


def ELSE():
    code("else {")
    config.depth += 4


def ENDIF():
    config.depth -= 4
    code("}")


def mult(text, i, n):
    text = text + "1"
    for nn in range(0, i):
        text = f"{text}* args[{n}].dat->size[{nn}]"

    return text


def para_parse(text, j, op_b, cl_b):
    """Parsing code block, i.e. text to find the correct closing brace"""

    depth = 0
    loc2 = j

    while 1:
        if text[loc2] == op_b:
            depth = depth + 1

        elif text[loc2] == cl_b:
            depth = depth - 1
            if depth == 0:
                break
        loc2 = loc2 + 1
    return loc2


def comment_remover(text):
    """Remove comments from text"""

    def replacer(match):
        s = match.group(0)
        if s.startswith("/"):
            return ""
        else:
            return s

    pattern = re.compile(
        r'//.*?$|/\*.*?\*/|\'(?:\\.|[^\\\'])*\'|"(?:\\.|[^\\"])*"',
        re.DOTALL | re.MULTILINE,
    )
    return re.sub(pattern, replacer, text)


def remove_trailing_w_space(text):
    return "\n".join(map(lambda line: line.rstrip(), text.split("\n")))


def arg_parse_list(text, j):
    """Parsing arguments in function to find the correct closing brace"""

    depth = 0
    loc2 = j
    arglist = []
    prev_start = j
    while 1:
        if text[loc2] == "(":
            if depth == 0:
                prev_start = loc2 + 1
            depth = depth + 1

        elif text[loc2] == ")":
            depth = depth - 1
            if depth == 0:
                arglist.append(text[prev_start:loc2].strip())
                break

        elif text[loc2] == ",":
            if depth == 1:
                arglist.append(text[prev_start:loc2].strip())
                prev_start = loc2 + 1
        elif text[loc2] == "{":
            depth = depth + 1
        elif text[loc2] == "}":
            depth = depth - 1
        loc2 = loc2 + 1
    return arglist


def parse_replace_ACC_signature(text, arg_typ, dims, opencl=0, accs=[], typs=[]):
    for i in range(0, len(dims)):
        if arg_typ[i] == "ops_arg_dat":
            if not dims[i].isdigit() or int(dims[i]) > 1:
                text = re.sub(r"ACC<([a-zA-Z0-9]*)>\s*&", r"ptrm_\1 ", text, 1)
            else:
                text = re.sub(r"ACC<([a-zA-Z0-9]*)>\s*&", r"ptr_\1 ", text, 1)
        elif (
            opencl == 1
            and arg_typ[i] == "ops_arg_gbl"
            and accs[i] == 1
            and (not dims[i].isdigit() or int(dims[i]) > 1)
        ):
            # if multidim global read, then it is passed in as a global pointer, otherwise it's local
            args = text.split(",")
            text = ""
            for j in range(0, len(args)):
                if j == i:
                    text = (
                        text
                        + args[j]
                        .replace(typs[j], "__global " + typs[j])
                        .replace("*", "* restrict ")
                        + ", "
                    )
                else:
                    text = text + args[j] + ", "
            text = text[:-2]

    return text


def convert_ACC_signature(text, arg_typ):
    arg_list = arg_parse_list(text, 0)
    for i in range(0, len(arg_list)):
        if arg_typ[i] == "ops_arg_dat" and not ("ACC" in arg_list[i]):
            arg_list[i] = re.sub(r"\bint\b", "ACC<int>", arg_list[i])
            arg_list[i] = re.sub(r"\bfloat\b", "ACC<float>", arg_list[i])
            arg_list[i] = re.sub(r"\bdouble\b", "ACC<double>", arg_list[i])
            arg_list[i] = re.sub(r"\blong long\b", "ACC<long long>", arg_list[i])
            arg_list[i] = re.sub(r"[^<]\blong\b", "ACC<long>", arg_list[i])
            arg_list[i] = re.sub(r"\bll\b", "ACC<ll>", arg_list[i])
            arg_list[i] = re.sub(r"\bshort\b", "ACC<short>", arg_list[i])
            arg_list[i] = re.sub(r"\bchar\b", "ACC<char>", arg_list[i])
            arg_list[i] = re.sub(r"\bcomplexf\b", "ACC<complexf>", arg_list[i])
            arg_list[i] = re.sub(r"\bcomplexd\b", "ACC<complexd>", arg_list[i])
            arg_list[i] = arg_list[i].replace("*", "&")
    signature = ""
    for i in range(0, len(arg_list)):
        signature = signature + arg_list[i] + ",\n  "
    return signature[:-4]


def convert_ACC_body(text):
    text = re.sub(r"\[OPS_ACC_MD[0-9]+(\([ -A-Za-z0-9,+]*\))\]", r"\1", text)
    text = re.sub(r"\[OPS_ACC[0-9]+(\([ -A-Za-z0-9,+]*\))\]", r"\1", text)
    return text


def convert_ACC(text, arg_typ):
    openb = text.find("(")
    closeb = text[0 : text.find("{")].rfind(")") + 1
    text = (
        text[0:openb]
        + f"({convert_ACC_signature(text[openb:closeb], arg_typ)})"
        + text[closeb:]
    )
    body_start = text.find("{")
    text = text[0:body_start] + convert_ACC_body(text[body_start:])
    return text


def parse_signature(text):
    text2 = re.sub(r"\bll\b", "", text)
    text2 = re.sub(r"\bconst\b", "", text2)
    text2 = re.sub(r"\bACC<", "", text2)
    text2 = re.sub(r">", "", text2)
    text2 = re.sub(r"\bint\b", "", text2)
    text2 = re.sub(r"\blong long\b", "", text2)
    text2 = re.sub(r"\blong\b", "", text2)
    text2 = re.sub(r"\bshort\b", "", text2)
    text2 = re.sub(r"\bchar\b", "", text2)
    text2 = re.sub(r"\bfloat\b", "", text2)
    text2 = re.sub(r"\bdouble\b", "", text2)
    text2 = re.sub(r"\bcomplexf\b", "", text2)
    text2 = re.sub(r"\bcomplexd\b", "", text2)
    text2 = text2.replace("*", "")
    text2 = text2.replace("&", "")
    text2 = text2.replace(")", "")
    text2 = text2.replace("(", "")
    text2 = text2.replace("\n", "")
    text2 = re.sub(r"\[[0-9]*\]", "", text2)
    arg_list = []
    args = text2.split(",")
    for n in range(0, len(args)):
        arg_list.append(args[n].strip())
    return arg_list


def find_consts(text, consts):
    found_consts = []

    for cn in range(0, len(consts)):
        pattern = consts[cn]["name"][1:-1]
        if re.search("\\b" + pattern + "\\b", text):
            found_consts.append(cn)

    return found_consts


def parse_signature_opencl(text2):
    text2 = text2.replace("*", "* restrict ")
    text2 = text2.replace("ll", "__global long long")
    text2 = text2.replace("long", "__global long")
    text2 = text2.replace("short", "__global short")
    text2 = text2.replace("char", "__global char")
    text2 = text2.replace("int", "__global int")
    text2 = text2.replace("float", "__global float")
    text2 = text2.replace("double", "__global double")
    return text2


def complex_numbers_cuda(text):
    """Handle complex numbers, and translate to the relevant CUDA function in cuComplex.h"""

    # Complex number assignment
    p = re.compile(
        r"([a-zA-Z_][a-zA-Z0-9]+)(\s+\_\_complex\_\_\s+)([a-zA-Z_][a-zA-Z0-9]*)\s*=\s*(.+)\s*;"
    )
    result = p.finditer(text)
    new_code = text
    complex_variable_names = []
    for match in result:
        complex_variable_names.append(match.group(3))
        rhs = match.group(4)
        if rhs in complex_variable_names:
            # Assignment of another complex variable already defined.
            if match.group(1) == "double":
                new_statement = f"cuDoubleComplex {match.group(3)} = {rhs};"
            elif match.group(1) == "float":
                new_statement = f"cuFloatComplex {match.group(3)} = {rhs};"
            else:
                continue
        else:
            # Assignment of a complex number in real and imaginary parts.
            p = re.compile(r"(\S+I?)\s*([+-]?)\s*(\S*I?)?")
            complex_number = p.search(rhs)
            assert complex_number is not None
            if complex_number.group(1)[-1] == "I":  # Real after imaginary part
                imag = complex_number.group(1)[:-1]
                if complex_number.group(3):  # If real part specified
                    real = complex_number.group(3)
                else:
                    real = "0.0"
            elif complex_number.group(3)[-1] == "I":  # Imaginary after real part
                if complex_number.group(2) == "-":
                    imag = "-" + complex_number.group(3)[:-1]
                else:
                    imag = complex_number.group(3)[:-1]
                if complex_number.group(1):  # If real part specified
                    real = complex_number.group(1)
                else:
                    real = "0.0"
            else:  # No imaginary part
                real = complex_number.group(0)
                imag = "0.0"
            if match.group(1) == "double":
                new_statement = f"cuDoubleComplex {match.group(3)} = make_cuDoubleComplex({real}, {imag});"
            elif match.group(1) == "float":
                new_statement = f"cuFloatComplex {match.group(3)} = make_cuFloatComplex({real}, {imag});"
            else:
                continue

        # Perform replacement.
        new_code = new_code.replace(match.group(0), new_statement)

    # Complex number __real__ and __imag__
    p = re.compile(r"(\_\_real\_\_)\s+([a-zA-Z_][a-zA-Z0-9]*)")
    result = p.finditer(new_code)
    for match in result:
        new_code = new_code.replace(match.group(0), f"cuCreal({match.group(2)})")
    p = re.compile(r"(\_\_imag\_\_)\s+([a-zA-Z_][a-zA-Z0-9]*)")
    result = p.finditer(new_code)
    for match in result:
        new_code = new_code.replace(match.group(0), f"cuCimag({match.group(2)})")

    # Multiplication of two complex numbers
    p = re.compile(r"([a-zA-Z_][a-zA-Z0-9]*)\s*\*\s*([a-zA-Z_][a-zA-Z0-9]*)")
    result = p.finditer(new_code)
    for match in result:
        if (
            match.group(1) in complex_variable_names
            or match.group(2) in complex_variable_names
        ):
            new_code = new_code.replace(
                match.group(0), f"cuCmul({match.group(1)}, {match.group(2)})"
            )

    return new_code


def arg_parse(text, j):
    """Parsing arguments in ops_par_loop to find the correct closing brace"""

    depth = 0
    loc2 = j
    while 1:
        if text[loc2] == "(":
            depth = depth + 1

        elif text[loc2] == ")":
            depth = depth - 1
            if depth == 0:
                return loc2
        loc2 = loc2 + 1


def check_accs(name, arg_list, arg_typ, text):
    for n in range(0, len(arg_list)):
        if arg_typ[n] == "ops_arg_dat":
            pos = 0
            while 1:
                match = re.search(f"\\b{arg_list[n]}\\b", text[pos:])
                if match == None:
                    break
                pos = pos + match.start(0)
                if pos < 0:
                    break
                pos = pos + len(arg_list[n])

                match0 = re.search(r"OPS_ACC_MD\d", text[pos:])
                match1 = re.search(r"OPS_ACC\d", text[pos:])

                if match0 != None:
                    if match1 != None:
                        if match0.start(0) > match1.start(0):
                            match = re.search(r"OPS_ACC\d", text[pos:])
                            assert match is not None
                            pos = pos + match.start(0)
                            pos2 = text[pos + 7 :].find("(")
                            num = int(text[pos + 7 : pos + 7 + pos2])
                            if num != n:
                                print(
                                    f"Access mismatch in {name}, arg {n}({arg_list[n]}) with OPS_ACC{num}"
                                )
                            pos = pos + 7 + pos2
                        elif match0.start(0) < match1.start(0):
                            match = re.search(r"OPS_ACC_MD\d", text[pos:])
                            assert match is not None
                            pos = pos + match.start(0)
                            pos2 = text[pos + 10 :].find("(")
                            num = int(text[pos + 10 : pos + 10 + pos2])
                            if num != n:
                                print(
                                    f"Access mismatch in {name}, arg {n}({arg_list[n]}) with OPS_ACC_MD{num}"
                                )
                            pos = pos + 10 + pos2
                    else:
                        match = re.search(r"OPS_ACC_MD\d", text[pos:])
                        assert match is not None
                        pos = pos + match.start(0)
                        pos2 = text[pos + 10 :].find("(")
                        num = int(text[pos + 10 : pos + 10 + pos2])
                        if num != n:
                            print(
                                f"Access mismatch in {name}, arg {n}({arg_list[n]}) with OPS_ACC_MD{num}"
                            )
                        pos = pos + 10 + pos2
                else:
                    if match1 != None:
                        match = re.search(r"OPS_ACC\d", text[pos:])
                        assert match is not None
                        pos = pos + match.start(0)
                        pos2 = text[pos + 7 :].find("(")
                        num = int(text[pos + 7 : pos + 7 + pos2])
                        if num != n:
                            print(
                                f"Access mismatch in {name}, arg {n}({arg_list[n]}) with OPS_ACC{num}"
                            )
                        pos = pos + 7 + pos2
                    else:
                        break


def replace_ACC_kernel_body(kernel_text, arg_list, arg_typ, nargs, opencl=0, dims=[]):
    # replace all data args with macros
    for n in range(0, nargs):
        if arg_typ[n] == "ops_arg_dat":
            pattern = re.compile(r"\b" + arg_list[n] + r"\b")
            match = pattern.search(kernel_text, 0)
            while match:
                closeb = para_parse(kernel_text, match.start(), "(", ")") + 1
                openb = kernel_text.find("(", match.start())
                if opencl == 1:
                    if not dims[n].isdigit() or int(dims[n]) > 1:
                        acc = (
                            f"OPS_ACCM({arg_list[n]}, {kernel_text[openb+1:closeb-1]})"
                        )
                    else:
                        acc = (
                            f"OPS_ACCS({arg_list[n]}, {kernel_text[openb+1:closeb-1]})"
                        )
                else:
                    acc = f"OPS_ACC({arg_list[n]}, {kernel_text[openb+1:closeb-1]})"
                kernel_text = (
                    kernel_text[0 : match.start()] + acc + kernel_text[closeb:]
                )
                match = pattern.search(kernel_text, match.start() + 10)
    return kernel_text


def write_text_to_file(file_name):
    header_text = f"// Auto-generated at {datetime.now()} by ops-translator legacy\n\n"
    with open(file_name, "w") as fid:
        fid.write(header_text)
        fid.write(config.file_text)
    # side effects:
    config.depth = 0
    config.file_text = ""


def get_file_text_for_kernel(kernel_name, src_dir):
    for file in glob.glob(os.path.join(src_dir, "*.h")):
        with open(file, "r") as fid:
            f_content = remove_trailing_w_space(comment_remover(fid.read()))
            p = re.compile("void\\s+\\b" + kernel_name + "\\b")
            if p.search(f_content):
                return f_content

    print("COULDN'T NOT FIND KERNEL", kernel_name)
    return None


def get_kernel_func_text(name, src_dir, arg_typ):
    text = get_file_text_for_kernel(name, src_dir)
    if text is None:
        print("\n********")
        print(
            f"Error: cannot locate user kernel function: {name} - Aborting code generation"
        )
        exit(2)

    p = re.compile(f"void\\s+\\b{name}\\b")
    match = p.search(text)
    assert match  # match shouldn't be None unless text was None
    text = text[max(0, text[: match.start()].rfind("\n")) :]
    kernel_text = convert_ACC(
        text[: para_parse(text, text.find("{"), "{", "}") + 2], arg_typ
    )
    return kernel_text


def get_kernel_body_and_arg_list(name, src_dir, arg_typ):
    kernel_text = get_kernel_func_text(name, src_dir, arg_typ)
    j = kernel_text.find("{")
    kernel_body = kernel_text[j + 1 : kernel_text.rfind("}")]
    arg_list = parse_signature(kernel_text[kernel_text.find(name) + len(name) : j])
    return kernel_body, arg_list


def generate_extern_global_consts_declarations(consts, for_cuda=False, for_hip=False):
    comm(" global constants")
    prefix = "__constant__" if for_cuda or for_hip else "extern"
    for const in consts:
        name = const["name"].replace('"', "").strip()
        if for_hip:
            code(f"#define {name} {name}_OPSCONSTANT")
        if const["dim"].isdigit() and int(const["dim"]) == 1:
            code(f'{prefix} {const["type"]} {name};')
        else:
            if const["dim"].isdigit() and int(const["dim"]) > 1:
                code(f'{prefix} {const["type"]} {name}[{const["dim"]}];')
            else:
                code(f'{prefix} {const["type"]} *{name};')


def group_n_per_line(vals, n_per_line=4, sep=",", group_sep="\n", spec_group_sep=None):
    if spec_group_sep:
        group_sep = spec_group_sep
    else:
        group_sep = sep + group_sep
    return (group_sep).join(
        [
            sep.join([vals[i] for i in range(s, min(len(vals), s + n_per_line))])
            for s in range(0, len(vals), n_per_line)
        ]
    )


def create_kernel_info(kernel):
    arg_typ = kernel["arg_type"]
    name = kernel["name"]
    nargs = kernel["nargs"]
    dim = kernel["dim"]
    dims = kernel["dims"]
    stens = kernel["stens"]
    accs = kernel["accs"]
    typs = kernel["typs"]
    NDIM = int(dim)
    # parse stencil to locate strided access
    stride = [[1] * NDIM for _ in range(nargs)]

    if NDIM == 2:
        for n, sten in enumerate(stens):
            if sten.find("STRID2D_X") > 0:
                stride[n][1] = 0
            elif sten.find("STRID2D_Y") > 0:
                stride[n][0] = 0

    if NDIM == 3:
        for n, sten in enumerate(stens):
            if sten.find("STRID3D_XY") > 0:
                stride[n][2] = 0
            elif sten.find("STRID3D_YZ") > 0:
                stride[n][0] = 0
            elif sten.find("STRID3D_XZ") > 0:
                stride[n][1] = 0
            elif sten.find("STRID3D_X") > 0:
                stride[n][1] = 0
                stride[n][2] = 0
            elif sten.find("STRID3D_Y") > 0:
                stride[n][0] = 0
                stride[n][2] = 0
            elif sten.find("STRID3D_Z") > 0:
                stride[n][0] = 0
                stride[n][1] = 0

    ### Determine if this is a MULTI_GRID LOOP with
    ### either restrict or prolong
    restrict = [str(sten).find("RESTRICT") > 0 for sten in stens]
    prolong = [str(sten).find("PROLONG") > 0 for sten in stens]
    MULTI_GRID = any(prolong) or any(restrict)

    has_reduction = any(
        map(lambda x: x[0] == "ops_arg_gbl" and x[1] != OPS_READ, zip(arg_typ, accs))
    )
    GBL_READ = any(
        map(lambda x: x[0] == "ops_arg_gbl" and x[1] == OPS_READ, zip(arg_typ, accs))
    )
    GBL_READ_MDIM = any(
        map(
            lambda x: x[0] == "ops_arg_gbl"
            and x[1] == OPS_READ
            and (not x[2].isdigit() or int(x[2]) > 1),
            zip(arg_typ, accs, dims),
        )
    )

    arg_idx = -1
    for n, typ in enumerate(arg_typ):
        if typ == "ops_arg_idx":
            arg_idx = n

    needDimList = []
    for n, (typ, acc, dim) in enumerate(zip(arg_typ, accs, dims)):
        if typ == "ops_arg_dat" or (typ == "ops_arg_gbl" and acc != OPS_READ):
            if not dim.isdigit():
                needDimList.append(n)

    return (
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
        GBL_READ,
        GBL_READ_MDIM,
        has_reduction,
        arg_idx,
        needDimList,
    )
