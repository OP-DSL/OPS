import os
import re
import copy
from pathlib import Path
from typing import Dict, Any

import fortran.translator.kernels as ftk
from fortran.parser import getChild, parseIdentifier

import ops as OPS
from language import Lang
from scheme import Scheme
from store import Application, ParseError, Program
from target import Target
from util import find

import fparser.two.Fortran2003 as f2003
from fparser.common.readfortran import FortranStringReader
from fparser.two.parser import ParserFactory

def retrieve_subroutine_by_name(file_path, subroutine_name):
    if not os.path.exists(file_path):
        raise ParseError(f"Unable to find file {file_path} for subroutine: {subroutine_name}")

    path = Path(file_path)
    source = path.read_text()
    reader = FortranStringReader(source, ignore_comments=False)
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
#    with open(file_path, 'r') as f:
#        fortran_code = f.read()

#    beg = re.search(r'\s*\bsubroutine\s*'+subroutine_name+r'\b\s*\(', fortran_code, re.IGNORECASE)
#    if beg == None:
#        raise ParseError(f"Unable to find subroutine: {subroutine_name}")
#        exit(1)
#    beg_pos = beg.start()
#    end = re.search(r'\s*end\s*subroutine\b', fortran_code[beg_pos:], re.IGNORECASE)
#    if end == None:
#        raise ParseError(f"'Could not find matching end subroutine for {subroutine_name}")
#        exit(1)
#
#    req_kernel = fortran_code[beg_pos:beg_pos+end.end()]
#    return req_kernel+'\n'


class FortranMPIOpenMP(Scheme):
    lang = Lang.find("F90")
    target = Target.find("mpi_openmp")

    fallback = None

    consts_template = None
    loop_host_template = Path("fortran/mpi_openmp/loop_host.F90.j2")
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

        kernel_entities = retrieve_subroutine_by_name(filename, loop.kernel)

        if kernel_entities is None or (kernel_entities is not None and len(kernel_entities) == 0):
            raise ParseError(f"unable to find kernel function: {loop.kernel}")

        return kernel_entities.strip()

Scheme.register(FortranMPIOpenMP)

class FortranCuda(Scheme):
    lang = Lang.find("F90")
    target = Target.find("cuda")

    fallback = None

    consts_template = None
    loop_host_template = Path("fortran/cuda/loop_host.F90.j2")
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

        kernel_entities = retrieve_subroutine_by_name(filename, loop.kernel)

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
