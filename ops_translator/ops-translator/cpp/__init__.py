import os
from functools import lru_cache
from io import StringIO
from pathlib import Path
from typing import FrozenSet, List, Set, Tuple, Any

import clang.cindex

import cpp.optimizer
import cpp.parser
import cpp.translator.program
import cpp.preprocessor
import ops
from language import Lang
from store import Program

libclang_path = os.getenv("LIBCLANG_PATH")
if libclang_path is not None:
    clang.cindex.Config.set_library_file(libclang_path)

class Cpp(Lang):
    name = "C++"

    source_exts = ["cpp"]
    include_ext = "h"
    kernel_dir = True

    com_delim = "//"
    zero_idx = True

    @lru_cache(maxsize=None)
    def parseFile(
        self, path: Path, include_dirs: FrozenSet[Path], defines: FrozenSet[str], preprocess: bool = False
        ) -> Tuple[clang.cindex.TranslationUnit, str, Any]:
        args = [f"-I{dir}" for dir in include_dirs]
        args = args + [f"-D{define}" for define in defines]
        args = args +['-std=c++11']
        source = path.read_text()
        isl_directives = []
        
        if preprocess:
            preprocessor = cpp.preprocessor.Preprocessor() 

            for dir in include_dirs:
                preprocessor.add_path(str(dir.resolve()))

            for define in defines:
                if "=" not in define:
                    define = f"{define}=1"

                preprocessor.define(define.replace("=", " ", 1))

            preprocessor.parse(source, str(path.resolve()))
                
            source_io = StringIO()
            preprocessor.write(source_io)

            source_io.seek(0)
            source = source_io.read()
            
            isl_directives = preprocessor.get_isl_directives()
        else:
            isl_directives = None

        translation_unit = clang.cindex.Index.create().parse(
            path,
            unsaved_files=[(path, source)],
            args=args,
            options=clang.cindex.TranslationUnit.PARSE_DETAILED_PROCESSING_RECORD
        )

        for diagnostic in iter(translation_unit.diagnostics):
            print(
                f"Clang parse diagnotic message: severity {diagnostic.severity} at: "
                f"{cpp.parser.parseLocation(diagnostic)}: {diagnostic.spelling}"
            )

        if isl_directives is not None:
            return translation_unit, source, isl_directives
        else:
            return translation_unit, source,  

    def parseProgram(self, path: Path, include_dirs: Set[Path], defines: List[str]) -> Program:
        ast, source = self.parseFile(path, frozenset(include_dirs), frozenset(defines))
        ast_pp, source_pp, isl_directives =  self.parseFile(path, frozenset(include_dirs), frozenset(defines), preprocess = True)

        with open("./source_pp.txt", "w") as f:        
            f.write("=================================================================================")
            f.write("================================== source PP ====================================")
            f.write("=================================================================================")
            f.write(source_pp)
            f.write("=================================================================================")
            f.write("=================================================================================")
            
        if isl_directives:
            for directive in isl_directives:
                print(directive)
                
        # TODO: Find the global ndim programatically
        program = Program(path, ast, ast_pp, source_pp, isl_directives)

        cpp.parser.parseLoops(ast, program)
        cpp.parser.parseMeta(ast_pp.cursor, program)

        return program
    
    def translateProgram(self, program: Program, include_dirs: Set[Path], defines: List[str], app_consts: List[ops.Const], force_soa: bool = False, hls: bool = False) -> str:
        if hls:
            return cpp.translator.program.translateProgramHLS(program.path.read_text(), program, app_consts, force_soa)
        else:
            return cpp.translator.program.translateProgram(program.path.read_text(), program, app_consts, force_soa)

    def formatType(self, typ: ops.Type) -> str:
        int_types = {
            (True, 16): "short",
            (True, 32): "int",
            (True, 64): "long long",
            (False, 16): "unsigned short",
            (False, 32): "unsigned int",
            (False, 64): "unsigned long long",
        }

        float_type = {32: "float", 64: "double"}

        if isinstance(typ, ops.Int):
            return int_types[(typ.signed, typ.size)]
        elif isinstance(typ, ops.Float):
            return float_type[typ.size]
        elif isinstance(typ, ops.Bool):
            return "bool"
        elif isinstance(typ, ops.Char):
            return "char"
        elif isinstance(typ, ops.ComplexD):
            return "complexd"
        elif isinstance(typ, ops.ComplexF):
            return "complexf"
        elif isinstance(typ, ops.Custom):
            return typ.name
        else:
            assert False

Lang.register(Cpp)

import cpp.schemes
