import copy
import io
import os
import re
import subprocess
from pathlib import Path
from typing import FrozenSet, List, Optional, Set, Tuple

import fparser.two.Fortran2003 as f2003
import fparser.two.utils
import pcpp
from fparser.common.readfortran import FortranStringReader
from fparser.two.parser import ParserFactory
from fparser.two.utils import Base, _set_parent

import fortran.parser
import fortran.translator.program
import fortran.validator
import ops as OPS
from language import Lang
from store import Application, Location, ParseError, Program

def base_deepcopy(self, memo):
    cls = self.__class__
    result = object.__new__(cls)

    memo[id(self)] = result

    for k, v in self.__dict__.items():
        if k == "parent":
            continue

        setattr(result, k, copy.deepcopy(v, memo))

    if hasattr(result, "items"):
        _set_parent(result, result.items)

    return result

def string_reader_deepcopy(self, memo):
    cls = self.__class__
    result = cls.__new__(cls)

    memo[id(self)] = result

    setattr(result, "source", None)
    setattr(result, "file", None)

    for k, v in self.__dict__.items():
        if hasattr(result, k):
            continue

        setattr(result, k, copy.deepcopy(v, memo))

    return result

# Patch the fparser2 Base class to allow deepcopies
Base.__deepcopy__ = base_deepcopy
FortranStringReader.__deepcopy__ = string_reader_deepcopy

kind_selector_aliases = {"*PS": "*8"}


def kind_selector_match(string):
    if string in kind_selector_aliases:
        string = kind_selector_aliases[string]

    return f2003.Kind_Selector.match_(string)


f2003.Kind_Selector.match_ = f2003.Kind_Selector.match
f2003.Kind_Selector.match = staticmethod(kind_selector_match)


# Patch the updated fparser2 walk function that visits tuples
# TODO: remove this when it has been included in an fparser release
def walk(node_list, types=None, indent=0, debug=False):
    local_list = []

    if not isinstance(node_list, (list, tuple)):
        node_list = [node_list]

    for child in node_list:
        if debug:
            if isinstance(child, str):
                print(indent * "  " + "child type = ", type(child), repr(child))
            else:
                print(indent * "  " + "child type = ", type(child))
        if types is None or isinstance(child, types):
            local_list.append(child)
        # Recurse down
        if isinstance(child, Base):
            local_list += walk(child.children, types, indent + 1, debug)
        elif isinstance(child, tuple):
            for component in child:
                local_list += walk(component, types, indent + 1, debug)

    return local_list


fparser.two.utils.walk = walk


class Preprocessor(pcpp.Preprocessor):
    def __init__(self, lexer=None):
        super(Preprocessor, self).__init__(lexer)

        self.line_directive = None

    def on_comment(self, tok: str) -> bool:
        return tok.type == self.t_COMMENT2

    def on_error(self, file: str, line: int, msg: str) -> None:
        loc = Location(file, line, 0)
        raise ParseError(msg, loc)

    def on_include_not_found(self, is_malformed, is_system_include, curdir, includepath) -> None:
        if is_system_include:
            raise pcpp.OutputDirective(pcpp.Action.IgnoreAndPassThrough)

        super(Preprocessor, self).on_include_not_found(is_malformed, is_system_include, curdir, includepath)


class Fortran(Lang):
    name = "Fortran"

    source_exts = ["F90", "F95"]
    include_ext = "inc"
    kernel_dir = True

    com_delim = "!"

#   fallback_wrapper_template = Path("fortran/fallback_wrapper.F90.jinja")

    extra_consts_list = None
    user_consts_module = None
    use_regex_translator = False

    def __init__(self):
        with open("constants_list.txt", 'w') as file:
            pass

    def validate(self, app: Application) -> None:
        # TODO: see fortran.parser
        for program in app.programs:
            fortran.parser.parseFunctionDependencies(program, app)

        for loop, program in app.loops():
            fortran.validator.validateLoop(loop, program, app)

    def preprocess(self, path: Path, include_dirs: FrozenSet[Path], defines: FrozenSet[str]) -> str:
        fpp = os.getenv("OPS_FPP")
        if fpp is not None:
            args = [fpp, "-P", "-free", "-f90"]

            for dir in include_dirs:
                args.append(f"-I{dir}")

            for define in defines:
                args.append(f"-D{define}")

            args.append(str(path))

            print(" ".join(args))
            res = subprocess.run(args, capture_output=True, check=True)
            print(res.stderr.decode("utf-8"))

            return res.stdout.decode("utf-8")

        preprocessor = Preprocessor()

        for dir in include_dirs:
            preprocessor.add_path(str(dir.resolve()))

        for define in defines:
            if "=" not in define:
                define = f"{define}=1"

            preprocessor.define(define.replace("=", " ", 1))

        preprocessor.parse(path.read_text(), str(path))

        source = io.StringIO()
        source.name = str(path)

        preprocessor.write(source)

        source.seek(0)

        source = source.read()

        source = re.sub(r"__FILE__", f'"{path}"', source)
        source = re.sub(r"__LINE__", "0", source)

        return source

    def parseFile(
        self, path: Path, include_dirs: FrozenSet[Path], defines: FrozenSet[str]
    ) -> Tuple[f2003.Program, str]:
        #source = self.preprocess(path, include_dirs, defines)
        #with open("_preprocessed.F90", "w") as f:
        #           f.write(source)
        source = path.read_text()
        reader = FortranStringReader(source, include_dirs=list(include_dirs), ignore_comments=False)
        parser = ParserFactory().create(std="f2003")
#        try:
#            print(parser(reader))
#        except Exception as e:
#            print(f"An error occurred: {e}")
        return parser(reader), source

    def parseProgram(self, path: Path, include_dirs: Set[Path], defines: List[str]) -> Program:
        ast, source = self.parseFile(path, frozenset(include_dirs), frozenset(defines))
        return fortran.parser.parseProgram(ast, source, path)

    def translateProgram(self, program: Program, include_dirs: Set[Path], defines: List[str], app_consts: List[OPS.Const], force_soa: bool) -> str:
        #if self.use_regex_translator:
        #    return fortran.translator.program.translateProgram2(program, force_soa)

        return fortran.translator.program.translateProgram(program, force_soa)

    def formatType(self, typ: OPS.Type) -> str:
        if isinstance(typ, OPS.Int):
            if not typ.signed:
                raise NotImplementedError("Fortran does not support unsigned integers")

            return f"integer(kind={int(typ.size / 8)})"
        elif isinstance(typ, OPS.Float):
            return f"real(kind={int(typ.size / 8)})"
        elif isinstance(typ, OPS.Bool):
            return "logical"
        elif isinstance(typ, OPS.Custom):
            return typ.name
        else:
            assert False


Lang.register(Fortran)

import fortran.schemes

