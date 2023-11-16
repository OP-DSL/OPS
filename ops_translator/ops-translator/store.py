from __future__ import annotations

import os
import copy
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Set, Tuple, Union
from util import flatten, uniqueBy

import ops
from ops import OpsError

if TYPE_CHECKING:
    from language import Lang


@dataclass(frozen=True)
class Location:
    file: str
    line: int
    column: int

    def __str__(self) -> str:
        return f"{os.path.abspath(self.file)}/{self.line}:{self.column}"


@dataclass
class ParseError(Exception):
    message: str
    loc: Optional[Location] = None

    def __str__(self) -> str:
        if self.loc:
            return f"Parse error at {self.loc}: {self.message}"
        else:
            return f"Parse error: {self.message}"


@dataclass
class Entity:
    name: str
    ast: Any

    program: Program
    scope: List[str] = field(default_factory=list)
    depends: Set[str] = field(default_factory=set)

    #Deep-copy everything but the program reference
    def __deepcopy__(self, memo) -> Optional["Entity"]:
        cls = self.__class__
        result = cls.__new__(cls)

        setattr(result, "program", self.program)

        memo[id(self)] = result

        for k, v in self.__dict__.items():
            if hasattr(result, k):
                continue

            setattr(result, k, copy.deepcopy(v, memo))

        return result

@dataclass
class Type(Entity):
    def __str__(self) -> str:
        return f"Type(name='{self.name}', scope={self.scope}, depends={self.depends})"

@dataclass
class Function(Entity):
    parameters: List(str) = field(default_factory=list)
    returns: Optional[ops.Type] = None
    loc: Location = None

    def __str__(self) -> str:
        return f"Function(name='{self.name}', loc={self.loc}, scope={self.scope}, depends={self.depends}, parameters={self.parameters}, ast={self.ast})"

@dataclass 
class KernelDef(Function):
    def __str__(self) -> str:
        return f"Function(name='{self.name}', scope={self.scope}, depends={self.depends}, parameters={self.parameters}, ast={self.ast})"
        
@dataclass
class Program:
    path: Path

    ast: Any
    source: str

    consts: List[ops.Const] = field(default_factory=list)
    stencils: List[ops.Stencil] = field(default_factory=list)
    loops: List[ops.Loop] = field(default_factory=list)

    entities: List[Entity] = field(default_factory=list)
    
    ndim: Optional[int] = None
    soa_val: Optional[bool] = False

    def findEntities(self, name: str, scope: List[str] = []) -> List[Entity]:
        def in_scope(entity: Entity):
            return len(entity.scope) <= len(scope) and all(map(lambda s1, s2: s1 == s2, zip(entity.scope, scope)))

        candidates = list(filter(lambda e: e.name == name and in_scope(e), self.entities))

        if len(candidates) == 0:
            return []

        candidates.sort(key=lambda e: len(e.scope), reverse=True)
        min_scope = len(candidates[0].scope)

        #returning canditages with min scope    
        return list(filter(lambda e: len(e.scope) == min_scope, candidates))
    
    def findStencil(self, name_ptr: str)->Union[ops.Stencil, None]:
        for stencil in self.stencils:
            if stencil.stencil_ptr == name_ptr:
                return stencil
        return None
    
    def __str__(self) -> str:
        outString = "\nprogram path=" + str(self.path)  + ",\n"
        outString += "ast=" + str(self.ast) + ",\n"
        outString += "ndim=" + str(self.ndim) + ",\n"
        outString += "\n---------------------\n"
        outString += "       consts        \n"
        outString += "---------------------\n"
        for const in self.consts:
            outString += str(const) + "\n"

        outString += "\n---------------------\n"    
        outString += "        loops        \n"
        outString += "---------------------\n"
        for loop in self.loops:
            outString += str(loop) + "\n"

        outString += "\n---------------------\n"    
        outString += "       Entities      \n"
        outString += "---------------------\n"
        for entity in self.entities:
            outString += str(entity) + "\n"  
            
        outString += "\n---------------------\n"    
        outString += "       Stencils      \n"
        outString += "---------------------\n"
        for stencil in self.stencils:
            outString += str(stencil) + "\n"   
        
        return outString

@dataclass
class Application:
    programs: List[Program] = field(default_factory=list)
    global_dim: Optional[int] = None

    def __str__(self) -> str:
        if len(self.programs) > 0:
            programs_str = "\n".join([str(p) for p in self.programs])
        else:
            programs_str = "No programs"

        return programs_str

    def findEntities(self, name: str, program: Program = None, scope: List[str] = []) -> List[Entity]:
        candidates = []

        if program is not None:
            candidates = program.findEntities(name, scope)

        if len(candidates) > 0:
             return candidates

        for program2 in self.programs:
            if program2 == program:
                continue

            candidates = program2.findEntities(name)
            if len(candidates) > 0:
                break

        return candidates

    def consts(self) -> List[ops.Const]:
        consts = flatten(program.consts for program in self.programs)
        return uniqueBy(consts, lambda c: c.ptr)

    def loops(self) -> List[Tuple[ops.Loop, Program]]:
        return flatten(map(lambda l: (l, p), p.loops) for p in self.programs)

    def uniqueLoops(self) -> List[ops.Loop]:
        return uniqueBy(self.loops(), lambda m: m[0].kernel)
        for p in self.programs:
            id = findId

    def validate(self, lang: Lang) -> None:
        self.validateConst(lang)
        self.validateLoops(lang)

        for program in self.programs:
            if self.global_dim == None:
                self.global_dim = program.ndim
            elif program.ndim == None:
                program.ndim = self.global_dim
            elif self.global_dim != program.ndim:
                raise OpsError(f"ndim mismatch with global dim={self.global_dim} and program dim={program.ndim} of program={program.path}")


    def validateConst(self, lang: Lang) -> None:
        seen_const_ptrs: Set[str] = set()

        for program in self.programs:
            for const in program.consts:
                if const.ptr in seen_const_ptrs:
                    raise OpsError(f"Duplicate const declaration: {const.ptr}", const.loc)

                seen_const_ptrs.add(const.ptr)

                if const.dim.isdigit() and int(const.dim) < 1:
                    raise OpsError(f"Invalid const dimension: {const.dim} of const: {const.ptr}", const.loc)


    def validateLoops(self, lang: Lang) -> None:
        for loop, Program in self.loops():
            num_opts = len([arg for arg in loop.args if getattr(arg, "opt", False)])
            if num_opts > 32:
                raise OpsError(f"number of optional arguments exceeds 32: {num_opts}", loop.loc)
            for arg in loop.args:
                if isinstance(arg, ops.ArgDat):
                    self.validateArgDat(arg, loop, lang)

                if isinstance(arg, ops.ArgGbl):
                    self.validateArgGbl(arg, loop, lang)

            #self.validateKernel(loop, program, lang) TODO


    def validateArgDat(self, arg: ops.ArgDat, loop: ops.Loop, lang: Lang) -> None:
        valid_access_types = [
            ops.AccessType.OPS_READ, 
            ops.AccessType.OPS_WRITE, 
            ops.AccessType.OPS_RW, 
            ops.AccessType.OPS_INC
            ]

        if arg.access_type not in valid_access_types:
            raise OpsError(f"Invalid access type for dat argument: {arg.access_type}, arg: {arg}", arg.loc)


    def validateArgGbl(self, arg: ops.ArgGbl, loop: ops.Loop, lang: Lang) -> None:
        valid_access_types = [
            ops.AccessType.OPS_READ, 
            ops.AccessType.OPS_WRITE, 
            ops.AccessType.OPS_RW, 
            ops.AccessType.OPS_INC,
            ops.AccessType.OPS_MIN,
            ops.AccessType.OPS_MAX
            ]

        if arg.access_type not in valid_access_types:
            raise OpsError(f"Invalid access type for gbl argument: {arg.access_type}", arg.loc)

        if arg.access_type in [ops.AccessType.OPS_INC, ops.AccessType.OPS_MIN, ops.AccessType.OPS_MAX] and arg.typ not in \
            [ops.Float(64), ops.Float(32), ops.Int(True, 32), ops.Int(False, 32), ops.Bool]:
            raise OpsError(f"Invalid access type for reduced gbl argument: {arg.access_type}", arg.loc)

        if str(arg.dim).isdigit() and int(str(arg.dim)) < 1:
            raise OpsError(f"Invalid gbl argument dimension: {arg.dim}", arg.loc)

    # TODO: Implement Kernel Validation




