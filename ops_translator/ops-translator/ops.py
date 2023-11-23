from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Callable, List, Optional, Union, Tuple

from util import ABDC, findIdx

if TYPE_CHECKING:
    from store import Location

class AccessType(Enum):
    OPS_READ = 0
    OPS_WRITE = 1 
    OPS_RW = 2

    OPS_INC = 3
    OPS_MIN = 4
    OPS_MAX = 5

    @staticmethod
    def values() -> List[str]:
        return [x.value for x in list(AccessType)]

# class ArgType(Enum):
#     ARGDAT = 0
#     ARGGBL = 1

#     ARGIDX = 2

#     @staticmethod
#     def values() -> List[str]:
#         return [x.value for x in list(AccessType)]


class OpsError(Exception):
    message: str
    loc: Location

    def __init__(self, message: str, loc: Location = None) -> None:
        self.message = message
        self.loc = loc

    def __str__(self) -> str:
        if self.loc:
            return f"{self.loc}: OPS Error: {self.message}"
        else:
            return f"OPS error: {self.message}"


class Type:
    formatter: Callable[["Type"], str]

    @classmethod
    def set_formatter(cls, formatter: Callable[["Type"], str]) -> None:
        cls.formatter = formatter

    def __str__(self) -> str:
        return self.__class__.formatter(self)


@dataclass(frozen=True)
class Int(Type):
    signed: bool
    size: int

    def __repr__(self) -> str:
        if self.signed and self.size == 16:
            return "short"
        elif not self.signed and self.size == 16:
            return "unsigned short"
        elif self.signed and self.size == 32:
            return "int"
        elif not self.signed and self.size == 32:
            return "unsigned int"
        elif self.signed and self.size == 64:
            return "long long"
        elif not self.signed and self.size == 64:
            return "unsigned long long"
        else:
            return f"{'i' if self.signed else 'u'}{self.size}"


@dataclass(frozen=True)
class Float(Type):
    size: int

    def __repr__(self) -> str:
        if self.size == 32:
            return "float"
        elif self.size == 64:
            return "double"
        else:
            return f"f{self.size}"


@dataclass(frozen=bool)
class ComplexD(Type):
    pass

    def __repr__(self) -> str:
        return "complexd"


@dataclass(frozen=bool)
class ComplexF(Type):
    pass

    def __repr__(self) -> str:
        return "complexf"


@dataclass(frozen=bool)
class Bool(Type):
    pass

    def __repr__(self) -> str:
        return "bool"

@dataclass(frozen=bool)
class Char(Type):
    pass

    def __repr__(self) -> str:
        return "char"


@dataclass(frozen=True)
class Custom(Type):
    name: str

    def __repr__(self) -> str:
        return self.name


@dataclass(frozen=True)
class Const:
    loc: Location
    ptr: str

    dim: str
    typ: Type
    name: str

    def __str__(self) -> str:
        return f"Const(name='{self.name}', loc={self.loc}, ptr='{self.ptr}', dim={self.dim}, type={self.typ})"


@dataclass(frozen=True)
class Range:
    loc: Location
    ptr: str

    dim: int

    def __str__(self) -> str:
        return f"Range(loc={self.loc}, ptr='{self.ptr}', dim={self.dim})"


@dataclass(frozen=True)
class Dat:
    id: int

    ptr: str
    dim: int
    # size: List[int]
    # base: List[int]
    # d_m: List[int]
    # d_p: List[int]

    typ: Type
    soa: bool

    block_id: Optional(int) = field(default_factory=int)
    name: Optional(str) = field(default_factory=str)

    # def __post_init__(self) -> None:
    #     if len(self.size) != self.dim:
    #         OpsError(f"dim of size={self.size} is not same as dat dim={self.dim} of dat='{self.name}'")
    #     elif len(self.base) != self.dim:
    #         OpsError(f"dim of base={self.base} is not same as dat dim={self.dim} of dat='{self.name}'")
    #     elif len(self.d_m) != self.dim:
    #         OpsError(f"dim of d_m={self.d_m} is not same as dat dim={self.dim} of dat='{self.name}'")
    #     elif len(self.d_p) != self.dim:
    #         OpsError(f"dim of d_p={self.d_p} is not same as dat dim={self.dim} of dat='{self.name}'")

    def __str__(self) -> str:
        return f"Dat(block_id={self.block_id}, id={self.id}, ptr='{self.ptr}', dim={self.dim}, type={self.typ}, soa={self.soa})"


@dataclass(frozen=False)
class Point:
    x: int
    y: int = 0
    z: int = 0
    
    def __str__(self) -> str:
        return f"({self.x},{self.y},{self.z})"
    
    def __init__(self, input: List[int]):
        if (len(input) > 3):
            raise ValueError("ops.Point cannot be initialized more than 3 dim")
        
        for i in range(len(input)):
            self[i] = input[i]
        
    def __getitem__(self, i:int) -> Union[int, None]:
        if i < 0 or i > 2:
            return None
        else:
            if i == 0:
                return self.x
            elif i == 1:
                return self.y
            else:
                return self.z  
    
    def __setitem__(self, i:int, newval: int):
        if i == 0:
            self.x = newval
        elif i == 1:
            self.y = newval
        else:
            self.z  = newval
    
@dataclass(frozen=True)
class Stencil:
    id: int
    dim: int

    stencil_ptr: str

    num_points: int
    points: List[Point]
    stencil_size: int
    window_buffers : List[str]
    chains: List[Tuple[str, str]]
    stride: Optional[list] = field(default_factory=list)
    
    def __str__(self) -> str:
        return f"Stencil(id={self.id}, dim={self.dim}, stencil_ptr='{self.stencil_ptr}', \
number of points={self.num_points}, points={self.points}, stride_ptr='{self.stride}')"

@dataclass(frozen=True)
class Arg(ABDC):
    id: int
    loc: Location

# TODO: Remove Dat and incorpareate into ArgDat
@dataclass(frozen=True)
class ArgDat(Arg):
    access_type: AccessType
    opt: bool

    dat_id: int
    stencil_ptr: str

    dim: int
    restrict: Optional[bool] = False
    prolong: Optional[bool] = False

#    stride: Optional[List] = None

#    def __post_init__(self):
#        object.__setattr__(self, 'stride', [1]*3)

    def __str__(self) -> str:
        return (
            f"ArgDat(id={self.id}, loc={self.loc}, access_type={str(self.access_type) + ',':17} opt={self.opt}, dat_id={self.dat_id}, stencil_id={self.stencil_ptr})"
            )

@dataclass(frozen=True)
class ArgGbl(Arg):
    access_type: AccessType

    ptr: str

    dim: str
    typ: Type

    #opt : bool

    def __str__(self) -> str:
        return (
            f"ArgGbl(id={self.id}, loc={self.loc}, access_type={str(self.access_type) + ',':17}" 
            f"ptr={self.ptr}, dim={self.dim}, type={self.typ})"
        )

@dataclass(frozen=True)
class ArgReduce(Arg):
    access_type: AccessType

    ptr: str

    dim: int
    typ: Type

    def __str__(self) -> str:
        return (
            f"ArgReduce(id={self.id}, loc={self.loc}, access_type={str(self.access_type) + ',':17}), " ##opt={self.opt}, "
            f"ptr={self.ptr}, dim={self.dim}, type={self.typ})"
        )

@dataclass(frozen=True)
class ArgIdx(Arg):
    pass

    def __str__(self) -> str:
        return f"ArgIdx(id={self.id}, loc={self.loc})"    

class Block:
    loc: Location
    ptr: str
    id: int

    dim: int
    dats: List[Dat]

    def __init__(self, loc: Location, ptr: str, dim: int) -> None:
        self.loc = loc
        self.ptr = ptr
        self.dim = dim
        self.dats = []

    def __str__(self) -> str:
        dat_str = "\n   ".join([str(dat) for dat in self.dats])

        if len(self.dats) > 0:
            dat_str = f"\n    {dat_str}\n"

        return f"Block(id={self.id}, loc={self.loc}, ptr='{self.ptr}', dim={self.dim}, dats={dat_str})"

    def addDat(self, dat: Dat):
        dat_id = findIdx(self.dats, lambda d: d.ptr == dat.ptr)

        if dat_id is None:
            dat_id = len(self.dats)
            self.dats.append(dat)


class Loop:
    loc: Location
    kernel: str

    block: str
    range: Range
    ndim: int

    args: List[Arg]
    dats: List[Dat]
    stencils: List[str]

    arg_idx: Optional[int] = -1
    multiGrid: Optional[bool] = False
    isGblRead: Optional[bool] = False
    isGblReadMDIM: Optional[bool] = False
    has_reduction: Optional[bool] = False

    def __init__(self, loc: Location, kernel: str, block: str, range: Range, ndim: int) -> None:
        self.loc = loc
        self.kernel = kernel
        self.block = block
        self.range = range
        self.ndim = ndim

        self.dats = []
        self.args = []
        self.stencils = []

    def addArgDat(
        self,
        loc: Location,
        dat_ptr: str,
        dat_dim: int,
        dat_typ: Type,
        dat_soa: bool,
        stencil_ptr: str,
        access_type: AccessType,
         opt: bool
    ) -> None: 

        arg_id = len(self.args)
        dat_id = findIdx(self.dats, lambda d: d.ptr == dat_ptr)

        if dat_id is None:
            dat_id = len(self.dats)
            # if findIdx(self.block.dats, lambda d: d.ptr == dat_ptr) is not None:
            self.dats.append(Dat(dat_id, dat_ptr, dat_dim, dat_typ, dat_soa))
            # else:
            #     OpsError(f"Parsing Dat='{dat_ptr}' as argument of loop in {self.loc} which is not belong to block='{self.block.ptr}'", loc)

        # stencil_id = findIdx(self.stencils, lambda s: s.stencil_ptr == stencil_ptr)

        if stencil_ptr not in self.stencils:
            self.stencils.append(stencil_ptr)

        restrict = stencil_ptr.find("RESTRICT") > 0
        prolong = stencil_ptr.find("PROLONG") > 0

        if not self.multiGrid and (restrict or prolong):
            self.multiGrid = True

        arg = ArgDat(arg_id, loc, access_type, opt, dat_id, stencil_ptr, dat_dim, restrict, prolong)
        self.args.append(arg)

    def addArgReduce(
        self,
        loc: Location,
        reduct_handle: str,
        dim: int,
        typ: Type,
        access_type: AccessType
    ) -> None: 

        arg_id = len(self.args)

        if not self.has_reduction:
            self.has_reduction = True

        arg = ArgReduce(arg_id, loc, access_type, reduct_handle, dim, typ)
        self.args.append(arg)

    def addArgGbl(
        self,
        loc: Location,
        ptr: str,
        dim: str,
        typ: Type,
        access_type: AccessType
    ) -> None:
        arg_id = len(self.args)
        arg = ArgGbl(arg_id, loc, access_type, ptr, dim, typ)

        if not self.isGblRead:
            if access_type == AccessType.OPS_READ:
                self.isGblRead = True

        if not self.isGblReadMDIM:
            if access_type == AccessType.OPS_READ:
                if not dim.isdigit() or (dim.isdigit() and int(dim) > 1):
                    self.isGblReadMDIM = True

        self.args.append(arg)

    def addArgIdx(
        self,
        loc: Location
    ) -> None:
        arg_id = len(self.args)
        arg = ArgIdx(arg_id, loc)
        self.arg_idx = arg_id
        self.args.append(arg)

    def get_dat(self, x: Union[ArgDat, int]) -> Optional[Dat]:
        if isinstance(x, ArgDat) and x.dat_id < len(self.dats):
            return self.dats[x.dat_id]

        if isinstance(x, int) and x < len(self.dats):
            return self.dats[x]

        return None

    def __str__(self) -> str:
        kernel_detail_str = f"Loop at {self.loc}:\n Kernel function: {self.kernel}\n \
            range dim: {self.ndim}, block: {self.block}, range: {self.range}, arg_idx: {self.arg_idx}"
        args_str = "\n    ".join([str(a) for a in self.args])
        dat_str = "\n    ".join([str(d) for d in self.dats])

        if len(self.dats) > 0:
            dat_str = f"\n    {dat_str}\n"

        return f"{kernel_detail_str}\n\n    {args_str}\n {dat_str}\n"

