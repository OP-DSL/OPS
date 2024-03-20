from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Callable, List, Optional, Union, Tuple, Any

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


class BufferType(Enum):
    LINE_BUFF = 0
    PLANE_BUFF = 1 

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

    block_id: Optional[int] = field(default_factory=int)
    name: Optional[str] = field(default_factory=str)

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
            
    def __add__(self, other: Point)->Point:
        return Point([self.x + other.x, self.y + other.y, self.z + other.z])
    
    def __neg__(self)->Point:
        return Point([-self.x, -self.y, -self.z])

@dataclass(frozen=False)
class StencilRowDiscriptor:
    row_id: Tuple[int, int]
    base_point: Point = field(default=Point([0,0,0]))
    row_points: List[Point] = field(default_factory=list, init=False) 
    
    def _key(self):
        return self.row_id
    def __hash__(self):
        return hash(self._key())
    def __eq__(self, other):
        return self._key() == other._key()


@dataclass(frozen=False)
class WindowBuffer:
    name: str
    buffer_type: BufferType
    read_point: Point
    write_point: Point
    
    def __str__(self) -> str:
        return f"WindowBuffer(name={self.name}, read_point={self.read_point}, write_point={self.write_point})"
    
@dataclass(frozen=True)
class Stencil:
    id: int
    dim: int
    stencil_ptr: str
    num_points: int
    points: List[Point]
    base_point: Point
    stencil_size: int
    window_buffers : List[WindowBuffer]
    chains: List[Tuple[Union[int, WindowBuffer], Union[int, WindowBuffer, str]]]
    d_m: Point
    d_p: Point
    row_discriptors: Optional[List[StencilRowDiscriptor]] = field(default_factory=list, init=False)
    stride: Optional[list] = field(default_factory=list)
    
    def __post_init__(self):
        for point in self.points:
            if StencilRowDiscriptor((point.y, point.z)) in self.row_discriptors:
                self.row_discriptors[self.row_discriptors.index(StencilRowDiscriptor((point.y, point.z)))].row_points.append(point)
            else:
                self.row_discriptors.append(StencilRowDiscriptor((point.y, point.z), self.base_point))
                self.row_discriptors[-1].row_points.append(point)
                    
    def __eq__(self, __value: str) -> bool:
        return self.stencil_ptr == __value 
         
    def __str__(self) -> str:
        return f"Stencil(id={self.id}, dim={self.dim}, stencil_ptr='{self.stencil_ptr}', \
number of points={self.num_points}, points={self.points}, base_point={self.base_point}, stride_ptr='{self.stride}')"

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
    global_dat_id: Optional[int] = -1
    

#    stride: Optional[List] = None

#    def __post_init__(self):
#        object.__setattr__(self, 'stride', [1]*3)

    def __str__(self) -> str:
        return (
            f"ArgDat(id={self.id}, loc={self.loc}, access_type={str(self.access_type) + ',':17} opt={self.opt}, dat_id={self.dat_id}, global_dat_id={self.global_dat_id}, stencil_id={self.stencil_ptr})"
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


@dataclass
class DependancyEdge:
    source_id: int
    source_arg_id: int
    dat_id: int
    sink_id: int
    sink_arg_id: int
    
    def __str__(self) -> str:
        return f"source_id: {self.source_id}, source_arg_id: {self.source_arg_id}, dat_id: {self.dat_id}, sink_id:{self.sink_id}, sink_arg_id: {self.sink_arg_id}"
       
class IterLoop:
    unique_name: str
    id: int
    num_iter: Union[int, str]
    scope: List[Location]
    itr_args: List[Any]
    dats: List[List[Dat, AccessType]] = []
    joint_args: List[Arg] = []
    unique_id: int
    source_dats: List[Union[int, ArgDat]]
    sink_dats: List[Union[int, ArgDat]]
    edges: List[DependancyEdge]
    dat_swap_map: List[int]
    raw_dat_swap_map: List[ParCopy]
    PE_args: List[List[str]]
    interconnector_names: List[str]
    ops_range: str = None
    
    def __init__(self, unique_name: str, id: int, num_iter: Union[int, str], scope: List[Location], args: List[Any] = []) -> None:
        self.unique_name = unique_name
        self.id = id
        self.num_iter = num_iter
        self.scope = scope
        self.itr_args = args
        self.raw_dat_swap_map = filter(lambda x: isinstance(x, ParCopy), args)
        
        key =  ""
        for arg in args:
            if isinstance(arg, Loop):
                key += arg.kernel
                self.addLoop(arg)
                
        self.unique_id = hash(key)
        self.gen_graph()
        self.gen_PE_args()
        
        self.dat_swap_map = [i for i in range(len(self.dats))]
        
        for arg in args:
            if isinstance(arg, ParCopy):
                self.addParCopy(arg)
                

    def gen_graph(self) -> None:
        temp_channels = {}
        edges = []
        source_dats = []
        sink_dats = []
        for i, v in enumerate(self.itr_args):
            if isinstance(v, Loop):
                for arg in filter(lambda x: isinstance(x, ArgDat), v.args):
                    dat_id = findIdx(self.dats, lambda d: d[0].ptr == v.dats[arg.dat_id].ptr)
                    if arg.access_type == AccessType.OPS_READ or arg.access_type == AccessType.OPS_RW:
                        if dat_id in temp_channels.keys():
                            temp_channel = temp_channels.pop(dat_id)
                            edges.append(DependancyEdge(temp_channel[0], temp_channel[1].id, dat_id, i, arg.id))
                        else:
                            edges.append(DependancyEdge(-1, len(source_dats), dat_id, i, arg.id))
                            source_dats.append([dat_id, arg])
                    if arg.access_type == AccessType.OPS_WRITE or arg.access_type == AccessType.OPS_RW:
                        temp_channels[dat_id] = [i, arg]
        
        for key in temp_channels.keys():
            temp_channel = temp_channels[key]   
            edges.append(DependancyEdge(temp_channel[0], temp_channel[1].id, key, len(self.dats), len(sink_dats)))
            sink_dats.append([key, temp_channel[1]])
        
        self.source_dats = source_dats
        self.sink_dats = sink_dats
        self.edges = edges
        
        for dat_id, arg in source_dats:
            arg_id = len(self.joint_args)

            idx = findIdx(sink_dats, lambda x: x[0] == dat_id)
            if idx:
                self.joint_args.append(ArgDat(arg_id, arg.loc, AccessType.OPS_RW, arg.opt, dat_id, arg.stencil_ptr, arg.dim, arg.restrict, arg.prolong, dat_id))
                sink_dats.remove(dat_id)
            else:
                self.joint_args.append(ArgDat(arg_id, arg.loc, AccessType.OPS_READ, arg.opt, dat_id, arg.stencil_ptr, arg.dim, arg.restrict, arg.prolong, dat_id))
                
        for dat_id, arg in sink_dats:
            arg_id = len(self.joint_args)
            self.joint_args.append(ArgDat(arg_id, arg.loc, AccessType.OPS_WRITE, arg.opt, dat_id, arg.stencil_ptr, arg.dim, arg.restrict, arg.prolong, dat_id))    

    def gen_PE_args(self) -> None:
        PE_args = []
        for i, v in enumerate(self.itr_args):
            if isinstance(v, Loop):
                PE_args.append(self.gen_PE_args_loop(i, v))
        self.PE_args = PE_args
        
    def gen_PE_args_loop(self, i: int,  v: Loop) -> List[str]:
        arg_map = {}
        PE_args = []
        for edge in self.edges:
            if i == edge.source_id:
                if edge.sink_id == len(self.dats):
                    search_list = filter(lambda x: x.access_type == AccessType.OPS_WRITE and x.dat_id == edge.dat_id, self.joint_args)
                    # print(f"search list: {search_list}")
                    arg_map[edge.source_arg_id] = f"arg{next(search_list).id}_hls_stream_out"
                else:
                    connector_name = f"node{edge.source_id}_{edge.source_arg_id}_to_node{edge.sink_id}_{edge.sink_arg_id}"
                    if connector_name not in self.interconnector_names:
                        self.interconnector_names.append(connector_name)
                    arg_map[edge.source_arg_id] = connector_name
            elif i == edge.sink_id:
                if edge.source_id == -1:
                    search_list = filter(lambda x: x.access_type == AccessType.OPS_READ and x.dat_id == edge.dat_id, self.joint_args)
                    arg_map[edge.sink_arg_id] = f"arg{next(search_list).id}_hls_stream_in"
                else:
                    connector_name = f"node{edge.source_id}_{edge.source_arg_id}_to_node{edge.sink_id}_{edge.sink_arg_id}"
                    if connector_name not in self.interconnector_names:
                        self.interconnector_names.append(connector_name)
                    arg_map[edge.sink_arg_id] = connector_name
        
        # print(f"PE_args: {arg_map}")
        for k in range(len(v.args)):
            if k in arg_map.keys():
                PE_args.append(arg_map[k])
        
        return PE_args
    
    def __str__(self) -> str:
        outer_loop_str = ""
        outer_loop_str += f"OPS Iterative Loop at {self.scope[0]}:\n ID: {self.id}, UID: {self.unique_id}, with num of iteration: {self.num_iter}\n\n DATS: \n ------ \n"
        
        for i,dat in enumerate(self.dats):
            outer_loop_str += f"dat{i}: " + str(dat) + "\n"
        
        outer_loop_str += "SOURCE DATS: " + str(self.source_dats) + "\n"
        outer_loop_str += "SINK DATS: " + str(self.sink_dats) + "\n"
        
        outer_loop_str +="\n JOINT_ARGS: \n ------ \n"
        for i,arg in enumerate(self.joint_args):
            outer_loop_str += f"arg{i}: " + str(arg) + "\n"
            
        outer_loop_str +="\n EDGES: \n ------ \n"

        for i, edge in enumerate(self.edges):
            outer_loop_str += f"edges{i}: " + str(edge) + "\n"
        
        outer_loop_str +="\n ARGS: \n ------ \n"
        
        dat_arg_i = 0
        for i,arg in enumerate(self.itr_args):
            outer_loop_str += f" arg{i}: " + str(arg)
            if (isinstance(arg, Loop)):
                outer_loop_str += f"   PE_args: {self.PE_args[dat_arg_i]} \n\n"
                dat_arg_i += 1
        return outer_loop_str
    
    def addParCopy(self, ParCopy: ParCopy) -> None:
        target_dat_id = findIdx(self.dats, lambda d: d[0].ptr == ParCopy.target)
        source_dat_id = findIdx(self.dats, lambda d: d[0].ptr == ParCopy.source)
        if target_dat_id is None:
            OpsError(f"ParCopy missing target dat used in any par-loop {ParCopy.target}")
        elif source_dat_id is None:
            OpsError(f"ParCopy missing source dat used in any par-loop {ParCopy.source}")

        self.dat_swap_map[target_dat_id] = source_dat_id
        self.dat_swap_map[source_dat_id] = target_dat_id 
         
        
    def addLoop(self, loop: Loop) -> None:
        for arg in loop.args:
            if isinstance(arg, ArgDat):
                dat_id = findIdx(self.dats, lambda d: d[0].ptr == loop.dats[arg.dat_id].ptr)
                
                if dat_id is None:
                    dat_id = len(self.dats)
                    self.dats.append([Dat(dat_id, loop.dats[arg.dat_id].ptr,loop.dats[arg.dat_id].dim, loop.dats[arg.dat_id].typ, loop.dats[arg.dat_id].soa), arg.access_type])
                else:
                    if (self.dats[dat_id][1] == AccessType.OPS_READ and arg.access_type == AccessType.OPS_WRITE) or \
                        (self.dats[dat_id][1] == AccessType.OPS_WRITE and arg.access_type == AccessType.OPS_READ):
                            self.dats[dat_id][1] = AccessType.OPS_RW    
                        
                loop.args[arg.id] = ArgDat(arg.dat_id, arg.loc, arg.access_type, arg.opt, arg.dat_id, arg.stencil_ptr, arg.dim, arg.restrict, arg.prolong, dat_id)    
        
            if self.ops_range == None:
                self.ops_range = loop.range.ptr
            elif self.ops_range != loop.range.ptr:
                OpsError("Missmatching ranging in par_loops within iter_par_loop scope")
                

            # elif isinstance(arg, ArgIdx):
            # elif isinstance(arg, ArgReduce):
            # elif isinstance(arg, ArgGbl):
    def getLoops(self)-> List[Loop]:
        return filter(lambda x: isinstance(x, Loop), self.itr_args)
    
    def getReadStencil(self) -> str:
        for arg in self.joint_args:
            if arg.access_type == AccessType.OPS_READ:
                return arg.stencil_ptr
        return None

    def getUniqueDatSwaps(self) -> List[Union[int,int]]:
        unique_swap_map = []
        for i in range(len(self.dat_swap_map)):
            found = False
            for j in unique_swap_map:
                if i == j[1]:
                    found = True
            if not found:
                unique_swap_map.append((i, self.dat_swap_map[i]))
        
        return unique_swap_map
    
    def getOrderedSwapPair(self, dat_id: int) -> Union[int, int]:
        other_dat = self.dat_swap_map[dat_id]
        pair = [other_dat, dat_id]
        pair.sort()
        return (pair)
               
            
    
class ParCopy:
    target: str
    source: str
    
    def __init__(self, target: str, source: str) -> None:
        self.target = target
        self.source = source
        
    def __str__(self) -> str:
        return f"OPS par copy from {self.source} to {self.target}"
        
class Loop:
    loc: Location
    kernel: str
    ast: str

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
    iterativeLoopId: Optional[int] = -1
    
    def __init__(self, ast: Any, loc: Location, kernel: str, block: str, range: Range, ndim: int) -> None:
        self.ast = ast
        self.loc = loc
        self.kernel = kernel
        self.block = block
        self.range = range
        self.ndim = ndim

        self.dats = []
        self.args = []
        self.stencils = []

    def __eq__ (self, other: Loop) -> bool:
        return self.ast == other.ast
    
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
        kernel_detail_str = f"Loop at {self.loc}:\n Kernel function: {self.kernel}, loop ast: {self.ast}, iter_loop_id: {self.iterativeLoopId}\n \
            range dim: {self.ndim}, block: {self.block}, range: {self.range}, arg_idx: {self.arg_idx}"
        args_str = "\n    ".join([str(a) for a in self.args])
        dat_str = "\n    ".join([str(d) for d in self.dats])

        if len(self.dats) > 0:
            dat_str = f"\n    {dat_str}\n"

        return f"{kernel_detail_str}\n  ARGS:\n{args_str}\n  DATS:\n{dat_str}\n"

    def get_read_stencil(self) -> str:
        for arg in filter(lambda x: isinstance(x, ArgDat), self.args):
            if arg.access_type == AccessType.OPS_READ or arg.access_type == AccessType.OPS_RW:
                return arg.stencil_ptr    
        return None
    
    def get_write_stencil(self) -> str:
        for arg in filter(lambda x: isinstance(x, ArgDat), self.args):
            if arg.access_type == AccessType.OPS_WRITE:
                return arg.stencil_ptr    
        return None
    