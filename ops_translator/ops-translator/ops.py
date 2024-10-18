from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Callable, List, Optional, Union, Tuple, Any

from util import ABDC, findIdx
from functools import cmp_to_key
import logging
import pygraphviz

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
            raise ValueError("Point cannot be initialized more than 3 dim")
        
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
    
    def __sub__(self, other: Point)->Point:
        return Point([self.x - other.x, self.y - other.y, self.z - other.z])
    
    def __neg__(self)->Point:
        return Point([-self.x, -self.y, -self.z])
    
    def __hash__(self) -> int:
        return hash(str(self.x) + "_" + str(self.y) + "_" + str(self.z))

def pointsToArray(points: List[Point], ndim: int) -> List[int]:
    array = []
    for point in points:
        for i in range(ndim):
            array.append(point[i])    
    return array

def pointCompare(point1: Point, point2: Point) -> int:
    
    if point1.z != point2.z:
        return point1.z - point2.z
    elif point1.y != point2.y:
        return point1.y - point2.y
    else:
        return point1.x - point2.x

def arrayToPoints(npoints: int, ndim: int, array: List[int]) -> List[Point]:
    
    points = []
    if len(array) != npoints * ndim :
        raise OpsError(f"Missmatch of parsed array with the stencil specification. Array: {array}, npoints: {npoints}, ndim: {ndim}")
    
    for i in range(npoints):
        point = []
        for j in range(ndim):
            # logging.debug(f"accessing: {array[i*ndim + j]}")
            point.append(array[i*ndim + j])
        points.append(Point(point))
    
    return points

def stencilPointsSort(npoints: int, ndim: int, array: List[int])-> List[Point]: 
    
    points = arrayToPoints(npoints, ndim, array)
    # logging.debug(f"Points before sort: {points}")
    sorted_points = sorted(points, key=cmp_to_key(pointCompare))
    # logging.debug(f"Points after sort: {sorted_points}")
    return sorted_points

def getStencilSize(array: List[Point])->int:
    xes = [point.x for point in array]
    minX = min(xes)
    maxX = max(xes)
    return (maxX - minX + 1)

def isInSameRow(one: Point , two: Point):
    if one.y == two.y and one.z == two.z:
        return True
    return False

def getMinPoint(array: List[Point]) -> Point:
    minPoint = Point([100,100,100])
    for  point in array:
        minPoint.x = min(minPoint.x, point.x)
        minPoint.y = min(minPoint.y, point.y)
        minPoint.z = min(minPoint.z, point.z)
    
    return minPoint

def cordinateOriginTranslation(origin: Point, array: List[Point]) -> List[Point]:
    translated = []
    for point in array:
        translated.append(Point([point.x + origin.x, point.y + origin.y, point.z + origin.z]))
    return translated

#Window buffer algo uses adjusted index, where base is (x_min, y_min, z_min)
def windowBuffChainingAlgo(sorted_array: List[Point], ndim: int) -> Tuple[List[str], List[Tuple[str, str]]]:
    chains = []
    unique_buffers = []
    # chains.append(("rd_val", "axis_read"))
    prev_buff = []
    feeding_point = []
    
    for p_idx in range(len(sorted_array)):
        if p_idx == len(sorted_array) - 1:
            chains.append((p_idx, "read_val"))
            if prev_buff:
                chains.append((prev_buff.pop(), feeding_point.pop()))
        elif isInSameRow(sorted_array[p_idx], sorted_array[p_idx+1]):
            chains.append((p_idx, p_idx+1))
        else:
            if sorted_array[p_idx+1].z == sorted_array[p_idx].z:
                buffer_type = BufferType.LINE_BUFF
                curr_buff_name = "buf_r" + str(sorted_array[p_idx].y) + "_" + str(sorted_array[p_idx+1].y) + "_p" + str(sorted_array[p_idx].z)
            else:
                buffer_type = BufferType.PLANE_BUFF
                curr_buff_name = "buf_p" + str(sorted_array[p_idx].z) + "_" + str(sorted_array[p_idx+1].z)
            curr_buff = WindowBuffer(curr_buff_name, buffer_type, sorted_array[p_idx+1], sorted_array[p_idx])
            unique_buffers.append(curr_buff)
            chains.append((p_idx, curr_buff))
            if prev_buff:
                chains.append((prev_buff.pop(), feeding_point.pop()))
            # print(p_idx)
            feeding_point.append(p_idx+1)
            prev_buff.append(curr_buff)

    return (unique_buffers, chains)  

def genRowDiscriptors(array: List[Point], base_point: Point = Point([0,0,0]))-> List[StencilRowDiscriptor]:
    
    row_discriptors = []
    
    for point in array:
        if StencilRowDiscriptor((point.y, point.z)) in row_discriptors:
            row_discriptors[row_discriptors.index(StencilRowDiscriptor((point.y, point.z)))].row_points.append(point)
        else:
            row_discriptors.append(StencilRowDiscriptor((point.y, point.z), base_point))
            row_discriptors[-1].row_points.append(point)
    
    return row_discriptors
    
def  computeWidenPoints(row_discriptors: List[StencilRowDiscriptor], vector_factor: int):
    widen_points = []
    init_point_to_widen_point_map = {}
    point_to_widen_point_map = {}
    
    for row in row_discriptors:
        base_point = row.base_point
        
        for point in row.row_points:
            
            if point.x - base_point.x < 0:
                widen_x = int((point.x - base_point.x - vector_factor + 1) / vector_factor)
            else: 
                widen_x = int((point.x - base_point.x + vector_factor - 1) / vector_factor)
                
            widen_point = Point([widen_x, row.row_id[0], row.row_id[1]])
            
            if widen_point not in widen_points:
                index = len(widen_points)
                widen_points.append(widen_point)
            else:
                index = widen_points.index(widen_point)
            init_point_to_widen_point_map[point] = index
            
    minWidenPoint = getMinPoint(widen_points)
    print("minwiden point: %d", minWidenPoint)
    widen_points = cordinateOriginTranslation(-minWidenPoint, widen_points)
    
    for key in init_point_to_widen_point_map.keys():
        point_to_widen_point_map[key] = widen_points[init_point_to_widen_point_map[key]]
    
    return widen_points, point_to_widen_point_map

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
class WindowBufferDiscriptor:
    widen_stencil: Stencil
    window_buffers: List[WindowBuffer]
    chains: List[Tuple[str, str]]
    point_to_widen_map: Optional[dict[Point, Point]]

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
    # window_buffers : List[WindowBuffer]
    # chains: List[Tuple[Union[int, WindowBuffer], Union[int, WindowBuffer, str]]]
    d_m: Point
    d_p: Point
    row_discriptors: Optional[List[StencilRowDiscriptor]] = field(default_factory=list)
    stride: Optional[list] = field(default_factory=list)
    read_origin_diff: Optional[List[Point]] = field(default_factory=list)
    
    def __post_init__(self):
        super().__setattr__('read_origin_diff', self.points[-1] - self.base_point)
        
    # def __post_init__(self):
    #     for point in self.points:
    #         if StencilRowDiscriptor((point.y, point.z)) in self.row_discriptors:
    #             self.row_discriptors[self.row_discriptors.index(StencilRowDiscriptor((point.y, point.z)))].row_points.append(point)
    #         else:
    #             self.row_discriptors.append(StencilRowDiscriptor((point.y, point.z), self.base_point))
    #             self.row_discriptors[-1].row_points.append(point)
                    
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
    is_read_only: Optional[bool] = False # if OPS_RW this is to disinguise in FPGA implementation wether via stream or RW stream.
    

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


# OPS Dataflow components 
class DFNodeType(Enum):
    DF_START = -1
    DF_END = -2 
    
    @staticmethod
    def values() -> List[str]:
        return [x.value for x in list(AccessType)]

@dataclass
class DependancyEdge:
    source_id: int
    source_arg_id: int
    dat_id: int
    sink_id: int
    sink_arg_id: int
    is_stray: Optional[bool] = False
    
    def __str__(self) -> str:
        return f"Edge:-> source_id: {self.source_id}, source_arg_id: {self.source_arg_id}, dat_id: {self.dat_id}, sink_id:{self.sink_id}, sink_arg_id: {self.sink_arg_id}, is_stray: {self.is_stray}"


@dataclass
class DataflowNode:
    loop: Loop
    node_id: int
    internal_dat_swap_map: Optional[map[int]] = field(default_factory=dict)
    
    def __post_init__(self) -> None:
        self.internal_dat_swap_map = {}
    
    def __str__(self) -> str:
        return f"Node:-> Loop: {self.loop.kernel}, node_id: {self.node_id}, interna_dat_swap_map: {self.internal_dat_swap_map}"

    def getArgDat(self, dat_ptr: str) -> Union[ArgDat, None]:
        local_dat_id = findIdx(self.loop.dats, lambda d: d.ptr == dat_ptr)
        
        for arg in self.loop.args:
            if isinstance(arg, ArgDat) and arg.dat_id == local_dat_id:
                return arg
        return None
    
@dataclass
class DataFlowGraph:
    unique_name: str
    nodes: Optional[List[DataflowNode]] = field(default_factory=list)
    edges: Optional[List[DependancyEdge]] = field(default_factory=list)
    global_dats: Optional[List[Dat]] = field(default_factory=list)
    global_dat_swap_map: Optional[List[int]] = field(default_factory=list)
    
    def __str__(self) -> str:
        prompt =  f"DataflowGraph {self.unique_name} \n"
        prompt += f"================================ \n\n"
        prompt += f"  nodes \n"
        prompt += f"  ----- \n"
        
        for node in self.nodes:
            prompt += f"    |- {node} \n"
            
        prompt += f"\n  edges \n"
        prompt += f"  ----- \n"
        
        for edge in self.edges:
            prompt += f"    |- {edge} \n"       
        
        prompt += f"\n  dats \n"
        prompt += f"  ---- \n"
        
        for dat in self.global_dats:
            prompt += f"    |- {dat} \n"    
        
        return prompt
    
    def print(self, filename: str, format: str = "png") -> None:
        logging.debug("Generating Dataflow graph image")
        g = pygraphviz.AGraph(strict=True, directed=True)
        
        nodeMap = {}
        nodeNameList = []
        
        for node in self.nodes:
            nodeMap[node.node_id] = f"{node.loop.kernel}({node.node_id})"
            nodeNameList.append(f"{node.loop.kernel}({node.node_id})")
        
        g.add_node("start", color="red")
        g.add_nodes_from(nodeNameList, shape="rectangle", color="blue")
        g.add_node("end", color="red")
        
        for edge in self.edges:
            type = 0
            if edge.source_id == DFNodeType.DF_START:
                source_node = "start"
                type = 1
            else:
                source_node = nodeMap[edge.source_id]
            
            if edge.sink_id == DFNodeType.DF_END:
                sink_node = "end"
                type = 2
            else:
                sink_node =  nodeMap[edge.sink_id]
            
            if type == 1:
                g.add_edge(source_node, f"{self.global_dats[edge.dat_id].ptr}", style="dashed")
            else:
                g.add_edge(source_node, f"{self.global_dats[edge.dat_id].ptr}")
            if type == 2:
                g.add_edge(f"{self.global_dats[edge.dat_id].ptr}", sink_node, style="dashed")
                if self.global_dat_swap_map[edge.dat_id] != edge.dat_id:
                    g.add_edge(f"{self.global_dats[edge.dat_id].ptr}", f"{self.global_dats[self.global_dat_swap_map[edge.dat_id]].ptr}", color="red")
            else:
                g.add_edge(f"{self.global_dats[edge.dat_id].ptr}", sink_node)
        
        g.layout(prog="dot")
        g.draw(f"{filename}.{format}")
        

    def addDat(self, dat: Dat)-> None:
        if not findIdx(self.global_dats, lambda d: d.ptr == dat.ptr):
            self.global_dats.append(dat)
    
    def getGlobalSourceDatIndices(self) -> List[int]:
        sourceDats = []
        
        for edge in self.edges:
            if edge.source_id == DFNodeType.DF_START:
                if edge.dat_id not in sourceDats:
                    sourceDats.append(edge.dat_id)
        return sourceDats
    
    def getGlobalSinkDatIndices(self) -> List[int]:
        sinkDats = []
        
        for edge in self.edges:
            if edge.sink_id == DFNodeType.DF_END:
                if edge.dat_id not in sinkDats:
                    sinkDats.append(edge.dat_id)    
        return sinkDats
        
    def getFirstReadingNode(self, dat_id: int) -> Union[DataflowNode, None]:
        fr_edge_idx = findIdx(self.edges, lambda edge: edge.dat_id == dat_id and edge.source_id == DFNodeType.DF_START)
        print (f"found first read edge idx: {fr_edge_idx}\n")
        if fr_edge_idx != None:
            print (f"firs read edge: {self.edges[fr_edge_idx]}, first read node: {self.getNode(self.edges[fr_edge_idx].sink_id)}")
            return self.getNode(self.edges[fr_edge_idx].sink_id)
        return None
    
    def getWritingNode(self, dat_id: int) -> Union[DataflowNode, None]:
        w_edge_idx = findIdx(self.edges, lambda edge: edge.dat_id == dat_id and edge.sink_id == DFNodeType.DF_END)
        print (f"found writing edge idx: {w_edge_idx}\n")
        if w_edge_idx != None:
            print (f"writing edge: {self.edges[w_edge_idx]}, first read node: {self.getNode(self.edges[w_edge_idx].source_id)}")
            return self.getNode(self.edges[w_edge_idx].source_id)
        return None

    def getNode(self, node_id: int) -> Optional[DataflowNode]:
        idx = findIdx(self.nodes, lambda node: node.node_id == node_id)
        if (idx != None):
            return self.nodes[idx]
        
        logging.warning(f"Node id: {node_id} does not exist in the dataflow graph: {self.unique_name}")
        return None
    
    def getEdge(self, node_id: int, dat_id: int = None, acc_dir: int = None) -> Optional[List[DependancyEdge]]:
        
        edges = []
        logging.debug(f"searching node: {node_id}, with dat_id: {dat_id}")
        for edge in self.edges:
            if (not acc_dir == None):
                if acc_dir == 0:
                    if not edge.source_id == node_id:
                        continue
                elif acc_dir == 1:
                    if not edge.sink_id == node_id:
                        continue 
            elif not (edge.sink_id == node_id or edge.source_id == node_id):
                continue
            if not((not dat_id == None) and dat_id == edge.dat_id):
                continue

            edges.append(edge)
        return edges
            
class IterLoop:
    unique_name: str
    id: int
    num_iter: Union[int, str]
    scope: List[Location]
    itrloop_args: List[Any]
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
    dataflow_graph: DataFlowGraph = None
    
    def __init__(self, unique_name: str, id: int, num_iter: Union[int, str], scope: List[Location], args: List[Any] = []) -> None:
        self.unique_name = unique_name
        self.id = id
        self.num_iter = num_iter
        self.scope = scope
        self.itrloop_args = args
        self.raw_dat_swap_map = filter(lambda x: isinstance(x, ParCopy), args)
        self.interconnector_names = []

        key =  ""
        for arg in args:
            if isinstance(arg, Loop):
                key += arg.kernel
                self.addLoop(arg)
                
        self.unique_id = hash(key)
        
        self.dat_swap_map = [i for i in range(len(self.dats))]
        
        for arg in args:
            if isinstance(arg, ParCopy):
                self.addParCopy(arg)
        
        self.gen_graph_v3()
        
        if logging.DEBUG >= logging.root.level:
            self.printDataflowGraph(f"{self.unique_name}")
        
        self.gen_global_dat_args() 
        self.gen_PE_args()
        self.gen_global_const_args()

    def gen_graph_v3(self) -> None:
        '''
        This generates pure dataflow graph IR
        '''
        # temporary_map_of_maps to store consumers from the node-produced-dat
        # key: dat value: {key: origin_loop.id, value: [consumer_loop.id, ....] }
        read_map_of_maps = {}
        
        # map to store current update. key: dat val: (loop.id, local arg_id)
        dat_current_update_map = {}
        
        self.dataflow_graph = DataFlowGraph(f"{self.unique_name}")
        
        for dat_id, (dat, AccessType) in enumerate(self.dats):
            self.dataflow_graph.addDat(dat)
            self.dataflow_graph.global_dat_swap_map.append(self.dat_swap_map[dat_id])
            
        for node_id, node in enumerate(self.itrloop_args):
            if not isinstance(node, Loop):
                continue
            
            self.dataflow_graph.nodes.append(DataflowNode(node, node_id))
            
            for arg in filter(lambda x: isinstance(x, ArgDat), node.args):
                global_dat_id = findIdx(self.dats, lambda d: d[0].ptr == node.dats[arg.dat_id].ptr)
                
                if arg.access_type == AccessType.OPS_READ or arg.access_type == AccessType.OPS_RW: 
                    if global_dat_id in dat_current_update_map.keys(): #if current reading dat being updated by previous nodes, then current node should have dependancy edge
                        new_edge = DependancyEdge(dat_current_update_map[global_dat_id][0], dat_current_update_map[global_dat_id][1], global_dat_id, node_id, arg.id)
                        self.dataflow_graph.edges.append(new_edge)
                        read_map_of_maps[global_dat_id][dat_current_update_map[global_dat_id][0]].append((node_id, arg.id))
                    else:
                        if global_dat_id not in read_map_of_maps.keys():
                            new_edge = DependancyEdge(DFNodeType.DF_START, 0, global_dat_id, node_id, arg.id)
                            read_map_of_maps[global_dat_id] = {DFNodeType.DF_START:[(node_id, arg.id)]}
                        elif (-1 not in read_map_of_maps[global_dat_id].keys()):
                            new_edge = DependancyEdge(DFNodeType.DF_START, 0, global_dat_id, node_id, arg.id)
                            read_map_of_maps[global_dat_id] = {DFNodeType.DF_START:[(node_id, arg.id)]}
                        else:
                            new_edge = DependancyEdge(DFNodeType.DF_START, len(read_map_of_maps[global_dat_id]), global_dat_id, node_id, arg.id)
                            read_map_of_maps[global_dat_id][DFNodeType.DF_START].append((node_id, arg.id))
                            
                        self.dataflow_graph.edges.append(new_edge)
                        
                        
                if arg.access_type == AccessType.OPS_WRITE or arg.access_type == AccessType.OPS_RW:
                    # Error if the dat is WAW as 
                    if global_dat_id in dat_current_update_map.keys():
                        if global_dat_id not in read_map_of_maps.keys() \
                            or not(read_map_of_maps[global_dat_id][dat_current_update_map[global_dat_id][0]]):
                            #TODO: if this is not an error make sure proper warning given as this might be a design flaw from user's side
                            raise OpsError(f"Dataflow failure: arg {dat_current_update_map[global_dat_id][1]} \
                                of par_loop {self.itrloop_args[dat_current_update_map[global_dat_id][0]].kernel} will be redundant \
                                write as no consumer for updated values as dat been overide by arg {arg.id} of par_loop {node.kernel}")
                    
                    dat_current_update_map[global_dat_id] = (node_id, arg.id)
                    
                    # Initializing read_map_of_map entries if not in the map_of_maps
                    if global_dat_id in read_map_of_maps.keys():
                        if node_id not in read_map_of_maps[global_dat_id]:
                            read_map_of_maps[global_dat_id][node_id] = []
                    else:
                        read_map_of_maps[global_dat_id]= {node_id: []}
                             
        # Analysing left over write dats that updated by ISL reigon
        for dat_id in dat_current_update_map.keys():
            # stray write, as this will be overide each iteration mark this stray write
            if dat_id not in read_map_of_maps.keys():
                new_edge = DependancyEdge(dat_current_update_map[dat_id][0], dat_current_update_map[dat_id][1], dat_id, DFNodeType.DF_END, -1, True)
                self.dataflow_graph.edges.append(new_edge)
            elif len(read_map_of_maps[dat_id][dat_current_update_map[dat_id][0]]) == 0:
                new_edge = DependancyEdge(dat_current_update_map[dat_id][0], dat_current_update_map[dat_id][1], dat_id, DFNodeType.DF_END, -1, True)
                self.dataflow_graph.edges.append(new_edge)
        
        self.read_map_of_maps = read_map_of_maps       
        
                   
                            
    def gen_graph_v2(self) -> None:
        # temporary_map_of_maps to store consumers from the node-produced-dat
        # key: dat value: {key: origin_loop.id, value: [consumer_loop.id, ....] }
        read_map_of_maps = {}
        # map to store current update. key: dat val: (loop.id, local arg_id)
        dat_current_update_map = {}
        edges = []
        source_dats = []
        sink_dats = []
        self.dataflow_graph = DataFlowGraph()
        
        # for aditional write channels
        loop_read_arg_sinks = {}
        
        # First Iteration initial edge generatitons inbetween loops
        for v_id,v in enumerate(self.itrloop_args):
            
            loop_read_arg_sinks[v_id] = []
            if not isinstance(v, Loop):
                continue
            
            self.dataflow_graph.nodes.append(DataflowNode(v, v_id))
            
            for arg in filter(lambda x: isinstance(x, ArgDat), v.args):
                global_dat_id = findIdx(self.dats, lambda d: d[0].ptr == v.dats[arg.dat_id].ptr) #finding the global dat_id from the dat info of par_loop node in iter_par_loop region
                
                if arg.access_type == AccessType.OPS_READ or arg.access_type == AccessType.OPS_RW:    
                    if global_dat_id in dat_current_update_map.keys(): #if current reading dat being updated by previous nodes, then current node should have dependancy edge
                        new_edge = DependancyEdge(dat_current_update_map[global_dat_id][0], dat_current_update_map[global_dat_id][1], global_dat_id, v_id, arg.id)
                        edges.append(new_edge)
                        self.dataflow_graph.edges.append(new_edge)
                        read_map_of_maps[global_dat_id][(dat_current_update_map[global_dat_id][0])].append((v_id, arg.id))
                    else:
                        new_edge = DependancyEdge(-1, len(source_dats), global_dat_id, v_id, arg.id)
                        edges.append(new_edge) #This dat is never been updated. Therefore, should come from the source dat 
                        self.dataflow_graph.edges.append(new_edge)
                        # dat_current_update_map[global_dat_id] = (-1, len(source_dats)) #source (-1) is recorded as the upding node of the dat
                        if (-1 not in read_map_of_maps[global_dat_id].keys()) or len(read_map_of_maps[global_dat_id]) == 0:
                            source_dats.append([global_dat_id, arg])
                        # print (f"global id: {global_dat_id}")
                        read_map_of_maps[global_dat_id] = {-1:[(v_id, arg.id)]}
                        
                        # By assuming one read and one right in each channels. There can be multiple reads from a source, where need to 
                        # implement how to add a streamSpliter.
                        
                        # here if a read from source dat without a swap map, the vertex loop need a output channel to be able to chain without
                        # reading from the memory. 
                        if self.dat_swap_map[global_dat_id] == global_dat_id:
                            dat_current_update_map[global_dat_id] = (v_id, arg.id)
                            if global_dat_id in read_map_of_maps.keys():
                                read_map_of_maps[global_dat_id][v_id] = []
                            else:
                                read_map_of_maps[global_dat_id]= {v_id: []}
                            
                        
                if arg.access_type == AccessType.OPS_WRITE or arg.access_type == AccessType.OPS_RW:
                    # checking previous update properly mapped
                    if global_dat_id in dat_current_update_map.keys():
                        if global_dat_id not in read_map_of_maps.keys() \
                            or not(read_map_of_maps[global_dat_id][dat_current_update_map[global_dat_id][0]]):
                            raise OpsError(f"Dataflow analysis failed: arg {dat_current_update_map[global_dat_id][1]} \
                                of par_loop {self.itrloop_args[dat_current_update_map[global_dat_id][0]].kernel} overide before read")
                            
                    dat_current_update_map[global_dat_id] = (v_id, arg.id)
                    if global_dat_id in read_map_of_maps.keys():
                        read_map_of_maps[global_dat_id][v_id] = []
                    else:
                        read_map_of_maps[global_dat_id]= {v_id: []}
                                              
        # Second iteration       
        for key in dat_current_update_map.keys():
            # check every update is properly mapped oterwise map to sink
            for src_v_id in read_map_of_maps[key]:
                if not read_map_of_maps[key][src_v_id]:
                    edges.append(DependancyEdge(dat_current_update_map[key][0], dat_current_update_map[key][1], key, -2, len(sink_dats)))
                    sink_dats.append([key, self.itrloop_args[dat_current_update_map[key][0]].args[dat_current_update_map[key][1]]])
                
        self.source_dats = source_dats
        self.sink_dats = sink_dats
        self.edges = edges 
        self.read_map_of_maps = read_map_of_maps
        
        for dat_id, arg in source_dats:
            arg_id = len(self.joint_args)
            print(f"Check dat id: {dat_id}")
            idx = findIdx(sink_dats, lambda x: x[0] == dat_id)
            print(f"found idx: {idx}")
            
            if idx != None:
                if self.dat_swap_map[dat_id] == dat_id:
                    # This means this is a via stream for chaining not to be store back to memory.
                    self.joint_args.append(ArgDat(arg_id, arg.loc, AccessType.OPS_RW, arg.opt, dat_id, arg.stencil_ptr, arg.dim, arg.restrict, arg.prolong, dat_id, True))
                else:
                    self.joint_args.append(ArgDat(arg_id, arg.loc, AccessType.OPS_RW, arg.opt, dat_id, arg.stencil_ptr, arg.dim, arg.restrict, arg.prolong, dat_id))
                del sink_dats[idx]
            else:
                self.joint_args.append(ArgDat(arg_id, arg.loc, AccessType.OPS_READ, arg.opt, dat_id, arg.stencil_ptr, arg.dim, arg.restrict, arg.prolong, dat_id))
                
        for dat_id, arg in sink_dats:
            arg_id = len(self.joint_args)
            self.joint_args.append(ArgDat(arg_id, arg.loc, AccessType.OPS_WRITE, arg.opt, dat_id, arg.stencil_ptr, arg.dim, arg.restrict, arg.prolong, dat_id))    

        print (f'read_map_of_maps: {read_map_of_maps}') 
        for edge in edges:
            print (f"edge: {edge}")
        
    def gen_global_dat_args(self) -> None:
        source_dats = self.dataflow_graph.getGlobalSourceDatIndices()
        sink_dats = self.dataflow_graph.getGlobalSinkDatIndices()
        
        print (f"source dats: {source_dats}, sink_dats: {sink_dats}")
        for dat_id in source_dats:
            idx = findIdx(sink_dats, lambda x: x == dat_id)
            node = self.dataflow_graph.getFirstReadingNode(dat_id)
            
            if not node:
                raise OpsError(f"Error finding node that read from dat: {self.dats[dat_id][0].ptr}")
            
            arg = node.getArgDat(self.dats[dat_id][0].ptr)
            
            if not arg:
                raise OpsError(f"Error finding ArgDat from node: {node.node_id} of parloop: {node.loop.kernel}, loc: {node.loop.loc}")
            if idx:
                if self.dat_swap_map[dat_id] == dat_id:
                    self.joint_args.append(ArgDat(len(self.joint_args), arg.loc, AccessType.OPS_RW, arg.opt, dat_id, arg.stencil_ptr, arg.dim, arg.restrict, arg.prolong, dat_id, True))
                else:
                     self.joint_args.append(ArgDat(len(self.joint_args), arg.loc, AccessType.OPS_RW, arg.opt, dat_id, arg.stencil_ptr, arg.dim, arg.restrict, arg.prolong, dat_id))
                del sink_dats[idx]
            else:
                self.joint_args.append(ArgDat(len(self.joint_args), arg.loc, AccessType.OPS_READ, arg.opt, dat_id, arg.stencil_ptr, arg.dim, arg.restrict, arg.prolong, dat_id))
   
        for dat_id in sink_dats:
            node = self.dataflow_graph.getWritingNode(dat_id)
            
            if not node:
                raise OpsError(f"Error finding node that read from dat: {self.dats[dat_id][0].ptr}")
            
            arg = node.getArgDat(self.dats[dat_id][0].ptr)
            
            if not arg:
                raise OpsError(f"Error finding ArgDat from node: {node.node_id} of parloop: {node.loop.kernel}, loc: {node.loop.loc}")

            self.joint_args.append(ArgDat(len(self.joint_args), arg.loc, AccessType.OPS_WRITE, arg.opt, dat_id, arg.stencil_ptr, arg.dim, arg.restrict, arg.prolong, dat_id))    


    def gen_global_const_args(self) -> None:
            
            global_args_ptrs = []
            global_args = []
            for v_id,v in enumerate(self.itrloop_args):
                if isinstance(v, Loop):
                    print(f"args: {v.args}")
                    for arg in filter(lambda x: isinstance(x, ArgGbl), v.args):
                        if arg.ptr not in global_args_ptrs:
                            global_args_ptrs.append(arg.ptr)
                            global_args.append(arg)
            
            for arg in global_args:
                self.joint_args.append(arg)
        
            
                    
    # def gen_graph(self) -> None:
    #     temp_channels = {}
    #     edges = []
    #     source_dats = []
    #     sink_dats = []
    #     for i, v in enumerate(self.itrloop_args):
    #         if isinstance(v, Loop):
    #             for arg in filter(lambda x: isinstance(x, ArgDat), v.args):
    #                 dat_id = findIdx(self.dats, lambda d: d[0].ptr == v.dats[arg.dat_id].ptr)
    #                 if arg.access_type == AccessType.OPS_READ or arg.access_type == AccessType.OPS_RW:
    #                     if dat_id in temp_channels.keys():
    #                         temp_channel = temp_channels.pop(dat_id)
    #                         edges.append(DependancyEdge(temp_channel[0], temp_channel[1].id, dat_id, i, arg.id))
    #                     else:
    #                         edges.append(DependancyEdge(-1, len(source_dats), dat_id, i, arg.id))
    #                         source_dats.append([dat_id, arg])
    #                 if arg.access_type == AccessType.OPS_WRITE or arg.access_type == AccessType.OPS_RW:
    #                     temp_channels[dat_id] = [i, arg]
        
    #     for key in temp_channels.keys():
    #         temp_channel = temp_channels[key]   
    #         edges.append(DependancyEdge(temp_channel[0], temp_channel[1].id, key, -2, len(sink_dats)))
    #         sink_dats.append([key, temp_channel[1]])
        
    #     self.source_dats = source_dats
    #     self.sink_dats = sink_dats
    #     self.edges = edges
        
    #     for dat_id, arg in source_dats:
    #         arg_id = len(self.joint_args)

    #         idx = findIdx(sink_dats, lambda x: x[0] == dat_id)
    #         if idx:
    #             self.joint_args.append(ArgDat(arg_id, arg.loc, AccessType.OPS_RW, arg.opt, dat_id, arg.stencil_ptr, arg.dim, arg.restrict, arg.prolong, dat_id))
    #             sink_dats.remove(dat_id)
    #         else:
    #             self.joint_args.append(ArgDat(arg_id, arg.loc, AccessType.OPS_READ, arg.opt, dat_id, arg.stencil_ptr, arg.dim, arg.restrict, arg.prolong, dat_id))
                
    #     for dat_id, arg in sink_dats:
    #         arg_id = len(self.joint_args)
    #         self.joint_args.append(ArgDat(arg_id, arg.loc, AccessType.OPS_WRITE, arg.opt, dat_id, arg.stencil_ptr, arg.dim, arg.restrict, arg.prolong, dat_id))    

        
    def gen_PE_args(self) -> None:
        PE_args = []
        for i, v in enumerate(self.itrloop_args):
            if isinstance(v, Loop):
                PE_args.append(self.gen_PE_args_loop(i, v))
        self.PE_args = PE_args
        
    def gen_PE_args_loop(self, i: int,  v: Loop) -> List[str]:
        arg_map = {}
        PE_args = []
        for edge in self.dataflow_graph.edges:
            if i == edge.source_id:
                if edge.sink_id == DFNodeType.DF_END:
                    search_list = filter(lambda x: (x.access_type == AccessType.OPS_WRITE or x.access_type == AccessType.OPS_RW) and x.dat_id == edge.dat_id, self.joint_args)
                    # print(f"search list: {[i for i in search_list]}")
                    if edge.source_arg_id in arg_map.keys():
                        arg_map[edge.source_arg_id].append(f"arg{next(search_list).id}_hls_stream_out")
                    else:
                        arg_map[edge.source_arg_id] = [f"arg{next(search_list).id}_hls_stream_out"]
                else:
                    connector_name = f"node{edge.source_id}_{edge.source_arg_id}_to_node{edge.sink_id}_{edge.sink_arg_id}"
                    if connector_name not in self.interconnector_names:
                        self.interconnector_names.append(connector_name)
                    arg_map[edge.source_arg_id] = connector_name
            elif i == edge.sink_id:
                if edge.source_id == DFNodeType.DF_START:
                    search_list = filter(lambda x: x.access_type in [AccessType.OPS_READ, AccessType.OPS_RW] and x.dat_id == edge.dat_id, self.joint_args)
                    arg_id = next(search_list).id
                    if edge.sink_arg_id in arg_map.keys():
                        arg_map[edge.sink_arg_id].append(f"arg{arg_id}_hls_stream_in")
                    else:
                        arg_map[edge.sink_arg_id] = [f"arg{arg_id}_hls_stream_in"]
                        
                else:
                    connector_name = f"node{edge.source_id}_{edge.source_arg_id}_to_node{edge.sink_id}_{edge.sink_arg_id}"
                    if connector_name not in self.interconnector_names:
                        self.interconnector_names.append(connector_name)
                    arg_map[edge.sink_arg_id] = [connector_name]
        
        # print(f"PE_args: {arg_map}")
        for k in range(len(v.args)):
            if k in arg_map.keys():
                if isinstance(arg_map[k], list):
                    [PE_args.append(s) for s in arg_map[k]]
                    continue
                PE_args.append(arg_map[k])
        
        return PE_args
    
    def __str__(self) -> str:
        outer_loop_str = ""
        outer_loop_str += f"OPS Iterative Loop at {self.scope[0]}:\n ID: {self.id}, UID: {self.unique_id}, with num of iteration: {self.num_iter}\n\n DATS: \n ------ \n"
        
        # for i,dat in enumerate(self.dats):
        #     outer_loop_str += f"dat{i}: " + str(dat) + "\n"
        
        # outer_loop_str += "SOURCE DATS: " + str(self.source_dats) + "\n"
        # outer_loop_str += "SINK DATS: " + str(self.sink_dats) + "\n"
        
        # outer_loop_str +="\n JOINT_ARGS: \n ------ \n"
        # for i,arg in enumerate(self.joint_args):
        #     outer_loop_str += f"arg{i}: " + str(arg) + "\n"
            
        # outer_loop_str +="\n EDGES: \n ------ \n"

        # for i, edge in enumerate(self.edges):
        #     outer_loop_str += f"edges{i}: " + str(edge) + "\n"
        
        # outer_loop_str +="\n SWAP MAP: \n ------ \n"
            
        # for i,j in enumerate(self.dat_swap_map):
        #     outer_loop_str += f"dat{i} - dat{j}\n"
        
        # outer_loop_str +="\n ARGS: \n ------ \n"
        
        # dat_arg_i = 0
        # for i,arg in enumerate(self.itrloop_args):
        #     outer_loop_str += f" arg{i}: " + str(arg)
        #     if (isinstance(arg, Loop)):
        #         outer_loop_str += f"   PE_args: {self.PE_args[dat_arg_i]} \n\n"
        #         dat_arg_i += 1
        return outer_loop_str
    
    def addParCopy(self, ParCopy: ParCopy) -> None:
        target_dat_id = findIdx(self.dats, lambda d: d[0].ptr == ParCopy.target)
        source_dat_id = findIdx(self.dats, lambda d: d[0].ptr == ParCopy.source)
        if target_dat_id is None:
            raise OpsError(f"ParCopy missing target dat used in any par-loop '{ParCopy.target}'")
        elif source_dat_id is None:
            raise OpsError(f"ParCopy missing source dat used in any par-loop '{ParCopy.source}'")

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
                raise OpsError("Missmatching ranging in par_loops within iter_par_loop scope")
                

            # elif isinstance(arg, ArgIdx):
            # elif isinstance(arg, ArgReduce):
            # elif isinstance(arg, ArgGbl):
    def getLoops(self)-> List[Loop]:
        return filter(lambda x: isinstance(x, Loop), self.itrloop_args)
    
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
         
    def getArg(self, dat_id: int) -> Union[ArgDat, None]:
        for arg in self.joint_args:
            if arg.dat_id == dat_id:
                return arg
        return None      
            
    def printDataflowGraph(self, filename: str) -> None:
        print(self.dataflow_graph)
        self.dataflow_graph.print(self.unique_name)
        
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
    read_stencil: Stencil = None
    write_stencil: Stencil = None

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

    def get_read_stencil(self, prog) -> str:
        """! return the union of all read stenils

        Returns:
            str: _description_
        """
        
        if self.read_stencil:
            return self.read_stencil
        
        unique_stencil_names = self.get_unique_read_stencil_names()
        unique_stencils = []
        
        for stencil_name in unique_stencil_names:
            unique_stencils.append(prog.findStencil(stencil_name))
        
        id = -1
        dim = 0
        stencil_ptr = "read_stencil"
        num_points = 0
        base_afine_points = []
        base_point = Point([0,0,0])
        d_m = [0,0,0]
        d_p = [0,0,0]
          
        for i, stencil in enumerate(unique_stencils):
            print(f"Unique stencil {i} - {stencil}")
            
            dim = max(dim, stencil.dim)
            local_base_afine_points = [(p - stencil.base_point) for p in stencil.points]
            base_afine_points.extend([p for p in local_base_afine_points if p not in base_afine_points])
            
            print (f"stencil_dim: {stencil.dim}")
            for i in range(stencil.dim):
                d_m[i] = max(d_m[i], stencil.d_m[i])
                d_p[i] = max(d_p[i], stencil.d_p[i])
        
        minPoint = getMinPoint(base_afine_points)
        base_point = -minPoint
        points = cordinateOriginTranslation(base_point, base_afine_points)
                
        num_points = len(points)
        xes = [point.x for point in points]
        minX = min(xes)
        maxX = max(xes)
        stencil_size = (maxX - minX + 1)
        
        row_discriptors = genRowDiscriptors(points, base_point)
        
        self.read_stencil = Stencil(id, dim, stencil_ptr, num_points, points, base_point, stencil_size, d_m, d_p, row_discriptors)
        
        print(f"read stencil: {self.read_stencil}")
        
        return self.read_stencil
        
        
    def get_write_stencil(self) -> str:
        """ Write Stencil is always 

        Returns:
            str: _description_
        """
        if self.write_stencil:
            return self.write_stencil
        
        self.write_stencil = Stencil(-1, self.ndim, "default_write_stencil", 1, [Point([0,0,0])], Point([0,0,0,]), 1, [], [], Point([0,0,0]), Point([0,0,0]))
        return self.write_stencil
    
    def get_unique_read_stencil_names(self) -> List[str]:
        unique_stencil_names = []
        for arg in filter(lambda x: isinstance(x, ArgDat), self.args):
            if arg.access_type in [AccessType.OPS_READ, AccessType.OPS_RW] and arg.stencil_ptr not in unique_stencil_names:
                unique_stencil_names.append(arg.stencil_ptr)
        return unique_stencil_names
    
    def get_dat_name(self, arg_id: int) -> str:
        assert isinstance(self.args[arg_id], ArgDat)
        dat_id = self.args[arg_id].dat_id 
        return self.dats[dat_id].ptr
        
    def get_arg_dat(self, dat_ptr: str, acc_type: List[AccessType] = []) -> Optional[List[ArgDat]]:
        candidate_arg_dats = []
        for arg in self.args:
            if not isinstance(arg, ArgDat):
                continue
            if not self.dats[arg.dat_id].ptr == dat_ptr:
                continue 
            if not acc_type:
                candidate_arg_dats.append(arg)
            elif arg.access_type in acc_type:
                candidate_arg_dats.append(arg)
        
        return candidate_arg_dats