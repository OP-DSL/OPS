from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Callable, List, Optional, Union, Tuple, Any, Dict

from util import ABDC, findIdx
from functools import cmp_to_key
import logging
import rustworkx as rx
from rustworkx.visualization import graphviz_draw
import pygraphviz
if TYPE_CHECKING:
    from store import Location
from copy import deepcopy

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
    DF_LOOP = -3
    DF_DAT = -4
    
    @staticmethod
    def values() -> List[str]:
        return [x.value for x in list(AccessType)]

# @dataclass
# class DependancyEdge:
#     source_id: int
#     source_arg_id: int
#     dat_id: int
#     sink_id: int
#     sink_arg_id: int
#     is_stray: Optional[bool] = False
    
#     def __str__(self) -> str:
#         return f"Edge:-> source_id: {self.source_id}, source_arg_id: {self.source_arg_id}, dat_id: {self.dat_id}, sink_id:{self.sink_id}, sink_arg_id: {self.sink_arg_id}, is_stray: {self.is_stray}"


@dataclass
class BaseDataflowNode:
    type: DFNodeType
    node_uid: Optional[int] = -1
    
DF_START_NODE = BaseDataflowNode(DFNodeType.DF_START)
DF_END_NODE = BaseDataflowNode(DFNodeType.DF_END)

@dataclass
class DatDataflowNode:
    dat_name: str
    type: Optional[DFNodeType] = DFNodeType.DF_DAT
@dataclass
class DataflowNode:
    loop: Loop
    node_uid: Optional[int] = -1
    internal_dat_swap_map: Optional[Dict[str, str]] = field(default_factory=dict)
    type: Optional[DFNodeType] = DFNodeType.DF_LOOP
    
    def __post_init__(self) -> None:
        self.internal_dat_swap_map = {}
    
    def __str__(self) -> str:
        return f"Node:-> Loop: {self.loop.kernel}, node_uid: {self.node_uid}, internal_dat_swap_map: {self.internal_dat_swap_map}"

    def getArgDat(self, dat_ptr: str) -> Union[ArgDat, None]:
        local_dat_id = findIdx(self.loop.dats, lambda d: d.ptr == dat_ptr)
        
        for arg in self.loop.args:
            if isinstance(arg, ArgDat) and arg.dat_id == local_dat_id:
                return arg
        return None
    
@dataclass
class DataflowGraph_v2:
    unique_name: str
    __graph: Optional[rx.PyDiGraph] = rx.PyDiGraph(multigraph=True)
    __global_dats: Optional[List[Dat]] = field(default_factory=list)
    __global_dat_swap_map: Optional[Dict[str, str]] = field(default_factory=dict)
    
    def __post_init__(self) -> None:
        if not self.__graph.nodes():
            self.addNode(DF_START_NODE)
            self.addNode(DF_END_NODE)
        
    def __str__(self) -> str:
        prompt =  f"DataflowGraph {self.unique_name} \n"
        prompt += f"================================ \n\n"
        prompt += f"  nodes \n"
        prompt += f"  ----- \n"
        
        for node in self.__graph.nodes():
            prompt += f"    |- {node} \n"
            
        prompt += f"\n  edges \n"
        prompt += f"  ----- \n"
        
        for edge in self.__graph.edges():
            prompt += f"    |- {edge} \n"       
        
        prompt += f"\n  dats \n"
        prompt += f"  ---- \n"
        
        for dat in self.__global_dats:
            prompt += f"    |- {dat} \n"    
        
        return prompt

    def addDat(self, dat: Dat)-> None:
        if not findIdx(self.__global_dats, lambda d: d.ptr == dat.ptr):
            self.__global_dats.append(dat)
        if dat.ptr not in self.__global_dat_swap_map.keys():
            self.__global_dat_swap_map[dat.ptr] = dat.ptr

    def addDatSwapUpdate(self, dat_a: str, dat_b: str):
        if dat_a in self.__global_dat_swap_map.keys():
            self.__global_dat_swap_map[dat_a] = dat_b
        else:
            OpsError(f"dat {dat_a} is not in the dat_swap_map")
        if dat_b in self.__global_dat_swap_map.keys():
            self.__global_dat_swap_map[dat_b] = dat_a
        else:
            OpsError(f"dat {dat_b} is not in the dat_swap_map")
            
    def addEdge(self, src_id: int, src_arg_id: int, dat_str: str, sink_id: int, sink_arg_id: int, isStray: bool = False) -> None:
        src_node = self.findNodeById(src_id)
        sink_node = self.findNodeById(sink_id)
        
        if src_node is None:
            logging.error(f"{self}")
            raise OpsError(f"Couldn't find node with id: {src_id}")
        if sink_node is None:
            raise OpsError(f"Couldn't find node with id: {sink_id}")
        
        datIdx = findIdx(self.__global_dats, lambda dat: dat.ptr == dat_str)
        if datIdx is None:
            raise OpsError(f"Failed to find {dat_str} in global dats")

        self.__graph.add_edge(self.__graph.nodes().index(src_node), self.__graph.nodes().index(sink_node), {"weight": 1, "src_arg_id": src_arg_id, "sink_arg_id": sink_arg_id, "dat_str": dat_str, "isStray": isStray})

    def addNode(self, df_node: DataflowNode) -> int:
        if findIdx(self.__graph.nodes(), lambda node: node.node_uid == df_node.node_uid):
            raise OpsError(f"two nodes can't have same uid, {self.unique_name}")
        node_id = self.__graph.add_node(df_node)
        self.__graph[node_id].node_uid = node_id
        logging.debug(f"Adding node: {self.__graph[node_id]}")
        return node_id

    def copy(self, new_unique_name: str = None) -> DataflowGraph_v2:
        if not new_unique_name is None:
            unique_name = new_unique_name
        else:
            unique_name = self.unique_name + "_copy"
        copy_inst = DataflowGraph_v2(unique_name, self.__graph.copy(), deepcopy(self.__global_dats), (self.__global_dat_swap_map))
        return copy_inst

    def deleteNode(self, node_uid: int) -> None:
        if not self.__graph.has_node(node_uid):
            OpsError(f"Delete failed. Node with node id {node_uid} does not exist.")
        self.__graph.remove_node(node_uid)
        
    def findNodeById(self, node_uid: int) -> Optional[DataflowNode]:
        idx = findIdx(self.__graph.nodes(), lambda node: node.node_uid == node_uid)
        
        if idx is None:
            return None
        return self.__graph.get_node_data(idx)

    def findNodesByKernelName(self, node_kernel_name: str) -> Optional[List[DataflowNode]]:
        return filter(lambda node: node.loop.kernel == node_kernel_name, self.__graph.nodes())

    def getEdges(self) -> List[Any]:
        edge_list = self.__graph.edge_list()
        edge_attr_list = self.__graph.edges()
        
        merged = [(edge_list[i][0], edge_list[i][1], edge_attr_list[i]) for i in range(0,len(edge_list))]
        return merged

    def getEndNodeIdx(self) -> None:
        return self.__graph.nodes().index(DF_END_NODE)

    def getFirstReadingNode(self, dat_ptr: str) -> Optional[DataflowNode]:
        start_node_id = self.getStartNodeIdx()
        
        for src, sink, edge_attr in self.__graph.out_edges(start_node_id):
            if edge_attr["dat_str"] == dat_ptr:
                return self.getNode(sink)
        return None 

    def getFirstWritingNode(self, dat_ptr: str) -> Optional[DataflowNode]:
        end_node_id = self.getEndNodeIdx()
        
        for src, sink, edge_attr in self.__graph.in_edges(end_node_id):
            if edge_attr["dat_str"] == dat_ptr:
                return self.getNode(src)
        return None

    def getGlobalDats(self) -> List[Dat]:
        return self.__global_dats
    
    def getGlobalDatsSwapMap(self) -> Dict[str, str]:
        return self.__global_dat_swap_map

    def getGlobalSinkDatIndices(self) -> List[int]:
        sink_dat_names = self.getGlobalSinkDatNames()
        sink_dat_indices = []

        for dat_name in sink_dat_names:
            sink_dat_index = findIdx(self.__global_dats, lambda dat: dat.ptr == dat_name)
            if sink_dat_index is None:
                raise OpsError(f"Failed to find sink dat: {dat_name} in DataflowGraph {self.unique_name} global dats")
            sink_dat_indices.append(sink_dat_index)
        return sink_dat_indices

    def getGlobalSinkDatNames(self) -> List[str]:
        sink_dat_names = []
        end_node_id = self.getEndNodeIdx()
        for src, sink, edge_attr in self.__graph.in_edges(end_node_id):
            sink_dat_names.append(edge_attr["dat_str"])
        return sink_dat_names


    def getGlobalSourceDatIndices(self) -> List[int]:
        source_dat_names = self.getGlobalSourceDatNames()
        source_dat_indices = []
        
        for dat_name in source_dat_names:
            source_dat_index = findIdx(self.__global_dats, lambda dat: dat.ptr == dat_name)
            if source_dat_index is None:
                raise OpsError(f"Failed to find source dat: {dat_name} in DataflowGraph {self.unique_name} global dats")
            source_dat_indices.append(source_dat_index)
        return source_dat_indices

    def getGlobalSourceDatNames(self) -> List[str]:
        source_dat_names = []
        start_node_id = self.getStartNodeIdx()
        
        for src, sink, edge_attr in self.__graph.out_edges(start_node_id):
            source_dat_names.append(edge_attr["dat_str"])
        return source_dat_names

    def getInEdgesFromNode(self, node_uid: int) -> List[Any]:
        if not self.__graph.has_node(node_uid):
            OpsError(f"Cannot retrieve in edges from node_id: {node_uid} as it does not exist")
        return self.__graph.in_edges(node_uid)

    def getNode(self, node_uid: int) -> Optional[DataflowNode]:
        if node_uid in self.__graph.node_indices():
            return self.__graph[node_uid]
        return None

    def getAllLoopNodes(self) -> List[DataflowNode]:
        return [node for node in self.__graph.nodes() if isinstance(node, DataflowNode)]
    
    def getOutEdgesFromNode(self, node_uid: int) -> List[Any]:
        if not self.__graph.has_node(node_uid):
            OpsError(f"Cannot retrieve out edges from node_id: {node_uid} as it does not exist")
        return self.__graph.out_edges(node_uid)
    
    def getRXGraph(self) -> rx.PyDiGraph:
        return self.__graph

    def getStartNodeIdx(self) -> None:
        return self.__graph.nodes().index(DF_START_NODE)  

    def isNodeExist(self, node_uid: int) -> Bool:
        if findIdx(self.__graph.nodes(), lambda node: node.node_uid == node_uid):
            return True
        return False

    def print(self, filename: str, format: str = "png", make_dats_node: bool = False) -> None:
        def node_attr(node):
            if node.type == DFNodeType.DF_START:
                return {"color": "red", "label": "START"}
            elif node.type == DFNodeType.DF_END:
                return {"color": "red", "label": "END"}
            elif node.type == DFNodeType.DF_LOOP:
                return {"color": "blue", "label": f"{node.node_uid}:{node.loop.kernel}", "shape" : "box"}
            elif node.type == DFNodeType.DF_DAT:
                return {"label": f"{node.dat_name}", "color" : "green"}
            
        def edge_attr(edge_det):
            # edge, attr =  edge_det
            if "dat_connect" in edge_det.keys():
                if edge_det["swap_connect"]:
                    return {"color": "red"}
                return {}
        
            else:
                return {"label": f"{edge_det['dat_str']}"}
        
        if not make_dats_node:  
            graphviz_draw(self.__graph, node_attr_fn=node_attr, edge_attr_fn=edge_attr, filename=f"{filename}.{format}", image_type=f"{format}")
        else:
            copy_graph = self.__graph.copy()
            
            edge_list = copy_graph.edge_list()
            edge_attr_list = copy_graph.edges()
        
            edges = [(edge_list[i][0], edge_list[i][1], edge_attr_list[i]) for i in range(0,len(edge_list))]

            copy_graph.clear_edges()
            added_dat_id_map = {}
            
            for src_id, sink_id, attr in edges:
                if not attr["dat_str"] in added_dat_id_map.keys():
                    dat_node_id = copy_graph.add_node(DatDataflowNode(attr["dat_str"]))
                    added_dat_id_map[attr["dat_str"]] = dat_node_id
                dat_node_id = added_dat_id_map[attr["dat_str"]]
                copy_graph.add_edge(src_id, dat_node_id, {"weight" : 1, "dat_connect" : True, "swap_connect": False})
                copy_graph.add_edge(dat_node_id, sink_id, {"weight" : 1, "dat_connect" : True, "swap_connect": False})
            
                if sink_id == self.getEndNodeIdx() and not self.__global_dat_swap_map[attr["dat_str"]] == attr["dat_str"]:
                    if not self.__global_dat_swap_map[attr["dat_str"]] in added_dat_id_map.keys():
                        dat_node_id = copy_graph.add_node(DatDataflowNode(self.__global_dat_swap_map[attr["dat_str"]]))
                        added_dat_id_map[self.__global_dat_swap_map[attr["dat_str"]]] = dat_node_id
                    sink_dat_node_id = added_dat_id_map[self.__global_dat_swap_map[attr["dat_str"]]]
                    copy_graph.add_edge(dat_node_id, sink_dat_node_id, {"weight" : 1, "dat_connect" : True, "swap_connect": True})
            
            graphviz_draw(copy_graph, node_attr_fn=node_attr, edge_attr_fn=edge_attr, filename=f"{filename}.{format}", image_type=f"{format}")
                # graphviz_obj = rx.visualization.graphviz_graph(self.__graph)
                # graphviz_obj.layout(prog="dot")
                # graphviz_obj.draw(f"{filename}.{format}")


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
    dat_swap_map: List[int]
    raw_dat_swap_map: List[ParCopy]
    PE_args: List[List[str]]
    interconnector_names: List[str]
    ops_range: str = None
    df_graph: DataflowGraph_v2 = None
    opt_df_graph: DataflowGraph_v2 = None
        
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
        
        self.df_graph = DataflowGraph_v2(f"{self.unique_name}")
        
        for dat_id, (dat, AccessType) in enumerate(self.dats):
            self.df_graph.addDat(dat)
            self.df_graph.addDatSwapUpdate(dat.ptr, self.dats[self.dat_swap_map[dat_id]][0].ptr)
            # self.df_graph.global_dat_swap_map.append(self.dat_swap_map[dat_id])
            
        for node in self.itrloop_args:
            if not isinstance(node, Loop):
                continue
            
            node_id = self.df_graph.addNode(DataflowNode(node))
            
            for arg in filter(lambda x: isinstance(x, ArgDat), node.args):
                dat_ptr = node.dats[arg.dat_id].ptr
                global_dat_id = findIdx(self.dats, lambda d: d[0].ptr == dat_ptr)
                if global_dat_id is None:
                    OpsError(f"couldn't find dat: {dat_ptr} in IterParLoop dats")
                
                if arg.access_type == AccessType.OPS_READ or arg.access_type == AccessType.OPS_RW: 
                    if global_dat_id in dat_current_update_map.keys(): #if current reading dat being updated by previous nodes, then current node should have dependancy edge
                        # new_edge = DependancyEdge(dat_current_update_map[global_dat_id][0], dat_current_update_map[global_dat_id][1], global_dat_id, node_id, arg.id)
                        # self.df_graph.edges.append(new_edge)
                        self.df_graph.addEdge(dat_current_update_map[global_dat_id][0], dat_current_update_map[global_dat_id][1], dat_ptr, node_id, arg.id)
                        read_map_of_maps[global_dat_id][dat_current_update_map[global_dat_id][0]].append((node_id, arg.id))
                    else:
                        if global_dat_id not in read_map_of_maps.keys():
                            # new_edge = DependancyEdge(DFNodeType.DF_START, 0, global_dat_id, node_id, arg.id)
                            self.df_graph.addEdge(self.df_graph.getStartNodeIdx(), 0, dat_ptr, node_id, arg.id)
                            read_map_of_maps[global_dat_id] = {DFNodeType.DF_START:[(node_id, arg.id)]}
                        elif (-1 not in read_map_of_maps[global_dat_id].keys()):
                            # new_edge = DependancyEdge(DFNodeType.DF_START, 0, global_dat_id, node_id, arg.id)
                            self.df_graph.addEdge(self.df_graph.getStartNodeIdx(), 0, dat_ptr, node_id, arg.id)
                            read_map_of_maps[global_dat_id] = {DFNodeType.DF_START:[(node_id, arg.id)]}
                        else:
                            # new_edge = DependancyEdge(DFNodeType.DF_START, len(read_map_of_maps[global_dat_id]), global_dat_id, node_id, arg.id)
                            self.df_graph.addEdge(self.df_graph.getStartNodeIdx(), len(read_map_of_maps[global_dat_id]), dat_ptr, node_id, arg.id)
                            read_map_of_maps[global_dat_id][DFNodeType.DF_START].append((node_id, arg.id))
                            
                        # self.df_graph.edges.append(new_edge)
                        
                        
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
                # new_edge = DependancyEdge(dat_current_update_map[dat_id][0], dat_current_update_map[dat_id][1], dat_id, DFNodeType.DF_END, -1, True)
                # self.df_graph.edges.append(new_edge)
                self.df_graph.addEdge(dat_current_update_map[dat_id][0], dat_current_update_map[dat_id][1], self.dats[dat_id][0].ptr, self.df_graph.getEndNodeIdx(), 0, True)
            elif len(read_map_of_maps[dat_id][dat_current_update_map[dat_id][0]]) == 0:
                # new_edge = DependancyEdge(dat_current_update_map[dat_id][0], dat_current_update_map[dat_id][1], dat_id, DFNodeType.DF_END, -1, True)
                # self.df_graph.edges.append(new_edge)
                self.df_graph.addEdge(dat_current_update_map[dat_id][0], dat_current_update_map[dat_id][1], self.dats[dat_id][0].ptr, self.df_graph.getEndNodeIdx(), 0)
        
        self.read_map_of_maps = read_map_of_maps       

    def gen_global_dat_args(self) -> None:
        source_dats = self.df_graph.getGlobalSourceDatIndices()
        sink_dats = self.df_graph.getGlobalSinkDatIndices()
        
        print (f"source dats: {source_dats}, sink_dats: {sink_dats}")
        for dat_id in source_dats:
            idx = findIdx(sink_dats, lambda x: x == dat_id)
            node = self.df_graph.getFirstReadingNode(self.dats[dat_id][0].ptr)
            
            if node is None:
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
            node = self.df_graph.getFirstWritingNode(self.dats[dat_id][0].ptr)
            
            if node is None:
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

    def gen_PE_args(self) -> None:
        PE_args = []
        for i, v in enumerate(self.itrloop_args):
            if isinstance(v, Loop):
                PE_args.append(self.gen_PE_args_loop(i, v))
        self.PE_args = PE_args
        
    def gen_PE_args_loop(self, i: int,  v: Loop) -> List[str]:
        arg_map = {}
        PE_args = []

        for source_id, sink_id, attr in self.df_graph.getEdges():
            print(attr)
            if i == source_id:
                if sink_id == self.df_graph.getEndNodeIdx():
                    search_list = filter(lambda x: (x.access_type == AccessType.OPS_WRITE or x.access_type == AccessType.OPS_RW) and self.dats[x.dat_id].ptr == attr["dat_str"], self.joint_args)
                    # print(f"search list: {[i for i in search_list]}")
                    if attr['src_arg_id'] in arg_map.keys():
                        arg_map[attr['src_arg_id']].append(f"arg{next(search_list).id}_hls_stream_out")
                    else:
                        arg_map[attr['src_arg_id']] = [f"arg{next(search_list).id}_hls_stream_out"]
                else:
                    connector_name = f"node{source_id}_{attr['src_arg_id']}_to_node{sink_id}_{attr['sink_arg_id']}"
                    if connector_name not in self.interconnector_names:
                        self.interconnector_names.append(connector_name)
                    arg_map[attr['src_arg_id']] = connector_name
            elif i == sink_id:
                if source_id == self.df_graph.getStartNodeIdx():
                    search_list = filter(lambda x: x.access_type in [AccessType.OPS_READ, AccessType.OPS_RW] and self.dats[x.dat_id].ptr == attr["dat_str"], self.joint_args)
                    arg_id = next(search_list).id
                    if attr['sink_arg_id'] in arg_map.keys():
                        arg_map[attr['sink_arg_id']].append(f"arg{arg_id}_hls_stream_in")
                    else:
                        arg_map[attr['sink_arg_id']] = [f"arg{arg_id}_hls_stream_in"]
                        
                else:
                    connector_name = f"node{source_id}_{attr['src_arg_id']}_to_node{sink_id}_{attr['sink_arg_id']}"
                    if connector_name not in self.interconnector_names:
                        self.interconnector_names.append(connector_name)
                    arg_map[attr['sink_arg_id']] = [connector_name]
        
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
        
        for i,dat in enumerate(self.dats):
            outer_loop_str += f"dat{i}: " + str(dat) + "\n"
        
        outer_loop_str += "SOURCE DATS: " + str(self.df_graph.getGlobalSourceDatNames()) + "\n"
        outer_loop_str += "SINK DATS: " + str(self.df_graph.getGlobalSinkDatNames()) + "\n"
        
        outer_loop_str +="\n JOINT_ARGS: \n ------ \n"
        for i,arg in enumerate(self.joint_args):
            outer_loop_str += f"arg{i}: " + str(arg) + "\n"
            
        outer_loop_str +="\n EDGES: \n ------ \n"

        for i, edge in enumerate(self.df_graph.getEdges()):
            outer_loop_str += f"edges{i}: " + str(edge) + "\n"
        
        outer_loop_str +="\n SWAP MAP: \n ------ \n"
            
        for i,j in enumerate(self.dat_swap_map):
            outer_loop_str += f"dat{i} - dat{j}\n"
        
        outer_loop_str +="\n ARGS: \n ------ \n"
        
        dat_arg_i = 0
        for i,arg in enumerate(self.itrloop_args):
            outer_loop_str += f" arg{i}: " + str(arg)
            if (isinstance(arg, Loop)):
                outer_loop_str += f"   PE_args: {self.PE_args[dat_arg_i]} \n\n"
                dat_arg_i += 1
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
        print(self.df_graph)
        self.df_graph.print(self.unique_name, make_dats_node = True)
        
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