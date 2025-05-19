import os
from ops import DataflowGraph_v2, DFNodeType, DataflowNode, Dat, AccessType, ArgDat
from typing import Optional, List, Tuple, Any, Union, Set
from store import Program, Location, Application
import logging
from dataclasses import dataclass, field
from util import KernelProcess, findIdx, function_name, print_rx_graph
from scheme import Scheme
from cpp.parser import ASTtoString, CursorKind, getBinaryOp, Cursor, getAccessorAccessIndices, decend
from copy import deepcopy
# import pygraphviz
import rustworkx as rx
import jaro
@dataclass
class OptError(Exception):
    message: str
    loc: Optional[Location] = None

    def __str__(self) -> str:
        if self.loc:
            logging.error(f"[OPTIMIZER_ERROR] at {self.loc}: {self.message}")
            return f"Optimizer error at {self.loc}: {self.message}"
        else:
            logging.error(f"[OPTIMIZER_ERROR]: {self.message}")
            return f"Optimizer error: {self.message}"


def findCopyPairsInsideKernelCompoundStatement(node: Cursor) -> List[Tuple]: 
    if not node.kind == CursorKind.COMPOUND_STMT:
        raise OptError(f"node should be compound statement")
    
    copy_pairs = []
    
    for child in node.get_children(): 
        if child.kind == CursorKind.BINARY_OPERATOR:
            ops = getBinaryOp(child)
            
            if not ops.spelling == "=":
                continue
            
            operands = [i for i in child.get_children()]

            LHS = getAccessorAccessIndices(operands[0])
            RHS = getAccessorAccessIndices(operands[1])
            
            if not LHS or not RHS:
                continue
            LHS_op,LHS_indices = LHS
            RHS_op,RHS_indices = RHS
            logging.debug(f"{function_name()}: LHS_operand: {LHS_op}, RHS_operand: {RHS_op}, LHS index: {LHS_indices}, RHS index: {RHS_indices}")
            
            if not len(LHS_indices) == len(RHS_indices) or not LHS_indices == RHS_indices:
                continue
            copy_pairs.append((LHS_op, RHS_op))
    return copy_pairs

def ISLUpdateNodeSwapPairs(graph: DataflowGraph_v2) -> None:
    """ This is to update local dat_swap_pairs of each node based on global swap pairs.
    
    Args: 
        graph (DataFlowGraph_v2): Original dataflow graph
        
    Returns: 
        None
    """
    global_dat_swap_map = graph.getGlobalDatsSwapMap()
    
    for node in graph.getAllLoopNodes():  
        for dat in node.loop.dats:
            global_swap_dat_name = global_dat_swap_map[dat.ptr]
            
            if global_swap_dat_name == dat.ptr:
                node.internal_dat_swap_map[dat.ptr] = dat.ptr
            else:
                local_dat_idx = findIdx(node.loop.dats, lambda dat: dat.ptr == global_swap_dat_name)
                if local_dat_idx is None:
                    node.internal_dat_swap_map[dat.ptr] = dat.ptr
                else:
                    node.internal_dat_swap_map[dat.ptr] = global_swap_dat_name

    logging.debug(f"Internal dat swap updated graph: \n {graph}")
    
def ISLCopyDetection(original_graph: DataflowGraph_v2, prog: Program, app: Application, scheme: Scheme) -> DataflowGraph_v2:
    
    """ This dataflow analysis check ops_par_loop nodes writing output to DF_END node and check weather there are any copy kernels that can be identified and
    eliminated to avoid synthesis of hardware for data copy 

    Args:
        original_graph (DataFlowGraph_v2): Original dataflow graph
        prog (Program): OPS program
        app (Application): OPS app
        scheme (Scheme): The OPS translation scheme calling the optimization

    Returns:
        DataFlowGraph: Adjusted dataflow graph without copy kernel nodes and redundant edges removed
    """
    
    kernel_processor = KernelProcess()
    
    copy_graph = original_graph.copy()

    checked_node_ids = []
    
    for src_id, sink_id, edge_attr in copy_graph.getEdges():
        if not sink_id == copy_graph.getEndNodeIdx():
            continue
        
        logging.debug(edge_attr)
        #get the kernel_entity
        
        if src_id in checked_node_ids:
            # copy_graph.edges.remove(edge)
            continue
        
        node = copy_graph.getNode(src_id)
            
        if node is None: 
            raise OptError(f"Failed to access node information of node_id: {edge.source_id} from {copy_graph.unique_name}")
        
        kernel_entities = prog.findEntities(node.loop.kernel)
        
        if len(kernel_entities) == 0:
            raise OptError(f"Failed to get AST entity of the kernel {node.loop.kernel} from program: {id(prog)}")
        
        logging.debug(f"kernel entity: {kernel_entities[0].ast.spelling}, {kernel_entities[0].ast.extent}, {id(kernel_entities[0].ast)}")
        lines = ASTtoString(kernel_entities[0].ast)
            # logging.debug(f"AST dump child of kernel kind: {child.kind}, type: {child.type}, val: {child.spelling}")
            
        logging.debug(f"AST dump of the body of kernel declaration of kernel {node.loop.kernel}")
        for line in lines:
            logging.debug(line)
            
        kernel_func = scheme.translateKernel(node.loop, prog, app, 1)
        logging.debug(f"translated kernel function: {kernel_func}")
        kernel_func = kernel_processor.clean_kernel_func_text(kernel_func)
        logging.debug(f"kernel function after cleaning: {kernel_func}")
        kernel_body, kernel_args = kernel_processor.get_kernel_body_and_arg_list(kernel_func)
        logging.debug(f"kernel arguments: {kernel_args}, kernel body: {kernel_body}")
        logging.debug(f"ops_par_loop arguments: {node.loop.args}")
        
        if not len(kernel_args) == len(node.loop.args):
            raise OptError("Critical error, number of kernel arg should match with ops_par_loop args")
        
        kernel_children = [child for child in kernel_entities[0].ast.get_children()]
        copy_pairs = findCopyPairsInsideKernelCompoundStatement(kernel_children[-1])
        logging.debug(f"copy pairs found {copy_pairs}")
        
        unique_copy_pairs = []
        for pair in copy_pairs:
            if pair not in unique_copy_pairs:
                unique_copy_pairs.append(pair)
                
        if not len(unique_copy_pairs) == len(kernel_args) / 2:
            logging.warning(f"kernel {node.loop.kernel} cannot be a copy kernel")
        else:
            for pair in copy_pairs:
                lhs_idx = kernel_args.index(pair[0])
                rhs_idx = kernel_args.index(pair[1])
                logging.debug(f"lhs_idx: {lhs_idx}, rhs_idx:{rhs_idx}")
                lhs_dat_name = node.loop.get_dat_name(lhs_idx)
                rhs_dat_name = node.loop.get_dat_name(rhs_idx)
                logging.debug(f"lhs_dat_name: {lhs_dat_name}, rhs_dat_name: {rhs_dat_name}")
                # lhs_global_dat_idx = findIdx(copy_graph.global_dats, lambda dat: dat.ptr == lhs_dat_name)
                # rhs_global_dat_idx = findIdx(copy_graph.global_dats, lambda dat: dat.ptr == rhs_dat_name)
                # logging.debug(f"lhs_global_dat_idx: {lhs_global_dat_idx}, rhs_global_dat_idx: {rhs_global_dat_idx}")
                # copy_graph.global_dat_swap_map[lhs_global_dat_idx] = rhs_global_dat_idx
                # copy_graph.global_dat_swap_map[rhs_global_dat_idx] = lhs_global_dat_idx 
                copy_graph.addDatSwapUpdate(lhs_dat_name, rhs_dat_name)
            checked_node_ids.append(src_id)
            # copy_graph.edges.remove(edge)
    
    logging.debug(f"Nodes need to be remove: {checked_node_ids}")
    
    for node_id in checked_node_ids:
        in_edges = copy_graph.getInEdgesFromNode(node_id)
        for src_id, sink_id, attr in in_edges:
            copy_graph.addEdge(src_id, attr["src_arg_id"], attr["dat_str"], copy_graph.getEndNodeIdx(), 0)
        copy_graph.deleteNode(node_id)
    
    # for node in remove_nodes:
    #     copy_graph.nodes.remove(node)
        
    # remove_edges = []
    # for edge in copy_graph.edges:
    #     if edge.sink_id in checked_node_ids:
    #         remove_edges.append(edge)
    
    # for edge in remove_edges:
    #     edge.sink_id = DFNodeType.DF_END
    #     # copy_graph.edges.remove(edge)
    
    logging.debug(f"global swap map is ISL COpy detect: {copy_graph.getGlobalDatsSwapMap()}")
    ISLUpdateNodeSwapPairs(copy_graph)
    copy_graph.print("after_copy_detection", make_dats_node=True, attr={'show_arg_id': False})
    return copy_graph

def ISLReadBufferPropagation(original_graph: DataflowGraph_v2, prog: Program, app: Application, scheme: Scheme) -> DataflowGraph_v2:
    
    #1. Read Only buffer propagation as they require stray connection
    read_only_dats = []
    copy_graph = original_graph.copy(original_graph.unique_name + "_buff_prop")
    #initializing propagation path from start node
    propagation_paths = {}
    
    for dat in copy_graph.getGlobalDats():
        edges = [(src_id, sink_id, attr) for src_id, sink_id, attr in copy_graph.getEdges() if attr["dat_str"] == dat.ptr]

        for src_id, sink_id, attr in edges:
            if src_id == copy_graph.getStartNodeIdx():
                propagation_paths[dat.ptr] = [(copy_graph.getStartNodeIdx(), 0)] #(node, node_arg_id)
            
        is_read_only_dat = True
        
        for (src_id, sink_id, attr) in edges:
            if not src_id == copy_graph.getStartNodeIdx():
                is_read_only_dat = False
                break
        
        if not copy_graph.getGlobalDatsSwapMap()[attr["dat_str"]] == attr["dat_str"]:
            is_read_only_dat = False
            
        if is_read_only_dat:
            read_only_dats.append(dat)
    
    for dat in read_only_dats:
        if not copy_graph.getGlobalDatsSwapMap()[dat.ptr] == dat.ptr:
            read_only_dats.remove(dat)

    logging.debug(f"{function_name()}: read_only_dats: {read_only_dats}")
    
    read_only_dat_names = []
    for dat in read_only_dats:
        read_only_dat_names.append(dat.ptr)
      
    #topological sort
    sorted_nodes = rx.topological_sort(copy_graph.getRXGraph())
    logging.debug(f"{function_name()}: topological sorted nodes: {sorted_nodes}")
    
    for node_uid in sorted_nodes:
        if node_uid == copy_graph.getStartNodeIdx(): #skip start node as propogation paths are initialized from start node earlier
            continue
        in_edges = copy_graph.getInEdgesFromNode(node_uid)
        
        for src_id, sink_id, attr in in_edges:
            if not attr["dat_str"] in read_only_dat_names:
                continue
            if not propagation_paths[attr["dat_str"]][0] == src_id: #remove edge and connect with previous
                copy_graph.addEdge(propagation_paths[attr["dat_str"]][-1][0], propagation_paths[attr["dat_str"]][-1][1], attr["dat_str"], sink_id, attr["sink_arg_id"])
                copy_graph.deleteEdge(src_id, sink_id, attr)
                
            propagation_paths[attr["dat_str"]].append((node_uid, attr["sink_arg_id"]))
    
    #final sweep to create edges to sink nodes
    for key in propagation_paths.keys():
        if key in read_only_dat_names:
            copy_graph.addEdge(propagation_paths[key][-1][0], propagation_paths[key][-1][1], key, copy_graph.getEndNodeIdx(), 0, True)

    # phase 2
    
    for node_uid in sorted_nodes:
        #checking each read_only nodes
        df_node = copy_graph.getNode(node_uid)
        if not isinstance(df_node, DataflowNode):
            continue
        
        for arg in df_node.loop.args:
            if not isinstance(arg, ArgDat):
                continue
            dat_name = df_node.loop.dats[arg.dat_id].ptr
            
            if dat_name in read_only_dat_names:
                continue
            
            if not (isinstance(arg, ArgDat) and (arg.access_type == AccessType.OPS_READ or arg.access_type == AccessType.OPS_RW)):
                continue
            
            if not dat_name in propagation_paths.keys():
                logging.debug(f"not in {dat_name}, node: {df_node.node_uid}|{df_node.loop.kernel}")
                continue
            elif not propagation_paths[dat_name][-1] == copy_graph.getStartNodeIdx():
                logging.debug(f"removing in {dat_name}, node: {df_node.node_uid}|{df_node.loop.kernel}")
                # remove edge connecting dat to current node
                filtered_edges = [(src_id, sink_id, attr) for src_id, sink_id, attr in copy_graph.getInEdgesFromNode(node_uid) if (attr["dat_str"] == dat_name and sink_id == node_uid)]
                
                if not filtered_edges or not len(filtered_edges) == 1:
                    raise OptError(f"{function_name()}: couldn't find edge read dat: {dat_name} from node: {df_node.node_uid}|{df_node.loop.kernel}")
                
                src_id,sink_id, attr = filtered_edges[0]
                
                if not src_id == copy_graph.getStartNodeIdx():
                    logging.debug(f"src id is not start {src_id}, node: {df_node.node_uid}|{df_node.loop.kernel}")
                    continue
                 
                copy_graph.addEdge(propagation_paths[dat_name][-1][0], propagation_paths[dat_name][-1][1], dat_name, node_uid, arg.id)
                copy_graph.deleteEdge(src_id, sink_id, attr)
                propagation_paths[dat_name].append((node_uid, arg.id)) 
                
            else:
                logging.debug(f"first read {dat_name}, node: {df_node.node_uid}|{df_node.loop.kernel}")
                propagation_paths[dat_name].append((node_uid, arg.id)) 
            
                
         
    # # creating propagation path for hierarchical nodes via single read from start
    # for x in range(len(sorted_nodes)-1):
    #     for y in range(x+1, len(sorted_nodes)):
    #         if (sorted_nodes[x] == copy_graph.getStartNodeIdx() or sorted_nodes[y] == copy_graph.getEndNodeIdx()):
    #             break
            
    #         first_node = copy_graph.getNode(sorted_nodes[x])
    #         second_node = copy_graph.getNode(sorted_nodes[y])
            
    #         #TODO if first and second node
            
            
    logging.debug(f"{function_name()}: propagation paths: {propagation_paths}")
    logging.debug(f"{copy_graph}")
    copy_graph.print("after_buffer_propagation", make_dats_node=True, attr={'show_arg_id': False})
    return copy_graph
    
@dataclass
class BasicDataDepNode:
    dat_ptr: str
    df_node: Union[DataflowNode, DFNodeType]
    head_dat_ptr: str = None
    is_assigned: bool = False
    
    def __str__(self) -> str:
        if self.df_node.type == DFNodeType.DF_START:
            node_name_suffix = "start"
        elif self.df_node.type == DFNodeType.DF_END:
            node_name_suffix = "end"
        else:
            node_name_suffix = self.df_node.loop.kernel
            
        return (f"{self.dat_ptr}:{self.df_node.node_uid}_{node_name_suffix}")
    
# @dataclass
# class DataDepNode(BasicDataDepNode):
#     children: Optional[List[Any]] = field(default_factory=list)
    
#     def add_child(self, child: Any) -> None:
#         self.children.append(child)
    
#     def get_children(self) -> List[Any]:
#         return self.children
    
#     def remove_child(self, child: Any) -> None:
#         if child in self.children:
#             self.children.remove(child)
            
#     def __eq__(self, other) -> bool:
#         if isinstance(other, DataDepNode):
#             return (self.df_node == other.df_node and self.dat_ptr == other.dat_ptr)
#         return False
#     def add_node_to_graph(self, graph: pygraphviz.AGraph) -> None:
#         for child in self.children:
#             if child.df_node.node_uid == 0:
#                 graph.add_edge(f"{child.dat_ptr}:START", f"{self.dat_ptr}:{self.df_node.loop.kernel}")
#             else:
#                 graph.add_edge(f"{child.dat_ptr}:{child.df_node.loop.kernel}", f"{self.dat_ptr}:{self.df_node.loop.kernel}")
#         for child in self.children:
#             child.add_node_to_graph(graph)
    
#     def print_tree(self) -> str:
#         logging.debug(f"Generating dependency graph for {self.dat_ptr}:{self.df_node.loop.kernel}")
#         g = pygraphviz.AGraph(strict=True, directed=True)
#         self.add_node_to_graph(g)
        
#         g.layout(prog="dot")
#         g.draw(f"{self.dat_ptr}_data_dependency.png")
    
#     # def search
#     def str_dep_tree(self) -> str:
#         out_str = f"{self.dat_ptr}"
#         if (self.children):
#             out_str += " <- "
#         for child in self.children:
#             out_str += f"({child.str_dep_tree()}), "
#         return out_str
    
#     def __str__(self) -> str:
#         out_str = f"loop kernel: {self.df_node.loop.kernel}, dep tree: {self.str_dep_tree()}"
#         return out_str
def jaro_weight_distance(first: str, second: str) -> float:
    return (1 - jaro.jaro_winkler_metric(first, second))
    
def findVariables(astnode: Cursor, variables: Set) -> None:
    if astnode.kind == CursorKind.DECL_REF_EXPR:
        variables.add(astnode.spelling)
    
    for child in astnode.get_children():
        findVariables(child, variables)

def getDependencyVariables(astnode: Cursor, dependant: str, visited: Set[str] = None, original_compound_ast_node: Cursor = None, curr_compound_idx: int = -1) -> Optional[Set[str]]:
    if visited is None:
        visited = set()
    
    # print(f"astnode: {astnode.spelling}, kind: {astnode.kind}, {ASTtoString(decend(astnode))}")
    if original_compound_ast_node is None:
        if astnode.kind is CursorKind.IF_STMT:
            original_compound_ast_node = decend(astnode)
        elif astnode.kind is CursorKind.COMPOUND_STMT:
            original_compound_ast_node = astnode
        else:
            raise OptError("Original first node should be a compound statement")
        
    
    # exit condition 1    
    if dependant in visited:
        return set()
    
    dependencies = set()
    dependant_found = False
    is_self_dependency = False
    local_vars = set()
    
    children = list(original_compound_ast_node.get_children())
    
    # traversing reverse to find dependant
    # find dependent
    for child in children[curr_compound_idx::-1]:
        variables_found = set()
        if child.kind is CursorKind.DECL_STMT:
            var_name = decend(child).spelling
            local_vars.add(var_name)
            
            if not var_name == dependant:
                continue
            dependant_found = True
            
            visited.add(var_name) #since it is declared here no more dependency check is need 
            findVariables(decend(decend(child)), variables_found)
        
        elif child.kind == CursorKind.BINARY_OPERATOR:
            operator = getBinaryOp(child).spelling
            if not operator == "=":
                continue
            operands = list(child.get_children())
        
            if not len(operands) == 2:
                raise OptError("Binary operator or compound assignment should have RHS and LHS")
            lhs_operand = operands[0]
            rhs_operand = operands[1]
            
            if lhs_operand.kind == CursorKind.UNEXPOSED_EXPR:
                lhs_operand = decend(lhs_operand)
            if not lhs_operand.spelling == dependant:
                continue
            dependant_found = True
            visited.add(dependant)
            findVariables(rhs_operand, variables_found)
            
        elif child.kind == CursorKind.COMPOUND_ASSIGNMENT_OPERATOR:
            operands = list(child.get_children())
        
            if not len(operands) == 2:
                raise OptError("Binary operator or compound assignment should have RHS and LHS")
            lhs_operand = operands[0]
            rhs_operand = operands[1]
            
            if lhs_operand.kind == CursorKind.UNEXPOSED_EXPR:
                lhs_operand = decend(lhs_operand)
            if not lhs_operand.spelling == dependant:
                continue
            dependant_found = True
            visited.add(dependant)
            dependencies.add(dependant) #self dependency
            findVariables(rhs_operand, variables_found)
            
        elif child.kind == CursorKind.FOR_STMT or child.kind == CursorKind.IF_STMT:
            dependencies.update(getDependencyVariables(list(child.get_children())[-1], dependant))
            
        if variables_found:
            idx_cur_child_in_compound_children = children.index(child)
            dependencies.update(variables_found)
            
            for var in variables_found:
                dependencies.update(getDependencyVariables(child, var, visited, original_compound_ast_node, idx_cur_child_in_compound_children))
    
    for var in local_vars:
        if not var in dependencies:
            continue
        dependencies.remove(var)
        
    return dependencies    

def ISLDataDependencyCyclesDetection(original_graph: DataflowGraph_v2, prog: Program, app: Application, scheme: Scheme) -> DataflowGraph_v2:
    """This Dataflow analysis read an original dataflow graph and detects buffer cycles which is crucial to maintain the halo flow in FPGAs

    Args:
        original_graph (DataFlowGraph): Input dataflow graph
        prog (Program): OPS program
        app (Application):OPS application
        scheme (Scheme): The OPS translation scheme calling the optimization

    Returns:
        DataFlowGraph: Output dataflow graph with each ops_par_loop node will have internal swap map adjusted with the data dependency cycles detected to 
        maintain halo flow.
    """
    kernel_processor = KernelProcess()
    
    copy_graph = original_graph.copy(original_graph.unique_name + "_dat_cycle_detected")
    
    logging.debug(f"df_graph before ISLDataDepCyclesDet \n {copy_graph}")
    
    dependency_graph = rx.PyDiGraph(multigraph=False)
    check_queue_uids = []
    cycle_heads_uids = []
    checked_uids = []
    
    for src_id, sink_id, attr in copy_graph.getEdges():
        if sink_id == copy_graph.getEndNodeIdx() and not attr["isStray"]:
            dep_node = BasicDataDepNode(attr["dat_str"], copy_graph.getNode(src_id))
            dep_node_uid = dependency_graph.add_node(dep_node)
            cycle_heads_uids.append(dep_node_uid)
            check_queue_uids.append(dep_node_uid)
    init_dats = []
    for dep_node_uid in check_queue_uids:        
        init_dats.append(dependency_graph[dep_node_uid].dat_ptr)#(dep_node.dat_ptr)
        
    logging.debug(f"Initial dats being checked for dataDependency: {init_dats}")
            
    while check_queue_uids:
        dep_node_uid = check_queue_uids.pop(0)
        depNode = dependency_graph[dep_node_uid]
        if depNode.df_node.node_uid == copy_graph.getStartNodeIdx():
            continue
        cur_dat_name = depNode.dat_ptr
        logging.debug(f"{function_name()}: Checking dat: {cur_dat_name}")
        arg_dats = depNode.df_node.loop.get_arg_dat(cur_dat_name, [AccessType.OPS_WRITE, AccessType.OPS_RW])
        if not len(arg_dats) == 1:
            if len(arg_dats) == 0:
                # continue
                raise OptError(f"Failed to find arg_dat of {depNode.dat_ptr} in loop: {depNode.df_node.loop.kernel} ({depNode.df_node.loop})")
            raise OptError(f"One ops_par_loop can only write a dat once. Critical error in {depNode.df_node.loop}")
        kernel_func = scheme.translateKernel(depNode.df_node.loop, prog, app, 1)
        # logging.debug(f"translated kernel function: {kernel_func}")
        kernel_func = kernel_processor.clean_kernel_func_text(kernel_func)
        # logging.debug(f"kernel function after cleaning: {kernel_func}")
        kernel_body, kernel_args = kernel_processor.get_kernel_body_and_arg_list(kernel_func)
        # logging.debug(f"kernel arguments: {kernel_args}, kernel body: {kernel_body}")
        # logging.debug(f"ops_par_loop arugments: {depNode.df_node.loop.args}")
        relevant_kernel_arg = kernel_args[depNode.df_node.loop.args.index(arg_dats[0])]
        # logging.debug(f"arg_dats: {arg_dats}, cur_dat_name: {cur_dat_name}")
        logging.debug(f"relevant_kernel_parameter to be checked for dependency: {relevant_kernel_arg} for argument {depNode.df_node.loop.dats[arg_dats[0].id].ptr}")
        kernel_entities = prog.findEntities(depNode.df_node.loop.kernel)
        kernel_children = [child for child in kernel_entities[0].ast.get_children()]
        # logging.debug(f"kernel entity: {kernel_entities[0].ast.spelling}, {kernel_entities[0].ast.extent}, {id(kernel_entities[0].ast)}")
        # lines = ASTtoString(kernel_entities[0].ast)
        # logging.debug(f"AST dump of the body of kernel declaration of kernel {depNode.df_node.loop.kernel}")
        # for line in lines:
        #     logging.debug(line)
        
        supporting_vars = getDependencyVariables(kernel_children[-1], relevant_kernel_arg)
            
        logging.debug(f"supporting vars: {supporting_vars}")
        
        for var in supporting_vars:
            if var not in kernel_args:
                continue
            arg_idx = kernel_args.index(var)
            loop_arg = depNode.df_node.loop.args[arg_idx]
            
            if not isinstance(loop_arg, ArgDat):
                continue
            
            dat_name = depNode.df_node.loop.dats[loop_arg.dat_id].ptr
            logging.debug(f"found dat in supporting vars: {dat_name}")
            global_dat_id = findIdx(copy_graph.getGlobalDats(), lambda dat: dat.ptr == dat_name)
            logging.debug(f"found dat_id: {global_dat_id}")
            searched_edges = copy_graph.getInEdgesFromNode(depNode.df_node.node_uid, dat_name)   #getEdge(depNode.df_node.node_id, global_dat_id, 1)
            logging.debug(f"found searched_edges: {searched_edges}")
            if not len(searched_edges) == 1:
                logging.warning(f"Couldn't find edge connecting to sink node {depNode.df_node.loop.kernel} with dat: {dat_name}({global_dat_id})")
                continue
            # if searched_edges[0][0] == copy_graph.getStartNodeIdx():
            #     newDepNode = BasicDataDepNode(dat_name, copy_graph.getNode(copy_graph.getStartNodeIdx()))
            #     newDepNode_uid = dependency_graph.add_node(newDepNode)
            # else:
            src_node = copy_graph.getNode(searched_edges[0][0])
            provider_dep_node_id = findIdx(dependency_graph.nodes(), lambda dep_node: dep_node.df_node.node_uid == src_node.node_uid and dep_node.dat_ptr == dat_name)
            
            if provider_dep_node_id is None:
                provider_dep_node = BasicDataDepNode(dat_name, src_node)
                provider_dep_node_id = dependency_graph.add_node(provider_dep_node)
            
            if (not provider_dep_node_id in checked_uids) and (not provider_dep_node_id in check_queue_uids):
                logging.debug(f"Adding new dat to be checked : {dependency_graph[provider_dep_node_id].dat_ptr}")
                check_queue_uids.append(provider_dep_node_id)
                
            dependency_graph.add_edge(provider_dep_node_id, dep_node_uid, {"weight" : 1, "src_dat_name" : provider_dep_node.dat_ptr, "sink_dat_name" : dep_node.dat_ptr})
        
        # Add a dep node if RW arg
        # searched_edges = copy_graph.getInEdgesFromNode(depNode.df_node.node_uid, dat_name)
        
        # provider_dep_node_id = findIdx(dependency_graph.nodes(), lambda dep_node: dep_node.df_node.node_uid == src_node.node_uid and dep_node.dat_ptr == dat_name)
        logging.debug(f"arg_dats_searched: {arg_dats[0]}")
        if arg_dats[0].access_type == AccessType.OPS_RW:
            logging.debug("degfsregb")
            searched_edges = copy_graph.getInEdgesFromNode(depNode.df_node.node_uid, cur_dat_name)
            if not len(searched_edges) == 1:
                logging.error(f"Couldn't find edge connecting to sink node {depNode.df_node.loop.kernel} with dat: {cur_dat_name}({global_dat_id})")
            src_node = copy_graph.getNode(searched_edges[0][0])
            provider_dep_node_id = findIdx(dependency_graph.nodes(), lambda dep_node: dep_node.df_node.node_uid == src_node.node_uid and dep_node.dat_ptr == cur_dat_name)
            
            if provider_dep_node_id is None:
                provider_dep_node = BasicDataDepNode(cur_dat_name, src_node)
                provider_dep_node_id = dependency_graph.add_node(provider_dep_node)
            
            if (not provider_dep_node_id in checked_uids) and (not provider_dep_node_id in check_queue_uids):
                logging.debug(f"Adding new dat to be checked : {dependency_graph[provider_dep_node_id].dat_ptr}")
                check_queue_uids.append(provider_dep_node_id)
            
            dependency_graph.add_edge(provider_dep_node_id, dep_node_uid, {"weight" : 1, "src_dat_name" : provider_dep_node.dat_ptr, "sink_dat_name" : dep_node.dat_ptr})

        checked_uids.append(dep_node_uid)
        
    def node_attr(node):
        if node.df_node.node_uid == copy_graph.getStartNodeIdx():
            node_name_suffix = "start"
        elif node.df_node.node_uid == copy_graph.getEndNodeIdx():
            node_name_suffix = "end"
        else:
            node_name_suffix = node.df_node.loop.kernel
            
        return {"label" : f"{node.dat_ptr}:{node.df_node.node_uid}_{node_name_suffix}"}
    
    print_rx_graph(f"{copy_graph.unique_name}", dependency_graph, node_attr=node_attr)
    
    # Phase 2: Explore shortest paths.
    # The cycle heads, the dats writing to end should have a swap pair dat (sometimes it can be same dat) that read from start
    # if the swap pair dat is read from start it is an error in the ISL definition from user side.
    
    logging.debug(f"head dep nodes: {[str(dependency_graph[node_id]) for node_id in cycle_heads_uids]}")
    
    for head_dep_node_id in cycle_heads_uids:
        head_dep_node = dependency_graph[head_dep_node_id]
        swap_dep_node_id  = findIdx(dependency_graph.nodes(), lambda dep_node: dep_node.dat_ptr == copy_graph.getGlobalDatsSwapMap()[head_dep_node.dat_ptr] \
            and dep_node.df_node.node_uid == copy_graph.getStartNodeIdx())
        
        if swap_dep_node_id is None:
            raise OptError(f"Error in the generated dependency graph: the writing dat {head_dep_node.dat_ptr}," \
                f"swap pair dat {copy_graph.getGlobalDatsSwapMap()[head_dep_node.dat_ptr]} doesn't have a read from start. Fix the ISL region definition")
        swap_dep_node = dependency_graph[swap_dep_node_id]
        
        def dijkstra_distance_function(edge_attr):
            # distance = jaro_weight_distance(head_dep_node.dat_ptr, edge_attr["sink_dat_name"]) + jaro_weight_distance(edge_attr["src_dat_name"], edge_attr["sink_dat_name"])
            # logging.debug(f"jaro weight call: {edge_attr}, distance: {jaro_distance}")
            distance = edge_attr["weight"]
            return distance
    
        all_shortest_paths = rx.digraph_all_shortest_paths(dependency_graph, swap_dep_node_id, head_dep_node_id)
        all_shortest_paths_by_dats = ""
        for path in all_shortest_paths:
            all_shortest_paths_by_dats += "["
            for i, uid in enumerate(path):
                all_shortest_paths_by_dats += dependency_graph[uid].dat_ptr
                if i < len(path) - 1:
                    all_shortest_paths_by_dats += ","
            all_shortest_paths_by_dats += "]\n"
        logging.debug(f"all shortest paths: {all_shortest_paths_by_dats}")
        
        plausible_shortest_paths = find_plausible_shortest_paths(dependency_graph, swap_dep_node_id, head_dep_node_id, all_shortest_paths)
        
        if not plausible_shortest_paths:
            raise OptError(f"{function_name()}: Cannot find any plausible path from {head_dep_node.dat_ptr} to {swap_dep_node.dat_ptr}")
        
        
        # logging.debug(f"shortest paths from {swap_dep_node}|{swap_dep_node_id} to {head_dep_node}|{head_dep_node_id}: {[str(dependency_graph[node_id]) for node_id in all_shortest_paths[head_dep_node_id]]}")
        
        # removing path internal connection and update dataflow graph internal swap map. This will allow each source to sink path in ISL region is disjoint
        removeInternalDependencyPathEdgesAndUpdateSwap(copy_graph, dependency_graph,  head_dep_node_id, swap_dep_node_id, plausible_shortest_paths[0][1])
        # print_rx_graph(f"{copy_graph.unique_name}_{head_dep_node_id}", dependency_graph, node_attr=node_attr)
        
        #updating internal swap map
    logging.debug(f"df_graph after ISLDataDepCyclesDet \n {copy_graph}")
    copy_graph.print("after_buffer_DataDepCycleDet", make_dats_node=True, attr={'show_arg_id':False})
    return copy_graph
    # for head in cycle_heads:
    #     logging.debug(f"dependency train: {head}")
    #     head.print_tree()
    #     dependancy_kernel_arguments = findDependancyArguments(kernel_children[-1], relevent_kernel_args[0])

def find_plausible_shortest_paths(dep_graph: rx.PyDiGraph, source_id: int, sink_id: int, paths: List[List[int]]) -> List[List[Any]]:
    
    plausible_paths = []
    
    for path in paths:     
        updated_kernel_args_map = {} # To store kernel arg_id with updating node according to current path
        
        prev_node_id = path[0]
        valid_path = True
        weighted_path_length = 0
        
        for i in range(1,len(path)):
            curr_node_id = path[i]
            curr_node = dep_graph[curr_node_id]
            prev_node = dep_graph[prev_node_id]
            curr_dat_name = curr_node.dat_ptr
            prev_dat_name = prev_node.dat_ptr
            curr_kernel_name = curr_node.df_node.loop.kernel
            
            '''
            Finding the reading ArgDat of current_kernel ops_par_loop args. 
            If previously stored arg_id is not same as this then this path can be flagged invalid.
            '''
            read_arg_dats = curr_node.df_node.loop.get_arg_dat(prev_dat_name, [AccessType.OPS_READ, AccessType.OPS_RW]) 
            
            if len(read_arg_dats) == 0:
                raise OptError(f"{function_name()} - Unable to find READ or RW arg_dat in {curr_node.df_node.loop.kernel} of dat: {prev_dat_name}")
            if not len(read_arg_dats) == 1:
                raise OptError(f"{function_name()} -  READ or RW arg_dats in {curr_node.df_node.loop.kernel} of dat: {prev_dat_name} has to be exactly one")

            write_arg_dats = curr_node.df_node.loop.get_arg_dat(curr_dat_name, [AccessType.OPS_WRITE, AccessType.OPS_RW])
            
            if len(write_arg_dats) == 0:
                raise OptError(f"{function_name()} - Unable to find WRITE or RW arg_dat in {curr_node.df_node.loop.kernel} of dat: {curr_dat_name}")
            if not len(write_arg_dats) == 1:
                raise OptError(f"{function_name()} -  WRITE or RW arg_dats in {curr_node.df_node.loop.kernel} of dat: {curr_dat_name} has to be exactly one")
            
            if curr_kernel_name in updated_kernel_args_map.keys():
                idx = findIdx(updated_kernel_args_map[curr_kernel_name], lambda arg_pairs: arg_pairs[1] == write_arg_dats[0].id)
                
                if not idx is None:
                    #If already updated, then check the update dep_nodes are with the same source argument id as it was used before
                    prev_updated_args_pair = updated_kernel_args_map[curr_kernel_name][idx]
                    if not prev_updated_args_pair[0] == read_arg_dats[0]:
                        valid_path = False
                else:
                    updated_kernel_args_map[curr_kernel_name].append([read_arg_dats[0].id, write_arg_dats[0].id])
            else:
                updated_kernel_args_map[curr_kernel_name] = [[read_arg_dats[0].id, write_arg_dats[0].id]]
            
            weighted_path_length += jaro_weight_distance(dep_graph[source_id].dat_ptr, curr_dat_name) + jaro_weight_distance(prev_dat_name, curr_dat_name)
            prev_node_id = curr_node_id
            
        if valid_path:
            plausible_paths.append([weighted_path_length, path])
    
    plausible_paths.sort(key=lambda val: val[0])
          
    logging.debug(f"Plausible paths for {dep_graph[source_id]} -> {dep_graph[sink_id]}")
    for i in range(len(plausible_paths)):
        if not  i == len(plausible_paths) - 1:
            logging.debug(f"  |- weight: { plausible_paths[i][0]}, path: {[str(dep_graph[k]) for k in plausible_paths[i][1]]}")
        else:
            logging.debug(f"  `- weight: { plausible_paths[i][0]}, path: {[str(dep_graph[k]) for k in plausible_paths[i][1]]}")

    return plausible_paths

def removeInternalDependencyPathEdgesAndUpdateSwap(df_graph: DataflowGraph_v2, dep_graph: rx.PyDiGraph, src_dep_node_id: int, sink_dep_node_id: int, path: List[int]) -> None:
    """_summary_

    Args:
        df_graph (DataflowGraph_v2): _description_
        dep_graph (rx.PyDiGraph): _description_
        src_dep_node_id (int): _description_
        sink_dep_node_id (int): _description_
        path_map (rx.PathMapping): _description_
        updated_kernel_args_map (_type_, optional): Map to hold the previous updating arg_id, dep_node pair for a particular kernel argument. 
            Defaults to map[str, List[Any]].

    """    
    prev_node_id = path[0]
    logging.debug(f"first dep node: {dep_graph[prev_node_id]}")
    
    for i in range(1,len(path)):
        curr_node_id = path[i]
        
        if not dep_graph.has_edge(prev_node_id, curr_node_id):
            raise OptError(f"The edge in-between node: {dep_graph[prev_node_id]}, node: {dep_graph[curr_node_id]}")

        # check updated kernel arg map whether the sink dat arg was updated in the context of the kernel
        already_updated = False

        curr_node = dep_graph[curr_node_id]
        prev_node = dep_graph[prev_node_id]
        curr_dat_name = curr_node.dat_ptr

        curr_kernel_name = curr_node.df_node.loop.kernel
        
        arg_dats = curr_node.df_node.loop.get_arg_dat(curr_dat_name, [AccessType.OPS_WRITE, AccessType.OPS_RW])
        
        if len(arg_dats) == 0:
            raise OptError(f"{function_name()} - Unable to find WRITE or RW arg_dat in {curr_node.df_node.loop.kernel} of dat: {curr_dat_name}")
        if not len(arg_dats) == 1:
            raise OptError(f"{function_name()} -  WRITE or RW arg_dats in {curr_node.df_node.loop.kernel} of dat: {curr_dat_name} has to be exactly one")
        
        
        dep_graph.remove_edge(prev_node_id, curr_node_id)
        
        if not curr_node_id == sink_dep_node_id:
            dep_graph[curr_node_id].is_assigned = True
            
            for in_n, out_n, attr in dep_graph.in_edges(curr_node_id):
                dep_graph.remove_edge(in_n, out_n)
        
            #updating internal swap map
            curr_node.df_node.internal_dat_swap_map[curr_dat_name] = prev_node.dat_ptr
            curr_node.df_node.internal_dat_swap_map[prev_node.dat_ptr] = curr_dat_name
        prev_node_id = curr_node_id
    