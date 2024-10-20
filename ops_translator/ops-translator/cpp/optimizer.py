import os
from ops import DataflowGraph_v2, DFNodeType, DataflowNode, Dat, AccessType, ArgDat
from typing import Optional, List, Tuple, Any, Union, Set
from store import Program, Location, Application
import logging
from dataclasses import dataclass, field
from util import KernelProcess, findIdx
from scheme import Scheme
from cpp.parser import ASTtoString, CursorKind, getBinaryOp, Cursor, getAccessorAccessIndices, decend
from copy import deepcopy
import pygraphviz
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
            
            print (f"child kind: {child.kind}, child spelling: {child.spelling}, op: {ops.spelling}")
            operands = [i for i in child.get_children()]
            print(f"operands: {operands}")
            LHS = getAccessorAccessIndices(operands[0])
            RHS = getAccessorAccessIndices(operands[1])
            
            if not LHS or not RHS:
                continue
            LHS_op,LHS_indices = LHS
            RHS_op,RHS_indices = RHS
            
            if not len(LHS_indices) == len(RHS_indices) or not LHS_indices == RHS_indices:
                continue
            copy_pairs.append((LHS_op, RHS_op))
    return copy_pairs
        
def ISLCopyDetection(original_graph: DataflowGraph_v2, prog: Program, app: Application, scheme: Scheme) -> DataflowGraph_v2:
    
    """ This dataflow analysis check ops_par_loop nodes writing output to DF_END node and check weather there are any copy kernels that can be identified and
    eliminated to avoid synthesis of hardware for data copy 

    Args:
        original_graph (DataFlowGraph): Original dataflow graph
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
        
        if not len(copy_pairs) == len(kernel_args) / 2:
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
    copy_graph.print("after_copy_detection", make_dats_node=True)
    return copy_graph


@dataclass
class BasicDataDepNode:
    dat_ptr: str
    df_node: Union[DataflowNode, DFNodeType]
@dataclass
class DataDepNode(BasicDataDepNode):
    children: Optional[List[Any]] = field(default_factory=list)
    
    def add_child(self, child: Any) -> None:
        self.children.append(child)
    
    def get_children(self) -> List[Any]:
        return self.children
    
    def remove_child(self, child: Any) -> None:
        if child in self.children:
            self.children.remove(child)
            
    def __eq__(self, other) -> bool:
        if isinstance(other, DataDepNode):
            return (self.df_node == other.df_node and self.dat_ptr == other.dat_ptr)
        return False
    def add_node_to_graph(self, graph: pygraphviz.AGraph) -> None:
        for child in self.children:
            if child.df_node == DFNodeType.DF_START:
                graph.add_edge(f"{child.dat_ptr}:START", f"{self.dat_ptr}:{self.df_node.loop.kernel}")
            else:
                graph.add_edge(f"{child.dat_ptr}:{child.df_node.loop.kernel}", f"{self.dat_ptr}:{self.df_node.loop.kernel}")
        for child in self.children:
            child.add_node_to_graph(graph)
    
    def print_tree(self) -> str:
        logging.debug(f"Generating dependency graph for {self.dat_ptr}:{self.df_node.loop.kernel}")
        g = pygraphviz.AGraph(strict=True, directed=True)
        self.add_node_to_graph(g)
        
        g.layout(prog="dot")
        g.draw(f"{self.dat_ptr}_data_dependency.png")
    
    # def search
    def str_dep_tree(self) -> str:
        out_str = f"{self.dat_ptr}"
        if (self.children):
            out_str += " <- "
        for child in self.children:
            out_str += f"({child.str_dep_tree()}), "
        return out_str
    
    def __str__(self) -> str:
        out_str = f"loop kernel: {self.df_node.loop.kernel}, dep tree: {self.str_dep_tree()}"
        return out_str

def findVariables(astnode: Cursor, variables: Set) -> None:
    if astnode.kind == CursorKind.DECL_REF_EXPR:
        variables.add(astnode.spelling)
    
    for child in astnode.get_children():
        findVariables(child, variables)

def getDependencyVariables_v2(astnode: Cursor, dependant: str, visited: Set[str] = None, original_compound_ast_node: Cursor = None, curr_compound_idx: int = -1) -> Optional[Set[str]]:
    if visited is None:
        visited = set()
    
    if original_compound_ast_node is None:
        if not astnode.kind is CursorKind.COMPOUND_STMT:
            raise OptError("Original first node should be a compound statement")
        original_compound_ast_node = astnode
    
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
            dependencies.update(getDependencyVariables_v2(list(child.get_children())[-1], dependant))
            
        if variables_found:
            idx_cur_child_in_compound_children = children.index(child)
            dependencies.update(variables_found)
            
            for var in variables_found:
                dependencies.update(getDependencyVariables_v2(child, var, visited, original_compound_ast_node, idx_cur_child_in_compound_children))
    
    for var in local_vars:
        if not var in dependencies:
            continue
        dependencies.remove(var)
        
    return dependencies    
    
# def getDependencyVariables(astnode: Cursor, dependant: str, visited: List[str] = None, original_ast_node: Cursor = None) -> Optional[Set[str]]:
#     if visited is None:
#         visited = set()

#     # The exit condition
#     if dependant in visited:
#         return set()
    
#     dependencies = set()
#     dependant_found = False
    
#     if astnode.kind == CursorKind.VAR_DECL or astnode.kind == CursorKind.DECL_REF_EXPR:
#         if astnode.spelling == dependant:
#             dependant_found = True
#             visited.add(dependant)
#             newly_found = set()
#             findVariables(decend(astnode), newly_found)
#             dependencies.update(newly_found)
#             for var in newly_found:
#                 dependencies += getDependencyVariables(astnode, var, visited)
        
#     elif astnode.kind == CursorKind.BINARY_OPERATOR or astnode.kind == CursorKind.COMPOUND_ASSIGNMENT_OPERATOR:
#         operands = list(astnode.get_children())
        
#         if not len(operands) == 2:
#             raise OptError("Binary operator or compound assignment should have RHS and LHS")
        
#         lhs_operand = operands[0]
#         rhs_operand = operands[1]
#         print(f"Original ast node kind: {astnode.kind}, spelling: {astnode.spelling}, type: {astnode.type}, location: {astnode.extent}")
#         print(f"lhs: kind: {lhs_operand.kind}, spelling: {lhs_operand.spelling}, type: {lhs_operand.type}, location: {lhs_operand.extent}")
#         # print(f"lhs: kind: {rhs_operand.kind}, spelling: {rhs_operand.spelling}, type: {rhs_operand.type}, location: {rhs_operand.extent}")
        
#         if lhs_operand.kind == CursorKind.UNEXPOSED_EXPR:
#             lhs_operand = decend(lhs_operand)
#         if lhs_operand.spelling == dependant:
#             dependant_found = True
#             visited.add(dependant)
#             newly_found = set()
#             findVariables(rhs_operand, newly_found)
#             dependencies.update(newly_found)
#             for var in newly_found:
#                 new_dependencies = getDependencyVariables(astnode, var, visited)
#                 dependencies.update(new_dependencies)
    
#     elif not dependant_found:
#         for child in astnode.get_children():
#             new_dependencies = getDependencyVariables(child, dependant, visited)
#             if new_dependencies:
#                 dependencies.update(new_dependencies)
#                 dependant_found = True
#                 break
            
#     if not dependant_found:
#         logging.warning(f"Couldn't find dependant: {dependant}")
#         # return None
        
#     return dependencies
        
            # # Found a declaration or assignment for the variable
            # for child in cursor.get_children():
            #     if child.kind == clang.cindex.CursorKind.DECL_REF_EXPR:
            #         dep_var = child.spelling
            #         dependencies.append(dep_var)
            #         # Recursively find dependencies for this variable (indirect dependencies)
            #         dependencies += get_variable_dependencies(translation_unit.cursor, dep_var, visited)

            
# def  getSupportingVariablesInsideKernel(node: Cursor, dependant: str, last_child_node_idx: int = -1) -> list[str]:
#     if not node.kind == CursorKind.COMPOUND_STMT:
#         raise OptError(f"node should be compound statement")
    
#     children = [child for child in node.get_children()]
    
    
#     # if last_child_node_idx == -1:
#     #     end_child_node = len(children)
#     # else:
#     #     end_child_node =last_child_node_idx
    
#     support_var_list = [] 
#     for child in children[last_child_node_idx::-1]:
#         if child.kind == CursorKind.BINARY_OPERATOR:
#             ops = getBinaryOp(child)
            
#             if not ops.spelling == "=":
#                 continue
        
#             operands = list(child.get_children())
            
#             if not len(operands) == 2:
#                 raise OptError(f"Checking Binary operator in line {children.index(child)} should have LHS and RHS")
            
#             lhs = operands[0]
#             rhs = operands[1]
            
#             if lhs.kind == CursorKind.UNEXPOSED_EXPR:
#                 lhs_var = decend(lhs).spelling
#             elif lhs.kind == CursorKind.DECL_REF_EXPR:
#                 lhs_var = lhs.spelling
                 
#             if lhs_var == dependant:
#                 rhs_vars = [] 
#                 findVariables(rhs, rhs_vars)
#                 support_var_list.extend(rhs_vars)
#                 for rhs_var in rhs_vars:
#                     support_var_list.extend(v for v in getSupportingVariablesInsideKernel(node, rhs_var, children.index(child)) if v not in support_var_list)
#                 break
#         elif child.kind == CursorKind.DECL_STMT:
#             lhs_var = decend(child).spelling
            
#             if lhs_var == dependant:
#                 rhs_vars = []
#                 findVariables(decend(decend(child)), rhs_vars)
#                 support_var_list.extend(rhs_vars)
#                 for rhs_var in rhs_vars:
#                     support_var_list.extend(v for v in getSupportingVariablesInsideKernel(node, rhs_var, children.index(child)) if v not in support_var_list)
#                 break
#     return support_var_list

def  ISLDataDependancyCyclesDetection(original_graph: DataflowGraph_v2, prog: Program, app: Application, scheme: Scheme) -> DataflowGraph_v2:
    pass
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
    # kernel_processor = KernelProcess()
    
    # copy_graph = DataFlowGraph(original_graph.unique_name + "_dat_cycle_detected")
    # copy_graph.nodes = [node for node in original_graph.nodes]
    # copy_graph.edges = [edge for edge in original_graph.edges]
    # copy_graph.global_dats = [global_dat for global_dat in original_graph.global_dats]
    # copy_graph.global_dat_swap_map = [swap_id for swap_id in original_graph.global_dat_swap_map]
    
    
    # check_queue = []
    # cycle_heads = []
    # for edge in copy_graph.edges:
    #     if edge.sink_id == DFNodeType.DF_END:
    #         dep_node = DataDepNode(copy_graph.global_dats[edge.dat_id].ptr, copy_graph.getNode(edge.source_id))
    #         cycle_heads.append(dep_node)
    #         check_queue.append(dep_node)
    # init_dats = []
    # for dep_node in check_queue:        
    #     init_dats.append(dep_node.dat_ptr)
        
    # logging.debug(f"Initial dats being checked for dataDependency: {init_dats}")
            
    # while check_queue:
    #     depNode = check_queue.pop(0)
    #     if depNode.df_node == DFNodeType.DF_START:
    #         continue
    #     cur_dat_name = depNode.dat_ptr
    #     arg_dats = depNode.df_node.loop.get_arg_dat(cur_dat_name, [AccessType.OPS_WRITE, AccessType.OPS_RW])
    #     if not len(arg_dats) == 1:
    #         if len(arg_dats) == 0:
    #             raise OptError(f"Failed to find arg_dat of {depNode.dat_ptr} in loop: {depNode.df_node.loop.kernel} ({depNode.df_node.loop})")
    #         raise OptError(f"One ops_par_loop can only write a dat once. Critical error in {depNode.df_node.loop}")
    #     kernel_func = scheme.translateKernel(depNode.df_node.loop, prog, app, 1)
    #     # logging.debug(f"translated kernel function: {kernel_func}")
    #     kernel_func = kernel_processor.clean_kernel_func_text(kernel_func)
    #     # logging.debug(f"kernel function after cleaning: {kernel_func}")
    #     kernel_body, kernel_args = kernel_processor.get_kernel_body_and_arg_list(kernel_func)
    #     # logging.debug(f"kernel arguments: {kernel_args}, kernel body: {kernel_body}")
    #     # logging.debug(f"ops_par_loop arugments: {depNode.df_node.loop.args}")
    #     relevant_kernel_arg = kernel_args[depNode.df_node.loop.args.index(arg_dats[0])]
    #     # logging.debug(f"arg_dats: {arg_dats}, cur_dat_name: {cur_dat_name}")
    #     logging.debug(f"relevant_kernel_parameter to be checked for dependency: {relevant_kernel_arg} for argument {depNode.df_node.loop.dats[arg_dats[0].id].ptr}")
    #     kernel_entities = prog.findEntities(depNode.df_node.loop.kernel)
    #     kernel_children = [child for child in kernel_entities[0].ast.get_children()]
    #     # logging.debug(f"kernel entity: {kernel_entities[0].ast.spelling}, {kernel_entities[0].ast.extent}, {id(kernel_entities[0].ast)}")
    #     # lines = ASTtoString(kernel_entities[0].ast)
    #     # logging.debug(f"AST dump of the body of kernel declaration of kernel {depNode.df_node.loop.kernel}")
    #     # for line in lines:
    #     #     logging.debug(line)
        
    #     supporting_vars = getDependencyVariables_v2(kernel_children[-1], relevant_kernel_arg)
            
    #     logging.debug(f"supporting vars: {supporting_vars}")
    #     for var in supporting_vars:
    #         if var not in kernel_args:
    #             continue
    #         arg_idx = kernel_args.index(var)
    #         loop_arg = depNode.df_node.loop.args[arg_idx]
            
    #         if not isinstance(loop_arg, ArgDat):
    #             continue
            
    #         dat_name = depNode.df_node.loop.dats[loop_arg.dat_id].ptr
    #         logging.debug(f"found dat in supporting vars: {dat_name}")
    #         global_dat_id = findIdx(copy_graph.global_dats, lambda dat: dat.ptr == dat_name)
    #         logging.debug(f"found dat_id: {global_dat_id}")
    #         searched_edges = copy_graph.getEdge(depNode.df_node.node_id, global_dat_id, 1)
    #         logging.debug(f"found searched_edges: {searched_edges}")
    #         if not len(searched_edges) == 1:
    #             logging.warning(f"Couldn't find edge connecting to sink node {depNode.df_node.loop.kernel} with dat: {dat_name}({global_dat_id}), is there any nodes: {copy_graph.getEdge(depNode.df_node.node_id, global_dat_id)}")
    #             continue
    #         if searched_edges[0].source_id == DFNodeType.DF_START:
    #             newDepNode = DataDepNode(copy_graph.global_dats[global_dat_id].ptr, DFNodeType.DF_START)
    #         else:
    #             newDepNode = DataDepNode(copy_graph.global_dats[global_dat_id].ptr, copy_graph.getNode(searched_edges[0].source_id))
    #         depNode.add_child(newDepNode)
    #         check_queue.append(newDepNode)
    #         # dat_id = findIdx(copy_graph.global_dats, lambda dat: dat.ptr == dat_name)
    #         # # for 
    #         # # node = DataDepNode(dat_id, )
    # for head in cycle_heads:
    #     logging.debug(f"dependency train: {head}")
    #     head.print_tree()
    #     # dependancy_kernel_arguments = findDependancyArguments(kernel_children[-1], relevent_kernel_args[0])