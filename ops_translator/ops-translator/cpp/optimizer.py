import os
from ops import DataFlowGraph, DFNodeType
from typing import Optional, List, Tuple
from store import Program, Location, Application
import logging
from dataclasses import dataclass
from util import KernelProcess, findIdx
from scheme import Scheme
from cpp.parser import ASTtoString, CursorKind, getBinaryOp, Cursor, getAccessorAccessIndices
from copy import deepcopy
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
            
            LHS_op,LHS_indices = getAccessorAccessIndices(operands[0])
            RHS_op,RHS_indices = getAccessorAccessIndices(operands[1])
            
            if not len(LHS_indices) == len(RHS_indices) or not LHS_indices == RHS_indices:
                continue
            copy_pairs.append((LHS_op, RHS_op))
    return copy_pairs
        
def ISLCopyDetection(original_graph: DataFlowGraph, prog: Program, app: Application, scheme: Scheme) -> DataFlowGraph:
    ''' Checking the kernels writing output from the ISL region '''
    
    kernel_processor = KernelProcess()
    
    copy_graph = DataFlowGraph(original_graph.unique_name + "_copy_detected")
    copy_graph.nodes = [node for node in original_graph.nodes]
    copy_graph.edges = [edge for edge in original_graph.edges]
    copy_graph.global_dats = [global_dat for global_dat in original_graph.global_dats]
    copy_graph.global_dat_swap_map = [swap_id for swap_id in original_graph.global_dat_swap_map]
    
    checked_node_ids = []
    remove_nodes = []
    
    for edge in copy_graph.edges:
        if not edge.sink_id == DFNodeType.DF_END:
            continue
        
        logging.debug(edge)
        #get the kernel_entity
        
        if edge.source_id in checked_node_ids:
            copy_graph.edges.remove(edge)
            continue
        
        node = copy_graph.getNode(edge.source_id)
            
        if not node:
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
        logging.debug(f"ops_par_loop arugments: {node.loop.args}")
        
        if not len(kernel_args) == len(node.loop.args):
            OptError("Critical error, number of kernel arg should match with ops_par_loop args")
        
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
                lhs_global_dat_idx = findIdx(copy_graph.global_dats, lambda dat: dat.ptr == lhs_dat_name)
                rhs_global_dat_idx = findIdx(copy_graph.global_dats, lambda dat: dat.ptr == rhs_dat_name)
                logging.debug(f"lhs_global_dat_idx: {lhs_global_dat_idx}, rhs_global_dat_idx: {rhs_global_dat_idx}")
                copy_graph.global_dat_swap_map[lhs_global_dat_idx] = rhs_global_dat_idx
                copy_graph.global_dat_swap_map[rhs_global_dat_idx] = lhs_global_dat_idx 
            checked_node_ids.append(edge.source_id)
            remove_nodes.append(node)
            copy_graph.edges.remove(edge)
    
    for node in remove_nodes:
        copy_graph.nodes.remove(node)
        
    remove_edges = []
    for edge in copy_graph.edges:
        if edge.sink_id in checked_node_ids:
            remove_edges.append(edge)
    
    for edge in remove_edges:
        edge.sink_id = DFNodeType.DF_END
        # copy_graph.edges.remove(edge)
    
    logging.debug(f"global swap map is ISL COpy detect: {copy_graph.global_dat_swap_map}")
    copy_graph.print("after_copy_detection")
    return copy_graph
        # logging.debug(f"arguments of the kernel: {Ler}")
        

