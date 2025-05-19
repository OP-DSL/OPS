import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any
from clang import cindex
from clang.cindex import Cursor, CursorKind, TranslationUnit, TypeKind, conf, Token

import ops
from store import Function, Location, ParseError, Program, Type
from util import safeFind, function_name, findIdx #TODO: implement safe find
import logging
from math import floor
from dataclasses import field
from cpp.preprocessor import isl_directive

def ASTtoString(node: Cursor, indent: str = "", is_last=True, print_lines: Optional[List[str]] = None) -> List[str]:
    if not print_lines:
        print_lines = []
    prefix = indent + ("└─" if is_last else "├─")   
    print_lines.append(f"{prefix} {node.kind} {node.spelling} [{node.location}] (type: {node.type})")
    indent += "  " if is_last else "| "
    children = list(node.get_children())
    
    for i, child in enumerate(children):
        ASTtoString(child, indent, i == len(children) - 1, print_lines)

    return print_lines

def dumpAST(node: Cursor, filename: Path)-> None:
    printLines = ASTtoString(node)
    
    with open(f"{filename}_ast.txt", "w+") as f:
        for line in printLines:
            f.write(line + "\n")
            
def parseMeta(node: Cursor, program: Program) -> None:
    if node.kind == CursorKind.TYPE_REF:
        parseTypeRef(node, program)

    if node.kind == CursorKind.FUNCTION_DECL:
        parseFunction(node, program) #TODO

    for child in node.get_children():
        parseMeta(child, program) 

def getBinaryOp(node: Cursor) -> Token:
    assert node.kind == CursorKind.BINARY_OPERATOR
    children = [child for child in node.get_children()]
    assert len(children) == 2
    token_offset = len([i for i in children[0].get_tokens()])
    return [i for i in node.get_tokens()][token_offset]

def getUnaryOp(node: Cursor) -> Token:
    assert node.kind == CursorKind.UNARY_OPERATOR
    return [i for i in node.get_tokens()][0]

def getAccessorAccessIndices(node: Cursor) -> Optional[Tuple[str, List]]:
    if not node.kind == CursorKind.UNEXPOSED_EXPR:
        logging.warning(f"Accessor access operand {node.spelling} is {node.kind}. It has to be UNEXPOSED_EXPR")
        return None
    children = [child for child in node.get_children()]
    operand = children[0].spelling
    indices = [parseIntLiteral(child) for child in children[1:]]
    return (operand, indices)
    
def parseTypeRef(node: Cursor, program: Program) -> None:
    node = node.get_definition()

    if node is None or Path(str(node.location.file)) != program.path:
        return

    matching_entities = program.findEntities(node.spelling)

    for entity in matching_entities: 
        if entity.ast == node: #The entitiy is already exist
            return

    typ = Type(node.spelling, node, program) 

    for n in node.walk_preorder():
        if n.kind != CursorKind.CALL_EXPR and n.kind != CursorKind.TYPE_REF:
            continue

        n = n.get_definition()

        if n is None or Path(str(n.location.file)) != program.path:
            continue

        typ.depends.add(n.spelling)

    program.entities.append(typ)


def parseFunction(node: Cursor, program: Program) -> None:
    node = node.get_definition()

    if node is None or Path(str(node.location.file)) != program.path: 
        return

    matching_entities = program.findEntities(node.spelling)

    for entity in matching_entities:
        if entity.ast == node: #The entitiy is already exist
            return

    function = Function(node.spelling, node, program)

    for n in node.get_children():
        if n.kind != CursorKind.PARM_DECL:
            continue

        function.parameters.append(n.spelling)

    for n in node.walk_preorder():
        if n.kind != CursorKind.CALL_EXPR and n.kind != CursorKind.TYPE_REF:
            continue

        n = n.get_definition()

        if n is None or Path(str(n.location.file)) != program.path:
            continue

        function.depends.add(n.spelling) 
    
    function.loc = parseLocation(node)
    
    program.entities.append(function)


def parseLocation(node: Cursor) -> Location:
    return Location(node.location.file.name, node.location.line, node.location.column)

def getLocation(location: Cursor.location) -> Location:
    return Location(location.file.name, location.line, location.column)

def parseLoops(translation_unit: TranslationUnit, program: Program) -> None:
    logging.debug(f"{function_name()}: Starting Parsing")
    macros: Dict[Location, str] = {}
    nodes: List[Cursor] = []
    isl_directives = program.isl_directives
    cur_isl_directive_id = None
    
    if not isl_directives:
        isl_directives = None
    else:
        cur_isl_directive_id = 0
        
    # counter = 0
    for node in translation_unit.cursor.get_children():
        # print(f"node {counter}: {node.spelling}")
        # counter += 1
            
        if node.kind == CursorKind.MACRO_DEFINITION:
            continue
                
        if node.location.file.name != translation_unit.spelling:
            continue

        if node.kind == CursorKind.MACRO_INSTANTIATION:
            macros[parseLocation(node)] = node.spelling
            continue
        
        nodes.append(node)

    for node in nodes:
        for child in node.walk_preorder(): 
            # print(f"node {node.spelling}: child - {child.spelling}/{child.kind}")  
            if child.kind == CursorKind.FOR_STMT:
                logging.debug(f" For loop found: {parseLocation(child)}")
                
                if isl_directives is None:
                    logging.debug(f" Not a ISL")
    
                elif child.location.line >  isl_directives[cur_isl_directive_id].get_lineno():
                    logging.debug(f" found iter_parloop: {isl_directives[cur_isl_directive_id].get_isl_name()}, param: {isl_directives[cur_isl_directive_id].get_max_iter_param()}")
                    
                    outerLoop = parseIterForLoop(child, isl_directives[cur_isl_directive_id], macros, program)
                    program.outerloops.append(outerLoop)
                    
                    if cur_isl_directive_id < len(isl_directives) - 1:
                        cur_isl_directive_id +=1
                    else:
                        isl_directives = None  
                
            # if child.kind == CursorKind.CALL_EXPR:
            #     parseCall(child, macros, program)
                
            elif child.kind.is_unexposed():
                parseCallUnexposed(child, macros, program)
                
            elif child.kind in [CursorKind.VAR_DECL, CursorKind.BINARY_OPERATOR]:
                parseVariableDeclaration(child, macros, program)
                
            # elif child.kind == CursorKind.OVERLOADED_DECL_REF:
            #     parseStencil()
                
    return program

# def parseForLoop(node: Cursor):
#     for child in node.get_children():
#         if child.kind == CursorKind.DECL_STMT:
#             parseDeclStmt(child)
#         elif child.kind == CursorKind.BINARY_OPERATOR:
#             parseConditions(child)
#         elif child.kind == CursorKind.UNARY_OPERATOR:
#             parseIters(child)    
        
# def parseDeclStmt(node: Cursor):
#     print (f"found declaration: {parseLocation(node)}")
#     for child in node.get_children():
#         print(f"   child - {child.spelling}/{child.kind}")
    
# def parseConditions(node: Cursor):
#     print (f"found condition: {parseLocation(node)}")
#     for child in node.get_children():
#         print(f"   child - {child.spelling}/{child.kind}")
    
# def parseIters(node: Cursor):
#     print (f"found Iteration: {parseLocation(node)}")
#     for child in node.get_children():
#         print(f"   child - {child.spelling}/{child.kind}")    
          

def parseVariableDeclaration(node: Cursor, macros: Dict[Location, str], program: Program) -> None:
    children = []

    if (node.kind == CursorKind.VAR_DECL):

        for child in node.get_children():
            children.append(child)
           
        if len(children) < 2:
            return
        var_name = node.spelling
        var_name_raw = node.spelling
        rval_expr = children[1]
            
    elif (node.kind == CursorKind.BINARY_OPERATOR):
        for child in node.get_children():
            children.append(child)
        
        if len(children) < 2:
            return
        
        l_val = children[0]
        var_name_raw =  parseIdentifier(l_val)
        if l_val.kind == CursorKind.ARRAY_SUBSCRIPT_EXPR:
            var_name = decend(l_val).spelling
        else:
            var_name = l_val.spelling
        rval_expr = children[1]
    
    logging.debug("Variable detected: %s, r_val_kind: %s", var_name, rval_expr.kind)

    # print(f"rval_expr.spelling: {rval_expr.spelling}, kind: {rval_expr.kind}")
    if rval_expr.kind == CursorKind.CALL_EXPR or rval_expr.kind.is_unexposed():
        if rval_expr.kind.is_unexposed():
            logging.debug(" |- This is unexposed")
            out = parseUnexposedFunction(rval_expr)
        else:
            logging.debug(" |- This is exposed")
            out = parseFunctionCall(rval_expr)
        
        if not out:
            return
        
        (name, args) = out
        
        if name == "ops_decl_stencil":
            parseStencil(var_name, args, parseLocation(node), macros, program)
    
     
        if name == "ops_decl_block":
            logging.debug("found ops_decl_block")
            if len(args) != 2:
                raise ParseError("ops_decl_block has 2 arguments", parseLocation(node))
            parseBlock(node, var_name, args, parseLocation(node), program)
            
        if name == "ops_decl_dat":
            logging.debug("found ops_dec_dat")
            if len(args) != 9:
                raise ParseError("ops_decl_dat has 9 arguments", parseLocation(node))
            parseDat(node, var_name_raw, var_name, args, parseLocation(node), program)
        
def parseFunctionCall(node: Cursor) -> Union[Tuple[str, List[Cursor]], None]:
    args = []
    if node.kind != CursorKind.CALL_EXPR:
        logging.debug(" |- This is not a call expression")
        return None
    
    # aststringlist = ASTtoString(decend(node))
    # aststring = ""
    
    # for line in aststringlist:
    #     aststring += line + "\n"
        
    # logging.debug(f"AST: \n" + aststring)
    
    for arg in node.get_arguments():
        args.append(arg)

    name = node.spelling
    if name == "":
        if decend(node).kind == CursorKind.DECL_REF_EXPR and decend(decend(node)).kind == CursorKind.OVERLOADED_DECL_REF:
            name = decend(decend(node)).spelling

    logging.debug("Detected function: %s, args: %s", name, args)
        
    return (name, args)


# def parseCall(node: Cursor, macros: Dict[Location, str], program: Program) -> None:
#     logging.debug(f"{function_name()} at {parseLocation(node)}")
#     out = parseFunctionCall(node)
    
#     if out == None:
#         return

#     (name, args) = out  
#     loc = parseLocation(node)

#     if name == "ops_decl_stencil":
#         if len(args) != 4:
#             raise ParseError("ops_decl_stencil need 4 arguments", loc)
        
#         # print(f"node {node}: {node.spelling}: type: {node.kind}")
#         # for arg in args: 
#         #     if arg.kind == CursorKind.UNEXPOSED_EXPR and arg.get_definition():
#         #         print(f"|-- argument {arg.spelling}: type: {arg.kind}, definition: {arg.get_definition().kind}, defLoc:{str(parseLocation(arg.get_definition()))}")
#         #     else:
#         #         print(f"|-- argument {arg.spelling}: type: {arg.kind}")

#     # print(f"node:{node.spelling}, type: {node.type}")
#     ov_node = decend(decend(node))   
    
#     # print (f"node: {name}, ov_node: {ov_node.spelling}, args: {args}") 
#     if not ov_node:
#         return
    
#     name = ov_node.spelling        
#     if name == "ops_iter_par_loop":
#         scope = [getLocation(node.extent.start), getLocation(node.extent.end)]
#         outerLoop = parseIterLoop(node, args, scope, macros, program)
#         program.outerloops.append(outerLoop)
        
        
def parseStencil(name: str, args: List[Cursor], loc: Location, macros: Dict[Location, str], program: Program, vector_factor: int = 1) -> Union[ops.Stencil, None]:
    
    if len(args) != 4:
        return None
    
    ndim = parseIntLiteral(args[0])
    npoints = parseIntLiteral(args[1])
    array = ops.stencilPointsSort(npoints, ndim, parseArrayIntLit(args[2]))
    d_m = ops.Point([abs(min([point.x for point in array])), abs(min([point.y for point in array])), abs(min([point.z for point in array]))])
    d_p = ops.Point([abs(max([point.x for point in array])), abs(max([point.y for point in array])), abs(max([point.z for point in array]))])

    minPoint = ops.getMinPoint(array)
    base_offset = -minPoint
    array = ops.cordinateOriginTranslation(base_offset, array)
    stencilSize = ops.getStencilSize(array)
    row_discriptors = ops.genRowDiscriptors(array, base_offset)
    # widen_array, point_to_widen_map = ops.computeWidenPoints(row_discriptors, vector_factor)
    # widen_npoints = len(widen_array)
    # windowBuffers, chains = ops.windowBuffChainingAlgo(widen_array, ndim)
    
    
    logging.info("Stencil found name:%s ndim: %d, npoints: %d, array: %s", name, ndim, npoints, str(array))
    # logging.debug("Stencil found addition info: windowBuffers: %s", str(windowBuffers))
    # logging.debug("Stencil found addition info: chains: %s", str(chains) )
    logging.debug("Stencil found addition info: row discriptors: %s", str(row_discriptors))
    # logging.debug("Stencil found addition info: point to widen point map: %s", str(point_to_widen_map))
    program.stencils.append(ops.Stencil(len(program.stencils), ndim, name, npoints, array, base_offset, stencilSize, d_m, d_p, row_discriptors))   


    
def parseArrayIntLit(node: Cursor)->List[int]:
    logging.debug("array: %s, array_decl: %s, type:%s, loc:%s", node, node.get_definition(), 
            node.get_definition().kind, parseLocation(node.get_definition()))
    if node.get_definition().kind == CursorKind.VAR_DECL:
        logging.debug("array typekind: %s", node.get_definition().type.kind)
        if node.get_definition().type.kind != TypeKind.CONSTANTARRAY:
            raise ParseError("The array of stencil points should be contant sized array", parseLocation(node.get_definition()))
    else:
        raise ParseError("Not an array node", parseLocation(node.get_definition()))
    
    first_child = decend(node.get_definition())

    if first_child.kind != CursorKind.INIT_LIST_EXPR:
        #TODO: make sure this is not breaking the other platforms. Technically, this has to be the case.
        raise ParseError("Array is not declared with initializer. OPS stencil point array has to be intialized", parseLocation(node.get_definition()))
    outList=[]
    
    for child in first_child.get_children():
        logging.debug("|--- aray elem: %s, kind: %s", child.spelling, child.kind)
        if child.kind == CursorKind.INTEGER_LITERAL:
            outList.append(parseIntLiteral(child))

        if child.kind == CursorKind.UNARY_OPERATOR:
            tokens = [token.spelling for token in child.get_tokens()]
            logging.debug("  |--- tokens: %s", tokens)
            
            if tokens[0] == '-':
                outList.append(-parseIntLiteral(decend(child)))
            elif tokens[0] == '+':
                outList.append(parseIntLiteral(decend(child)))
            else:
                raise ParseError(f"Not supported unary operator", parseLocation(node))
      
    logging.debug("Array: %s", outList)
    return outList
             
    
def parseUnexposedFunction(node: Cursor) -> Union[Tuple[str, List[Cursor]], None]:
    args = []
                    
    for child in node.get_children():
        args.append(child)

    if len(args) == 0:
        return None

    first_child = args.pop(0)
    # print(f"   Unexposed Func first child: {first_child.spelling}, kind: {first_child.kind}")
    if (first_child.kind == CursorKind.MEMBER_REF_EXPR
        and len(list(first_child.get_children())) >= 2):
        name_token = list(first_child.get_children())[1]
        name = name_token.spelling
    elif first_child.kind == CursorKind.DECL_REF_EXPR:
        name = list(first_child.get_tokens())[0].spelling
    else:
        return None

    return (name, args)

    
def parseCallUnexposed(node: Cursor, macros: Dict[Location, str], program: Program) -> None:

    out = parseUnexposedFunction(node)
    
    if out == None:
        return
    else:
        (name, args) = out

    loc = parseLocation(node)

    if name == "ops_decl_const" or name == "decl_const":
        program.consts.append(parseConst(args, loc, macros))

    elif name == "ops_par_loop":
        loop = parseLoop(node, args, loc, macros)
        if loop not in program.loops:
            program.loops.append(loop)

        if program.ndim == None:
            program.ndim = loop.ndim
        elif program.ndim < loop.ndim:
            program.ndim = loop.ndim

    
        
def decend(node: Cursor) -> Optional[Cursor]:
    if not node:
        return None
    return next(node.get_children(), None)


def parseStringLit(node: Cursor) -> str:
    if node.kind == CursorKind.UNEXPOSED_EXPR:
        node = decend(node)
        if node.kind != CursorKind.STRING_LITERAL:
            raise ParseError("Expected string literal", parseLocation(node))

    elif node.kind != CursorKind.STRING_LITERAL:
        raise ParseError("Expected string literal", parseLocation(node))

    return node.spelling[1:-1]


def parseIntExpression(node: Cursor) -> int:
    if node.type.kind != TypeKind.INT:
        raise ParseError("Expected int expression", parseLocation(node))

    eval_result = conf.lib.clang_Cursor_Evaluate(node)
    val = conf.lib.clang_EvalResult_getAsInt(eval_result)
    conf.lib.clang_EvalResult_dispose(eval_result)

    return val


def parseIntLiteral(node: Cursor) -> Optional[int]:
    if node.type.kind != TypeKind.INT:
        raise ParseError("Expected int expression", parseLocation(node))

    if node.kind == CursorKind.INTEGER_LITERAL:
        eval_result = conf.lib.clang_Cursor_Evaluate(node)
        val = conf.lib.clang_EvalResult_getAsInt(eval_result)
        conf.lib.clang_EvalResult_dispose(eval_result)
        return val
    else:
        try:
            eval_result = conf.lib.clang_Cursor_Evaluate(node)
            
            if eval_result:
                eval_kind = conf.lib.clang_EvalResult_getKind(eval_result) #should produce CXEval_int == 1
                if eval_kind == 1:
                    val = conf.lib.clang_EvalResult_getAsInt(eval_result)
                    conf.lib.clang_EvalResult_dispose(eval_result)
                    return val
                else:
                    raise ParseError()
            else:
                raise ParseError()
        except Exception as e:
            conf.lib.clang_EvalResult_dispose(eval_result)
            raise ParseError(f"Invalid node type {str(node.kind)} raised error{e}", parseLocation(node))


def parseType(typ: str, loc: Location, include_custom=False) -> Tuple[ops.Type, bool]:
    typ_clean = typ.strip()
    typ_clean = re.sub(r"\s*const\s*", "", typ_clean) # Why removing const?

    soa = False
    if re.search(r":soa", typ_clean):
        soa = True

    typ_clean = re.sub(r"\s*:soa\s*", "", typ_clean)

    typ_map = {
        "short": ops.Int(True, 16),
        "unsigned short": ops.Int(False, 16),
        "ushort": ops.Int(False, 16),
        "int": ops.Int(True, 32),
        "long": ops.Int(True, 32),
        "long int": ops.Int(True, 32),
        "uint": ops.Int(False, 32),
        "unsigned int": ops.Int(False, 32),
        "ll": ops.Int(True, 64),
        "long long": ops.Int(True, 64),
        "ull": ops.Int(False, 64),
        "unsigned long long": ops.Int(False, 64),
        "half": ops.Float(16),
        "float": ops.Float(32),
        "double": ops.Float(64),
        "bool": ops.Bool(),
        "char": ops.Char(),
        "complexd": ops.ComplexD(),
        "complexf": ops.ComplexF()
    }

    if typ_clean in typ_map:
        return typ_map[typ_clean], soa

    if include_custom:
        return ops.Custom(typ_clean), soa

    raise ParseError(f"Unable to parse type: '{typ}'", loc)


def parseIdentifier(node: Cursor, raw: bool = True) -> str:
    if raw:
        return "".join([t.spelling for t in node.get_tokens()])
    
    while node.kind == CursorKind.CSTYLE_CAST_EXPR:
        node = list(node.get_children())[1]
        
    if node.kind == CursorKind.UNEXPOSED_EXPR:
        logging.debug(f"{function_name()}: unexposed")
        return parseIdentifier(decend(node), raw)

    elif node.kind == CursorKind.ARRAY_SUBSCRIPT_EXPR:
        logging.debug(f"{function_name()}: array_subscript")
        return parseIdentifier(decend(node), raw)
        
    elif node.kind == CursorKind.UNARY_OPERATOR and next(node.get_tokens()).spelling in ("&", "*"):
        logging.debug(f"{function_name()}: ref or pointer")
        return parseIdentifier(decend(node), raw)

    elif node.kind == CursorKind.GNU_NULL_EXPR:
        raise ParseError("Expected identifier, found NULL", parseLocation(node))

    elif node.kind != CursorKind.DECL_REF_EXPR:
        raise ParseError("Expected identifier", parseLocation(node))

    return node.spelling

def getIdentiferDefinition(node: Cursor) -> Optional[Cursor]:
    if node.kind == CursorKind.UNEXPOSED_EXPR:
        node = decend(node)

    if node.kind == CursorKind.UNARY_OPERATOR and next(node.get_tokens()).spelling in ("&", "*"):
        node = decend(node)

    if node.kind == CursorKind.GNU_NULL_EXPR:
        raise ParseError("Expected identifier, found NULL", parseLocation(node))

    if node.kind != CursorKind.DECL_REF_EXPR:
        raise ParseError("Expected identifier", parseLocation(node))
    
    return node.get_definition()

def parseConst(args: List[Cursor], loc: Location, macros: Dict[Location, str]) -> ops.Const:
    if(len(args) != 4):
        raise ParseError(f"Incorrect number({len(args)}) of args passed to ops_decl_const", loc)

    name = parseStringLit(args[0])
    if parseLocation(args[1]) in macros.keys():
        dim = macros[parseLocation(args[1])]
    else:
        dim = parseIdentifier(args[1])
    typ, _ = parseType(parseStringLit(args[2]), loc, True)
    ptr = parseIdentifier(args[3], raw=False)
    # ptr_def_node = getIdentiferDefinition(args[3])
    # print(f"Parse Const: ptr_node: {args[3]}, definition_node:{ptr_def_node.spelling}, is definition extern: {ptr_def_node.extent}")
    return ops.Const(loc, ptr, dim, typ, name)


def parseAccessType(node: Cursor, loc: Location, macros: Dict[Location, str]) -> ops.AccessType:
    access_type_raw = parseIntExpression(node)

    if access_type_raw not in ops.AccessType.values():
        raise ParseError(
            f"Invalid access type {access_type_raw}, expected one of {', '.join(ops.AccessType.values())}", 
            loc)

    return ops.AccessType(access_type_raw)


def parseArgDat(loop: ops.Loop, args: List[Cursor], loc: Location, macros: Dict[Location, str]) -> None:
    if len(args) != 5:
        raise ParseError(f"Incorrect number({len(args)}) of args passed to ops_arg_dat", loc)

    aststringlist = ASTtoString(args[0])
    aststring = ""
    
    for line in aststringlist:
        aststring += line + "\n"
        
    logging.debug(f"AST: \n" + aststring)
    
    dat_raw_ptr = parseIdentifier(args[0])
    dat_ptr = parseIdentifier(args[0], raw=False)
    dim = parseIntLiteral(args[1])
    stencil_ptr = parseIdentifier(args[2])
    dat_typ, dat_soa = parseType(parseStringLit(args[3]), loc)
    access_type = parseAccessType(args[4], loc, macros)

    logging.debug(f"Found arg dat {dat_ptr}")
    loop.addArgDat(loc, dat_raw_ptr, dat_ptr, dim, dat_typ, dat_soa, stencil_ptr, access_type, True)


def parseArgDatOpt(loop: ops.Loop, args: List[Cursor], loc: Location, macros: Dict[Location, str]) -> None:
    if len(args) != 6:
        raise ParseError(f"Incorrect number({len(args)}) of args passed to ops_arg_dat_opt", loc)

    dat_raw_ptr = parseIdentifier(args[0])
    dat_ptr = parseIdentifier(args[0], raw=False)
    dim = parseIntLiteral(args[1])
    stencil_ptr = parseIdentifier(args[2])
    dat_typ, dat_soa = parseType(parseStringLit(args[3]), loc)
    access_type = parseAccessType(args[4], loc, macros)
    opt = parseIntExpression(args[5])

    loop.addArgDat(loc, dat_raw_ptr, dat_ptr, dim, dat_typ, dat_soa, stencil_ptr, access_type, bool(opt))


def parseArgReduce(loop: ops.Loop, args: List[Cursor], loc: Location, macros: Dict[Location, str]) -> None:
    if len(args) != 4:
        raise ParseError(f"Incorrect number of args passed to ops_arg_reduce: {len(args)}", loc)

    reduct_handle_ptr = parseIdentifier(args[0])
    dim = parseIntLiteral(args[1])
    typ, soa = parseType(parseStringLit(args[2]), loc)
    access_type = parseAccessType(args[3], loc, macros)

    loop.addArgReduce(loc, reduct_handle_ptr, dim, typ, access_type)

def parseBlock(node: Cursor, ptr: str, args: List[Any], loc: Location, prog: Program) -> ops.Block:
    dim = parseIntExpression(args[0])
    try:
        prompt = parseStringLit(args[1])
    except ParseError as e:
        logging.warning(f"{e.args[0]}")
        prompt = ""
        
    block = ops.Block(loc, ptr, dim, prompt)
    
    # if prog.ndim < dim: 
    #     raise ParseError(f"Defining block dim {dim} is not permited inside a {prog.ndim}D program", loc)
    
    idx = safeFind(prog.blocks, lambda b: b.ptr == ptr)
    
    if not idx is None:
        raise ParseError(f"Block {ptr} is redefined. First defined in {prog.blocks[idx].loc} and again in {loc}")
    
    block.id = len(prog.blocks)
    prog.blocks.append(block)

def parseDat(node: Cursor, ptr_raw: str, ptr: str, args: List[Any], loc: Location, prog: Program) -> None:
    block_ptr = parseIdentifier(args[0], raw=False)
    multidim_dim = parseIntExpression(args[1])
    typ, soa = parseType(parseStringLit(args[7]), parseLocation(args[7]))
    blk_idx = findIdx(prog.blocks, lambda blk: blk.ptr == block_ptr)
    
    if block_ptr is None:
        raise ParseError(f"Unable to find Block ({block_ptr})", parseLocation(node))
    
    prog.blocks[blk_idx].dats.append(ops.Dat(len(prog.blocks[blk_idx].dats), ptr_raw, ptr, multidim_dim, typ, soa, blk_idx))
    
    

def parseRange(node: Cursor, dim: int) -> ops.Range:
    ptr = parseIdentifier(node)
    loc = parseLocation(node)
    return ops.Range(loc, ptr, dim)


def parseArgGbl(loop: ops.Loop, args: List[Cursor], loc: Location, macros: Dict[Location, str]) -> None:
    if len(args) !=4:
        raise ParseError("Incorrect number of args passed to ops_arg_gbl", loc)

    ptr = parseIdentifier(args[0])
    if parseLocation(args[1]) in macros.keys():
        dim = macros[parseLocation(args[1])]
    else:
        dim = parseIdentifier(args[1])
    typ, _ = parseType(parseStringLit(args[2]), loc)
    access_type = parseAccessType(args[3], loc, macros)

    loop.addArgGbl(loc, ptr, dim, typ, access_type)


def parseArgIdx(loop: ops.Loop, args: List[Cursor], loc: Location, macros: Dict[Location, str]) -> None:
    if len(args) !=0:
        raise ParseError("Incorrect number of args passed to ops_arg_idx", loc)

    loop.addArgIdx(loc)


def parseIterForLoop(node: Cursor, isl_dir: isl_directive, macros: Dict[Location, str], program: Program)-> ops.IterLoop:
    if node.kind != CursorKind.FOR_STMT:
        raise ParseError("The ISL region shuld be a FOR statement", parseLocation(node))
    lines = ASTtoString(node)
    
    logging.debug("ISL Region AST")
    for  l in  lines:
        logging.debug(l)
    
    cond_stmt = None
    init_stmt = None
    inc_stmt = None
    comp_stmt = None
    iterLoop_args = []
     
    for child in node.get_children():
        if child.kind == CursorKind.BINARY_OPERATOR:
            cond_stmt = child
        elif child.kind == CursorKind.DECL_STMT:
            init_stmt = child
        elif child.kind == CursorKind.UNARY_OPERATOR:
            inc_stmt = child
        elif child.kind == CursorKind.COMPOUND_STMT:
            comp_stmt = child
    
    init_val_expr = None
    if decend(decend(init_stmt)).kind == CursorKind.INTEGER_LITERAL:
        init_val_expr = decend(decend(init_stmt))
    elif decend(decend(init_stmt)).kind == CursorKind.UNEXPOSED_EXPR \
            and decend(decend(decend(init_stmt))).kind == CursorKind.INTEGER_LITERAL:
        init_val_expr = decend(decend(decend(init_stmt)))
    else:
        raise ParseError("ISL For loop should be a simple iteration variable with 0 initializer")
    
    if parseIntLiteral(init_val_expr) != 0:
        raise ParseError("Currenty loop iterator supports for ISL FOR STMT with 0 intialization", parseLocation(node))
    
    if not getBinaryOp(cond_stmt).spelling == "<":
        raise ParseError(f"Only the < operator in ISL FOR STMT supported. got {getBinaryOp(cond_stmt).spelling}")
    
    #TODO: Fix this check
    # if inc_stmt == None or not getUnaryOp(inc_stmt).spelling == "++":
    #     raise ParseError(f"Only supported increament operation in ISL FOR is ++", parseLocation(node))
    
    for child in comp_stmt.get_children():
        if child.kind != CursorKind.UNEXPOSED_EXPR:
            raise ParseError(f"The ISL FOR STMT only allow ops_par_loop statements")

        out = parseUnexposedFunction(child)
    
        if out == None:
            raise ParseError("Incorrect argument passed to ops_iter_par_loop")
        else:
            (name, loop_args) = out
            
        if name == "ops_par_loop":
            loop = parseLoop(child, loop_args, parseLocation(child), macros)
            loop.iterativeLoopId = len(program.outerloops)
            
            iterLoop_args.append(loop)
            
            if loop not in program.loops:
                program.loops.append(loop)

            if program.ndim == None:
                program.ndim = loop.ndim
            elif program.ndim < loop.ndim:
                program.ndim = loop.ndim
        
        elif name == "ops_par_copy":
            parCpyObj = parse_par_copy(child, loop_args, parseLocation(child), macros)
            iterLoop_args.append(parCpyObj)
     
    iterLoopObj = ops.IterLoop(isl_dir.get_isl_name(), len(program.outerloops), isl_dir.get_max_iter_param(), [getLocation(node.extent.start), getLocation(node.extent.end)], iterLoop_args)
    if iterLoopObj.unique_id not in program.uniqueOuterloopMap.keys():
        program.uniqueOuterloopMap[iterLoopObj.unique_id] = len(program.uniqueOuterloopMap.keys())
    
    return iterLoopObj       
        
    
def parseIterLoop(node: Cursor, args: List[Cursor], scope: List[Location], macros: Dict[Location, str], program: Program)-> ops.IterLoop:
    logging.debug(f"Found iterLoop: {node.spelling}, scope: (start - {getLocation(node.extent.start)}, end - {getLocation(node.extent.end)}), args: {print([(arg.spelling, arg.kind) for arg in args])}")
    is_iter_literal = args[1].kind == CursorKind.INTEGER_LITERAL
    iterLoop_args = []
    unique_name = parseStringLit(args[0])

    if is_iter_literal:
        iter = parseIntLiteral(args[1])
    else:
        iter = args[1].spelling
    
    for arg in args[2:]:
        out = parseUnexposedFunction(arg)
    
        if out == None:
            raise ParseError("Incorrect argument passed to ops_iter_par_loop")
        else:
            (name, loop_args) = out
        
        if name == "ops_par_loop":
            loop = parseLoop(arg, loop_args, parseLocation(arg), macros)
            loop.iterativeLoopId = len(program.outerloops)
            
            iterLoop_args.append(loop)
            
            if loop not in program.loops:
                program.loops.append(loop)

            if program.ndim == None:
                program.ndim = loop.ndim
            elif program.ndim < loop.ndim:
                program.ndim = loop.ndim
                
        elif name == "ops_par_copy":
            parCpyObj = parse_par_copy(arg, loop_args, parseLocation(arg), macros)
            iterLoop_args.append(parCpyObj)
            
            
    iterLoopObj = ops.IterLoop(unique_name, len(program.outerloops), iter, scope, iterLoop_args)
    if iterLoopObj.unique_id not in program.uniqueOuterloopMap.keys():
        program.uniqueOuterloopMap[iterLoopObj.unique_id] = len(program.uniqueOuterloopMap.keys())
    
    return iterLoopObj
            
        
def parse_par_copy(loopNode: Cursor, args: List[Cursor], loc: Location, macros: Dict[Location, str]) -> ops.ParCopy:
    if len(args) != 2:
        raise ParseError("Incorrect number of args passed to ops_par_copy")
    
    target = parseIdentifier(args[0], raw=False)
    source = parseIdentifier(args[1], raw=False)
    
    return ops.ParCopy(target, source)
        
    
def parseLoop(loopNode: Cursor, args: List[Cursor], loc: Location, macros: Dict[Location, str]) -> ops.Loop:
    if len(args) < 6:
        raise ParseError("Incorrect number of args passed to ops_par_loop")

    kernel = parseIdentifier(args[0])
    name   = parseStringLit(args[1])
    dim    = parseIntLiteral(args[3])
    block  = parseIdentifier(args[2])
    range = parseRange(args[4], dim)

    logging.debug(f"{function_name()}: kernel={kernel}, name={name}, dim={dim}, block={block}, range={range}\n")
    
    loop = ops.Loop(loopNode, loc, kernel, block, range, dim)

    for node in args[5:]:
        node_name = node.spelling

        arg_loc = parseLocation(node)
        arg_args = list(node.get_arguments())

        if node_name == "ops_arg_dat":
            parseArgDat(loop, arg_args, arg_loc, macros)

        elif node_name == "ops_arg_dat_opt":
            parseArgDatOpt(loop, arg_args, arg_loc, macros)

        elif node_name == "ops_arg_idx":
            parseArgIdx(loop, arg_args, arg_loc, macros)

        elif node_name == "ops_arg_reduce":
            parseArgReduce(loop, arg_args, arg_loc, macros)

        elif node.kind.is_unexposed() and parseUnexposedFunction(node) != None:
            (_, arg_args) = parseUnexposedFunction(node)
            parseArgGbl(loop, arg_args, arg_loc, macros)

        else:
            raise ParseError(f"Invalid loop argument {node_name}", parseLocation(node))

    return loop


