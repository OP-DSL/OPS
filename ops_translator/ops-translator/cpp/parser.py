import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any

from clang.cindex import Cursor, CursorKind, TranslationUnit, TypeKind, conf

import ops
from store import Function, Location, ParseError, Program, Type
from util import safeFind #TODO: implement safe find
import logging
from functools import cmp_to_key

def parseMeta(node: Cursor, program: Program) -> None:
    if node.kind == CursorKind.TYPE_REF:
        parseTypeRef(node, program)

    if node.kind == CursorKind.FUNCTION_DECL:
        parseFunction(node, program) #TODO

    for child in node.get_children():
        parseMeta(child, program) 


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

# def parseDats(translation_unit: TranslationUnit, program: Program) -> None:
#     return None

def parseLoops(translation_unit: TranslationUnit, program: Program) -> None:
    macros: Dict[Location, str] = {}
    nodes: List[Cursor] = []
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
 
            if child.kind.is_unexposed():
                parseCallUnexposed(child, macros, program)
                
            elif child.kind in [CursorKind.VAR_DECL, CursorKind.BINARY_OPERATOR]:
                parseVariableDeclaration(child, macros, program)
                
    return program


def parseVariableDeclaration(node: Cursor, macros: Dict[Location, str], program: Program) -> None:
    children = []

    if (node.kind == CursorKind.VAR_DECL):

        for child in node.get_children():
            children.append(child)
           
        if len(children) < 2:
            return
        var_name = node.spelling
        rval_expr = children[1]
            
    elif (node.kind == CursorKind.BINARY_OPERATOR):
        for child in node.get_children():
            children.append(child)
        
        if len(children) < 2:
            return
        
        var_name = children[0].spelling
        rval_expr = children[1]
    
    logging.debug("Variable detected: %s", var_name)
    
    if rval_expr.kind == CursorKind.CALL_EXPR and rval_expr.spelling == "ops_decl_stencil":
        out = parseFunctionCall(children[1])
        
        if not out:
            return
        
        (name, args) = out
        if name != "ops_decl_stencil":
            ParseError("Expected ops_decl stencil as RHS expression", parseLocation(node))
            
        parseStencil(var_name, args, parseLocation(node), macros, program)
      
        
def parseFunctionCall(node: Cursor) -> Union[Tuple[str, List[Cursor]], None]:
    args = []
    if node.kind != CursorKind.CALL_EXPR:
        return None
    
    for arg in node.get_arguments():
        args.append(arg)
    
    name = node.spelling

    logging.debug("Detected function: %s, args: %s", name, args)
        
    return (name, args)


def parseCall(node: Cursor, macros: Dict[Location, str], program: Program) -> None:
    out = parseFunctionCall(node)
    
    if out == None:
        return

    (name, args) = out  
    loc = parseLocation(node)
    
    if name == "ops_decl_stencil":
        if len(args) != 4:
            raise ParseError("ops_decl_stencil need 4 arguments", parseLocation(node))
        
        print(f"node {node}: {node.spelling}: type: {node.kind}")
        for arg in args: 
            if arg.kind == CursorKind.UNEXPOSED_EXPR and arg.get_definition():
                print(f"|-- argument {arg.spelling}: type: {arg.kind}, definition: {arg.get_definition().kind}, defLoc:{str(parseLocation(arg.get_definition()))}")
            else:
                print(f"|-- argument {arg.spelling}: type: {arg.kind}")
        
def pointsToArray(points: List[ops.Point], ndim: int) -> List[int]:
    array = []
    for point in points:
        for i in range(ndim):
            array.append(point[i])    
    return array

def pointCompare(point1: ops.Point, point2: ops.Point) -> int:
    
    if point1.z != point2.z:
        return point1.z - point2.z
    elif point1.y != point2.y:
        return point1.y - point2.y
    else:
        return point1.x - point2.y

def arrayToPoints(npoints: int, ndim: int, array: List[int]) -> List[ops.Point]:
    
    points = []
    if len(array) != npoints * ndim :
        raise ParseError(f"Missmatch of parsed array with the stencil specification. Array: {array}, npoints: {npoints}, ndim: {ndim}")
    
    for i in range(npoints):
        point = []
        for j in range(ndim):
            logging.debug(f"accessing: {array[i*ndim + j]}")
            point.append(array[i*ndim + j])
        points.append(ops.Point(point))
    
    return points

def stencilPointsSort(npoints: int, ndim: int, array: List[int])-> List[ops.Point]: 
    
    points = arrayToPoints(npoints, ndim, array)
    logging.debug(f"Points before sort: {points}")
    sorted_points = sorted(points, key=cmp_to_key(pointCompare))
    logging.debug(f"Points after sort: {sorted_points}")
    return sorted_points

def parseStencil(name: str, args: List[Cursor], loc: Location, macros: Dict[Location, str], program: Program) -> Union[ops.Stencil, None]:
    
    if len(args) != 4:
        return None
    
    ndim = parseIntLiteral(args[0])
    npoints = parseIntLiteral(args[1])
    array = stencilPointsSort(npoints, ndim, parseArrayIntLit(args[2]))
         
    logging.info("Stencil found - name:%s ndim: %d, npoints: %d, array: %s", name, ndim, npoints, array)

    program.stencils.append(ops.Stencil(len(program.stencils), ndim, name, npoints, array))   
 

    
def parseArrayIntLit(node: Cursor)->List[int]:
    logging.debug("array: %s, array_decl: %s, type:%s, loc:%s", node, node.get_definition(), 
            node.get_definition().kind, parseLocation(node.get_definition()))
     
    first_child = decend(node.get_definition())
    
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
        loop = parseLoop(args, loc, macros)
        program.loops.append(loop)

        if program.ndim == None:
            program.ndim = loop.ndim
        elif program.ndim < loop.ndim:
            program.ndim = loop.ndim
    
        
def decend(node: Cursor) -> Optional[Cursor]:
    return next(node.get_children(), None)


def parseStringLit(node: Cursor) -> str:
    if node.kind == CursorKind.UNEXPOSED_EXPR:
        node = decend(node)
        if node.kind != CursorKind.STRING_LITERAL:
            raise ParseError("Expected string literal")

    elif node.kind != CursorKind.STRING_LITERAL:
        raise ParseError("Expected string literal")

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
        raise ParseError("Invalid node type " + str(node.kind), parseLocation(node))


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
        node = decend(node)

    if node.kind == CursorKind.UNARY_OPERATOR and next(node.get_tokens()).spelling in ("&", "*"):
        node = decend(node)

    if node.kind == CursorKind.GNU_NULL_EXPR:
        raise ParseError("Expected identifier, found NULL", parseLocation(node))

    if node.kind != CursorKind.DECL_REF_EXPR:
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

    dat_ptr = parseIdentifier(args[0])
    dim = parseIntLiteral(args[1])
    stencil_ptr = parseIdentifier(args[2])
    dat_typ, dat_soa = parseType(parseStringLit(args[3]), loc)
    access_type = parseAccessType(args[4], loc, macros)

    loop.addArgDat(loc, dat_ptr, dim, dat_typ, dat_soa, stencil_ptr, access_type, True)


def parseArgDatOpt(loop: ops.Loop, args: List[Cursor], loc: Location, macros: Dict[Location, str]) -> None:
    if len(args) != 6:
        raise ParseError(f"Incorrect number({len(args)}) of args passed to ops_arg_dat_opt", loc)

    dat_ptr = parseIdentifier(args[0])
    dim = parseIntLiteral(args[1])
    stencil_ptr = parseIdentifier(args[2])
    dat_typ, dat_soa = parseType(parseStringLit(args[3]), loc)
    access_type = parseAccessType(args[4], loc, macros)
    opt = parseIntExpression(args[5])

    loop.addArgDat(loc, dat_ptr, dim, dat_typ, dat_soa, stencil_ptr, access_type, bool(opt))


def parseArgReduce(loop: ops.Loop, args: List[Cursor], loc: Location, macros: Dict[Location, str]) -> None:
    if len(args) != 4:
        raise ParseError(f"Incorrect number of args passed to ops_arg_reduce: {len(args)}", loc)

    reduct_handle_ptr = parseIdentifier(args[0])
    dim = parseIntLiteral(args[1])
    typ, soa = parseType(parseStringLit(args[2]), loc)
    access_type = parseAccessType(args[3], loc, macros)

    loop.addArgReduce(loc, reduct_handle_ptr, dim, typ, access_type)

def parseBlock(node: Cursor, dim: int) -> ops.Block:
    ptr = parseIdentifier(node)
    loc = parseLocation(node)
    return ops.Block(loc, ptr, dim)


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


def parseLoop(args: List[Cursor], loc: Location, macros: Dict[Location, str]) -> ops.Loop:
    if len(args) < 6:
        raise ParseError("Incorrect number of args passed to ops_par_loop")

    kernel = parseIdentifier(args[0])
    name   = parseStringLit(args[1])
    dim    = parseIntLiteral(args[3])
    block  = parseIdentifier(args[2])
    range = parseRange(args[4], dim)

    loop = ops.Loop(loc, kernel, block, range, dim)

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


