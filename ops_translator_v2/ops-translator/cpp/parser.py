import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

from clang.cindex import Cursor, CursorKind, TranslationUnit, TypeKind, conf

import ops
from store import Function, Location, ParseError, Program, Type
from util import safeFind #TODO: implement safe find

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

    program.entities.append(function)


def parseLocation(node: Cursor) -> Location:
    return Location(node.location.file.name, node.location.line, node.location.column)


def parseLoops(translation_unit: TranslationUnit, program: Program) -> None:
    macros: Dict[Location, str] = {}
    nodes: List[Cursor] = []

    for node in translation_unit.cursor.get_children():

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
                parseCall(child, macros, program)

    return program


def parseUnexposedFunction(node: Cursor) -> Union[Tuple[str, List[Cursor]], None]:
    args = []
    # child_string=""
    for child in node.get_children():
        args.append(child)

    if len(args) == 0:
        return None

    first_child = args.pop(0)

    if first_child.kind == CursorKind.DECL_REF_EXPR:
        name = list(first_child.get_tokens())[0].spelling
    else:
        return None

    return (name, args)


def parseCall(node: Cursor, macros: Dict[Location, str], program: Program) -> None:

    if parseUnexposedFunction(node) == None:
        return
    else:
        (name, args) = parseUnexposedFunction(node)

    loc = parseLocation(node)

    if name == "ops_decl_const":
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
        "int": ops.Int(True, 32),
        "uint": ops.Int(False, 32),
        "ll": ops.Int(True, 64),
        "ull": ops.Int(False, 64),
        "float": ops.Float(32),
        "double": ops.Float(64),
        "bool": ops.Bool()
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
    name = parseStringLit(args[1])
    dim = parseIntLiteral(args[3])
    block = parseBlock(args[2], dim)
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


