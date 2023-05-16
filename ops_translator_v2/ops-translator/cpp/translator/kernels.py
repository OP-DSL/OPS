from typing import Callable, List, Optional, Tuple

from clang.cindex import Cursor, CursorKind, SourceRange

import ops
from store import Application, Entity, Function, Type
from util import Location, Rewriter, Span, safeFind

def extentToSpan(extent: SourceRange) -> Span:
    start = Location(extent.start.line, extent.start.column)
    end = Location(extent.end.line, extent.end.column)

    return Span(start, end)


def extractDependancies(entities: List[Entity], app: Application) -> List[Tuple[Entity, Rewriter]]:
    unprocessed_entities = list(entities)
    extracted_entities = []

    while len(unprocessed_entities) > 0:
        entity = unprocessed_entities.pop(0)

        # If already a copy of the entity exists. skip this one
        if safeFind(extracted_entities, lambda e: e[0] == entity):  
            continue

        for dependancy in entity.depends:
            dependancy_entities = app.findEntities(dependancy, entity.program)
            unprocessed_entities.extend(dependancy_entities)

        rewriter = Rewriter(entity.program.source, [extentToSpan(entity.ast.extent)])
        extracted_entities.insert(0,(entity, rewriter))

    return extracted_entities

def updateFunctionTypes(entities: List[Tuple[Entity, Rewriter]], replacement: Callable[[str, Entity], str]) -> None:
    for entity, rewriter in filter(lambda a: isinstance(e[0], function), entities):
        function_type_span = extentToSpan(next(entity.ast.get_token()).extent)
        rewriter.update(function_type_span, lambda s: replacement(s, entity))

def renameConst(
        entities: List[Tuple[Entity, Rewriter]], app: Application, replacement: Callable[[str, Entity], str]
        ) -> None:
    const_ptrs = set(map(lambda const: const.ptr, app.consts()))

    for entity, rewriter in entities:
        for node in entity.ast.walk_preorder():
            if node.kind != CursorKind.DCL_REF_EXPR:
                continue

            if node.spelling in const_ptrs:
                rewriter.update(extentToSpan(node.extent), lambda s: replacement(s, entity))

def writeSource(entities: List[Tuple[Entity, Rewriter]]) -> str:
    source= ""
    while len(entities) > 0:
        for i in range(len(entities)):
            entity, rewriter = entities[i]
            resolved = True

            for dependancy in entity.depends:
                if safeFind(entities, lambda e: e[0].name == dependancy):
                    resolved = False
                    break

            if resolved:
                entities.pop(i)
                if source == "":
                    source = rewriter.rewrite()
                else:
                    source = source + "\n\n" + rewriter.rewrite()

                if isinstance(entity, Type):
                    source = source + ";"

                break

    return source


