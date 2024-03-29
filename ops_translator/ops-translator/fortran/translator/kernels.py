from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import fparser.two.Fortran2003 as f2003
import fparser.two.utils as fpu

import fortran.util as fu
import ops as OPS
from language import Lang
from ops import OpsError
from store import Application, Entity, Function
from util import find, safeFind

def extractDependencies(
    entities: List[Entity], app: Application, scope: List[str] = []
) -> Tuple[List[Entity], List[str]]:
    unprocessed_entities = list(entities)
    extracted_entities = []
    unknown_entities = []

    while len(unprocessed_entities) > 0:
        entity = unprocessed_entities.pop(0)

        if safeFind(extracted_entities, lambda e: e == entity):
            continue

        for dependency in entity.depends:
            dependency_entities = app.findEntities(dependency, entity.program, scope)  # TODO: Loop scope

            if len(dependency_entities) == 0:
                unknown_entities.append(dependency)
            else:
                unprocessed_entities.extend(dependency_entities)

        if not safeFind(entities, lambda e: e == entity):
            extracted_entities.insert(0, entity)

    return extracted_entities, unknown_entities


# TODO: types
def renameEntities(entities: List[Entity], replacement: Callable[[str], str]) -> None:
    for entity in entities:
        new_name = replacement(entity.name)
        renameFunctionDefinition(entity, new_name)

        for entity2 in entities:
            renameFunctionCalls(entity2, entity.name, new_name)


def renameFunctionDefinition(entity: Entity, replacement: str) -> None:
    subroutine_statement = fpu.get_child(entity.ast, f2003.Subroutine_Stmt)
    kernel_name = fpu.get_child(subroutine_statement, f2003.Name)

    kernel_name.string = replacement


def renameFunctionCalls(entity: Entity, name: str, replacement: str) -> None:
    for node in fpu.walk(entity.ast, f2003.Call_Stmt):
        name_node = fpu.get_child(node, f2003.Name)

        if name_node.string.lower() == name:
            name_node.string = replacement


def renameConsts(lang: Lang, entities: List[Entity], app: Application, replacement: Callable[[str], str]) -> None:
    const_ptrs = app.constPtrs()

    for entity in entities:
        for name in fpu.walk(entity.ast, f2003.Name):
            if name.string.lower() in const_ptrs and name.string.lower() not in entity.parameters:
                name.string = replacement(name.string.lower())


def removeExternals(func: Function) -> None:
    for spec in fpu.walk(func.ast, f2003.Specification_Part):
        content = list(spec.content)
        content = filter(lambda n: not isinstance(n, f2003.External_Stmt), content)
        spec.content = list(content)


def replaceChild(node: f2003.Base, index: int, replacement: Any) -> None:
    children = []
    use_tuple = False
    use_content = False

    if hasattr(replacement, "parent"):
        replacement.parent = node

    if getattr(node, "items", None) is not None:
        if isinstance(node.items, tuple):
            use_tuple = True

        children = list(node.items)
    else:
        if isinstance(node.content, tuple):
            use_tuple = True

        children = list(node.content)
        use_content = True

    children[index] = replacement

    if use_tuple:
        children = tuple(children)

    if not use_content:
        node.items = children
    else:
        node.content = children


def replaceNodes(node: Any, match: Callable[[f2003.Base], bool], replacement: f2003.Base) -> Optional[Any]:
    if isinstance(node, tuple) or isinstance(node, list):
        children = list(node)

        for i in range(len(children)):
            if children[i] is None:
                continue

            child_replacement = replaceNodes(children[i], match, replacement)
            if child_replacement is not None:
                children[i] = child_replacement

        if isinstance(node, tuple):
            return tuple(children)
        else:
            return children

    if not isinstance(node, f2003.Base):
        return None

    if match(node):
        return replacement

    for i in range(len(node.children)):
        if node.children[i] is None:
            continue

        child_replacement = replaceNodes(node.children[i], match, replacement)

        if child_replacement is not None:
            replaceChild(node, i, child_replacement)

    return None


def writeSource(entities: List[Entity], prologue: Optional[str] = None) -> str:
    if len(entities) == 0:
        return ""

    source = (prologue or "") + str(entities[-1].ast)
    for entity in reversed(entities[:-1]):
        source = source + "\n\n" + (prologue or "") + str(entity.ast)

    return source
