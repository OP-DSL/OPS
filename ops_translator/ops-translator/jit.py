"""Support of `ops-translator --jit`: the table of kernels the build-time importer needs (jit_kernels.json).

For every unique kernel: the source file whose parse sees the kernel's definition (normally the application file
that contains the ops_par_loop calls; another file of the application when the kernel header is only included there;
for Fortran the kernel's .inc file), its language, the dimension of the block, and the distinct patterns of arguments
the kernel is called with (kind, access, element type, dimension). `kinds` is the kind of each parameter when all call sites agree.
"""
import json
from pathlib import Path

import ops


def _arg_pattern(loop, arg):
    if isinstance(arg, ops.ArgDat):
        dat = loop.dats[arg.dat_id]
        return {
            "kind": "dat",
            "access": arg.access_type.name,
            "type": str(dat.typ),
            "dim": dat.dim,
            "multigrid": bool(getattr(arg, "restrict", False) or getattr(arg, "prolong", False)),
            "strides": [int(s) for s in getattr(arg, "strides", [])],
        }
    if isinstance(arg, ops.ArgGbl):
        return {"kind": "gbl", "access": arg.access_type.name, "type": str(arg.typ), "dim": str(arg.dim)}
    if isinstance(arg, ops.ArgReduce):
        return {"kind": "reduce", "access": arg.access_type.name, "type": str(arg.typ), "dim": str(arg.dim)}
    if isinstance(arg, ops.ArgIdx):
        return {"kind": "idx"}
    return {"kind": "unknown"}


_IMPORTER_KIND = {"dat": "acc", "gbl": "gbl", "reduce": "red", "idx": "idx"}


def _definition_program(app, kernel, program):
    """The program (translation unit) in which the definition of `kernel` is visible, preferring the caller's."""
    if program.findEntities(kernel):
        return program
    found = app.findEntities(kernel, program)
    return found[0].program if found else program


_FORTRAN_SUFFIXES = {".f", ".f90", ".f95", ".f03", ".f08", ".for", ".cuf"}


def _is_fortran(program):
    return Path(program.path).suffix.lower() in _FORTRAN_SUFFIXES


def _fortran_kernel_file(loop, program):
    """The file that defines a Fortran kernel: `<name up to "kernel">kernel.inc`, as the Fortran schemes look it up."""
    name = loop.kernel[:loop.kernel.find("kernel")] + "kernel.inc"
    return Path(program.path).resolve().parent / name


def kernel_table(app):
    kernels = {}
    for loop, program in app.loops():
        if _is_fortran(program):
            source, language = _fortran_kernel_file(loop, program), "fortran"
        else:
            source, language = Path(_definition_program(app, loop.kernel, program).path).resolve(), "c++"
        entry = kernels.setdefault(
            loop.kernel,
            {"kernel": loop.kernel, "source": str(source), "language": language, "ndim": loop.ndim, "patterns": []},
        )
        pattern = [_arg_pattern(loop, arg) for arg in loop.args]
        if pattern not in entry["patterns"]:
            entry["patterns"].append(pattern)
    for entry in kernels.values():
        kinds = None
        for pattern in entry["patterns"]:
            these = [_IMPORTER_KIND.get(p["kind"]) for p in pattern]
            if kinds is None:
                kinds = these
            elif kinds != these:
                kinds = None
                break
        entry["kinds"] = kinds
    return {"jit_kernels_version": 1, "kernels": list(kernels.values())}


def write_kernel_table(app, path):
    with open(path, "w") as f:
        json.dump(kernel_table(app), f, indent=2)
        f.write("\n")
