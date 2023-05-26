from pathlib import Path

import cpp.translator.kernels as ctk
import ops
from language import Lang
from scheme import Scheme
from store import Application, ParseError, Program
from target import Target

class CppMPIOpenMP(Scheme):
    lang = Lang.find("cpp")
    target = Target.find("mpi_openmp")

    const_template = None
    loop_host_template = Path("cpp/mpi_openmp/loop_host.cpp.j2")
    master_kernel_template = Path("cpp/mpi_openmp/master_kernel.cpp.j2")

    def translateKernel(
        self, 
        loop: ops.Loop, 
        program: Program, 
        app: Application, 
        kernel_idx: int
    ) -> str:
        kernel_entities = app.findEntities(loop.kernel, program)

        if len(kernel_entities) == 0:
            raise ParseError(f"Unable to find kernel: {loop.kernel}")

        extracted_entities = ctk.extractDependancies(kernel_entities, app)
        return ctk.writeSource(extracted_entities)

Scheme.register(CppMPIOpenMP)
