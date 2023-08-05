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

    loop_kernel_extension = "cpp"
    master_kernel_extension = "cpp"

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


class CppCuda(Scheme):
    lang = Lang.find("cpp")
    target = Target.find("cuda")

    const_template = None
    loop_host_template = Path("cpp/cuda/loop_host.cpp.j2")
    master_kernel_template = Path("cpp/cuda/master_kernel.cpp.j2")

    loop_kernel_extension = "cu"
    master_kernel_extension = "cu"

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

Scheme.register(CppCuda)


class CppHip(Scheme):
    lang = Lang.find("cpp")
    target = Target.find("hip")

    const_template = None
    loop_host_template = Path("cpp/cuda/loop_host.cpp.j2")
    master_kernel_template = Path("cpp/cuda/master_kernel.cpp.j2")

    loop_kernel_extension = "cpp"
    master_kernel_extension = "cpp"

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

Scheme.register(CppHip)


class CppOpenMPOffload(Scheme):
    lang = Lang.find("cpp")
    target = Target.find("openmp_offload")

    const_template = None
    loop_host_template = Path("cpp/openmp_offload/loop_host.cpp.j2")
    master_kernel_template = Path("cpp/openmp_offload/master_kernel.cpp.j2")

    loop_kernel_extension = "cpp"
    master_kernel_extension = "cpp"

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

Scheme.register(CppOpenMPOffload)


class CppSycl(Scheme):
    lang = Lang.find("cpp")
    target = Target.find("sycl")

    const_template = None
    loop_host_template = Path("cpp/sycl/loop_host.cpp.j2")
    master_kernel_template = Path("cpp/sycl/master_kernel.cpp.j2")

    loop_kernel_extension = "cpp"
    master_kernel_extension = "cpp"

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

Scheme.register(CppSycl)
