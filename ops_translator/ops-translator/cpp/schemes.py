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


class CppHLS(Scheme):
    lang = Lang.find("cpp")
    target = Target.find("hls")    
    loop_host_template = Path("cpp/hls/loop_hls.cpp.j2")
    master_kernel_template = Path("cpp/hls/master_kernel.cpp.j2")
    
    loop_kernel_extension = "cpp"
    master_kernel_extension = "cpp"
    
    def translateKernel(
        self, 
        loop: ops.Loop, 
        program: Program, 
        app: Application, 
        kernel_idx: int
    ) -> str:
                
        print ("Range of loop: ", str(loop.range))
        
        kernel_entities = app.findEntities(loop.kernel, program)

        if len(kernel_entities) == 0:
            raise ParseError(f"Unable to find kernel: {loop.kernel}")

        print ("Found loop entity: ", kernel_entities[0])
        
        extracted_entities = ctk.extractDependancies(kernel_entities, app)
        return ctk.writeSource(extracted_entities)


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


Scheme.register(CppMPIOpenMP)
Scheme.register(CppCuda)
Scheme.register(CppHip)
<<<<<<< HEAD:ops_translator_v2/ops-translator/cpp/schemes.py
Scheme.register(CppHLS)
=======


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


class CppOpenACC(Scheme):
    lang = Lang.find("cpp")
    target = Target.find("openacc")

    const_template = None
    loop_host_template = Path("cpp/openacc/loop_host.cpp.j2")
    master_kernel_template = Path("cpp/openacc/master_kernel.cpp.j2")

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

Scheme.register(CppOpenACC)


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
>>>>>>> 8deae066f33d4c1523d2d0a45abe5ca420aa23b6:ops_translator/ops-translator/cpp/schemes.py
