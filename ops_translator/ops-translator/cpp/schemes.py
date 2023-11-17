from pathlib import Path

import cpp.translator.kernels as ctk
import ops
from language import Lang
from scheme import Scheme
from store import Application, ParseError, Program
from target import Target
from jinja2 import Environment
from typing import List, Tuple
from util import KernelProcess

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
    common_config_template = Path("cpp/hls/common_config_dev_hls.hpp.j2")
    loop_device_inc_template = Path("cpp/hls/loop_dev_inc_hls.hpp.j2")
    loop_device_src_template = Path("cpp/hls/loop_dev_src_hls.cpp.j2")
    loop_datamover_inc_template = Path("cpp/hls/datamover_dev_inc_hls.hpp.j2")
    loop_datamover_src_template = Path("cpp/hls/datamover_dev_src_hls.cpp.j2")
    
    loop_kernel_extension = "hpp"
    master_kernel_extension = "hpp"
    common_config_extension = "hpp"
    loop_device_inc_extension = "hpp"
    loop_device_src_extension = "cpp"
    loop_datamover_inc_extension = "hpp"
    loop_datamover_src_extension = "cpp"
    
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

    def genLoopDevice(
        self,
        env: Environment,
        loop: ops.Loop,
        program: Program,
        app: Application,
        config: dict,
        kernel_idx: int
    ) -> List[Tuple[str, str]]:
        
        #load datamover_templates
        datamover_inc_template = env.get_template(str(self.loop_datamover_inc_template))
        datamover_src_template = env.get_template(str(self.loop_datamover_src_template))
        kernel_inc_template = env.get_template(str(self.loop_device_inc_template))
        kernel_src_template = env.get_template(str(self.loop_device_src_template))
        
        kernel_processor = KernelProcess()
        
        kernel_func = self.translateKernel(loop, program, app, kernel_idx)
        kernel_func = kernel_processor.clean_kernel_func_text(kernel_func)
        
        return (
            [(datamover_inc_template.render(
                lh=loop),
            self.loop_datamover_inc_extension),
            (datamover_src_template.render(
                lh=loop,
                config=config), 
            self.loop_datamover_src_extension),
            (kernel_inc_template.render(
                 lh=loop,
                 kernel_func=kernel_func,
                 prog=program,
                 config=config
                 ),self.loop_device_inc_extension),
            (kernel_src_template.render(
                lh=loop,
                kernel_func=kernel_func,
                prog=program,
                config=config),
            self.loop_device_src_extension)]
        )
    
    def genConfigDevice(
        self,
        env: Environment,
        config: dict,
    ) -> Tuple[str, str]:
        
        template = env.get_template(str(self.common_config_template))     
        return (
            template.render(
                config=config
            ), self.common_config_extension
        ) 

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
Scheme.register(CppHLS)


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
