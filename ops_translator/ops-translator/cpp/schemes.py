from pathlib import Path

import cpp.translator.kernels as ctk
import ops
from language import Lang
from scheme import Scheme
from store import Application, ParseError, Program
from target import Target
from jinja2 import Environment
from typing import List, Tuple, Set, Union
from util import KernelProcess
import re
import logging

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
    stencil_device_template = Path("cpp/hls/stencil_dev_hls.hpp.j2")
    
    loop_kernel_extension = "hpp"
    master_kernel_extension = "hpp"
    common_config_extension = "hpp"
    loop_device_inc_extension = "hpp"
    loop_device_src_extension = "cpp"
    loop_datamover_inc_extension = "hpp"
    loop_datamover_src_extension = "cpp"
    stencil_device_extension = "hpp"
    
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
    
    # def genLoopHost(
    #     self,
    #     include_dirs: Set[Path],
    #     defines: List[str],
    #     env: Environment,
    #     loop: ops.Loop,
    #     program: Program,
    #     app: Application,
    #     kernel_idx: int,
    #     force_soa: bool
    # ) -> Tuple[str, str]:
    #     template = env.get_template(str(self.loop_host_template))
    #     kernel_func = self.translateKernel(loop, program, app, kernel_idx)
        
    #     kp_ = KernelProcess()
        
    #     kernel_func = kp_.clean_kernel_func_text(kernel_func)
    #     kernel_body, args_list = kp_.get_kernel_body_and_arg_list(kernel_func)
        
    #     return (
    #         template.render (
    #             ops=ops,
    #             lh=loop,
    #             kernel_func=kernel_func,
    #             kernel_idx=kernel_idx,
    #             kernel_body=kernel_body,
    #             args_list=args_list,
    #             lang=self.lang,
    #             target=self.target,
    #             soa_set=force_soa
    #         ),
    #         self.loop_kernel_extension
    #     )
    def hls_replace_accessors_with_registers(self, kernel_body: str, kernel_args: List[str], loop: ops.Loop, prog: Program):
        logging.getLogger(__name__)
        assert len(kernel_args) == len(loop.args), f"kernel arguments of kernel {loop.kernel} count mismatch with loop"
        logging.debug("starting accessor replacer")
        
        while(True):

            if loop.ndim == 1:
                match_string = "[A-Za-z0-9_]+\s*\(\s*-?\s*\d+\s*\)"
            elif loop.ndim == 2:
                match_string = "[A-Za-z0-9_]+\s*\(\s*-?\s*\d+\s*,\s*-?\s*\d+\s*\)"
            else:
                match_string = "[A-Za-z0-9_]+\s*\(\s*-?\s*\d+\s*,\s*-?\s*\d+\s*,\s*-?\s*\d+\s*\)"
            logging.debug(f"matching string: {match_string}")
            match = re.search(match_string, kernel_body)
            if not match:
                break
            
            name = re.search(r"[A-Za-z0-9_]+", match.group(0))
            access_raw_indices = re.search(r"\(.*\)",match.group(0))
            arg_idx = kernel_args.index(name.group(0))
            logging.debug("match found: %s, name: %s, access_raw_indices: %s", match.group(0), name.group(0), access_raw_indices.group(0))

            if not isinstance(loop.args[arg_idx], ops.ArgDat):
                logging.error("Transltor failed finding matchin argument for: %s in loop: %s data", match.group(0), loop.kernel)
                raise ParseError(f"Translator failed finding relevent Dat argument of loop{loop.kernel}")
                
            stencil_ptr = loop.args[arg_idx].stencil_ptr
            stencil = prog.findStencil(stencil_ptr)
            logging.debug("Matching stencil: %s", str(stencil))
            
            if not stencil:
                raise ParseError(f"Translator failed finding relevent stencil: {stencil_ptr} in program: {str(prog.path)}")
            try:
                access_indices = ops.Point(list(eval(access_raw_indices.group(0))))
                access_indices = access_indices + stencil.base_point
                logging.debug(f"corrected access indice point: {access_indices}")
                
            except Exception as e:
                raise ParseError(f"Transaltor filed with error: {str(e)}")
                
            kernel_body = re.sub(match_string, f"reg_{arg_idx}_{stencil.points.index(access_indices)}", kernel_body, count = 1)
            
        return kernel_body
    
    def findConstInKernel(self, kernel_body: str, global_consts: List[ops.Const]) -> List[ops.Const]:
        selected_consts = []
        for const in global_consts:
            if const.name in kernel_body:
                selected_consts.append(const)
        
        return selected_consts
                
    def findKernelArgOfOpsIdx(self, kernel_args: List[str], loopArgs: List[ops.Arg])-> Union[str, None]:
        for i, arg in enumerate(loopArgs):
            if isinstance(arg, ops.ArgIdx):
                return kernel_args[i]
        return None
    
    def replaceIdxAccess(self, kernel_body: str, idx_arg_name: str)->str:
        new_kernel_body = re.sub(idx_arg_name, "indexConv.index", kernel_body)
        return new_kernel_body

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
        kernel_body, kernel_args = kernel_processor.get_kernel_body_and_arg_list(kernel_func)
        kernel_body = self.hls_replace_accessors_with_registers(kernel_body, kernel_args, loop, program)
        kernel_consts = self.findConstInKernel(kernel_body, program.consts)
        kernel_idx_arg_name = self.findKernelArgOfOpsIdx(kernel_args, loop.args)
        logging.debug("kernel_idx_arg_name: %s", kernel_idx_arg_name)
        
        if kernel_idx_arg_name:
            kernel_body = self.replaceIdxAccess(kernel_body, kernel_idx_arg_name)
        
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
                 kernel_body=kernel_body,
                 kernel_args=kernel_args,
                 prog=program,
                 consts=kernel_consts,
                 config=config
                 ),self.loop_device_inc_extension),
            (kernel_src_template.render(
                lh=loop,
                kernel_body=kernel_body,
                kernel_args=kernel_args,
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
    
    def genStencilDecl(
        self,
        env: Environment,
        config: dict,
        stencil: ops.Stencil,
    ) -> Tuple[str, str]:
        template = env.get_template(str(self.stencil_device_template))
        return (
            template.render(
                config = config,
                stencil = stencil
            ), self.stencil_device_extension
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
