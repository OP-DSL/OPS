from pathlib import Path

import cpp.translator.kernels as ctk
import ops
from language import Lang
from scheme import Scheme
from store import Application, ParseError, Program
from target import Target
from jinja2 import Environment
from typing import List, Tuple, Set, Union, Optional
from util import KernelProcess
import re
import logging
from cpp import optimizer

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
    loop_host_kernelwrap_template = Path("cpp/hls/loop_kernelwrap.hpp.j2")
    loop_host_cpu_template = Path("cpp/hls/loop_host_cpu.hpp.j2")
    # loop_device_inc_template = Path("cpp/hls/loop_dev_inc_hls.hpp.j2")
    # loop_device_src_template = Path("cpp/hls/loop_dev_src_hls.cpp.j2")
    # loop_datamover_inc_template = Path("cpp/hls/datamover_dev_inc_hls.hpp.j2")
    # loop_datamover_src_template = Path("cpp/hls/datamover_dev_src_hls.cpp.j2")
    loop_device_PE_template = Path("cpp/hls/loop_dev_PE_hls_V2.hpp.j2")
    
    iterloop_datamover_inc_template = Path("cpp/hls/iter_loop_datamover_dev_inc_hls.hpp.j2")
    iterloop_datamover_src_template = Path("cpp/hls/iter_loop_datamover_dev_src_hls.cpp.j2")
    iterloop_device_inc_template = Path("cpp/hls/iter_loop_dev_inc_hls.hpp.j2")
    iterloop_device_src_template = Path("cpp/hls/iter_loop_dev_src_hls.cpp.j2")
    iterloop_host_kernelwrap_template = Path("cpp/hls/iter_loop_host_kernelwrap.hpp.j2")
    
    stencil_device_template = Path("cpp/hls/stencil_dev_hls.hpp.j2")
    master_kernel_template = Path("cpp/hls/master_kernel.cpp.j2")
    common_config_template = Path("cpp/hls/common_config_dev_hls.hpp.j2")
    host_config_template = Path("cpp/hls/xrt_config.cfg.j2")
    
    loop_kernel_extension = "hpp"
    master_kernel_extension = "hpp"
    common_config_extension = "hpp"
    host_config_extension = "cfg"
    iterloop_device_inc_extension = "hpp"
    iterloop_device_src_extension = "cpp"
    iterloop_datamover_inc_extension = "hpp"
    iterloop_datamover_src_extension = "cpp"
    iterloop_host_kernelwrap_extension = "hpp"
    loop_device_PE_extension = "hpp"
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
    
    def genLoopHost(
        self,
        include_dirs: Set[Path],
        defines: List[str],
        env: Environment,
        loop: ops.Loop,
        program: Program,
        app: Application,
        kernel_idx: int,
        force_soa: bool
    ) -> Tuple[str, str]:
        
        template = env.get_template(str(self.loop_host_cpu_template))
        kernel_func = self.translateKernel(loop, program, app, kernel_idx)
        
        kernel_processor = KernelProcess()
        
        kernel_func = self.translateKernel(loop, program, app, kernel_idx)
        kernel_func = kernel_processor.clean_kernel_func_text(kernel_func)
        kernel_body, kernel_args = kernel_processor.get_kernel_body_and_arg_list(kernel_func)
        kernel_body = self.hls_replace_accessors(kernel_body, kernel_args, loop, program, False)
        kernel_consts = self.find_const_in_kernel(kernel_body, program.consts)
        kernel_idx_arg_name = self.find_kernel_arg_of_ops_idx(kernel_args, loop.args)
        logging.debug("kernel_idx_arg_name: %s", kernel_idx_arg_name)
        
        if kernel_idx_arg_name:
            kernel_body = self.replace_idx_access(kernel_body, kernel_idx_arg_name, False)
        
        return (
            template.render (
                ops=ops,
                lh=loop,
                prog=program,
                kernel_func=kernel_func,
                kernel_idx=kernel_idx,
                kernel_body=kernel_body,
                kernel_args=kernel_args,
                consts=kernel_consts
            ),
            self.loop_kernel_extension
        )
    
    def genIterLoopHost(
            self,
            include_dirs: Set[Path],
            defines: List[str],
            env: Environment,
            iterloop: ops.IterLoop,
            program: Program,
            app: Application,
            kernel_idx: int,
            force_soa: bool,
            config: dict
    ) -> Tuple[str, str]:
        
        template = env.get_template(str(self.iterloop_host_kernelwrap_template))
        
        kernel_processor = KernelProcess()
        consts = []
        
        for kernel_idx, loop in enumerate(iterloop.getLoops()):
                kernel_func = self.translateKernel(loop, program, app, kernel_idx)
                kernel_func = kernel_processor.clean_kernel_func_text(kernel_func)
                kernel_body, kernel_args = kernel_processor.get_kernel_body_and_arg_list(kernel_func)
                kernel_body = self.hls_replace_accessors(kernel_body, kernel_args, loop, program)
                kernel_consts = self.find_const_in_kernel(kernel_body, program.consts)
                consts.extend(x for x in kernel_consts if x not in consts)

        return (
            template.render (
                ops=ops,
                ilh=iterloop,
                prog=program,
                ndim=program.ndim,
                consts=consts,
                config=config
            ),
            self.iterloop_host_kernelwrap_extension
        )
            
    def hls_replace_accessors(self, kernel_body: str, kernel_args: List[str], loop: ops.Loop, prog: Program, isReplaceWithReg = True):
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
                print(f"re search: {access_raw_indices.group(0)}, eval: {eval(access_raw_indices.group(0))}")
                if loop.ndim == 1:
                    access_indices = ops.Point(list([eval(access_raw_indices.group(0))]))
                else:
                    access_indices = ops.Point(list(eval(access_raw_indices.group(0))))
                access_indices = access_indices + stencil.base_point
                logging.debug(f"corrected access indice point: {access_indices}")
                
            except Exception as e:
                raise ParseError(f"Transaltor filed with error: {str(e)}")
                
            if isReplaceWithReg:
                kernel_body = re.sub(match_string, f"reg_{loop.args[arg_idx].dat_id}_{stencil.points.index(access_indices)}", kernel_body, count = 1)
            else:
                kernel_body = re.sub(match_string, f"arg{loop.args[arg_idx].dat_id}_{stencil.points.index(access_indices)}", kernel_body, count = 1)
        return kernel_body

    def generateWidenStencilandBufferDiscriptor(self, stencil: ops.Stencil, vector_factor: int) -> Tuple[ops.Stencil, ops.WindowBufferDiscriptor]:
        widen_points, point_to_widen_map = ops.computeWidenPoints(stencil.row_discriptors, vector_factor)
        
        print(f"widen points: {widen_points}, stencil: {stencil}")
        windows_buffers, chains = ops.windowBuffChainingAlgo(widen_points, stencil.dim)
        stencilSize = ops.getStencilSize(widen_points)
        row_discriptors = ops.genRowDiscriptors(widen_points, stencil.base_point)
        widen_stencil = ops.Stencil(stencil.id, stencil.dim, stencil.stencil_ptr, len(widen_points), widen_points, stencil.base_point, stencilSize, stencil.d_m, stencil.d_p, row_discriptors)
        
        return (ops.WindowBufferDiscriptor(widen_stencil, windows_buffers, chains, point_to_widen_map))
      
    def find_kernel_arg_of_ops_idx(self, kernel_args: List[str], loopArgs: List[ops.Arg])-> Union[str, None]:
        for i, arg in enumerate(loopArgs):
            if isinstance(arg, ops.ArgIdx):
                return kernel_args[i]
        return None
    
    def replace_idx_access(self, kernel_body: str, idx_arg_name: str, isHLS = True)->str:
        new_kernel_body = re.sub(idx_arg_name, "idx", kernel_body)
        return new_kernel_body

    def optimize(self, program: Program, app: Application):
        for iterLoop in program.outerloops:
            iterLoop.opt_df_graph = optimizer.ISLCopyDetection(iterLoop.df_graph, program, app, self).copy()
            
    def genIterLoopDevice(
        self,
        env: Environment,
        iterLoop: ops.IterLoop,
        program: Program,
        app: Application,
        config: dict
    ) -> List[Tuple[str, str]]:
        
        iterloop_datamover_inc_template = env.get_template(str(self.iterloop_datamover_inc_template))
        iterLoop_datamover_src_template = env.get_template(str(self.iterloop_datamover_src_template))
        iterLoop_kernel_inc_template = env.get_template(str(self.iterloop_device_inc_template))
        iterLoop_kernel_src_template = env.get_template(str(self.iterloop_device_src_template))
        
        kernel_processor = KernelProcess()
        consts = []
        consts_map = []
        for kernel_idx, loop in enumerate(iterLoop.getLoops()):
                kernel_func = self.translateKernel(loop, program, app, kernel_idx)
                kernel_func = kernel_processor.clean_kernel_func_text(kernel_func)
                kernel_body, kernel_args = kernel_processor.get_kernel_body_and_arg_list(kernel_func)
                kernel_body = self.hls_replace_accessors(kernel_body, kernel_args, loop, program)
                kernel_consts = self.find_const_in_kernel(kernel_body, program.consts)
                consts_map.append(kernel_consts)
                consts.extend(x for x in kernel_consts if x not in consts)
        
        return [(iterloop_datamover_inc_template.render(ilh=iterLoop, ndim=program.ndim), self.iterloop_datamover_inc_extension),
                (iterLoop_datamover_src_template.render(ilh=iterLoop, ndim=program.ndim, config=config), self.iterloop_datamover_src_extension),
                (iterLoop_kernel_inc_template.render(ilh=iterLoop, ndim=program.ndim, config=config, consts=consts), self.iterloop_device_inc_extension),
                (iterLoop_kernel_src_template.render(ilh=iterLoop, ndim=program.ndim, config=config, consts=consts, consts_map = consts_map), self.iterloop_device_src_extension)]
    
    def genLoopDevice(
        self,
        env: Environment,
        loop: ops.Loop,
        program: Program,
        app: Application,
        config: dict,
        kernel_idx: int,
        outerLoop: Optional[ops.IterLoop] = None
    ) -> List[Tuple[str, str]]:
        
        #load datamover_templates
        loop_PE_template = env.get_template(str(self.loop_device_PE_template))
        kernel_processor = KernelProcess()
        
        kernel_func = self.translateKernel(loop, program, app, kernel_idx)
        kernel_func = kernel_processor.clean_kernel_func_text(kernel_func)
        kernel_body, kernel_args = kernel_processor.get_kernel_body_and_arg_list(kernel_func)
        kernel_body = self.hls_replace_accessors(kernel_body, kernel_args, loop, program)
        kernel_consts = self.find_const_in_kernel(kernel_body, program.consts)
        kernel_idx_arg_name = self.find_kernel_arg_of_ops_idx(kernel_args, loop.args)
        logging.debug("kernel_idx_arg_name: %s", kernel_idx_arg_name)
        
        widen_stencil_desc = {}
        
        for stencil_ptr in loop.stencils:
            stencil = program.findStencil(stencil_ptr)
            
            if not stencil:
                raise ParseError("Failed to find stencil of a loop in the program")
            
            window_stencil_buff = self.generateWidenStencilandBufferDiscriptor(stencil, config["vector_factor"])
            widen_stencil_desc[stencil.stencil_ptr] = window_stencil_buff
            
        widen_read_stencil_desc = self.generateWidenStencilandBufferDiscriptor(loop.get_read_stencil(program), config["vector_factor"])
        
        
        if kernel_idx_arg_name:
            kernel_body = self.replace_idx_access(kernel_body, kernel_idx_arg_name)
        
        if outerLoop:
            isFullyMapped, datMap = kernel_processor.gen_local_dependancy_map(loop, outerLoop)
        else:
            isFullyMapped = False
            datMap = []
        return (
            [(loop_PE_template.render(
                 lh=loop,
                 kernel_body=kernel_body,
                 kernel_args=kernel_args,
                 prog=program,
                 consts=kernel_consts,
                 config=config,
                 isFullyMapped = isFullyMapped,
                 datMap = datMap,
                 is_arg_idx = (kernel_idx_arg_name != None),
                 ops=ops,
                 widen_stencil_disc=widen_stencil_desc,
                 widen_read_stencil_desc = widen_read_stencil_desc
                 ),self.loop_device_PE_extension)]
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
    
    def genConfigHost(
        self,
        env: Environment,
        config: dict,
        app: Application
    ) -> Tuple[str, str]:
        template = env.get_template(str(self.host_config_template))     
        return (
            template.render(
                config=config,
                app=app
            ), self.host_config_extension
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
