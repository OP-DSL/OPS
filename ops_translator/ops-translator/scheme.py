from __future__ import annotations

from pathlib import Path 
from typing import List, Optional, Set, Tuple

from jinja2 import Environment

import ops
from language import Lang
from store import Application, Program
from target import Target
from util import sycl_set_flat_parallel
from util import Findable
from util import KernelProcess


class Scheme(Findable):
    lang: Lang
    target: Target

    loop_host_template: Path
    master_kernel_template: Optional[Path]

    def __str__(self) -> str:
        return f"{self.lang.name}/{self.target.name}: \n \
            lang_details: {self.lang}, target_details: {self.target}"
    
    def genLoopDevice(
        self,
        env: Environment,
        loop: ops.Loop,
        program: Program,
        app: Application,
        config: dict,
    ) -> List[Tuple[str, str]]:
            return None
    
    def genConfigHost(
        self,
        env: Environment,
        config: dict,
        app: Application
    ) -> Tuple[str, str]:
        return None
    
    def genStencilDecl(
        self,
        env: Environment,
        config: dict,
        stencil: ops.Stencil
    ) -> Tuple[str, str]:
        return None
    
    def genConfigDevice(
        self,
        env: Environment,
        config: dict,
    ) -> Tuple[str, str]:
        return None
    
    
    def find_const_in_kernel(self, kernel_body: str, global_consts: List[ops.Const]) -> List[ops.Const]:
        selected_consts = []
        for const in global_consts:
            if const.name in kernel_body:
                selected_consts.append(const)
        
        return selected_consts
    
     
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
        # Load the loop host template
        template = env.get_template(str(self.loop_host_template))
        #extention = self.loop_host_template.suffixes[-2][1:]

        kernel_func = self.translateKernel(loop, program, app, kernel_idx)

        kp_obj = KernelProcess()
        if(self.lang.name == "C++"):
            kernel_func = kp_obj.clean_kernel_func_text(kernel_func)
        
            if(self.target.name == "cuda"):
                kernel_func = kp_obj.cuda_complex_numbers(kernel_func)

            consts_in_kernel = []
            if(self.target.name == "sycl"):
                kernel_func, consts_in_kernel = kp_obj.sycl_kernel_func_text(kernel_func, app.consts())

            if(self.target.name == "hls"):
                consts_in_kernel = self.find_const_in_kernel(kernel_func, program.consts)
            #TODO : Complex arguments in HIP
            const_dims = []
            if(self.target.name == "openacc"):
                consts_in_kernel, const_dims = kp_obj.openacc_get_const_names_and_dim(kernel_func, app.consts())

            kernel_body, args_list = kp_obj.get_kernel_body_and_arg_list(kernel_func)
            flat_parallel, ops_cpu = sycl_set_flat_parallel(loop.has_reduction)

        elif (self.lang.name == "Fortran"):
            kernel_body = None
            consts_in_kernel = None
            const_dims = None
            args_list = None
            flat_parallel = None
            ops_cpu = None

        # Generalte source from the template
        return (
            template.render (
                ops=ops,
                lh=loop,
                kernel_func=kernel_func,
                kernel_idx=kernel_idx,
                kernel_body=kernel_body,
                consts_in_kernel=consts_in_kernel,
                const_dims=const_dims,
                args_list=args_list,
                lang=self.lang,
                target=self.target,
                soa_set=force_soa,
                flat_parallel=flat_parallel,
                ops_cpu=ops_cpu
            ),
            self.loop_kernel_extension
        )

    def genMasterKernel(self, env: Environment, app: Application, user_types_file: Optional[Path], target_config: dict, force_soa: bool, outerloop_enbl: bool = False) -> Tuple[str, str]:
        if self.master_kernel_template is None:
            exit(f"No master kernel template registered for {self}")

        user_types = None
        if user_types_file is not None:
            user_types = user_types_file.read_text()

        # Load the kernel master template
        template = env.get_template(str(self.master_kernel_template))

        #extension = self.master_kernel_template.suffixes[-2][1:]
        #name = f"{self.target.name}_kernels.{extension}"
        name = f"{self.target.name}_kernels.{self.master_kernel_extension}"

        # Generate source from the template
        return (
            template.render(
                ops=ops,
                app=app,
                lang=self.lang,
                target=self.target,
                user_types=user_types,
                include_extension=self.master_kernel_extension,
                target_config=target_config,
                soa_set=force_soa,
                outerloop_enbl=outerloop_enbl
            ),
            name
        )
        
    def translateKernel(
        self,
        loop: ops.Loop,
        program: Program,
        app: Application,
        kernel_idx: int
    ) -> str:
        pass

    def matches(self, key: Tuple[Lang, Target]) -> bool:
        return self.lang == key[0] and self.target == key[1]
