from __future__ import annotations

from pathlib import Path 
from typing import List, Optional, Set, Tuple

from jinja2 import Environment

import ops
from language import Lang
from store import Application, Program
from target import Target
from util import sycl_set_flat_parallel
from util import extract_intrinsic_functions
from util import extract_arglist_fortran
from util import Findable
from util import KernelProcess
from abc import abstractmethod

class Scheme(Findable):
    lang: Lang
    target: Target

    loop_host_template: Path
    loop_host_f2c_template: Optional[Path]
    master_kernel_template: Optional[Path]

    def __str__(self) -> str:
        return f"{self.lang.name}/{self.target.name}: \n \
            lang_details: {self.lang}, target_details: {self.target}"
    
    def optimize(self, program: Program, app: Application) -> None:
        return None
    
    def genLoopDevice(
        self,
        env: Environment,
        loop: ops.Loop,
        program: Program,
        app: Application,
        config: dict,
    ) -> List[Tuple[str, str]]:
            return None
    
    def genIterLoopDevice(
        self,
        env: Environment,
        iterLoop: ops.IterLoop,
        program: Program,
        app: Application,
        config: dict
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
    ) -> Tuple[str, str, str]:
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
            intrinsic_funcs = ""

        elif (self.lang.name == "Fortran"):
            kernel_body = None
            consts_in_kernel = None
            const_dims = None
            args_list = extract_arglist_fortran(kernel_func)
            flat_parallel = None
            ops_cpu = None
            intrinsic_funcs = extract_intrinsic_functions(kernel_func)

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
                intrinsic_funcs=intrinsic_funcs,
                lang=self.lang,
                target=self.target,
                soa_set=force_soa,
                flat_parallel=flat_parallel,
                ops_cpu=ops_cpu
            ),
            self.loop_kernel_extension, kernel_func
        )


    def genF2CLoopHost(
        self,
        include_dirs: Set[Path],
        defines: List[str],
        env: Environment,
        loop: ops.Loop,
        program: Program,
        app: Application,
        kernel_idx: int,
        force_soa: bool,
        kernel_func: str
    ) -> Tuple[str, str]:

        template = env.get_template(str(self.loop_host_f2c_template))

        kp_obj = KernelProcess()
        kernel_func = kp_obj.clean_kernel_func_text(kernel_func)

        if(self.target.name == "f2c_cuda"):
            kernel_func = kp_obj.cuda_complex_numbers(kernel_func)

        if(self.target.name == "f2c_cuda" or self.target.name == "f2c_hip"):
            kernel_func = kp_obj.comment_stdcout(kernel_func)

        #TODO : Complex arguments in HIP

        kernel_body, args_list = kp_obj.get_kernel_body_and_arg_list(kernel_func)
        consts_in_kernel = None
        const_dims = None
        flat_parallel = None
        ops_cpu = None
        intrinsic_funcs = ""

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
                intrinsic_funcs=intrinsic_funcs,
                lang=self.lang,
                target=self.target,
                soa_set=force_soa,
                flat_parallel=flat_parallel,
                ops_cpu=ops_cpu
            ),
            self.loop_kernel_f2c_extension
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

        const_c_type = []

        if(self.target.name == "f2c_mpi_openmp" or self.target.name == "f2c_cuda" or self.target.name == "f2c_hip"):
            for const in app.consts():
                const_f90_type = str(const.typ).lower().strip()
                const_f90_type.replace(" ", "")
                if const_f90_type=="integer" or const_f90_type=="integer(4)" or const_f90_type=="integer(kind=4)":
                    const_c_type.append("int")
                elif const_f90_type=="integer(8)" or const_f90_type=="integer(kind=8)":
                    const_c_type.append("int64_t")
                elif const_f90_type=="real(4)" or const_f90_type=="real(kind=4)":
                    const_c_type.append("float")
                elif const_f90_type=="real" or const_f90_type=="real(8)" or const_f90_type=="real(kind=8)":
                    const_c_type.append("double")

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
                const_c_type=const_c_type,
                outerloop_enbl=outerloop_enbl
            ),
            name
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
        pass
            
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
