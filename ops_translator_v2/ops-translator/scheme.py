from __future__ import annotations

from pathlib import Path 
from typing import List, Optional, Set, Tuple

from jinja2 import Environment

import ops
from language import Lang
from store import Application, Program
from target import Target
from util import Findable
from util import KernelProcess


class Scheme(Findable):
    lang: Lang
    target: Target

    loop_host_template: Path
    master_kernel_template: Optional[Path]

    def __str__(self) -> str:
        return f"{self.lang.name}/{self.target.name}"
    
    def genLoopHost(
        self,
        include_dirs: Set[Path],
        defines: List[str],
        env: Environment,
        loop: ops.Loop,
        program: Program,
        app: Application,
        kernel_idx: int
    ) -> Tuple[str, str]:
        # Load the loop host template
        template = env.get_template(str(self.loop_host_template))
        extention = self.loop_host_template.suffixes[-2][1:]

        kernel_func = self.translateKernel(loop, program, app, kernel_idx)

        kp_obj = KernelProcess()

        kernel_body, args_list = kp_obj.get_kernel_body_and_arg_list(kernel_func)

        # Generalte source from the template
        return (
            template.render (
                ops=ops, lh=loop, kernel_func=kernel_func, kernel_idx=kernel_idx, kernel_body=kernel_body, args_list=args_list, lang=self.lang, target=self.target
            ),
            extention
        )

    def genMasterKernel(self, env: Environment, app: Application, user_types_file: Optional[Path]) -> Tuple[str, str]:
        if self.master_kernel_template is None:
            exit(f"No master kernel template registered for {self}")

        user_types = None
        if user_types_file is not None:
            user_types = user_types_file.read_text()

        # Load the kernel master template
        template = env.get_template(str(self.master_kernel_template))

        extension = self.master_kernel_template.suffixes[-2][1:]
        name = f"{self.target.name}_kernels.{extension}"

        # Generate source from the template
        return template.render(ops=ops, app=app, lang=self.lang, target=self.target, user_types=user_types), name      

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