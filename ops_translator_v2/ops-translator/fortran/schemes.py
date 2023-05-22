import copy
from pathlib import Path
from typing import Dict, Any

import fortran.translator.kernels as ftk
import ops as OPS
from language import Lang
from scheme import Scheme
from store import Application, ParseError, Program
from target import Target
from util import find

class FortranSeq(Scheme):
    lang = Lang.find("F90")
    target = Target.find("seq")

    fallback = None

    consts_template = None
#   consts_template = Path("fortran/seq/consts.F90.jinja")
    loop_host_template = None
#   loop_host_template = Path("fortran/seq/loop_host.F90.jinja")
    master_kernel_template = None    
#   master_kernel_template = Path("fortran/seq/master_kernel.F90.jinja")


    def translateKernel(
        self,
        loop: OPS.Loop,
        program: Program,
        app: Application,
        config: Dict[str, Any],
        kernel_idx: int,
    ) -> str:
        kernel_entities = app.findEntities(loop.kernel, program, [])  # TODO: Loop scope
        if len(kernel_entities) == 0:
            raise ParseError(f"unable to find kernel function: {loop.kernel}")

        dependencies, _ = ftk.extractDependencies(kernel_entities, app, [])  # TODO: Loop scope

        kernel_entities = copy.deepcopy(kernel_entities)
        dependencies = copy.deepcopy(dependencies)

#        if self.lang.user_consts_module is None:
#            ftk.renameConsts(self.lang, kernel_entities + dependencies, app, lambda const: f"op2_const_{const}")

        return ftk.writeSource(kernel_entities + dependencies)


Scheme.register(FortranSeq)
