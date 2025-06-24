from __future__ import annotations

from typing import Any, Dict
from util import Findable

#TODO: Add documentaion (numpy style)
class Target(Findable):
    name: str
    kernel_translation: bool
    config: Dict[str, Any]

    def __str__(self) -> str:
        return self.name

    def __eq__(self, other) -> bool:
        return self.name == other.name if type(other) is type(self) else False

    def __hash__(self) -> int:
        return hash(self.name)

    def matches(self, key: str) -> bool:
        return self.name == key.lower()

class MPIOpenMP(Target):
    name = "mpi_openmp"
    suffix = "seq"
    kernel_translation = False
    config = {
        "grouped" : False, 
        "device" : 1
        }

class F2CMPIOpenMP(Target):
    name = "f2c_mpi_openmp"
    suffix = "f2c"
    kernel_translation = False
    config = {
        "grouped" : False,
        "device" : 2
        }

class Cuda(Target):
    name = "cuda"
    suffix = "cuda"
    kernel_translation = True
    config = {
        "grouped" : True,
        "device" : 3,
        "atomics": True,
        "color2": False
        }

class F2CCuda(Target):
    name = "f2c_cuda"
    suffix = "f2c"
    kernel_translation = True
    config = {
        "grouped" : True,
        "device" : 4,
        "atomics": True,
        "color2": False
        }

class Hip(Target):
    name = "hip"
    kernel_translation = True
    config = {
        "grouped" : True,
        "device" : 5,
        "atomics": True,
        "color2": False
        }

class F2CHip(Target):
    name = "f2c_hip"
    suffix = "f2c"
    kernel_translation = True
    config = {
        "grouped" : True,
        "device" : 6,
        "atomics": True,
        "color2": False
        }

class OpenMPOffload(Target):
    name = "openmp_offload"
    kernel_translation = True
    suffix = "ompoffload"
    config = {
        "grouped" : True,
        "device" : 7,
        "atomics": True,
        "color2": False
        }

#class OpenACC(Target):
#    name = "openacc"
#    kernel_translation = True
#    config = {
#        "grouped" : True,
#        "device" : 5,
#        "atomics": True,
#        "color2": False
#        }

class Sycl(Target):
    name = "sycl"
    kernel_translation = True
    config = {
        "grouped" : True,
        "device" : 8,
        "atomics": True,
        "color2": False
        }

class HLS(Target):
    name = "hls"
    kernel_translation = True
    config = {
        "grouped" : False,
        "SLR_count" : 1,
        "device" : 9
        }

Target.register(MPIOpenMP)
Target.register(F2CMPIOpenMP)
Target.register(Cuda)
Target.register(F2CCuda)
Target.register(Hip)
Target.register(F2CHip)
Target.register(OpenMPOffload)
#Target.register(OpenACC)
Target.register(Sycl)
Target.register(HLS)
