from __future__ import annotations

from typing import Any, Dict
from util import Findable

#TODO: Add documentaion (numpy style)
class Target(Findable):
    name: str
    kernel_translation: bool
    config: Dict[str, Any]

    def __str__(self) -> str:
        return f"{self.name}: config: {self.config}"
    
    def __eq__(self, other) -> bool:
        return self.name == other.name if type(other) is type(self) else False

    def __hash__(self) -> int:
        return hash(self.name)

    def matches(self, key: str) -> bool:
        return self.name == key.lower()

class MPIOpenMP(Target):
    name = "mpi_openmp"
    kernel_translation = False
    config = {
        "grouped" : False, 
        "device" : 1
        }

class Cuda(Target):
    name = "cuda"
    kernel_translation = True
    config = {
        "grouped" : True,
        "device" : 2,
        "atomics": True,
        "color2": False
        }

class Hip(Target):
    name = "hip"
    kernel_translation = True
    config = {
        "grouped" : True,
        "device" : 3,
        "atomics": True,
        "color2": False
        }

class OpenMPOffload(Target):
    name = "openmp_offload"
    kernel_translation = True
    config = {
        "grouped" : True,
        "device" : 4,
        "atomics": True,
        "color2": False
        }

class OpenACC(Target):
    name = "openacc"
    kernel_translation = True
    config = {
        "grouped" : True,
        "device" : 5,
        "atomics": True,
        "color2": False
        }

class Sycl(Target):
    name = "sycl"
    kernel_translation = True
    config = {
        "grouped" : True,
        "device" : 6,
        "atomics": True,
        "color2": False
        }

class HLS(Target):
    name = "hls"
    kernel_translation = True
    config = {
        "grouped" : False,
        "SLR_count" : 1,
        "device" : 7
        }

Target.register(MPIOpenMP)
Target.register(Cuda)
Target.register(Hip)
Target.register(OpenMPOffload)
Target.register(OpenACC)
Target.register(Sycl)
Target.register(HLS)
