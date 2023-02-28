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

class Seq(Target):
    name = "seq"
    kernel_translation = False
    config = {
        "grouped" : False, 
        "device" : 1
        }

class HLS(Target):
    name = "hls"
    kernel_translation = True
    config = {
        "grouped" : False,
        "SLR_count" : 1,
        "device" : 3
        }

Target.register(Seq)
Target.register(HLS)