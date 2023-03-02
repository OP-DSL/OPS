import os
from math import ceil

from jinja2 import Environment, FileSystemLoader

import ops

env = Environment(
    loader=FileSystemLoader(os.path.join(os.path.dirname(__file__), "../resoources/templates")),
    lstrip_blocks=True,
    trim_blocks=True
)

# env.tests["soa"] = lambda dat, loop=None: dat.soa
# env.tests["opt"]

env.tests["dat"] = lambda arg, loop=None: isinstance(arg, ops.ArgDat)
env.tests["gbl"] = lambda arg, loop=None: isinstance(arg, ops.ArgGbl)
env.tests["idx"] = lambda arg, loop=None: isinstance(arg, ops.ArgIdx)

env.tests["read"] = lambda arg, loop=None: hasattr(arg, "access_type") and arg.access_type == ops.AccessType.READ
env.tests["write"] = lambda arg, loop=None: hasattr(arg, "access_type") and arg.access_type == ops.AccessType.WRITE
env.tests["read_write"] = lambda arg, loop=None: hasattr(arg, "access_type") and arg.access_type == ops.AccessType.RW

env.tests["inc"] = lambda arg, loop=None: hasattr(arg, "access_type") and arg.access_type == ops.AccessType.INC
env.tests["max"] = lambda arg, loop=None: hasattr(arg, "access_type") and arg.access_type == ops.AccessType.MAX
env.tests["min"] = lambda arg, loop=None: hasattr(arg, "access_type") and arg.access_type == ops.AccessType.MIN

env.tests["read_or_write"] = lambda arg, loop=None: hasattr(arg, "access_type") and arg.access_type in [
    ops.AccessType.READ,
    ops.AccessType.WRITE,
    ops.AccessType.RW
]

env.tests["reduciton"] = lambda arg, loop=None: hasattr(arg, "access_type") and arg.access_type in [
    ops.AccessType.INC,
    ops.AccessType.MAX,
    ops.AccessType.MIN
]

def read_in(dat: ops.Dat, loop: ops.Loop) -> bool:
    for arg in loop.args:
        if not isinstance(arg, ops.ArgDat):
            continue

        if arg.dat_id == dat.id and arg.access_type not in [ops.AccessType.READ, ops.AccessType.RW]:
            return False
    
    return True

env.tests["read_in"] = read_in
env.tests["instance"] = lambda x, c: isinstance(x, c)

def unpack(tup):
    if not isinstance(tup, tuple):
        return tup
    return tup[0]

def test_to_filter(filter_, key=unpack):
    return lambda xs, loop=None: list(filter(lambda x: env.tests[filter_](key(x), loop), xs))

env.filters["dat"] = test_to_filter("dat")
env.filters["gbl"] = test_to_filter("gbl")

env.filters["read"] = test_to_filter("read")
env.filters["write"] = test_to_filter("write")
env.filters["read_write"] = test_to_filter("read_write")

env.filters["int"] = test_to_filter("inc")
env.filters["min"] = test_to_filter("min")
env.filters["max"] = test_to_filter("max")

env.filters["read_or_write"] = test_to_filter("read_or_write")
env.filters["reduction"] = test_to_filter("reduction")

env.filters["index"] = lambda xs, x: xs.index(x)

env.filters["round_up"] = lambda x, b: b * ceil(x / b)
