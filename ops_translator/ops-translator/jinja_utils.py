import os
from math import ceil

from jinja2 import Environment, FileSystemLoader

import ops

env = Environment(
    loader=FileSystemLoader(os.path.join(os.path.dirname(__file__), "../resources/templates")),
    lstrip_blocks=True,
    trim_blocks=True
)

# env.tests["soa"] = lambda dat, loop=None: dat.soa
# env.tests["opt"]

env.tests["ops_dat"]    = lambda arg, loop=None: isinstance(arg, ops.ArgDat)
env.tests["ops_gbl"]    = lambda arg, loop=None: isinstance(arg, ops.ArgGbl)
env.tests["ops_reduce"] = lambda arg, loop=None: isinstance(arg, ops.ArgReduce)
env.tests["ops_idx"]    = lambda arg, loop=None: isinstance(arg, ops.ArgIdx)

env.tests["ops_read"]  = lambda arg, loop=None: hasattr(arg, "access_type") and arg.access_type == ops.AccessType.OPS_READ
env.tests["ops_write"] = lambda arg, loop=None: hasattr(arg, "access_type") and arg.access_type == ops.AccessType.OPS_WRITE
env.tests["ops_rw"]    = lambda arg, loop=None: hasattr(arg, "access_type") and arg.access_type == ops.AccessType.OPS_RW

env.tests["ops_inc"] = lambda arg, loop=None: hasattr(arg, "access_type") and arg.access_type == ops.AccessType.OPS_INC
env.tests["ops_min"] = lambda arg, loop=None: hasattr(arg, "access_type") and arg.access_type == ops.AccessType.OPS_MIN
env.tests["ops_max"] = lambda arg, loop=None: hasattr(arg, "access_type") and arg.access_type == ops.AccessType.OPS_MAX

env.tests["read_or_write"] = lambda arg, loop=None: hasattr(arg, "access_type") and arg.access_type in [
    ops.AccessType.OPS_READ,
    ops.AccessType.OPS_WRITE,
    ops.AccessType.OPS_RW
]

env.tests["reduction"] = lambda arg, loop=None: hasattr(arg, "access_type") and arg.access_type in [
    ops.AccessType.OPS_INC,
    ops.AccessType.OPS_MIN,
    ops.AccessType.OPS_MAX
]

def read_in(dat: ops.Dat, loop: ops.Loop) -> bool:
    for arg in loop.args:
        if not isinstance(arg, ops.ArgDat):
            continue

        if arg.dat_id == dat.id and arg.access_type not in [ops.AccessType.OPS_READ, ops.AccessType.OPS_RW]:
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

env.filters["ops_dat"] = test_to_filter("ops_dat")
env.filters["ops_gbl"] = test_to_filter("ops_gbl")
env.filters["ops_reduce"] = test_to_filter("ops_reduce")
env.filters["ops_idx"] = test_to_filter("ops_idx")

env.filters["ops_read"]  = test_to_filter("ops_read")
env.filters["ops_write"] = test_to_filter("ops_write")
env.filters["ops_rw"]    = test_to_filter("ops_rw")

env.filters["ops_inc"] = test_to_filter("inc")
env.filters["ops_min"] = test_to_filter("min")
env.filters["ops_max"] = test_to_filter("max")

env.filters["read_or_write"] = test_to_filter("read_or_write")
env.filters["reduction"] = test_to_filter("reduction")

env.filters["index"] = lambda xs, x: xs.index(x)

env.filters["round_up"] = lambda x, b: b * ceil(x / b)

env.filters["max_value"] = lambda values: max(values)
