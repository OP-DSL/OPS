# Flags for Intel compiler
# C
set(OPS_CFLAGS "-xHost")
set(OPS_CFLAGS_RELEASE "-O3")
set(OPS_CFLAGS_DEBUG   "-O0 -g")
# CXX
set(OPS_CXXFLAGS "-xHost")
set(OPS_CXXFLAGS_RELEASE "-O3")
set(OPS_CXXFLAGS_DEBUG   "-O0 -g -DOPS_DEBUG")
