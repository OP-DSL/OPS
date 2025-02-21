# Flags for Clang compiler
# CXX
set(OPS_CXXFLAGS "-fPIC -fPIC -Wall -g -std=c++11 -g")
set(OPS_CXXFLAGS_RELEASE "-O3")
set(OPS_CXXFLAGS_DEBUG   "-O0")
if(OpenMP_CXX_FOUND)
  set(OPS_CXXFLAGS_RELEASE "${OPS_CXXFLAGS_RELEASE} -fopenmp")
endif(OpenMP_CXX_FOUND)
if(IEEE)
  set(OPS_CXXFLAGS_RELEASE "${OPS_CXXFLAGS_RELEASE} -fno-fast-math -ffp-contract=off -fdenormal-fp-math=ieee -fno-associative-math")
endif(IEEE)
	
