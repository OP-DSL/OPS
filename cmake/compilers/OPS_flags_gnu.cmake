# Flags for GNU compiler
# CXX
set(OPS_CXXFLAGS "-fPIC -Wall -std=c++11 -g")
set(OPS_CXXFLAGS_RELEASE "-O3")
set(OPS_CXXFLAGS_DEBUG   "-O0 -ffloat-store")
if(OpenMP_CXX_FOUND)
  set(OPS_CXXFLAGS_RELEASE "${OPS_CXXFLAGS_RELEASE} -fopenmp")
endif(OpenMP_CXX_FOUND)

