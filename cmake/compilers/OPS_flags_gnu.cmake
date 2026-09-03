# Flags for GNU compiler
# CXX
set(OPS_CXXFLAGS "-fPIC -Wall -std=c++11 -g")
set(OPS_CXXFLAGS_RELEASE "-O3")
set(OPS_CXXFLAGS_DEBUG   "-O0 -ffloat-store")
if(OpenMP_CXX_FOUND)
  set(OPS_CXXFLAGS_RELEASE "${OPS_CXXFLAGS_RELEASE} -fopenmp")
  # To test if GNU 14 is needed or 13 is OK with different set-up options
  # Library compile the apps no
  #if (CMAKE_CXX_COMPILER_VERSION GREATER_EQUAL "13")
  #  if(SET_CUDA_ARCH)
  #    #set(OPS_CXXFLAGS_OMPOFFLOAD "-fopenmp-simd" "-foffload-options=nvptx-none=\"march=sm_${SET_CUDA_ARCH}\"")
  #    # set(OPS_CXXFLAGS_OMPOFFLOAD "-fopenmp-simd" "-foffload=\"-march=sm_${SET_CUDA_ARCH}\"")
  #    #set(OPS_CXXFLAGS_OMPOFFLOAD "-fopenmp-simd")
  #    #set(OPS_CXXFLAGS_OMPOFFLOAD "-fopenmp-simd" "-march=sm_${SET_CUDA_ARCH}")
  #  endif()
  #endif()
endif(OpenMP_CXX_FOUND)
# Fortran
if(OPS_FORTRAN)
  set(OPS_FORTFLAGS "-g -ffree-form -ffree-line-length-none")
  set(OPS_FORTFLAGS_RELEASE "-O3 -march=native")
  set(OPS_FORTFLAGS_DEBUG   "-O0 -ffloat-store -g3 -Og -ffpe-trap=invalid,zero -fcheck=all -fimplicit-none")
  if(OpenMP_FORTRAN_FOUND)
    set(OPS_FORTFLAGS_RELEASE "${OPS_FORTFLAGS_RELEASE} -fopenmp")
  endif()
endif()
