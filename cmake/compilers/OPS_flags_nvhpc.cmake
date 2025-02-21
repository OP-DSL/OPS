# Flags for NVHPC compiler
# CXX
set(OPS_CXXFLAGS "-g")
set(OPS_CXXFLAGS_RELEASE "-O3 -fast -gopt -Xcudafe \"--diag_suppress=unrecognized_gcc_pragma\" ")
#set(OPS_CXXFLAGS_RELEASE "-O3 -fast -gopt ")
set(OPS_CXXFLAGS_DEBUG   "-O0 ")
if(OpenMP_CXX_FOUND)
  set(OPS_CXXFLAGS_RELEASE "${OPS_CXXFLAGS_RELEASE} -mp")
  if(SET_CUDA_ARCH)
    # This is only for this specific target 
    # This flags are passed as target_compile_options
    # Each flag must be append to the list    
    set(OPS_CXXFLAGS_OMPOFFLOAD "-target=gpu")
    list(APPEND OPS_CXXFLAGS_OMPOFFLOAD "-Minfo=accel")
    #set(OPS_CXXFLAGS_OMPOFFLOAD "-mp=gpu -arch=sm_${SET_CUDA_ARCH}")
    #set(OPS_CXXFLAGS_OMPOFFLOAD "-mp=gpu -gencode arch=compute_${SET_CUDA_ARCH},code=sm_${SET_CUDA_ARCH}")
  endif()
endif(OpenMP_CXX_FOUND)
if(IEEE)
  set(OPS_CXXFLAGS_RELEASE "${OPS_CXXFLAGS_RELEASE} -Kieee -nofma")
endif(IEEE)
