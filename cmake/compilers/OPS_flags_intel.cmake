# Flags for Intel compiler
# CXX
set(OPS_CXXFLAGS "-std=c++17 -Wall -fPIC")
set(OPS_CXXFLAGS_RELEASE "-O3 -Ofast -march=native")
set(OPS_CXXFLAGS_DEBUG   "-O0 -ffloat-store -g -DSYCL_COPY -DSYCL_USM")
if(IEEE)
  set(OPS_CXXFLAGS_RELEASE "${OPS_CXXFLAGS_RELEASE} -fp-model=strict")
endif(IEEE)
if(OpenMP_CXX_FOUND)
  set(OPS_CXXFLAGS_RELEASE "${OPS_CXXFLAGS_RELEASE} -qopenmp")
endif(OpenMP_CXX_FOUND)

if (IntelDPCPP_FOUND)
  # This is only for this specific target 
  # This flags are passed as target_compile_options
  # Each flag must be append to the list    
  #set(OPS_CXXFLAGS_OMPOFFLOAD "-fiopenmp")
  #list(APPEND OPS_CXXFLAGS_OMPOFFLOAD "-fopenmp-targets=spir64")
  if(CUDAToolkit_FOUND)
    # For NVIDIA	  
    set(OPS_CXXFLAGS_SYCL "--fsycl")
    list(APPEND OPS_CXXFLAGS_SYCL "-fsycl-targets=nvptx64-nvidia-cuda")
  else()
    # For INTEL GPUs
    set(OPS_CXXFLAGS_SYCL "--fsycl")
    list(APPEND OPS_CXXFLAGS_SYCL "-fsycl-targets=spir64")
  endif()
  # SYCL for CPU 
  #ifeq ($(GPU_VENDOR), UNKNOWN)
  #SYCL_FLAGS := -fsycl -fsycl-targets=spir64_x86_64
  #OMPOFFLOADFLAGS := -fiopenmp -fopenmp-targets=spir64_x86_64
endif()
