# Add local module path
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

# Here we set all the lists for building the different targets
include(OPS_SET_SRC)
set(lib_prefix "ops_")

# Add the includes 
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

# Base build with sequenatial and MPI
include(OPS_BUILD_BASE)
# HDF5 IO
if(HDF5_FOUND)
  include(OPS_BUILD_HDF5)
endif()
# seq with CUDA
if(CUDAToolkit_FOUND)
  include(OPS_BUILD_CUDA)
endif()
## seq with HIP
#if(HIP_FOUND)
#  message(STATUS "Call BUILD_HIP")
#  include(OPS_BUILD_HIP)
#endif()
# OMPOFFLOAD
# If the FLAFS are set for the OMPOFFLOAD than we build
if (OPS_CXXFLAGS_OMPOFFLOAD)
  include(OPS_BUILD_OMPOFFLOAD)
endif()


#   Configure the installation of "include"
file(GLOB_RECURSE INCLUDE "${CMAKE_CURRENT_SOURCE_DIR}/include/*.h")
install(FILES ${INCLUDE} DESTINATION include)
