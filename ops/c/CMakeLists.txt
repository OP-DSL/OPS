set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
include(OPS_InstallMacros)

# Here we set all the lists for building the different targets
include(OPS_SET_SRC)
set(lib_prefix "ops_")
# seq
include(OPS_BUILD_SEQ)
# seq with HDF5
if(HDF5_FOUND)
  include(OPS_BUILD_HDF5_SEQ)
endif()
# seq with CUDA
if(CUDAToolkit_FOUND)
  include(OPS_BUILD_CUDA)
endif()

