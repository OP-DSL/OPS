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
# OMPOFFLOAD
if (OpenMP_CXX_VERSION GREATER_EQUAL 4.5)
  include(OPS_BUILD_OMPOFFLOAD)
endif()


#   Configure the installation of "include"
file(GLOB_RECURSE INCLUDE "${CMAKE_CURRENT_SOURCE_DIR}/include/*.h")
install(FILES ${INCLUDE} DESTINATION include)
# Add the configure file
configure_package_config_file(${CMAKE_CURRENT_SOURCE_DIR}/cmake/Config.cmake.in
                              "${CMAKE_BINARY_DIR}/OPSConfig.cmake"
                              INSTALL_DESTINATION ${ConfigPackageLocation})

