set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
include(OPS_InstallMacros)

# Here we set all the lists for building the different targets
include(OPS_SET_SRC)
set(lib_prefix "ops_")
# Base build with sequenatial and MPI
include(OPS_BUILD_BASE)
# seq with HDF5
if(HDF5_FOUND)
  include(OPS_BUILD_HDF5)
endif()
# seq with CUDA
if(CUDAToolkit_FOUND)
  include(OPS_BUILD_CUDA)
endif()
# OMPOFFLOAD
# If the FLAFS are set for the OMPOFFLOAD than we build
if (OPS_CXXFLAGS_OMPOFFLOAD)
  include(OPS_BUILD_OMPOFFLOAD)
endif()


#   Configure the installation of "include"
file(GLOB_RECURSE INCLUDE "${CMAKE_CURRENT_SOURCE_DIR}/include/*.h")
install(FILES ${INCLUDE} DESTINATION include)
# Add the configure file
configure_package_config_file(${CMAKE_CURRENT_SOURCE_DIR}/cmake/Config.cmake.in
                              "${CMAKE_BINARY_DIR}/OPSConfig.cmake"
                              INSTALL_DESTINATION ${ConfigPackageLocation})

