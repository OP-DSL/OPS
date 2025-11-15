# Add local module path
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_SOURCE_DIR}/../c/cmake")

# Add the additional Fortran Sources
include(OPS_SET_SRC)
include(OPS_SET_SRC_FORT)
set(lib_prefix "ops_for_")

# Add the includes 
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../c/include)

# Base build with sequential and MPI
include(OPS_BUILD_BASE_FORT)

## HDF5 IO
if(HDF5_FOUND)
  include(OPS_BUILD_HDF5_FORT)
endif()

# Configure the installation of "include"
file(GLOB_RECURSE INCLUDE_F "${CMAKE_CURRENT_SOURCE_DIR}/include/*.inc")
install(FILES ${INCLUDE_F} DESTINATION include)

set(CMAKE_Fortran_MODULE_DIRECTORY ${CMAKE_BINARY_DIR}/include)
install(DIRECTORY ${CMAKE_Fortran_MODULE_DIRECTORY}/
        DESTINATION include
        FILES_MATCHING PATTERN "*.mod")

