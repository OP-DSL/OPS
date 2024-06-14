# List the core subrotines
file(GLOB_RECURSE CORE "${CMAKE_CURRENT_SOURCE_DIR}/src/core/*.cpp")
file(GLOB_RECURSE SEQ "${CMAKE_CURRENT_SOURCE_DIR}/src/sequential/*.cpp")
file(GLOB_RECURSE EXTERN "${CMAKE_CURRENT_SOURCE_DIR}/src/externlib/*.cpp")
# Remove some specific routines for the lists
list(FILTER EXTERN EXCLUDE REGEX "hdf5")
list(FILTER CORE EXCLUDE REGEX "device")
# Set the source for HDF5
file(GLOB_RECURSE HDF "${CMAKE_CURRENT_SOURCE_DIR}/src/externlib/*hdf5*.cpp")
message(STATUS "HDF5 List ${HDF}")
# Set the sources for CUDA
file(GLOB_RECURSE CUDA "${CMAKE_CURRENT_SOURCE_DIR}/src/cuda/*"
                       "${CMAKE_CURRENT_SOURCE_DIR}/src/core/ops_device_singlenode_common.cpp")
