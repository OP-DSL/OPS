# Add the Include files
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
# Build the basic sequential target
set(TargetName "hdf5_seq")
set(SRC ${HDF})
#
set(LibName "${lib_prefix}${TargetName}")
#set(Links "hdf5::hdf5 hdf5::hdf5_hl")
set(Links "hdf5::hdf5"
          "hdf5::hdf5_hl"
          "MPI::MPI_CXX")
#message(STATUS "SRC LIST1: ${SRC}")
setlib(${LibName} "${SRC}" "${Links}")
