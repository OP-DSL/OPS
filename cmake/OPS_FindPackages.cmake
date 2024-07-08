# Look for dependancies 

# OpenMP
find_package(OpenMP QUIET)
if(NOT OpenMP_FOUND)
  message(FATAL_ERROR "OpenMP support NOT FOUND!")
endif()
# HDF5
find_package(HDF5 QUIET COMPONENTS C HL)
if(NOT HDF5_FOUND)
  message(WARNING "HDF5 support NOT FOUND. The HDF5 IO routines won't work! Please use -DHDF5_ROOT to specify the path!")
else()
  set(HDF5_PREFER_PARALLEL true)
endif()
# Python 
find_package(Python3 REQUIRED)
if(NOT Python3_FOUND)
  message(FATAL_ERROR "Python3 support NOT FOUND! The Python translator needs Python3! Please use -DPython3_EXECUTABLE to specify the path.")
endif()
# CUDA Support
# Set the NUMBER of GPU to 1 in any case
# CUDA and HIP will look if more are available
set(GPU_NUMBER 1)
if(OPS_CUDA)
  include(OPS_CUDA)
endif()
# HIP Support
if(OPS_HIP)
  include(OPS_HIP)
endif()
# MPI
find_package(MPI QUIET)
if(NOT MPI_FOUND)
  message(WARNING "MPI environment NOT FOUND! Only sequential code will be compiled!")
endif()
message(STATUS "MPICXX FOUND ${MPI_CXX_FOUND}")
message(STATUS "MPICC FOUND ${MPI_C_FOUND}")
