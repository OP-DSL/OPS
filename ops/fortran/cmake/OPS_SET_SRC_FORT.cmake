set(SRC_FORT_DIR "${CMAKE_SOURCE_DIR}/ops/fortran/src")

# Here we set the sources and we create support libraries 
# Support libraries are necessary to avoid race condition during // builds
file(GLOB_RECURSE F_COMMON "${SRC_FORT_DIR}/ops_for_declarations.F90"
                           "${SRC_FORT_DIR}/ops_for_rt_support.F90")
add_library(ops_for_common ${F_COMMON})

set(F_SUPPORT "${SRC_FORT_DIR}/ops_for_math_support.F90")
add_library(ops_for_math_support ${F_SUPPORT})

#set(F_CUDA "${SRC_FORT_DIR}/ops_for_cuda_reduction.CUF")
#add_library(ops_for_cuda_reduction ${F_SUPPORT})

# HDF5 is build afterwards 
set(F_HDF "${SRC_FORT_DIR}/ops_for_hdf5_declarations.F90")
