# List the core subrotines
set(SRC_CXX_DIR "${CMAKE_SOURCE_DIR}/ops/c/src")
file(GLOB_RECURSE CORE "${SRC_CXX_DIR}/core/*.cpp")
file(GLOB_RECURSE SEQ "${SRC_CXX_DIR}/sequential/*.cpp")
file(GLOB_RECURSE EXTERN "${SRC_CXX_DIR}/externlib/*.cpp")
file(GLOB_RECURSE MPICORE "${SRC_CXX_DIR}/core/*.cpp")

# Remove some specific routines for the lists
list(FILTER EXTERN EXCLUDE REGEX "hdf5")
list(FILTER CORE EXCLUDE REGEX "device")
list(FILTER MPICORE EXCLUDE REGEX "singlenode")
# Set the source for HDF5
set(HDF "${SRC_CXX_DIR}/externlib/ops_hdf5_common.cpp"
        "${SRC_CXX_DIR}/externlib/ops_hdf5.cpp")
set(HDF_MPI "${SRC_CXX_DIR}/externlib/ops_hdf5_common.cpp"
            "${SRC_CXX_DIR}/mpi/ops_mpi_hdf5.cpp")
# Set the sources for CUDA
file(GLOB_RECURSE CUDA "${SRC_CXX_DIR}/cuda/*"
                       "${SRC_CXX_DIR}/core/ops_device_singlenode_common.cpp")
# Set sources for OMPOFFLOAD
file(GLOB_RECURSE OMPOFFLOAD "${SRC_CXX_DIR}/ompoffload/ops_ompoffload_rt_support_kernels.cpp"
                             "${SRC_CXX_DIR}/ompoffload/ops_ompoffload_singlenode.cpp"
                             "${SRC_CXX_DIR}/core/ops_device_singlenode_common.cpp")
if (OpenMP_CXX_VERSION GREATER_EQUAL "5")
  list(APPEND OMPOFFLOAD "${SRC_CXX_DIR}/ompoffload/ops_ompoffload_common_omp5.cpp")
else()
  list(APPEND OMPOFFLOAD "${SRC_CXX_DIR}/ompoffload/ops_ompoffload_common_omp4.cpp")
endif()
#
set(MPICommonFiles
                "${SRC_CXX_DIR}/mpi/ops_mpi_core.cpp"
                "${SRC_CXX_DIR}/mpi/ops_mpi_decl.cpp"
                "${SRC_CXX_DIR}/mpi/ops_mpi_partition.cpp"
	        "${SRC_CXX_DIR}/mpi/ops_mpi_rt_support.cpp")

file(GLOB_RECURSE PUREMPI
            "${SRC_CXX_DIR}/mpi/ops_mpi_rt_support_host.cpp"
            "${SRC_CXX_DIR}/sequential/ops_host_common.cpp")
#
file(GLOB_RECURSE MPICUDA "${SRC_CXX_DIR}/mpi/*cuda*.cu"
                          "${SRC_CXX_DIR}/cuda/*")
list(FILTER MPICUDA EXCLUDE REGEX "singlenode")
#
file(GLOB_RECURSE MPIOMPOFFLOAD "${SRC_CXX_DIR}/mpi/*ompoffload*.cpp"
                                "${SRC_CXX_DIR}/ompoffload/ops_ompoffload_rt_support_kernels.cpp")

if (OpenMP_CXX_VERSION GREATER_EQUAL "5")
  list(APPEND MPIOMPOFFLOAD "${SRC_CXX_DIR}/ompoffload/ops_ompoffload_common_omp5.cpp")
else()
  list(APPEND MPIOMPOFFLOAD "${SRC_CXX_DIR}/ompoffload/ops_ompoffload_common_omp4.cpp")
endif()
