# List the core subrotines
file(GLOB_RECURSE CORE "${CMAKE_CURRENT_SOURCE_DIR}/src/core/*.cpp")
file(GLOB_RECURSE SEQ "${CMAKE_CURRENT_SOURCE_DIR}/src/sequential/*.cpp")
file(GLOB_RECURSE EXTERN "${CMAKE_CURRENT_SOURCE_DIR}/src/externlib/*.cpp")
file(GLOB_RECURSE MPICORE "${CMAKE_CURRENT_SOURCE_DIR}/src/core/*.cpp")

# Remove some specific routines for the lists
list(FILTER EXTERN EXCLUDE REGEX "hdf5")
list(FILTER CORE EXCLUDE REGEX "device")
list(FILTER MPICORE EXCLUDE REGEX "singlenode")
# Set the source for HDF5
set(HDF "${CMAKE_CURRENT_SOURCE_DIR}/src/externlib/ops_hdf5_common.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/src/externlib/ops_hdf5.cpp")
set(HDF_MPI "${CMAKE_CURRENT_SOURCE_DIR}/src/externlib/ops_hdf5_common.cpp"
            "${CMAKE_CURRENT_SOURCE_DIR}/src/mpi/ops_mpi_hdf5.cpp")
# Set the sources for CUDA
file(GLOB_RECURSE CUDA "${CMAKE_CURRENT_SOURCE_DIR}/src/cuda/*"
                       "${CMAKE_CURRENT_SOURCE_DIR}/src/core/ops_device_singlenode_common.cpp")
# Set sources for OMPOFFLOAD
file(GLOB_RECURSE OMPOFFLOAD "${CMAKE_CURRENT_SOURCE_DIR}/src/ompoffload/*"
                             "${CMAKE_CURRENT_SOURCE_DIR}/src/core/ops_device_singlenode_common.cpp")
#
set(MPICommonFiles
                "${CMAKE_CURRENT_SOURCE_DIR}/src/mpi/ops_mpi_core.cpp"
                "${CMAKE_CURRENT_SOURCE_DIR}/src/mpi/ops_mpi_decl.cpp"
                "${CMAKE_CURRENT_SOURCE_DIR}/src/mpi/ops_mpi_partition.cpp"
	        "${CMAKE_CURRENT_SOURCE_DIR}/src/mpi/ops_mpi_rt_support.cpp")

file(GLOB_RECURSE PUREMPI
            "${CMAKE_CURRENT_SOURCE_DIR}/src/mpi/ops_mpi_rt_support_host.cpp"
            "${CMAKE_CURRENT_SOURCE_DIR}/src/sequential/ops_host_common.cpp")
#
file(GLOB_RECURSE MPICUDA "${CMAKE_CURRENT_SOURCE_DIR}/src/mpi/*cuda*.cu"
                          "${CMAKE_CURRENT_SOURCE_DIR}/src/cuda/*")
list(FILTER MPICUDA EXCLUDE REGEX "singlenode")
#
file(GLOB_RECURSE MPIOMPOFFLOAD "${CMAKE_CURRENT_SOURCE_DIR}/src/mpi/*ompoffload*.cpp"
                                "${CMAKE_CURRENT_SOURCE_DIR}/src/ompoffload/*")
list(FILTER MPIOMPOFFLOAD EXCLUDE REGEX "singlenode")
