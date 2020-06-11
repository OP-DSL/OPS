cmake_minimum_required(VERSION 3.17)
project(API C CXX)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

# Prepare for files to compile the SEQ targets
file(GLOB_RECURSE CORE "${CMAKE_CURRENT_SOURCE_DIR}/src/core/*.cpp")
file(GLOB_RECURSE SEQ "${CMAKE_CURRENT_SOURCE_DIR}/src/sequential/*.cpp")
file(GLOB_RECURSE EXTERN "${CMAKE_CURRENT_SOURCE_DIR}/src/externlib/*.cpp")
list(FILTER EXTERN EXCLUDE REGEX "hdf5")
# seq
add_library(ops_seq ${CORE} ${EXTERN} ${SEQ} )
target_link_libraries(ops_seq PRIVATE)
install(TARGETS ops_seq DESTINATION lib)
# hdf5_seq
if (HDF5_FOUND)
    add_library(ops_hdf5_seq ${CMAKE_CURRENT_SOURCE_DIR}/src/externlib/ops_hdf5.cpp)
    target_link_libraries(ops_hdf5_seq PRIVATE ${HDF5_LIBRARIES})
    install(TARGETS ops_hdf5_seq DESTINATION lib)
endif ()
# CUDA seq
if (CUDAToolkit_FOUND)
    file(GLOB_RECURSE CUDA "${CMAKE_CURRENT_SOURCE_DIR}/src/cuda/*")
    add_library(ops_cuda ${CORE} ${EXTERN} ${CUDA})
    target_link_libraries(ops_cuda PRIVATE CUDA::cudart_static)
    install(TARGETS ops_cuda DESTINATION lib)
endif ()

if (MPI_FOUND)
    file(GLOB_RECURSE MPICORE "${CMAKE_CURRENT_SOURCE_DIR}/src/core/*.cpp")
    list(FILTER MPICORE EXCLUDE REGEX "singlenode")
    file(GLOB_RECURSE MPI "${CMAKE_CURRENT_SOURCE_DIR}/src/mpi/*.cpp")
    list(FILTER MPI EXCLUDE REGEX "opencl")
    list(FILTER MPI EXCLUDE REGEX "cuda")
    list(FILTER MPI EXCLUDE REGEX "hdf5")
    # MPI
    add_library(ops_mpi ${MPI} ${MPICORE} ${EXTERN})
    target_link_libraries(ops_mpi PRIVATE MPI::MPI_CXX ${HDF5_LIBRARIES})
    install(TARGETS ops_mpi DESTINATION lib)
    # HDF5_MPI
    if (HDF5_FOUND)
        add_library(ops_hdf5_mpi ${CMAKE_CURRENT_SOURCE_DIR}/src/mpi/ops_mpi_hdf5.cpp)
        target_link_libraries(ops_hdf5_mpi PRIVATE MPI::MPI_CXX ${HDF5_LIBRARIES})
        install(TARGETS ops_hdf5_mpi DESTINATION lib)
    endif ()
    #MPI_CUDA
    if (CUDAToolkit_FOUND)
    # ar -r $(LIB)/libops_mpi_cuda.a $(OBJ)/ops_lib_core.o $(OBJ)/ops_instance.o $(OBJ)/ops_lazy.o \
	# $(OBJ)/ops_checkpointing.o $(OBJ)/ops_util.o $(OBJ)/ops_mpi_decl_cuda_mpi.o $(OBJ)/ops_mpi_rt_support_mpi.o \
	# $(OBJ)/ops_mpi_core_mpi.o $(OBJ)/ops_mpi_rt_support_cuda.o $(OBJ)/ops_mpi_partition_mpi.o \
	# $(OBJ)/ops_cuda_rt_support.o $(OBJ)/ops_checkpointing_strategy.o
        file(GLOB_RECURSE MPICUDA "${CMAKE_CURRENT_SOURCE_DIR}/src/mpi/*")
        list(FILTER MPICUDA EXCLUDE REGEX "opencl")
        list(FILTER MPICUDA EXCLUDE REGEX "host")
        list(FILTER MPICUDA EXCLUDE REGEX "hdf5")
        list(FILTER MPICUDA EXCLUDE REGEX "decl.cpp")
        add_library(ops_mpi_cuda ${MPICORE} ${EXTERN} ${MPICUDA}  "${CMAKE_CURRENT_SOURCE_DIR}/src/cuda/ops_cuda_rt_support.cpp")
        target_link_libraries(ops_mpi_cuda PRIVATE CUDA::cudart_static MPI::MPI_CXX)
        install(TARGETS ops_mpi_cuda DESTINATION lib)
    endif ()
endif ()