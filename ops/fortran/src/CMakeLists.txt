cmake_minimum_required(VERSION 3.9)


option(CHECKPOINTING, "Enable Checkpointing support" OFF)

enable_language(C CXX Fortran)
include(CheckLanguage)
check_language(CUDA)
if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    find_library(CUDART_LIBRARY cudart)
    set(CUDA_FOUND ON)
endif()


find_package(OpenACC)
find_package(OpenMP)
find_package(MPI)

find_package(HDF5)
if (CHECKPOINTING)
    if(NOT HDF5_FOUND)
        message( FATAL_ERROR "Checkpointing requires HDF5" )
    endif()
    set(CHECKPOINT_FLAG "-DCHECKPOINTING")
endif()


if(NOT OPS_C_INCLUDE_DIR)
    message(FATAL_ERROR "This CMake file expects to be called from the higher-level file")
endif()


file(GLOB _OPS_FORTRAN_HEADERS ${OPS_FORTRAN_INCLUDE_DIR}/*.inc)
if (OPS_INSTALL_HEADERS)
install(FILES ${_OPS_FORTRAN_HEADERS} DESTINATION include)
endif()

# Use Free-from FORTRAN
string(APPEND CMAKE_Fortran_FLAGS " ${CMAKE_Fortran_FORMAT_FREE_FLAG}")
if(CMAKE_Fortran_COMPILER_ID MATCHES "GNU")
    string(APPEND CMAKE_Fortran_FLAGS " -ffree-line-length-none")
endif()

list(APPEND OPS_F_SRC
    "ops_for_declarations.F90"
    "ops_for_rt_support.F90"
    "ops_for_hdf5_declarations.F90")


add_library(ops_for_seq STATIC ${OPS_F_SRC})
target_link_libraries(ops_for_seq PUBLIC ops_seq)
if (OPS_INSTALL_LIBS)
install(TARGETS ops_for_seq DESTINATION lib)
endif()

if(MPI_CXX_FOUND)
    add_library(ops_for_mpi STATIC ${OPS_F_SRC})
    target_link_libraries(ops_for_mpi PUBLIC ops_mpi)
	if (OPS_INSTALL_LIBS)
    install(TARGETS ops_for_mpi DESTINATION lib)
	endif()
endif()

if(CUDA_FOUND)
    add_library(ops_for_cuda STATIC ${OPS_F_SRC})
    target_compile_definitions(ops_for_cuda PRIVATE OPS_WITH_CUDAFOR)
    target_link_libraries(ops_for_cuda PUBLIC ops_cuda)
	if (OPS_INSTALL_LIBS)
    install(TARGETS ops_for_cuda DESTINATION lib)
	endif()
endif()

if(MPI_CXX_FOUND AND CUDA_FOUND)
    add_library(ops_for_mpi_cuda STATIC ${OPS_F_SRC})
    target_compile_definitions(ops_for_mpi_cuda PRIVATE OPS_WITH_CUDAFOR)
    target_link_libraries(ops_for_mpi_cuda PUBLIC ops_mpi_cuda)
	if (OPS_INSTALL_LIBS)
    install(TARGETS ops_for_mpi_cuda DESTINATION lib)
	endif()
endif()

