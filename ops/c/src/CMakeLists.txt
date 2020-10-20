cmake_minimum_required(VERSION 3.9)


option(CHECKPOINTING "Enable Checkpointing support" OFF)

enable_language(C CXX)
include(CheckLanguage)
check_language(CUDA)
if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 11)
    find_library(CUDART_LIBRARY cudart)
    set(CUDA_FOUND ON)
endif()


find_package(OpenCL)
find_package(OpenMP)
find_package(MPI)

# WARNING:  This is a hack.  Eventually, Cmake will have MPI::MPI_CUDA.  Until then, we need
# to remove '-pthread' from the INTERFACE_LINK_LIBRARIES, or else 'nvcc' gets mad
if(CMAKE_VERSION VERSION_LESS "3.14.0")
get_property(_prop TARGET MPI::MPI_CXX PROPERTY INTERFACE_LINK_LIBRARIES)
string(REPLACE " -pthread" "" _prop2 "${_prop}")
set_property(TARGET MPI::MPI_CXX PROPERTY INTERFACE_LINK_LIBRARIES "${_prop2}")
if(CMAKE_VERSION VERSION_LESS "3.13.0")
get_property(_prop TARGET MPI::MPI_CXX PROPERTY INTERFACE_COMPILE_OPTIONS)
string(REPLACE "-pthread" "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:-Xcompiler >-pthread" _prop2 "${_prop}")
set_property(TARGET MPI::MPI_CXX PROPERTY INTERFACE_COMPILE_OPTIONS "${_prop2}")
endif()
endif()

find_package(HDF5 COMPONENTS C HL)
if (CHECKPOINTING)
    if(NOT HDF5_FOUND)
        message( FATAL_ERROR "Checkpointing requires HDF5" )
    endif()
    set(CHECKPOINT_FLAG "-DCHECKPOINTING")
endif()


# In case we're called without our higher-level CMake file
if(NOT OPS_C_INCLUDE_DIR)
    set(OPS_C_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../include")
endif()


file(GLOB _OPS_C_HEADERS ${OPS_C_INCLUDE_DIR}/*.h)
if (OPS_INSTALL_HEADERS)
install(FILES ${_OPS_C_HEADERS} DESTINATION include)
endif()

if ( OpenMP_CXX_FOUND )
    set_source_files_properties(
        sequential/ops_seq.cpp
        externlib/ops_util.cpp
        mpi/ops_mpi_rt_support_host.cpp
        PROPERTIES
            COMPILE_OPTIONS ${OpenMP_CXX_FLAGS})
endif()

add_library(ops_core STATIC
    core/ops_lib_core.cpp
    core/ops_instance.cpp
    core/ops_lazy.cpp
    externlib/ops_util.cpp
    externlib/ops_checkpointing.cpp
    externlib/ops_checkpointing_strategy.cpp
)
set_property(TARGET ops_core PROPERTY POSITION_INDEPENDENT_CODE ON)

if( OpenMP_CXX_FOUND )
    target_link_libraries(ops_core PUBLIC OpenMP::OpenMP_CXX)
endif()

target_compile_definitions(ops_core PRIVATE ${CHECKPOINT_FLAG})
if (CHECKPOINTING)
    target_link_libraries(ops_core
        PUBLIC
            ${HDF5_C_LIBRARIES}
            ${HDF5_C_HL_LIBRARIES}
     )
    target_include_directories(ops_core PRIVATE ${HDF5_INCLUDE_DIR})
endif()


add_library(ops_seq STATIC
    sequential/ops_seq.cpp
    core/ops_dummy_singlenode.cpp)
set_property(TARGET ops_seq PROPERTY POSITION_INDEPENDENT_CODE ON)
target_link_libraries(ops_seq PUBLIC ops_core)

if(MPI_CXX_FOUND)
    add_library(ops_mpi STATIC
        mpi/ops_mpi_decl.cpp
        mpi/ops_mpi_rt_support.cpp
        mpi/ops_mpi_core.cpp
        mpi/ops_mpi_rt_support_host.cpp
        mpi/ops_mpi_partition.cpp)
    set_property(TARGET ops_mpi PROPERTY POSITION_INDEPENDENT_CODE ON)
    target_link_libraries(ops_mpi
        PUBLIC
            ops_core
            MPI::MPI_CXX
            )
	if (OPS_INSTALL_LIBS)
    install(TARGETS ops_mpi DESTINATION lib)
	endif()
endif()

if(CUDA_FOUND)
    add_library(ops_cuda STATIC
        cuda/ops_cuda_rt_support.cu
        cuda/ops_cuda_rt_support.cpp
        cuda/ops_cuda_decl.cpp
        core/ops_dummy_singlenode.cpp)
    set_property(TARGET ops_cuda PROPERTY POSITION_INDEPENDENT_CODE ON)
    set_source_files_properties(cuda/ops_cuda_decl.cpp PROPERTIES COMPILE_DEFINITIONS SET_CUDA_CACHE_CONFIG)
    set_source_files_properties(cuda/ops_cuda_rt_support.cpp.cpp PROPERTIES COMPILE_DEFINITIONS SET_CUDA_CACHE_CONFIG)
    target_include_directories(ops_cuda PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
    target_link_libraries(ops_cuda
        PUBLIC ops_core)

	if (OPS_INSTALL_LIBS)
    install(TARGETS ops_cuda DESTINATION lib)
	endif()
endif()

if(MPI_CXX_FOUND AND CUDA_FOUND)
    add_library(ops_mpi_cuda STATIC
            mpi/ops_mpi_decl_cuda.cpp
            mpi/ops_mpi_rt_support.cpp
            mpi/ops_mpi_core.cpp
            mpi/ops_mpi_rt_support_cuda.cu
            mpi/ops_mpi_partition.cpp
            cuda/ops_cuda_rt_support.cpp
    )
    set_property(TARGET ops_mpi_cuda PROPERTY POSITION_INDEPENDENT_CODE ON)
    set_source_files_properties(mpi/ops_mpi_decl_cuda.cpp PROPERTIES COMPILE_DEFINITIONS SET_CUDA_CACHE_CONFIG)
    target_include_directories(ops_mpi_cuda PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
    target_link_libraries(ops_mpi_cuda
        PUBLIC
            ops_core
            MPI::MPI_CXX
    )
	if (OPS_INSTALL_LIBS)
    install(TARGETS ops_mpi_cuda DESTINATION lib)
	endif()
endif()

if(OPENCL_FOUND)
    add_library(ops_opencl STATIC
        opencl/ops_opencl_rt_support.cpp
        opencl/ops_opencl_rt_support_kernels.cpp
        opencl/ops_opencl_decl.cpp
        core/ops_dummy_singlenode.cpp)
    set_property(TARGET ops_opencl PROPERTY POSITION_INDEPENDENT_CODE ON)
    target_link_libraries(ops_opencl
        PUBLIC
            ops_core
            OpenCL::OpenCL
    )
	if (OPS_INSTALL_LIBS)
    install(TARGETS ops_opencl DESTINATION lib)
	endif()
endif()

if(MPI_CXX_FOUND AND OPENCL_FOUND)
    add_library(ops_mpi_opencl STATIC
        mpi/ops_mpi_decl_opencl.cpp
        mpi/ops_mpi_rt_support.cpp
        mpi/ops_mpi_core.cpp
        mpi/ops_mpi_rt_support_opencl.cpp
        mpi/ops_mpi_partition.cpp
        opencl/ops_opencl_rt_support.cpp
    )
    set_property(TARGET ops_mpi_opencl PROPERTY POSITION_INDEPENDENT_CODE ON)
    target_link_libraries(ops_mpi_opencl
        PUBLIC
            ops_core
            OpenCL::OpenCL
            MPI::MPI_CXX
    )
	if (OPS_INSTALL_LIBS)
    install(TARGETS ops_mpi_opencl DESTINATION lib)
	endif()
endif()

if(HDF5_FOUND AND NOT HDF5_IS_PARALLEL)
    add_library(ops_hdf5_seq STATIC
        externlib/ops_hdf5)
    set_property(TARGET ops_hdf5_seq PROPERTY POSITION_INDEPENDENT_CODE ON)
    target_compile_definitions(ops_hdf5_seq PRIVATE ${CHECKPOINT_FLAG})
    target_include_directories(ops_hdf5_seq PRIVATE ${HDF5_INCLUDE_DIR})
    target_link_libraries(ops_hdf5_seq
        PUBLIC
            ${HDF5_C_LIBRARIES} ${HDF5_C_HL_LIBRARIES}
    )
	if (OPS_INSTALL_LIBS)
    install(TARGETS ops_hdf5_seq DESTINATION lib)
	endif()
endif()

if(MPI_CXX_FOUND AND HDF5_FOUND AND HDF5_IS_PARALLEL)
    add_library(ops_hdf5_mpi STATIC
        mpi/ops_mpi_hdf5.cpp)
    set_property(TARGET ops_hdf5_mpi PROPERTY POSITION_INDEPENDENT_CODE ON)
    target_compile_definitions(ops_hdf5_mpi PRIVATE ${CHECKPOINT_FLAG})
    target_include_directories(ops_hdf5_mpi PRIVATE ${HDF5_INCLUDE_DIR})
    target_link_libraries(ops_hdf5_mpi
        PUBLIC
            MPI::MPI_CXX
            ${HDF5_C_LIBRARIES} ${HDF5_C_HL_LIBRARIES}
    )
	if (OPS_INSTALL_LIBS)
    install(TARGETS ops_hdf5_mpi DESTINATION lib)
	endif()
endif()

if(LIBTRID_PATH)
    add_library(ops_trid STATIC
        tridiag/ops_tridiag.cpp)
    set_property(TARGET ops_trid PROPERTY POSITION_INDEPENDENT_CODE ON)
    target_include_directories(ops_trid PRIVATE ${LIBTRID_PATH}/include)
    target_link_libraries(ops_trid PUBLIC ops_core -L${LIBTRID_PATH}/lib -ltridcpu)
    if (OPS_INSTALL_LIBS)
        install(TARGETS ops_trid DESTINATION lib)
    endif()

    if(MPI_CXX_FOUND)
        add_library(ops_trid_mpi STATIC
            tridiag/ops_tridiag_mpi)
        set_property(TARGET ops_trid_mpi PROPERTY POSITION_INDEPENDENT_CODE ON)
        target_include_directories(ops_trid_mpi PRIVATE ${LIBTRID_PATH}/include)
        target_link_libraries(ops_trid_mpi PUBLIC ops_core -L${LIBTRID_PATH}/lib -ltridcpu)
        if (OPS_INSTALL_LIBS)
            install(TARGETS ops_trid_mpi DESTINATION lib)
        endif()
        if(CUDA_FOUND)
            if (OPS_TRID_USE_NCCL)
              find_package(NCCL)
              if (NCCL_FOUND)
                add_definitions(-DTRID_NCCL)
                include_directories(SYSTEM ${NCCL_INCLUDE_DIR})
              endif ()
            endif ()
            add_library(ops_trid_mpi_cuda STATIC
                tridiag/ops_tridiag_mpi_cuda.cpp)
            set_property(TARGET ops_trid_mpi_cuda PROPERTY POSITION_INDEPENDENT_CODE ON)
            target_include_directories(ops_trid_mpi_cuda PRIVATE ${LIBTRID_PATH}/include ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
            target_link_libraries(ops_trid_mpi_cuda PUBLIC ops_core ops_mpi_cuda -L${LIBTRID_PATH}/lib -ltridcuda_mpi MPI::MPI_CXX)

        	if (OPS_INSTALL_LIBS)
            install(TARGETS ops_trid_mpi_cuda DESTINATION lib)
        	endif()
        endif()
    endif()
endif()
