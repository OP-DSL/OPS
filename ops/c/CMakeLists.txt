cmake_minimum_required(VERSION 3.18)
project(API C CXX)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

# Prepare for files to compile the SEQ targets
file(GLOB_RECURSE CORE "${CMAKE_CURRENT_SOURCE_DIR}/src/core/*.cpp")
file(GLOB_RECURSE SEQ "${CMAKE_CURRENT_SOURCE_DIR}/src/sequential/*.cpp")
file(GLOB_RECURSE EXTERN "${CMAKE_CURRENT_SOURCE_DIR}/src/externlib/*.cpp")
list(FILTER EXTERN EXCLUDE REGEX "hdf5")
macro(InstallTarget Name Pos)
  install(TARGETS ops_${Name} EXPORT ops_${Name}_targets
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
    INCLUDES DESTINATION include
  )
  install(EXPORT ops_${Name}_targets
    FILE ops_${Name}_targets.cmake
    NAMESPACE
      OPS::
    DESTINATION ${Pos}
  )
endmacro()
# seq
add_library(ops_seq ${CORE} ${EXTERN} ${SEQ} )
#generate_export_header(ops_seq)

target_link_libraries(ops_seq PRIVATE)
InstallTarget(seq ${ConfigPackageLocation})

# hdf5_seq
if (HDF5_FOUND)
    add_library(ops_hdf5_seq ${CMAKE_CURRENT_SOURCE_DIR}/src/externlib/ops_hdf5.cpp ${CMAKE_CURRENT_SOURCE_DIR}/src/externlib/ops_hdf5_common.cpp)
    target_link_libraries(ops_hdf5_seq PRIVATE hdf5::hdf5 hdf5::hdf5_hl MPI::MPI_CXX)
    InstallTarget(hdf5_seq ${ConfigPackageLocation})
endif ()
# CUDA seq
if (CUDAToolkit_FOUND)
    file(GLOB_RECURSE CUDA "${CMAKE_CURRENT_SOURCE_DIR}/src/cuda/*")
    add_library(ops_cuda ${CORE} ${EXTERN} ${CUDA})
    target_link_libraries(ops_cuda PRIVATE CUDA::cudart_static)
    InstallTarget(cuda ${ConfigPackageLocation})
endif ()

if (OpenCL_FOUND)
    file(GLOB_RECURSE OPENCL "${CMAKE_CURRENT_SOURCE_DIR}/src/opencl/*")
    add_library(ops_opencl ${CORE} ${EXTERN} ${OPENCL})
    target_link_libraries(ops_opencl PRIVATE ${OpenCL_LIBRARIES})
    target_include_directories(ops_opencl PRIVATE ${OpenCL_INCLUDE_DIRS})
    InstallTarget(opencl ${ConfigPackageLocation})
endif ()
if (OPS_HIP)
if (HIP_FOUND)
    file(GLOB_RECURSE HIP "${CMAKE_CURRENT_SOURCE_DIR}/src/hip/*.cpp")
    add_library(ops_hip ${CORE} ${EXTERN} ${HIP})
    #-D__HIP_PLATFORM_NVCC__= -D__HIP_PLATFORM_NVIDIA__ -I/opt/rocm-4.5.0/hip/include -I/usr/local/cuda/include
    target_compile_definitions(ops_hip PRIVATE  __HIP_PLATFORM_NVIDIA__)
    #target_compile_options(ops_hip PRIVATE -I/opt/rocm-4.5.0/hip/include)
    target_link_libraries(ops_hip PRIVATE CUDA::cudart_static)

    target_include_directories(ops_hip PRIVATE /opt/rocm-4.5.0/hip/include)
    InstallTarget(hip ${ConfigPackageLocation})
endif ()
endif ()

if (MPI_FOUND)
    file(GLOB_RECURSE MPICORE "${CMAKE_CURRENT_SOURCE_DIR}/src/core/*.cpp")
    list(FILTER MPICORE EXCLUDE REGEX "singlenode")
    file(GLOB_RECURSE MPI "${CMAKE_CURRENT_SOURCE_DIR}/src/mpi/*.cpp")
    list(FILTER MPI EXCLUDE REGEX "opencl")
    list(FILTER MPI EXCLUDE REGEX "cuda")
    list(FILTER MPI EXCLUDE REGEX "hdf5")
    list(FILTER MPI EXCLUDE REGEX "hip")
    # MPI
    add_library(ops_mpi ${MPI} ${MPICORE} ${EXTERN})
    target_link_libraries(ops_mpi PRIVATE MPI::MPI_CXX)
    InstallTarget(mpi ${ConfigPackageLocation})
    # HDF5_MPI
    if (HDF5_FOUND)
        add_library(ops_hdf5_mpi ${CMAKE_CURRENT_SOURCE_DIR}/src/mpi/ops_mpi_hdf5.cpp ${CMAKE_CURRENT_SOURCE_DIR}/src/externlib/ops_hdf5_common.cpp)
        target_link_libraries(ops_hdf5_mpi PRIVATE hdf5::hdf5 hdf5::hdf5_hl MPI::MPI_CXX)
        InstallTarget(hdf5_mpi ${ConfigPackageLocation})
    endif ()
    #MPI_CUDA
    if (CUDAToolkit_FOUND)
        file(GLOB_RECURSE MPICUDA "${CMAKE_CURRENT_SOURCE_DIR}/src/mpi/*")
        list(FILTER MPICUDA EXCLUDE REGEX "opencl")
        list(FILTER MPICUDA EXCLUDE REGEX "host")
        list(FILTER MPICUDA EXCLUDE REGEX "hdf5")
        list(FILTER MPICUDA EXCLUDE REGEX "decl.cpp")
        list(FILTER MPICUDA EXCLUDE REGEX "hip")
        add_library(ops_mpi_cuda ${MPICORE} ${EXTERN} ${MPICUDA}  "${CMAKE_CURRENT_SOURCE_DIR}/src/cuda/ops_cuda_rt_support.cpp")
        target_link_libraries(ops_mpi_cuda PRIVATE CUDA::cudart_static MPI::MPI_CXX)
        InstallTarget(mpi_cuda ${ConfigPackageLocation})
    endif ()
    if (OpenCL_FOUND)
        file(GLOB_RECURSE MPIOPENCL "${CMAKE_CURRENT_SOURCE_DIR}/src/mpi/*")
        list(FILTER MPIOPENCL EXCLUDE REGEX "cuda")
        list(FILTER MPIOPENCL EXCLUDE REGEX "host")
        list(FILTER MPIOPENCL EXCLUDE REGEX "hdf5")
        list(FILTER MPIOPENCL EXCLUDE REGEX "decl.cpp")
        list(FILTER MPIOPENCL EXCLUDE REGEX "hip")
        add_library(ops_mpi_opencl ${MPICORE} ${EXTERN} ${MPIOPENCL} "${CMAKE_CURRENT_SOURCE_DIR}/src/opencl/ops_opencl_rt_support.cpp")
        target_include_directories(ops_mpi_opencl PRIVATE ${OpenCL_INCLUDE_DIRS})
        target_link_libraries(ops_mpi_opencl PRIVATE ${OpenCL_LIBRARIES} MPI::MPI_CXX )
        InstallTarget(mpi_opencl ${ConfigPackageLocation})
    endif()
    if (OPS_HIP)
    if (HIP_FOUND)
    file(GLOB_RECURSE MPIOPENCL "${CMAKE_CURRENT_SOURCE_DIR}/src/mpi/*")
    list(FILTER MPIOPENCL EXCLUDE REGEX "cuda")
    list(FILTER MPIOPENCL EXCLUDE REGEX "host")
    list(FILTER MPIOPENCL EXCLUDE REGEX "hdf5")
    list(FILTER MPIOPENCL EXCLUDE REGEX "decl.cpp")
    list(FILTER MPIOPENCL EXCLUDE REGEX "hip")
    add_library(ops_mpi_opencl ${MPICORE} ${EXTERN} ${MPIOPENCL} "${CMAKE_CURRENT_SOURCE_DIR}/src/opencl/ops_opencl_rt_support.cpp")
    target_include_directories(ops_mpi_opencl PRIVATE ${OpenCL_INCLUDE_DIRS})
    target_link_libraries(ops_mpi_opencl PRIVATE ${OpenCL_LIBRARIES} MPI::MPI_CXX )
    InstallTarget(mpi_opencl ${ConfigPackageLocation})
endif()
endif()
endif ()

# Tridiagonal library
if (LIBTRID_PATH)
    # Sequential Tridiagonal Library
    add_library(ops_trid_seq ${CMAKE_CURRENT_SOURCE_DIR}/src/tridiag/ops_tridiag.cpp)
    target_include_directories(ops_trid_seq PRIVATE ${LIBTRID_PATH}/include)
    InstallTarget(trid_seq ${ConfigPackageLocation})

    # Sequential CUDA Tridiagonal Library
    if (CUDAToolkit_FOUND)
        add_library(ops_trid_cuda ${CMAKE_CURRENT_SOURCE_DIR}/src/tridiag/ops_tridiag_cuda.cpp)
        target_include_directories(ops_trid_cuda PRIVATE ${LIBTRID_PATH}/include)
        target_link_libraries(ops_trid_cuda PRIVATE CUDA::cudart_static)
        InstallTarget(trid_cuda ${ConfigPackageLocation})
    endif ()

    if (MPI_FOUND)
        # MPI Tridiagonal Library
        add_library(ops_trid_mpi ${CMAKE_CURRENT_SOURCE_DIR}/src/tridiag/ops_tridiag_mpi.cpp)
        target_include_directories(ops_trid_mpi PRIVATE ${LIBTRID_PATH}/include)
        target_link_libraries(ops_trid_mpi PRIVATE MPI::MPI_CXX )
        InstallTarget(trid_mpi ${ConfigPackageLocation})

        if (CUDAToolkit_FOUND)
            # MPI + CUDA Tridiagonal Library
            add_library(ops_trid_mpi_cuda ${CMAKE_CURRENT_SOURCE_DIR}/src/tridiag/ops_tridiag_mpi_cuda.cpp)
            if (NCCL_FOUND)
                target_compile_definitions(ops_trid_mpi_cuda PRIVATE -DTRID_NCCL)
                target_include_directories(ops_trid_mpi_cuda PRIVATE ${NCCL_INCLUDE_DIR})
            endif ()
            target_include_directories(ops_trid_mpi_cuda PRIVATE ${LIBTRID_PATH}/include)
            target_link_libraries(ops_trid_mpi_cuda PRIVATE CUDA::cudart_static MPI::MPI_CXX )
            InstallTarget(trid_mpi_cuda ${ConfigPackageLocation})
        endif ()
    endif ()
endif ()
# if (LIBTRID_PATH)
#     set(LIBTRID ${LIBTRID_PATH})
# else()
#     set(LIBTRID "NONE")
# endif ()
configure_package_config_file(${CMAKE_CURRENT_SOURCE_DIR}/Config.cmake.in
    "${CMAKE_BINARY_DIR}/OPSConfig.cmake"
    INSTALL_DESTINATION  ${ConfigPackageLocation}
)
