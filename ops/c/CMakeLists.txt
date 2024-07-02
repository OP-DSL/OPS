
    cmake_minimum_required(VERSION 3.18)

    project(API C CXX)

    include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

#   Prepare for files to compile the SEQ targets
    file(GLOB_RECURSE CORE "${CMAKE_CURRENT_SOURCE_DIR}/src/core/*.cpp")
    file(GLOB_RECURSE SEQ "${CMAKE_CURRENT_SOURCE_DIR}/src/sequential/*.cpp")
    file(GLOB_RECURSE EXTERN "${CMAKE_CURRENT_SOURCE_DIR}/src/externlib/*.cpp")

    list(FILTER EXTERN EXCLUDE REGEX "hdf5")
    list(FILTER CORE EXCLUDE REGEX "device")

    macro(InstallTarget Name Pos)
        install(
                TARGETS ops_${Name}
                EXPORT ops_${Name}_targets
                LIBRARY DESTINATION lib
                ARCHIVE DESTINATION lib
                RUNTIME DESTINATION bin
                INCLUDES
                DESTINATION include)
        install(
                EXPORT ops_${Name}_targets
                FILE ops_${Name}_targets.cmake
                NAMESPACE OPS::
                DESTINATION ${Pos})
    endmacro()

#   TARGET: SEQ
    add_library(ops_seq ${CORE} ${EXTERN} ${SEQ})

    target_link_libraries(ops_seq PRIVATE OpenMP::OpenMP_CXX)
    installtarget(seq ${ConfigPackageLocation})

#   TARGET: HDF5_SEQ
    if(HDF5_FOUND)
        add_library(
                    ops_hdf5_seq ${CMAKE_CURRENT_SOURCE_DIR}/src/externlib/ops_hdf5_common.cpp
                    ${CMAKE_CURRENT_SOURCE_DIR}/src/externlib/ops_hdf5.cpp)
        target_link_libraries(ops_hdf5_seq PRIVATE hdf5::hdf5 hdf5::hdf5_hl
                                             MPI::MPI_CXX)
        installtarget(hdf5_seq ${ConfigPackageLocation})
    endif()

#   TARGET: CUDA
    if(CUDAToolkit_FOUND)
        file(GLOB_RECURSE CUDA "${CMAKE_CURRENT_SOURCE_DIR}/src/cuda/*"
            "${CMAKE_CURRENT_SOURCE_DIR}/src/core/ops_device_singlenode_common.cpp")
        add_library(ops_cuda ${CORE} ${EXTERN} ${CUDA})
        target_link_libraries(ops_cuda PRIVATE OpenMP::OpenMP_CXX CUDA::cudart_static)
        installtarget(cuda ${ConfigPackageLocation})

        if(${CMAKE_C_COMPILER_ID} STREQUAL NVHPC)
            set(BUILD_OMPOFFLOAD TRUE)
            set(OMPOFFLOAD_FLAGS -Minline=reshape -Mvect=levels:5 -mp=gpu -gpu=cc${CMAKE_CUDA_ARCHITECTURES})
        endif()

    endif()

#   TARGET: OMPOFFLOAD
    if(BUILD_OMPOFFLOAD)
        file(GLOB_RECURSE OMPOFFLOAD "${CMAKE_CURRENT_SOURCE_DIR}/src/ompoffload/*"
                "${CMAKE_CURRENT_SOURCE_DIR}/src/core/ops_device_singlenode_common.cpp")
        add_library(ops_ompoffload ${CORE} ${EXTERN} ${OMPOFFLOAD})
        target_link_libraries(ops_ompoffload PRIVATE OpenMP::OpenMP_CXX)
        target_compile_options(ops_ompoffload PRIVATE ${OMPOFFLOAD_FLAGS})
        installtarget(ompoffload ${ConfigPackageLocation})
    endif()

#   TARGET: HIP
    if(OPS_HIP)
        if(HIP_FOUND)
            file(GLOB_RECURSE HIP "${CMAKE_CURRENT_SOURCE_DIR}/src/hip/*.cpp"
                "${CMAKE_CURRENT_SOURCE_DIR}/src/core/ops_device_singlenode_common.cpp")
            add_library(ops_hip ${CORE} ${EXTERN} ${HIP})
            # -D__HIP_PLATFORM_NVCC__= -D__HIP_PLATFORM_NVIDIA__
            # -I/opt/rocm-4.5.0/hip/include -I/usr/local/cuda/include
            target_compile_definitions(ops_hip PRIVATE OpenMP::OpenMP_CXX
                                               __HIP_PLATFORM_NVIDIA__)
            # target_compile_options(ops_hip PRIVATE -I/opt/rocm-4.5.0/hip/include)
            target_link_libraries(ops_hip PRIVATE CUDA::cudart_static)

            target_include_directories(ops_hip PRIVATE /opt/rocm-4.5.0/hip/include)
            installtarget(hip ${ConfigPackageLocation})
        endif()
    endif()

    if(MPI_FOUND)
        file(GLOB_RECURSE MPICORE "${CMAKE_CURRENT_SOURCE_DIR}/src/core/*.cpp")
        list(FILTER MPICORE EXCLUDE REGEX "singlenode")
        set(CommonFiles
                "${CMAKE_CURRENT_SOURCE_DIR}/src/mpi/ops_mpi_core.cpp"
                "${CMAKE_CURRENT_SOURCE_DIR}/src/mpi/ops_mpi_decl.cpp"
                "${CMAKE_CURRENT_SOURCE_DIR}/src/mpi/ops_mpi_partition.cpp"
                "${CMAKE_CURRENT_SOURCE_DIR}/src/mpi/ops_mpi_rt_support.cpp")

#       TARGET: MPI
        file(GLOB_RECURSE MPI
            "${CMAKE_CURRENT_SOURCE_DIR}/src/mpi/ops_mpi_rt_support_host.cpp"
            "${CMAKE_CURRENT_SOURCE_DIR}/src/sequential/ops_host_common.cpp")
        list(APPEND MPI ${CommonFiles})
        add_library(ops_mpi ${MPI} ${MPICORE} ${EXTERN})
        target_link_libraries(ops_mpi PRIVATE OpenMP::OpenMP_CXX MPI::MPI_CXX)
        installtarget(mpi ${ConfigPackageLocation})

#       TARGET: HDF5_MPI
        if(HDF5_FOUND)
            add_library(ops_hdf5_mpi ${CMAKE_CURRENT_SOURCE_DIR}/src/externlib/ops_hdf5_common.cpp
                        ${CMAKE_CURRENT_SOURCE_DIR}/src/mpi/ops_mpi_hdf5.cpp)
            target_link_libraries(ops_hdf5_mpi PRIVATE hdf5::hdf5 hdf5::hdf5_hl
                                               MPI::MPI_CXX)
            installtarget(hdf5_mpi ${ConfigPackageLocation})
        endif()

#       TARGET: MPI_CUDA
        if(CUDAToolkit_FOUND)
            file(GLOB_RECURSE MPICUDA "${CMAKE_CURRENT_SOURCE_DIR}/src/mpi/*cuda*.cu"
                "${CMAKE_CURRENT_SOURCE_DIR}/src/cuda/ops_cuda_common.cu")
            list(FILTER MPICUDA EXCLUDE REGEX "singlenode")
            list(APPEND MPICUDA ${CommonFiles})
            add_library(ops_mpi_cuda ${MPICORE} ${EXTERN} ${MPICUDA})
            target_link_libraries(ops_mpi_cuda PRIVATE OpenMP::OpenMP_CXX
                                               CUDA::cudart_static MPI::MPI_CXX)
            installtarget(mpi_cuda ${ConfigPackageLocation})

            if(${CMAKE_C_COMPILER_ID} STREQUAL NVHPC)
                set(BUILD_MPIOMPOFFLOAD TRUE)
            endif()
        endif()

#       TARGET: MPI_OMPOFFLOAD
        if(BUILD_MPIOMPOFFLOAD)
            file(GLOB_RECURSE MPIOMPOFFLOAD "${CMAKE_CURRENT_SOURCE_DIR}/src/mpi/*ompoffload*.cpp"
                    "${CMAKE_CURRENT_SOURCE_DIR}/src/ompoffload/*")
                list(FILTER MPIOMPOFFLOAD EXCLUDE REGEX "singlenode")
                list(APPEND MPIOMPOFFLOAD ${CommonFiles})
                add_library(ops_mpi_ompoffload ${MPICORE} ${EXTERN} ${MPIOMPOFFLOAD})
                target_link_libraries(ops_mpi_ompoffload PRIVATE OpenMP::OpenMP_CXX
                                                            MPI::MPI_CXX)
                target_compile_options(ops_mpi_ompoffload PRIVATE ${OMPOFFLOAD_FLAGS})
                installtarget(mpi_ompoffload ${ConfigPackageLocation})
        endif()

#       TARGET: MPI_HIP
        if(OPS_HIP)
            if(HIP_FOUND)
                file(GLOB_RECURSE MPIHIP "${CMAKE_CURRENT_SOURCE_DIR}/src/mpi/*hip*"
                    "${CMAKE_CURRENT_SOURCE_DIR}/src/hip/*")
                list(FILTER MPIHIP EXCLUDE REGEX "singlenode")
                list(APPEND MPIHIP ${CommonFiles})
                add_library(ops_mpi_hip ${MPICORE} ${EXTERN} ${MPIHIP})
                target_include_directories(ops_mpi_hip PRIVATE ${HIP_INCLUDE_DIRS})
                target_link_libraries(ops_mpi_hip PRIVATE OpenMP::OpenMP_CXX
                                                ${HIP_LIBRARIES} MPI::MPI_CXX)
                installtarget(mpi_hip ${ConfigPackageLocation})
            endif()
        endif()

    endif()


#   Tridiagonal library
    if(LIBTRID_PATH)
#       Sequential Tridiagonal Library
        add_library(ops_trid_seq
                    ${CMAKE_CURRENT_SOURCE_DIR}/src/tridiag/ops_tridiag.cpp)
        target_include_directories(ops_trid_seq PRIVATE ${LIBTRID_PATH}/include)
        installtarget(trid_seq ${ConfigPackageLocation})

#       Sequential CUDA Tridiagonal Library
        if(CUDAToolkit_FOUND)
            add_library(ops_trid_cuda
                        ${CMAKE_CURRENT_SOURCE_DIR}/src/tridiag/ops_tridiag_cuda.cpp)
            target_include_directories(ops_trid_cuda PRIVATE ${LIBTRID_PATH}/include)
            target_link_libraries(ops_trid_cuda PRIVATE CUDA::cudart_static)
            installtarget(trid_cuda ${ConfigPackageLocation})
        endif()

        if(MPI_FOUND)
#           MPI Tridiagonal Library
            add_library(ops_trid_mpi
                        ${CMAKE_CURRENT_SOURCE_DIR}/src/tridiag/ops_tridiag_mpi.cpp)
            target_include_directories(ops_trid_mpi PRIVATE ${LIBTRID_PATH}/include)
            target_link_libraries(ops_trid_mpi PRIVATE MPI::MPI_CXX)
            installtarget(trid_mpi ${ConfigPackageLocation})

            if(CUDAToolkit_FOUND)
#               MPI + CUDA Tridiagonal Library
                add_library(ops_trid_mpi_cuda
                            ${CMAKE_CURRENT_SOURCE_DIR}/src/tridiag/ops_tridiag_mpi_cuda.cpp)
                if(NCCL_FOUND)
                    target_compile_definitions(ops_trid_mpi_cuda PRIVATE -DTRID_NCCL)
                    target_include_directories(ops_trid_mpi_cuda
                                               PRIVATE ${NCCL_INCLUDE_DIR})
                endif()
                target_include_directories(ops_trid_mpi_cuda
                                            PRIVATE ${LIBTRID_PATH}/include)
                target_link_libraries(ops_trid_mpi_cuda PRIVATE CUDA::cudart_static
                                                      MPI::MPI_CXX)
                installtarget(trid_mpi_cuda ${ConfigPackageLocation})
            endif()
        endif()
    endif()

    configure_package_config_file(${CMAKE_CURRENT_SOURCE_DIR}/Config.cmake.in
                                  "${CMAKE_BINARY_DIR}/OPSConfig.cmake"
                                  INSTALL_DESTINATION ${ConfigPackageLocation})
