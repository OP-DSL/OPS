
    cmake_minimum_required(VERSION 3.18)

    project(API Fortran CXX)

    include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
    include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../c/include)

    include(CheckLanguage)
    check_language(Fortran)

    if(CMAKE_Fortran_COMPILER)
        message(STATUS "Fortran compiler: ${CMAKE_Fortran_COMPILER}")

        set(CMAKE_Fortran_MODULE_DIRECTORY ${CMAKE_INSTALL_PREFIX}/mod)

        if(${CMAKE_Fortran_COMPILER_ID} STREQUAL Intel)
            set(CMAKE_Fortran_FLAGS         "${CMAKE_Fortran_FLAGS} -O3 -g -xHost -qopenmp")
            set(CMAKE_Fortran_FLAGS_DEBUG   "-g -traceback -qopenmp")
            set(CMAKE_Fortran_FLAGS_RELEASE "-O3 -ip -xHOST -qopenmp")
        elseif(${CMAKE_Fortran_COMPILER_ID} STREQUAL GNU)
            set(CMAKE_Fortran_FLAGS         "${CMAKE_Fortran_FLAGS} -O3 -g -ffree-form -ffree-line-length-none -fopenmp")
            set(CMAKE_Fortran_FLAGS_DEBUG   "-O0 -g3 -ffree-form -ffree-line-length-none -fopenmp")
            set(CMAKE_Fortran_FLAGS_RELEASE "-Ofast -march=native -ffree-form -ffree-line-length-none -fopenmp")
        elseif(${CMAKE_Fortran_COMPILER_ID} STREQUAL NVHPC)
            set(CMAKE_Fortran_FLAGS         "${CMAKE_Fortran_FLAGS} -O3 -fast -gopt -mp")
        else()
            message(
                WARNING
                    "There is no default flag for the chosen Fortran compiler!"
            )
        endif()

        if(CMAKE_BUILD_TYPE STREQUAL "Release")
            message(STATUS "CMAKE_Fortran_FLAGS_RELEASE: ${CMAKE_Fortran_FLAGS_RELEASE}")
        elseif(CMAKE_BUILD_TYPE STREQUAL "Debug")
            message(STATUS "CMAKE_Fortran_FLAGS_DEBUG: ${CMAKE_Fortran_FLAGS_DEBUG}")
        elseif(CMAKE_BUILD_TYPE STREQUAL "")
            message(STATUS "CMAKE_Fortran_FLAGS: ${CMAKE_Fortran_FLAGS}")
        else()
            message(STATUS "Unknown build type: ${CMAKE_BUILD_TYPE}")
        endif()

#       Prepare for files to compile the SEQ targets
        list(APPEND F_HDF5
                    "${CMAKE_CURRENT_SOURCE_DIR}/src/ops_for_hdf5_declarations.F90")

        list(APPEND F_COMMON
                    "${CMAKE_CURRENT_SOURCE_DIR}/src/ops_for_declarations.F90"
                    "${CMAKE_CURRENT_SOURCE_DIR}/src/ops_for_rt_support.F90")

        file(GLOB_RECURSE CORE "${CMAKE_CURRENT_SOURCE_DIR}/../c/src/core/*.cpp")
        file(GLOB_RECURSE SEQ "${CMAKE_CURRENT_SOURCE_DIR}/../c/src/sequential/*.cpp")
        file(GLOB_RECURSE EXTERN "${CMAKE_CURRENT_SOURCE_DIR}/../c/src/externlib/*.cpp")

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


#       TARGET: SEQ
        add_library(ops_for_seq ${CORE} ${SEQ} ${EXTERN} ${F_COMMON})
        target_compile_definitions(ops_for_seq PRIVATE OPS_FTN)
        target_link_libraries(ops_for_seq PRIVATE OpenMP::OpenMP_CXX)
        installtarget(for_seq ${ConfigPackageLocation})

#       TARGET: HDF5_SEQ
        if(HDF5_FOUND)
            file(GLOB_RECURSE HDF5_SEQ
                "${CMAKE_CURRENT_SOURCE_DIR}/../c/src/externlib/ops_hdf5_common.cpp"
                "${CMAKE_CURRENT_SOURCE_DIR}/../c/src/externlib/ops_hdf5.cpp")
            add_library(ops_for_hdf5_seq ${HDF5_SEQ} ${F_HDF5})
            add_dependencies(ops_for_hdf5_seq ops_for_seq)
            target_link_libraries(ops_for_hdf5_seq PRIVATE hdf5::hdf5 hdf5::hdf5_hl)
            target_compile_definitions(ops_for_hdf5_seq PRIVATE OPS_FTN)
            installtarget(for_hdf5_seq ${ConfigPackageLocation})
        endif()

#       TARGET: CUDA
        if(CUDAToolkit_FOUND)
            if(${CMAKE_Fortran_COMPILER_ID} STREQUAL NVHPC)

                list(APPEND CUDA
                    "${CMAKE_CURRENT_SOURCE_DIR}/../c/src/cuda/ops_cuda_common.cu"
                    "${CMAKE_CURRENT_SOURCE_DIR}/../c/src/cuda/ops_cuda_rt_support.cu"
                    "${CMAKE_CURRENT_SOURCE_DIR}/../c/src/cuda/ops_cuda_singlenode.cu"
                    "${CMAKE_CURRENT_SOURCE_DIR}/../c/src/core/ops_device_singlenode_common.cpp")
                add_library(ops_for_cuda ${CORE} ${EXTERN} ${CUDA} ${F_COMMON})
                target_compile_definitions(ops_for_cuda PRIVATE OPS_FTN)
                installtarget(for_cuda ${ConfigPackageLocation})
            endif()
        endif()

        if(MPI_FOUND)
            file(GLOB_RECURSE MPICORE "${CMAKE_CURRENT_SOURCE_DIR}/../c/src/core/*.cpp")
            list(FILTER MPICORE EXCLUDE REGEX "singlenode")
            set(CommonFiles
                "${CMAKE_CURRENT_SOURCE_DIR}/../c/src/mpi/ops_mpi_core.cpp"
                "${CMAKE_CURRENT_SOURCE_DIR}/../c/src/mpi/ops_mpi_decl.cpp"
                "${CMAKE_CURRENT_SOURCE_DIR}/../c/src/mpi/ops_mpi_partition.cpp"
                "${CMAKE_CURRENT_SOURCE_DIR}/../c/src/mpi/ops_mpi_rt_support.cpp")


#           TARGET: MPI
            file(GLOB_RECURSE MPI_HOST
                "${CMAKE_CURRENT_SOURCE_DIR}/../c/src/mpi/ops_mpi_rt_support_host.cpp"
                "${CMAKE_CURRENT_SOURCE_DIR}/../c/src/sequential/ops_host_common.cpp")

            list(APPEND MPI_HOST ${CommonFiles})
            add_library(ops_for_mpi ${MPI_HOST} ${MPICORE} ${EXTERN} ${F_COMMON})
            target_compile_definitions(ops_for_mpi PRIVATE OPS_FTN)
            target_link_libraries(ops_for_mpi PRIVATE OpenMP::OpenMP_CXX MPI::MPI_CXX MPI::MPI_Fortran)
            installtarget(for_mpi ${ConfigPackageLocation})


#           TARGET: HDF5_MPI
            if(HDF5_FOUND)
                file(GLOB_RECURSE HDF5_MPI
                    "${CMAKE_CURRENT_SOURCE_DIR}/../c/src/mpi/ops_mpi_hdf5.cpp"
                    "${CMAKE_CURRENT_SOURCE_DIR}/../c/src/externlib/ops_hdf5_common.cpp")

                add_library(ops_for_hdf5_mpi ${HDF5_MPI} ${F_HDF5})
                add_dependencies(ops_for_hdf5_mpi ops_for_mpi)
                target_compile_definitions(ops_for_hdf5_mpi PRIVATE OPS_FTN)
                target_link_libraries(ops_for_hdf5_mpi PRIVATE hdf5::hdf5 hdf5::hdf5_hl
                                                    MPI::MPI_Fortran)
                installtarget(for_hdf5_mpi ${ConfigPackageLocation})
            endif()

#           TARGET: MPI_CUDA
            if(CUDAToolkit_FOUND)
                if(${CMAKE_Fortran_COMPILER_ID} STREQUAL NVHPC)

                    list(APPEND MPI_DEVICE
                        "${CMAKE_CURRENT_SOURCE_DIR}/../c/src/cuda/ops_cuda_common.cu"
                        "${CMAKE_CURRENT_SOURCE_DIR}/../c/src/mpi/ops_mpi_rt_support_cuda.cu")
                    list(APPEND MPI_DEVICE ${CommonFiles})

                    add_library(ops_for_mpi_cuda ${MPI_DEVICE} ${MPICORE} ${EXTERN} ${F_COMMON})
                    target_compile_definitions(ops_for_mpi_cuda PRIVATE OPS_FTN)
                    target_link_libraries(ops_for_mpi_cuda PRIVATE OpenMP::OpenMP_CXX MPI::MPI_CXX MPI::MPI_Fortran)
                    installtarget(for_mpi_cuda ${ConfigPackageLocation})
                endif()
            endif()

        endif()

    endif(CMAKE_Fortran_COMPILER)
