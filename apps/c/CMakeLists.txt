cmake_minimum_required(VERSION 3.17)
project(APP C CXX)
# if APP is compiled alone then steup compilers
if (${CMAKE_PROJECT_NAME} STREQUAL APP)
    # To show the compiling process in detail
    set(CMAKE_VERBOSE_MAKEFILE on)
    # Configure Compilers
    # C
    set(CMAKE_C_STANDARD 99)
    #TODO:Shall we keep the "-g" in the release mode? It increases file size.
    if (${CMAKE_C_COMPILER_ID} STREQUAL GNU)
        set(CMAKE_C_FLAGS "-fPIC -Wall -ffloat-store")
        set(CMAKE_C_FLAGS_RELEASE "-O3 -ftree-vectorize")
        set(CMAKE_C_FLAGS_DEBUG "-O0 -g")
    elseif (${CMAKE_C_COMPILER_ID} STREQUAL Clang)
        set(CMAKE_C_FLAGS "-fPIC -DUNIX -Wall")
        set(CMAKE_C_FLAGS_RELEASE "-O3")
        set(CMAKE_C_FLAGS_DEBUG "-g -O0")
    elseif (${CMAKE_C_COMPILER_ID} STREQUAL MSVC)
        set(CMAKE_C_FLAGS "-W1")
        set(CMAKE_C_FLAGS_RELEASE "/O2 /Ot ")
        set(CMAKE_C_FLAGS_DEBUG "/Z7 /DEBUG")
    else ()
        message(FATAL_ERROR "We can't recongnise the C compiler!")
    endif ()
    # C++
    set(CMAKE_CXX_STANDARD 11)
    if (${CMAKE_CXX_COMPILER_ID} STREQUAL GNU)
        set(CMAKE_CXX_FLAGS "-fPIC -Wall -ffloat-store")
        set(CMAKE_CXX_FLAGS_RELEASE "-O3")
        set(CMAKE_CXX_FLAGS_DEBUG "-g -O0 -DUNIX")
    elseif (${CMAKE_CXX_COMPILER_ID} STREQUAL Clang)
        set(CMAKE_CXX_FLAGS "-fPIC -DUNIX -Wall")
        set(CMAKE_CXX_FLAGS_RELEASE "-O3")
        set(CMAKE_CXX_FLAGS_DEBUG "-g -O0")
    elseif (${CMAKE_CXX_COMPILER_ID} STREQUAL MSVC)
        set(CMAKE_CXX_FLAGS "-W1")
        set(CMAKE_CXX_FLAGS_RELEASE "/O2 /Ot ")
        set(CMAKE_CXX_FLAGS_DEBUG "/Z7 /DEBUG")
    else ()
        message(FATAL_ERROR "We can't recongnise the C++ compiler!")
    endif ()

    # Try to find the required dependency
    find_package(MPI)
    find_package(HDF5)
    find_package(CUDAToolkit)
    find_package(OpenACC)
    find_package(OpenCL)
    find_package(OpenMP)
    # Configure the "include" dir for compiling
    if (HDF5_FOUND)
        include_directories(${HDF5_INCLUDE_DIRS})
    endif ()
    if (MPI_FOUND)
        include_directories(${MPI_C_INCLUDE_DIRS})
        include_directories(${MPI_CXX_INCLUDE_DIRS})
    endif ()

    if (CUDAToolkit_FOUND)
        include_directories(${CUDAToolkit_INCLUDE_DIRS})
        set(CMAKE_CUDA_COMPILER ${CUDAToolkit_NVCC_EXECUTABLE})
        enable_language(CUDA)
    endif ()

    if (OpenCL_FOUND)
        include_directories(${OpenCL_INCLUDE_DIRS})
    endif ()
endif()

if (${CMAKE_PROJECT_NAME} STREQUAL OPS)
    include_directories("${CMAKE_SOURCE_DIR}/ops/c/include")
    set(DO_MPI ${MPI_FOUND})
    set(UseHDF5 ${HDF5_FOUND})
    set(UseCUDA ${CUDAToolkit_FOUND})
    set(UseOpenCL ${OpenCL_FOUND})
endif()


# Pass 3rd argument to use HDF5, 4th is a list of additional sources
macro(BUILD_OPS_C_SAMPLE Name Odd)
    # Get the file to be compiled, DEV, OPS, OTHERS
    file(GLOB DEV "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp")
    list(FILTER DEV EXCLUDE REGEX "ops")
    file(GLOB OPS "${CMAKE_CURRENT_SOURCE_DIR}/*ops*.cpp")
    set(OTHERS ${DEV})
    foreach(OpsFile ${OPS})
        string(REPLACE "_ops" "" NoOPS ${OpsFile})
        list(REMOVE_ITEM OTHERS ${NoOPS})
    endforeach()
    if (NOT (${Odd} STREQUAL "NONE"))
        list(FILTER OTHERS EXCLUDE REGEX ${Odd})
        list(FILTER DEV EXCLUDE REGEX ${Odd})
    endif()

    add_executable(${Name}_seq_dev ${DEV})
    target_include_directories(${Name}_seq_dev PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
    add_executable(${Name}_seq ${OPS} ${OTHERS} "${CMAKE_CURRENT_SOURCE_DIR}/MPI_OpenMP/${Name}_cpu_kernels.cpp")
    target_include_directories(${Name}_seq PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
    target_link_libraries(${Name}_seq ops_seq)
    target_link_libraries(${Name}_seq_dev ops_seq)
    if (UseHDF5)
        target_link_libraries(${Name}_seq ops_hdf5_seq ${HDF5_LIBRARIES})
        target_link_libraries(${Name}_seq_dev ops_hdf5_seq ${HDF5_LIBRARIES})
    endif()

    if (UseCUDA)
        add_executable(${Name}_cuda ${OPS} ${OTHERS} "${CMAKE_CURRENT_SOURCE_DIR}/CUDA/${Name}_kernels.cu")
        target_include_directories(${Name}_cuda PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
        target_link_libraries(${Name}_cuda CUDA::cudart_static ops_cuda)
        if (UseHDF5)
            target_link_libraries(${Name}_cuda ops_hdf5_seq ${HDF5_LIBRARIES})
        endif()
    endif()

    if (UseOpenCL)
        add_executable(${Name}_opencl ${OPS} ${OTHERS} "${CMAKE_CURRENT_SOURCE_DIR}/OpenCL/${Name}_opencl_kernels.cpp")
        target_include_directories(${Name}_opencl PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
        target_link_libraries(${Name}_opencl  ${OpenCL_LIBRARIES} ops_opencl)
        if (UseHDF5)
            target_link_libraries(${Name}_opencl ops_hdf5_seq ${HDF5_LIBRARIES})
        endif()
    endif()



    # if (DO_MPI)
    #     add_executable(${Name}_mpi
    #         ${OPS_${Name}_BASEOUTPUT}
    #         ${OPS_${Name}_MPI_OpenMP_KERNELS}
    #         ${${EXTRA_SRC}}
    #         )
    #     target_include_directories(${Name}_mpi
    #         PRIVATE
    #             ${CMAKE_CURRENT_SOURCE_DIR}
    #             ${OPS_C_INCLUDE_DIR}
    #             )
    #     target_link_libraries(${Name}_mpi ops_mpi)
    #     if (UseHDF5)
    #         target_link_libraries(${Name}_mpi ops_hdf5_mpi)
    #     endif()

    #     if (TARGET ops_mpi_cuda)
    #         add_executable(${Name}_cuda_mpi
    #             ${OPS_${Name}_BASEOUTPUT}
    #             ${OPS_${Name}_CUDA_KERNELS}
    #             ${${EXTRA_SRC}}
    #             )
    #         target_include_directories(${Name}_cuda_mpi
    #             PRIVATE
    #                 ${CMAKE_CURRENT_SOURCE_DIR}
    #                 ${OPS_C_INCLUDE_DIR}
    #                 )
    #         target_link_libraries(${Name}_cuda_mpi ops_mpi_cuda)
    #         if (UseHDF5)
    #             target_link_libraries(${Name}_cuda_mpi ops_hdf5_mpi)
    #         endif()
    #     endif()
    # endif()



endmacro()


# add_subdirectory(access)
add_subdirectory(CloverLeaf)
#add_subdirectory(CloverLeaf_3D)
#add_subdirectory(CloverLeaf_3D_HDF5)
#add_subdirectory(complex_numbers)
#add_subdirectory(laplace2d_tutorial)
# add_subdirectory(lowdim_test)
# add_subdirectory(mblock)
#add_subdirectory(mblock4D)
# add_subdirectory(mb_shsgc)
#add_subdirectory(mgrid)
#add_subdirectory(multiDim)
#add_subdirectory(multiDim3D)
#add_subdirectory(multiDim_HDF5)
add_subdirectory(poisson)
# add_subdirectory(shsgc)
#add_subdirectory(TeaLeaf)
