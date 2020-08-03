cmake_minimum_required(VERSION 3.17)
project(APP C CXX)
if ( NOT APP_INSTALL)
    message(FATAL_ERROR "We need to set APP_INSTALL variable for installing apps!")
endif()
#if APPs are compiled independently then setup compilers and dependencies.
if (${CMAKE_PROJECT_NAME} STREQUAL APP)
    # if show the compiling process in detail
    option(OPS_VERBOSE_WARNINGS "Turn on verbose warning messages" OFF)
    option(OPS_APP_TEST "Turn on tests for Apps" OFF)
    if (NOT OPS_VERBOSE_WARNINGS)
        message("We show concise compiling information by defautl! Use -DOPS_VERBOSE_WARNING=ON to switch on.")
    endif()
    set(CMAKE_VERBOSE_MAKEFILE ${OPS_VERBOSE_WARNINGS})
    set(OPS_APP_SRC ${CMAKE_SOURCE_DIR})
    # Use the Release mode by default
    if ( NOT CMAKE_BUILD_TYPE )
        message("We choose the Release mode! Please uses -DCMAKE_BUILD_TYPE to change.")
        set(CMAKE_BUILD_TYPE Release)
    endif()
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

     # Trying to find the OPS library
     # TODO this shall become FindOPS.cmake later
     SET (SEARCH_PATHS
     "${OPS_INSTALL_DIR}"
         /usr/local/
         /usr/
     )
     # Tyring to find the include directory
     set(OPS_HEAD ops_checkpointing.h  ops_hdf5.h  ops_lib_core.h
                   ops_opencl_reduction.h   ops_tridiag.h ops_cuda_reduction.h
                   ops_instance.h   ops_lib_cpp.h   ops_opencl_rt_support.h
                   ops_util.h ops_cuda_rt_support.h  ops_internal1.h  ops_macros.h
                   ops_seq.h  queue.h ops_exceptions.h  ops_internal2.h
                   ops_mpi_core.h  ops_seq_v2.h)
     set(OPS_INCLUDE_DIR "")
     foreach(head ${OPS_HEAD})
         unset(head_inc CACHE)
         FIND_PATH (head_inc
         NAMES ${head}
         PATHS ${SEARCH_PATHS}
         PATH_SUFFIXES include)
         if (head_inc)
             list(APPEND OPS_INCLUDE_DIR ${head_inc})
         else()
             message(FATAL_ERROR "We cannot find ${head}")
         endif()
     endforeach()
     list(REMOVE_DUPLICATES OPS_INCLUDE_DIR)
     include_directories("${OPS_INCLUDE_DIR}")
     # Trying to find the library directory
     set(OPS_LIB ops_cuda  ops_hdf5_seq  ops_mpi_cuda
         ops_opencl ops_hdf5_mpi  ops_mpi  ops_mpi_opencl ops_seq)
     foreach(library ${OPS_LIB})
         FIND_LIBRARY (${library}_FOUND
         NAMES ${library}
         PATHS ${SEARCH_PATHS}
         PATH_SUFFIXES lib)
         if (${library}_FOUND)
             unset(dir CACHE)
             get_filename_component(dir ${${library}_FOUND} DIRECTORY)
             list(APPEND OPS_LIBRARY_DIR ${dir})
         else()
             message("We cannot find ${library}")
         endif()
     endforeach()
     list(REMOVE_DUPLICATES OPS_LIBRARY_DIR)
     link_directories("${OPS_LIBRARY_DIR}")
     # set the OPS_INSTALL_PATH for testing
     set(OPS_INSTALL_PATH "${OPS_INSTALL_DIR}")

    # Try to find the required dependency
    find_package(MPI QUIET)
    find_package(HDF5 QUIET COMPONENTS C HL)
    find_package(CUDAToolkit QUIET)
    find_package(OpenACC QUIET)
    find_package(OpenCL QUIET)
    find_package(OpenMP QUIET)
    find_package(Python2 QUIET)
    if (NOT Python2_FOUND)
        message (FATAL_ERROR "We cannot find Python2 and the Python translator needs Python2!")
    else()
        message(${SEARCH_PATHS})
        FIND_PATH (opsc NAMES ops_c PATHS ${SEARCH_PATHS} PATH_SUFFIXES bin)
        FIND_PATH (opsfortran NAMES ops_fortran PATHS ${SEARCH_PATHS} PATH_SUFFIXES bin)
        if (NOT opsc)
            message (FATAL_ERROR "We cannot find the python C/C++ translator!")
        endif()
        if (NOT opsfortran)
            message (FATAL_ERROR "We cannot find the python fortran translator!")
        endif()
    endif()
    set(OPS_C_TRANSLATOR  "${opsc}/ops_c")
    set(OPS_F_TRANSLATOR  "${opsfortran}/ops_fortran")
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

    set(HDF5_SEQ ${HDF5_FOUND})
    if (ops_cuda_FOUND)
        set(CUDA ${CUDAToolkit_FOUND})
    endif()
    if (ops_opencl_FOUND)
        set(OPENCL ${OpenCL_FOUND})
    endif()
    if (${MPI_FOUND})
        set(MPI TRUE)
        set(HDF5_MPI ${HDF5_FOUND})
        if (ops_cuda_FOUND)
            set(CUDA_MPI ${CUDAToolkit_FOUND})
        endif()
        if (ops_opencl_FOUND)
            set(OPENCL_MPI ${OpenCL_FOUND})
        endif()
    endif()
    #prepare ctest
    if (OPS_APP_TEST)
        enable_testing()
        include(ProcessorCount)
        ProcessorCount(CPU_NUMBER)
        if(CPU_NUMBER GREATER 12)
            set(CPU_NUMBER 12)
        endif()
        if (NOT GPU_NUMBER)
            message("GPU tests are disabled since the GPU number is not specified!")
        endif()
    endif()
endif()
#if APPs are compiled together with the OPS API
if (${CMAKE_PROJECT_NAME} STREQUAL OPS)
    include_directories("${CMAKE_SOURCE_DIR}/ops/c/include")
    set(HDF5_SEQ ${HDF5_FOUND})
    set(CUDA ${CUDAToolkit_FOUND})
    set(OPENCL ${OpenCL_FOUND})
    if (${MPI_FOUND})
        set(MPI TRUE)
        set(HDF5_MPI ${HDF5_FOUND})
        set(CUDA_MPI ${CUDAToolkit_FOUND})
        set(OPENCL_MPI ${OpenCL_FOUND})
    endif()
    if (NOT Python2_FOUND)
        message (FATAL_ERROR "We cannot find Python2 and the Python translator needs Python2!")
    else()
        set(OPS_C_TRANSLATOR  "${CMAKE_SOURCE_DIR}/ops_translator/c/ops.py")
        set(OPS_F_TRANSLATOR  "${opsfortran}/ops_translator/fortran/ops_fortran.py")
    endif()
    # set the OPS_INSTALL_PATH for testing
    set(OPS_INSTALL_PATH "${CMAKE_SOURCE_DIR}/ops/c")
    set(OPS_APP_SRC ${CMAKE_SOURCE_DIR}/apps/c)
    if (OPS_APP_TEST)
        enable_testing()
    endif()
endif()



# Prepare the macro for compiling apps
# Name: App name
# Odd: Key words for source files are included in other source files and need no explicit compilation
# Others: Key words for source codes need no code generation using ops.py
# Extra: Key words for ource codes that cannot be treated by the macro
# TODO MPI_inline, titled OpenACC (Gnu?)
macro(BUILD_OPS_C_SAMPLE Name Odd Others Extra)
    # clean the tmp directory for the code generation
    if (UNIX)
        execute_process (
            COMMAND rm -r -f tmp
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        )
    endif()
    if (UNIX)
        execute_process (
            COMMAND mkdir tmp
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        )
    endif()
    # Copy all source and head files into tmp
    file(GLOB CPP "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp")
    list(FILTER CPP EXCLUDE REGEX "ops.cpp")
    file(COPY ${CPP} DESTINATION "${CMAKE_CURRENT_BINARY_DIR}/tmp/")
    file(GLOB HEADS "${CMAKE_CURRENT_SOURCE_DIR}/*.h")
    file(COPY ${HEADS} DESTINATION "${CMAKE_CURRENT_BINARY_DIR}/tmp/")
    set(TMP_SOURCE_DIR "${CMAKE_CURRENT_BINARY_DIR}/tmp")
    # Get the file to be compiled,
    # DEV: non-optimised codes (full directory and filename)
    # OPS: codes generated by ops.py (full directory and filename)
    # OTHERS: Source codes need no code generation (full directory and filename)
    file(GLOB DEV "${TMP_SOURCE_DIR}/*.cpp")
    list(FILTER DEV EXCLUDE REGEX ${Odd})
    foreach(oth ${Others})
        list(FILTER DEV EXCLUDE REGEX ${oth})
    endforeach()
    foreach(ext ${Extra})
        list(FILTER DEV EXCLUDE REGEX ${ext})
    endforeach()
    list(GET DEV 0 Kernel)
    get_filename_component(KernerName ${Kernel} NAME_WE)
    if (UNIX)
        execute_process (
            COMMAND ${OPS_C_TRANSLATOR} ${DEV}
            WORKING_DIRECTORY ${TMP_SOURCE_DIR}
        )
    endif()
    file(GLOB OPS "${TMP_SOURCE_DIR}/*ops*.cpp")
    file(GLOB OTHERS "${TMP_SOURCE_DIR}/*.cpp")
    foreach(OpsFile ${OPS})
        list(REMOVE_ITEM OTHERS ${OpsFile})
        string(REPLACE "_ops" "" NoOPS ${OpsFile})
        list(REMOVE_ITEM OTHERS ${NoOPS})
    endforeach()
    list(FILTER OTHERS EXCLUDE REGEX ${Odd})
    foreach(ext ${Extra})
        list(FILTER OTHERS EXCLUDE REGEX ${ext})
    endforeach()
    # Copy input parameters
    file(GLOB_RECURSE INPUT "${CMAKE_CURRENT_SOURCE_DIR}/*.in")
    # OpenCL needs the kenrel codes for first running
    INSTALL(DIRECTORY "${TMP_SOURCE_DIR}/OpenCL" DESTINATION ${APP_INSTALL}/${Name})
    file(COPY ${INPUT} DESTINATION "${TMP_SOURCE_DIR}")
    if (EXISTS "${TMP_SOURCE_DIR}/user_types.h")
        INSTALL(FILES "${TMP_SOURCE_DIR}/user_types.h" DESTINATION ${APP_INSTALL}/${Name})
    endif()
    INSTALL(FILES ${INPUT} DESTINATION ${APP_INSTALL}/${Name})
    add_executable(${Name}_seq_dev ${DEV} ${OTHERS})
    target_include_directories(${Name}_seq_dev PRIVATE ${TMP_SOURCE_DIR})
    add_executable(${Name}_seq ${OPS} ${OTHERS} "${TMP_SOURCE_DIR}/MPI_OpenMP/${KernerName}_cpu_kernels.cpp")
    target_include_directories(${Name}_seq PRIVATE ${TMP_SOURCE_DIR})
    if (HDF5_SEQ)
        target_link_libraries(${Name}_seq ops_hdf5_seq ${HDF5_LIBRARIES}  ${HDF5_HL_LIBRARIES} MPI::MPI_CXX)
        target_link_libraries(${Name}_seq_dev ops_hdf5_seq ${HDF5_LIBRARIES}  ${HDF5_HL_LIBRARIES} MPI::MPI_CXX)
    endif()
    target_link_libraries(${Name}_seq ops_seq)
    target_link_libraries(${Name}_seq_dev ops_seq)
    install(TARGETS ${Name}_seq DESTINATION ${APP_INSTALL}/${Name})
    if (OPS_APP_TEST)
        set(args " ") # here we must have a space to be parsed by the macro
        set(cmd "$<TARGET_FILE:${Name}_seq>" )
        add_test(NAME ${Name}_seq
        COMMAND ${CMAKE_COMMAND} -DCMD=${cmd}  -DARG=${args} -DOPS_INSTALL_PATH=${OPS_INSTALL_PATH}
        -P ${OPS_APP_SRC}/runtests.cmake
        WORKING_DIRECTORY "${TMP_SOURCE_DIR}"
        )
    endif()
    install(TARGETS ${Name}_seq_dev DESTINATION ${APP_INSTALL}/${Name})

    if (CUDA)
        add_executable(${Name}_cuda ${OPS} ${OTHERS} "${TMP_SOURCE_DIR}/CUDA/${KernerName}_kernels.cu")
        target_include_directories(${Name}_cuda PRIVATE ${TMP_SOURCE_DIR})
        target_link_libraries(${Name}_cuda CUDA::cudart_static ops_cuda)
        if (HDF5_SEQ)
            target_link_libraries(${Name}_cuda ops_hdf5_seq ${HDF5_LIBRARIES} ${HDF5_HL_LIBRARIES} MPI::MPI_CXX)
        endif()
        install(TARGETS ${Name}_cuda DESTINATION ${APP_INSTALL}/${Name})
        if ((OPS_APP_TEST) AND (GPU_NUMBER GREATER_EQUAL 1))
            set(args "OPS_CL_DEVICE=1 OPS_BLOCK_SIZE_X=512 OPS_BLOCK_SIZE_Y=1")
            set(cmd "$<TARGET_FILE:${Name}_cuda>")
            add_test(NAME ${Name}_cuda
                    COMMAND ${CMAKE_COMMAND} -DCMD=${cmd}  -DARG=${args} -DOPS_INSTALL_PATH=${OPS_INSTALL_PATH}
                    -P ${OPS_APP_SRC}/runtests.cmake
                    WORKING_DIRECTORY "${TMP_SOURCE_DIR}"
                    )
        endif()
    endif()

    if (OPENCL)
        add_executable(${Name}_opencl ${OPS} ${OTHERS} "${TMP_SOURCE_DIR}/OpenCL/${KernerName}_opencl_kernels.cpp")
        target_include_directories(${Name}_opencl PRIVATE ${TMP_SOURCE_DIR})
        target_link_libraries(${Name}_opencl  ${OpenCL_LIBRARIES} ops_opencl)
        if (HDF5_SEQ)
            target_link_libraries(${Name}_opencl ops_hdf5_seq ${HDF5_LIBRARIES} ${HDF5_HL_LIBRARIES}  MPI::MPI_CXX)
        endif()
        install(TARGETS ${Name}_opencl DESTINATION ${APP_INSTALL}/${Name})
            if ((OPS_APP_TEST) AND (GPU_NUMBER GREATER_EQUAL 1))
            set(args "OPS_CL_DEVICE=1 OPS_BLOCK_SIZE_X=512 OPS_BLOCK_SIZE_Y=1")
            add_test(NAME ${Name}_opencl
            COMMAND ${CMAKE_COMMAND} -DCMD=$<TARGET_FILE:${Name}_opencl>  -DARG=${args} -DOPS_INSTALL_PATH=${OPS_INSTALL_PATH}
            -P ${OPS_APP_SRC}/runtests.cmake
            WORKING_DIRECTORY "${TMP_SOURCE_DIR}"
            )
        endif()
    endif()
    if (MPI)
        add_executable(${Name}_mpi_dev ${DEV} ${OTHERS})
        target_include_directories(${Name}_mpi_dev PRIVATE ${TMP_SOURCE_DIR})
        target_compile_definitions(${Name}_mpi_dev PRIVATE "-DOPS_MPI")
        add_executable(${Name}_mpi ${OPS} ${OTHERS} "${TMP_SOURCE_DIR}/MPI_OpenMP/${KernerName}_cpu_kernels.cpp")
        target_include_directories(${Name}_mpi PRIVATE ${TMP_SOURCE_DIR})
        target_compile_definitions(${Name}_mpi PRIVATE "-DOPS_MPI")
        if (HDF5_MPI)
            target_link_libraries(${Name}_mpi ops_hdf5_mpi ${HDF5_LIBRARIES} ${HDF5_HL_LIBRARIES} MPI::MPI_CXX)
            target_link_libraries(${Name}_mpi_dev ops_hdf5_mpi ${HDF5_LIBRARIES} ${HDF5_HL_LIBRARIES} MPI::MPI_CXX)
        endif()
        target_link_libraries(${Name}_mpi ops_mpi MPI::MPI_CXX)
        set(args "-n ${CPU_NUMBER} $<TARGET_FILE:${Name}_mpi>")
        set(cmd "mpirun")
        add_test(NAME ${Name}_mpi
        COMMAND ${CMAKE_COMMAND} -DCMD=${cmd}  -DARG=${args} -DOPS_INSTALL_PATH=${OPS_INSTALL_PATH}
        -P ${OPS_APP_SRC}/runtests.cmake
        WORKING_DIRECTORY "${TMP_SOURCE_DIR}"
        )
        target_link_libraries(${Name}_mpi_dev ops_mpi MPI::MPI_CXX)
        install(TARGETS ${Name}_mpi DESTINATION ${APP_INSTALL}/${Name})
        install(TARGETS ${Name}_mpi_dev DESTINATION ${APP_INSTALL}/${Name})
        if (CUDA_MPI)
            add_executable(${Name}_mpi_cuda ${OPS} ${OTHERS} "${TMP_SOURCE_DIR}/CUDA/${KernerName}_kernels.cu")
            target_include_directories(${Name}_mpi_cuda PRIVATE ${TMP_SOURCE_DIR})
            target_compile_definitions(${Name}_mpi_cuda PRIVATE "-DOPS_MPI")
            if (HDF5_MPI)
                target_link_libraries(${Name}_mpi_cuda ops_hdf5_mpi ${HDF5_LIBRARIES} ${HDF5_HL_LIBRARIES} MPI::MPI_CXX)
            endif()
            target_link_libraries(${Name}_mpi_cuda CUDA::cudart_static ops_mpi_cuda MPI::MPI_CXX)

            install(TARGETS ${Name}_mpi_cuda DESTINATION ${APP_INSTALL}/${Name})
        endif()

        if (OPENCL_MPI)
            add_executable(${Name}_mpi_opencl ${OPS} ${OTHERS} "${TMP_SOURCE_DIR}/OpenCL/${KernerName}_opencl_kernels.cpp")
            target_include_directories(${Name}_mpi_opencl PRIVATE ${TMP_SOURCE_DIR})
            target_compile_definitions(${Name}_mpi_opencl PRIVATE "-DOPS_MPI")
            if (HDF5_MPI)
                target_link_libraries(${Name}_mpi_opencl ops_hdf5_mpi ${HDF5_LIBRARIES} ${HDF5_HL_LIBRARIES} MPI::MPI_CXX)
            endif()
            target_link_libraries(${Name}_mpi_opencl  ${OpenCL_LIBRARIES} ops_mpi_opencl MPI::MPI_CXX)
            install(TARGETS ${Name}_mpi_opencl DESTINATION ${APP_INSTALL}/${Name})
        endif()
    endif()
endmacro()

# TODO fetch_data.cpp cannot be found for access
#add_subdirectory(access)
# add_subdirectory(CloverLeaf) # Compile OK
# add_subdirectory(CloverLeaf_3D) # Compile OK
# add_subdirectory(CloverLeaf_3D_HDF5) # Compile OK
#TODO there seem bugs in the codes of complex_numbers
#add_subdirectory(complex_numbers)
#TODO Need to figure out the structure
#add_subdirectory(laplace2d_tutorial)
add_subdirectory(lowdim_test) # Compile OK
#TODO no PASSED for testing
add_subdirectory(mblock) # Compile OK
#TODO there seems problem for compiling mblock4D
#add_subdirectory(mblock4D)
#TODO needs to replace  acceptable with PASSED for standard test
#add_subdirectory(mb_shsgc)  # Compile OK
add_subdirectory(mgrid)  # Compile OK
add_subdirectory(multiDim)  # Compile OK
add_subdirectory(multiDim3D)  # Compile OK
#TODO Find a bug of Python translators
#add_subdirectory(multiDim_HDF5)
add_subdirectory(poisson) # Compile OK
add_subdirectory(shsgc) # Compile OK
# add_subdirectory(TeaLeaf) # Compile OK