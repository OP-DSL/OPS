cmake_minimum_required(VERSION 3.18)
project(APP C CXX)
if(NOT APP_INSTALL_DIR)
  set(APP_INSTALL_DIR "$ENV{HOME}/OPS-APPS")
  message(
    WARNING
      "We will install the applications to ${APP_INSTALL_DIR}! Please use -DAPP_INSTALL_DIR to specify!"
  )
endif()
# if APPs are compiled independently then setup compilers and dependencies.
if(${CMAKE_PROJECT_NAME} STREQUAL APP)
  # if show the compiling process in detail
  option(OPS_VERBOSE_WARNING "Turn on verbose warning messages" OFF)
  option(OPS_TEST "Turn on tests for Apps" OFF)
  if(NOT OPS_VERBOSE_WARNING)
    message(
      "We show concise compiling information by defautl! Use -DOPS_VERBOSE_WARNING=ON to switch on."
    )
  endif()
  set(CMAKE_VERBOSE_MAKEFILE ${OPS_VERBOSE_WARNING})
  set(OPS_APP_SRC ${CMAKE_SOURCE_DIR})
  # Use the Release mode by default
  if(NOT CMAKE_BUILD_TYPE)
    message(
      "We choose the Release mode! Please uses -DCMAKE_BUILD_TYPE to change.")
    set(CMAKE_BUILD_TYPE Release)
  endif()

  if(NOT LIBTRID_PATH)
    message(
      WARNING
        "The install path of the scalar tridiagonal solver library (LIBTRID_PATH) is not specified and the apps which need it will not compiled!"
    )
  endif()
  # Configure Compilers C
  set(CMAKE_C_STANDARD 99)
  # TODO:Shall we keep the "-g" in the release mode? It increases file size.
  if(${CMAKE_C_COMPILER_ID} STREQUAL GNU)
    set(CMAKE_C_FLAGS "-fPIC -Wall")
    set(CMAKE_C_FLAGS_RELEASE "-O3")
    set(CMAKE_C_FLAGS_DEBUG "-O0 -g")
  elseif(${CMAKE_C_COMPILER_ID} STREQUAL Clang)
    set(CMAKE_C_FLAGS "-fPIC -DUNIX -Wall")
    set(CMAKE_C_FLAGS_RELEASE "-O3")
    set(CMAKE_C_FLAGS_DEBUG "-g -O0")
  elseif(${CMAKE_C_COMPILER_ID} STREQUAL MSVC)
    set(CMAKE_C_FLAGS "-W1")
    set(CMAKE_C_FLAGS_RELEASE "/O2 /Ot ")
    set(CMAKE_C_FLAGS_DEBUG "/Z7 /DEBUG")
  elseif(${CMAKE_C_COMPILER_ID} STREQUAL Intel)
    set(CMAKE_C_FLAGS "-xHost")
    set(CMAKE_C_FLAGS_RELEASE "-O3")
    set(CMAKE_C_FLAGS_DEBUG "-O0 -g")
  else()
    message(
      WARNING
        "There is no default flag for the chosen C compiler! Please use -DCFLAG=XXX to supply!"
    )
  endif()

  # C++
  set(CMAKE_CXX_STANDARD 11)
  if(${CMAKE_CXX_COMPILER_ID} STREQUAL GNU)
    set(CMAKE_CXX_FLAGS "-fPIC -Wall")
    set(CMAKE_CXX_FLAGS_RELEASE "-O3")
    set(CMAKE_CXX_FLAGS_DEBUG "-g -O0")
  elseif(${CMAKE_CXX_COMPILER_ID} STREQUAL Clang)
    set(CMAKE_CXX_FLAGS "-fPIC -DUNIX -Wall")
    set(CMAKE_CXX_FLAGS_RELEASE "-O3")
    set(CMAKE_CXX_FLAGS_DEBUG "-g -O0")
  elseif(${CMAKE_CXX_COMPILER_ID} STREQUAL MSVC)
    set(CMAKE_CXX_FLAGS "-W1")
    set(CMAKE_CXX_FLAGS_RELEASE "/O2 /Ot ")
    set(CMAKE_CXX_FLAGS_DEBUG "/Z7 /DEBUG")
  elseif(${CMAKE_CXX_COMPILER_ID} STREQUAL Intel)
    set(CMAKE_CXX_FLAGS "-xHost")
    set(CMAKE_CXX_FLAGS_RELEASE "-O3")
    set(CMAKE_CXX_FLAGS_DEBUG " -O0 -g -DOPS_DEBUG ")
  else()
    message(
      WARNING
        "There is no default flag for the chosen C++ compiler! Please use -DCXXFLAG=XXX to supply!"
    )
  endif()

  if(CFLAG)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${CFLAG}")
  else()
    message(
      "Extra C compiler options by using CFLAG, e.g., -DCFLAG=\"-ffloat-store\" for GNU!"
    )
  endif()

  if(CXXFLAG)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${CXXFLAG}")
  else()
    message(
      "Extra C++ compiler options by using CXXFLAG, e.g., -DCXXFLAG=\"-ffloat-store\" for GNU!"
    )
  endif()

  # Trying to find the OPS library TODO this shall become FindOPS.cmake later
  set(SEARCH_PATHS "${OPS_INSTALL_DIR}" "/usr/local" "/usr")
  # Tyring to find the include directory
  set(OPS_HEAD
      ops_checkpointing.h
      ops_hdf5.h
      ops_lib_core.h
      ops_opencl_reduction.h
      ops_tridiag.h
      ops_cuda_reduction.h
      ops_instance.h
      ops_device_rt_support.h
      ops_util.h
      ops_device_rt_support.h
      ops_internal1.h
      ops_macros.h
      ops_seq.h
      queue.h
      ops_exceptions.h
      ops_internal2.h
      ops_mpi_core.h
      ops_seq_v2.h)
  set(OPS_INCLUDE_DIR "")
  foreach(head ${OPS_HEAD})
    unset(head_inc CACHE)
    find_path(
      head_inc
      NAMES ${head}
      PATHS ${SEARCH_PATHS}
      PATH_SUFFIXES include)
    if(head_inc)
      list(APPEND OPS_INCLUDE_DIR ${head_inc})
    else()
      message(FATAL_ERROR "We cannot find ${head}")
    endif()
  endforeach()
  list(REMOVE_DUPLICATES OPS_INCLUDE_DIR)
  include_directories("${OPS_INCLUDE_DIR}")
  # Trying to find the library directory
  if(LIBTRID_PATH)
    set(OPS_LIB
        ops_cuda
        ops_hdf5_seq
        ops_mpi_cuda
        ops_opencl
        ops_hdf5_mpi
        ops_mpi
        ops_mpi_opencl
        ops_seq
        ops_trid_seq
        ops_trid_cuda
        ops_trid_mpi
        ops_trid_mpi_cuda)
  else()
    set(OPS_LIB
        ops_cuda
        ops_hdf5_seq
        ops_mpi_cuda
        ops_opencl
        ops_hdf5_mpi
        ops_mpi
        ops_mpi_opencl
        ops_seq)
  endif()
  foreach(library ${OPS_LIB})
    find_library(
      ${library}_FOUND
      NAMES ${library}
      PATHS ${SEARCH_PATHS}
      PATH_SUFFIXES lib)
    if(${library}_FOUND)
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
  #find_package(OpenACC QUIET)
  #find_package(OpenCL QUIET)
  find_package(OpenMP QUIET)
  find_package(Python3 REQUIRED)
  if(NOT Python3_FOUND)
    message(
      FATAL_ERROR
        "We cannot find Python3 and the Python translator needs Python3! Please use -DPython3_EXECUTABLE to specify the path."
    )
  else()
    find_path(
      opsc
      NAMES "ops.py"
      PATHS ${SEARCH_PATHS}
      PATH_SUFFIXES bin/ops_translator_legacy/c)
    find_path(
      opsfortran
      NAMES "ops_fortran.py"
      PATHS ${SEARCH_PATHS}
      PATH_SUFFIXES bin/ops_translator_legacy/fortran)
    if(NOT opsc)
      message(FATAL_ERROR "We cannot find the python C/C++ translator!")
    endif()
    if(NOT opsfortran)
      message(FATAL_ERROR "We cannot find the python fortran translator!")
    endif()
  endif()
  set(OPS_C_TRANSLATOR "${opsc}/ops.py")
  set(OPS_F_TRANSLATOR "${opsfortran}/ops_fortran.py")
  # Configure the "include" dir for compiling
  if(NOT HDF5_FOUND)
    message(
      WARNING
        "We cannot find the parallel HDF5 library. The HDF5 IO routines won't work! Please use -DHDF5_ROOT to specify the path!"
    )
  endif()
  if(NOT MPI_FOUND)
    message(
      WARNING "We cannot find a MPI environment! The MPI codes won't compile!")
  endif()

  if(LIBTRID_PATH)
    include_directories(${LIBTRID_PATH}/include)
  endif()

  if(CUDAToolkit_FOUND)
    set(CMAKE_CUDA_COMPILER ${CUDAToolkit_NVCC_EXECUTABLE})
    if(GPU_ARCH)
      set(CMAKE_CUDA_ARCHITECTURES
          70
          CACHE STRING "CUDA architectures")
    else()
      message(WARNING "Please the GPU architecture using -DGPU_ARCH=XXX!")
    endif()
    enable_language(CUDA)
  else()
    message(WARNING "We cannot find a CUDA kit! The CUDA codes won't compile!")
  endif()

  #if(NOT OpenCL_FOUND)
  #  message(
  #    WARNING "We cannot find a OpenCL kit! The OpenCL codes won't compile!")
  #endif()

  if(USE_NCCL)
    find_package(NCCL)
    if(NOT NCCL_FOUND)
      message(
        WARNING
          "We cannot find NCCL! Tridiagonal library wrapper will be compiled with NCCL"
      )
    endif()
  endif()

  set(HDF5_SEQ ${HDF5_FOUND})
  if(ops_cuda_FOUND)
    set(CUDA ${CUDAToolkit_FOUND})
  endif()
  #if(ops_opencl_FOUND)
  #  set(OPENCL ${OpenCL_FOUND})
  #endif()
  if(${MPI_FOUND})
    set(MPI TRUE)
    set(HDF5_MPI ${HDF5_FOUND})
    if(ops_cuda_FOUND)
      set(CUDA_MPI ${CUDAToolkit_FOUND})
    endif()
    if(ops_opencl_FOUND)
      set(OPENCL_MPI ${OpenCL_FOUND})
    endif()
  endif()
  # prepare ctest
  if(OPS_TEST)
    enable_testing()
    include(ProcessorCount)
    ProcessorCount(CPU_NUMBER)
    if(CPU_NUMBER GREATER 12)
      set(CPU_NUMBER 12)
    endif()
    if(NOT GPU_NUMBER)
      message(
        WARNING "GPU tests are disabled! Please use -DGPU_NUMBER to specify!")
    endif()
  endif()
endif()
# if APPs are compiled together with the OPS API
if(${CMAKE_PROJECT_NAME} STREQUAL OPS)
  include_directories("${CMAKE_SOURCE_DIR}/ops/c/include")
  set(HDF5_SEQ ${HDF5_FOUND})
  set(CUDA ${CUDAToolkit_FOUND})
  #set(OPENCL ${OpenCL_FOUND})
  if(OPS_HIP)
    set(HIP ${HIP_FOUND})
  endif()
  if(${MPI_FOUND})
    set(MPI TRUE)
    set(HDF5_MPI ${HDF5_FOUND})
    set(CUDA_MPI ${CUDAToolkit_FOUND})
    #set(OPENCL_MPI ${OpenCL_FOUND})
  endif()
  if(NOT Python3_FOUND)
    message(
      FATAL_ERROR
        "We cannot find Python3 and the Python translator needs Python3!")
  else()
    set(OPS_C_TRANSLATOR "${CMAKE_SOURCE_DIR}/ops_translator_legacy/c/ops.py")
    set(OPS_F_TRANSLATOR "${opsfortran}/ops_translator_legacy/fortran/ops_fortran.py")
  endif()
  # set the OPS_INSTALL_PATH for testing
  set(OPS_INSTALL_PATH "${CMAKE_SOURCE_DIR}/ops/c")
  set(OPS_APP_SRC ${CMAKE_SOURCE_DIR}/apps/c)
  if(OPS_TEST)
    enable_testing()
  endif()
endif()

# clean the tmp directory for the code generation
macro(CreateTempDir)
  file(REMOVE_RECURSE "${CMAKE_CURRENT_BINARY_DIR}/tmp")
  file(MAKE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/tmp")
endmacro()

# Prepare the macro for compiling apps Name: App name Odd: Key words for source
# files are included in other source files and need no explicit compilation
# Others: Key words for source codes need no code generation using ops.py Extra:
# Key words for ource codes that cannot be treated by the macro Trid: Whether
# the tridiagonal library is required for this application GenerateTest: if
# generating the testings, some apps need specific testing targets. TODO
# MPI_inline, titled OpenACC (Gnu?)
macro(
  BUILD_OPS_C_SAMPLE
  Name
  Odd
  Others
  Extra
  Trid
  GenerateTest)
  # Copy all source and head files into tmp
  file(GLOB CPP "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp")
  list(FILTER CPP EXCLUDE REGEX "ops.cpp")
  file(COPY ${CPP} DESTINATION "${CMAKE_CURRENT_BINARY_DIR}/tmp/")
  file(GLOB HEADS "${CMAKE_CURRENT_SOURCE_DIR}/*.h")
  file(COPY ${HEADS} DESTINATION "${CMAKE_CURRENT_BINARY_DIR}/tmp/")
  set(TMP_SOURCE_DIR "${CMAKE_CURRENT_BINARY_DIR}/tmp")
  # Get the file to be compiled, DEV: non-optimised codes (full directory and
  # filename) OPS: codes generated by ops.py (full directory and filename)
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
  execute_process(COMMAND ${OPS_C_TRANSLATOR} ${DEV}
                  WORKING_DIRECTORY ${TMP_SOURCE_DIR})
  file(GLOB OPS "${TMP_SOURCE_DIR}/*ops*.cpp")
  # message(${OPS})
  list(FILTER OPS EXCLUDE REGEX ${Odd})
  # message(${OPS})
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
  install(DIRECTORY "${TMP_SOURCE_DIR}/OpenCL"
          DESTINATION ${APP_INSTALL_DIR}/${Name})
  file(COPY ${INPUT} DESTINATION "${TMP_SOURCE_DIR}")
  if(EXISTS "${TMP_SOURCE_DIR}/user_types.h")
    install(FILES "${TMP_SOURCE_DIR}/user_types.h"
            DESTINATION ${APP_INSTALL_DIR}/${Name})
  endif()
  install(FILES ${INPUT} DESTINATION ${APP_INSTALL_DIR}/${Name})
  add_executable(${Name}_seq_dev ${DEV} ${OTHERS})
  target_include_directories(${Name}_seq_dev PRIVATE ${TMP_SOURCE_DIR})
  add_executable(
    ${Name}_seq ${OPS} ${OTHERS}
                "${TMP_SOURCE_DIR}/mpi_openmp/mpi_openmp_kernels.cpp")
  target_include_directories(${Name}_seq PRIVATE ${TMP_SOURCE_DIR})
  if(HDF5_SEQ)
    target_link_libraries(${Name}_seq ops_hdf5_seq hdf5::hdf5 hdf5::hdf5_hl
                          MPI::MPI_CXX)
    target_link_libraries(${Name}_seq_dev ops_hdf5_seq hdf5::hdf5 hdf5::hdf5_hl
                          MPI::MPI_CXX)
  endif()
  if(${Trid} STREQUAL "YES")
    set(TRID Yes)
  else()
    set(TRID No)
  endif()
  if(TRID)
    target_link_libraries(${Name}_seq ops_trid_seq -L${LIBTRID_PATH}/lib
                          -ltridcpu)
    target_link_libraries(${Name}_seq_dev ops_trid_seq -L${LIBTRID_PATH}/lib
                          -ltridcpu)
  endif()
  target_link_libraries(${Name}_seq ops_seq OpenMP::OpenMP_CXX)
  target_link_libraries(${Name}_seq_dev ops_seq OpenMP::OpenMP_CXX)
  install(TARGETS ${Name}_seq DESTINATION ${APP_INSTALL_DIR}/${Name})
  # Sequential tests disabled since it is too time consuming for some tests at
  # this moment if ((OPS_TEST) AND (${GenerateTest} STREQUAL "YES")) set(args "
  # ") # here we must have a space to be parsed by the macro set(cmd
  # "$<TARGET_FILE:${Name}_seq>" ) add_test(NAME ${Name}_seq COMMAND
  # ${CMAKE_COMMAND} -DCMD=${cmd}  -DARG=${args}
  # -DOPS_INSTALL_PATH=${OPS_INSTALL_PATH} -P ${OPS_APP_SRC}/runtests.cmake
  # WORKING_DIRECTORY "${TMP_SOURCE_DIR}" ) endif()
  install(TARGETS ${Name}_seq_dev DESTINATION ${APP_INSTALL_DIR}/${Name})

  if(CUDA)
    add_executable(
      ${Name}_cuda ${OPS} ${OTHERS}
                   "${TMP_SOURCE_DIR}/cuda/cuda_kernels.cu")
    target_include_directories(${Name}_cuda PRIVATE ${TMP_SOURCE_DIR})
    target_link_libraries(${Name}_cuda ops_cuda CUDA::cudart_static CUDA::curand
                          OpenMP::OpenMP_CXX)
    target_compile_options(${Name}_cuda PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
                                                --fmad=false>)
    if(HDF5_SEQ)
      target_link_libraries(${Name}_cuda ops_hdf5_seq hdf5::hdf5 hdf5::hdf5_hl
                            MPI::MPI_CXX)
    endif()
    if(TRID)
      target_link_libraries(
        ${Name}_cuda
        ops_trid_cuda
        -L${LIBTRID_PATH}/lib
        -ltridcuda
        CUDA::cublas_static
        CUDA::cublasLt_static
        CUDA::culibos)
    endif()
    install(TARGETS ${Name}_cuda DESTINATION ${APP_INSTALL_DIR}/${Name})
    if((OPS_TEST)
       AND (GPU_NUMBER GREATER_EQUAL 1)
       AND (${GenerateTest} STREQUAL "YES"))
      set(args "OPS_CL_DEVICE=1 OPS_BLOCK_SIZE_X=512 OPS_BLOCK_SIZE_Y=1")
      set(cmd "$<TARGET_FILE:${Name}_cuda>")
      add_test(
        NAME ${Name}_cuda
        COMMAND
          ${CMAKE_COMMAND} -DCMD=${cmd} -DARG=${args}
          -DOPS_INSTALL_PATH=${OPS_INSTALL_PATH} -P
          ${OPS_APP_SRC}/runtests.cmake
        WORKING_DIRECTORY "${TMP_SOURCE_DIR}")
    endif()
  endif()

  # Tridiagonal library does not currently support OpenCL
  #if(OPENCL AND NOT TRID)
  #  add_executable(
  #    ${Name}_opencl
  #    ${OPS} ${OTHERS}
  #    "${TMP_SOURCE_DIR}/opencl/opencl_kernels.cpp")
  #  target_include_directories(${Name}_opencl PRIVATE ${TMP_SOURCE_DIR}
  #                                                    ${OpenCL_INCLUDE_DIRS})
  #  target_link_libraries(${Name}_opencl ops_opencl ${OpenCL_LIBRARIES}
  #                        OpenMP::OpenMP_CXX)
  #  if(HDF5_SEQ)
  #    target_link_libraries(${Name}_opencl ops_hdf5_seq hdf5::hdf5
  #                          hdf5::hdf5_hl MPI::MPI_CXX)
  #  endif()
  #  install(TARGETS ${Name}_opencl DESTINATION ${APP_INSTALL_DIR}/${Name})
  #  if((OPS_TEST)
  #     AND (GPU_NUMBER GREATER_EQUAL 1)
  #     AND (${GenerateTest} STREQUAL "YES"))
  #    set(args "OPS_CL_DEVICE=1 OPS_BLOCK_SIZE_X=512 OPS_BLOCK_SIZE_Y=1")
  #    add_test(
  #      NAME ${Name}_opencl
  #      COMMAND
  #        ${CMAKE_COMMAND} -DCMD=$<TARGET_FILE:${Name}_opencl> -DARG=${args}
  #        -DOPS_INSTALL_PATH=${OPS_INSTALL_PATH} -P
  #        ${OPS_APP_SRC}/runtests.cmake
  #      WORKING_DIRECTORY "${TMP_SOURCE_DIR}")
  #  endif()
  #endif()
  if(OPS_HIP)
    if(HIP AND NOT TRID)
      add_executable(
        ${Name}_hip ${OPS} ${OTHERS}
                    "${TMP_SOURCE_DIR}/hip/hip_kernels.cpp")
      message("${TMP_SOURCE_DIR}/hip/${KernerName}_hip_kernel.cpp")
      target_include_directories(${Name}_hip PRIVATE ${TMP_SOURCE_DIR})
      target_link_libraries(${Name}_hip ops_hip hip::device OpenMP::OpenMP_CXX)
      # if (HDF5_SEQ) target_link_libraries(${Name}_hip ops_hdf5_seq hdf5::hdf5
      # hdf5::hdf5_hl  MPI::MPI_CXX) endif()
      install(TARGETS ${Name}_hip DESTINATION ${APP_INSTALL_DIR}/${Name})
      if((OPS_TEST)
         AND (GPU_NUMBER GREATER_EQUAL 1)
         AND (${GenerateTest} STREQUAL "YES"))
        set(args "OPS_CL_DEVICE=1 OPS_BLOCK_SIZE_X=512 OPS_BLOCK_SIZE_Y=1")
        add_test(
          NAME ${Name}_hip
          COMMAND
            ${CMAKE_COMMAND} -DCMD=$<TARGET_FILE:${Name}_opencl> -DARG=${args}
            -DOPS_INSTALL_PATH=${OPS_INSTALL_PATH} -P
            ${OPS_APP_SRC}/runtests.cmake
          WORKING_DIRECTORY "${TMP_SOURCE_DIR}")
      endif()
    endif()
  endif()

  if(MPI)
    add_executable(${Name}_mpi_dev ${DEV} ${OTHERS})
    target_include_directories(${Name}_mpi_dev PRIVATE ${TMP_SOURCE_DIR})
    target_compile_definitions(${Name}_mpi_dev PRIVATE "-DOPS_MPI")
    add_executable(
      ${Name}_mpi ${OPS} ${OTHERS}
                  "${TMP_SOURCE_DIR}/mpi_openmp/mpi_openmp_kernels.cpp")
    target_include_directories(${Name}_mpi PRIVATE ${TMP_SOURCE_DIR})
    target_compile_definitions(${Name}_mpi PRIVATE "-DOPS_MPI")
    if(HDF5_MPI)
      target_link_libraries(${Name}_mpi PRIVATE ops_hdf5_mpi hdf5::hdf5
                                                hdf5::hdf5_hl MPI::MPI_CXX)
      target_link_libraries(${Name}_mpi_dev PRIVATE ops_hdf5_mpi hdf5::hdf5
                                                    hdf5::hdf5_hl MPI::MPI_CXX)
    endif()
    if(TRID)
      target_link_libraries(
        ${Name}_mpi PRIVATE ops_trid_mpi -L${LIBTRID_PATH}/lib -ltridcpu_mpi
                            MPI::MPI_CXX)
      target_link_libraries(
        ${Name}_mpi_dev PRIVATE ops_trid_mpi -L${LIBTRID_PATH}/lib
                                -ltridcpu_mpi MPI::MPI_CXX)
    endif()
    target_link_libraries(${Name}_mpi PRIVATE ops_mpi MPI::MPI_CXX
                                              OpenMP::OpenMP_CXX)
    target_link_libraries(${Name}_mpi_dev PRIVATE ops_mpi MPI::MPI_CXX
                                                  OpenMP::OpenMP_CXX)
    if((OPS_TEST) AND (${GenerateTest} STREQUAL "YES"))
      set(args "-n ${CPU_NUMBER} $<TARGET_FILE:${Name}_mpi>")
      set(cmd "mpirun")
      add_test(
        NAME ${Name}_mpi
        COMMAND
          ${CMAKE_COMMAND} -DCMD=${cmd} -DARG=${args}
          -DOPS_INSTALL_PATH=${OPS_INSTALL_PATH} -P
          ${OPS_APP_SRC}/runtests.cmake
        WORKING_DIRECTORY "${TMP_SOURCE_DIR}")
    endif()
    install(TARGETS ${Name}_mpi DESTINATION ${APP_INSTALL_DIR}/${Name})
    install(TARGETS ${Name}_mpi_dev DESTINATION ${APP_INSTALL_DIR}/${Name})
    if(CUDA_MPI)
      add_executable(
        ${Name}_mpi_cuda ${OPS} ${OTHERS}
                         "${TMP_SOURCE_DIR}/cuda/cuda_kernels.cu")
      target_include_directories(${Name}_mpi_cuda PRIVATE ${TMP_SOURCE_DIR})
      target_compile_definitions(${Name}_mpi_cuda PRIVATE "-DOPS_MPI")
      target_compile_options(${Name}_mpi_cuda
                             PRIVATE $<$<COMPILE_LANGUAGE:CUDA>: --fmad=false >)
      if(HDF5_MPI)
        target_link_libraries(
          ${Name}_mpi_cuda PRIVATE ops_hdf5_mpi hdf5::hdf5 hdf5::hdf5_hl
                                   MPI::MPI_CXX)
      endif()
      if(TRID)
        target_link_libraries(
          ${Name}_mpi_cuda PRIVATE ops_trid_mpi_cuda -L${LIBTRID_PATH}/lib
                                   -ltridcuda_mpi MPI::MPI_CXX)
        if(NCCL_FOUND)
          target_link_libraries(${Name}_mpi_cuda PRIVATE ${NCCL_LIBRARY})
        endif()
      endif()
      target_link_libraries(
        ${Name}_mpi_cuda PRIVATE ops_mpi_cuda CUDA::cudart_static CUDA::curand MPI::MPI_CXX
                                 OpenMP::OpenMP_CXX)
      install(TARGETS ${Name}_mpi_cuda DESTINATION ${APP_INSTALL_DIR}/${Name})
    endif()

    #if(OPENCL_MPI AND NOT TRID)
    #  add_executable(
    #    ${Name}_mpi_opencl
    #    ${OPS} ${OTHERS}
    #    "${TMP_SOURCE_DIR}/opencl/opencl_kernels.cpp")
    #  target_include_directories(
    #    ${Name}_mpi_opencl PRIVATE ${TMP_SOURCE_DIR} ${OpenCL_INCLUDE_DIRS})
    #  target_compile_definitions(${Name}_mpi_opencl PRIVATE "-DOPS_MPI")
    #  if(HDF5_MPI)
    #    target_link_libraries(
    #      ${Name}_mpi_opencl PRIVATE ops_hdf5_mpi hdf5::hdf5 hdf5::hdf5_hl
    #                                 MPI::MPI_CXX)
    #  endif()
    #  target_link_libraries(
    #    ${Name}_mpi_opencl PRIVATE ops_mpi_opencl ${OpenCL_LIBRARIES}
    #                               MPI::MPI_CXX OpenMP::OpenMP_CXX)
    #  install(TARGETS ${Name}_mpi_opencl DESTINATION ${APP_INSTALL_DIR}/${Name})
    #endif()
  endif()
endmacro()

# TODO fetch_data.cpp cannot be found for access add_subdirectory(access)
if(LIBTRID_PATH)
  add_subdirectory(adi)
  add_subdirectory(adi_burger)
  add_subdirectory(adi_burger_3D)
endif()

add_subdirectory(CloverLeaf) # Compile OK
#add_subdirectory(CloverLeaf_3D) # Compile OK
#add_subdirectory(CloverLeaf_3D_HDF5) # Compile OK
## TODO there seem bugs in the codes of complex_numbers
## add_subdirectory(complex_numbers) TODO Need to figure out the structure
## add_subdirectory(laplace2d_tutorial)
#add_subdirectory(lowdim_test)
## TODO no PASSED for testing
#add_subdirectory(mblock)
## TODO there seems problem for compiling mblock4D add_subdirectory(mblock4D) #
## ignored at this moment TODO needs to replace  acceptable with PASSED for
## standard test
#add_subdirectory(mb_shsgc)
## add_subdirectory(mgrid)  # Compile OK
#add_subdirectory(multiDim)
#add_subdirectory(multiDim3D)
#add_subdirectory(multiDim_HDF5)
#add_subdirectory(poisson)
#add_subdirectory(shsgc)
#add_subdirectory(TeaLeaf)
#add_subdirectory(compact_scheme)
#add_subdirectory(hdf5_slice)
