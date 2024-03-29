# To specifiy the cmake version range Currenty work under 3.18
cmake_minimum_required(VERSION 3.18)
project(OPS C CXX)
# if show the compiling process in detail
option(OPS_VERBOSE_WARNING "Turn on verbose warning messages" OFF)
option(OPS_TEST "Turn on tests for Apps" OFF)
option(OPS_HIP "Turn on the HIP backend" OFF)
if(NOT OPS_VERBOSE_WARNING)
  message(
    "We show concise compiling information by defautl! Use -DOPS_VERBOSE_WARNING=ON to switch on."
  )
endif()
set(CMAKE_VERBOSE_MAKEFILE ${OPS_VERBOSE_WARNING})
# Use the Release mode by default
if(NOT CMAKE_BUILD_TYPE)
  message(
    "We choose the Release mode! Please uses -DCMAKE_BUILD_TYPE to change.")
  set(CMAKE_BUILD_TYPE Release)
endif()
message(
  WARNING
    "The installation directory is ${CMAKE_INSTALL_PREFIX}! Use -DCMAKE_INSTALL_PREFIX to change"
)

# Tridiagonal library

if(NOT LIBTRID_PATH)
  message(
    WARNING
      "The install path of the scalar tridiagonal solver library (LIBTRID_PATH) is not specified and it will not compiled!"
  )
endif()

set(HDF5_PREFER_PARALLEL true)
# Configure Compilers C
set(CMAKE_C_STANDARD 99)

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
# include(GenerateExportHeader)
include(CMakePackageConfigHelpers)

set(VERSION latest)
set(ConfigPackageLocation ${CMAKE_INSTALL_PREFIX}/lib/cmake)
# if (CFLAG) add_compile_options(${CFLAG}) endif () Add find scripts
list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)

# Try to find the required dependency
find_package(MPI QUIET)
find_package(HDF5 QUIET COMPONENTS C HL)
find_package(CUDAToolkit QUIET)
find_package(OpenACC QUIET)
find_package(OpenCL QUIET)
find_package(OpenMP QUIET)
find_package(Python3 QUIET)

if(OPS_HIP)
  # Search for rocm in common locations
  list(APPEND CMAKE_PREFIX_PATH /opt/rocm/hip /opt/rocm)
  # Find hip
  find_package(hip)
  # enable_language(HIP) message($ENV{HIP_PLATFORM})
  set(ENV{HIP_PLATFORM} "nvidia")
  # message($ENV{HIP_PLATFORM})
  set(HIP_FOUND TRUE)

  if({$hip::host} STREQUAL "" OR {$hip::device} STREQUAL "")
    message(
      WARNING
        "We cannot find the HIP environment. The HIP codes won't work! You might need to use CMAKE_PREFIX_PATH to specify the path for HIP!"
    )
    set(HIP_FOUND FALSE)
  endif()
endif()

if(NOT Python3_FOUND)
  message(
    FATAL_ERROR
      "We cannot find Python3 and the Python translator needs Python3! Please use -DPython3_EXECUTABLE to specify the path."
  )
endif()
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

if(CUDAToolkit_FOUND)
  set(CMAKE_CUDA_COMPILER ${CUDAToolkit_NVCC_EXECUTABLE})
  if(GPU_ARCH)
    set(CMAKE_CUDA_ARCHITECTURES
        ${GPU_ARCH}
        CACHE STRING "CUDA architectures")
  else()
    message(WARNING "Please the GPU architecture using -DGPU_ARCH=XXX!")
  endif()
  enable_language(CUDA)
else()
  message(WARNING "We cannot find a CUDA kit! The CUDA codes won't compile!")
endif()

if(NOT OpenCL_FOUND)
  message(
    WARNING "We cannot find a OpenCL kit! The OpenCL codes won't compile!")
endif()

if(USE_NCCL)
  find_package(NCCL)
  if(NOT NCCL_FOUND)
    message(
      WARNING
        "We cannot find NCCL! Tridiagonal library wrapper will be compiled with NCCL"
    )
  endif()
endif()

if(LIBTRID_PATH)
  include_directories(${LIBTRID_PATH}/include)
endif()
# option(BUILD_OPS_FORTRAN "Build FORTRAN version of OPS" OFF)
option(BUILD_OPS_APPS "Build Sample OPS applications" OFF)

# option(OPS_ERROR_ON_WARNING "Error upon encountering a warning" ON)
# option(CHECKPOINTING "Enable OPS checkpointing support" OFF)
# option(OPS_INSTALL_TRANSLATOR "Install OPS Source Translator" ON)
# option(OPS_INSTALL_HEADERS "Install OPS Dev Headers" ON)
# option(OPS_INSTALL_LIBS "Install OPS Libraries" ON)

# Turn on all the warnings if(OPS_VERBOSE_WARNINGS) if (WIN32) list(APPEND
# OPS_WARNINGS /W4) list(APPEND OPS_WARNINGS /wd4849) list(APPEND OPS_WARNINGS
# /wd4127) list(APPEND OPS_WARNINGS /D_CRT_SECURE_NO_WARNINGS) else()
# list(APPEND OPS_WARNINGS -Wall -Wextra -pedantic -Wno-unused-function
# -Wno-unused-parameter) endif() foreach(flag IN LISTS OPS_WARNINGS) # NVCC
# complicates life a little here # 1) We want to pass our warning flags to the
# compiler, not NVCC, so #    each need prefaced with -Xcompiler # 2) NVCC
# generates code which is not pedantically compliant, so need #    to filter the
# -pedantic from compiling cuda code
# add_compile_options("$<IF:$<COMPILE_LANGUAGE:CUDA>,$<$<NOT:$<STREQUAL:-pedantic,${flag}>>:SHELL:-Xcompiler
# ${flag}>,${flag}>") endforeach() endif() if(OPS_ERROR_ON_WARNING) if (WIN32)
# set(werrorFlag "/WX") else() set(werrorFlag "-Werror") endif()
# add_compile_options("$<$<COMPILE_LANGUAGE:CUDA>:SHELL:-Xcompiler
# >${werrorFlag}") endif()

# install python translator
add_subdirectory(ops_translator_legacy)
# Configure the C/C++ library compiling
add_subdirectory(ops/c)
# Configure the installation of "include"
file(GLOB_RECURSE INCLUDE "${CMAKE_CURRENT_SOURCE_DIR}/ops/c/include/*.h"
     "${CMAKE_CURRENT_SOURCE_DIR}/ops/fortran/include/*.inc")

install(FILES ${INCLUDE} DESTINATION include)
# prepare the script for setting OPS environment variable variables
if(UNIX)
  install(
    CODE "file(WRITE \"${CMAKE_INSTALL_PREFIX}/bin/ops_env\" \"export OPS_INSTALL_PATH=${CMAKE_INSTALL_PREFIX}\n\")"
  )
  if(NOT ((CMAKE_INSTALL_PREFIX STREQUAL "/usr") OR (CMAKE_INSTALL_PREFIX
                                                     STREQUAL "/usr/local")))
    install(
      CODE "file(APPEND \"${CMAKE_INSTALL_PREFIX}/bin/ops_env\" \"export PATH=$PATH:${CMAKE_INSTALL_PREFIX}/bin\n\")"
    )
  endif()
endif()

# if(BUILD_OPS_FORTRAN) add_subdirectory(ops/fortran) endif()

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
install(FILES "${CMAKE_BINARY_DIR}/OPSConfig.cmake"
        DESTINATION ${ConfigPackageLocation})
if(BUILD_OPS_APPS)
  add_subdirectory(apps/c)
  # if(BUILD_OPS_FORTRAN) add_subdirectory(apps/fortran) endif()
endif()
