# To specifiy the cmake version range
# Currenty work under 3.17.2
cmake_minimum_required(VERSION 3.17)
project(OPS C CXX)
# if show the compiling process in detail
option(OPS_VERBOSE_WARNINGS "Turn on verbose warning messages" OFF)
if (NOT OPS_VERBOSE_WARNINGS)
    message("We show concise compiling information by defautl! Use -DOPS_VERBOSE_WARNING=ON to switch on.")
endif()
set(CMAKE_VERBOSE_MAKEFILE ${OPS_VERBOSE_WARNINGS})
# Use the Release mode by default
if ( NOT CMAKE_BUILD_TYPE )
    message("We choose the Release mode! Please uses -DCMAKE_BUILD_TYPE to change.")
    set(CMAKE_BUILD_TYPE Release)
endif()
message("The installation directory is ${CMAKE_INSTALL_PREFIX}! Use -DCMAKE_INSTALL_PREFIX to change")


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
find_package(MPI QUIET)
find_package(HDF5 QUIET)
find_package(CUDAToolkit QUIET)
find_package(OpenACC QUIET)
find_package(OpenCL QUIET)
find_package(OpenMP QUIET)
find_package(Python2 QUIET)
if (NOT Python2_FOUND)
  message ("We cannot find Python2 and the Python translator needs Python2!")
endif ()
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


# option(BUILD_OPS_FORTRAN "Build FORTRAN version of OPS" OFF)
# option(BUILD_OPS_APPS "Build Sample OPS applications" OFF)

# option(OPS_ERROR_ON_WARNING "Error upon encountering a warning" ON)
# option(CHECKPOINTING "Enable OPS checkpointing support" OFF)
# option(OPS_INSTALL_TRANSLATOR "Install OPS Source Translator" ON)
# option(OPS_INSTALL_HEADERS "Install OPS Dev Headers" ON)
# option(OPS_INSTALL_LIBS "Install OPS Libraries" ON)

# Turn on all the warnings
# if(OPS_VERBOSE_WARNINGS)
#     if (WIN32)
#         list(APPEND OPS_WARNINGS /W4)
#         list(APPEND OPS_WARNINGS /wd4849)
#         list(APPEND OPS_WARNINGS /wd4127)
# 	list(APPEND OPS_WARNINGS /D_CRT_SECURE_NO_WARNINGS)
#     else()
#        list(APPEND OPS_WARNINGS -Wall -Wextra -pedantic -Wno-unused-function -Wno-unused-parameter)
#     endif()
#     foreach(flag IN LISTS OPS_WARNINGS)
#         # NVCC complicates life a little here
#         # 1) We want to pass our warning flags to the compiler, not NVCC, so
#         #    each need prefaced with -Xcompiler
#         # 2) NVCC generates code which is not pedantically compliant, so need
#         #    to filter the -pedantic from compiling cuda code
#         add_compile_options("$<IF:$<COMPILE_LANGUAGE:CUDA>,$<$<NOT:$<STREQUAL:-pedantic,${flag}>>:SHELL:-Xcompiler ${flag}>,${flag}>")
#     endforeach()
# endif()
# if(OPS_ERROR_ON_WARNING)
#     if (WIN32)
#         set(werrorFlag "/WX")
#     else()
#         set(werrorFlag "-Werror")
#     endif()
#     add_compile_options("$<$<COMPILE_LANGUAGE:CUDA>:SHELL:-Xcompiler >${werrorFlag}")
# endif()


#add_subdirectory(ops_translator)
#Configure the C/C++ library compiling
add_subdirectory(ops/c)
#add_subdirectory(apps/c)
# Configure the installation of "include"
file(GLOB_RECURSE INCLUDE "${CMAKE_CURRENT_SOURCE_DIR}/ops/c/include/*.h" "${CMAKE_CURRENT_SOURCE_DIR}/ops/fortran/include/*.inc")
install(FILES ${INCLUDE} DESTINATION include)
#install python translator
INSTALL(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/ops_translator" DESTINATION bin)
#The following operations are OS dependent, we do UNIX first
if (UNIX)
    INSTALL(CODE "execute_process (
        COMMAND ln -s ${CMAKE_INSTALL_PREFIX}/bin/ops_translator/c/ops.py ${CMAKE_INSTALL_PREFIX}/bin/ops_c
    )")
    INSTALL(CODE "execute_process (
        COMMAND chmod a+x ${CMAKE_INSTALL_PREFIX}/bin/ops_translator/c/ops.py
    )")
    INSTALL(CODE "execute_process (
        COMMAND chmod a+x ${CMAKE_INSTALL_PREFIX}/bin/ops_translator/fortran/ops_fortran.py
    )")
    INSTALL(CODE "execute_process (
        COMMAND ln -s ${CMAKE_INSTALL_PREFIX}/bin/ops_translator/fortran/ops_fortran.py ${CMAKE_INSTALL_PREFIX}/bin/ops_fortran
    )")
endif()
#prepare the script for setting OPS environment variable variables
if (UNIX)
    INSTALL(
        CODE "file(WRITE \"${CMAKE_INSTALL_PREFIX}/bin/ops_env\" \"export OPS_INSTALL_PATH=${CMAKE_INSTALL_PREFIX}\n\")"
    )
    if (NOT ((CMAKE_INSTALL_PREFIX STREQUAL "/usr") OR (CMAKE_INSTALL_PREFIX STREQUAL "/usr/local")))
        INSTALL(
            CODE "file(APPEND \"${CMAKE_INSTALL_PREFIX}/bin/ops_env\" \"export PATH=$PATH:${CMAKE_INSTALL_PREFIX}/bin\n\")"
        )
    endif()
endif()

# if(BUILD_OPS_FORTRAN)
#     add_subdirectory(ops/fortran)
# endif()

# if(BUILD_OPS_APPS)
#     add_subdirectory(apps/c)
#     if(BUILD_OPS_FORTRAN)
#         add_subdirectory(apps/fortran)
#     endif()
# endif()