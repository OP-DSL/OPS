# To specifiy the cmake version range
# Currenty work under 3.17.2
cmake_minimum_required(VERSION 3.17)
project(OPS)
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

# option(BUILD_OPS_FORTRAN "Build FORTRAN version of OPS" OFF)
# option(BUILD_OPS_APPS "Build Sample OPS applications" OFF)
# option(OPS_VERBOSE_WARNINGS "Turn on verbose warning messages" ON)
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
# Configure the installation of "include"
file(GLOB_RECURSE INCLUDE "${CMAKE_CURRENT_SOURCE_DIR}/ops/c/include/*.h")
install(FILES ${INCLUDE} DESTINATION include)

# if(BUILD_OPS_FORTRAN)
#     add_subdirectory(ops/fortran)
# endif()

# if(BUILD_OPS_APPS)
#     add_subdirectory(apps/c)
#     if(BUILD_OPS_FORTRAN)
#         add_subdirectory(apps/fortran)
#     endif()
# endif()