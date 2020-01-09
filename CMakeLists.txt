cmake_minimum_required(VERSION 3.0)
project(OPS)

option(BUILD_OPS_FORTRAN "Build FORTRAN version of OPS" OFF)
option(BUILD_OPS_APPS "Build Sample OPS applications" OFF)
option(OPS_VERBOSE_WARNINGS "Turn on verbose warning messages" ON)
option(CHECKPOINTING "Enable OPS checkpointing support" OFF)

# Enable C++11 standard
set(CMAKE_CXX_STANDARD 11)

# Turn on all the warnings
if(OPS_VERBOSE_WARNINGS)
    if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "MSVC")
        list(APPEND OPS_WARNINGS /W4)
    else()
       list(APPEND OPS_WARNINGS -Wall -Wextra -pedantic -Wno-unused-function -Wno-unused-parameter)
    endif()
    foreach(flag IN LISTS OPS_WARNINGS)
        # NVCC complicates life a little here
        # 1) We want to pass our warning flags to the compiler, not NVCC, so 
        #    each need prefaced with -Xcompiler
        # 2) NVCC generates code which is not pedantically compliant, so need 
        #    to filter the -pedantic from compiling cuda code
        add_compile_options("$<IF:$<COMPILE_LANGUAGE:CUDA>,$<$<NOT:$<STREQUAL:-pedantic,${flag}>>:SHELL:-Xcompiler ${flag}>,${flag}>")
    endforeach()
endif()
 if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "MSVC")
     set(werrorFlag "/WX")
 else()
     set(werrorFlag "-Werror")
 endif()
 add_compile_options("$<$<COMPILE_LANGUAGE:CUDA>:SHELL:-Xcompiler >${werrorFlag}")


add_subdirectory(ops_translator)
add_subdirectory(ops/c)

if(BUILD_OPS_FORTRAN)
    add_subdirectory(ops/fortran)
endif()

if(BUILD_OPS_APPS)
    add_subdirectory(apps/c)
    if(BUILD_OPS_FORTRAN)
        add_subdirectory(apps/fortran)
    endif()
endif()
