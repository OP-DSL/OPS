cmake_minimum_required(VERSION 3.0)
project(OPS)

option(BUILD_OPS_FORTRAN "Build FORTRAN version of OPS" OFF)
option(BUILD_OPS_APPS "Build Sample OPS applications" OFF)
option(OPS_VERBOSE_WARNINGS "Turn on verbose warning messages" ON)
option(OPS_ERROR_ON_WARNING "Error upon encountering a warning" ON)
option(CHECKPOINTING "Enable OPS checkpointing support" OFF)
option(OPS_INSTALL_TRANSLATOR "Install OPS Source Translator" ON)
option(OPS_INSTALL_HEADERS "Install OPS Dev Headers" ON)
option(OPS_INSTALL_LIBS "Install OPS Libraries" ON)

# Enable C++11 standard
set(CMAKE_CXX_STANDARD 11)

# Turn on all the warnings
if(OPS_VERBOSE_WARNINGS)
    if (WIN32)
        list(APPEND OPS_WARNINGS /W4)
        list(APPEND OPS_WARNINGS /wd4849)
        list(APPEND OPS_WARNINGS /wd4127)
	list(APPEND OPS_WARNINGS /D_CRT_SECURE_NO_WARNINGS)
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
if(OPS_ERROR_ON_WARNING)
    if (WIN32)
        set(werrorFlag "/WX")
    else()
        set(werrorFlag "-Werror")
    endif()
    add_compile_options("$<$<COMPILE_LANGUAGE:CUDA>:SHELL:-Xcompiler >${werrorFlag}")
endif()


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
