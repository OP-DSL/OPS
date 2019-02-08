cmake_minimum_required(VERSION 3.0)
project(OPS)

option(BUILD_OPS_FORTRAN "Build FORTRAN version of OPS" OFF)
option(BUILD_OPS_APPS "Build Sample OPS applications" OFF)

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
