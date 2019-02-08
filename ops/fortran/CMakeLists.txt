cmake_minimum_required(VERSION 3.0)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
set(OPS_FORTRAN_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/include")
set(OPS_C_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../c/include")
set(OPS_C_SRC_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../c/src")


add_subdirectory(src)
