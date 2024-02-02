cmake_minimum_required(VERSION 3.0)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
set(OPS_C_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/include" CACHE PATH "Path to OPS C Includes")

set(FLAGS_INTEL_RELWITHDEBINFO "-O2 -g -DNDEBUG -march=native")#-ip -restrict -xHost") #-fp-model fast
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO   " ${FLAGS_INTEL_RELWITHDEBINFO}  ")
set(CMAKE_CXX_FLAGS_DEBUG   "${CMAKE_CXX_FLAGS_DEBUG} -O0 ")

add_subdirectory(src)
