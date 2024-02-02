cmake_minimum_required(VERSION 3.0)
project(cloverleaf_adjoint LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_EXTENSIONS OFF)

# Add the OPS and Tridsolver dependencies
add_subdirectory(./dependencies/ops)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-O3 -g -DNDEBUG -march=native")#-ip -restrict -xHost")

option(BUILD_FOR_CPU "Build CPU implementations" ON)
option(BUILD_FOR_GPU "Build CUDA implementation" OFF)

if(BUILD_FOR_GPU)
  include(CheckLanguage)
  set(CMAKE_CUDA_STANDARD 14)
  check_language(CUDA)
  if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    find_library(CUDART_LIBRARY cudart)
    set(CUDA_FOUND ON)
    if( NOT CMAKE_CUDA_FLAGS)
       string(APPEND CMAKE_CUDA_FLAGS " -Xptxas -v -arch=sm_60 ")
    endif()
  else()
    message("CUDA could not found BUILD FOR GPU turned OFF")
    set(BUILD_FOR_GPU OFF)
  endif()
endif()


add_subdirectory(src)
