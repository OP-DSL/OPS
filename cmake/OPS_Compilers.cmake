# Compilers CMakeLists

set(CMAKE_CXX_STANDARD 11)
set(CXX_COMPILER_NAME ${CMAKE_CXX_COMPILER_ID} )
message(STATUS "COMP ID CXX ${CXX_COMPILER_NAME}")
message(STATUS "CXX compiler version ${CMAKE_CXX_COMPILER_VERSION}")

if (CXX_COMPILER_NAME MATCHES "GNU")
  message(STATUS "Setting GNU flags")
  include(OPS_flags_gnu)
elseif (CXX_COMPILER_NAME MATCHES "Intel")
  message(STATUS "Setting Intel flags")
  include(OPS_flags_intel)
elseif (CXX_COMPILER_NAME MATCHES "NVHPC")
  message(STATUS "Setting NVHPC flags")
  include(OPS_flags_nvidia)
elseif (CXX_COMPILER_NAME MATCHES "Clang")
  message(STATUS "Setting Clang flags")
  include(OPS_flags_clang)
elseif (CXX_COMPILER_NAME MATCHES "MSVC")
  message(STATUS "Setting MSVC flags")
  include(OPS_flags_msvc)
else (CXX_COMPILER_NAME MATCHES "GNU")
  message ("CMAKE_CXX_COMPILER full path: " ${CMAKE_CXX_COMPILER})
  message ("CXX compiler: " ${CXX_COMPILER_NAME})
  message ("No optimized CXX compiler flags are known, we just try -O2...")
  set(OPS_CXXFLAGS_RELEASE "-O2")
  set(OPS_CXXFLAGS_DEBUG   "-O0 -g")
endif (CXX_COMPILER_NAME MATCHES "GNU")

if (NOT FLAGS_SET)
  set(FLAGS_SET 1 CACHE INTERNAL "Flags are set")
  # CXX
  set(CMAKE_CXX_FLAGS ${OPS_CXXFLAGS} CACHE STRING 
	  "Base CXXFLAGS for build" FORCE)
  set(CMAKE_CXX_FLAGS_RELEASE ${OPS_CXXFLAGS_RELEASE} CACHE STRING
	  "Additional CXXFLAGS for Release (optimised) build" FORCE)
  set(CMAKE_CXX_FLAGS_DEBUG ${OPS_CXXFLAGS_DEBUG} CACHE STRING
	  "Additional CXXFLAGS for Debug build" FORCE)
  # If CUDA found
  if(CUDAToolkit_FOUND)          
    set(CMAKE_CUDA_FLAGS ${OPS_CUDAFLAGS} CACHE STRING 
	"Base CUDAFLAGS for build" FORCE)
    set(CMAKE_CUDA_FLAGS_RELEASE ${OPS_CUDAFLAGS_RELEASE} CACHE STRING
 	"Additional CUDAFLAGS for Release (optimised) build" FORCE)
    set(CMAKE_CUDA_FLAGS_DEBUG ${OPS_CUDAFLAGS_DEBUG} CACHE STRING
	"Additional CUDAFLAGS for Debug build" FORCE)
  endif(CUDAToolkit_FOUND)                                                              
endif()
