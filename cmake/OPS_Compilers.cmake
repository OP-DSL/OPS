# Compilers CMakeLists

set(CMAKE_C_STANDARD 99)
set(CMAKE_CXX_STANDARD 11)
set(C_COMPILER_NAME ${CMAKE_C_COMPILER_ID} )
set(CXX_COMPILER_NAME ${CMAKE_CXX_COMPILER_ID} )
message(STATUS "COMP ID C ${C_COMPILER_NAME}")
message(STATUS "C compiler version ${CMAKE_C_COMPILER_VERSION}")

message(STATUS "COMP ID CXX ${C_COMPILER_NAME}")
message(STATUS "CXX compiler version ${CMAKE_CXX_COMPILER_VERSION}")


if (C_COMPILER_NAME MATCHES "GNU")
  message(STATUS "Setting GNU flags")
  include(OPS_flags_gnu)
elseif (C_COMPILER_NAME MATCHES "Intel")
  message(STATUS "Setting Intel flags")
  include(OPS_flags_intel)
elseif (C_COMPILER_NAME MATCHES "NVHPC")
  message(STATUS "Setting NVHPC flags")
  include(OPS_flags_nvidia)
elseif (C_COMPILER_NAME MATCHES "Clang")
  message(STATUS "Setting Clang flags")
  include(OPS_flags_clang)
elseif (C_COMPILER_NAME MATCHES "MSVC")
  message(STATUS "Setting Clang flags")
  include(OPS_flags_clang)
else (C_COMPILER_NAME MATCHES "GNU")
  message ("CMAKE_C_COMPILER full path: " ${CMAKE_C_COMPILER})
  message ("C compiler: " ${C_COMPILER_NAME})
  message ("No optimized C compiler flags are known, we just try -O2...")
  set(OPS_CFLAGS_RELEASE "-O2")
  set(OPS_CFLAGS_DEBUG   "-O0 -g")
endif (C_COMPILER_NAME MATCHES "GNU")

if (NOT FLAGS_SET)
  # C
  set(CMAKE_C_FLAGS ${OPS_CFLAGS} CACHE STRING 
	  "Base CFLAGS for build" FORCE)
  set(CMAKE_C_FLAGS_RELEASE ${OPS_CFLAGS_RELEASE} CACHE STRING
	  "Additional CFLAGS for Release (optimised) build" FORCE)
  set(CMAKE_C_FLAGS_DEBUG ${OPS_CFLAGS_DEBUG} CACHE STRING
	  "Additional CFLAGS for Debug build" FORCE)
  # CXX
  set(CMAKE_CXX_FLAGS ${OPS_CXXFLAGS} CACHE STRING 
	  "Base CXXFLAGS for build" FORCE)
  set(CMAKE_CXX_FLAGS_RELEASE ${OPS_CXXFLAGS_RELEASE} CACHE STRING
	  "Additional CXXFLAGS for Release (optimised) build" FORCE)
  set(CMAKE_CXX_FLAGS_DEBUG ${OPS_CXXFLAGS_DEBUG} CACHE STRING
	  "Additional CXXFLAGS for Debug build" FORCE)
  set(FLAGS_SET 1 CACHE INTERNAL "Flags are set")
endif()

