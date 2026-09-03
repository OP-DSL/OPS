# Search for rocm in common locations
#list(APPEND CMAKE_PREFIX_PATH /opt/rocm/hip /opt/rocm)
# Find hip
set(ENV{HIP_PLATFORM} "nvidia")
find_package(HIP QUIET)
if(HIP_FOUND)
    message(STATUS "Found HIP: " ${HIP_VERSION})
else()
    message(FATAL_ERROR "Could not find HIP. Ensure that HIP is either installed in /opt/rocm/hip or the variable HIP_PATH is set to point to the right location.")
endif()
enable_language(HIP)
if(NOT DEFINED HIP_PATH)
    if(NOT DEFINED ENV{HIP_PATH})
        set(HIP_PATH "/opt/rocm/hip" CACHE PATH "Path to which HIP has been installed")
    else()
        set(HIP_PATH $ENV{HIP_PATH} CACHE PATH "Path to which HIP has been installed")
    endif()
endif()

if(NOT DEFINED ROCM_PATH)
    if(DEFINED ENV{ROCM_PATH})
        set(ROCM_PATH $ENV{ROCM_PATH} CACHE PATH "Path to which ROCM has been installed")
    elseif(DEFINED ENV{HIP_PATH})
        set(ROCM_PATH "$ENV{HIP_PATH}/.." CACHE PATH "Path to which ROCM has been installed")
    else()
        set(ROCM_PATH "/opt/rocm" CACHE PATH "Path to which ROCM has been installed")
    endif()
endif()

if(NOT DEFINED HCC_PATH)
    if(DEFINED ENV{HCC_PATH})
        set(HCC_PATH $ENV{HCC_PATH} CACHE PATH "Path to which HCC has been installed")
    else()
        set(HCC_PATH "${ROCM_PATH}/hcc" CACHE PATH "Path to which HCC has been installed")
    endif()
    set(HCC_HOME "${HCC_PATH}")
endif()

if(NOT DEFINED HIP_CLANG_PATH)
    if(NOT DEFINED ENV{HIP_CLANG_PATH})
        set(HIP_CLANG_PATH "${ROCM_PATH}/llvm/bin" CACHE PATH "Path to which HIP compatible clang binaries have been installed")
    else()
        set(HIP_CLANG_PATH $ENV{HIP_CLANG_PATH} CACHE PATH "Path to which HIP compatible clang binaries have been installed")
    endif()
endif()

set(CMAKE_MODULE_PATH "${HIP_PATH}/lib/cmake" "${HIP_PATH}/lib/cmake/hip" ${CMAKE_MODULE_PATH})
list(APPEND CMAKE_PREFIX_PATH
    "${HIP_PATH}/lib/cmake"
)
include(FindHIP)

message(STATUS "CMAKE_MODULE_PATH in HIP set ${CMAKE_MODULE_PATH}")

message(STATUS "HIP_PATH: ${HIP_PATH}")
message(STATUS "CMAKE_HIP_PLATFORM ${CMAKE_HIP_PLATFORM}")
message(STATUS "CMAKE_HIP_ARCHITECTURES ${CMAKE_HIP_ARCHITECTURES}")
message(STATUS "HIP_EXTENSIONS ${HIP_EXTENSIONS}")
#message(STATUS "Host and Device $ENV{hip::host} $ENV{hip::device}")
message(STATUS "ENV HIP PLATFORM: $ENV{HIP_PLATFORM}")
set(HIP_FOUND TRUE)

if({$hip::host} STREQUAL "" OR {$hip::device} STREQUAL "")
  message(
    WARNING
    "HIP support required but not found! You might need to use CMAKE_PREFIX_PATH to specify the path for HIP!"
  )
  set(HIP_FOUND FALSE)
endif()
message(STATUS "Status of HIP_FOUND ${HIP_FOUND}")
