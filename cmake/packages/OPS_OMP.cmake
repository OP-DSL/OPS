#Check for OpenMP for CPU: this is activated by defauls with the exception of NVHPC above V23.1

message(STATUS "Set up OMP")
set(USE_OMP TRUE)
if (CMAKE_CXX_COMPILER_ID MATCHES "NVHPC")
  message(STATUS "We are using NVHPC")
  if (CMAKE_CXX_COMPILER_VERSION GREATER "23.1")
    set(USE_OMP FALSE)
    message(STATUS "OpenMP for multicore not enabled")
  endif()
endif()

option(OPS_OMP_CPU "Turn on OpenMP for multicore CPU" ON)
message(STATUS "Setting for option OMP_CPU ${OPS_OMP_CPU}")

find_package(OpenMP QUIET)
if(NOT OpenMP_FOUND)
  message(FATAL_ERROR "OpenMP support NOT FOUND!")
else()
  message(STATUS "OpenMP version ${OpenMP_CXX_VERSION}")
endif()
find_package(OpenMP REQUIRED)

