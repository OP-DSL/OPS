# Search for rocm in common locations
list(APPEND CMAKE_PREFIX_PATH /opt/rocm/hip /opt/rocm)
# Find hip
find_package(hip)
# enable_language(HIP) message($ENV{HIP_PLATFORM})
set(ENV{HIP_PLATFORM} "nvidia")
# message($ENV{HIP_PLATFORM})
set(HIP_FOUND TRUE)

if({$hip::host} STREQUAL "" OR {$hip::device} STREQUAL "")
  message(
    WARNING
    "HIP support required but not found! You might need to use CMAKE_PREFIX_PATH to specify the path for HIP!"
  )
  set(HIP_FOUND FALSE)
endif()
