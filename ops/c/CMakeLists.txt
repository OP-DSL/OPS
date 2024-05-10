set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
include(OPS_InstallMacros)

# List the core subrotines
file(GLOB_RECURSE CORE "${CMAKE_CURRENT_SOURCE_DIR}/src/core/*.cpp")
file(GLOB_RECURSE SEQ "${CMAKE_CURRENT_SOURCE_DIR}/src/sequential/*.cpp")
file(GLOB_RECURSE EXTERN "${CMAKE_CURRENT_SOURCE_DIR}/src/externlib/*.cpp")
# Remove some specific routines for the lists
list(FILTER EXTERN EXCLUDE REGEX "hdf5")
list(FILTER CORE EXCLUDE REGEX "device")
set(lib_prefix "ops_")
# seq
include(OPS_BUILD_BASE)
