set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
include(OPS_BUILD_APPS)

# Create a folder directory for testing
set(app_dir_c "${app_dir}/c")
file(MAKE_DIRECTORY ${app_dir_c})

# Include include directory for C++
include_directories("${CMAKE_SOURCE_DIR}/ops/c/include")

# Set the OPS Translator 
if(LEGACY_CODEGEN)
  set(OPS_C_TRANSLATOR "${opsc}/ops.py")
else()
  set(OPS_C_TRANSLATOR "${opsc}")
endif()
add_subdirectory(poisson)


