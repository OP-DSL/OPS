# To specifiy the cmake version range Currenty work under 3.18
cmake_minimum_required(VERSION 3.18)
project(OPS CXX)

# if show the compiling process in detail
option(OPS_TEST "Turn on tests for Apps" OFF)
option(OPS_HIP "Turn on the HIP backend" OFF)
option(OPS_CUDA "Turn on the CUDA backend" ON)
option(IEEE "Use IEEE flags" ON)

# Output of the installation directory
include(GNUInstallDirs)
set(LIBRARY_OUTPUT_PATH ${PROJECT_BINARY_DIR}/${CMAKE_INSTALL_LIBDIR})
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_BINARY_DIR}/${CMAKE_INSTALL_BINDIR})
set(CMAKE_Fortran_MODULE_DIRECTORY ${PROJECT_BINARY_DIR}/${CMAKE_INSTALL_INCLUDEDIR})
if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
  set(CMAKE_INSTALL_PREFIX "${PROJECT_BINARY_DIR}/opt" CACHE PATH "..." FORCE)
endif()
message(STATUS
    "The installation directory is ${CMAKE_INSTALL_PREFIX}! Use -DCMAKE_INSTALL_PREFIX to change"
)

# Use the Release mode by default
if(NOT CMAKE_BUILD_TYPE)
  message(STATUS
	  "BUILD_TYPE is Release mode! Please uses -DCMAKE_BUILD_TYPE to change.")
  set (CMAKE_BUILD_TYPE RELEASE CACHE STRING
      "Choose the type of build, options are: None Dev Debug Release."
      FORCE)
endif(NOT CMAKE_BUILD_TYPE)


# Set the verbose level
option(OPS_VERBOSE_WARNING "Turn on verbose warning messages" OFF)
if(NOT OPS_VERBOSE_WARNING)
  message(STATUS 
      "Verbose output for compilation is set to concise! Use -DOPS_VERBOSE_WARNING=ON to increase verbose."
  )
endif()
set(CMAKE_VERBOSE_MAKEFILE ${OPS_VERBOSE_WARNING})

# Tridiagonal library
if(NOT LIBTRID_PATH)
  message(WARNING
      "The install path of the scalar tridiagonal solver library (LIBTRID_PATH) is not specified and it will not be compiled!"
  )
endif()


# Try to find the required dependency
# Add support for CMAKE_DEPENDENT_OPTION
INCLUDE(CMakeDependentOption)
INCLUDE(CMakeParseArguments)
include(CMakePackageConfigHelpers)
set(VERSION latest)
set(ConfigPackageLocation ${CMAKE_INSTALL_PREFIX}/lib/cmake)

# Include paths to find pachages 
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake" "${CMAKE_SOURCE_DIR}/cmake/compilers" "${CMAKE_SOURCE_DIR}/cmake/packages")

include(OPS_FindPackages)
include(OPS_Compilers)
# install python translator
add_subdirectory(ops)
#add_subdirectory(ops_translator)
