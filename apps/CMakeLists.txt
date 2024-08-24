# Create a folder directory for testing
set(app_dir "${PROJECT_BINARY_DIR}/APPS_BUILD")
file(MAKE_DIRECTORY ${app_dir})
# Activate the full list of tests
option(OPS_TEST_FULL "Turn on tests for all Apps" OFF)
option(OPS_TEST_DEV "Turn on tests for dev backends" OFF)

# Find the OPS Translator
if(LEGACY_CODEGEN)
    find_path(opsc
            NAMES "ops.py"
            PATHS ${SEARCH_PATHS}
            PATH_SUFFIXES translator/ops_translator_legacy/c)

    find_path(opsfortran
            NAMES "ops_fortran.py"
            PATHS ${SEARCH_PATHS}
            PATH_SUFFIXES translator/ops_translator_legacy/fortran)
else()
    find_path(opsc
            NAMES "__main__.py"
            PATHS ${SEARCH_PATHS}
            PATH_SUFFIXES translator/ops_translator/ops-translator)

    find_path(opsfortran
            NAMES "__main__.py"
            PATHS ${SEARCH_PATHS}
            PATH_SUFFIXES translator/ops_translator/ops-translator)
endif()

if(NOT opsc)
    message(FATAL_ERROR "We cannot find the python C/C++ translator!")
endif()

if(NOT opsfortran)
    message(FATAL_ERROR "We cannot find the python fortran translator!")
endif()

# Add include and link directories
#${CMAKE_INSTALL_PREFIX}/lib
#include_directories("${CMAKE_INSTALL_PREFIX}/include")
#message(STATUS "SOURCE DIR ${CMAKE_SOURCE_DIR}")
#message(STATUS "INSTALL PREFIX ${CMAKE_INSTALL_PREFIX}")
link_directories("${PROJECT_BINARY_DIR}/lib")
add_subdirectory(c)
