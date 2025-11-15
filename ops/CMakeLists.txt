# Add local module path
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
include(OPS_InstallMacros)

# Includes to be done at level below to avoid issue of inclusion in the wrong target

# Library build
add_subdirectory(c)
if (OPS_FORTRAN)
  add_subdirectory(fortran)
endif()

# Add the configure file
configure_package_config_file(${CMAKE_CURRENT_SOURCE_DIR}/cmake/Config.cmake.in
                              "${CMAKE_BINARY_DIR}/OPSConfig.cmake"
                              INSTALL_DESTINATION ${ConfigPackageLocation})

#   prepare the script for setting OPS environment variable variables
if(UNIX)
    install(
        CODE "file(WRITE \"${CMAKE_INSTALL_PREFIX}/bin/ops_env\" \"export OPS_INSTALL_PATH=${CMAKE_INSTALL_PREFIX}\n\")"
    )

    if(NOT ((CMAKE_INSTALL_PREFIX STREQUAL "/usr")
         OR (CMAKE_INSTALL_PREFIX STREQUAL "/usr/local")))
        install(
            CODE "file(APPEND \"${CMAKE_INSTALL_PREFIX}/bin/ops_env\" \"export PATH=$PATH:${CMAKE_INSTALL_PREFIX}/bin\n\")"
        )
    endif()
endif()

