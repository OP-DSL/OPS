add_subdirectory(c)
#check_language(Fortran)
#if(CMAKE_Fortran_COMPILER)
#  add_subdirectory(fortran)
#endif(CMAKE_Fortran_COMPILER)

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

