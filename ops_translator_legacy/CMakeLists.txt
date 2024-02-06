cmake_minimum_required(VERSION 3.18)
project(OPS_TRANSLATOR)
#install python translator
INSTALL(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}" DESTINATION bin)
#The following operations are OS dependent, we do UNIX first
if (UNIX)
    INSTALL(CODE "execute_process (
        COMMAND ln -s ${CMAKE_INSTALL_PREFIX}/bin/ops_translator_legacy/c/ops.py ${CMAKE_INSTALL_PREFIX}/bin/ops_c
    )")
    INSTALL(CODE "execute_process (
        COMMAND chmod a+x ${CMAKE_INSTALL_PREFIX}/bin/ops_translator_legacy/c/ops.py
    )")
    INSTALL(CODE "execute_process (
        COMMAND chmod a+x ${CMAKE_INSTALL_PREFIX}/bin/ops_translator_legacy/fortran/ops_fortran.py
    )")
    INSTALL(CODE "execute_process (
        COMMAND ln -s ${CMAKE_INSTALL_PREFIX}/bin/ops_translator_legacy/fortran/ops_fortran.py ${CMAKE_INSTALL_PREFIX}/bin/ops_fortran
    )")
endif()

