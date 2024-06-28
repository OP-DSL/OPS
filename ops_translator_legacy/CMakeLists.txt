#   Specifiy the cmake version range, currenty work under 3.18
    cmake_minimum_required(VERSION 3.18)

    project(OPS_TRANSLATOR)

#   The following operations are OS dependent, we do UNIX first
    if (UNIX)
#       Copy the entire directory content to CMAKE_INSTALL_PREFIX/translator
        execute_process(COMMAND mkdir -p ${CMAKE_INSTALL_PREFIX}/translator)
        execute_process(COMMAND cp -r ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_INSTALL_PREFIX}/translator/.)

#       Set the execute permission and create soft link for CPP translator
        execute_process(COMMAND chmod a+x ${CMAKE_INSTALL_PREFIX}/translator/ops_translator_legacy/c/ops.py)
        execute_process(COMMAND ln -s ${CMAKE_INSTALL_PREFIX}/translator/ops_translator_legacy/c/ops.py ${CMAKE_INSTALL_PREFIX}/translator/ops_c)

#       Set the execute permission and create soft link for Fortran translator
        execute_process(COMMAND chmod a+x ${CMAKE_INSTALL_PREFIX}/translator/ops_translator_legacy/fortran/ops_fortran.py)
        execute_process(COMMAND ln -s ${CMAKE_INSTALL_PREFIX}/translator/ops_translator_legacy/fortran/ops_fortran.py ${CMAKE_INSTALL_PREFIX}/translator/ops_fortran)
    endif()

