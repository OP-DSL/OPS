#   Specifiy the cmake version range, currenty work under 3.18
    cmake_minimum_required(VERSION 3.18)

    project(OPS_TRANSLATOR)

#   Copy the entire directory content to CMAKE_INSTALL_PREFIX/bin
    INSTALL(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}" DESTINATION bin)

#   The following operations are OS dependent, we do UNIX first
    if (UNIX)
        INSTALL(CODE 
            "execute_process (
            COMMAND ln -s ${CMAKE_INSTALL_PREFIX}/bin/ops_translator/ops-translator/__main__.py ${CMAKE_INSTALL_PREFIX}/bin/ops_c
            )"
        )

        INSTALL(CODE 
            "execute_process (
            COMMAND ln -s ${CMAKE_INSTALL_PREFIX}/bin/ops_translator/ops-translator/__main__.py ${CMAKE_INSTALL_PREFIX}/bin/ops_fortran
            )"
        )

        INSTALL(CODE 
            "execute_process (
            COMMAND chmod a+x ${CMAKE_INSTALL_PREFIX}/bin/ops_translator/setup_venv_cmake.sh
            )"
        )

#       Install the required Python dependencies at CMAKE_INSTALL_PREFIX/bin/ops_translator/ops_venv
        file(GLOB OPS_VENV ${CMAKE_INSTALL_PREFIX}/bin/ops_translator/ops_venv)
        if(NOT OPS_VENV)
            message(STATUS "Python Virtual Environment not found in OPS_TRANSLATOR, proceeding with installation")

            install(CODE 
                "execute_process(
                COMMAND /bin/bash ${CMAKE_INSTALL_PREFIX}/bin/ops_translator/setup_venv_cmake.sh ${CMAKE_INSTALL_PREFIX}/bin/ops_translator
                )"
            )

        else()
            message(STATUS "Python Virtual Environment found, skipping installation")
        endif()

    endif() # UNIX if end
