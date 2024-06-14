#   The following operations are OS dependent, we do UNIX first
if (UNIX)
    # Copy the entire directory content to CMAKE_INSTALL_PREFIX/translator
    execute_process(COMMAND mkdir -p ${CMAKE_INSTALL_PREFIX}/translator)
    execute_process(COMMAND cp -r ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_INSTALL_PREFIX}/translator/.)

    # Set the execute permission to a script which will install python virtual environment and required python
    # packages to use Clang/Fparser based code-gen
    execute_process(COMMAND chmod a+x ${CMAKE_INSTALL_PREFIX}/translator/ops_translator/setup_venv_cmake.sh
                    RESULT_VARIABLE chmod_result
                    OUTPUT_VARIABLE chmod_output)

    if(NOT chmod_result EQUAL 0)
        message(FATAL_ERROR "Error: Failed to set execute permission on setup_venv_cmake.sh: ${chmod_output}")
    endif()

    # Install the required Python dependencies at CMAKE_INSTALL_PREFIX/translator/ops_translator/ops_venv
    file(GLOB OPS_VENV ${CMAKE_INSTALL_PREFIX}/translator/ops_translator/ops_venv)
    if(NOT OPS_VENV)
        message(STATUS "Python Virtual Environment not found in ${OPS_VENV}, proceeding with installation")

        execute_process(COMMAND /bin/bash ${CMAKE_INSTALL_PREFIX}/translator/ops_translator/setup_venv_cmake.sh ${CMAKE_INSTALL_PREFIX}/translator/ops_translator
                        RESULT_VARIABLE req_install_result
                        OUTPUT_VARIABLE req_install_output)
        if(NOT req_install_result EQUAL 0)
            message(FATAL_ERROR "Error: Failed to execute setup_venv_cmake.sh: ${req_install_output}")
        endif()
    else()
        message(STATUS "Python Virtual Environment found, skipping installation")
    endif()

endif() # UNIX if end
