#   Specifiy the cmake version range, currently work under 3.18
    cmake_minimum_required(VERSION 3.18)

#           Project-NAME LANGUAGE/s
    project(APP Fortran)

    include(CheckLanguage)
    check_language(Fortran)

    if(CMAKE_Fortran_COMPILER)
        message(STATUS "Fortran compiler: ${CMAKE_Fortran_COMPILER}")
        set(CMAKE_Fortran_MODULE_DIRECTORY ${CMAKE_INSTALL_PREFIX}/mod)

        if(${CMAKE_Fortran_COMPILER_ID} STREQUAL Intel)
            set(CMAKE_Fortran_FLAGS         "${CMAKE_Fortran_FLAGS} -Wall -O3 -g -xHost -qopenmp")
            set(CMAKE_Fortran_FLAGS_DEBUG   "-g -traceback -qopenmp")
            set(CMAKE_Fortran_FLAGS_RELEASE "-O3 -ip -xHOST -qopenmp")
        elseif(${CMAKE_Fortran_COMPILER_ID} STREQUAL GNU)
            set(CMAKE_Fortran_FLAGS         "${CMAKE_Fortran_FLAGS} -Wall -O3 -g -ffree-form -ffree-line-length-none -fopenmp")
            set(CMAKE_Fortran_FLAGS_DEBUG   "-O0 -g3 -ffree-form -ffree-line-length-none -fopenmp")
            set(CMAKE_Fortran_FLAGS_RELEASE "-Ofast -march=native -ffree-form -ffree-line-length-none -fopenmp")
        elseif(${CMAKE_Fortran_COMPILER_ID} STREQUAL NVHPC)
            set(CMAKE_Fortran_FLAGS         "${CMAKE_Fortran_FLAGS} -Wall O3 -fast -gopt")
        else()
            message(
                WARNING
                    "There is no default flag for the chosen Fortran compiler!"
            )
        endif()

#       If APPs are compiled together with the OPS API
        if(${CMAKE_PROJECT_NAME} STREQUAL OPS)

            set(HDF5_SEQ ${HDF5_FOUND})
            if(NOT Python3_FOUND)
                message(
                    WARNING
                        "We cannot find Python3 and the OPS code-generator needs Python3! Please use -DPython3_EXECUTABLE to specify the path."
                )
            else()
                if(LEGACY_CODEGEN)
                    set(OPS_C_TRANSLATOR "${CMAKE_INSTALL_PREFIX}/translator/ops_translator_legacy/c/ops.py")
                    set(OPS_F_TRANSLATOR "${CMAKE_INSTALL_PREFIX}/translator/ops_translator_legacy/fortran/ops_fortran.py")
                else()
                    set(OPS_C_TRANSLATOR "${CMAKE_INSTALL_PREFIX}/translator/ops_translator/ops-translator")
                    set(OPS_F_TRANSLATOR "${CMAKE_INSTALL_PREFIX}/translator/ops_translator/ops-translator")
                endif()
            endif()

#           Set the OPS_INSTALL_PATH for testing
            set(OPS_CMAKE_INSTALL_PATH "${CMAKE_INSTALL_PREFIX}")
            set(OPS_INSTALL_PATH "${CMAKE_SOURCE_DIR}/ops/fortran")
            set(OPS_APP_SRC "${CMAKE_SOURCE_DIR}/apps/fortran")

            if(OPS_TEST)
                enable_testing()
            endif()

        endif()

#       Clean the tmp directory for the code generation
        macro(CreateTempDir)
            file(REMOVE_RECURSE "${CMAKE_CURRENT_BINARY_DIR}/tmp")
            file(MAKE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/tmp")
        endmacro()

        function(add_cmake_test EXE_NAME SCRIPT_PATH ARGUMENTS)
            if(EXISTS ${SCRIPT_PATH})
                separate_arguments(args NATIVE_COMMAND ${ARGUMENTS})
                execute_process(COMMAND /bin/bash ${SCRIPT_PATH} ${EXE_NAME} ${args}
                                RESULT_VARIABLE test_result
                                OUTPUT_VARIABLE test_output)

                if(test_result EQUAL 0)
                    string(REGEX REPLACE "\n" "" test_output "${test_output}")

#                    message(STATUS "run command: ${test_output}")
                    set(args " ")
                    set(cmd "${test_output}")
                    add_test(NAME ${EXE_NAME}
                            COMMAND ${CMAKE_COMMAND}
                            -DCMD=${cmd} -DARG=${args}
                            -DOPS_INSTALL_PATH=${OPS_CMAKE_INSTALL_PATH} -P ${OPS_APP_SRC}/runtests.cmake
                            WORKING_DIRECTORY "${TMP_SOURCE_DIR}")
                else()
                    message(FATAL_ERROR "Error in executing script to get runtime command string ${test_result}: ${test_output}")
                endif()
            endif()
        endfunction()

#       Prepare the macro for compiling apps 
#       Name: App name 
#       Odd: Key words for source files are included in other source files and need no explicit compilation
#       Others: Key words for source codes need no code generation using ops.py 
#       Extra: Key words for ource codes that cannot be treated by the macro 
#       Trid: Whether the tridiagonal library is required for this application 
#       GenerateTest: if generating the testings, some apps need specific testing targets.
#       TODO: ompoffload
        macro(
            BUILD_OPS_C_SAMPLE
            Name
            Odd
            Others
            Extra
            Trid
            GenerateTest)

#           Copy all source and head files into tmp
            file(GLOB FTN "${CMAKE_CURRENT_SOURCE_DIR}/*.F90")

            list(FILTER FTN EXCLUDE REGEX "ops.F90")

            file(COPY ${FTN} DESTINATION "${CMAKE_CURRENT_BINARY_DIR}/tmp/")
            file(GLOB HEADS "${CMAKE_CURRENT_SOURCE_DIR}/*.inc")
            file(COPY ${HEADS} DESTINATION "${CMAKE_CURRENT_BINARY_DIR}/tmp/")

            set(TMP_SOURCE_DIR "${CMAKE_CURRENT_BINARY_DIR}/tmp")

#           Get the file to be compiled
#           DEV: non-optimised codes (full directory and filename)
#           OPS: codes generated by ops.py (full directory and filename)
#           OTHERS: Source codes need no code generation (full directory and filename)
            file(GLOB DEV "${TMP_SOURCE_DIR}/*.F90")
            list(FILTER DEV EXCLUDE REGEX ${Odd})
            foreach(oth ${Others})
                list(FILTER DEV EXCLUDE REGEX ${oth})
            endforeach()
            foreach(ext ${Extra})
                list(FILTER DEV EXCLUDE REGEX ${ext})
            endforeach()

            list(GET DEV 0 Kernel)

            get_filename_component(KernelName ${Kernel} NAME_WE)

#           If not LEGACY_CODEGEN then activate the Installed Python Virtual Environment
            if(NOT LEGACY_CODEGEN)
                message("Activating Python Virtual Environment for Code-Generation")
                set(VENV_ACTIVATE "source ${OPS_CMAKE_INSTALL_PATH}/translator/ops_translator/ops_venv/bin/activate")

                execute_process(COMMAND /bin/bash -c ${VENV_ACTIVATE}
                                WORKING_DIRECTORY ${TMP_SOURCE_DIR}
                                RESULT_VARIABLE result
                                OUTPUT_VARIABLE output)

                if(result EQUAL 0)
                    set(PYTHON_EXECUTABLE "${OPS_CMAKE_INSTALL_PATH}/translator/ops_translator/ops_venv/bin/python3")
                    message(STATUS "Python Virtual Enviroment activated successfully")
                else()
                    message("command: ${VENV_ACTIVATE}")
                    if(NOT "${output}" STREQUAL "")
                        message(FATAL_ERROR "Activate Python Virtual Environment command failed with error code ${result}: ${output}")
                    else()
                        message(FATAL_ERROR "Activate Python Virtual Environment command failed with error code ${result}")
                    endif()
                endif()
            endif()

#           Run OPS code-generation
            if(LEGACY_CODEGEN)
                message(STATUS "Code-gen command: ${OPS_F_TRANSLATOR} ${DEV}")
                execute_process(COMMAND ${OPS_F_TRANSLATOR} ${DEV}
                                WORKING_DIRECTORY ${TMP_SOURCE_DIR}
                                RESULT_VARIABLE result
                                OUTPUT_VARIABLE output)
            else()
                if(${CMAKE_PROJECT_NAME} STREQUAL OPS)
                    set(PYTHON_FILE_ARGS "-DOPS_ACC_IGNORE -I ${CMAKE_SOURCE_DIR}/ops/c/include --file_paths")
                    set(PYTHON_OPS_INCLUDE ${CMAKE_SOURCE_DIR}/ops/c/include)
                endif()
                if(${CMAKE_PROJECT_NAME} STREQUAL APP)
                    set(PYTHON_FILE_ARGS "-DOPS_ACC_IGNORE -I ${OPS_CMAKE_INSTALL_PATH}/include --file_paths")
                    set(PYTHON_OPS_INCLUDE ${OPS_CMAKE_INSTALL_PATH}/include)
                endif()
                foreach(filepath ${DEV})
                    get_filename_component(filename ${filepath} NAME)
                    list(APPEND FILENAMES ${filename})
                endforeach()

                message(STATUS "Code-gen command: ${PYTHON_EXECUTABLE} ${OPS_F_TRANSLATOR} ${PYTHON_FILE_ARGS} ${FILENAMES}")
                execute_process(COMMAND ${PYTHON_EXECUTABLE} ${OPS_F_TRANSLATOR}
                                -DOPS_ACC_IGNORE
                                -I ${PYTHON_OPS_INCLUDE}
                                --file_paths ${FILENAMES}
                                WORKING_DIRECTORY ${TMP_SOURCE_DIR}
                                RESULT_VARIABLE result
                                OUTPUT_VARIABLE output)
            endif()

            if(result EQUAL 0)
                message(STATUS "OPS Python code generation runs successfully")
            else()
                message(FATAL_ERROR "OPS Python code generation command failed with error code ${result}: ${output}")
            endif()

            file(GLOB OPS "${TMP_SOURCE_DIR}/*ops*.F90")

            file(GLOB OTHERS "${TMP_SOURCE_DIR}/*.F90")
            foreach(OpsFile ${OPS})
                list(REMOVE_ITEM OTHERS ${OpsFile})
                string(REPLACE "_ops" "" NoOPS ${OpsFile})
                list(REMOVE_ITEM OTHERS ${NoOPS})
            endforeach()

            list(FILTER OTHERS EXCLUDE REGEX ${Odd})

            foreach(ext ${Extra})
                list(FILTER OTHERS EXCLUDE REGEX ${ext})
            endforeach()

            file(GLOB_RECURSE SEQ_TEMPLATES "${TMP_SOURCE_DIR}/mpi_openmp/*_seq_kernel.F90")

#            message(STATUS "OPS: ${OPS}")
#            message(STATUS "OTHERS: ${OTHERS}")
#            message(STATUS "SEQ_TEMPLATES: ${SEQ_TEMPLATES}")
            add_executable(${Name}_seq ${OPS} ${OTHERS} ${SEQ_TEMPLATES})
            target_include_directories(${Name}_seq PRIVATE ${TMP_SOURCE_DIR})

            add_executable(${Name}_openmp ${OPS} ${OTHERS} ${SEQ_TEMPLATES})
            target_include_directories(${Name}_openmp PRIVATE ${TMP_SOURCE_DIR})

            add_executable(${Name}_tiled ${OPS} ${OTHERS} ${SEQ_TEMPLATES})           
            target_include_directories(${Name}_tiled PRIVATE ${TMP_SOURCE_DIR})
            target_compile_definitions(${Name}_tiled PRIVATE "-DOPS_LAZY")

            target_link_libraries(${Name}_seq     ops_for_seq)
            target_link_libraries(${Name}_openmp     ops_for_seq)
            target_link_libraries(${Name}_tiled     ops_for_seq)

            if(HDF5_SEQ)
                target_link_libraries(${Name}_seq ops_for_hdf5_seq hdf5::hdf5 hdf5::hdf5_hl)
                target_link_libraries(${Name}_openmp ops_for_hdf5_seq hdf5::hdf5 hdf5::hdf5_hl)
                target_link_libraries(${Name}_tiled ops_for_hdf5_seq hdf5::hdf5 hdf5::hdf5_hl)
            endif()

            install(TARGETS ${Name}_seq     DESTINATION ${APP_INSTALL_DIR}/${Name})
            install(TARGETS ${Name}_openmp     DESTINATION ${APP_INSTALL_DIR}/${Name})
            install(TARGETS ${Name}_tiled     DESTINATION ${APP_INSTALL_DIR}/${Name})

            if((OPS_TEST) AND (${GenerateTest} STREQUAL "YES"))
                add_cmake_test(${CMAKE_CURRENT_BINARY_DIR}/${Name}_seq ${CMAKE_CURRENT_SOURCE_DIR}/cmake_test.sh seq)
                add_cmake_test(${CMAKE_CURRENT_BINARY_DIR}/${Name}_openmp ${CMAKE_CURRENT_SOURCE_DIR}/cmake_test.sh openmp)
                add_cmake_test(${CMAKE_CURRENT_BINARY_DIR}/${Name}_tiled ${CMAKE_CURRENT_SOURCE_DIR}/cmake_test.sh tiled)
            endif()

        endmacro()

#       Build all listed apps

#       Current Targets: seq

#       App: Poisson
        add_subdirectory(poisson)

#       App: Laplace2d
        add_subdirectory(laplace2dtutorial/step7)

    endif(CMAKE_Fortran_COMPILER)
