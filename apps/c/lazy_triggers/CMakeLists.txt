cmake_minimum_required(VERSION 3.18)
createtempdir()
build_ops_c_sample(lazy_triggers "NONE" "NONE" "NONE" "NO" "YES")

# The _tiled executables are compiled with -DOPS_LAZY (loops become descriptors). The generic tests
# run them with OPS_TILING; these additional tests run them with queueing but without tiling.
function(add_queue_test EXE TARGET)
    execute_process(COMMAND /bin/bash ${CMAKE_CURRENT_SOURCE_DIR}/cmake_test.sh ${EXE} ${TARGET}
                    RESULT_VARIABLE test_result OUTPUT_VARIABLE test_output)
    if(NOT test_result EQUAL 0)
        message(FATAL_ERROR "Error in executing script to get runtime command string ${test_result}: ${test_output}")
    endif()
    string(REGEX REPLACE "\n" "" test_output "${test_output}")
    add_test(NAME ${EXE}_${TARGET}
             COMMAND ${CMAKE_COMMAND} -DCMD=${test_output} -DARG=" "
                     -DOPS_INSTALL_PATH=${OPS_CMAKE_INSTALL_PATH} -P ${OPS_APP_SRC}/runtests.cmake
             WORKING_DIRECTORY "${TMP_SOURCE_DIR}")
endfunction()

if(OPS_TEST)
    add_queue_test(${CMAKE_CURRENT_BINARY_DIR}/lazy_triggers_tiled queue)
    add_queue_test(${CMAKE_CURRENT_BINARY_DIR}/lazy_triggers_tiled executor_split)
    add_queue_test(${CMAKE_CURRENT_BINARY_DIR}/lazy_triggers_tiled executor_manual)
    add_queue_test(${CMAKE_CURRENT_BINARY_DIR}/lazy_triggers_tiled executor_split_tiled)
    if(MPI)
        add_queue_test(${CMAKE_CURRENT_BINARY_DIR}/lazy_triggers_mpi_tiled mpi_queue)
        add_queue_test(${CMAKE_CURRENT_BINARY_DIR}/lazy_triggers_mpi_tiled mpi_executor_split)
        add_queue_test(${CMAKE_CURRENT_BINARY_DIR}/lazy_triggers_mpi_tiled mpi_executor_manual)
        add_queue_test(${CMAKE_CURRENT_BINARY_DIR}/lazy_triggers_mpi_tiled mpi_executor_split_tiled)
    endif()
endif()
