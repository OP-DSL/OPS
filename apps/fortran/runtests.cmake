macro(TEST CMD ARG OPS_INSTALL_PATH)
    set(ENV{OPS_INSTALL_PATH} ${OPS_INSTALL_PATH})

    string(FIND ${CMD} "_openmp" openmp_index)
    if(openmp_index EQUAL -1)
        set(ENV{OMP_NUM_THREADS} 1)
    else()
        set(ENV{OMP_NUM_THREADS} 2)
    endif()

    #set(ENV{OMP_NUM_THREADS} 1)
    separate_arguments(args NATIVE_COMMAND ${ARG})
    separate_arguments(cmds NATIVE_COMMAND ${CMD})

    execute_process(COMMAND ${cmds} ${args} OUTPUT_FILE perf.out)
    execute_process(COMMAND grep "PASSED" perf.out RESULT_VARIABLE RES)
    # message("RES=${RES}")
    if(RES)
        message(FATAL_ERROR "Error running ${CMD}")
    endif()
endmacro()

TEST(${CMD} ${ARG} ${OPS_INSTALL_PATH})
