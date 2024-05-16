execute_process(COMMAND ${SEQ} OUTPUT_FILE perf.out RESULT_VARIABLE RES)
if(RES)
    message(FATAL_ERROR "Error running ${SEQ}")
endif()
file(COPY "cloverdata.h5" DESTINATION "cloverdata_seq.h5")

#execute_process(COMMAND mpirun -n ${NUM} ${MPI} OUTPUT_FILE perf.out RESULT_VARIABLE RES)
#if(RES)
#    message(FATAL_ERROR "Error running ${MPI}")
#endif()

#file(RENAME "cloverdata.h5" "cloverdata_mpi.h5")
#execute_process(COMMAND "${H5D}" cloverdata.h5 cloverdata_seq.h5 OUTPUT_FILE diff.out RESULT_VARIABLE RES)
#if(RES)
#    message(FATAL_ERROR "h5diff fails")
#endif()

