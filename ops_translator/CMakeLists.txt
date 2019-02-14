cmake_minimum_required(VERSION 3.8)
project(OPS_TRANSLATOR)

install(PROGRAMS
    ${CMAKE_CURRENT_SOURCE_DIR}/c/ops.py
    ${CMAKE_CURRENT_SOURCE_DIR}/c/format.sh
    DESTINATION bin/ops_translator/c)
install(FILES
    ${CMAKE_CURRENT_SOURCE_DIR}/c/config.py
    ${CMAKE_CURRENT_SOURCE_DIR}/c/ops_gen_mpi_cuda.py
    ${CMAKE_CURRENT_SOURCE_DIR}/c/ops_gen_mpi_inline.py
    ${CMAKE_CURRENT_SOURCE_DIR}/c/ops_gen_mpi_lazy.py
    ${CMAKE_CURRENT_SOURCE_DIR}/c/ops_gen_mpi_openacc.py
    ${CMAKE_CURRENT_SOURCE_DIR}/c/ops_gen_mpi_opencl.py
    ${CMAKE_CURRENT_SOURCE_DIR}/c/ops_mpi_seq_gen.py
    ${CMAKE_CURRENT_SOURCE_DIR}/c/util.py
    DESTINATION bin/ops_translator/c)
set(OPS_C_TRANSLATOR_EXECUTABLE "${CMAKE_CURRENT_SOURCE_DIR}/c/ops.py" CACHE FILEPATH "OPS C Translator path")

if(BUILD_OPS_FORTRAN)
    install(PROGRAMS
        ${CMAKE_CURRENT_SOURCE_DIR}/fortran/ops_fortran.py
        DESTINATION bin/ops_translator/fortran)
    install(FILES
        ${CMAKE_CURRENT_SOURCE_DIR}/fortran/config.py
        ${CMAKE_CURRENT_SOURCE_DIR}/fortran/ops_fortran_gen_mpi_cuda.py
        ${CMAKE_CURRENT_SOURCE_DIR}/fortran/ops_fortran_gen_mpi_openacc.py
        ${CMAKE_CURRENT_SOURCE_DIR}/fortran/ops_fortran_gen_mpi_openmp.py
        ${CMAKE_CURRENT_SOURCE_DIR}/fortran/ops_fortran_gen_mpi.py
        ${CMAKE_CURRENT_SOURCE_DIR}/fortran/util_fortran.py
        DESTINATION bin/ops_translator/fortran)
set(OPS_Fortran_TRANSLATOR_EXECUTABLE ${CMAKE_CURRENT_SOURCE_DIR}/fortran/ops_fortran.py CACHE FILEPATH "OPS Fortran Translator")
endif()


macro(OPS_C_TARGET Name OpsInput OpsTargetDir)

    list(LENGTH ${OpsInput} InLength)
    if (${InLength} EQUAL 0)
        list(APPEND Sources ${OpsInput})
    else()
        set(Sources ${${OpsInput}})
    endif()
    list(LENGTH Sources InLength2)
    list(TRANSFORM Sources PREPEND "${CMAKE_CURRENT_SOURCE_DIR}/")

    foreach(input ${Sources})
        get_filename_component(input_BASENAME ${input} NAME_WE)
        get_filename_component(input_EXTN ${input} EXT)
        list(APPEND OPS_${Name}_BASEOUTPUT "${OpsTargetDir}/${input_BASENAME}_ops${input_EXTN}")
    endforeach()
    list(GET Sources 0 OpsMainInput)
    get_filename_component(input_BASENAME ${OpsMainInput} NAME_WE)
    get_filename_component(input_EXTN ${OpsMainInput} EXT)
    set(OPS_${Name}_CUDA_KERNELS "${OpsTargetDir}/CUDA/${input_BASENAME}_kernels.cu")
    set(OPS_${Name}_MPI_inline_KERNELS "${OpsTargetDir}/MPI_inline/${input_BASENAME}_kernels.cpp")
    set(OPS_${Name}_MPI_OpenMP_KERNELS "${OpsTargetDir}/MPI_OpenMP/${input_BASENAME}_cpu_kernels.cpp")
    set(OPS_${Name}_OpenACC_KERNELS "${OpsTargetDir}/OpenACC/${input_BASENAME}_kernels.cpp")
    set(OPS_${Name}_OpenCL_KERNELS "${OpsTargetDir}/OpenCL/${input_BASENAME}_opencl_kernels.cpp")

    list(APPEND OPS_${Name}_TARGET_outputs
            "${OPS_${Name}_BASEOUTPUT}"
            "${OPS_${Name}_CUDA_KERNELS}"
            "${OPS_${Name}_MPI_inline_KERNELS}"
            "${OPS_${Name}_MPI_OpenMP_KERNELS}"
            "${OPS_${Name}_OpenACC_KERNELS}"
            "${OPS_${Name}_OpenCL_KERNELS}"
        )

    file(MAKE_DIRECTORY ${OpsTargetDir})
    add_custom_command(
            OUTPUT ${OPS_${Name}_TARGET_outputs}
            COMMAND ${OPS_C_TRANSLATOR_EXECUTABLE} ${Sources}
            DEPENDS ${Sources}
            BYPRODUCTS ${OPS_${Name}_TARGET_outputs}
            COMMENT "[OPS][${Name}] Building OPS sources"
            WORKING_DIRECTORY ${OpsTargetDir})

    set(OPS_${Name}_DEFINED TRUE)
    set(OPS_${Name}_INPUT ${OpsInput})
endmacro()

