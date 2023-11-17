cmake_minimum_required(VERSION 3.18)
project(OPS_TRANSLATOR)
#install python translator
INSTALL(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}" DESTINATION bin)
#The following operations are OS dependent, we do UNIX first
if (UNIX)
    INSTALL(CODE "execute_process (
        COMMAND ln -s ${CMAKE_INSTALL_PREFIX}/bin/ops_translator/c/ops.py ${CMAKE_INSTALL_PREFIX}/bin/ops_c
    )")
    INSTALL(CODE "execute_process (
        COMMAND chmod a+x ${CMAKE_INSTALL_PREFIX}/bin/ops_translator/c/ops.py
    )")
    INSTALL(CODE "execute_process (
        COMMAND chmod a+x ${CMAKE_INSTALL_PREFIX}/bin/ops_translator/fortran/ops_fortran.py
    )")
    INSTALL(CODE "execute_process (
        COMMAND ln -s ${CMAKE_INSTALL_PREFIX}/bin/ops_translator/fortran/ops_fortran.py ${CMAKE_INSTALL_PREFIX}/bin/ops_fortran
    )")
endif()

set(OPS_C_TRANSLATOR_EXECUTABLE "${CMAKE_CURRENT_SOURCE_DIR}/c/ops.py" CACHE FILEPATH "OPS C Translator path")
set(OPS_Fortran_TRANSLATOR_EXECUTABLE ${CMAKE_CURRENT_SOURCE_DIR}/fortran/ops_fortran.py CACHE FILEPATH "OPS Fortran Translator")



# Prepare macro for calling the translator from CMake
# Name: App name
# OpsInput: List of OPS source files
# OpsTargetDir: Directory where the generated files will be placed
# The generated files are stored for each target in a list with name:
# OPS_<Name>_BASEOUTPUT: The list of transformed input files
# OPS_<Name>_<Target>_KERNELS: The generated kernel files
# OPS_<Name>_TARGET_outputs: The list of all generated files
macro(OPS_C_TARGET Name OpsInput OpsTargetDir)

    unset(_Sources)
    list(LENGTH ${OpsInput} InLength)
    if (${InLength} EQUAL 0)
        list(APPEND _Sources ${OpsInput})
    else()
        set(_Sources ${${OpsInput}})
    endif()
    list(LENGTH _Sources InLength2)
    list(TRANSFORM _Sources PREPEND "${CMAKE_CURRENT_SOURCE_DIR}/")

    foreach(input ${_Sources})
        get_filename_component(input_BASENAME ${input} NAME_WE)
        get_filename_component(input_EXTN ${input} EXT)
        list(APPEND OPS_${Name}_BASEOUTPUT "${OpsTargetDir}/${input_BASENAME}_ops${input_EXTN}")
    endforeach()
    list(GET _Sources 0 OpsMainInput)
    get_filename_component(input_BASENAME ${OpsMainInput} NAME_WE)
    get_filename_component(input_EXTN ${OpsMainInput} EXT)
    set(OPS_${Name}_CUDA_KERNELS "${OpsTargetDir}/CUDA/${input_BASENAME}_kernels.cu")
    set(OPS_${Name}_HIP_KERNELS "${OpsTargetDir}/HIP/${input_BASENAME}_kernels.cpp")
    set(OPS_${Name}_MPI_inline_KERNELS "${OpsTargetDir}/MPI_inline/${input_BASENAME}_kernels.cpp")
    set(OPS_${Name}_MPI_OpenMP_KERNELS "${OpsTargetDir}/MPI_OpenMP/${input_BASENAME}_cpu_kernels.cpp")
    set(OPS_${Name}_OpenMP_offload_KERNELS "${OpsTargetDir}/OpenMP_offload/${input_BASENAME}_ompoffload_kernels.cpp")
    set(OPS_${Name}_OpenACC_KERNELS "${OpsTargetDir}/OpenACC/${input_BASENAME}_kernels.cpp")
    set(OPS_${Name}_OpenCL_KERNELS "${OpsTargetDir}/OpenCL/${input_BASENAME}_opencl_kernels.cpp")
    set(OPS_${Name}_SYCL_KERNELS "${OpsTargetDir}/SYCL/${input_BASENAME}_sycl_kernels.cpp")

    list(APPEND OPS_${Name}_TARGET_outputs
            "${OPS_${Name}_BASEOUTPUT}"
            "${OPS_${Name}_CUDA_KERNELS}"
            "${OPS_${Name}_HIP_KERNELS}"
            "${OPS_${Name}_MPI_inline_KERNELS}"
            "${OPS_${Name}_MPI_OpenMP_KERNELS}"
            "${OPS_${Name}_OpenMP_offload_KERNELS}"
            "${OPS_${Name}_OpenACC_KERNELS}"
            "${OPS_${Name}_OpenCL_KERNELS}"
            "${OPS_${Name}_SYCL_KERNELS}"
        )

    file(MAKE_DIRECTORY ${OpsTargetDir})
    add_custom_command(
            OUTPUT ${OPS_${Name}_TARGET_outputs}
            COMMAND ${OPS_C_TRANSLATOR_EXECUTABLE} ${_Sources}
            DEPENDS ${_Sources}
            BYPRODUCTS ${OPS_${Name}_TARGET_outputs}
            COMMENT "[OPS][${Name}] Building OPS sources"
            WORKING_DIRECTORY ${OpsTargetDir})

    set(OPS_${Name}_DEFINED TRUE)
    set(OPS_${Name}_INPUT ${OpsInput})
endmacro()

