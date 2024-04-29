#   Specifiy the cmake version range, currently work under 3.18
    cmake_minimum_required(VERSION 3.18)

    project(APP C CXX)

    if(NOT APP_INSTALL_DIR)
        set(APP_INSTALL_DIR "$ENV{HOME}/OPS-APPS")
        message(
            WARNING
                "We will install the applications to ${APP_INSTALL_DIR}! Please use -DAPP_INSTALL_DIR to specify!"
        )
    endif()

#   If APPs are compiled independently then setup compilers and dependencies.
    if(${CMAKE_PROJECT_NAME} STREQUAL APP)
#       Show the compiling process in detail
        option(OPS_VERBOSE_WARNING "Turn on verbose warning messages" OFF)
        option(OPS_TEST "Turn on tests for Apps" OFF)
        option(ENABLE_IEEE "Enable IEEE flags" OFF)
        option(LEGACY_CODEGEN "Build apps with Legacy code-gen" OFF)

        if(NOT OPS_VERBOSE_WARNING)
            message(
                "We show concise compiling information by defautl! Use -DOPS_VERBOSE_WARNING=ON to switch on."
            )
        endif()

        set(CMAKE_VERBOSE_MAKEFILE ${OPS_VERBOSE_WARNING})
        set(OPS_APP_SRC ${CMAKE_SOURCE_DIR})
  
        if(NOT CMAKE_BUILD_TYPE)
            message("Using default mode. Please use -DCMAKE_BUILD_TYPE to change to Release or Debug.")
            set(CMAKE_BUILD_TYPE "")
        endif()

        if(NOT LIBTRID_PATH)
            message(
                WARNING
                    "The install path of the scalar tridiagonal solver library (LIBTRID_PATH) is not specified and the apps which need it will not compiled!"
            )
        endif()

#       Configure C Compiler
        set(CMAKE_C_STANDARD 99)

        if(${CMAKE_C_COMPILER_ID} STREQUAL GNU)
            set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O3 -std=c99 -fPIC -Wall -ffloat-store -g -ftree-vectorize -fopenmp")
            set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} -O3 -std=c99 -fPIC -Wall -ffloat-store -g -ftree-vectorize -fopenmp")
            set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} -O0 -std=c99 -fPIC -Wall -ffloat-store -g -fopenmp")
        elseif(${CMAKE_C_COMPILER_ID} STREQUAL Clang)
            set(CMAKE_C_FLAGS "-fPIC -DUNIX -Wall")
            set(CMAKE_C_FLAGS_RELEASE "-O3")
            set(CMAKE_C_FLAGS_DEBUG "-g -O0")
        elseif(${CMAKE_C_COMPILER_ID} STREQUAL MSVC)
            set(CMAKE_C_FLAGS "-W1")
            set(CMAKE_C_FLAGS_RELEASE "/O2 /Ot ")
            set(CMAKE_C_FLAGS_DEBUG "/Z7 /DEBUG")
        elseif(${CMAKE_C_COMPILER_ID} STREQUAL NVHPC)
            set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O3 -fast -gopt")
            set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} -O3 -fast -gopt")
            set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} -O0 -g")
        elseif(${CMAKE_C_COMPILER_ID} STREQUAL Intel)
            set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O3 -g -xHost -std=c99 -qopenmp -qopt-report")
            set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} -O3 -g -xHost -std=c99 -qopenmp -qopt-report")
            set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} -O0 -xHost -g -std=c99")
        else()
            message(
                WARNING
                    "There is no default flag for the chosen C compiler! Please use -DCFLAG=XXX to supply!"
            )
        endif()

#       Configure CXX Compiler
        set(CMAKE_CXX_STANDARD 11)

        if(${CMAKE_CXX_COMPILER_ID} STREQUAL GNU)
            set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -fPIC -Wall -ffloat-store -g -std=c++11 -fopenmp")
            set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -fPIC -Wall -ffloat-store -g -std=c++11 -fopenmp")
            set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -O0 -fPIC -Wall -ffloat-store -g -std=c++11")
        elseif(${CMAKE_CXX_COMPILER_ID} STREQUAL Clang)
            set(CMAKE_CXX_FLAGS "-fPIC -DUNIX -Wall")
            set(CMAKE_CXX_FLAGS_RELEASE "-O3")
            set(CMAKE_CXX_FLAGS_DEBUG "-g -O0")
        elseif(${CMAKE_CXX_COMPILER_ID} STREQUAL MSVC)
            set(CMAKE_CXX_FLAGS "-W1")
            set(CMAKE_CXX_FLAGS_RELEASE "/O2 /Ot ")
            set(CMAKE_CXX_FLAGS_DEBUG "/Z7 /DEBUG")
        elseif(${CMAKE_C_COMPILER_ID} STREQUAL NVHPC)
            set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -fast -gopt -std=c++11 -mp")
            set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -fast -gopt -std=c++11 -mp")
            set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -O0 -g -std=c++11 -mp")
        elseif(${CMAKE_CXX_COMPILER_ID} STREQUAL Intel)
            set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -g -xHost -qopenmp -qopt-report -std=c++11 -qopt-zmm-usage=high")
            set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -g -xHost -qopenmp -qopt-report -std=c++11 -qopt-zmm-usage=high")
            set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -O0 -g -xHost -DOPS_DEBUG -std=c++11 -DOPS_DEBUG")
        else()
            message(
                WARNING
                    "There is no default flag for the chosen C++ compiler! Please use -DCXXFLAG=XXX to supply!"
            )
        endif()


#       Adding CFLAGS provided to CMAKE_C_FLAGS
        if(CFLAG)
            set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${CFLAG}")
        else()
            message(
                "Extra C compiler options can be provided using CFLAG, e.g., -DCFLAG=\"-ffloat-store\" for GNU!"
            )
        endif()

#       Adding CXXFLAGS provided to CMAKE_CXX_FLAGS
        if(CXXFLAG)
            set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${CXXFLAG}")
        else()
            message(
                "Extra CXX compiler options can be provided using CXXFLAG, e.g., -DCXXFLAG=\"-ffloat-store\" for GNU!"
            )
        endif()

#       If IEEE is enabled
        if(ENABLE_IEEE)
#           Add IEEE flags to CMAKE_CXX_FLAGS
            if(${CMAKE_CXX_COMPILER_ID} STREQUAL Intel)
                set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fp-model strict -fp-model source -prec-div -prec-sqrt")
                set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fp-model strict -fp-model source -prec-div -prec-sqrt")
            elseif(${CMAKE_C_COMPILER_ID} STREQUAL NVHPC)
                set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Kieee -nofma")
                set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Kieee -nofma")
            else()
                message(
                    WARNING
                        "There is no default IEEE flags for chosen compiler"
                )
            endif()
        endif()

        if(CMAKE_BUILD_TYPE STREQUAL "Release")
            message(STATUS "CMAKE_C_FLAGS_RELEASE: ${CMAKE_C_FLAGS_RELEASE}")
            message(STATUS "CMAKE_CXX_FLAGS_RELEASE: ${CMAKE_CXX_FLAGS_RELEASE}")
        elseif(CMAKE_BUILD_TYPE STREQUAL "Debug")
            message(STATUS "CMAKE_C_FLAGS_DEBUG: ${CMAKE_C_FLAGS_DEBUG}")
            message(STATUS "CMAKE_CXX_FLAGS_DEBUG: ${CMAKE_CXX_FLAGS_DEBUG}")
        elseif(CMAKE_BUILD_TYPE STREQUAL "")
            message(STATUS "CMAKE_C_FLAGS: ${CMAKE_C_FLAGS}")
            message(STATUS "CMAKE_CXX_FLAGS: ${CMAKE_CXX_FLAGS}")
        else()
            message(STATUS "Unknown build type: ${CMAKE_BUILD_TYPE}")
        endif()


#       Trying to find the OPS library TODO this shall become FindOPS.cmake later
        set(SEARCH_PATHS "${OPS_INSTALL_DIR}" "/usr/local" "/usr")
#       Tyring to find the include directory
        set(OPS_HEAD
            ops_checkpointing.h
            ops_cuda_reduction.h
            ops_cuda_rt_support.h
            ops_device_rt_support.h
            ops_exceptions.h
            ops_hdf5.h
            ops_hdf5_common.h
            ops_hip_reduction.h
            ops_hip_rt_support.h
            ops_instance.h
            ops_internal1.h
            ops_internal2.h
            ops_lib_core.h
            ops_macros.h
            ops_mpi_core.h
            ops_seq.h
            ops_seq_v2.h
            ops_sycl_reduction.h
            ops_sycl_rt_support.h
            ops_tridiag.h
            ops_util.h
            queue.h)

        set(OPS_INCLUDE_DIR "")
        foreach(head ${OPS_HEAD})
            unset(head_inc CACHE)
            find_path(
                    head_inc
                    NAMES ${head}
                    PATHS ${SEARCH_PATHS}
                    PATH_SUFFIXES include)

            if(head_inc)
                list(APPEND OPS_INCLUDE_DIR ${head_inc})
            else()
                message(FATAL_ERROR "We cannot find ${head}")
            endif()
        endforeach()

        list(REMOVE_DUPLICATES OPS_INCLUDE_DIR)
        include_directories("${OPS_INCLUDE_DIR}")

#       Trying to find the library directory
        if(LIBTRID_PATH)
            set(OPS_LIB
                ops_cuda
                ops_mpi_cuda
                ops_hdf5_seq
                ops_hdf5_mpi
                ops_seq
                ops_mpi
                ops_trid_seq
                ops_trid_mpi
                ops_trid_cuda
                ops_trid_mpi_cuda)
        else()
            set(OPS_LIB
                ops_cuda
                ops_mpi_cuda
                ops_hdf5_seq
                ops_hdf5_mpi
                ops_seq
                ops_mpi)
        endif()

        foreach(library ${OPS_LIB})
            find_library(
                    ${library}_FOUND
                    NAMES ${library}
                    PATHS ${SEARCH_PATHS}
                    PATH_SUFFIXES lib)
            if(${library}_FOUND)
                unset(dir CACHE)
                get_filename_component(dir ${${library}_FOUND} DIRECTORY)
                list(APPEND OPS_LIBRARY_DIR ${dir})
            else()
                message("We cannot find ${library}")
            endif()
        endforeach()

        list(REMOVE_DUPLICATES OPS_LIBRARY_DIR)
        link_directories("${OPS_LIBRARY_DIR}")

#       Set the OPS_INSTALL_PATH for testing
        set(OPS_INSTALL_PATH "${OPS_INSTALL_DIR}")
        set(OPS_CMAKE_INSTALL_PATH "${OPS_INSTALL_DIR}")

#       Try to find the required dependency
        find_package(MPI QUIET)
        find_package(HDF5 QUIET COMPONENTS C HL)
        find_package(CUDAToolkit QUIET)
        find_package(OpenMP QUIET)

        find_package(Python3 3.8 REQUIRED)
        if(NOT Python3_FOUND)
            message(
                WARNING
                    "We cannot find Python3 and the OPS code-generator needs Python3! Please use -DPython3_EXECUTABLE to specify the path."
            )  
        else()
            if(LEGACY_CODEGEN)
                find_path(opsc
                        NAMES "ops.py"
                        PATHS ${SEARCH_PATHS}
                        PATH_SUFFIXES bin/ops_translator_legacy/c)

                find_path(opsfortran
                        NAMES "ops_fortran.py"
                        PATHS ${SEARCH_PATHS}
                        PATH_SUFFIXES bin/ops_translator_legacy/fortran)
            else()
                find_path(opsc
                        NAMES "__main__.py"
                        PATHS ${SEARCH_PATHS}
                        PATH_SUFFIXES bin/ops_translator/ops-translator)

                find_path(opsfortran
                        NAMES "__main__.py"
                        PATHS ${SEARCH_PATHS}
                        PATH_SUFFIXES bin/ops_translator/ops-translator)
            endif()
    
            if(NOT opsc)
                message(FATAL_ERROR "We cannot find the python C/C++ translator!")
            endif()

            if(NOT opsfortran)
                message(FATAL_ERROR "We cannot find the python fortran translator!")
            endif()
        endif()

        if(LEGACY_CODEGEN)
            set(OPS_C_TRANSLATOR "${opsc}/ops.py")
            set(OPS_F_TRANSLATOR "${opsfortran}/ops_fortran.py")
        else()
            set(OPS_C_TRANSLATOR "${opsc}")
            set(OPS_F_TRANSLATOR "${opsfortran}")
        endif()

#       Configure the "include" dir for compiling
        if(NOT HDF5_FOUND)
            message(
                WARNING
                    "We cannot find the parallel HDF5 library. The HDF5 IO routines won't work! Please use -DHDF5_ROOT to specify the path!"
            )
        endif()

        if(NOT MPI_FOUND)
            message(
                WARNING "We cannot find a MPI environment! The MPI codes won't compile!")
        endif()

        if(LIBTRID_PATH)
            include_directories(${LIBTRID_PATH}/include)
        endif()

        if(CUDAToolkit_FOUND)
            set(CMAKE_CUDA_COMPILER ${CUDAToolkit_NVCC_EXECUTABLE})
            if(GPU_ARCH)
                set(CMAKE_CUDA_ARCHITECTURES
                    ${GPU_ARCH}
                    CACHE STRING "CUDA architectures")
            else()
                set(CMAKE_CUDA_ARCHITECTURES
                    70
                    CACHE STRING "CUDA architectures")
                message(WARNING "Using Volta architecure for compiling CUDA. \nIf different please provide correct GPU architecture using -DGPU_ARCH=XXX  where XXX is compute capability of an architecture!")
            endif()

            enable_language(CUDA)

        else()
            message(WARNING "We cannot find a CUDA toolkit! The CUDA codes won't compile!")
        endif()

        if(USE_NCCL)
            find_package(NCCL)
            if(NOT NCCL_FOUND)
                message(
                    WARNING
                        "We cannot find NCCL! Tridiagonal library wrapper will be compiled with NCCL"
                )
            endif()
        endif()

        set(HDF5_SEQ ${HDF5_FOUND})

        if(ops_cuda_FOUND)
            set(CUDA ${CUDAToolkit_FOUND})
        endif()

        if(${MPI_FOUND})
            set(MPI TRUE)
            set(HDF5_MPI ${HDF5_FOUND})
            if(ops_cuda_FOUND)
                set(CUDA_MPI ${CUDAToolkit_FOUND})
            endif()

#           Find the MPI INCLUDE directory, required to resolve mpi.h not found error for some targets
            if(DEFINED ENV{I_MPI_ROOT})     #Intel MPI
                set(MPI_INCLUDE_DIRS $ENV{I_MPI_ROOT}/include)
            elseif(DEFINED ENV{MPI_ROOT})       # HPE MPT
                set(MPI_INCLUDE_DIRS $ENV{MPI_ROOT}/include)
            elseif(DEFINED ENV{MPICH_DIR})      # Cray
                set(MPI_INCLUDE_DIRS $ENV{MPICH_DIR}/include)
            else()                          # OpenMPI
                execute_process(COMMAND mpicc --showme:incdirs
                                OUTPUT_VARIABLE MPI_INCLUDE_DIRS
                                OUTPUT_STRIP_TRAILING_WHITESPACE
                                ERROR_VARIABLE MPI_INCLUDE_DIRS_ERROR)
                if(MPI_INCLUDE_DIRS_ERROR)
                    message(FATAL_ERROR "Failed to determine MPI include directories: ${MPI_INCLUDE_DIRS_ERROR}")
                endif()
            endif()

            if(NOT EXISTS ${MPI_INCLUDE_DIRS})
                message(FATAL_ERROR "MPI include directory ${MPI_INCLUDE_DIRS} does not exist.")
            else()
                message(STATUS "MPI include directories: ${MPI_INCLUDE_DIRS}")
            endif()

        endif()

#       Prepare ctest
        if(OPS_TEST)
            enable_testing()
            include(ProcessorCount)
            ProcessorCount(CPU_NUMBER)
            if(CPU_NUMBER GREATER 12)
                set(CPU_NUMBER 12)
            endif()
            if(NOT GPU_NUMBER)
                message(
                    WARNING "GPU tests are disabled! Please use -DGPU_NUMBER to specify!")
            endif()
        endif()
    endif()

#   If APPs are compiled together with the OPS API
    if(${CMAKE_PROJECT_NAME} STREQUAL OPS)

        include_directories("${CMAKE_SOURCE_DIR}/ops/c/include")
        set(OPS_INCLUDE_DIR "${CMAKE_SOURCE_DIR}/ops/c/include")

        set(HDF5_SEQ ${HDF5_FOUND})
        set(CUDA ${CUDAToolkit_FOUND})

        if(OPS_HIP)
            set(HIP ${HIP_FOUND})
        endif()

        if(${MPI_FOUND})
            set(MPI TRUE)
            set(HDF5_MPI ${HDF5_FOUND})
            set(CUDA_MPI ${CUDAToolkit_FOUND})
        endif()

        if(NOT Python3_FOUND)
            message(
                WARNING
                    "We cannot find Python3 and the OPS code-generator needs Python3! Please use -DPython3_EXECUTABLE to specify the path."
            )
        else()
            if(LEGACY_CODEGEN)
                set(OPS_C_TRANSLATOR "${CMAKE_INSTALL_PREFIX}/bin/ops_translator_legacy/c/ops.py")
                set(OPS_F_TRANSLATOR "${CMAKE_INSTALL_PREFIX}/bin/ops_translator_legacy/fortran/ops_fortran.py")
            else()
                set(OPS_C_TRANSLATOR "${CMAKE_INSTALL_PREFIX}/bin/ops_translator/ops-translator")
                set(OPS_F_TRANSLATOR "${CMAKE_INSTALL_PREFIX}/bin/ops_translator/ops-translator")
            endif()
        endif()

#       Set the OPS_INSTALL_PATH for testing
        set(OPS_CMAKE_INSTALL_PATH "${CMAKE_INSTALL_PREFIX}")
        set(OPS_INSTALL_PATH "${CMAKE_SOURCE_DIR}/ops/c")
        set(OPS_APP_SRC "${CMAKE_SOURCE_DIR}/apps/c")

        if(OPS_TEST)
            enable_testing()
        endif()
    endif()

#   Clean the tmp directory for the code generation
    macro(CreateTempDir)
        file(REMOVE_RECURSE "${CMAKE_CURRENT_BINARY_DIR}/tmp")
        file(MAKE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/tmp")
    endmacro()

#   Prepare the macro for compiling apps Name: App name Odd: Key words for source
#   files are included in other source files and need no explicit compilation
#   Others: Key words for source codes need no code generation using ops.py Extra:
#   Key words for ource codes that cannot be treated by the macro Trid: Whether
#   the tridiagonal library is required for this application GenerateTest: if
#   generating the testings, some apps need specific testing targets. 
#   TODO: tiled, sycl, openmpoffload
    macro(
        BUILD_OPS_C_SAMPLE
        Name
        Odd
        Others
        Extra
        Trid
        GenerateTest)

#       Copy all source and head files into tmp
        file(GLOB CPP "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp")
        
        list(FILTER CPP EXCLUDE REGEX "ops.cpp")

        file(COPY ${CPP} DESTINATION "${CMAKE_CURRENT_BINARY_DIR}/tmp/")
        file(GLOB HEADS "${CMAKE_CURRENT_SOURCE_DIR}/*.h")
        file(COPY ${HEADS} DESTINATION "${CMAKE_CURRENT_BINARY_DIR}/tmp/")

        set(TMP_SOURCE_DIR "${CMAKE_CURRENT_BINARY_DIR}/tmp")

#       Get the file to be compiled, DEV: non-optimised codes (full directory and filename) 
#       OPS: codes generated by ops.py (full directory and filename)
#       OTHERS: Source codes need no code generation (full directory and filename)
        file(GLOB DEV "${TMP_SOURCE_DIR}/*.cpp")
        list(FILTER DEV EXCLUDE REGEX ${Odd})
        foreach(oth ${Others})
                list(FILTER DEV EXCLUDE REGEX ${oth})
        endforeach()
        foreach(ext ${Extra})
            list(FILTER DEV EXCLUDE REGEX ${ext})
        endforeach()

        list(GET DEV 0 Kernel)

        get_filename_component(KernelName ${Kernel} NAME_WE)

#       If not LEGACY_CODEGEN then activate the Installed Python Virtual Environment
        if(${CMAKE_PROJECT_NAME} STREQUAL APP)
            if(NOT LEGACY_CODEGEN)
                message("Activating Python Virtual Environment for Code-Generation")
                set(VENV_DEACTIVATE "deactivate")
                set(VENV_ACTIVATE "source ${OPS_CMAKE_INSTALL_PATH}/bin/ops_translator/ops_venv/bin/activate")
                #execute_process(COMMAND /bin/bash -c ${VENV_DEACTIVATE}
                #                WORKING_DIRECTORY ${TMP_SOURCE_DIR})

                execute_process(COMMAND /bin/bash -c ${VENV_ACTIVATE}
                                WORKING_DIRECTORY ${TMP_SOURCE_DIR}
                                RESULT_VARIABLE result
                                OUTPUT_VARIABLE output)

                if(result EQUAL 0)
                    set(PYTHON_EXECUTABLE "${OPS_CMAKE_INSTALL_PATH}/bin/ops_translator/ops_venv/bin/python3")
                    message(STATUS "Python Virtual Enviroment activated successfully")
                else()
                    message("command: ${VENV_ACTIVATE}")
                    message(FATAL_ERROR "Activate Python Virtual Environment command failed with error code ${result}: ${output}")
                endif()
            endif()
        endif()

#       Run OPS code-generation
        if(LEGACY_CODEGEN)
            message(STATUS "Code-gen command: ${OPS_C_TRANSLATOR} ${DEV}")
            execute_process(COMMAND ${OPS_C_TRANSLATOR} ${DEV}
                            WORKING_DIRECTORY ${TMP_SOURCE_DIR}
                            RESULT_VARIABLE result
                            OUTPUT_VARIABLE output)
        else()
            set(PYTHON_FILE_ARGS "-DOPS_ACC_IGNORE -I ${OPS_CMAKE_INSTALL_PATH}/include --file_paths")

            foreach(filepath ${DEV})
                get_filename_component(filename ${filepath} NAME)
                list(APPEND FILENAMES ${filename})
            endforeach()

            message(STATUS "Code-gen command: ${PYTHON_EXECUTABLE} ${OPS_C_TRANSLATOR} ${PYTHON_FILE_ARGS} ${FILENAMES}")
            execute_process(COMMAND ${PYTHON_EXECUTABLE} ${OPS_C_TRANSLATOR}
                            -DOPS_ACC_IGNORE
                            -I ${OPS_CMAKE_INSTALL_PATH}/include
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


        file(GLOB OPS "${TMP_SOURCE_DIR}/*ops*.cpp")
#       message(${OPS})

        list(FILTER OPS EXCLUDE REGEX ${Odd})
#       message(${OPS})
  
        file(GLOB OTHERS "${TMP_SOURCE_DIR}/*.cpp")
        foreach(OpsFile ${OPS})
                list(REMOVE_ITEM OTHERS ${OpsFile})
            string(REPLACE "_ops" "" NoOPS ${OpsFile})
            list(REMOVE_ITEM OTHERS ${NoOPS})
        endforeach()

        list(FILTER OTHERS EXCLUDE REGEX ${Odd})

        foreach(ext ${Extra})
            list(FILTER OTHERS EXCLUDE REGEX ${ext})
        endforeach()

#       Copy input parameters
        file(GLOB_RECURSE INPUT "${CMAKE_CURRENT_SOURCE_DIR}/*.in")
        file(COPY ${INPUT} DESTINATION "${TMP_SOURCE_DIR}")
        if(EXISTS "${TMP_SOURCE_DIR}/user_types.h")
            install(FILES "${TMP_SOURCE_DIR}/user_types.h"
                    DESTINATION ${APP_INSTALL_DIR}/${Name})
        endif()

        install(FILES ${INPUT} DESTINATION ${APP_INSTALL_DIR}/${Name})


#       TARGET: DEV_SEQ, SEQ, OPENMP, TILED
        add_executable(${Name}_dev_seq ${DEV} ${OTHERS})
        target_include_directories(${Name}_dev_seq PRIVATE ${TMP_SOURCE_DIR})

        add_executable(${Name}_seq ${OPS} ${OTHERS}
                       "${TMP_SOURCE_DIR}/mpi_openmp/mpi_openmp_kernels.cpp")
        target_include_directories(${Name}_seq PRIVATE ${TMP_SOURCE_DIR})

        add_executable(${Name}_openmp ${OPS} ${OTHERS}
                       "${TMP_SOURCE_DIR}/mpi_openmp/mpi_openmp_kernels.cpp")
        target_include_directories(${Name}_openmp PRIVATE ${TMP_SOURCE_DIR})

        add_executable(${Name}_tiled ${OPS} ${OTHERS}
                       "${TMP_SOURCE_DIR}/mpi_openmp/mpi_openmp_kernels.cpp")
        target_include_directories(${Name}_tiled PRIVATE ${TMP_SOURCE_DIR})
        target_compile_definitions(${Name}_tiled PRIVATE "-DOPS_LAZY")

        if(HDF5_SEQ)
            target_link_libraries(${Name}_dev_seq ops_hdf5_seq hdf5::hdf5 hdf5::hdf5_hl
                                    MPI::MPI_CXX)
            target_link_libraries(${Name}_seq ops_hdf5_seq hdf5::hdf5 hdf5::hdf5_hl
                                    MPI::MPI_CXX)
            target_link_libraries(${Name}_openmp ops_hdf5_seq hdf5::hdf5 hdf5::hdf5_hl
                                    MPI::MPI_CXX)
            target_link_libraries(${Name}_tiled ops_hdf5_seq hdf5::hdf5 hdf5::hdf5_hl
                                    MPI::MPI_CXX)
        endif()

        if(${Trid} STREQUAL "YES")
            set(TRID Yes)
        else()
            set(TRID No)
        endif()

        if(TRID)
            target_link_libraries(${Name}_dev_seq ops_trid_seq -L${LIBTRID_PATH}/lib -ltridcpu)
            target_link_libraries(${Name}_seq     ops_trid_seq -L${LIBTRID_PATH}/lib -ltridcpu)
            target_link_libraries(${Name}_openmp  ops_trid_seq -L${LIBTRID_PATH}/lib -ltridcpu)
            target_link_libraries(${Name}_tiled   ops_trid_seq -L${LIBTRID_PATH}/lib -ltridcpu)
        endif()

        target_link_libraries(${Name}_dev_seq ops_seq OpenMP::OpenMP_CXX)
        target_link_libraries(${Name}_seq     ops_seq OpenMP::OpenMP_CXX)
        target_link_libraries(${Name}_openmp  ops_seq OpenMP::OpenMP_CXX)
        target_link_libraries(${Name}_tiled   ops_seq OpenMP::OpenMP_CXX)

        install(TARGETS ${Name}_dev_seq DESTINATION ${APP_INSTALL_DIR}/${Name})
        install(TARGETS ${Name}_seq     DESTINATION ${APP_INSTALL_DIR}/${Name})
        install(TARGETS ${Name}_openmp  DESTINATION ${APP_INSTALL_DIR}/${Name})
        install(TARGETS ${Name}_tiled   DESTINATION ${APP_INSTALL_DIR}/${Name})

#       Sequential tests disabled since it is too time consuming for some tests at this moment 
#       if ((OPS_TEST) AND (${GenerateTest} STREQUAL "YES")) 
#           set(args "") 
#           Here we must have a space to be parsed by the macro 
#           set(cmd "$<TARGET_FILE:${Name}_seq>") 
#           add_test(NAME ${Name}_seq 
#                    COMMAND ${CMAKE_COMMAND} 
#                    -DCMD=${cmd}  -DARG=${args}
#                    -DOPS_INSTALL_PATH=${OPS_CMAKE_INSTALL_PATH} -P ${OPS_APP_SRC}/runtests.cmake
#                    WORKING_DIRECTORY "${TMP_SOURCE_DIR}" ) 
#          endif()

        if(CUDA)
            add_executable(${Name}_cuda ${OPS} ${OTHERS}
                            "${TMP_SOURCE_DIR}/cuda/cuda_kernels.cu")
            target_include_directories(${Name}_cuda PRIVATE ${TMP_SOURCE_DIR})
            target_link_libraries(${Name}_cuda ops_cuda CUDA::cudart_static -lcurand
                                  OpenMP::OpenMP_CXX)
            target_compile_options(${Name}_cuda PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
                                                --fmad=false>)
            if(HDF5_SEQ)
                target_link_libraries(${Name}_cuda ops_hdf5_seq hdf5::hdf5 hdf5::hdf5_hl
                                        MPI::MPI_CXX)
            endif()

            if(TRID)
                target_link_libraries(${Name}_cuda ops_trid_cuda -L${LIBTRID_PATH}/lib -ltridcuda
                                CUDA::cublas_static CUDA::cublasLt_static CUDA::culibos)
            endif()

            install(TARGETS ${Name}_cuda DESTINATION ${APP_INSTALL_DIR}/${Name})
    
            if((OPS_TEST)
                AND (GPU_NUMBER GREATER_EQUAL 1)
                AND (${GenerateTest} STREQUAL "YES"))

                set(args "OPS_CL_DEVICE=1 OPS_BLOCK_SIZE_X=512 OPS_BLOCK_SIZE_Y=1")
                set(cmd "$<TARGET_FILE:${Name}_cuda>")
                add_test(NAME ${Name}_cuda
                        COMMAND
                        ${CMAKE_COMMAND} -DCMD=${cmd} -DARG=${args}
                        -DOPS_INSTALL_PATH=${OPS_CMAKE_INSTALL_PATH} -P
                        ${OPS_APP_SRC}/runtests.cmake
                        WORKING_DIRECTORY "${TMP_SOURCE_DIR}")
            endif()
        endif()

        if(OPS_HIP)
            if(HIP AND NOT TRID)
                add_executable(${Name}_hip ${OPS} ${OTHERS}
                                "${TMP_SOURCE_DIR}/hip/hip_kernels.cpp")
                message("${TMP_SOURCE_DIR}/hip/${KernelName}_hip_kernel.cpp")

            target_include_directories(${Name}_hip PRIVATE ${TMP_SOURCE_DIR})
            target_link_libraries(${Name}_hip ops_hip hip::device OpenMP::OpenMP_CXX)

#           if (HDF5_SEQ) target_link_libraries(${Name}_hip ops_hdf5_seq hdf5::hdf5
#                                               hdf5::hdf5_hl  MPI::MPI_CXX) endif()
            install(TARGETS ${Name}_hip DESTINATION ${APP_INSTALL_DIR}/${Name})
            if((OPS_TEST)
                AND (GPU_NUMBER GREATER_EQUAL 1)
                AND (${GenerateTest} STREQUAL "YES"))

                set(args "OPS_CL_DEVICE=1 OPS_BLOCK_SIZE_X=512 OPS_BLOCK_SIZE_Y=1")
                add_test(NAME ${Name}_hip
                        COMMAND
                        ${CMAKE_COMMAND} -DCMD=$<TARGET_FILE:${Name}_opencl> -DARG=${args}
                        -DOPS_INSTALL_PATH=${OPS_CMAKE_INSTALL_PATH} -P
                        ${OPS_APP_SRC}/runtests.cmake
                        WORKING_DIRECTORY "${TMP_SOURCE_DIR}")
                endif()
            endif()
        endif()

        if(MPI)
            add_executable(${Name}_dev_mpi ${DEV} ${OTHERS})
            target_include_directories(${Name}_dev_mpi PRIVATE ${TMP_SOURCE_DIR})
            target_compile_definitions(${Name}_dev_mpi PRIVATE "-DOPS_MPI")

            add_executable(${Name}_mpi ${OPS} ${OTHERS}
                            "${TMP_SOURCE_DIR}/mpi_openmp/mpi_openmp_kernels.cpp")
            target_include_directories(${Name}_mpi PRIVATE ${TMP_SOURCE_DIR})
            target_compile_definitions(${Name}_mpi PRIVATE "-DOPS_MPI")

            add_executable(${Name}_mpi_openmp ${OPS} ${OTHERS}
                            "${TMP_SOURCE_DIR}/mpi_openmp/mpi_openmp_kernels.cpp")
            target_include_directories(${Name}_mpi_openmp PRIVATE ${TMP_SOURCE_DIR})
            target_compile_definitions(${Name}_mpi_openmp PRIVATE "-DOPS_MPI")

            add_executable(${Name}_mpi_tiled ${OPS} ${OTHERS}
                            "${TMP_SOURCE_DIR}/mpi_openmp/mpi_openmp_kernels.cpp")
            target_include_directories(${Name}_mpi_tiled PRIVATE ${TMP_SOURCE_DIR})
            target_compile_definitions(${Name}_mpi_tiled PRIVATE "-DOPS_MPI -DOPS_LAZY")

            if(HDF5_MPI)
                target_link_libraries(${Name}_dev_mpi PRIVATE ops_hdf5_mpi hdf5::hdf5
                                                    hdf5::hdf5_hl MPI::MPI_CXX)
                target_link_libraries(${Name}_mpi PRIVATE ops_hdf5_mpi hdf5::hdf5
                                                hdf5::hdf5_hl MPI::MPI_CXX)
                target_link_libraries(${Name}_mpi_openmp PRIVATE ops_hdf5_mpi hdf5::hdf5
                                                hdf5::hdf5_hl MPI::MPI_CXX)
                target_link_libraries(${Name}_mpi_tiled PRIVATE ops_hdf5_mpi hdf5::hdf5
                                                hdf5::hdf5_hl MPI::MPI_CXX)
            endif()

            if(TRID)
                target_link_libraries(${Name}_dev_mpi PRIVATE ops_trid_mpi -L${LIBTRID_PATH}/lib -ltridcpu_mpi
                                        MPI::MPI_CXX)
                target_link_libraries(${Name}_mpi PRIVATE ops_trid_mpi -L${LIBTRID_PATH}/lib -ltridcpu_mpi
                                        MPI::MPI_CXX)
                target_link_libraries(${Name}_mpi_openmp PRIVATE ops_trid_mpi -L${LIBTRID_PATH}/lib -ltridcpu_mpi
                                        MPI::MPI_CXX)
                target_link_libraries(${Name}_mpi_tiled PRIVATE ops_trid_mpi -L${LIBTRID_PATH}/lib -ltridcpu_mpi
                                        MPI::MPI_CXX)
            endif()

            target_link_libraries(${Name}_dev_mpi PRIVATE ops_mpi MPI::MPI_CXX
                                              OpenMP::OpenMP_CXX)
            target_link_libraries(${Name}_mpi PRIVATE ops_mpi MPI::MPI_CXX
                                              OpenMP::OpenMP_CXX)
            target_link_libraries(${Name}_mpi_openmp PRIVATE ops_mpi MPI::MPI_CXX
                                              OpenMP::OpenMP_CXX)
            target_link_libraries(${Name}_mpi_tiled PRIVATE ops_mpi MPI::MPI_CXX
                                              OpenMP::OpenMP_CXX)

            if((OPS_TEST) AND (${GenerateTest} STREQUAL "YES"))
                set(args "-n ${CPU_NUMBER} $<TARGET_FILE:${Name}_mpi>")
                set(cmd "mpirun")
                add_test(NAME ${Name}_mpi
                        COMMAND
                        ${CMAKE_COMMAND} -DCMD=${cmd} -DARG=${args}
                        -DOPS_INSTALL_PATH=${OPS_CMAKE_INSTALL_PATH} -P
                        ${OPS_APP_SRC}/runtests.cmake
                        WORKING_DIRECTORY "${TMP_SOURCE_DIR}")
            endif()

            install(TARGETS ${Name}_dev_mpi DESTINATION ${APP_INSTALL_DIR}/${Name})
            install(TARGETS ${Name}_mpi DESTINATION ${APP_INSTALL_DIR}/${Name})
            install(TARGETS ${Name}_mpi_openmp DESTINATION ${APP_INSTALL_DIR}/${Name})
            install(TARGETS ${Name}_mpi_tiled DESTINATION ${APP_INSTALL_DIR}/${Name})

            if(CUDA_MPI)
                add_executable(${Name}_mpi_cuda ${OPS} ${OTHERS}
                                "${TMP_SOURCE_DIR}/cuda/cuda_kernels.cu")
                target_include_directories(${Name}_mpi_cuda PRIVATE ${TMP_SOURCE_DIR} ${MPI_INCLUDE_DIRS})
                target_compile_definitions(${Name}_mpi_cuda PRIVATE "-DOPS_MPI -DMPICH_IGNORE_CXX_SEEK")
                target_compile_options(${Name}_mpi_cuda
                             PRIVATE $<$<COMPILE_LANGUAGE:CUDA>: --fmad=false >)

                add_executable(${Name}_mpi_cuda_tiled ${OPS} ${OTHERS}
                                "${TMP_SOURCE_DIR}/cuda/cuda_kernels.cu")
                target_include_directories(${Name}_mpi_cuda_tiled PRIVATE ${TMP_SOURCE_DIR} ${MPI_INCLUDE_DIRS})
                target_compile_definitions(${Name}_mpi_cuda_tiled PRIVATE "-DOPS_MPI -DMPICH_IGNORE_CXX_SEEK -DOPS_LAZY")
                target_compile_options(${Name}_mpi_cuda_tiled
                             PRIVATE $<$<COMPILE_LANGUAGE:CUDA>: --fmad=false >)

                if(HDF5_MPI)
                    target_link_libraries(${Name}_mpi_cuda PRIVATE ops_hdf5_mpi hdf5::hdf5 hdf5::hdf5_hl
                                            MPI::MPI_CXX)
                    target_link_libraries(${Name}_mpi_cuda_tiled PRIVATE ops_hdf5_mpi hdf5::hdf5 hdf5::hdf5_hl
                                            MPI::MPI_CXX)
                endif()

                if(TRID)
                    target_link_libraries(${Name}_mpi_cuda PRIVATE ops_trid_mpi_cuda -L${LIBTRID_PATH}/lib -ltridcuda_mpi
                                            MPI::MPI_CXX)
                    target_link_libraries(${Name}_mpi_cuda_tiled PRIVATE ops_trid_mpi_cuda -L${LIBTRID_PATH}/lib -ltridcuda_mpi
                                            MPI::MPI_CXX)
                    if(NCCL_FOUND)
                        target_link_libraries(${Name}_mpi_cuda PRIVATE ${NCCL_LIBRARY})
                        target_link_libraries(${Name}_mpi_cuda_tiled PRIVATE ${NCCL_LIBRARY})
                    endif()
                endif()

                target_link_libraries(${Name}_mpi_cuda PRIVATE ops_mpi_cuda CUDA::cudart_static -lcurand MPI::MPI_CXX
                                        OpenMP::OpenMP_CXX)
                target_link_libraries(${Name}_mpi_cuda_tiled PRIVATE ops_mpi_cuda CUDA::cudart_static -lcurand MPI::MPI_CXX
                                        OpenMP::OpenMP_CXX)

                install(TARGETS ${Name}_mpi_cuda DESTINATION ${APP_INSTALL_DIR}/${Name})
                install(TARGETS ${Name}_mpi_cuda_tiled DESTINATION ${APP_INSTALL_DIR}/${Name})
            endif()

        endif()

    endmacro()

#   TODO fetch_data.cpp cannot be found for access add_subdirectory(access)
    if(LIBTRID_PATH)
        add_subdirectory(adi)
        add_subdirectory(adi_burger)
        add_subdirectory(adi_burger_3D)
    endif()

    add_subdirectory(CloverLeaf) # Compile OK
    add_subdirectory(CloverLeaf_3D) # Compile OK
    add_subdirectory(CloverLeaf_3D_HDF5) # Compile OK
#   TODO there seem bugs in the codes of complex_numbers
#   add_subdirectory(complex_numbers) TODO Need to figure out the structure
#   add_subdirectory(laplace2d_tutorial)
    add_subdirectory(lowdim_test)
#   TODO no PASSED for testing
    add_subdirectory(mblock)
#   TODO there seems problem for compiling mblock4D add_subdirectory(mblock4D) #
#   ignored at this moment TODO needs to replace  acceptable with PASSED for
#   standard test
    add_subdirectory(mb_shsgc)
#   add_subdirectory(mgrid)  # Compile OK
    add_subdirectory(multiDim)
    add_subdirectory(multiDim3D)
    add_subdirectory(multiDim_HDF5)
    add_subdirectory(poisson)
    add_subdirectory(shsgc)
    add_subdirectory(TeaLeaf)
    add_subdirectory(compact_scheme)
    add_subdirectory(hdf5_slice)
