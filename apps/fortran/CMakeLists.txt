#   Specifiy the cmake version range, currently work under 3.18
    cmake_minimum_required(VERSION 3.18)

#           Project-NAME LANGUAGE/s
    project(APP C CXX Fortran)

    include(CheckLanguage)
    check_language(Fortran)

    if(CMAKE_Fortran_COMPILER)
        if(NOT APP_INSTALL_DIR)
            set(APP_INSTALL_DIR "$ENV{HOME}/OPS-APPS")
            message(
                WARNING
                    "We will install the applications to ${APP_INSTALL_DIR} \n! Please use -DAPP_INSTALL_DIR to specify!"
            )
        endif()

#       If APPs are compiled independently then setup compilers and dependencies.
        if(${CMAKE_PROJECT_NAME} STREQUAL APP)
#           Show the compiling process in detail
            option(OPS_VERBOSE_WARNING "Turn on verbose warning messages" OFF)
            option(OPS_TEST "Turn on tests for Apps" OFF)
            option(LEGACY_CODEGEN "Build apps with Legacy code-gen" OFF)

            if(NOT OPS_VERBOSE_WARNING)
                message(
                    "We show concise compiling information by default \n! Use -DOPS_VERBOSE_WARNING=ON to switch on."
                )
            endif()

            message(STATUS "Fortran compiler: ${CMAKE_Fortran_COMPILER}")

            if(${CMAKE_Fortran_COMPILER_ID} STREQUAL Intel)
                set(CMAKE_Fortran_FLAGS         "${CMAKE_Fortran_FLAGS} -O3 -g -xHost -qopenmp")
                set(CMAKE_Fortran_FLAGS_DEBUG   "-g -traceback -qopenmp")
                set(CMAKE_Fortran_FLAGS_RELEASE "-O3 -ip -xHOST -qopenmp")
            elseif(${CMAKE_Fortran_COMPILER_ID} STREQUAL GNU)
                set(CMAKE_Fortran_FLAGS         "${CMAKE_Fortran_FLAGS} -O3 -g -ffree-form -ffree-line-length-none -fopenmp")
                set(CMAKE_Fortran_FLAGS_DEBUG   "-O0 -g3 -ffree-form -ffree-line-length-none -fopenmp")
                set(CMAKE_Fortran_FLAGS_RELEASE "-Ofast -march=native -ffree-form -ffree-line-length-none -fopenmp")
            elseif(${CMAKE_Fortran_COMPILER_ID} STREQUAL NVHPC)
                set(CMAKE_Fortran_FLAGS         "${CMAKE_Fortran_FLAGS} -O3 -fast -gopt -mp")
                set(MPI_LINK "-lmpi_cxx")
            else()
                message(
                    WARNING
                        "There is no default flag for the chosen Fortran compiler!"
                )
            endif()

            set(CMAKE_VERBOSE_MAKEFILE ${OPS_VERBOSE_WARNING})
            set(OPS_APP_SRC ${CMAKE_SOURCE_DIR})

            if(NOT CMAKE_BUILD_TYPE)
                message("Using default mode. Please use -DCMAKE_BUILD_TYPE to change to Release or Debug.")
                set(CMAKE_BUILD_TYPE "")
            endif()

            set(SEARCH_PATHS "${OPS_INSTALL_DIR}" "/usr/local" "/usr")

#           Tyring to find the include directory
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
                ops_macros.inc
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

            set(OPS_LIB
                ops_for_cuda
                ops_for_mpi_cuda
                ops_for_hdf5_seq
                ops_for_hdf5_mpi
                ops_for_seq
                ops_for_mpi)

            set(OPS_LIBRARY_DIR "")
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

#           Set the OPS_INSTALL_PATH for testing
            set(OPS_INSTALL_PATH "${OPS_INSTALL_DIR}")
            set(OPS_CMAKE_INSTALL_PATH "${OPS_INSTALL_DIR}")

#           Try to find the required dependency
            find_package(MPI QUIET)
            find_package(HDF5 QUIET COMPONENTS C Fortran HL)
            find_package(CUDAToolkit QUIET)
            find_package(OpenMP QUIET)

            find_package(Python3 3.8 REQUIRED)
            if(NOT Python3_FOUND)
                message(
                    WARNING
                        "We cannot find Python3.8 or higher and the OPS code-generator needs Python3! Please use -DPython3_EXECUTABLE to specify the path."
                )
            else()
                if(LEGACY_CODEGEN)
                    find_path(opsc
                            NAMES "ops.py"
                            PATHS ${SEARCH_PATHS}
                            PATH_SUFFIXES translator/ops_translator_legacy/c)

                    find_path(opsfortran
                            NAMES "ops_fortran.py"
                            PATHS ${SEARCH_PATHS}
                            PATH_SUFFIXES translator/ops_translator_legacy/fortran)
                else()
                    find_path(opsc
                            NAMES "__main__.py"
                            PATHS ${SEARCH_PATHS}
                            PATH_SUFFIXES translator/ops_translator/ops-translator)

                    find_path(opsfortran
                            NAMES "__main__.py"
                            PATHS ${SEARCH_PATHS}
                            PATH_SUFFIXES translator/ops_translator/ops-translator)
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

#           Configure the "include" dir for compiling
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
                    message(WARNING "Using Volta architecure for compiling CUDA. \n If different please provide correct GPU architecture using -DGPU_ARCH=XXX  where XXX is compute capability of an architecture \n For ex. -DGPU_ARCH=70 if you want to target sm_70")
                endif()

                enable_language(CUDA)

            else()
                message(WARNING "We cannot find a CUDA toolkit! The CUDA codes won't compile!")
            endif()

            set(HDF5_SEQ ${HDF5_FOUND})

            if(ops_cuda_FOUND)
                set(CUDA ${CUDAToolkit_FOUND})
            endif()

            if(${MPI_FOUND})
                set(CMAKE_Fortran_COMPILER ${MPI_Fortran_COMPILER})
                set(MPI TRUE)
                set(HDF5_MPI ${HDF5_FOUND})
                set(CUDA_MPI ${CUDAToolkit_FOUND})
            endif()

#           Prepare ctest
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

#       If APPs are compiled together with the OPS API
        if(${CMAKE_PROJECT_NAME} STREQUAL OPS)

            include_directories("${CMAKE_SOURCE_DIR}/ops/c/include")
            set(OPS_INCLUDE_DIR "${CMAKE_SOURCE_DIR}/ops/c/include")

            message(STATUS "Fortran compiler: ${CMAKE_Fortran_COMPILER}")

            if(${CMAKE_Fortran_COMPILER_ID} STREQUAL Intel)
                set(CMAKE_Fortran_FLAGS         "${CMAKE_Fortran_FLAGS} -O3 -g -xHost -qopenmp")
                set(CMAKE_Fortran_FLAGS_DEBUG   "-g -traceback -qopenmp")
                set(CMAKE_Fortran_FLAGS_RELEASE "-O3 -ip -xHOST -qopenmp")
            elseif(${CMAKE_Fortran_COMPILER_ID} STREQUAL GNU)
                set(CMAKE_Fortran_FLAGS         "${CMAKE_Fortran_FLAGS} -O3 -g -ffree-form -ffree-line-length-none -fopenmp")
                set(CMAKE_Fortran_FLAGS_DEBUG   "-O0 -g3 -ffree-form -ffree-line-length-none -fopenmp")
                set(CMAKE_Fortran_FLAGS_RELEASE "-Ofast -march=native -ffree-form -ffree-line-length-none -fopenmp")
            elseif(${CMAKE_Fortran_COMPILER_ID} STREQUAL NVHPC)
                set(CMAKE_Fortran_FLAGS         "${CMAKE_Fortran_FLAGS} -O3 -fast -gopt -mp")
                set(MPI_LINK "-lmpi_cxx")
            else()
                message(
                    WARNING
                        "There is no default flag for the chosen Fortran compiler!"
                )
            endif()

            set(HDF5_SEQ ${HDF5_FOUND})
            if(${MPI_FOUND})
                set(CMAKE_Fortran_COMPILER ${MPI_Fortran_COMPILER})
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

#       Find the MPI INCLUDE directory, required to resolve mpi.h not found error for some targets
        if(${MPI_FOUND})
            if(DEFINED ENV{I_MPI_ROOT})     #Intel MPI
                set(MPI_INC_LIST -I$ENV{I_MPI_ROOT}/include)
            elseif(DEFINED ENV{MPI_ROOT})       # HPE MPT
                set(MPI_INC_LIST -I$ENV{MPI_ROOT}/include)
            elseif(DEFINED ENV{MPICH_DIR})      # Cray
                set(MPI_INC_LIST -I$ENV{MPICH_DIR}/include)
            else()                          # OpenMPI
                execute_process(COMMAND mpicc --showme:compile
                                OUTPUT_VARIABLE MPI_INC_STRING
                                OUTPUT_STRIP_TRAILING_WHITESPACE
                                ERROR_VARIABLE MPI_INC_FIND)
                if(MPI_INC_FIND)
                    message(FATAL_ERROR "Failed to determine MPI include directories: ${MPI_INC_FIND}")
                endif()
                string(REPLACE " " ";" MPI_INC_LIST ${MPI_INC_STRING})

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

            file(GLOB CONSTANTS_MODULE "${TMP_SOURCE_DIR}/constants.F90")
            list(FILTER OTHERS EXCLUDE REGEX ${CONSTANTS_MODULE})

            file(GLOB_RECURSE SEQ_TEMPLATES "${TMP_SOURCE_DIR}/mpi_openmp/*_seq_kernel.F90")
            foreach(source_file ${SEQ_TEMPLATES})
                set_source_files_properties(${source_file} PROPERTIES GENERATED TRUE)
            endforeach()

#            message(STATUS "OPS: ${OPS}")
#            message(STATUS "OTHERS: ${OTHERS}")
#            message(STATUS "SEQ_TEMPLATES: ${SEQ_TEMPLATES}")

#           Copying the required mod files
            set(SOURCE_DIR "${OPS_CMAKE_INSTALL_PATH}/mod")
            set(DESTINATION_DIR "${OPS_CMAKE_INSTALL_PATH}/mod/${Name}")

            set(SOURCE_MOD_FILES
                "ops_fortran_declarations.mod"
                "ops_fortran_rt_support.mod"
            )

            if(HDF5_SEQ)
                list(APPEND SOURCE_MOD_FILES
                    "ops_fortran_hdf5_declarations.mod")
            endif()

            foreach(SOURCE_MOD_FILE ${SOURCE_MOD_FILES})
                set(SOURCE_FILE "${SOURCE_DIR}/${SOURCE_MOD_FILE}")
                set(DEST_FILE "${DESTINATION_DIR}/${SOURCE_MOD_FILE}")

                add_custom_command(
                    OUTPUT ${DEST_FILE}
                    COMMAND ${CMAKE_COMMAND} -E copy_if_different ${SOURCE_FILE} ${DEST_FILE}
                    DEPENDS ${SOURCE_FILE}
                    COMMENT "Copying ${SOURCE_FILE} to ${DEST_FILE}"
                    VERBATIM
                )

                list(APPEND COPY_OUTPUTS ${DEST_FILE})
            endforeach()

            add_custom_target(${Name}_copy_files
                DEPENDS ${COPY_OUTPUTS}
                COMMENT "Ensuring ${DESTINATION_DIR} is up-to-date"
            )

            set(CMAKE_Fortran_MODULE_DIRECTORY ${OPS_CMAKE_INSTALL_PATH}/mod/${Name})
            add_library(${Name}_constants OBJECT ${CONSTANTS_MODULE})
            add_dependencies(${Name}_constants ${Name}_copy_files)

#           TARGET: SEQ
            if(OTHERS)
                add_library(${Name}_others OBJECT ${OTHERS})
                add_dependencies(${Name}_others ${Name}_constants)
            endif()

            add_library(${Name}_seqtemplates OBJECT ${SEQ_TEMPLATES})
            add_dependencies(${Name}_seqtemplates ${Name}_constants)

            add_library(${Name}_seqopsgen OBJECT ${OPS})
            add_dependencies(${Name}_seqopsgen ${Name}_constants)
            add_dependencies(${Name}_seqopsgen ${Name}_seqtemplates)

            if(OTHERS)
                add_executable(${Name}_seq $<TARGET_OBJECTS:${Name}_others> $<TARGET_OBJECTS:${Name}_seqopsgen> $<TARGET_OBJECTS:${Name}_seqtemplates> $<TARGET_OBJECTS:${Name}_constants>)
            else()
                add_executable(${Name}_seq $<TARGET_OBJECTS:${Name}_seqopsgen> $<TARGET_OBJECTS:${Name}_seqtemplates> $<TARGET_OBJECTS:${Name}_constants>)
            endif()

            set_target_properties(${Name}_seq PROPERTIES LINKER_LANGUAGE Fortran)

            target_include_directories(${Name}_seq PRIVATE ${TMP_SOURCE_DIR})
            if(HDF5_SEQ)
                target_link_libraries(${Name}_seq PRIVATE ops_for_hdf5_seq hdf5::hdf5 hdf5::hdf5_hl)
            endif()
            target_link_libraries(${Name}_seq PRIVATE ops_for_seq)
            if(${CMAKE_Fortran_COMPILER_ID} STREQUAL NVHPC)
                target_link_libraries(${Name}_seq PRIVATE "-c++libs -lstdc++")
            endif()
            install(TARGETS ${Name}_seq     DESTINATION ${APP_INSTALL_DIR}/${Name})


#           TARGET: OpenMP
            if(OTHERS)
                add_executable(${Name}_openmp $<TARGET_OBJECTS:${Name}_others> $<TARGET_OBJECTS:${Name}_seqopsgen> $<TARGET_OBJECTS:${Name}_seqtemplates> $<TARGET_OBJECTS:${Name}_constants>)
            else()
                add_executable(${Name}_openmp $<TARGET_OBJECTS:${Name}_seqopsgen> $<TARGET_OBJECTS:${Name}_seqtemplates> $<TARGET_OBJECTS:${Name}_constants>)
            endif()

            set_target_properties(${Name}_openmp PROPERTIES LINKER_LANGUAGE Fortran)

            target_include_directories(${Name}_openmp PRIVATE ${TMP_SOURCE_DIR})
            if(HDF5_SEQ)
                target_link_libraries(${Name}_openmp PRIVATE ops_for_hdf5_seq hdf5::hdf5 hdf5::hdf5_hl)
            endif()
            target_link_libraries(${Name}_openmp PRIVATE ops_for_seq)
            if(${CMAKE_Fortran_COMPILER_ID} STREQUAL NVHPC)
                target_link_libraries(${Name}_openmp PRIVATE "-c++libs -lstdc++")
            endif()
            install(TARGETS ${Name}_openmp     DESTINATION ${APP_INSTALL_DIR}/${Name})


#           TARGET: tiled
            add_library(${Name}_tiledtemplates OBJECT ${SEQ_TEMPLATES})
            add_dependencies(${Name}_tiledtemplates ${Name}_constants)
            target_compile_definitions(${Name}_tiledtemplates PRIVATE "-DOPS_LAZY")

            add_library(${Name}_tiledopsgen OBJECT ${OPS})
            add_dependencies(${Name}_tiledopsgen ${Name}_constants)
            add_dependencies(${Name}_tiledopsgen ${Name}_tiledtemplates)
            target_compile_definitions(${Name}_tiledopsgen PRIVATE "-DOPS_LAZY")

            if(OTHERS)
                add_executable(${Name}_tiled $<TARGET_OBJECTS:${Name}_others> $<TARGET_OBJECTS:${Name}_tiledopsgen> $<TARGET_OBJECTS:${Name}_tiledtemplates> $<TARGET_OBJECTS:${Name}_constants>)
            else()
                add_executable(${Name}_tiled $<TARGET_OBJECTS:${Name}_tiledopsgen> $<TARGET_OBJECTS:${Name}_tiledtemplates> $<TARGET_OBJECTS:${Name}_constants>)
            endif()

            set_target_properties(${Name}_tiled PROPERTIES LINKER_LANGUAGE Fortran)

            target_include_directories(${Name}_tiled PRIVATE ${TMP_SOURCE_DIR})
            target_compile_definitions(${Name}_tiled PRIVATE "-DOPS_LAZY")
            if(HDF5_SEQ)
                target_link_libraries(${Name}_tiled PRIVATE ops_for_hdf5_seq hdf5::hdf5 hdf5::hdf5_hl)
            endif()
            target_link_libraries(${Name}_tiled PRIVATE ops_for_seq)
            if(${CMAKE_Fortran_COMPILER_ID} STREQUAL NVHPC)
                target_link_libraries(${Name}_tiled PRIVATE "-c++libs -lstdc++")
            endif()
            install(TARGETS ${Name}_tiled     DESTINATION ${APP_INSTALL_DIR}/${Name})

            if((OPS_TEST) AND (${GenerateTest} STREQUAL "YES"))
                add_cmake_test(${CMAKE_CURRENT_BINARY_DIR}/${Name}_seq ${CMAKE_CURRENT_SOURCE_DIR}/cmake_test.sh seq)
                add_cmake_test(${CMAKE_CURRENT_BINARY_DIR}/${Name}_openmp ${CMAKE_CURRENT_SOURCE_DIR}/cmake_test.sh openmp)
                add_cmake_test(${CMAKE_CURRENT_BINARY_DIR}/${Name}_tiled ${CMAKE_CURRENT_SOURCE_DIR}/cmake_test.sh tiled)
            endif()

            if(MPI)
#               TARGET: MPI
                add_library(${Name}_mpitemplates OBJECT ${SEQ_TEMPLATES})
                add_dependencies(${Name}_mpitemplates ${Name}_constants)
                target_compile_definitions(${Name}_mpitemplates PRIVATE "-DOPS_MPI")

                add_library(${Name}_mpiopsgen OBJECT ${OPS})
                add_dependencies(${Name}_mpiopsgen ${Name}_constants)
                add_dependencies(${Name}_mpiopsgen ${Name}_mpitemplates)
                target_compile_definitions(${Name}_mpiopsgen PRIVATE "-DOPS_MPI")

                if(OTHERS)
                    add_executable(${Name}_mpi $<TARGET_OBJECTS:${Name}_others> $<TARGET_OBJECTS:${Name}_mpiopsgen> $<TARGET_OBJECTS:${Name}_mpitemplates> $<TARGET_OBJECTS:${Name}_constants>)
                else()
                    add_executable(${Name}_mpi $<TARGET_OBJECTS:${Name}_mpiopsgen> $<TARGET_OBJECTS:${Name}_mpitemplates> $<TARGET_OBJECTS:${Name}_constants>)
                endif()

                set_target_properties(${Name}_mpi PROPERTIES LINKER_LANGUAGE Fortran)

                target_include_directories(${Name}_mpi PRIVATE ${TMP_SOURCE_DIR})
                target_compile_definitions(${Name}_mpi PRIVATE "-DOPS_MPI")
                if(HDF5_MPI)
                    target_link_libraries(${Name}_mpi PRIVATE ops_for_hdf5_mpi hdf5::hdf5 hdf5::hdf5_hl
                                                    MPI::MPI_Fortran)
                endif()
                target_link_libraries(${Name}_mpi PRIVATE ops_for_mpi MPI::MPI_Fortran "${MPI_LINK}")
                if(${CMAKE_Fortran_COMPILER_ID} STREQUAL NVHPC)
                    target_link_libraries(${Name}_mpi PRIVATE "-c++libs -lstdc++")
                endif()
                install(TARGETS ${Name}_mpi     DESTINATION ${APP_INSTALL_DIR}/${Name})


#               TARGET: MPI_OpenMP
                if(OTHERS)
                    add_executable(${Name}_mpi_openmp $<TARGET_OBJECTS:${Name}_others> $<TARGET_OBJECTS:${Name}_mpiopsgen> $<TARGET_OBJECTS:${Name}_mpitemplates> $<TARGET_OBJECTS:${Name}_constants>)
                else()
                    add_executable(${Name}_mpi_openmp $<TARGET_OBJECTS:${Name}_mpiopsgen> $<TARGET_OBJECTS:${Name}_mpitemplates> $<TARGET_OBJECTS:${Name}_constants>)
                endif()

                set_target_properties(${Name}_mpi_openmp PROPERTIES LINKER_LANGUAGE Fortran)

                target_include_directories(${Name}_mpi_openmp PRIVATE ${TMP_SOURCE_DIR})
                target_compile_definitions(${Name}_mpi_openmp PRIVATE "-DOPS_MPI")
                if(HDF5_MPI)
                    target_link_libraries(${Name}_mpi_openmp PRIVATE ops_for_hdf5_mpi hdf5::hdf5 hdf5::hdf5_hl
                                                    MPI::MPI_Fortran)
                endif()
                target_link_libraries(${Name}_mpi_openmp PRIVATE ops_for_mpi MPI::MPI_Fortran "${MPI_LINK}")
                if(${CMAKE_Fortran_COMPILER_ID} STREQUAL NVHPC)
                    target_link_libraries(${Name}_mpi_openmp PRIVATE "-c++libs -lstdc++")
                endif()
                install(TARGETS ${Name}_mpi_openmp     DESTINATION ${APP_INSTALL_DIR}/${Name})


#               TARGET: MPI_tiled
                add_library(${Name}_mpitiledtemplates OBJECT ${SEQ_TEMPLATES})
                add_dependencies(${Name}_mpitiledtemplates ${Name}_constants)
                target_compile_definitions(${Name}_mpitiledtemplates PRIVATE "-DOPS_MPI -DOPS_LAZY")

                add_library(${Name}_mpitiledopsgen OBJECT ${OPS})
                add_dependencies(${Name}_mpitiledopsgen ${Name}_constants)
                add_dependencies(${Name}_mpitiledopsgen ${Name}_mpitiledtemplates)
                target_compile_definitions(${Name}_mpitiledopsgen PRIVATE "-DOPS_MPI -DOPS_LAZY")

                if(OTHERS)
                    add_executable(${Name}_mpi_tiled $<TARGET_OBJECTS:${Name}_others> $<TARGET_OBJECTS:${Name}_mpitiledopsgen> $<TARGET_OBJECTS:${Name}_mpitiledtemplates> $<TARGET_OBJECTS:${Name}_constants>)
                else()
                    add_executable(${Name}_mpi_tiled $<TARGET_OBJECTS:${Name}_mpitiledopsgen> $<TARGET_OBJECTS:${Name}_mpitiledtemplates> $<TARGET_OBJECTS:${Name}_constants>)
                endif()

                set_target_properties(${Name}_mpi_tiled PROPERTIES LINKER_LANGUAGE Fortran)

                target_include_directories(${Name}_mpi_tiled PRIVATE ${TMP_SOURCE_DIR})
                target_compile_definitions(${Name}_mpi_tiled PRIVATE "-DOPS_MPI -DOPS_LAZY")
                if(HDF5_MPI)
                    target_link_libraries(${Name}_mpi_tiled PRIVATE ops_for_hdf5_mpi hdf5::hdf5 hdf5::hdf5_hl
                                                    MPI::MPI_Fortran)
                endif()
                target_link_libraries(${Name}_mpi_tiled PRIVATE ops_for_mpi MPI::MPI_Fortran "${MPI_LINK}")
                if(${CMAKE_Fortran_COMPILER_ID} STREQUAL NVHPC)
                    target_link_libraries(${Name}_mpi_tiled PRIVATE "-c++libs -lstdc++")
                endif()
                install(TARGETS ${Name}_mpi_tiled     DESTINATION ${APP_INSTALL_DIR}/${Name})


                if((OPS_TEST) AND (${GenerateTest} STREQUAL "YES"))
                    add_cmake_test(${CMAKE_CURRENT_BINARY_DIR}/${Name}_mpi ${CMAKE_CURRENT_SOURCE_DIR}/cmake_test.sh mpi)
                    add_cmake_test(${CMAKE_CURRENT_BINARY_DIR}/${Name}_mpi_openmp ${CMAKE_CURRENT_SOURCE_DIR}/cmake_test.sh mpi_openmp)
                    add_cmake_test(${CMAKE_CURRENT_BINARY_DIR}/${Name}_mpi_tiled ${CMAKE_CURRENT_SOURCE_DIR}/cmake_test.sh mpi_tiled)
                endif()

            endif()

#           TARGET: CUDA
            if(CUDAToolkit_FOUND)
                if(${CMAKE_Fortran_COMPILER_ID} STREQUAL NVHPC)

                    set(SOURCE_DIR "${OPS_CMAKE_INSTALL_PATH}/mod")
                    set(DESTINATION_DIR "${OPS_CMAKE_INSTALL_PATH}/mod/${Name}/cuda")

                    set(SOURCE_MOD_FILES
                        "ops_fortran_declarations.mod"
                        "ops_fortran_rt_support.mod"
                    )

                    if(HDF5_SEQ)
                        list(APPEND SOURCE_MOD_FILES
                            "ops_fortran_hdf5_declarations.mod")
                    endif()

                    foreach(SOURCE_MOD_FILE ${SOURCE_MOD_FILES})
                        set(SOURCE_FILE "${SOURCE_DIR}/${SOURCE_MOD_FILE}")
                        set(DEST_FILE "${DESTINATION_DIR}/${SOURCE_MOD_FILE}")

                        add_custom_command(
                            OUTPUT ${DEST_FILE}
                            COMMAND ${CMAKE_COMMAND} -E copy_if_different ${SOURCE_FILE} ${DEST_FILE}
                            DEPENDS ${SOURCE_FILE}
                            COMMENT "Copying ${SOURCE_FILE} to ${DEST_FILE}"
                            VERBATIM
                        )

                        list(APPEND COPY_OUTPUTS ${DEST_FILE})
                    endforeach()

                    add_custom_target(${Name}_copy_files_cuda
                        DEPENDS ${COPY_OUTPUTS}
                        COMMENT "Ensuring ${DESTINATION_DIR} is up-to-date"
                    )

                    set(CMAKE_Fortran_MODULE_DIRECTORY ${OPS_CMAKE_INSTALL_PATH}/mod/${Name}/cuda)

                    set_source_files_properties(${CONSTANTS_MODULE} PROPERTIES LANGUAGE Fortran)
                    add_library(${Name}_constants_cuda OBJECT ${CONSTANTS_MODULE})
                    add_dependencies(${Name}_constants_cuda ${Name}_copy_files_cuda)
                    target_compile_definitions(${Name}_constants_cuda PRIVATE $<$<COMPILE_LANGUAGE:Fortran>:OPS_WITH_CUDAFOR>)
                    target_compile_options(${Name}_constants_cuda PRIVATE $<$<COMPILE_LANGUAGE:Fortran>:-cuda -gpu=cc${CMAKE_CUDA_ARCHITECTURES}>)

                    file(GLOB_RECURSE CUDA_TEMPLATES "${TMP_SOURCE_DIR}/cuda/*_cuda_kernel.CUF")

                    foreach(CUDA_FILE ${CUDA_TEMPLATES})
                        set_source_files_properties(${CUDA_FILE} PROPERTIES LANGUAGE Fortran)
                    endforeach()

                    add_library(${Name}_cudatemplates OBJECT ${CUDA_TEMPLATES})
                    add_dependencies(${Name}_cudatemplates ${Name}_constants_cuda)
                    target_compile_definitions(${Name}_cudatemplates PRIVATE $<$<COMPILE_LANGUAGE:Fortran>:OPS_WITH_CUDAFOR>)
                    target_compile_options(${Name}_cudatemplates PRIVATE $<$<COMPILE_LANGUAGE:Fortran>:-cuda -gpu=cc${CMAKE_CUDA_ARCHITECTURES}>)

                    foreach(OPS_FILE ${OPS})
                        set_source_files_properties(${OPS_FILE} PROPERTIES LANGUAGE Fortran)
                    endforeach()
                    add_library(${Name}_cudaopsgen OBJECT ${OPS})
                    add_dependencies(${Name}_cudaopsgen ${Name}_constants_cuda)
                    add_dependencies(${Name}_cudaopsgen ${Name}_cudatemplates)
                    target_compile_definitions(${Name}_cudaopsgen PRIVATE $<$<COMPILE_LANGUAGE:Fortran>:OPS_WITH_CUDAFOR>)
                    target_compile_options(${Name}_cudaopsgen PRIVATE $<$<COMPILE_LANGUAGE:Fortran>:-cuda -gpu=cc${CMAKE_CUDA_ARCHITECTURES}>)

                    if(OTHERS)
                        add_executable(${Name}_cuda $<TARGET_OBJECTS:${Name}_others> $<TARGET_OBJECTS:${Name}_cudaopsgen> $<TARGET_OBJECTS:${Name}_cudatemplates> $<TARGET_OBJECTS:${Name}_constants_cuda>)
                    else()
                        add_executable(${Name}_cuda $<TARGET_OBJECTS:${Name}_cudaopsgen> $<TARGET_OBJECTS:${Name}_cudatemplates> $<TARGET_OBJECTS:${Name}_constants_cuda>)
                    endif()

                    set_target_properties(${Name}_cuda PROPERTIES LINKER_LANGUAGE Fortran)

                    target_link_options(${Name}_cuda PRIVATE "-DOPS_WITH_CUDAFOR" "-cuda" "-gpu=cc${CMAKE_CUDA_ARCHITECTURES}")

                    if(HDF5_SEQ)
                        target_link_libraries(${Name}_cuda PRIVATE ops_for_hdf5_seq hdf5::hdf5 hdf5::hdf5_hl)
                    endif()
                    target_link_libraries(${Name}_cuda PRIVATE ops_for_cuda "-cudalib=curand -c++libs -lstdc++ -lcudart")
                    install(TARGETS ${Name}_cuda     DESTINATION ${APP_INSTALL_DIR}/${Name})

                    if((OPS_TEST)
                        AND (GPU_NUMBER GREATER_EQUAL 1)
                        AND (${GenerateTest} STREQUAL "YES"))
                        add_cmake_test(${CMAKE_CURRENT_BINARY_DIR}/${Name}_cuda ${CMAKE_CURRENT_SOURCE_DIR}/cmake_test.sh cuda)
                    endif()

                endif()
            endif()

        endmacro()

#       Build all listed apps

#       Current Targets: seq, openmp, tiled, mpi, mpi_openmp, mpi_tiled

#       App: Poisson
        add_subdirectory(poisson)

#       App: Laplace2d
        add_subdirectory(laplace2dtutorial/step7)

    endif(CMAKE_Fortran_COMPILER)
