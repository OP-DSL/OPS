cmake_minimum_required(VERSION 3.18)
CreateTempDir()
BUILD_OPS_C_SAMPLE(write "read" "NONE" "NONE" "NO" "NO")
BUILD_OPS_C_SAMPLE(read "write" "NONE" "NONE" "NO" "NO")
if (OPS_TEST AND  MPI AND HDF5_FOUND)
    enable_testing()
    set(H5D "${HDF5_DIFF_EXECUTABLE}")
    set(READ "$<TARGET_FILE:read_cuda>")
    set(WRITE "$<TARGET_FILE:write_cuda>")
    set(WRITE_SEQ "$<TARGET_FILE:write_seq>")
    set(W_ARG " ")
    set(R_ARG " ")
    add_test(NAME multiDim_HDF5_cuda
        COMMAND ${CMAKE_COMMAND} -DREAD=${READ} -DWRITE_SEQ=${WRITE_SEQ} -DWRITE=${WRITE} -DR_ARG=${R_ARG} -DW_ARG=${W_ARG} -DH5D=${H5D} -DOPS_INSTALL_PATH=${OPS_INSTALL_PATH}  -P ${CMAKE_CURRENT_SOURCE_DIR}/runtests.cmake
        WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/tmp"
        )
    set(READ "$<TARGET_FILE:read_opencl>")
    set(WRITE "$<TARGET_FILE:write_opencl>")
    set(W_ARG "OPS_CL_DEVICE=1 OPS_BLOCK_SIZE_X=32 OPS_BLOCK_SIZE_Y=4")
    set(R_ARG "OPS_CL_DEVICE=1 OPS_BLOCK_SIZE_X=32 OPS_BLOCK_SIZE_Y=4")
    add_test(NAME multiDim_HDF5_opencl
        COMMAND ${CMAKE_COMMAND} -DREAD=${READ}  -DWRITE=${WRITE} -DR_ARG=${R_ARG} -DW_ARG=${W_ARG} -DH5D=${H5D} -DOPS_INSTALL_PATH=${OPS_INSTALL_PATH}  -P ${CMAKE_CURRENT_SOURCE_DIR}/runtests.cmake
        WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/tmp"
        )
    set(READ "mpirun")
    set(WRITE "mpirun")
    set(W_ARG "-n ${CPU_NUMBER} $<TARGET_FILE:write_mpi>")
    set(R_ARG "-n ${CPU_NUMBER} $<TARGET_FILE:read_mpi>")
    add_test(NAME multiDim_HDF5_mpi
        COMMAND ${CMAKE_COMMAND} -DREAD=${READ}  -DWRITE=${WRITE} -DR_ARG=${R_ARG} -DW_ARG=${W_ARG} -DH5D=${H5D} -DOPS_INSTALL_PATH=${OPS_INSTALL_PATH}  -P ${CMAKE_CURRENT_SOURCE_DIR}/runtests.cmake
        WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/tmp"
        )
endif()
