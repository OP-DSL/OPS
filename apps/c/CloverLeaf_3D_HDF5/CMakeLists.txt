cmake_minimum_required(VERSION 3.18)
CreateTempDir()
if (HDF5_SEQ)
    add_executable(generate_file generate_file.cpp build_field_hdf5.cpp read_input.cpp initialise_chunk.cpp
                    initialise.cpp generate_hdf5.cpp)
    target_include_directories(generate_file PRIVATE "${CMAKE_CURRENT_BINARY_DIR}/tmp")
    target_link_libraries(generate_file ops_seq ops_hdf5_seq ${HDF5_LIBRARIES}  ${HDF5_HL_LIBRARIES} MPI::MPI_CXX)
endif()
if (HDF5_MPI)
    add_executable(generate_file_mpi generate_file.cpp build_field_hdf5.cpp read_input.cpp initialise_chunk.cpp
                    initialise.cpp generate_hdf5.cpp)
    target_include_directories(generate_file_mpi PRIVATE "${CMAKE_CURRENT_BINARY_DIR}/tmp/")
    target_link_libraries(generate_file_mpi ops_mpi ops_hdf5_mpi ${HDF5_LIBRARIES}  ${HDF5_HL_LIBRARIES} MPI::MPI_CXX)
endif()
if (OPS_TEST AND  MPI AND HDF5_FOUND)
    enable_testing()
    set(H5D "${HDF5_DIFF_EXECUTABLE}")
    set(Gen_SEQ "$<TARGET_FILE:generate_file>")
    set(Gen_MPI "$<TARGET_FILE:generate_file_mpi>")
    add_test(NAME Generate_SEQ_MPI
        COMMAND ${CMAKE_COMMAND} -DSEQ=${Gen_SEQ}  -DMPI=${Gen_MPI} -DNUM=${CPU_NUMBER} -DH5D=${H5D}
        -P ${CMAKE_CURRENT_SOURCE_DIR}/runtests.cmake
        WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/tmp"
        )
endif()
set(others start.cpp time_step.cpp build_field.cpp read_input.cpp initialise.cpp advection.cpp)
set(extra generate_file.cpp generate_hdf5.cpp build_field_hdf5.cpp)
BUILD_OPS_C_SAMPLE(clover_leaf_3D_hdf5 profile "${others}" "${extra}" "YES")

