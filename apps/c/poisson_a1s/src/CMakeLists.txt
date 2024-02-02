cmake_minimum_required(VERSION 3.8)

OPS_C_TARGET(PoissonAD poisson.cpp "${CMAKE_CURRENT_BINARY_DIR}/ops")

if(BUILD_FOR_CPU)
add_executable(poisson_a1s_omp
    ${OPS_PoissonAD_BASEOUTPUT}
    ${OPS_PoissonAD_MPI_adjoint_KERNELS}
    )

target_include_directories(poisson_a1s_omp
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${OPS_C_INCLUDE_DIR}
    )

target_link_libraries(poisson_a1s_omp
    ops_seq
    )

endif()

if(BUILD_FOR_GPU)
add_executable(poisson_a1s_cuda
    ${OPS_PoissonAD_BASEOUTPUT}
    ${OPS_PoissonAD_CUDA_adjoint_KERNELS}
    )

target_include_directories(poisson_a1s_cuda
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${OPS_C_INCLUDE_DIR}
    )

target_link_libraries(poisson_a1s_cuda
    ops_cuda
    )

endif()
