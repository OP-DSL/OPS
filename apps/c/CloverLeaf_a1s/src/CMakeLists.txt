cmake_minimum_required(VERSION 3.8)

list(APPEND OTHER_FILES start.cpp time_step.cpp build_field.cpp read_input.cpp initialise.cpp advection.cpp)
list(APPEND SRC_FILES clover_leaf.cpp revert.cpp reset_field.cpp ideal_gas.cpp PdV.cpp advec_cell.cpp accelerate.cpp advec_mom.cpp calc_dt.cpp field_summary.cpp flux_calc.cpp viscosity.cpp initialise_chunk.cpp generate.cpp update_halo.cpp)

OPS_C_TARGET(CloverLeaf_AD SRC_FILES "${CMAKE_CURRENT_BINARY_DIR}/ops")


if(BUILD_FOR_CPU)
add_executable(cloverleaf_a1s_omp
    ${OPS_CloverLeaf_AD_BASEOUTPUT}
    ${OPS_CloverLeaf_AD_MPI_adjoint_KERNELS}
    ${OTHER_FILES}
    )

target_include_directories(cloverleaf_a1s_omp
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${OPS_C_INCLUDE_DIR}
    )

target_link_libraries(cloverleaf_a1s_omp
    ops_seq
    )

endif()

if(BUILD_FOR_GPU)
add_executable(cloverleaf_a1s_cuda
    ${OPS_CloverLeaf_AD_BASEOUTPUT}
    ${OPS_CloverLeaf_AD_CUDA_adjoint_KERNELS}
    ${OTHER_FILES}
    )

target_include_directories(cloverleaf_a1s_cuda
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${OPS_C_INCLUDE_DIR}
    )

target_link_libraries(cloverleaf_a1s_cuda
    ops_cuda
    )

endif()
