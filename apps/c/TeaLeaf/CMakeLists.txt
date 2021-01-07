cmake_minimum_required(VERSION 3.18)
CreateTempDir()
set(others build_field.cpp diffuse.cpp initialise.cpp read_input.cpp tea_solve.cpp timestep.cpp)
BUILD_OPS_C_SAMPLE(tea_leaf "NONE" "${others}" "NONE" "NO" "YES")
