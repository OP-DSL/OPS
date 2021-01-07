cmake_minimum_required(VERSION 3.18)
set(others start.cpp time_step.cpp build_field.cpp read_input.cpp initialise.cpp advection.cpp)
CreateTempDir()
BUILD_OPS_C_SAMPLE(clover_leaf_3D profile "${others}" "NONE" "NO" "YES")
