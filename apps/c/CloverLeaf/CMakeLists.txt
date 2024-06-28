cmake_minimum_required(VERSION 3.18)
set(others start.cpp time_step.cpp build_field.cpp read_input.cpp
           initialise.cpp advection.cpp)
createtempdir()
build_ops_c_sample(clover_leaf profile "${others}" "NONE" "NO" "YES")
