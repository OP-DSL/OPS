cmake_minimum_required(VERSION 3.18)
createtempdir()
set(others build_field.cpp diffuse.cpp initialise.cpp read_input.cpp
           tea_solve.cpp timestep.cpp)
build_ops_c_sample(tea_leaf "NONE" "${others}" "NONE" "NO" "YES")
