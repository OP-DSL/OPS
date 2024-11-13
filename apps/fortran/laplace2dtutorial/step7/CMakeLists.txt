cmake_minimum_required(VERSION 3.18)
set(others constants.F90)
createtempdir()
build_ops_c_sample(laplace2d_ftn "NONE" "${others}" "NONE" "NO" "YES")
