set(case "CloverLeaf")
set(CASE_DIR "${app_dir_c}/${case}")
createtempdir()
#Set additional sources
set(others start.cpp 
	   time_step.cpp 
	   build_field.cpp 
	   read_input.cpp
           initialise.cpp 
	   advection.cpp)
# Off files to avoid: generally they are idetified with a small string
set(odd "profile")
build_ops_c_sample(${case} ${odd} "${others}" "NONE" "NO" "YES")
