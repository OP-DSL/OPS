set(case "mblock")
set(CASE_DIR "${app_dir_c}/${case}")
createtempdir()
build_ops_c_sample(${case} "NONE" "NONE" "NONE" "NO" "${TestDev}" "2")
