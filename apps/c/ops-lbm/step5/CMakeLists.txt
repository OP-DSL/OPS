set(case "lattboltz2d")
set(CASE_DIR "${app_dir_c}/${case}")
createtempdir()
build_ops_c_sample(${case} "NONE" "NONE" "NONE" "NO" "${OPS_TEST_DEV}" "2")
