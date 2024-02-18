#!/bin/bash

#current_datetime=$(date +"%Y%m%d_%H%M%S")
current_datetime=$(date +"%d_%m_%Y")
log_file="$PWD/logfile_${current_datetime}.txt"
cpp_build_log="$PWD/build_log_CPP_${current_datetime}.txt"
f90_build_log="$PWD/build_log_F90_${current_datetime}.txt"


function check_env ( ) {
    if [[ -n "$OPS_COMPILER" ]]; then
        echo "OPS Compiler : "$OPS_COMPILER > $log_file
    else
        echo "Variable OPS_COMPILER is not set." > $log_file
        echo "Exiting now....." >> $log_file
        exit
    fi

    if [[ -n "$OPS_INSTALL_PATH" ]]; then
        echo "OPS Install Path: "$OPS_INSTALL_PATH >> $log_file
        echo "" >> $log_file
        is_path_exist "$OPS_INSTALL_PATH"
    else
        echo "Variable OPS_INSTALL_PATH is not set." >> $log_file
        echo "Exiting now....." >> $log_file
        exit
    fi

    if [[ -n "$CUDA_INSTALL_PATH" ]]; then
        echo "CUDA Install Path: "$CUDA_INSTALL_PATH >> $log_file
        echo "" >> $log_file
        is_path_exist "$CUDA_INSTALL_PATH"
    else
        echo "CUDA_INSTALL_PATH is not set" >> $log_file
        echo "Warning! CUDA targets will not be built" >> $log_file
        echo "" >> $log_file
    fi

    if [[ -n "$HIP_INSTALL_PATH" ]]; then
        echo "HIP Install Path: "$HIP_INSTALL_PATH >> $log_file
        echo "" >> $log_file
        is_path_exist "$HIP_INSTALL_PATH"
    else
        echo "HIP_INSTALL_PATH is not set" >> $log_file
        echo "Warning! HIP targets will not be built" >> $log_file
        echo "" >> $log_file
    fi

    if [[ -n "$SYCL_INSTALL_PATH" ]]; then
        echo "SYCL Install Path: "$SYCL_INSTALL_PATH >> $log_file
        echo "" >> $log_file
        is_path_exist "$SYCL_INSTALL_PATH"
    else
        echo "SYCL_INSTALL_PATH is not set" >> $log_file
        echo "Warning! SYCL targets will not be built" >> $log_file
        echo "" >> $log_file
    fi

    if [[ -n "$HDF5_INSTALL_PATH" ]]; then
        echo "HDF5 Install Path: "$HDF5_INSTALL_PATH >> $log_file
        echo "" >> $log_file
        is_path_exist "$HDF5_INSTALL_PATH"
    else
        echo "HDF5_INSTALL_PATH is not set" >> $log_file
        echo "Warning! HDF5 targets will not be built" >> $log_file
        echo "" >> $log_file
    fi

    if [[ -n "$TRID_INSTALL_PATH" ]]; then
        echo "TRID Install Path: "$TRID_INSTALL_PATH >> $log_file
        echo "" >> $log_file
        is_path_exist "$TRID_INSTALL_PATH"
    else
        echo "TRID_INSTALL_PATH is not set" >> $log_file
        echo "Warning! TRID targets will not be built" >> $log_file
        echo "" >> $log_file
    fi

}

function is_path_exist ( ) {

    local dir_path="$1"

   if [ ! -e "$dir_path" ]; then
        echo "" >> $log_file
        echo "Error! Path does not exist: "$dir_path >> $log_file
        echo "Exiting now....." >> $log_file
        echo "" >> $log_file
        exit
    fi
}

function is_lib_built ( ) { 
    local lib_name="$1"
    local lib_path="$2"
    
    if [ -e "$lib_path" ]; then
        echo "LIB build PASSED: "$lib_name >> $log_file
    else
        echo "" >> $log_file
        echo "Error! LIB build FAILED: "$lib_name >> $log_file
        echo "Error! Library doesn't exist: "$lib_path >> $log_file
        echo "Please check build logs for errors." >> $log_file 
        echo "Exiting now....." >> $log_file
        echo "" >> $log_file
        exit
    fi
}

function build_cpp_lib ( ) {
    cd $OPS_INSTALL_PATH/c
    make clean
    make IEEE=1 >> $cpp_build_log
    
    is_lib_built "CPP_SEQ" "$OPS_INSTALL_PATH/c/lib/$OPS_COMPILER/libops_seq.a"
    is_lib_built "CPP_MPI" "$OPS_INSTALL_PATH/c/lib/$OPS_COMPILER/libops_mpi.a"

    if [ -n "$HDF5_INSTALL_PATH" ]; then
        is_lib_built "CPP_HDF5_SEQ" "$OPS_INSTALL_PATH/c/lib/$OPS_COMPILER/libops_hdf5_seq.a"
        is_lib_built "CPP_HDF5_MPI" "$OPS_INSTALL_PATH/c/lib/$OPS_COMPILER/libops_hdf5_mpi.a"
    fi

    if [ -n "$CUDA_INSTALL_PATH" ]; then
        is_lib_built "CPP_CUDA_SEQ" "$OPS_INSTALL_PATH/c/lib/$OPS_COMPILER/libops_cuda.a"
        is_lib_built "CPP_CUDA_MPI" "$OPS_INSTALL_PATH/c/lib/$OPS_COMPILER/libops_mpi_cuda.a"
    fi

    if [ -n "$HIP_INSTALL_PATH" ]; then
        is_lib_built "CPP_HIP_SEQ" "$OPS_INSTALL_PATH/c/lib/$OPS_COMPILER/libops_hip.a"
        is_lib_built "CPP_HIP_MPI" "$OPS_INSTALL_PATH/c/lib/$OPS_COMPILER/libops_mpi_hip.a"
    fi

    if [ -n "$SYCL_INSTALL_PATH" ]; then
        is_lib_built "CPP_SYCL_SEQ" "$OPS_INSTALL_PATH/c/lib/$OPS_COMPILER/libops_sycl.a"
        is_lib_built "CPP_SYCL_MPI" "$OPS_INSTALL_PATH/c/lib/$OPS_COMPILER/libops_mpi_sycl.a"
    fi
    
    if [ -n "$TRID_INSTALL_PATH" ]; then
        is_lib_built "CPP_TRID_SEQ" "$OPS_INSTALL_PATH/c/lib/$OPS_COMPILER/libops_trid_seq.a"
        is_lib_built "CPP_TRID_MPI" "$OPS_INSTALL_PATH/c/lib/$OPS_COMPILER/libops_trid_mpi.a"
        if [ -n "$CUDA_INSTALL_PATH" ]; then
            is_lib_built "CPP_TRID_CUDA_SEQ" "$OPS_INSTALL_PATH/c/lib/$OPS_COMPILER/libops_trid_cuda.a"
            is_lib_built "CPP_TRID_CUDA_MPI" "$OPS_INSTALL_PATH/c/lib/$OPS_COMPILER/libops_trid_mpi_cuda.a"
        fi
    fi

    if [ "$OPS_COMPILER" != "intel" ] && [ "$OPS_COMPILER" != "gnu" ] && [ "$OPS_COMPILER" != "intel-sycl" ]; then
        is_lib_built "CPP_OMPOFFLOAD_SEQ" "$OPS_INSTALL_PATH/c/lib/$OPS_COMPILER/libops_ompoffload.a"
        is_lib_built "CPP_OMPOFFLOAD_MPI" "$OPS_INSTALL_PATH/c/lib/$OPS_COMPILER/libops_mpi_ompoffload.a"
    fi

    echo "OPS CPP library build Successful!!!" >> $log_file
    echo "" >> $log_file
}

function build_and_run_cpp_apps ( ) {

    target_names=("dev_seq" "dev_mpi" "seq" "tiled" "openmp" "mpi" "mpi_tiled" "mpi_openmp")
    if [ -n "$CUDA_INSTALL_PATH" ]; then
        target_names+=("cuda" "mpi_cuda" "mpi_cuda_tiled")
    fi
    if [ -n "$HIP_INSTALL_PATH" ]; then
        target_names+=("hip" "mpi_hip" "mpi_hip_tiled")
    fi
    if [ -n "$SYCL_INSTALL_PATH" ]; then
        target_names+=("sycl" "mpi_sycl" "mpi_sycl_tiled")
    fi
    if [ "$OPS_COMPILER" != "intel" ] && [ "$OPS_COMPILER" != "gnu" ] && [ "$OPS_COMPILER" != "intel-sycl" ]; then
        target_names+=("ompoffload" "mpi_ompoffload" "mpi_ompoffload_tiled")
    fi

    app_dir_list=("laplace2d_tutorial/step7" "poisson" "CloverLeaf" "CloverLeaf_3D" "CloverLeaf_3D_HDF5" "TeaLeaf" \
              "lowdim_test" "multiDim" "multiDim_HDF5" "mblock" "mgrid" "shsgc" "mb_shsgc/Max_datatransfer" \
              "hdf5_slice" "ops-lbm/step5")
    app_names_list=("laplace2d" "poisson" "cloverleaf" "cloverleaf" "cloverleaf" "tealeaf" \
                    "lowdim" "multidim" "multiDim_HDF5" "mblock" "mgrid" "shsgc" "shsgc" \
                    "hdf5_slice" "lattboltz2d")
    
    length=${#app_dir_list[@]}
    
    for ((i = 0; i < length; i++)); do
        app_name="${app_names_list[i]}"
        app_dir="${app_dir_list[i]}"
        test_cpp_app "$app_name" "$app_dir" "${target_names[@]}"
    done

    trid_app_dir_list=("adi" "adi_burger" "adi_burger_3D" "compact_scheme")
}

function is_app_built ( ) {
    local app_target_name="$1"
    local app_path="$2"

    if [ ! -e "$app_path/$app_target_name" ]; then
        echo "" >> $log_file
        echo "Error! APP build FAILED: "$app_target_name >> $log_file
        echo "Please check app build logs for errors." >> $log_file
#        echo "Exiting now....." >> $log_file
        echo "" >> $log_file
#        exit
        return 1
    fi
    return 0
}

function test_cpp_app ( ) {

    local app_name="$1"
    local app_dir_name="$2"
    shift;  shift
    local target_names=("$@")

    echo "Testing: $app_dir_name" >> $log_file

    local app_path=$OPS_INSTALL_PATH/../apps/c/$app_dir_name
    is_path_exist "$app_path"

    apps_with_hdf5=("CloverLeaf_3D_HDF5" "lowdim_test" "multiDim" "multiDim_HDF5" "mblock" "mgrid" "hdf5_slice")

    continue_test=true
    if [[ -z "$HDF5_INSTALL_PATH" ]]; then
        for item in "${apps_with_hdf5[@]}"; do
            if [[ "$item" = "$app_dir_name" ]]; then
                continue_test=false
                break
            fi
        done
    fi

    app_passed=()
    app_failed=()
    app_build_failed=()

#   Go to APP directory and make
    if [ $continue_test ]; then
#       Go to App directory
        cd $app_path

        if [[ "$app_dir_name" = "multiDim_HDF5" ]]; then
            make -f Makefile.write IEEE=1
            rm .generated
            make -f Makefile.read IEEE=1
        else
            make IEEE=1
            if [[ "$app_dir_name" = "CloverLeaf_3D_HDF5" ]]; then
                make generate_file generate_file_mpi
                ./generate_file
                mv cloverdata.h5 cloverdata_seq.h5
                mpirun -np 4 ./generate_file_mpi
                mv cloverdata.h5 cloverdata_mpi.h5

                $HDF5_INSTALL_PATH/bin/h5diff cloverdata_seq.h5 cloverdata_mpi.h5 > diff_out.log

                if [ -s ./diff_out.log ]; then
                    echo "Error! cloverdata.h5 file generated from sequential and MPI version doesn't match" >> $log_file
                else
                    echo "Success, cloverdata.h5 file generated from sequential and MPI version matches" >> $log_file
                    rm cloverdata_seq.h5 cloverdata_mpi.h5
                fi
                rm diff_out.log
                ./generate_file
            fi
        fi

#       Run all targets
        if [[ "$app_dir_name" = "multiDim_HDF5" ]]; then
            for target in "${target_names[@]}"; do
                app_name=write
                app_target_name=write_$target
                #   check if app exist, run it and report status
                is_app_built "$app_target_name" "$app_path"
                build_res=$?    
                if [[ $build_res != 0 ]]; then
                    app_build_failed+=("write_$target")
                fi
                run_cpp_target "$app_name" "$app_dir_name" "$target" "$app_target_name" "$app_path"
                result1=$?

                app_name=read
                app_target_name=read_$target
                #   check if app exist, run it and report status
                is_app_built "$app_target_name" "$app_path"
                build_res=$?
                if [[ $build_res != 0 ]]; then
                    app_build_failed+=("read_$target")
                fi
                run_cpp_target "$app_name" "$app_dir_name" "$target" "$app_target_name" "$app_path"
                result2=$?

                if [[ $((result1 + result2)) = 0 ]]; then
                    app_passed+=("$target")
                else
                    app_failed+=("$target")
                fi
            done        
        else
            for target in "${target_names[@]}"; do
                app_target_name=${app_name}_$target
                #   check if app exist, run it and report status
                is_app_built "$app_target_name" "$app_path"
                build_res=$?
                if [[ $build_res != 0 ]]; then
                    app_build_failed+=("$target")
                fi
                run_cpp_target "$app_name" "$app_dir_name" "$target" "$app_target_name" "$app_path"
                result=$?
                if [[ $result = 0 ]]; then
                    app_passed+=("$target")
                else
                    app_failed+=("$target")
                fi
            done
        fi

#       Report status to log file
        echo "=================================================================================================" >> $log_file
        echo "App Name: $app_dir_name" >> $log_file
        if [[ -n "$app_build_failed" ]]; then
            echo "Compilation FAILED targets:  ${app_build_failed[@]}" >> $log_file
        fi
        if [[ -n "$app_passed" ]]; then
            echo "PASSED targets:  ${app_passed[@]}" >> $log_file
        fi
        if [[ -n "$app_failed" ]]; then
            echo "FAILED targets:  ${app_failed[@]}" >> $log_file
        fi
        echo "=================================================================================================" >> $log_file
        if [[ "$app_dir_name" = "multiDim_HDF5" ]]; then
            make -f Makefile.read cleanall
            make -f Makefile.write cleanall
        else
            make cleanall
            if [[ "$app_dir_name" = "CloverLeaf_3D_HDF5" ]]; then
                rm cloverdata.h5 generate_file
            fi
            if [[ "$app_dir_name" = "mgrid" ]]; then
                rm data_ref.h5
            fi
            if [[ "$app_dir_name" = "hdf5_slice" ]]; then
                rm double_ref.h5 half_ref.h5 I1_ref.h5 I4_ref.h5 J16_ref.h5 J8_ref.h5 K15_ref.h5 
                rm K16_ref.h5 single_ref.h5 slab_ref.h5 slice3Du_ref.h5 slice3Dv_ref.h5
            fi
        fi
    else
        echo "HDF5_INSTALL_PATH is not set, cannot build and run the APP: $app_dir_name" >> $log_file
    fi
    echo "" >> $log_file
    echo "" >> $log_file
}

function run_cpp_target ( ) {
    local app_name="$1"
    local app_dir_name="$2"
    local target="$3"
    local app_target_name="$4"
    local app_path="$5"

    sleep 2

    echo "" >> $log_file

#   Set OpenMP num threads
    if [[ "$target" = *"openmp"* ]]; then
        export OMP_NUM_THREADS=2
    else
        export OMP_NUM_THREADS=1
    fi

    if [[ "$app_name" = "multidim" ]] || [[ "$app_name" = "cloverleaf" ]]; then
        export OMP_NUM_THREADS=1
    fi

#   ====================
#   IF MPI_Tiled version
#   ====================
    if [[ "$target" = *"mpi"* ]] && [[ "$target" = *"tiled"* ]]; then
        if [[ "$app_name" = "cloverleaf" ]]; then
            OPS_TILING_MAXDEPTH=6
        elif  [[ "$app_name" = "poisson" ]]; then
            OPS_TILING_MAXDEPTH=10
        fi
        if [[ "$target" = *"cuda"* ]] || [[ "$target" = *"hip"* ]] || [[ "$target" = *"sycl"* ]]; then
            NP=2
        else
            if [[ "$app_dir_name" = "CloverLeaf" ]] || [[ "$app_dir_name" = "CloverLeaf_3D" ]] || [[ "$app_dir_name" = "CloverLeaf_3D_HDF5" ]] || \
            [[ "$app_dir_name" = "mgrid" ]] || [[ "$app_dir_name" = "shsgc" ]] || [[ "$app_dir_name" = "hdf5_slice" ]] || \
            [[ "$app_dir_name" = "ops-lbm/step5" ]]; then
                NP=2
            else
                NP=4
            fi
        fi
#       Run tiled version
        if [[ -n "$OPS_TILING_MAXDEPTH" ]]; then
            echo "     command : mpirun -np $NP  ./$app_target_name OPS_TILING OPS_TILING_MAXDEPTH=$OPS_TILING_MAXDEPTH 2>&1 | tee log_out.txt" >> $log_file
            mpirun -np $NP  ./$app_target_name OPS_TILING OPS_TILING_MAXDEPTH=$OPS_TILING_MAXDEPTH 2>&1 | tee log_out.txt
        else
            if [[ "$app_dir_name" = "hdf5_slice" ]]; then
                echo "skipping running mpi tiled version for hdf5_slice, getting stuck for longer, need fix" >> $log_file
                return 1
            else
                echo "     command: mpirun -np $NP  ./$app_target_name OPS_TILING 2>&1 | tee log_out.txt" >> $log_file
                mpirun -np $NP  ./$app_target_name OPS_TILING 2>&1 | tee log_out.txt
            fi
        fi
#   ===============================
#   IF MPI version - Without tiling
#   ==============================
    elif [[ "$target" = *"mpi"* ]] && [[ "$target" != *"tiled"* ]]; then
        if [[ "$target" = *"cuda"* ]] || [[ "$target" = *"hip"* ]] || [[ "$target" = *"sycl"* ]]; then
            NP=2
        else
            if [[ "$app_dir_name" = "CloverLeaf" ]] || [[ "$app_dir_name" = "CloverLeaf_3D" ]] || [[ "$app_dir_name" = "CloverLeaf_3D_HDF5" ]] || \
            [[ "$app_dir_name" = "mgrid" ]] || [[ "$app_dir_name" = "shsgc" ]] || [[ "$app_dir_name" = "hdf5_slice" ]] || \
            [[ "$app_dir_name" = "ops-lbm/step5" ]]; then
                NP=2
            else
                NP=4
            fi
        fi
        echo "     command: mpirun -np $NP  ./$app_target_name 2>&1 | tee log_out.txt" >> $log_file
        mpirun -np $NP  ./$app_target_name 2>&1 | tee log_out.txt
#   =========================
#   IF Tiled version: Non-MPI
#   =========================
    elif [[ "$target" != *"mpi"* ]] && [[ "$target" = *"tiled"* ]]; then
        echo "     command: ./$app_target_name OPS_TILING 2>&1 | tee log_out.txt" >> $log_file
        ./$app_target_name OPS_TILING 2>&1 | tee log_out.txt
#   =====================
#   IF sequential version
#   =====================
    else
        echo "     command: ./$app_target_name 2>&1 | tee log_out.txt" >> $log_file
        ./$app_target_name 2>&1 | tee log_out.txt
    fi

#   Check the difference in HDF5 files created for following applications
#   CloverLeaf_3D_HDF5, multiDim_HDF5, mgrid, hdf5_slice
    if [[ "$app_dir_name" = "CloverLeaf_3D_HDF5" ]]; then
        $HDF5_INSTALL_PATH/bin/h5diff cloverdata.h5 test_cloverdata.h5 > diff_out.log
#       if file diff_out is not empty that means h5diff reported differences
        if [ -s ./diff_out.log ]; then
            mv test_cloverdata.h5 test_cloverdata_failed.h5
            echo " FAILURE - HDF5 file not-matched with reference file for target: $target" >> $log_file
            return 1
        else
            echo " SUCCESS - HDF5 file matched with reference file for target: $target" >> $log_file
        fi
        rm test_cloverdata.h5
        rm diff_out.log
    fi

    if [[ "$app_dir_name" = "multiDim_HDF5" ]]; then
        echo "checking hdf5 status "$app_name
#       If app is read, then check the difference from write and read HDF5
        if [[ "$app_name" = "read" ]]; then
            $HDF5_INSTALL_PATH/bin/h5diff write_data.h5 read_data.h5 > diff_out.log
#           if file diff_out is not empty that means h5diff reported differences
            if [ -s ./diff_out.log ]; then
                mv write_data.h5 write_data_failed.h5
                mv read_data.h5 read_data_failed.h5
                echo " FAILURE - HDF5 file not-matched with reference file for target: $target" >> $log_file
                return 1
            else
                echo " SUCCESS - HDF5 file matched with reference file for target: $target" >> $log_file
            fi
            rm write_data.h5 read_data.h5
            rm diff_out.log
        fi
    fi

    if [[ "$app_dir_name" = "mgrid" ]]; then
        if [[ "$target" = "seq" ]]; then
            mv data.h5 data_ref.h5
        else
            if [ -e "$app_path/data_ref.h5" ]; then
                $HDF5_INSTALL_PATH/bin/h5diff data.h5 data_ref.h5 > diff_out.log
                if [ -s ./diff_out.log ]; then
                    mv data.h5 data_failed.h5
                    echo " FAILURE - HDF5 file not-matched with reference file for target: $target" >> $log_file
                    return 1
                else
                    echo " SUCCESS - HDF5 file matched with reference file for target: $target" >> $log_file
                fi
                rm data.h5
                rm diff_out.log
            fi
        fi
    fi

    if [[ "$app_dir_name" = "hdf5_slice" ]]; then
        if [[ "$target" = "dev_seq" ]]; then
            mv double.h5 double_ref.h5; mv half.h5 half_ref.h5; mv I1.h5 I1_ref.h5; mv I4.h5 I4_ref.h5
            mv J16.h5 J16_ref.h5; mv J8.h5 J8_ref.h5; mv K15.h5 K15_ref.h5; mv K16.h5 K16_ref.h5
            mv single.h5 single_ref.h5; mv slab.h5 slab_ref.h5; mv slice3Du.h5 slice3Du_ref.h5; mv slice3Dv.h5 slice3Dv_ref.h5
        else
            hdf5_files=("double" "half" "I1" "I4" "J16" "J8" "K15" "K16" "single" "slab" "slice3Du" "slice3Dv")
            result=0
            for hdf5_file_name in "${hdf5_files[@]}"; do
                $HDF5_INSTALL_PATH/bin/h5diff ${hdf5_file_name}.h5 ${hdf5_file_name}_ref.h5 > diff_out.log
                if [ -s ./diff_out.log ]; then
                    mv ${hdf5_file_name}.h5 ${hdf5_file_name}_failed.h5
                    ((result++))
                else
                    rm ${hdf5_file_name}.h5
                    rm diff_out.log
                fi
            done
            if [[ $result != 0 ]]; then
                echo " FAILURE - HDF5 file/files not-matched with reference file/files for target: $target" >> $log_file
                return 1
            else
                echo " SUCCESS - HDF5 files matched with reference files for target: $target" >> $log_file
            fi
        fi
    fi

#   Check status from log file
    grep_msg="PASSED"
    if [[ "$app_dir_name" = "multiDim_HDF5" ]] || [[ "$app_dir_name" = "hdf5_slice" ]]; then
        grep_msg="Sucessful exit from OPS"
    fi
    grep "$grep_msg" log_out.txt > /dev/null
    result=$?
    if [[ $result != 0 ]]; then
        return 1
    else
#       if passed then only remove the log file, otherwise keep it for checking
        rm -f log_out.txt
        return 0
    fi
}

function test_trid_cpp_app ( ) {
    local app_name="$1"
    echo "Testing: $app_name"

    local app_path=$OPS_INSTALL_PATH/../apps/c/$app_name
    is_path_exist "$app_path"
}

# Call the check environment function
check_env

# Build the CPP OPS library
build_cpp_lib

build_and_run_cpp_apps

