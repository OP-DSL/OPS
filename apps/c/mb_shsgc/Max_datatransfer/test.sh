#!/bin/bash
set -e
cd $OPS_INSTALL_PATH/c

export SOURCE_INTEL=source_intel_2021.3_pythonenv
export SOURCE_PGI=source_pgi_nvhpc_23_pythonenv
export SOURCE_INTEL_SYCL=source_intel_2021.3_sycl_pythonenv
export SOURCE_AMD_HIP=source_amd_rocm-5.4.3_pythonenv

#export AMOS=TRUE
#export DMOS=TRUE
#export TELOS=TRUE
#export KOS=TRUE

if [[ -v TELOS || -v KOS ]]; then

#============================ Test with Intel Classic Compilers==========================================
echo "Testing Intel classic complier based applications ---- "
cd $OPS_INSTALL_PATH/c
source ../../scripts/$SOURCE_INTEL
make clean
make

cd $OPS_INSTALL_PATH/../apps/c/mb_shsgc/Max_datatransfer
make clean
make IEEE=1


#============================ Test SHSGC with Intel Compilers==========================================================
echo '============> Running OpenMP'
KMP_AFFINITY=compact OMP_NUM_THREADS=10 ./shsgc_openmp > perf_out
grep "Pre shock error is:" perf_out
grep "Post shock error is:" perf_out
grep "Post shock Error is" perf_out
grep "Total Wall time" perf_out
grep -e "acceptable" -e "correct"  perf_out
rc=$?; if [[ $rc != 0 ]]; then echo "TEST FAILED";exit $rc; fi
rm perf_out

echo '============> Running MPI+OpenMP'
export OMP_NUM_THREADS=2;$MPI_INSTALL_PATH/bin/mpirun -np 10 ./shsgc_mpi_openmp > perf_out
grep "Pre shock error is:" perf_out
grep "Post shock error is:" perf_out
grep "Post shock Error is" perf_out
grep "Total Wall time" perf_out
grep -e "acceptable" -e "correct"  perf_out
rc=$?; if [[ $rc != 0 ]]; then echo "TEST FAILED";exit $rc; fi
rm perf_out

echo '============> Running DEV_MPI'
$MPI_INSTALL_PATH/bin/mpirun -np 20 ./shsgc_dev_mpi > perf_out
grep "Pre shock error is:" perf_out
grep "Post shock error is:" perf_out
grep "Post shock Error is" perf_out
grep "Total Wall time" perf_out
grep -e "acceptable" -e "correct"  perf_out
rc=$?; if [[ $rc != 0 ]]; then echo "TEST FAILED";exit $rc; fi
rm perf_out

echo '============> Running MPI'
$MPI_INSTALL_PATH/bin/mpirun -np 20 ./shsgc_mpi > perf_out
grep "Pre shock error is:" perf_out
grep "Post shock error is:" perf_out
grep "Post shock Error is" perf_out
grep "Total Wall time" perf_out
grep -e "acceptable" -e "correct"  perf_out
rc=$?; if [[ $rc != 0 ]]; then echo "TEST FAILED";exit $rc; fi
rm perf_out

if [[ -v CUDA_INSTALL_PATH ]]; then
echo '============> Running CUDA'
./shsgc_cuda OPS_BLOCK_SIZE_X=64 OPS_BLOCK_SIZE_Y=4 > perf_out
grep "Pre shock error is:" perf_out
grep "Post shock error is:" perf_out
grep "Post shock Error is" perf_out
grep "Total Wall time" perf_out
grep -e "acceptable" -e "correct"  perf_out
rc=$?; if [[ $rc != 0 ]]; then echo "TEST FAILED";exit $rc; fi
rm perf_out

echo '============> Running MPI+CUDA'
$MPI_INSTALL_PATH/bin/mpirun -np 2 ./shsgc_mpi_cuda OPS_BLOCK_SIZE_X=64 OPS_BLOCK_SIZE_Y=4 > perf_out
grep "Pre shock error is:" perf_out
grep "Post shock error is:" perf_out
grep "Post shock Error is" perf_out
grep "Total Wall time" perf_out
grep -e "acceptable" -e "correct"  perf_out
rc=$?; if [[ $rc != 0 ]]; then echo "TEST FAILED";exit $rc; fi
rm perf_out

#echo '============> Running MPI+CUDA with GPU-Direct'
#MV2_USE_CUDA=1 $MPI_INSTALL_PATH/bin/mpirun -np 2 ./shsgc_mpi_cuda -gpudirect OPS_BLOCK_SIZE_X=64 OPS_BLOCK_SIZE_Y=4 > perf_out
#grep "Pre shock error is:" perf_out
#grep "Post shock error is:" perf_out
#grep "Post shock Error is" perf_out
#grep "Total Wall time" perf_out
#grep -e "acceptable" -e "correct"  perf_out
#rc=$?; if [[ $rc != 0 ]]; then echo "TEST FAILED";exit $rc; fi
#rm perf_out
fi

#cleanup
rm rhoin1.* x1.* rhoout1.*  rhoin1 rhoout1 x1

echo "All Intel classic complier based applications ---- PASSED"

fi


if [[ -v TELOS ]]; then
	#============================ Test with Intel SYCL Compilers==========================================
echo "Testing Intel SYCL complier based applications ---- "
cd $OPS_INSTALL_PATH/c
source ../../scripts/$SOURCE_INTEL_SYCL
make clean
make
cd $OPS_INSTALL_PATH/../apps/c/mb_shsgc/Max_datatransfer

make clean
#make IEEE=1 -j
make IEEE=1 shsgc_sycl shsgc_mpi_sycl shsgc_mpi_sycl_tiled

echo '============> Running SYCL on CPU'
./shsgc_sycl OPS_CL_DEVICE=0 OPS_BLOCK_SIZE_X=32 OPS_BLOCK_SIZE_Y=4 > perf_out
grep "Pre shock error is:" perf_out
grep "Post shock error is:" perf_out
grep "Post shock Error is" perf_out
grep "Total Wall time" perf_out
grep -e "acceptable" -e "correct"  perf_out
rc=$?; if [[ $rc != 0 ]]; then echo "TEST FAILED";exit $rc; fi
rm perf_out

echo '============> Running MPI+SYCL on CPU'
$MPI_INSTALL_PATH/bin/mpirun -np 12 ./shsgc_mpi_sycl OPS_CL_DEVICE=0 OPS_BLOCK_SIZE_X=256 OPS_BLOCK_SIZE_Y=1 > perf_out
grep "Pre shock error is:" perf_out
grep "Post shock error is:" perf_out
grep "Post shock Error is" perf_out
grep "Total Wall time" perf_out
grep -e "acceptable" -e "correct"  perf_out
rc=$?; if [[ $rc != 0 ]]; then echo "TEST FAILED";exit $rc; fi
rm perf_out

echo '============> Running MPI+SYCL Tiled on CPU'
$MPI_INSTALL_PATH/bin/mpirun -np 2 ./shsgc_mpi_sycl_tiled OPS_CL_DEVICE=1 OPS_BLOCK_SIZE_X=32 OPS_BLOCK_SIZE_Y=4 > perf_out
grep "Pre shock error is:" perf_out
grep "Post shock error is:" perf_out
grep "Post shock Error is" perf_out
grep "Total Wall time" perf_out
grep -e "acceptable" -e "correct"  perf_out
rc=$?; if [[ $rc != 0 ]]; then echo "TEST FAILED";exit $rc; fi
rm perf_out

#cleanup
rm rhoin1.* x1.* rhoout1.*  rhoin1 rhoout1 x1

echo "All Intel SYCL complier based applications ---- PASSED"

fi


if [[ -v TELOS ]]; then

#============================ Test with PGI Compilers==========================================
echo "Testing PGI/NVHPC complier based applications ---- "
cd $OPS_INSTALL_PATH/c
source ../../scripts/$SOURCE_PGI
make clean
#make -j
make
cd $OPS_INSTALL_PATH/../apps/c/mb_shsgc/Max_datatransfer
make clean
make IEEE=1

echo '============> Running OpenMP'
KMP_AFFINITY=compact OMP_NUM_THREADS=10 ./shsgc_openmp > perf_out
grep "Pre shock error is:" perf_out
grep "Post shock error is:" perf_out
grep "Post shock Error is" perf_out
grep "Total Wall time" perf_out
grep -e "acceptable" -e "correct"  perf_out
rc=$?; if [[ $rc != 0 ]]; then echo "TEST FAILED";exit $rc; fi
rm perf_out

echo '============> Running MPI+OpenMP'
export OMP_NUM_THREADS=2;$MPI_INSTALL_PATH/bin/mpirun -np 10 ./shsgc_mpi_openmp > perf_out
grep "Pre shock error is:" perf_out
grep "Post shock error is:" perf_out
grep "Post shock Error is" perf_out
grep "Total Wall time" perf_out
grep -e "acceptable" -e "correct"  perf_out
rc=$?; if [[ $rc != 0 ]]; then echo "TEST FAILED";exit $rc; fi
rm perf_out

echo '============> Running DEV_MPI'
$MPI_INSTALL_PATH/bin/mpirun -np 20 ./shsgc_dev_mpi > perf_out
grep "Pre shock error is:" perf_out
grep "Post shock error is:" perf_out
grep "Post shock Error is" perf_out
grep "Total Wall time" perf_out
grep -e "acceptable" -e "correct"  perf_out
rc=$?; if [[ $rc != 0 ]]; then echo "TEST FAILED";exit $rc; fi
rm perf_out

echo '============> Running MPI'
$MPI_INSTALL_PATH/bin/mpirun -np 20 ./shsgc_mpi > perf_out
grep "Pre shock error is:" perf_out
grep "Post shock error is:" perf_out
grep "Post shock Error is" perf_out
grep "Total Wall time" perf_out
grep -e "acceptable" -e "correct"  perf_out
rc=$?; if [[ $rc != 0 ]]; then echo "TEST FAILED";exit $rc; fi
rm perf_out

if [[ -v CUDA_INSTALL_PATH ]]; then
echo '============> Running CUDA'
./shsgc_cuda OPS_BLOCK_SIZE_X=64 OPS_BLOCK_SIZE_Y=4 > perf_out
grep "Pre shock error is:" perf_out
grep "Post shock error is:" perf_out
grep "Post shock Error is" perf_out
grep "Total Wall time" perf_out
grep -e "acceptable" -e "correct"  perf_out
rc=$?; if [[ $rc != 0 ]]; then echo "TEST FAILED";exit $rc; fi
rm perf_out

echo '============> Running MPI+CUDA'
$MPI_INSTALL_PATH/bin/mpirun -np 2 ./shsgc_mpi_cuda OPS_BLOCK_SIZE_X=64 OPS_BLOCK_SIZE_Y=4 > perf_out
grep "Pre shock error is:" perf_out
grep "Post shock error is:" perf_out
grep "Post shock Error is" perf_out
grep "Total Wall time" perf_out
grep -e "acceptable" -e "correct"  perf_out
rc=$?; if [[ $rc != 0 ]]; then echo "TEST FAILED";exit $rc; fi
rm perf_out

echo '============> Running MPI+CUDA+Tiled'
$MPI_INSTALL_PATH/bin/mpirun -np 2 ./shsgc_mpi_cuda_tiled OPS_BLOCK_SIZE_X=64 OPS_BLOCK_SIZE_Y=4 > perf_out
grep "Pre shock error is:" perf_out
grep "Post shock error is:" perf_out
grep "Post shock Error is" perf_out
grep "Total Wall time" perf_out
grep -e "acceptable" -e "correct"  perf_out
rc=$?; if [[ $rc != 0 ]]; then echo "TEST FAILED";exit $rc; fi
rm perf_out

#echo '============> Running MPI+CUDA with GPU-Direct'
#MV2_USE_CUDA=1 $MPI_INSTALL_PATH/bin/mpirun -np 2 ./shsgc_mpi_cuda -gpudirect OPS_BLOCK_SIZE_X=64 OPS_BLOCK_SIZE_Y=4 > perf_out
#grep "Pre shock error is:" perf_out
#grep "Post shock error is:" perf_out
#grep "Post shock Error is" perf_out
#grep "Total Wall time" perf_out
#grep -e "acceptable" -e "correct"  perf_out
#rc=$?; if [[ $rc != 0 ]]; then echo "TEST FAILED";exit $rc; fi
#rm perf_out

fi

echo '============> Running OMPOFFLOAD'
./shsgc_offload OPS_BLOCK_SIZE_X=64 OPS_BLOCK_SIZE_Y=4 > perf_out
grep "Pre shock error is:" perf_out
grep "Post shock error is:" perf_out
grep "Post shock Error is" perf_out
grep "Total Wall time" perf_out
grep -e "acceptable" -e "correct"  perf_out
rc=$?; if [[ $rc != 0 ]]; then echo "TEST FAILED";exit $rc; fi
rm perf_out

echo '============> Running MPI+OMPOFFLOAD'
$MPI_INSTALL_PATH/bin/mpirun -np 2 ./shsgc_mpi_offload OPS_BLOCK_SIZE_X=64 OPS_BLOCK_SIZE_Y=4 > perf_out
grep "Pre shock error is:" perf_out
grep "Post shock error is:" perf_out
grep "Post shock Error is" perf_out
grep "Total Wall time" perf_out
grep -e "acceptable" -e "correct"  perf_out
rc=$?; if [[ $rc != 0 ]]; then echo "TEST FAILED";exit $rc; fi
rm perf_out

#cleanup
rm rhoin1.* x1.* rhoout1.*  rhoin1 rhoout1 x1

echo "All PGI complier based applications ---- PASSED"

fi

if [[ -v AMOS ]]; then

echo "Testing AMD HIP complier based applications ---- "
cd $OPS_INSTALL_PATH/c
source ../../scripts/$SOURCE_AMD_HIP
make clean
make

cd $OPS_INSTALL_PATH/../apps/c/mb_shsgc/Max_datatransfer
make clean
make IEEE=1 shsgc_hip shsgc_mpi_hip

echo '============> Running HIP'
./shsgc_hip OPS_BLOCK_SIZE_X=64 OPS_BLOCK_SIZE_Y=4 > perf_out
grep "Pre shock error is:" perf_out
grep "Post shock error is:" perf_out
grep "Post shock Error is" perf_out
grep "Total Wall time" perf_out
grep -e "acceptable" -e "correct"  perf_out
rc=$?; if [[ $rc != 0 ]]; then echo "TEST FAILED";exit $rc; fi
rm perf_out

echo '============> Running MPI+HIP'
mpirun --allow-run-as-root -np 2 ./shsgc_mpi_hip OPS_BLOCK_SIZE_X=64 OPS_BLOCK_SIZE_Y=4 > perf_out
grep "Pre shock error is:" perf_out
grep "Post shock error is:" perf_out
grep "Post shock Error is" perf_out
grep "Total Wall time" perf_out
grep -e "acceptable" -e "correct"  perf_out
rc=$?; if [[ $rc != 0 ]]; then echo "TEST FAILED";exit $rc; fi
rm perf_out

echo "All AMD HIP complier based applications ---- PASSED"

#cleanup
rm rhoin1.* x1.* rhoout1.*  rhoin1 rhoout1 x1

fi

echo "---------- Exiting Test Script "
