set -e

export SOURCE_INTEL=source_intel_2021.3_pythonenv
export SOURCE_PGI=source_pgi_nvhpc_23_pythonenv
export SOURCE_INTEL_SYCL=source_intel_2021.3_sycl_pythonenv
export SOURCE_AMD_HIP=source_amd_rocm-5.4.3_pythonenv

#export AMOS=TRUE
#export DMOS=TRUE
export TELOS=TRUE
#export KOS=TRUE


cd $OPS_INSTALL_PATH/fortran
source ../../scripts/$SOURCE_INTEL
make
cd -
make clean
make IEEE=1

echo '============================ Test Poisson Intel Compilers=========================================================='
echo '============> Running OpenMP'
KMP_AFFINITY=compact OMP_NUM_THREADS=20 ./poisson_openmp > perf_out
grep "Total error:" perf_out
grep "Max total runtime" perf_out
grep "PASSED" perf_out
rc=$?; if [[ $rc != 0 ]]; then echo "TEST FAILED";exit $rc; fi
rm perf_out

echo '============> Running MPI+OpenMP'
export OMP_NUM_THREADS=2;KMP_AFFINITY=compact $MPI_INSTALL_PATH/bin/mpirun -np 10 ./poisson_mpi_openmp > perf_out
grep "Total error:" perf_out
grep "Max total runtime" perf_out
grep "PASSED" perf_out
rc=$?; if [[ $rc != 0 ]]; then echo "TEST FAILED";exit $rc; fi
rm perf_out

echo '============> Running MPI'
$MPI_INSTALL_PATH/bin/mpirun -np 20 ./poisson_mpi > perf_out
grep "Total error:" perf_out
grep "Max total runtime" perf_out
grep "PASSED" perf_out
rc=$?; if [[ $rc != 0 ]]; then echo "TEST FAILED";exit $rc; fi
rm perf_out

cd $OPS_INSTALL_PATH/fortran

source ../../scripts/$SOURCE_PGI
#make clean
#make
cd -
make clean
make


echo '============================ Test Poisson PGI Compilers=========================================================='
echo '============> Running OpenMP'
KMP_AFFINITY=compact OMP_NUM_THREADS=20 ./poisson_openmp > perf_out
grep "Total error:" perf_out
grep "Max total runtime" perf_out
grep "PASSED" perf_out
rc=$?; if [[ $rc != 0 ]]; then echo "TEST FAILED";exit $rc; fi
rm perf_out

echo '============> Running MPI+OpenMP'
export OMP_NUM_THREADS=2;$MPI_INSTALL_PATH/bin/mpirun -np 10 ./poisson_mpi_openmp > perf_out
grep "Total error:" perf_out
grep "Max total runtime" perf_out
grep "PASSED" perf_out
rc=$?; if [[ $rc != 0 ]]; then echo "TEST FAILED";exit $rc; fi
rm perf_out

echo '============> Running MPI'
$MPI_INSTALL_PATH/bin/mpirun -np 20 ./poisson_mpi > perf_out
grep "Total error:" perf_out
grep "Max total runtime" perf_out
grep "PASSED" perf_out
rc=$?; if [[ $rc != 0 ]]; then echo "TEST FAILED";exit $rc; fi
rm perf_out

echo '============> Running CUDA'
./poisson_cuda OPS_BLOCK_SIZE_X=64 OPS_BLOCK_SIZE_Y=4 > perf_out
grep "Total error:" perf_out
grep "Max total runtime" perf_out
grep "PASSED" perf_out
rc=$?; if [[ $rc != 0 ]]; then echo "TEST FAILED";exit $rc; fi
rm perf_out

echo '============> Running MPI+CUDA'
$MPI_INSTALL_PATH/bin/mpirun -np 2 ./poisson_mpi_cuda OPS_BLOCK_SIZE_X=64 OPS_BLOCK_SIZE_Y=4 > perf_out
grep "Total error:" perf_out
grep "Max total runtime" perf_out
grep "PASSED" perf_out
rc=$?; if [[ $rc != 0 ]]; then echo "TEST FAILED";exit $rc; fi
rm perf_out

#echo '============> Running MPI+CUDA with GPU-Direct'
#MV2_USE_CUDA=1 $MPI_INSTALL_PATH/bin/mpirun -np 2 ./poisson_mpi_cuda -gpudirect OPS_BLOCK_SIZE_X=64 OPS_BLOCK_SIZE_Y=4 > perf_out
#grep "Total error:" perf_out
#grep "Max total runtime" perf_out
#grep "PASSED" perf_out
#rc=$?; if [[ $rc != 0 ]]; then echo "TEST FAILED";exit $rc; fi
#rm perf_out

echo "All PGI tests passed. Exiting Test script"
