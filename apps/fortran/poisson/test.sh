#!/bin/bash
set -e
cd ../../../ops/fortran
source ../../scripts/source_intel
make
cd -
make clean
rm -f .generated
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
source ../../scripts/source_pgi_18

make clean
make
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
$MPI_INSTALL_PATH/bin/mpirun -np 2 numawrap2 ./poisson_mpi_cuda OPS_BLOCK_SIZE_X=64 OPS_BLOCK_SIZE_Y=4 > perf_out
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

echo '============> Running OpenACC'
./poisson_openacc OPS_BLOCK_SIZE_X=64 OPS_BLOCK_SIZE_Y=4 > perf_out
grep "Total error:" perf_out
grep "Max total runtime" perf_out
grep "PASSED" perf_out
rc=$?; if [[ $rc != 0 ]]; then echo "TEST FAILED";exit $rc; fi
rm perf_out

echo "All PGI tests passed. Exiting Test script"
