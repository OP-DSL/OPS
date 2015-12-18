#!/bin/bash

cd ../../../ops/c
source ../source_intel
make
cd -
make
echo '============> Running OpenMP'
KMP_AFFINITY=compact OMP_NUM_THREADS=20 ./multidim_openmp > perf_out
grep "Reduction result" perf_out
grep "Total Wall time" perf_out
rm perf_out
echo '============> Running MPI+OpenMP'
export OMP_NUM_THREADS=2;$MPI_INSTALL_PATH/bin/mpirun -np 10 ./multidim_mpi_openmp > perf_out
grep "Reduction result" perf_out
grep "Total Wall time" perf_out
rm perf_out
echo '============> Running MPI'
$MPI_INSTALL_PATH/bin/mpirun -np 20 ./multidim_mpi > perf_out
grep "Reduction result" perf_out
grep "Total Wall time" perf_out
rm perf_out


cd $OPS_INSTALL_PATH/fortran
source ../source_pgi
export LM_LICENSE_FILE=/opt/pgi/license.dat
make
cd -
make
echo '============> Running CUDA'
./multidim_cuda OPS_BLOCK_SIZE_X=64 OPS_BLOCK_SIZE_Y=4 > perf_out
grep "Reduction result" perf_out
grep "Total Wall time" perf_out
rm perf_out
echo '============> Running MPI+CUDA'
$MPI_INSTALL_PATH/bin/mpirun -np 2 ./multidim_mpi_cuda OPS_BLOCK_SIZE_X=64 OPS_BLOCK_SIZE_Y=4 > perf_out
grep "Reduction result" perf_out
grep "Total Wall time" perf_out
rm perf_out
echo '============> Running MPI+CUDA with GPU-Direct'
MV2_USE_CUDA=1 $MPI_INSTALL_PATH/bin/mpirun -np 2 ./multidim_mpi_cuda -gpudirect OPS_BLOCK_SIZE_X=64 OPS_BLOCK_SIZE_Y=4 > perf_out
grep "Reduction result" perf_out
grep "Total Wall time" perf_out
rm perf_out
