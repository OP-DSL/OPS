#!/bin/bash

#cd ../../ops/
#source ./source_intel
#make
#cd -
#make
#============================ Test Poisson ==========================================================
echo '============> Running OpenMP'
KMP_AFFINITY=compact OMP_NUM_THREADS=24 ./poisson_openmp > perf_out
grep "Total error:" perf_out
grep "Total Wall time" perf_out
rm perf_out
echo '============> Running MPI+OpenMP'
export OMP_NUM_THREADS=2;$MPI_INSTALL_PATH/bin/mpirun -np 12 ./poisson_mpi_openmp > perf_out
grep "Total error:" perf_out
grep "Total Wall time" perf_out
rm perf_out
echo '============> Running MPI'
$MPI_INSTALL_PATH/bin/mpirun -np 22 ./poisson_mpi > perf_out
grep "Total error:" perf_out
grep "Total Wall time" perf_out
rm perf_out
echo '============> Running CUDA'
./poisson_cuda OPS_BLOCK_SIZE_X=64 OPS_BLOCK_SIZE_Y=4 > perf_out
grep "Total error:" perf_out
grep "Total Wall time" perf_out
rm perf_out
echo '============> Running MPI+CUDA'
$MPI_INSTALL_PATH/bin/mpirun -np 2 ./poisson_mpi_cuda OPS_BLOCK_SIZE_X=64 OPS_BLOCK_SIZE_Y=4 > perf_out
grep "Total error:" perf_out
grep "Total Wall time" perf_out
rm perf_out
echo '============> Running MPI+CUDA with GPU-Direct'
MV2_USE_CUDA=1 $MPI_INSTALL_PATH/bin/mpirun -np 2 ./poisson_mpi_cuda -gpudirect OPS_BLOCK_SIZE_X=64 OPS_BLOCK_SIZE_Y=4 > perf_out
grep "Total error:" perf_out
grep "Total Wall time" perf_out
rm perf_out

#cd -
#source ./source_pgi
#make cuda
#cd -
#make poisson_openacc
#echo '============> Running OpenACC'
#./poisson_openacc OPS_BLOCK_SIZE_X=64 OPS_BLOCK_SIZE_Y=4 > perf_out
#grep "Total error:" perf_out
#grep "Total Wall time" perf_out
#rm perf_out
