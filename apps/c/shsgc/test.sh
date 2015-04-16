#!/bin/bash

cd ../../../ops/c
source ../rubly_intel_source
make
cd -
make
#============================ Test SHSGC ==========================================================
echo '============> Running OpenMP'
KMP_AFFINITY=compact OMP_NUM_THREADS=12 ./shsgc_openmp > perf_out
grep "RMS =" perf_out
grep "Total Wall time" perf_out
rm perf_out
echo '============> Running MPI+OpenMP'
export OMP_NUM_THREADS=2;$MPI_INSTALL_PATH/bin/mpirun -np 12 ./shsgc_mpi_openmp > perf_out
grep "RMS =" perf_out
grep "Total Wall time" perf_out
rm perf_out
echo '============> Running MPI'
$MPI_INSTALL_PATH/bin/mpirun -np 22 ./shsgc_mpi > perf_out
grep "RMS =" perf_out
grep "Total Wall time" perf_out
rm perf_out
echo '============> Running CUDA'
./shsgc_cuda OPS_BLOCK_SIZE_X=64 OPS_BLOCK_SIZE_Y=4 > perf_out
grep "RMS =" perf_out
grep "Total Wall time" perf_out
rm perf_out
echo '============> Running MPI+CUDA'
$MPI_INSTALL_PATH/bin/mpirun -np 2 ./shsgc_mpi_cuda OPS_BLOCK_SIZE_X=64 OPS_BLOCK_SIZE_Y=4 > perf_out
grep "RMS =" perf_out
grep "Total Wall time" perf_out
rm perf_out
echo '============> Running MPI+CUDA with GPU-Direct'
MV2_USE_CUDA=1 $MPI_INSTALL_PATH/bin/mpirun -np 2 ./shsgc_mpi_cuda -gpudirect OPS_BLOCK_SIZE_X=64 OPS_BLOCK_SIZE_Y=4 > perf_out
grep "RMS =" perf_out
grep "Total Wall time" perf_out
rm perf_out
echo '============> Running OpenCL on CPU'
./shsgc_opencl OPS_CL_DEVICE=0 OPS_BLOCK_SIZE_X=512 OPS_BLOCK_SIZE_Y=1 > perf_out
grep "RMS =" perf_out
grep "Total Wall time" perf_out
rm perf_out
echo '============> Running OpenCL on GPU'
./shsgc_opencl OPS_CL_DEVICE=1 OPS_BLOCK_SIZE_X=32 OPS_BLOCK_SIZE_Y=4 > perf_out
./shsgc_opencl OPS_CL_DEVICE=1 OPS_BLOCK_SIZE_X=32 OPS_BLOCK_SIZE_Y=4 > perf_out
grep "RMS =" perf_out
grep "Total Wall time" perf_out
rm perf_out
echo '============> Running MPI+OpenCL on CPU'
$MPI_INSTALL_PATH/bin/mpirun -np 24 ./shsgc_mpi_opencl OPS_CL_DEVICE=0 OPS_BLOCK_SIZE_X=256 OPS_BLOCK_SIZE_Y=1 > perf_out
$MPI_INSTALL_PATH/bin/mpirun -np 24 ./shsgc_mpi_opencl OPS_CL_DEVICE=0 OPS_BLOCK_SIZE_X=256 OPS_BLOCK_SIZE_Y=1 > perf_out
grep "RMS =" perf_out
grep "Total Wall time" perf_out
rm perf_out
echo '============> Running MPI+OpenCL on GPU'
$MPI_INSTALL_PATH/bin/mpirun -np 2 ./shsgc_mpi_opencl OPS_CL_DEVICE=1 OPS_BLOCK_SIZE_X=32 OPS_BLOCK_SIZE_Y=4 > perf_out
$MPI_INSTALL_PATH/bin/mpirun -np 2 ./shsgc_mpi_opencl OPS_CL_DEVICE=1 OPS_BLOCK_SIZE_X=32 OPS_BLOCK_SIZE_Y=4 > perf_out
grep "RMS =" perf_out
grep "Total Wall time" perf_out
rm perf_out

cd -
source ../rubly_pgi_source
make clean
make
cd -
make shsgc_openacc
make shsgc_mpi_openacc
echo '============> Running OpenACC'
./shsgc_openacc OPS_BLOCK_SIZE_X=64 OPS_BLOCK_SIZE_Y=4 > perf_out
grep "RMS =" perf_out
grep "Total Wall time" perf_out
rm perf_out
echo '============> Running MPI+OpenACC'
$MPI_INSTALL_PATH/bin/mpirun -np 2 ./shsgc_mpi_openacc OPS_BLOCK_SIZE_X=64 OPS_BLOCK_SIZE_Y=4 > perf_out
grep "RMS =" perf_out
grep "Total Wall time" perf_out
rm perf_out
