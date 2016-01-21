#!/bin/bash
#set -e
cd ../../../ops/c
source ../source_intel
make
cd -
make
#============================ Test Cloverleaf 2D With Intel Compilers==========================================================
echo '============> Running OpenMP'
KMP_AFFINITY=compact OMP_NUM_THREADS=20 ./cloverleaf_openmp > perf_out
grep "Total Wall time" clover.out
grep "step:   2955" clover.out
echo '============> Running MPI+OpenMP'
export OMP_NUM_THREADS=2;$MPI_INSTALL_PATH/bin/mpirun -np 10 ./cloverleaf_mpi_openmp > perf_out
grep "Total Wall time" clover.out
grep "step:   2955" clover.out
echo '============> Running DEV_MPI'
$MPI_INSTALL_PATH/bin/mpirun -np 20 ./cloverleaf_dev_mpi > perf_out
grep "Total Wall time" clover.out
grep "step:   2955" clover.out
echo '============> Running MPI'
$MPI_INSTALL_PATH/bin/mpirun -np 20 ./cloverleaf_mpi > perf_out
grep "Total Wall time" clover.out
grep "step:   2955" clover.out
echo '============> Running CUDA'
./cloverleaf_cuda OPS_BLOCK_SIZE_X=64 OPS_BLOCK_SIZE_Y=4 > perf_out
grep "Total Wall time" clover.out
grep "step:   2953" clover.out
echo '============> Running MPI+CUDA'
$MPI_INSTALL_PATH/bin/mpirun -np 2 ./cloverleaf_mpi_cuda OPS_BLOCK_SIZE_X=64 OPS_BLOCK_SIZE_Y=4 > perf_out
grep "Total Wall time" clover.out
grep "step:   2953" clover.out
echo '============> Running MPI+CUDA with GPU-Direct'
MV2_USE_CUDA=1 $MPI_INSTALL_PATH/bin/mpirun -np 2 ./cloverleaf_mpi_cuda -gpudirect OPS_BLOCK_SIZE_X=64 OPS_BLOCK_SIZE_Y=4 > perf_out
grep "Total Wall time" clover.out
grep "step:   2953" clover.out
echo '============> Running OpenCL on CPU'
./cloverleaf_opencl OPS_CL_DEVICE=0 OPS_BLOCK_SIZE_X=512 OPS_BLOCK_SIZE_Y=1 > perf_out
grep "Total Wall time" clover.out
grep "step:   2955" clover.out
echo '============> Running OpenCL on GPU'
./cloverleaf_opencl OPS_CL_DEVICE=1 OPS_BLOCK_SIZE_X=32 OPS_BLOCK_SIZE_Y=4 > perf_out
./cloverleaf_opencl OPS_CL_DEVICE=1 OPS_BLOCK_SIZE_X=32 OPS_BLOCK_SIZE_Y=4 > perf_out
grep "Total Wall time" clover.out
grep "step:   2953" clover.out
grep "step:   2954" clover.out
grep "step:   2955" clover.out
rm perf_out
echo '============> Running MPI+OpenCL on CPU'
$MPI_INSTALL_PATH/bin/mpirun -np 20 ./cloverleaf_mpi_opencl OPS_CL_DEVICE=0  > perf_out
$MPI_INSTALL_PATH/bin/mpirun -np 20 ./cloverleaf_mpi_opencl OPS_CL_DEVICE=0  > perf_out
grep "Total Wall time" clover.out
grep "step:   2953" clover.out
grep "step:   2954" clover.out
grep "step:   2955" clover.out
rm perf_out
echo '============> Running MPI+OpenCL on GPU'
$MPI_INSTALL_PATH/bin/mpirun -np 2 ./cloverleaf_mpi_opencl OPS_CL_DEVICE=1 OPS_BLOCK_SIZE_X=32 OPS_BLOCK_SIZE_Y=4 > perf_out
$MPI_INSTALL_PATH/bin/mpirun -np 2 ./cloverleaf_mpi_opencl OPS_CL_DEVICE=1 OPS_BLOCK_SIZE_X=32 OPS_BLOCK_SIZE_Y=4 > perf_out
grep "Total Wall time" clover.out
grep "step:   2953" clover.out
grep "step:   2954" clover.out
grep "step:   2955" clover.out
rm perf_out


cd -
source ../source_pgi_15.1
make clean
make
cd -
make


#============================ Test Cloverleaf 2D With PGI Compilers==========================================================
echo '============> Running OpenMP'
KMP_AFFINITY=compact OMP_NUM_THREADS=20 ./cloverleaf_openmp > perf_out
grep "Total Wall time" clover.out
grep "step:   2955" clover.out
echo '============> Running MPI+OpenMP'
export OMP_NUM_THREADS=2;$MPI_INSTALL_PATH/bin/mpirun -np 10 ./cloverleaf_mpi_openmp > perf_out
grep "Total Wall time" clover.out
grep "step:   2955" clover.out
echo '============> Running DEV_MPI'
$MPI_INSTALL_PATH/bin/mpirun -np 20 ./cloverleaf_dev_mpi > perf_out
grep "Total Wall time" clover.out
grep "step:   2955" clover.out
echo '============> Running MPI'
$MPI_INSTALL_PATH/bin/mpirun -np 20 ./cloverleaf_mpi > perf_out
grep "Total Wall time" clover.out
grep "step:   2955" clover.out
echo '============> Running CUDA'
./cloverleaf_cuda OPS_BLOCK_SIZE_X=64 OPS_BLOCK_SIZE_Y=4 > perf_out
grep "Total Wall time" clover.out
grep "step:   2953" clover.out
echo '============> Running MPI+CUDA'
$MPI_INSTALL_PATH/bin/mpirun -np 2 ./cloverleaf_mpi_cuda OPS_BLOCK_SIZE_X=64 OPS_BLOCK_SIZE_Y=4 > perf_out
grep "Total Wall time" clover.out
grep "step:   2953" clover.out
echo '============> Running MPI+CUDA with GPU-Direct'
MV2_USE_CUDA=1 $MPI_INSTALL_PATH/bin/mpirun -np 2 ./cloverleaf_mpi_cuda -gpudirect OPS_BLOCK_SIZE_X=64 OPS_BLOCK_SIZE_Y=4 > perf_out
grep "Total Wall time" clover.out
grep "step:   2953" clover.out
echo '============> Running OpenCL on CPU'
./cloverleaf_opencl OPS_CL_DEVICE=0 OPS_BLOCK_SIZE_X=512 OPS_BLOCK_SIZE_Y=1 > perf_out
grep "Total Wall time" clover.out
grep "step:   2955" clover.out
echo '============> Running OpenCL on GPU'
./cloverleaf_opencl OPS_CL_DEVICE=1 OPS_BLOCK_SIZE_X=32 OPS_BLOCK_SIZE_Y=4 > perf_out
./cloverleaf_opencl OPS_CL_DEVICE=1 OPS_BLOCK_SIZE_X=32 OPS_BLOCK_SIZE_Y=4 > perf_out
grep "Total Wall time" clover.out
grep "step:   2953" clover.out
grep "step:   2954" clover.out
grep "step:   2955" clover.out
rm perf_out
echo '============> Running MPI+OpenCL on CPU'
$MPI_INSTALL_PATH/bin/mpirun -np 20 ./cloverleaf_mpi_opencl OPS_CL_DEVICE=0  > perf_out
$MPI_INSTALL_PATH/bin/mpirun -np 20 ./cloverleaf_mpi_opencl OPS_CL_DEVICE=0  > perf_out
grep "Total Wall time" clover.out
grep "step:   2953" clover.out
grep "step:   2954" clover.out
grep "step:   2955" clover.out
rm perf_out
echo '============> Running MPI+OpenCL on GPU'
$MPI_INSTALL_PATH/bin/mpirun -np 2 ./cloverleaf_mpi_opencl OPS_CL_DEVICE=1 OPS_BLOCK_SIZE_X=32 OPS_BLOCK_SIZE_Y=4 > perf_out
$MPI_INSTALL_PATH/bin/mpirun -np 2 ./cloverleaf_mpi_opencl OPS_CL_DEVICE=1 OPS_BLOCK_SIZE_X=32 OPS_BLOCK_SIZE_Y=4 > perf_out
grep "Total Wall time" clover.out
grep "step:   2953" clover.out
grep "step:   2954" clover.out
grep "step:   2955" clover.out
rm perf_out

cd -
source ../source_pgi_15.1
make clean
make
cd -
make cloverleaf_openacc -j20
make cloverleaf_mpi_openacc -j20
echo '============> Running OpenACC'
./cloverleaf_openacc OPS_BLOCK_SIZE_X=64 OPS_BLOCK_SIZE_Y=4 > perf_out
grep "Total Wall time" clover.out
grep "step:   2952" clover.out
grep "step:   2953" clover.out
grep "step:   2954" clover.out
grep "step:   2955" clover.out
rm perf_out
echo '============> Running MPI+OpenACC'
$MPI_INSTALL_PATH/bin/mpirun -np 2 ./cloverleaf_mpi_openacc OPS_BLOCK_SIZE_X=64 OPS_BLOCK_SIZE_Y=4 > perf_out
grep "Total Wall time" clover.out
grep "step:   2952" clover.out
grep "step:   2953" clover.out
grep "step:   2954" clover.out
grep "step:   2955" clover.out
rm perf_out
