#!/bin/bash
set -e
cd ../../../ops/c
#<<COMMENT
source ../../scripts/source_intel
make
cd -

../../../ops_translator/c/ops.py complex_numbers.cpp

make clean
make

#============================ Test Complex Numbers 2D With Intel Compilers==========================================================
echo '============> Running OpenMP'
KMP_AFFINITY=compact OMP_NUM_THREADS=20 ./complex_numbers_openmp > perf_out
grep "Total Wall time" perf_out
rm -f perf_out

echo '============> Running OpenMP with Tiling'
KMP_AFFINITY=compact OMP_NUM_THREADS=20 ./complex_numbers_tiled OPS_TILING > perf_out
grep "Total Wall time" perf_out
rm -f perf_out

echo '============> Running MPI+OpenMP'
export OMP_NUM_THREADS=2;$MPI_INSTALL_PATH/bin/mpirun -np 10 ./complex_numbers_mpi_openmp > perf_out
grep "Total Wall time" perf_out
rm -f perf_out

echo '============> Running MPI_Inline with MPI+OpenMP'
export OMP_NUM_THREADS=2;$MPI_INSTALL_PATH/bin/mpirun -np 10 ./complex_numbers_mpi_inline > perf_out
grep "Total Wall time" perf_out
rm -f perf_out

echo '============> Running DEV_MPI'
$MPI_INSTALL_PATH/bin/mpirun -np 20 ./complex_numbers_dev_mpi > perf_out
grep "Total Wall time" perf_out
rm -f perf_out

echo '============> Running MPI'
$MPI_INSTALL_PATH/bin/mpirun -np 20 ./complex_numbers_mpi > perf_out
grep "Total Wall time" perf_out
rm -f perf_out

echo '============> Running MPI_Inline'
export OMP_NUM_THREADS=1;$MPI_INSTALL_PATH/bin/mpirun -np 20 ./complex_numbers_mpi_inline > perf_out
grep "Total Wall time" perf_out
rm -f perf_out


echo '============> Running CUDA'
./complex_numbers_cuda OPS_BLOCK_SIZE_X=64 OPS_BLOCK_SIZE_Y=4 > perf_out
grep "Total Wall time" perf_out
rm -f perf_out

echo '============> Running MPI+CUDA'
$MPI_INSTALL_PATH/bin/mpirun -np 2 ./complex_numbers_mpi_cuda OPS_BLOCK_SIZE_X=64 OPS_BLOCK_SIZE_Y=4 > perf_out
grep "Total Wall time" perf_out
rm -f perf_out

#echo '============> Running MPI+CUDA with GPU-Direct'
#MV2_USE_CUDA=1 $MPI_INSTALL_PATH/bin/mpirun -np 2 ./complex_numbers_mpi_cuda -gpudirect OPS_BLOCK_SIZE_X=64 OPS_BLOCK_SIZE_Y=4 > perf_out
#grep "Total Wall time" perf_out
####rm -f perf_out

echo '============> Running OpenCL on CPU'
./complex_numbers_opencl OPS_CL_DEVICE=0 OPS_BLOCK_SIZE_X=512 OPS_BLOCK_SIZE_Y=1 > perf_out
grep "Total Wall time" perf_out
rm -f perf_out

echo '============> Running OpenCL on GPU'
./complex_numbers_opencl OPS_CL_DEVICE=1 OPS_BLOCK_SIZE_X=32 OPS_BLOCK_SIZE_Y=4 > perf_out
./complex_numbers_opencl OPS_CL_DEVICE=1 OPS_BLOCK_SIZE_X=32 OPS_BLOCK_SIZE_Y=4 > perf_out
grep "Total Wall time" perf_out
rm -f perf_out

echo '============> Running MPI+OpenCL on CPU'
$MPI_INSTALL_PATH/bin/mpirun -np 20 ./complex_numbers_mpi_opencl OPS_CL_DEVICE=0 > perf_out
$MPI_INSTALL_PATH/bin/mpirun -np 20 ./complex_numbers_mpi_opencl OPS_CL_DEVICE=0 > perf_out
grep "Total Wall time" perf_out
rm -f perf_out

echo '============> Running MPI+OpenCL on GPU'
$MPI_INSTALL_PATH/bin/mpirun -np 2 ./complex_numbers_mpi_opencl OPS_CL_DEVICE=1 OPS_BLOCK_SIZE_X=32 OPS_BLOCK_SIZE_Y=4 > perf_out
$MPI_INSTALL_PATH/bin/mpirun -np 2 ./complex_numbers_mpi_opencl OPS_CL_DEVICE=1 OPS_BLOCK_SIZE_X=32 OPS_BLOCK_SIZE_Y=4 > perf_out
grep "Total Wall time" perf_out
rm -f perf_out

echo "All Intel complied applications PASSED : Exiting Test Script "
#exit

#COMMENT
cd -
source ../../scripts/source_pgi_16.9

make clean
make
cd -
make clean
make IEEE=1


#============================ Test Complex Numbers 2D With PGI Compilers==========================================================
echo '============> Running OpenMP'
KMP_AFFINITY=compact OMP_NUM_THREADS=20 ./complex_numbers_openmp > perf_out
grep "Total Wall time" perf_out
rm -f perf_out

echo '============> Running MPI+OpenMP'
export OMP_NUM_THREADS=2;$MPI_INSTALL_PATH/bin/mpirun -np 10 ./complex_numbers_mpi_openmp > perf_out
grep "Total Wall time" perf_out
rm -f perf_out

echo '============> Running DEV_MPI'
$MPI_INSTALL_PATH/bin/mpirun -np 20 ./complex_numbers_dev_mpi > perf_out
grep "Total Wall time" perf_out
rm -f perf_out

echo '============> Running MPI'
$MPI_INSTALL_PATH/bin/mpirun -np 20 ./complex_numbers_mpi > perf_out
grep "Total Wall time" perf_out
rm -f perf_out

echo '============> Running CUDA'
./complex_numbers_cuda OPS_BLOCK_SIZE_X=64 OPS_BLOCK_SIZE_Y=4 > perf_out
grep "Total Wall time" perf_out
rm -f perf_out

echo '============> Running MPI+CUDA'
$MPI_INSTALL_PATH/bin/mpirun -np 2 ./complex_numbers_mpi_cuda OPS_BLOCK_SIZE_X=64 OPS_BLOCK_SIZE_Y=4 > perf_out
grep "Total Wall time" perf_out
rm -f perf_out

#echo '============> Running MPI+CUDA with GPU-Direct'
#MV2_USE_CUDA=1 $MPI_INSTALL_PATH/bin/mpirun -np 2 ./complex_numbers_mpi_cuda -gpudirect OPS_BLOCK_SIZE_X=64 OPS_BLOCK_SIZE_Y=4 > perf_out
#grep "Total Wall time" perf_out
####rm -f perf_out


#echo '============> Running OpenCL on CPU'
#./complex_numbers_opencl OPS_CL_DEVICE=0 OPS_BLOCK_SIZE_X=512 OPS_BLOCK_SIZE_Y=1 > perf_out
#grep "Total Wall time" perf_out
####rm -f perf_out
#rm perf_out

echo '============> Running OpenCL on GPU'
./complex_numbers_opencl OPS_CL_DEVICE=1 OPS_BLOCK_SIZE_X=32 OPS_BLOCK_SIZE_Y=4 > perf_out
./complex_numbers_opencl OPS_CL_DEVICE=1 OPS_BLOCK_SIZE_X=32 OPS_BLOCK_SIZE_Y=4 > perf_out
grep "Total Wall time" perf_out
rm -f perf_out

#echo '============> Running MPI+OpenCL on CPU'
#$MPI_INSTALL_PATH/bin/mpirun -np 20 ./complex_numbers_mpi_opencl OPS_CL_DEVICE=0  > perf_out
#$MPI_INSTALL_PATH/bin/mpirun -np 20 ./complex_numbers_mpi_opencl OPS_CL_DEVICE=0  > perf_out
#grep "Total Wall time" perf_out
##rm perf_out

echo '============> Running MPI+OpenCL on GPU'
$MPI_INSTALL_PATH/bin/mpirun -np 2 ./complex_numbers_mpi_opencl OPS_CL_DEVICE=1 OPS_BLOCK_SIZE_X=32 OPS_BLOCK_SIZE_Y=4 > perf_out
$MPI_INSTALL_PATH/bin/mpirun -np 2 ./complex_numbers_mpi_opencl OPS_CL_DEVICE=1 OPS_BLOCK_SIZE_X=32 OPS_BLOCK_SIZE_Y=4 > perf_out
grep "Total Wall time" perf_out
rm -f perf_out

#echo '============> Running OpenACC'
#./complex_numbers_openacc OPS_BLOCK_SIZE_X=64 OPS_BLOCK_SIZE_Y=4 > perf_out
#grep "Total Wall time" perf_out
#rm -f perf_out

#echo '============> Running MPI+OpenACC'
#$MPI_INSTALL_PATH/bin/mpirun -np 2 ./complex_numbers_mpi_openacc OPS_BLOCK_SIZE_X=64 OPS_BLOCK_SIZE_Y=4 > perf_out
#grep "Total Wall time" perf_out
#rm -f perf_out

echo "All PGI complied applications PASSED : Exiting Test Script "
