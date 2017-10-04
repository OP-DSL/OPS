#!/bin/bash
set -e
cd ../../../../ops/c
source ../../scripts/source_intel
make
cd -
#../../../../translator/python/c/ops.py shsgc.cpp
make

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

echo '============> Running OpenCL on CPU'
./shsgc_opencl OPS_CL_DEVICE=0 OPS_BLOCK_SIZE_X=512 OPS_BLOCK_SIZE_Y=1 > perf_out
grep "Pre shock error is:" perf_out
grep "Post shock error is:" perf_out
grep "Post shock Error is" perf_out
grep "Total Wall time" perf_out
grep -e "acceptable" -e "correct"  perf_out
rc=$?; if [[ $rc != 0 ]]; then echo "TEST FAILED";exit $rc; fi
rm perf_out

echo '============> Running OpenCL on GPU'
./shsgc_opencl OPS_CL_DEVICE=1 OPS_BLOCK_SIZE_X=32 OPS_BLOCK_SIZE_Y=4 > perf_out
./shsgc_opencl OPS_CL_DEVICE=1 OPS_BLOCK_SIZE_X=32 OPS_BLOCK_SIZE_Y=4 > perf_out
grep "Pre shock error is:" perf_out
grep "Post shock error is:" perf_out
grep "Post shock Error is" perf_out
grep "Total Wall time" perf_out
grep -e "acceptable" -e "correct"  perf_out
rc=$?; if [[ $rc != 0 ]]; then echo "TEST FAILED";exit $rc; fi
rm perf_out

echo '============> Running MPI+OpenCL on CPU'
$MPI_INSTALL_PATH/bin/mpirun -np 20 ./shsgc_mpi_opencl OPS_CL_DEVICE=0 OPS_BLOCK_SIZE_X=256 OPS_BLOCK_SIZE_Y=1 > perf_out
$MPI_INSTALL_PATH/bin/mpirun -np 20 ./shsgc_mpi_opencl OPS_CL_DEVICE=0 OPS_BLOCK_SIZE_X=256 OPS_BLOCK_SIZE_Y=1 > perf_out
grep "Pre shock error is:" perf_out
grep "Post shock error is:" perf_out
grep "Post shock Error is" perf_out
grep "Total Wall time" perf_out
grep -e "acceptable" -e "correct"  perf_out
rc=$?; if [[ $rc != 0 ]]; then echo "TEST FAILED";exit $rc; fi
rm perf_out

echo '============> Running MPI+OpenCL on GPU'
$MPI_INSTALL_PATH/bin/mpirun -np 2 ./shsgc_mpi_opencl OPS_CL_DEVICE=1 OPS_BLOCK_SIZE_X=32 OPS_BLOCK_SIZE_Y=4 > perf_out
$MPI_INSTALL_PATH/bin/mpirun -np 2 ./shsgc_mpi_opencl OPS_CL_DEVICE=1 OPS_BLOCK_SIZE_X=32 OPS_BLOCK_SIZE_Y=4 > perf_out
grep "Pre shock error is:" perf_out
grep "Post shock error is:" perf_out
grep "Post shock Error is" perf_out
grep "Total Wall time" perf_out
grep -e "acceptable" -e "correct"  perf_out
rc=$?; if [[ $rc != 0 ]]; then echo "TEST FAILED";exit $rc; fi
rm perf_out

cd -
source ../../scripts/source_pgi_15.10

make clean
make
cd -
make

#============================ Test SHSGC with PGI Compilers==========================================================
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

#echo '============> Running OpenCL on CPU'
#./shsgc_opencl OPS_CL_DEVICE=0 OPS_BLOCK_SIZE_X=512 OPS_BLOCK_SIZE_Y=1 > perf_out
#grep "Pre shock error is:" perf_out
#grep "Post shock error is:" perf_out
#grep "Post shock Error is" perf_out
#grep "Total Wall time" perf_out
#grep -e "acceptable" -e "correct"  perf_out
#rc=$?; if [[ $rc != 0 ]]; then echo "TEST FAILED";exit $rc; fi
#rm perf_out

echo '============> Running OpenCL on GPU'
./shsgc_opencl OPS_CL_DEVICE=1 OPS_BLOCK_SIZE_X=32 OPS_BLOCK_SIZE_Y=4 > perf_out
./shsgc_opencl OPS_CL_DEVICE=1 OPS_BLOCK_SIZE_X=32 OPS_BLOCK_SIZE_Y=4 > perf_out
grep "Pre shock error is:" perf_out
grep "Post shock error is:" perf_out
grep "Post shock Error is" perf_out
grep "Total Wall time" perf_out
grep -e "acceptable" -e "correct"  perf_out
rc=$?; if [[ $rc != 0 ]]; then echo "TEST FAILED";exit $rc; fi
rm perf_out

#echo '============> Running MPI+OpenCL on CPU'
#$MPI_INSTALL_PATH/bin/mpirun -np 20 ./shsgc_mpi_opencl OPS_CL_DEVICE=0 OPS_BLOCK_SIZE_X=256 OPS_BLOCK_SIZE_Y=1 > perf_out
#$MPI_INSTALL_PATH/bin/mpirun -np 20 ./shsgc_mpi_opencl OPS_CL_DEVICE=0 OPS_BLOCK_SIZE_X=256 OPS_BLOCK_SIZE_Y=1 > perf_out
#grep "Pre shock error is:" perf_out
#grep "Post shock error is:" perf_out
#grep "Post shock Error is" perf_out
#grep "Total Wall time" perf_out
#grep -e "acceptable" -e "correct"  perf_out
#rc=$?; if [[ $rc != 0 ]]; then echo "TEST FAILED";exit $rc; fi
#rm perf_out

echo '============> Running MPI+OpenCL on GPU'
$MPI_INSTALL_PATH/bin/mpirun -np 2 ./shsgc_mpi_opencl OPS_CL_DEVICE=1 OPS_BLOCK_SIZE_X=32 OPS_BLOCK_SIZE_Y=4 > perf_out
$MPI_INSTALL_PATH/bin/mpirun -np 2 ./shsgc_mpi_opencl OPS_CL_DEVICE=1 OPS_BLOCK_SIZE_X=32 OPS_BLOCK_SIZE_Y=4 > perf_out
grep "Pre shock error is:" perf_out
grep "Post shock error is:" perf_out
grep "Post shock Error is" perf_out
grep "Total Wall time" perf_out
grep -e "acceptable" -e "correct"  perf_out
rc=$?; if [[ $rc != 0 ]]; then echo "TEST FAILED";exit $rc; fi

echo '============> Running OpenACC'
./shsgc_openacc OPS_BLOCK_SIZE_X=64 OPS_BLOCK_SIZE_Y=4 > perf_out
grep "Pre shock error is:" perf_out
grep "Post shock error is:" perf_out
grep "Post shock Error is" perf_out
grep "Total Wall time" perf_out
grep -e "acceptable" -e "correct"  perf_out
rc=$?; if [[ $rc != 0 ]]; then echo "TEST FAILED";exit $rc; fi
rm perf_out

echo '============> Running MPI+OpenACC'
$MPI_INSTALL_PATH/bin/mpirun -np 2 ./shsgc_mpi_openacc OPS_BLOCK_SIZE_X=64 OPS_BLOCK_SIZE_Y=4 > perf_out
grep "Pre shock error is:" perf_out
grep "Post shock error is:" perf_out
grep "Post shock Error is" perf_out
grep "Total Wall time" perf_out
grep -e "acceptable" -e "correct"  perf_out
rc=$?; if [[ $rc != 0 ]]; then echo "TEST FAILED";exit $rc; fi
rm perf_out
