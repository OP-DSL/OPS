#!/bin/bash
set -e
cd ../../../ops/c
source ../../scripts/source_intel
make
cd -
make clean
rm -f .generated
make IEEE=1

#============================ Test Cloverleaf 2D With Intel Compilers==========================================================
#<<COMMENT
echo '============> Running OpenMP'
KMP_AFFINITY=compact OMP_NUM_THREADS=20 ./cloverleaf_openmp > perf_out
grep "Total Wall time" clover.out
#grep -e "step:   2952" -e "step:   2953" -e "step:   2954" -e "step:   2955" clover.out
grep "PASSED" clover.out
rc=$?; if [[ $rc != 0 ]]; then echo "TEST FAILED";exit $rc; fi
rm -f clover.out

echo '============> Running OpenMP with Tiling'
KMP_AFFINITY=compact OMP_NUM_THREADS=20 ./cloverleaf_tiled OPS_TILING > perf_out
grep "Total Wall time" clover.out
#grep -e "step:   2952" -e "step:   2953" -e "step:   2954" -e "step:   2955" clover.out
grep "PASSED" clover.out
rc=$?; if [[ $rc != 0 ]]; then echo "TEST FAILED";exit $rc; fi
rm -f clover.out

echo '============> Running MPI+OpenMP'
export OMP_NUM_THREADS=2;$MPI_INSTALL_PATH/bin/mpirun -np 10 ./cloverleaf_mpi_openmp > perf_out
grep "Total Wall time" clover.out
#grep -e "step:   2952" -e "step:   2953" -e "step:   2954" -e "step:   2955" clover.out
grep "PASSED" clover.out
rc=$?; if [[ $rc != 0 ]]; then echo "TEST FAILED";exit $rc; fi
rm -f clover.out

echo '============> Running MPI_Inline with MPI+OpenMP'
export OMP_NUM_THREADS=2;$MPI_INSTALL_PATH/bin/mpirun -np 10 ./cloverleaf_mpi_inline > perf_out
grep "Total Wall time" clover.out
#grep -e "step:   2952" -e "step:   2953" -e "step:   2954" -e "step:   2955" clover.out
grep "PASSED" clover.out
rc=$?; if [[ $rc != 0 ]]; then echo "TEST FAILED";exit $rc; fi
rm -f clover.out

echo '============> Running DEV_MPI'
$MPI_INSTALL_PATH/bin/mpirun -np 20 ./cloverleaf_dev_mpi > perf_out
grep "Total Wall time" clover.out
#grep -e "step:   2952" -e "step:   2953" -e "step:   2954" -e "step:   2955" clover.out
grep "PASSED" clover.out
rc=$?; if [[ $rc != 0 ]]; then echo "TEST FAILED";exit $rc; fi
rm -f clover.out

echo '============> Running MPI'
$MPI_INSTALL_PATH/bin/mpirun -np 20 ./cloverleaf_mpi > perf_out
grep "Total Wall time" clover.out
#grep -e "step:   2952" -e "step:   2953" -e "step:   2954" -e "step:   2955" clover.out
grep "PASSED" clover.out
rc=$?; if [[ $rc != 0 ]]; then echo "TEST FAILED";exit $rc; fi
rm -f clover.out

echo '============> Running MPI_Inline'
export OMP_NUM_THREADS=1;$MPI_INSTALL_PATH/bin/mpirun -np 20 ./cloverleaf_mpi_inline > perf_out
grep "Total Wall time" clover.out
#grep -e "step:   2952" -e "step:   2953" -e "step:   2954" -e "step:   2955" clover.out
grep "PASSED" clover.out
rc=$?; if [[ $rc != 0 ]]; then echo "TEST FAILED";exit $rc; fi
rm -f clover.out

echo '============> Running MPI_Tiled'
export OMP_NUM_THREADS=10;$MPI_INSTALL_PATH/bin/mpirun -np 2 ./cloverleaf_mpi_tiled OPS_TILING OPS_TILING_MAXDEPTH=6 > perf_out
grep "Total Wall time" clover.out
#grep -e "step:   2952" -e "step:   2953" -e "step:   2954" -e "step:   2955" clover.out
grep "PASSED" clover.out
rc=$?; if [[ $rc != 0 ]]; then echo "TEST FAILED";exit $rc; fi
rm -f clover.out

echo '============> Running CUDA'
./cloverleaf_cuda OPS_BLOCK_SIZE_X=64 OPS_BLOCK_SIZE_Y=4 > perf_out
grep "Total Wall time" clover.out
#grep -e "step:   2952" -e "step:   2953" -e "step:   2954" -e "step:   2955" clover.out
grep "PASSED" clover.out
rc=$?; if [[ $rc != 0 ]]; then echo "TEST FAILED";exit $rc; fi
rm -f clover.out

echo '============> Running MPI+CUDA'
$MPI_INSTALL_PATH/bin/mpirun -np 2 ./cloverleaf_mpi_cuda OPS_BLOCK_SIZE_X=64 OPS_BLOCK_SIZE_Y=4 > perf_out
grep "Total Wall time" clover.out
#grep -e "step:   2952" -e "step:   2953" -e "step:   2954" -e "step:   2955" clover.out
grep "PASSED" clover.out
rc=$?; if [[ $rc != 0 ]]; then echo "TEST FAILED";exit $rc; fi
rm -f clover.out

#echo '============> Running MPI+CUDA with GPU-Direct'
#MV2_USE_CUDA=1 $MPI_INSTALL_PATH/bin/mpirun -np 2 ./cloverleaf_mpi_cuda -gpudirect OPS_BLOCK_SIZE_X=64 OPS_BLOCK_SIZE_Y=4 > perf_out
#grep "Total Wall time" clover.out
#grep -e "step:   2952" -e "step:   2953" -e "step:   2954" -e "step:   2955" clover.out
#grep "PASSED" clover.out
#rc=$?; if [[ $rc != 0 ]]; then echo "TEST FAILED";exit $rc; fi
#rm -f clover.out

echo '============> Running OpenCL on CPU'
./cloverleaf_opencl OPS_CL_DEVICE=0 OPS_BLOCK_SIZE_X=512 OPS_BLOCK_SIZE_Y=1 > perf_out
grep "Total Wall time" clover.out
#grep -e "step:   2952" -e "step:   2953" -e "step:   2954" -e "step:   2955" clover.out
grep "PASSED" clover.out
rc=$?; if [[ $rc != 0 ]]; then echo "TEST FAILED";exit $rc; fi
rm -f clover.out
rm perf_out


echo '============> Running OpenCL on GPU'
./cloverleaf_opencl OPS_CL_DEVICE=1 OPS_BLOCK_SIZE_X=32 OPS_BLOCK_SIZE_Y=4 > perf_out
./cloverleaf_opencl OPS_CL_DEVICE=1 OPS_BLOCK_SIZE_X=32 OPS_BLOCK_SIZE_Y=4 > perf_out
grep "Total Wall time" clover.out
#grep -e "step:   2952" -e "step:   2953" -e "step:   2954" -e "step:   2955" clover.out
grep "PASSED" clover.out
rc=$?; if [[ $rc != 0 ]]; then echo "TEST FAILED";exit $rc; fi
rm -f clover.out
rm perf_out

echo '============> Running MPI+OpenCL on CPU'
$MPI_INSTALL_PATH/bin/mpirun -np 20 ./cloverleaf_mpi_opencl OPS_CL_DEVICE=0  > perf_out
$MPI_INSTALL_PATH/bin/mpirun -np 20 ./cloverleaf_mpi_opencl OPS_CL_DEVICE=0  > perf_out
grep "Total Wall time" clover.out
#grep -e "step:   2952" -e "step:   2953" -e "step:   2954" -e "step:   2955" clover.out
grep "PASSED" clover.out
rc=$?; if [[ $rc != 0 ]]; then echo "TEST FAILED";exit $rc; fi
rm -f clover.out
rm perf_out

echo '============> Running MPI+OpenCL on GPU'
$MPI_INSTALL_PATH/bin/mpirun -np 2 ./cloverleaf_mpi_opencl OPS_CL_DEVICE=1 OPS_BLOCK_SIZE_X=32 OPS_BLOCK_SIZE_Y=4 > perf_out
$MPI_INSTALL_PATH/bin/mpirun -np 2 ./cloverleaf_mpi_opencl OPS_CL_DEVICE=1 OPS_BLOCK_SIZE_X=32 OPS_BLOCK_SIZE_Y=4 > perf_out
grep "Total Wall time" clover.out
#grep -e "step:   2952" -e "step:   2953" -e "step:   2954" -e "step:   2955" clover.out
grep "PASSED" clover.out
rc=$?; if [[ $rc != 0 ]]; then echo "TEST FAILED";exit $rc; fi
rm -f clover.out
rm perf_out

echo "All Intel complied applications PASSED : Exiting Test Script "
#exit

#COMMENT
cd -
source ../../scripts/source_pgi_15.10

make clean
make
cd -
make


#============================ Test Cloverleaf 2D With PGI Compilers==========================================================
echo '============> Running OpenMP'
KMP_AFFINITY=compact OMP_NUM_THREADS=20 ./cloverleaf_openmp > perf_out
grep "Total Wall time" clover.out
#grep -e "step:   2952" -e "step:   2953" -e "step:   2954" -e "step:   2955" clover.out
grep "PASSED" clover.out
rc=$?; if [[ $rc != 0 ]]; then echo "TEST FAILED";exit $rc; fi
rm -f clover.out

echo '============> Running MPI+OpenMP'
export OMP_NUM_THREADS=2;$MPI_INSTALL_PATH/bin/mpirun -np 10 ./cloverleaf_mpi_openmp > perf_out
grep "Total Wall time" clover.out
#grep -e "step:   2952" -e "step:   2953" -e "step:   2954" -e "step:   2955" clover.out
grep "PASSED" clover.out
rc=$?; if [[ $rc != 0 ]]; then echo "TEST FAILED";exit $rc; fi
rm -f clover.out

echo '============> Running DEV_MPI'
$MPI_INSTALL_PATH/bin/mpirun -np 20 ./cloverleaf_dev_mpi > perf_out
grep "Total Wall time" clover.out
#grep -e "step:   2952" -e "step:   2953" -e "step:   2954" -e "step:   2955" clover.out
grep "PASSED" clover.out
rc=$?; if [[ $rc != 0 ]]; then echo "TEST FAILED";exit $rc; fi
rm -f clover.out

echo '============> Running MPI'
$MPI_INSTALL_PATH/bin/mpirun -np 20 ./cloverleaf_mpi > perf_out
grep "Total Wall time" clover.out
#grep -e "step:   2952" -e "step:   2953" -e "step:   2954" -e "step:   2955" clover.out
grep "PASSED" clover.out
rc=$?; if [[ $rc != 0 ]]; then echo "TEST FAILED";exit $rc; fi
rm -f clover.out

echo '============> Running MPI_Tiled'
export OMP_NUM_THREADS=10;$MPI_INSTALL_PATH/bin/mpirun -np 2 ./cloverleaf_mpi_tiled OPS_TILING OPS_TILING_MAXDEPTH=6 > perf_out
grep "Total Wall time" clover.out
#grep -e "step:   2952" -e "step:   2953" -e "step:   2954" -e "step:   2955" clover.out
grep "PASSED" clover.out
rc=$?; if [[ $rc != 0 ]]; then echo "TEST FAILED";exit $rc; fi
rm -f clover.out


echo '============> Running CUDA'
./cloverleaf_cuda OPS_BLOCK_SIZE_X=64 OPS_BLOCK_SIZE_Y=4 > perf_out
grep "Total Wall time" clover.out
#grep -e "step:   2952" -e "step:   2953" -e "step:   2954" -e "step:   2955" clover.out
grep "PASSED" clover.out
rc=$?; if [[ $rc != 0 ]]; then echo "TEST FAILED";exit $rc; fi
rm -f clover.out

echo '============> Running MPI+CUDA'
$MPI_INSTALL_PATH/bin/mpirun -np 2 ./cloverleaf_mpi_cuda OPS_BLOCK_SIZE_X=64 OPS_BLOCK_SIZE_Y=4 > perf_out
grep "Total Wall time" clover.out
#grep -e "step:   2952" -e "step:   2953" -e "step:   2954" -e "step:   2955" clover.out
grep "PASSED" clover.out
rc=$?; if [[ $rc != 0 ]]; then echo "TEST FAILED";exit $rc; fi
rm -f clover.out

#echo '============> Running MPI+CUDA with GPU-Direct'
#MV2_USE_CUDA=1 $MPI_INSTALL_PATH/bin/mpirun -np 2 ./cloverleaf_mpi_cuda -gpudirect OPS_BLOCK_SIZE_X=64 OPS_BLOCK_SIZE_Y=4 > perf_out
#grep "Total Wall time" clover.out
#grep -e "step:   2952" -e "step:   2953" -e "step:   2954" -e "step:   2955" clover.out
#grep "PASSED" clover.out
#rc=$?; if [[ $rc != 0 ]]; then echo "TEST FAILED";exit $rc; fi
#rm -f clover.out


#echo '============> Running OpenCL on CPU'
#./cloverleaf_opencl OPS_CL_DEVICE=0 OPS_BLOCK_SIZE_X=512 OPS_BLOCK_SIZE_Y=1 > perf_out
#grep "Total Wall time" clover.out
#grep -e "step:   2952" -e "step:   2953" -e "step:   2954" -e "step:   2955" clover.out
#grep "PASSED" clover.out
#rc=$?; if [[ $rc != 0 ]]; then echo "TEST FAILED";exit $rc; fi
#rm -f clover.out
#rm perf_out

echo '============> Running OpenCL on GPU'
./cloverleaf_opencl OPS_CL_DEVICE=1 OPS_BLOCK_SIZE_X=32 OPS_BLOCK_SIZE_Y=4 > perf_out
./cloverleaf_opencl OPS_CL_DEVICE=1 OPS_BLOCK_SIZE_X=32 OPS_BLOCK_SIZE_Y=4 > perf_out
grep "Total Wall time" clover.out
#grep -e "step:   2952" -e "step:   2953" -e "step:   2954" -e "step:   2955" clover.out
grep "PASSED" clover.out
rc=$?; if [[ $rc != 0 ]]; then echo "TEST FAILED";exit $rc; fi
rm -f clover.out
rm perf_out

#echo '============> Running MPI+OpenCL on CPU'
#$MPI_INSTALL_PATH/bin/mpirun -np 20 ./cloverleaf_mpi_opencl OPS_CL_DEVICE=0  > perf_out
#$MPI_INSTALL_PATH/bin/mpirun -np 20 ./cloverleaf_mpi_opencl OPS_CL_DEVICE=0  > perf_out
#grep "Total Wall time" clover.out
#grep -e "step:   2952" -e "step:   2953" -e "step:   2954" -e "step:   2955" clover.out
#rm perf_out

echo '============> Running MPI+OpenCL on GPU'
$MPI_INSTALL_PATH/bin/mpirun -np 2 ./cloverleaf_mpi_opencl OPS_CL_DEVICE=1 OPS_BLOCK_SIZE_X=32 OPS_BLOCK_SIZE_Y=4 > perf_out
$MPI_INSTALL_PATH/bin/mpirun -np 2 ./cloverleaf_mpi_opencl OPS_CL_DEVICE=1 OPS_BLOCK_SIZE_X=32 OPS_BLOCK_SIZE_Y=4 > perf_out
grep "Total Wall time" clover.out
#grep -e "step:   2952" -e "step:   2953" -e "step:   2954" -e "step:   2955" clover.out
grep "PASSED" clover.out
rc=$?; if [[ $rc != 0 ]]; then echo "TEST FAILED";exit $rc; fi
rm -f clover.out
rm perf_out

echo '============> Running OpenACC'
./cloverleaf_openacc OPS_BLOCK_SIZE_X=64 OPS_BLOCK_SIZE_Y=4 > perf_out
grep "Total Wall time" clover.out
#grep -e "step:   2952" -e "step:   2953" -e "step:   2954" -e "step:   2955" clover.out
grep "PASSED" clover.out
rc=$?; if [[ $rc != 0 ]]; then echo "TEST FAILED";exit $rc; fi
rm -f clover.out
rm perf_out

echo '============> Running MPI+OpenACC'
$MPI_INSTALL_PATH/bin/mpirun -np 2 ./cloverleaf_mpi_openacc OPS_BLOCK_SIZE_X=64 OPS_BLOCK_SIZE_Y=4 > perf_out
grep "Total Wall time" clover.out
#grep -e "step:   2952" -e "step:   2953" -e "step:   2954" -e "step:   2955" clover.out
grep "PASSED" clover.out
rc=$?; if [[ $rc != 0 ]]; then echo "TEST FAILED";exit $rc; fi
rm -f clover.out
rm perf_out

echo "All PGI complied applications PASSED : Exiting Test Script "
