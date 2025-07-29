#!/bin/bash
set -e
cd ../../../ops/c

export SOURCE_INTEL=source_intel_2021.3_pythonenv
export SOURCE_PGI=source_pgi_nvhpc_23_pythonenv
export SOURCE_INTEL_SYCL=source_intel_2021.3_sycl_pythonenv
export SOURCE_AMD_HIP=source_amd_rocm-5.4.3_pythonenv

#export AMOS=TRUE
#export DMOS=TRUE
export TELOS=TRUE
#export KOS=TRUE

if [[ -v TELOS || -v KOS ]]; then

#============================ Test Complex Numbers 2D With Intel Compilers==========================================================

cd $OPS_INSTALL_PATH/c
source ../../scripts/$SOURCE_INTEL
#make -j -B
make clean
make IEEE=1 

cd  $OPS_INSTALL_PATH/../apps/c/complex_numbers/
make clean
make IEEE=1

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

fi


if [[ -v TELOS ]]; then

#============================ Test with Intel SYCL Compilers==========================================
echo "Testing Intel SYCL complier based applications ---- "
cd $OPS_INSTALL_PATH/c
source ../../scripts/$SOURCE_INTEL_SYCL
#make -j -B
make clean
make
cd $OPS_INSTALL_PATH/../apps/c/complex_numbers

make clean
#make IEEE=1 -j
make IEEE=1 complex_numbers_sycl complex_numbers_mpi_sycl complex_numbers_mpi_sycl_tiled

echo '============> Running SYCL on CPU'
./complex_numbers_sycl OPS_CL_DEVICE=0 OPS_BLOCK_SIZE_X=512 OPS_BLOCK_SIZE_Y=1 > perf_out
grep "Total Wall time" perf_out
grep "PASSED" perf_out
rc=$?; if [[ $rc != 0 ]]; then echo "TEST FAILED";exit $rc; fi
rm perf_out

echo '============> Running MPI+SYCL on CPU'
$MPI_INSTALL_PATH/bin/mpirun -np 10 ./complex_numbers_mpi_sycl OPS_CL_DEVICE=0 OPS_BLOCK_SIZE_X=256 OPS_BLOCK_SIZE_Y=1 > perf_out
grep "Total Wall time" perf_out
grep "PASSED" perf_out
rc=$?; if [[ $rc != 0 ]]; then echo "TEST FAILED";exit $rc; fi
rm perf_out

echo '============> Running MPI+SYCL Tiled on CPU'
$MPI_INSTALL_PATH/bin/mpirun -np 2 ./complex_numbers_mpi_sycl_tiled OPS_CL_DEVICE=0 OPS_TILING OPS_TILING_MAXDEPTH=6 OPS_BLOCK_SIZE_X=32 OPS_BLOCK_SIZE_Y=4 > perf_out
grep "Total Wall time" perf_out
grep "PASSED" perf_out
rc=$?; if [[ $rc != 0 ]]; then echo "TEST FAILED";exit $rc; fi
rm perf_out

echo "All Intel SYCL complier based applications ---- PASSED"

fi


if [[ -v TELOS ]]; then
#============================ Test Complex Numbers 2D With PGI Compilers==========================================================
cd $OPS_INSTALL_PATH/c
source ../../scripts/$SOURCE_PGI
make clean
#make -j
make
cd $OPS_INSTALL_PATH/../apps/c/complex_numbers
make clean
make IEEE=1

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
fi


if [[ -v AMOS ]]; then

echo "Testing AMD HIP complier based applications ---- "
cd $OPS_INSTALL_PATH/c
source ../../scripts/$SOURCE_AMD_HIP
#make -j -B
make clean
make
cd $OPS_INSTALL_PATH/../apps/c/complex_numbers

make clean
rm -f .generated
#make IEEE=1 -j
make IEEE=1 complex_numbers_hip complex_numbers_mpi_hip 

echo '============> Running HIP'
./complex_numbers_hip OPS_BLOCK_SIZE_X=64 OPS_BLOCK_SIZE_Y=4 > perf_out
grep "Total Wall time" perf_out
grep "PASSED" perf_out
rc=$?; if [[ $rc != 0 ]]; then echo "TEST FAILED";exit $rc; fi
rm perf_out

echo '============> Running MPI+HIP'
#mpirun --allow-run-as-root -np 2 ./cloverleaf_mpi_hip OPS_BLOCK_SIZE_X=64 OPS_BLOCK_SIZE_Y=4 > perf_out
mpirun -np 2 ./complex_numbers_mpi_hip OPS_BLOCK_SIZE_X=64 OPS_BLOCK_SIZE_Y=4 > perf_out
grep "Total Wall time" perf_out
grep "PASSED" perf_out
rc=$?; if [[ $rc != 0 ]]; then echo "TEST FAILED";exit $rc; fi
rm perf_out

echo "All AMD HIP complier based applications ---- PASSED"

fi

echo "---------- Exiting Test Script "

