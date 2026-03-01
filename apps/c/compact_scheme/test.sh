#!/bin/bash
set -e
cd $OPS_INSTALL_PATH/c

export SOURCE_INTEL=source_oneapi_sycl_pythonenv
export SOURCE_PGI=source_pgi_nvhpc_23_pythonenv
export SOURCE_INTEL_SYCL=source_oneapi_sycl_pythonenv
export SOURCE_AMD_HIP=source_amd_rocm-5.4.3_pythonenv

#export AMOS=TRUE
export DEMOS=TRUE
#export TELOS=TRUE
#export KOS=TRUE

if [[ -v TELOS || -v DEMOS || -v KOS ]]; then

#============================ Test with Intel Classic Compilers==========================================
echo "Testing Intel classic complier based applications ---- "
cd $OPS_INSTALL_PATH/c
source ../../scripts/$SOURCE_INTEL
#make -j -B
make clean
make

cd $OPS_INSTALL_PATH/../apps/c/compact_scheme
make clean
#make IEEE=1 -j
make IEEE=1 compact3d_dev_seq compact3d_dev_mpi compact3d_seq compact3d_tiled compact3d_openmp compact3d_mpi \
compact3d_mpi_tiled compact3d_mpi_openmp

rm -rf *.h5 diff_out

# set Relative Tolarance for solution check -- h5diff check only
export TOL="1.000E-14"

#============== Run refernace (OPS sequential) solution ========================
echo '============> Running Referance (OPS sequential) Solution compact3d_dev_seq'
pwd
./compact3d_dev_seq > perf_out
mv Compact3D.h5 Compact3d_dev_seq.h5
$HDF5_INSTALL_PATH/bin/h5diff -p $TOL Compact3d_dev_seq.h5 Compact3d_dev_seq.h5 >> diff_out
if [ -s ./diff_out ]
then
    echo "File not empty - Solution Not Valid";exit 1;
else
    echo "PASSED"
fi
grep "Successful exit from OPS!" perf_out
rm perf_out

echo '============> Running OpenMP'
KMP_AFFINITY=compact OMP_NUM_THREADS=20 ./compact3d_openmp > perf_out
mv Compact3D.h5 Compact3d_openmp.h5
$HDF5_INSTALL_PATH/bin/h5diff -p $TOL Compact3d_openmp.h5 Compact3d_dev_seq.h5 >> diff_out
if [ -s ./diff_out ]
then
    echo "File not empty - Solution Not Valid";exit 1;
else
    echo "PASSED"
fi
grep "Successful exit from OPS!" perf_out
rc=$?; if [[ $rc != 0 ]]; then echo "TEST FAILED";exit $rc; fi
rm perf_out

#echo '============> Running OpenMP with Tiling'
#KMP_AFFINITY=compact OMP_NUM_THREADS=10 ./compact3d_tiled OPS_TILING > perf_out
#mv Compact3D.h5 Compact3d_omp_tiled.h5
#$HDF5_INSTALL_PATH/bin/h5diff -p $TOL Compact3d_omp_tiled.h5 Compact3d_dev_seq.h5 >> diff_out
#if [ -s ./diff_out ]
#then
#    echo "File not empty - Solution Not Valid";exit 1;
#else
#    echo "PASSED"
#fi
#grep "Successful exit from OPS!" perf_out
#rc=$?; if [[ $rc != 0 ]]; then echo "TEST FAILED";exit $rc; fi
#rm perf_out

echo '============> Running MPI+OpenMP'
export OMP_NUM_THREADS=2;$MPI_INSTALL_PATH/bin/mpirun -np 10 ./compact3d_mpi_openmp > perf_out
mv Compact3D.h5 Compact3d_mpi_omp.h5
$HDF5_INSTALL_PATH/bin/h5diff -p $TOL Compact3d_mpi_omp.h5 Compact3d_dev_seq.h5 >> diff_out
if [ -s ./diff_out ]
then
    echo "File not empty - Solution Not Valid";exit 1;
else
    echo "PASSED"
fi
grep "Successful exit from OPS!" perf_out
rc=$?; if [[ $rc != 0 ]]; then echo "TEST FAILED";exit $rc; fi
rm perf_out

echo '============> Running DEV_MPI'
$MPI_INSTALL_PATH/bin/mpirun -np 20 ./compact3d_dev_mpi > perf_out
mv Compact3D.h5 Compact3d_dev_mpi.h5
$HDF5_INSTALL_PATH/bin/h5diff -p $TOL Compact3d_dev_mpi.h5 Compact3d_dev_seq.h5 >> diff_out
if [ -s ./diff_out ]
then
    echo "File not empty - Solution Not Valid";exit 1;
else
    echo "PASSED"
fi
grep "Successful exit from OPS!" perf_out
rc=$?; if [[ $rc != 0 ]]; then echo "TEST FAILED";exit $rc; fi
rm perf_out

echo '============> Running MPI'
$MPI_INSTALL_PATH/bin/mpirun -np 20 ./compact3d_mpi > perf_out
mv Compact3D.h5 Compact3d_mpi.h5
$HDF5_INSTALL_PATH/bin/h5diff -p $TOL Compact3d_mpi.h5 Compact3d_dev_seq.h5 >> diff_out
if [ -s ./diff_out ]
then
    echo "File not empty - Solution Not Valid";exit 1;
else
    echo "PASSED"
fi
grep "Successful exit from OPS!" perf_out
rc=$?; if [[ $rc != 0 ]]; then echo "TEST FAILED";exit $rc; fi
rm perf_out

#echo '============> Running MPI_Tiled'
#export OMP_NUM_THREADS=10;$MPI_INSTALL_PATH/bin/mpirun -np 2 numawrap2 ./compact3d_mpi_tiled OPS_TILING OPS_TILING_MAXDEPTH=10 > perf_out
#mv Compact3D.h5 Compact3d_mpi_tiled.h5
#$HDF5_INSTALL_PATH/bin/h5diff -p $TOL Compact3d_mpi_tiled.h5 Compact3d_dev_seq.h5 >> diff_out
#if [ -s ./diff_out ]
#then
#    echo "File not empty - Solution Not Valid";exit 1;
#else
#    echo "PASSED"
#fi
#grep "Successful exit from OPS!" perf_out
#rc=$?; if [[ $rc != 0 ]]; then echo "TEST FAILED";exit $rc; fi
#rm perf_out

if [[ -v CUDA_INSTALL_PATH ]]; then
make IEEE=1 compact3d_cuda compact3d_mpi_cuda compact3d_mpi_cuda_tiled

echo '============> Running CUDA'
./compact3d_cuda OPS_BLOCK_SIZE_X=64 OPS_BLOCK_SIZE_Y=4 > perf_out
mv Compact3D.h5 Compact3d_cuda.h5
$HDF5_INSTALL_PATH/bin/h5diff -p $TOL Compact3d_cuda.h5 Compact3d_dev_seq.h5 >> diff_out
if [ -s ./diff_out ]
then
    echo "File not empty - Solution Not Valid";exit 1;
else
    echo "PASSED"
fi
grep "Successful exit from OPS!" perf_out
rc=$?; if [[ $rc != 0 ]]; then echo "TEST FAILED";exit $rc; fi
rm perf_out

echo '============> Running MPI+CUDA'
$MPI_INSTALL_PATH/bin/mpirun -np 2 ./compact3d_mpi_cuda OPS_BLOCK_SIZE_X=64 OPS_BLOCK_SIZE_Y=4 > perf_out
mv Compact3D.h5 Compact3d_mpi_cuda.h5
$HDF5_INSTALL_PATH/bin/h5diff -p $TOL Compact3d_mpi_cuda.h5 Compact3d_dev_seq.h5 >> diff_out
if [ -s ./diff_out ]
then
    echo "File not empty - Solution Not Valid";exit 1;
else
    echo "PASSED"
fi
grep "Successful exit from OPS!" perf_out
rc=$?; if [[ $rc != 0 ]]; then echo "TEST FAILED";exit $rc; fi
rm perf_out

echo '============> Running MPI+CUDA Tiled'
$MPI_INSTALL_PATH/bin/mpirun -np 2 ./compact3d_mpi_cuda_tiled OPS_TILING OPS_TILING_MAXDEPTH=10 OPS_BLOCK_SIZE_X=64 OPS_BLOCK_SIZE_Y=4 > perf_out
mv Compact3D.h5 Compact3d_mpi_cuda_tiled.h5
$HDF5_INSTALL_PATH/bin/h5diff -p $TOL .h5 Compact3d_mpi_cuda_ti Compact3d_dev_seq.h5 >> diff_out
if [ -s ./diff_out ]
then
    echo "File not empty - Solution Not Valid";exit 1;
else
    echo "PASSED"
fi
grep "Successful exit from OPS!" perf_out
rc=$?; if [[ $rc != 0 ]]; then echo "TEST FAILED";exit $rc; fi
rm perf_out

fi
fi

echo "All Intel classic complier based applications ---- PASSED"


if [[ -v TELOS || -v DEMOS ]]; then

#============================ Test with Intel SYCL Compilers==========================================
echo "Testing Intel SYCL complier based applications ---- "
cd $OPS_INSTALL_PATH/c
source ../../scripts/$SOURCE_INTEL_SYCL
#make -j -B
#make clean
#make
cd $OPS_INSTALL_PATH/../apps/c/compact_scheme

make clean
#make IEEE=1 -j
make IEEE=1 compact3d_sycl compact3d_mpi_sycl compact3d_mpi_sycl_tiled

echo '============> Running SYCL on CPU'
./compact3d_sycl OPS_CL_DEVICE=0 OPS_BLOCK_SIZE_X=512 OPS_BLOCK_SIZE_Y=1 > perf_out
mv Compact3D.h5 Compact3d_sycl.h5
$HDF5_INSTALL_PATH/bin/h5diff -p $TOL Compact3d_sycl.h5 Compact3d_dev_seq.h5 >> diff_out
if [ -s ./diff_out ]
then
    echo "File not empty - Solution Not Valid";exit 1;
else
    echo "PASSED"
fi
grep "Successful exit from OPS!" perf_out
rc=$?; if [[ $rc != 0 ]]; then echo "TEST FAILED";exit $rc; fi
rm perf_out

echo '============> Running MPI+SYCL on CPU'
$MPI_INSTALL_PATH/bin/mpirun -np 20 ./compact3d_mpi_sycl OPS_CL_DEVICE=0 OPS_BLOCK_SIZE_X=256 OPS_BLOCK_SIZE_Y=1 > perf_out
mv Compact3D.h5 Compact3d_mpi_sycl.h5
$HDF5_INSTALL_PATH/bin/h5diff -p $TOL Compact3d_mpi_sycl.h5 Compact3d_dev_seq.h5 >> diff_out
if [ -s ./diff_out ]
then
    echo "File not empty - Solution Not Valid";exit 1;
else
    echo "PASSED"
fi
grep "Successful exit from OPS!" perf_out
rc=$?; if [[ $rc != 0 ]]; then echo "TEST FAILED";exit $rc; fi
rm perf_out

echo '============> Running MPI+SYCL Tiled on CPU'
$MPI_INSTALL_PATH/bin/mpirun -np 2 ./compact3d_mpi_sycl_tiled OPS_CL_DEVICE=1 OPS_BLOCK_SIZE_X=32 OPS_BLOCK_SIZE_Y=4 > perf_out
mv Compact3D.h5 Compact3d_mpi_sycl_tiled.h5
$HDF5_INSTALL_PATH/bin/h5diff -p $TOL .h5 Compact3d_mpi_sycl_ti Compact3d_dev_seq.h5 >> diff_out
if [ -s ./diff_out ]
then
    echo "File not empty - Solution Not Valid";exit 1;
else
    echo "PASSED"
fi
grep "Successful exit from OPS!" perf_out
rc=$?; if [[ $rc != 0 ]]; then echo "TEST FAILED";exit $rc; fi
rm perf_out

echo "All Intel SYCL complier based applications ---- PASSED"

fi

if [[ -v TELOS || -v DEMOS ]]; then

#============================ Test with PGI Compilers==========================================
echo "Testing PGI/NVHPC complier based applications ---- "
cd $OPS_INSTALL_PATH/c
source ../../scripts/$SOURCE_PGI
make clean
#make -j
#make
#
cd $OPS_INSTALL_PATH/../apps/c/compact_scheme
make clean
make IEEE=1 compact3d_dev_seq compact3d_dev_mpi compact3d_seq compact3d_tiled compact3d_openmp cocompact3d_mpi_openmp compact3d_ompoffload compact3d_mpi_ompoffload compact3d_mpi_ompoffload_tiled

echo '============> Running OpenMP'
KMP_AFFINITY=compact OMP_NUM_THREADS=20 ./compact3d_openmp > perf_out
mv Compact3D.h5 Compact3d_openmp.h5
$HDF5_INSTALL_PATH/bin/h5diff -p $TOL Compact3d_openmp.h5 Compact3d_dev_seq.h5 >> diff_out
if [ -s ./diff_out ]
then
    echo "File not empty - Solution Not Valid";exit 1;
else
    echo "PASSED"
fi
grep "Successful exit from OPS!" perf_out
rc=$?; if [[ $rc != 0 ]]; then echo "TEST FAILED";exit $rc; fi
rm perf_out

echo '============> Running MPI+OpenMP'
export OMP_NUM_THREADS=2;$MPI_INSTALL_PATH/bin/mpirun -np 10 ./compact3d_mpi_openmp > perf_out
mv Compact3D.h5 Compact3d_mpi_openmp.h5
$HDF5_INSTALL_PATH/bin/h5diff -p $TOL h5 Compact3d_mpi_openmp. Compact3d_dev_seq.h5 >> diff_out
if [ -s ./diff_out ]
then
    echo "File not empty - Solution Not Valid";exit 1;
else
    echo "PASSED"
fi
grep "Successful exit from OPS!" perf_out
rc=$?; if [[ $rc != 0 ]]; then echo "TEST FAILED";exit $rc; fi
rm perf_out

echo '============> Running OpenMP with Tiling'
KMP_AFFINITY=compact OMP_NUM_THREADS=20 ./compact3d_tiled OPS_TILING > perf_out
mv Compact3D.h5 Compact3d_mpi_openmp_tiled.h5
$HDF5_INSTALL_PATH/bin/h5diff -p $TOL .h5 Compact3d_mpi_openmp_ Compact3d_dev_seq.h5 >> diff_out
if [ -s ./diff_out ]
then
    echo "File not empty - Solution Not Valid";exit 1;
else
    echo "PASSED"
fi
grep "Successful exit from OPS!" perf_out
rc=$?; if [[ $rc != 0 ]]; then echo "TEST FAILED";exit $rc; fi
rm perf_out

echo '============> Running DEV_MPI'
$MPI_INSTALL_PATH/bin/mpirun -np 20 ./compact3d_dev_mpi > perf_out
mv Compact3D.h5 Compact3d_dev_mpi.h5
$HDF5_INSTALL_PATH/bin/h5diff -p $TOL Compact3d_dev_mpi.h5 Compact3d_dev_seq.h5 >> diff_out
if [ -s ./diff_out ]
then
    echo "File not empty - Solution Not Valid";exit 1;
else
    echo "PASSED"
fi
grep "Successful exit from OPS!" perf_out
rc=$?; if [[ $rc != 0 ]]; then echo "TEST FAILED";exit $rc; fi
rm perf_out

echo '============> Running MPI'
$MPI_INSTALL_PATH/bin/mpirun -np 20 ./compact3d_mpi > perf_out
mv Compact3D.h5 Compact3d_mpi.h5
$HDF5_INSTALL_PATH/bin/h5diff -p $TOL Compact3d_mpi.h5 Compact3d_dev_seq.h5 >> diff_out
if [ -s ./diff_out ]
then
    echo "File not empty - Solution Not Valid";exit 1;
else
    echo "PASSED"
fi
grep "Successful exit from OPS!" perf_out
rc=$?; if [[ $rc != 0 ]]; then echo "TEST FAILED";exit $rc; fi
rm perf_out

echo '============> Running MPI_Tiled'
export OMP_NUM_THREADS=10;$MPI_INSTALL_PATH/bin/mpirun -np 2 numawrap2 ./compact3d_mpi_tiled OPS_TILING OPS_TILING_MAXDEPTH=10 > perf_out
mv Compact3D.h5 Compact3d_mpi_tiled.h5
$HDF5_INSTALL_PATH/bin/h5diff -p $TOL Compact3d_mpi_tiled.h5 Compact3d_dev_seq.h5 >> diff_out
if [ -s ./diff_out ]
then
    echo "File not empty - Solution Not Valid";exit 1;
else
    echo "PASSED"
fi
grep "Successful exit from OPS!" perf_out
rc=$?; if [[ $rc != 0 ]]; then echo "TEST FAILED";exit $rc; fi
rm perf_out


if [[ -v CUDA_INSTALL_PATH ]]; then
make IEEE=1 compact3d_cuda compact3d_mpi_cuda compact3d_mpi_cuda_tiled
#compact3d_openacc compact3d_mpi_openacc compact3d_mpi_openacc_tiled

echo '============> Running CUDA'
./compact3d_cuda OPS_BLOCK_SIZE_X=64 OPS_BLOCK_SIZE_Y=4 > perf_out
mv Compact3D.h5 Compact3d_cuda.h5
$HDF5_INSTALL_PATH/bin/h5diff -p $TOL Compact3d_cuda.h5 Compact3d_dev_seq.h5 >> diff_out
if [ -s ./diff_out ]
then
    echo "File not empty - Solution Not Valid";exit 1;
else
    echo "PASSED"
fi
grep "Successful exit from OPS!" perf_out
rc=$?; if [[ $rc != 0 ]]; then echo "TEST FAILED";exit $rc; fi
rm perf_out
echo '============> Running MPI+CUDA'
$MPI_INSTALL_PATH/bin/mpirun -np 2 ./compact3d_mpi_cuda OPS_BLOCK_SIZE_X=64 OPS_BLOCK_SIZE_Y=4 > perf_out
mv Compact3D.h5 Compact3d_mpi_cuda.h5
$HDF5_INSTALL_PATH/bin/h5diff -p $TOL Compact3d_mpi_cuda.h5 Compact3d_dev_seq.h5 >> diff_out
if [ -s ./diff_out ]
then
    echo "File not empty - Solution Not Valid";exit 1;
else
    echo "PASSED"
fi
grep "Successful exit from OPS!" perf_out
rc=$?; if [[ $rc != 0 ]]; then echo "TEST FAILED";exit $rc; fi
rm perf_out

fi

echo '============> Running OMPOFFLOAD'
./compact3d_ompoffload OPS_BLOCK_SIZE_X=64 OPS_BLOCK_SIZE_Y=4 > perf_out
mv Compact3D.h5 Compact3d_ompoffload.h5
$HDF5_INSTALL_PATH/bin/h5diff -p $TOL h5 Compact3d_ompoffload. Compact3d_dev_seq.h5 >> diff_out
if [ -s ./diff_out ]
then
    echo "File not empty - Solution Not Valid";exit 1;
else
    echo "PASSED"
fi
grep "Successful exit from OPS!" perf_out
rc=$?; if [[ $rc != 0 ]]; then echo "TEST FAILED";exit $rc; fi
rm perf_out

echo '============> Running MPI+OMPOFFLOAD'
$MPI_INSTALL_PATH/bin/mpirun -np 2 ./compact3d_mpi_ompoffload OPS_BLOCK_SIZE_X=64 OPS_BLOCK_SIZE_Y=4 > perf_out
mv Compact3D.h5 Compact3d_mpi_ompoffload.h5
$HDF5_INSTALL_PATH/bin/h5diff -p $TOL .h5 Compact3d_mpi_ompoffl Compact3d_dev_seq.h5 >> diff_out
if [ -s ./diff_out ]
then
    echo "File not empty - Solution Not Valid";exit 1;
else
    echo "PASSED"
fi
grep "Successful exit from OPS!" perf_out
rc=$?; if [[ $rc != 0 ]]; then echo "TEST FAILED";exit $rc; fi
rm perf_out

echo "All PGI complier based applications ---- PASSED"

fi

if [[ -v AMOS ]]; then

echo "Testing AMD HIP complier based applications ---- "
cd $OPS_INSTALL_PATH/c
source ../../scripts/$SOURCE_AMD_HIP
#make -j -B
#make clean
#make
cd $OPS_INSTALL_PATH/../apps/c/compact_scheme

make clean
#make IEEE=1 -j
make IEEE=1 compact3d_hip compact3d_mpi_hip #compact3d_hip_tiled compact3d_mpi_hip_tiled

echo '============> Running HIP'
./compact3d_hip OPS_BLOCK_SIZE_X=64 OPS_BLOCK_SIZE_Y=4 > perf_out
mv Compact3D.h5 Compact3d_hip.h5
$HDF5_INSTALL_PATH/bin/h5diff -p $TOL Compact3d_hip.h5 Compact3d_dev_seq.h5 >> diff_out
if [ -s ./diff_out ]
then
    echo "File not empty - Solution Not Valid";exit 1;
else
    echo "PASSED"
fi
grep "Successful exit from OPS!" perf_out
rc=$?; if [[ $rc != 0 ]]; then echo "TEST FAILED";exit $rc; fi
rm perf_out

echo '============> Running MPI+HIP'
#mpirun --allow-run-as-root -np 2 ./compact3d_mpi_hip OPS_BLOCK_SIZE_X=64 OPS_BLOCK_SIZE_Y=4 > perf_out
mpirun -np 2 ./compact3d_mpi_hip OPS_BLOCK_SIZE_X=64 OPS_BLOCK_SIZE_Y=4 > perf_out
mv Compact3D.h5 Compact3d_mpi_hip.h5
$HDF5_INSTALL_PATH/bin/h5diff -p $TOL Compact3d_mpi_hip.h5 Compact3d_dev_seq.h5 >> diff_out
if [ -s ./diff_out ]
then
    echo "File not empty - Solution Not Valid";exit 1;
else
    echo "PASSED"
fi
grep "Successful exit from OPS!" perf_out
rc=$?; if [[ $rc != 0 ]]; then echo "TEST FAILED";exit $rc; fi
rm perf_out

echo "All AMD HIP complier based applications ---- PASSED"

fi

rm -rf diff_out perf_out

echo "---------- Exiting Test Script "
