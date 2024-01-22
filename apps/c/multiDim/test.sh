#!/bin/bash
set -e
cd $OPS_INSTALL_PATH/c

export SOURCE_INTEL=source_intel_2021.3_pythonenv
export SOURCE_PGI=source_pgi_nvhpc_23_pythonenv
export SOURCE_INTEL_SYCL=source_intel_2021.3_sycl_pythonenv
export SOURCE_AMD_HIP=source_amd_rocm-5.4.3_pythonenv

export AMOS=TRUE
#export DMOS=TRUE
#export TELOS=TRUE
#export KOS=TRUE

if [[ -v TELOS || -v KOS ]]; then

#============================ Test with Intel Classic Compilers==========================================
echo "Testing Intel classic complier based applications ---- "  
cd $OPS_INSTALL_PATH/c
source ../../scripts/$SOURCE_INTEL

make clean
make
cd $OPS_INSTALL_PATH/../apps/c/multiDim/

make clean
rm -f .generated
make IEEE=1 -j


echo '============> Running OpenMP'
KMP_AFFINITY=compact OMP_NUM_THREADS=1,12 ./multidim_openmp > perf_out
grep "Reduction result" perf_out
grep "Total Wall time" perf_out
grep "PASSED" perf_out
rc=$?; if [[ $rc != 0 ]]; then echo "TEST FAILED";exit $rc; fi
rm perf_out

echo '============> Running MPI+OpenMP'
export OMP_NUM_THREADS=1,2;$MPI_INSTALL_PATH/bin/mpirun -np 4 ./multidim_mpi_openmp > perf_out
grep "Reduction result" perf_out
grep "Total Wall time" perf_out
grep "PASSED" perf_out
rc=$?; if [[ $rc != 0 ]]; then echo "TEST FAILED";exit $rc; fi
rm perf_out

echo '============> Running DEV_MPI'
OMP_NUM_THREADS=1 $MPI_INSTALL_PATH/bin/mpirun -np 4 ./multidim_dev_mpi > perf_out
grep "Reduction result" perf_out
grep "Total Wall time" perf_out
grep "PASSED" perf_out
rc=$?; if [[ $rc != 0 ]]; then echo "TEST FAILED";exit $rc; fi
rm perf_out

echo '============> Running MPI'
OMP_NUM_THREADS=1 $MPI_INSTALL_PATH/bin/mpirun -np 4 ./multidim_mpi > perf_out
grep "Reduction result" perf_out
grep "Total Wall time" perf_out
grep "PASSED" perf_out
rc=$?; if [[ $rc != 0 ]]; then echo "TEST FAILED";exit $rc; fi
rm perf_out


echo '============> Running MPI_inline'
$MPI_INSTALL_PATH/bin/mpirun -np 4 ./multidim_mpi_inline > perf_out
grep "Reduction result" perf_out
grep "Total Wall time" perf_out
grep "PASSED" perf_out
rc=$?; if [[ $rc != 0 ]]; then echo "TEST FAILED";exit $rc; fi
rm perf_out

if [[ -v CUDA_INSTALL_PATH ]]; then
echo '============> Running CUDA'
./multidim_cuda OPS_BLOCK_SIZE_X=64 OPS_BLOCK_SIZE_Y=4 > perf_out
grep "Reduction result" perf_out
grep "Total Wall time" perf_out
grep "PASSED" perf_out
rc=$?; if [[ $rc != 0 ]]; then echo "TEST FAILED";exit $rc; fi
rm perf_out

echo '============> Running MPI+CUDA'
OMP_NUM_THREADS=1 $MPI_INSTALL_PATH/bin/mpirun -np 2 ./multidim_mpi_cuda OPS_BLOCK_SIZE_X=64 OPS_BLOCK_SIZE_Y=4 > perf_out
grep "Reduction result" perf_out
grep "Total Wall time" perf_out
grep "PASSED" perf_out
rc=$?; if [[ $rc != 0 ]]; then echo "TEST FAILED";exit $rc; fi
rm perf_out

#echo '============> Running MPI+CUDA with GPU-Direct'
#MV2_USE_CUDA=1 $MPI_INSTALL_PATH/bin/mpirun -np 2 ./multidim_mpi_cuda -gpudirect OPS_BLOCK_SIZE_X=64 OPS_BLOCK_SIZE_Y=4 > perf_out
#grep "Reduction result" perf_out
#grep "Total Wall time" perf_out
#grep "PASSED" perf_out
#rc=$?; if [[ $rc != 0 ]]; then echo "TEST FAILED";exit $rc; fi
#rm perf_out

fi

echo "All Intel classic complier based applications ---- PASSED"
fi

if [[ -v TELOS ]]; then

echo "Testing Intel SYCL complier based applications ---- "

cd $OPS_INSTALL_PATH/c
source ../../scripts/$SOURCE_INTEL_SYCL
#make -j -B
make clean
make
cd $OPS_INSTALL_PATH/../apps/c/multiDim/

make clean
#make IEEE=1 -j
make IEEE=1 multidim_sycl multidim_mpi_sycl multidim_mpi_sycl_tiled

if [[ -v CUDA_INSTALL_PATH ]]; then
echo '============> Running SYCL'
./multidim_sycl OPS_BLOCK_SIZE_X=64 OPS_BLOCK_SIZE_Y=4 > perf_out
grep "Reduction result" perf_out
grep "Total Wall time" perf_out
grep "PASSED" perf_out
rc=$?; if [[ $rc != 0 ]]; then echo "TEST FAILED";exit $rc; fi
rm perf_out

echo '============> Running MPI+SYCL'
OMP_NUM_THREADS=1 $MPI_INSTALL_PATH/bin/mpirun -np 2 ./multidim_mpi_sycl OPS_BLOCK_SIZE_X=64 OPS_BLOCK_SIZE_Y=4 > perf_out
grep "Reduction result" perf_out
grep "Total Wall time" perf_out
grep "PASSED" perf_out
rc=$?; if [[ $rc != 0 ]]; then echo "TEST FAILED";exit $rc; fi
rm perf_out

echo '============> Running MPI+SYCL Tiled'
OMP_NUM_THREADS=1 $MPI_INSTALL_PATH/bin/mpirun -np 2 ./multidim_mpi_sycl_tiled OPS_BLOCK_SIZE_X=64 OPS_BLOCK_SIZE_Y=4 > perf_out
grep "Reduction result" perf_out
grep "Total Wall time" perf_out
grep "PASSED" perf_out
rc=$?; if [[ $rc != 0 ]]; then echo "TEST FAILED";exit $rc; fi
rm perf_out

echo "All Intel SYCL complier based applications ---- PASSED"

fi


if [[ -v TELOS ]]; then

#============================ Test with PGI Compilers==========================================
echo "Testing PGI/NVHPC complier based applications ---- "

cd $OPS_INSTALL_PATH/c
source ../../scripts/$SOURCE_PGI

make clean
make 
cd $OPS_INSTALL_PATH/../apps/c/multiDim/
make clean
make

echo '============> Running OpenMP'
KMP_AFFINITY=compact OMP_NUM_THREADS=1,12 ./multidim_openmp > perf_out
grep "Reduction result" perf_out
grep "Total Wall time" perf_out
grep "PASSED" perf_out
rc=$?; if [[ $rc != 0 ]]; then echo "TEST FAILED";exit $rc; fi
rm perf_out

echo '============> Running MPI+OpenMP'
export OMP_NUM_THREADS=1,2;$MPI_INSTALL_PATH/bin/mpirun -np 4 ./multidim_mpi_openmp > perf_out
grep "Reduction result" perf_out
grep "Total Wall time" perf_out
grep "PASSED" perf_out
rc=$?; if [[ $rc != 0 ]]; then echo "TEST FAILED";exit $rc; fi
rm perf_out

echo '============> Running DEV_MPI'
OMP_NUM_THREADS=1 $MPI_INSTALL_PATH/bin/mpirun -np 4 ./multidim_dev_mpi > perf_out
grep "Reduction result" perf_out
grep "Total Wall time" perf_out
grep "PASSED" perf_out
rc=$?; if [[ $rc != 0 ]]; then echo "TEST FAILED";exit $rc; fi
rm perf_out

echo '============> Running MPI'
OMP_NUM_THREADS=1 $MPI_INSTALL_PATH/bin/mpirun -np 4 ./multidim_mpi > perf_out
grep "Reduction result" perf_out
grep "Total Wall time" perf_out
grep "PASSED" perf_out
rc=$?; if [[ $rc != 0 ]]; then echo "TEST FAILED";exit $rc; fi
rm perf_out

if [[ -v CUDA_INSTALL_PATH ]]; then
echo '============> Running CUDA'
./multidim_cuda OPS_BLOCK_SIZE_X=64 OPS_BLOCK_SIZE_Y=4 > perf_out
grep "Reduction result" perf_out
grep "Total Wall time" perf_out
grep "PASSED" perf_out
rc=$?; if [[ $rc != 0 ]]; then echo "TEST FAILED";exit $rc; fi
rm perf_out

echo '============> Running MPI+CUDA'
OMP_NUM_THREADS=1 $MPI_INSTALL_PATH/bin/mpirun -np 2 ./multidim_mpi_cuda OPS_BLOCK_SIZE_X=64 OPS_BLOCK_SIZE_Y=4 > perf_out
grep "Reduction result" perf_out
grep "Total Wall time" perf_out
grep "PASSED" perf_out
rc=$?; if [[ $rc != 0 ]]; then echo "TEST FAILED";exit $rc; fi
rm perf_out

#echo '============> Running MPI+CUDA with GPU-Direct'
#MV2_USE_CUDA=1 $MPI_INSTALL_PATH/bin/mpirun -np 2 ./multidim_mpi_cuda -gpudirect OPS_BLOCK_SIZE_X=64 OPS_BLOCK_SIZE_Y=4 > perf_out
#grep "Reduction result" perf_out
#grep "Total Wall time" perf_out
#grep "PASSED" perf_out
#rc=$?; if [[ $rc != 0 ]]; then echo "TEST FAILED";exit $rc; fi
#rm perf_out

#echo '============> Running OpenACC'
#./multidim_openacc OPS_BLOCK_SIZE_X=32 OPS_BLOCK_SIZE_Y=4 > perf_out
#grep "Reduction result" perf_out
#grep "Total Wall time" perf_out
#grep "PASSED" perf_out
#rc=$?; if [[ $rc != 0 ]]; then echo "TEST FAILED";exit $rc; fi
#rm perf_out
#COMMENT
#echo '============> Running MPI+OpenACC'
#OMP_NUM_THREADS=1 $MPI_INSTALL_PATH/bin/mpirun -np 2 ./multidim_mpi_openacc OPS_BLOCK_SIZE_X=32 OPS_BLOCK_SIZE_Y=4 > perf_out
#grep "Reduction result" perf_out
#grep "Total Wall time" perf_out
#grep "PASSED" perf_out
#rc=$?; if [[ $rc != 0 ]]; then echo "TEST FAILED";exit $rc; fi
#rm perf_out
#echo "All PGI complied applications PASSED : Exiting Test Script "

fi

echo '============> Running OMPOFFLOAD'
./multidim_offload OPS_BLOCK_SIZE_X=64 OPS_BLOCK_SIZE_Y=4 > perf_out
grep "Reduction result" perf_out
grep "Total Wall time" perf_out
grep "PASSED" perf_out
rc=$?; if [[ $rc != 0 ]]; then echo "TEST FAILED";exit $rc; fi
rm perf_out

echo '============> Running MPI+OMPOFFLOAD'
OMP_NUM_THREADS=1 $MPI_INSTALL_PATH/bin/mpirun -np 2 ./multidim_mpi_offload OPS_BLOCK_SIZE_X=64 OPS_BLOCK_SIZE_Y=4 > perf_out
grep "Reduction result" perf_out
grep "Total Wall time" perf_out
grep "PASSED" perf_out
rc=$?; if [[ $rc != 0 ]]; then echo "TEST FAILED";exit $rc; fi
rm perf_out


fi

echo "All PGI complier based applications ---- PASSED"

fi

if [[ -v AMOS ]]; then

echo "Testing AMD HIP complier based applications ---- "
cd $OPS_INSTALL_PATH/c
source ../../scripts/$SOURCE_AMD_HIP
#make -j -B
#make clean
#make 
cd $OPS_INSTALL_PATH/../apps/c/multiDim

rm -f .generated
make clean
#make IEEE=1 -j
make multidim_hip multidim_mpi_hip #multiDim_hip_tiled multiDim_mpi_hip_tiled

echo '============> Running HIP'
./multidim_hip OPS_BLOCK_SIZE_X=64 OPS_BLOCK_SIZE_Y=4 > perf_out
grep "Reduction result" perf_out
grep "Total Wall time" perf_out
grep "PASSED" perf_out
rc=$?; if [[ $rc != 0 ]]; then echo "TEST FAILED";exit $rc; fi
rm perf_out

echo '============> Running MPI+HIP'
OMP_NUM_THREADS=1 $MPI_INSTALL_PATH/bin/mpirun -np 2 ./multidim_mpi_hip OPS_BLOCK_SIZE_X=64 OPS_BLOCK_SIZE_Y=4 > perf_out
grep "Reduction result" perf_out
grep "Total Wall time" perf_out
grep "PASSED" perf_out
rc=$?; if [[ $rc != 0 ]]; then echo "TEST FAILED";exit $rc; fi
rm perf_out

echo "All AMD HIP complier based applications ---- PASSED"

fi

echo "---------- Exiting Test Script "