#!/bin/bash
set -e
cd ../../../ops/c

#export SOURCE_INTEL=source_intel_2021.3_pythonenv
#export SOURCE_PGI=source_pgi_nvhpc-23-new
#export SOURCE_INTEL_SYCL=source_intel_2021.3_sycl_pythonenv
#export SOURCE_AMD_HIP=source_amd_rocm-5.4.3_pythonenv

#export AMOS=TRUE
#export DMOS=TRUE
#export TELOS=TRUE
export KOS=TRUE

if [[ -v TELOS || -v KOS ]]; then

#============================ Test with Intel Classic Compilers==========================================
echo "Testing Intel classic complier based applications ---- "
cd $OPS_INSTALL_PATH/c
source ../../scripts/$SOURCE_INTEL
#make -j -B
make clean
make
cd $OPS_INSTALL_PATH/../apps/c/CloverLeaf

make clean
rm -f .generated
make IEEE=1 cloverleaf_dev_seq cloverleaf_dev_mpi cloverleaf_seq cloverleaf_tiled cloverleaf_openmp cloverleaf_mpi \
cloverleaf_mpi_tiled cloverleaf_mpi_openmp

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
export OMP_NUM_THREADS=10;$MPI_INSTALL_PATH/bin/mpirun -np 2 numawrap2 ./cloverleaf_mpi_tiled OPS_TILING OPS_TILING_MAXDEPTH=6 > perf_out
grep "Total Wall time" clover.out
#grep -e "step:   2952" -e "step:   2953" -e "step:   2954" -e "step:   2955" clover.out
grep "PASSED" clover.out
rc=$?; if [[ $rc != 0 ]]; then echo "TEST FAILED";exit $rc; fi
rm -f clover.out

if [[ -v CUDA_INSTALL_PATH ]]; then
make IEEE=1 cloverleaf_cuda cloverleaf_mpi_cuda cloverleaf_mpi_cuda_tiled

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

echo '============> Running MPI+CUDA+Tiled'
$MPI_INSTALL_PATH/bin/mpirun -np 2 ./cloverleaf_mpi_cuda_tiled OPS_TILING OPS_TILING_MAXDEPTH=6 OPS_BLOCK_SIZE_X=64 OPS_BLOCK_SIZE_Y=4 > perf_out
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

fi
fi

echo "All Intel classic complier based applications ---- PASSED"


if [[ -v TELOS ]]; then

#============================ Test with Intel SYCL Compilers==========================================
echo "Testing Intel SYCL complier based applications ---- "
cd $OPS_INSTALL_PATH/c
source ../../scripts/$SOURCE_INTEL_SYCL
#make -j -B
make clean
make
cd $OPS_INSTALL_PATH/../apps/c/CloverLeaf

make clean
#make IEEE=1 -j
make cloverleaf_sycl cloverleaf_mpi_sycl cloverleaf_mpi_sycl_tiled

echo '============> Running SYCL on CPU'
./cloverleaf_sycl OPS_CL_DEVICE=0 OPS_BLOCK_SIZE_X=512 OPS_BLOCK_SIZE_Y=1 > perf_out
grep "Total Wall time" perf_out
grep "PASSED" perf_out
rc=$?; if [[ $rc != 0 ]]; then echo "TEST FAILED";exit $rc; fi
rm perf_out

echo '============> Running MPI+SYCL on CPU'
$MPI_INSTALL_PATH/bin/mpirun -np 20 ./cloverleaf_mpi_sycl OPS_CL_DEVICE=0 OPS_BLOCK_SIZE_X=256 OPS_BLOCK_SIZE_Y=1 > perf_out
grep "Total Wall time" perf_out
grep "PASSED" perf_out
rc=$?; if [[ $rc != 0 ]]; then echo "TEST FAILED";exit $rc; fi
rm perf_out

echo '============> Running MPI+SYCL Tiled on CPU'
$MPI_INSTALL_PATH/bin/mpirun -np 2 ./cloverleaf_mpi_sycl_tiled OPS_CL_DEVICE=1 OPS_BLOCK_SIZE_X=32 OPS_BLOCK_SIZE_Y=4 > perf_out
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
#make -j
make
echo "in here "
cd $OPS_INSTALL_PATH/../apps/c/CloverLeaf
make clean
make cloverleaf_dev_seq cloverleaf_dev_mpi cloverleaf_seq cloverleaf_tiled cloverleaf_openmp cloverleaf_mpi cloverleaf_mpi_tiled \
cloverleaf_mpi_openmp cloverleaf_ompoffload cloverleaf_mpi_ompoffload cloverleaf_mpi_ompoffload_tiled

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

echo '============> Running OpenMP with Tiling'
KMP_AFFINITY=compact OMP_NUM_THREADS=20 ./cloverleaf_tiled OPS_TILING > perf_out
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

if [[ -v CUDA_INSTALL_PATH ]]; then
make IEEE=1 cloverleaf_cuda cloverleaf_mpi_cuda cloverleaf_mpi_cuda_tiled cloverleaf_openacc cloverleaf_mpi_openacc \
cloverleaf_mpi_openacc_tiled

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

#echo '============> Running MPI+CUDA+Tiled'
$MPI_INSTALL_PATH/bin/mpirun -np 2 ./cloverleaf_mpi_cuda_tiled OPS_TILING OPS_TILING_MAXDEPTH=6 OPS_BLOCK_SIZE_X=64 OPS_BLOCK_SIZE_Y=4 > perf_out
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

fi

echo '============> Running OMPOFFLOAD'
./cloverleaf_ompoffload OPS_BLOCK_SIZE_X=64 OPS_BLOCK_SIZE_Y=4 > perf_out
grep "Total Wall time" perf_out
grep "PASSED" perf_out
rc=$?; if [[ $rc != 0 ]]; then echo "TEST FAILED";exit $rc; fi
rm perf_out

echo '============> Running MPI+OMPOFFLOAD'
$MPI_INSTALL_PATH/bin/mpirun -np 2 ./cloverleaf_mpi_ompoffload OPS_BLOCK_SIZE_X=64 OPS_BLOCK_SIZE_Y=4 > perf_out
grep "Total Wall time" perf_out
grep "PASSED" perf_out
rc=$?; if [[ $rc != 0 ]]; then echo "TEST FAILED";exit $rc; fi
rm perf_out

echo "All PGI complier based applications ---- PASSED"

fi

if [[ -v AMOS ]]; then

echo "Testing AMD HIP complier based applications ---- "
cd $OPS_INSTALL_PATH/c
source ../../scripts/$SOURCE_AMD_HIP
#make -j -B
make clean
make
cd $OPS_INSTALL_PATH/../apps/c/CloverLeaf

make clean
rm -f .generated
#make IEEE=1 -j
make IEEE=1 cloverleaf_hip cloverleaf_mpi_hip #cloverleaf_hip_tiled cloverleaf_mpi_hip_tiled

echo '============> Running HIP'
./cloverleaf_hip OPS_BLOCK_SIZE_X=64 OPS_BLOCK_SIZE_Y=4 > perf_out
grep "Total Wall time" perf_out
grep "PASSED" perf_out
rc=$?; if [[ $rc != 0 ]]; then echo "TEST FAILED";exit $rc; fi
rm perf_out

echo '============> Running MPI+HIP'
#mpirun --allow-run-as-root -np 2 ./cloverleaf_mpi_hip OPS_BLOCK_SIZE_X=64 OPS_BLOCK_SIZE_Y=4 > perf_out
mpirun -np 2 ./cloverleaf_mpi_hip OPS_BLOCK_SIZE_X=64 OPS_BLOCK_SIZE_Y=4 > perf_out
grep "Total Wall time" perf_out
grep "PASSED" perf_out
rc=$?; if [[ $rc != 0 ]]; then echo "TEST FAILED";exit $rc; fi
rm perf_out

echo "All AMD HIP complier based applications ---- PASSED"

fi

echo "---------- Exiting Test Script "
