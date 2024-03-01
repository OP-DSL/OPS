#!/bin/bash
set -e
cd $OPS_INSTALL_PATH/c

export SOURCE_INTEL=source_intel_2021.3_pythonenv
export SOURCE_PGI=source_pgi_nvhpc_23_pythonenv
export SOURCE_INTEL_SYCL=source_intel_2021.3_sycl_pythonenv
export SOURCE_AMD_HIP=source_amd_rocm-5.4.3_pythonenv

#export AMOS=TRUE
#export DMOS=TRUE
#export TELOS=TRUE
#export KOS=TRUE

#<<comment

if [[ -v TELOS || -v KOS ]]; then

#============================ Test with Intel Classic Compilers==========================================
echo "Testing Intel classic complier based applications ---- "
cd $OPS_INSTALL_PATH/c
source ../../scripts/$SOURCE_INTEL
#make -j -B
make clean
make
cd $OPS_INSTALL_PATH/../apps/c/mblock

make clean
make IEEE=1 
rm -f *.h5

echo '============> Running OpenMP'
KMP_AFFINITY=compact OMP_NUM_THREADS=20 ./mblock_openmp > mblock.out
grep "Total Wall time" mblock.out
grep "PASSED" mblock.out
rc=$?; if [[ $rc != 0 ]]; then echo "TEST FAILED";exit $rc; fi
rm -f mblock.out *.h5 data*

#echo '============> Running OpenMP with Tiling'
#KMP_AFFINITY=compact OMP_NUM_THREADS=20 ./mblock_tiled OPS_TILING > mblock.out
#grep "Total Wall time" mblock.out
#grep "PASSED" mblock.out
#rc=$?; if [[ $rc != 0 ]]; then echo "TEST FAILED";exit $rc; fi
#rm -f mblock.out *.h5 data*

echo '============> Running MPI+OpenMP'
export OMP_NUM_THREADS=2;$MPI_INSTALL_PATH/bin/mpirun -np 10 ./mblock_mpi_openmp > mblock.out
grep "Total Wall time" mblock.out
grep "PASSED" mblock.out
rc=$?; if [[ $rc != 0 ]]; then echo "TEST FAILED";exit $rc; fi
rm -f mblock.out *.h5 data*

echo '============> Running DEV_MPI'
$MPI_INSTALL_PATH/bin/mpirun -np 20 ./mblock_dev_mpi > mblock.out
grep "Total Wall time" mblock.out
grep "PASSED" mblock.out
rc=$?; if [[ $rc != 0 ]]; then echo "TEST FAILED";exit $rc; fi
rm -f mblock.out *.h5 data*

echo '============> Running MPI'
$MPI_INSTALL_PATH/bin/mpirun -np 20 ./mblock_mpi > mblock.out
grep "Total Wall time" mblock.out
grep "PASSED" mblock.out
rc=$?; if [[ $rc != 0 ]]; then echo "TEST FAILED";exit $rc; fi
rm -f mblock.out *.h5 data*

#echo '============> Running MPI_Tiled'
#export OMP_NUM_THREADS=10;$MPI_INSTALL_PATH/bin/mpirun -np 2 numawrap2 ./mblock_mpi_tiled OPS_TILING OPS_TILING_MAXDEPTH=6 > mblock.out
#grep "Total Wall time" mblock.out
#grep "PASSED" mblock.out
#rc=$?; if [[ $rc != 0 ]]; then echo "TEST FAILED";exit $rc; fi
#rm -f mblock.out *.h5 data*

if [[ -v CUDA_INSTALL_PATH ]]; then	
echo '============> Running CUDA'
./mblock_cuda OPS_BLOCK_SIZE_X=64 OPS_BLOCK_SIZE_Y=4 > mblock.out
grep "Total Wall time" mblock.out
grep "PASSED" mblock.out
rc=$?; if [[ $rc != 0 ]]; then echo "TEST FAILED";exit $rc; fi
rm -f mblock.out *.h5 data*

echo '============> Running MPI+CUDA'
$MPI_INSTALL_PATH/bin/mpirun -np 2 ./mblock_mpi_cuda OPS_BLOCK_SIZE_X=64 OPS_BLOCK_SIZE_Y=4 > mblock.out
$MPI_INSTALL_PATH/bin/mpirun -np 2 ./mblock_mpi_cuda OPS_BLOCK_SIZE_X=64 OPS_BLOCK_SIZE_Y=4 > mblock.out
grep "Total Wall time" mblock.out
grep "PASSED" mblock.out
rc=$?; if [[ $rc != 0 ]]; then echo "TEST FAILED";exit $rc; fi
rm -f mblock.out *.h5 data*

#echo '============> Running MPI+CUDA with GPU-Direct'
#MV2_USE_CUDA=1 $MPI_INSTALL_PATH/bin/mpirun -np 2 ./mblock_mpi_cuda -gpudirect OPS_BLOCK_SIZE_X=64 OPS_BLOCK_SIZE_Y=4 > mblock.out
#grep "Total Wall time" mblock.out
#grep "PASSED" mblock.out
#rc=$?; if [[ $rc != 0 ]]; then echo "TEST FAILED";exit $rc; fi
#rm -f mblock.out *.h5 data*

#echo '============> Running MPI+CUDA+Tiled'
#$MPI_INSTALL_PATH/bin/mpirun -np 2 ./mblock_mpi_cuda_tiled OPS_TILING OPS_TILING_MAXDEPTH=6 OPS_BLOCK_SIZE_X=64 OPS_BLOCK_SIZE_Y=4 > mblock.out
#grep "Total Wall time" mblock.out
#grep "PASSED" mblock.out
#rc=$?; if [[ $rc != 0 ]]; then echo "TEST FAILED";exit $rc; fi
#rm -f mblock.out *.h5 data*
fi

echo "All Intel classic complier based applications ---- PASSED"

fi


if [[ -v TELOS ]]; then

#============================ Test with Intel SYCL Compilers==========================================
echo "Testing Intel SYCL complier based applications ---- "
cd $OPS_INSTALL_PATH/c
source ../../scripts/$SOURCE_INTEL_SYCL

make clean
make
cd $OPS_INSTALL_PATH/../apps/c/mblock

make clean
make IEEE=1 mblock_sycl mblock_mpi_sycl 

echo '============> Running SYCL on CPU'
./mblock_sycl OPS_CL_DEVICE=0 OPS_BLOCK_SIZE_X=512 OPS_BLOCK_SIZE_Y=1 > mblock.out
grep "Total Wall time" mblock.out
grep "PASSED" mblock.out
rc=$?; if [[ $rc != 0 ]]; then echo "TEST FAILED";exit $rc; fi
rm -f mblock.out *.h5 data*

echo '============> Running MPI+SYCL on CPU'
$MPI_INSTALL_PATH/bin/mpirun -np 10 ./mblock_mpi_sycl OPS_CL_DEVICE=0  > mblock.out
grep "Total Wall time" mblock.out
grep "PASSED" mblock.out
rc=$?; if [[ $rc != 0 ]]; then echo "TEST FAILED";exit $rc; fi
rm -f mblock.out *.h5 data*

echo "All Intel SYCL complier based applications ---- PASSED"

fi
#comment

if [[ -v TELOS ]]; then

#============================ Test with PGI Compilers==========================================
echo "Testing PGI/NVHPC complier based applications ---- "
cd $OPS_INSTALL_PATH/c
source ../../scripts/$SOURCE_PGI
make clean
make

cd $OPS_INSTALL_PATH/../apps/c/mblock
make clean
make IEEE=1 mblock_dev_seq mblock_dev_mpi mblock_seq mblock_openmp mblock_mpi \
mblock_mpi_openmp mblock_ompoffload mblock_mpi_ompoffload 

echo '============> Running OpenMP'
KMP_AFFINITY=compact OMP_NUM_THREADS=20 ./mblock_openmp > mblock.out
grep "Total Wall time" mblock.out
grep "PASSED" mblock.out
rc=$?; if [[ $rc != 0 ]]; then echo "TEST FAILED";exit $rc; fi
rm -f mblock.out *.h5 data*

echo '============> Running MPI+OpenMP'
export OMP_NUM_THREADS=2;$MPI_INSTALL_PATH/bin/mpirun -np 10 ./mblock_mpi_openmp > mblock.out
grep "Total Wall time" mblock.out
grep "PASSED" mblock.out
rc=$?; if [[ $rc != 0 ]]; then echo "TEST FAILED";exit $rc; fi
rm -f mblock.out *.h5 data*

#echo '============> Running OpenMP with Tiling'
#KMP_AFFINITY=compact OMP_NUM_THREADS=20 ./mblock_tiled OPS_TILING > mblock.out
#grep "Total Wall time" mblock.out
#grep "PASSED" mblock.out
#rc=$?; if [[ $rc != 0 ]]; then echo "TEST FAILED";exit $rc; fi
#rm -f mblock.out *.h5 data*

echo '============> Running DEV_MPI'
$MPI_INSTALL_PATH/bin/mpirun -np 20 ./mblock_dev_mpi > mblock.out
grep "Total Wall time" mblock.out
grep "PASSED" mblock.out
rc=$?; if [[ $rc != 0 ]]; then echo "TEST FAILED";exit $rc; fi
rm -f mblock.out *.h5 data*

echo '============> Running MPI'
$MPI_INSTALL_PATH/bin/mpirun -np 20 ./mblock_mpi > mblock.out
grep "Total Wall time" mblock.out
grep "PASSED" mblock.out
rc=$?; if [[ $rc != 0 ]]; then echo "TEST FAILED";exit $rc; fi
rm -f mblock.out *.h5 data*
#COMMENT


#echo '============> Running MPI_Tiled'
#export OMP_NUM_THREADS=10;$MPI_INSTALL_PATH/bin/mpirun -np 2 numawrap2 ./mblock_mpi_tiled OPS_TILING OPS_TILING_MAXDEPTH=6 > mblock.out
#grep "Total Wall time" mblock.out
#grep "PASSED" mblock.out
#rc=$?; if [[ $rc != 0 ]]; then echo "TEST FAILED";exit $rc; fi
#rm -f mblock.out *.h5 data*

if [[ -v CUDA_INSTALL_PATH ]]; then
make mblock_cuda mblock_mpi_cuda

echo '============> Running CUDA'
./mblock_cuda OPS_BLOCK_SIZE_X=64 OPS_BLOCK_SIZE_Y=4 > mblock.out
grep "Total Wall time" mblock.out
grep "PASSED" mblock.out
rc=$?; if [[ $rc != 0 ]]; then echo "TEST FAILED";exit $rc; fi
rm -f mblock.out *.h5 data*

echo '============> Running MPI+CUDA'
$MPI_INSTALL_PATH/bin/mpirun -np 2 ./mblock_mpi_cuda OPS_BLOCK_SIZE_X=64 OPS_BLOCK_SIZE_Y=4 > mblock.out
grep "Total Wall time" mblock.out
grep "PASSED" mblock.out
rc=$?; if [[ $rc != 0 ]]; then echo "TEST FAILED";exit $rc; fi
rm -f mblock.out *.h5 data*

#echo '============> Running MPI+CUDA with GPU-Direct'
#MV2_USE_CUDA=1 $MPI_INSTALL_PATH/bin/mpirun -np 2 ./mblock_mpi_cuda -gpudirect OPS_BLOCK_SIZE_X=64 OPS_BLOCK_SIZE_Y=4 > mblock.out
#grep "Total Wall time" mblock.out
#grep "PASSED" mblock.out
#rc=$?; if [[ $rc != 0 ]]; then echo "TEST FAILED";exit $rc; fi
#rm -f mblock.out *.h5 data*

#echo '============> Running MPI+CUDA+Tiled'
#$MPI_INSTALL_PATH/bin/mpirun -np 2 ./mblock_mpi_cuda_tiled OPS_TILING OPS_TILING_MAXDEPTH=6 OPS_BLOCK_SIZE_X=64 OPS_BLOCK_SIZE_Y=4 > mblock.out
#grep "Total Wall time" mblock.out
#grep "PASSED" mblock.out
#rc=$?; if [[ $rc != 0 ]]; then echo "TEST FAILED";exit $rc; fi
#rm -f mblock.out *.h5 data*

fi

echo '============> Running OMPOFFLOAD'
./mblock_ompoffload OPS_BLOCK_SIZE_X=64 OPS_BLOCK_SIZE_Y=4 > mblock.out
grep "Total Wall time" mblock.out
grep "PASSED" mblock.out
rc=$?; if [[ $rc != 0 ]]; then echo "TEST FAILED";exit $rc; fi
rm -f mblock.out *.h5 data*

echo '============> Running MPI+OMPOFFLOAD'
$MPI_INSTALL_PATH/bin/mpirun -np 2 ./mblock_mpi_ompoffload OPS_BLOCK_SIZE_X=64 OPS_BLOCK_SIZE_Y=4 > mblock.out
grep "Total Wall time" mblock.out
grep "PASSED" mblock.out
rc=$?; if [[ $rc != 0 ]]; then echo "TEST FAILED";exit $rc; fi
rm -f mblock.out *.h5 data*

echo "All PGI complier based applications ---- PASSED"

fi


if [[ -v AMOS ]]; then

echo "Testing AMD HIP complier based applications ---- "
cd $OPS_INSTALL_PATH/c
source ../../scripts/$SOURCE_AMD_HIP
#make -j -B
make clean
make
cd $OPS_INSTALL_PATH/../apps/c/mblock

make clean
rm -f .generated
#make IEEE=1 -j
make IEEE=1 mblock_hip mblock_mpi_hip #cloverleaf_hip_tiled cloverleaf_mpi_hip_tiled

echo '============> Running HIP'
./mblock_hip OPS_BLOCK_SIZE_X=64 OPS_BLOCK_SIZE_Y=4 > mblock.out
grep "Total Wall time" mblock.out
grep "PASSED" mblock.out
rc=$?; if [[ $rc != 0 ]]; then echo "TEST FAILED";exit $rc; fi
rm -f mblock.out *.h5 data*

echo '============> Running MPI+HIP'
$MPI_INSTALL_PATH/bin/mpirun -np 2 ./mblock_mpi_hip OPS_BLOCK_SIZE_X=64 OPS_BLOCK_SIZE_Y=4 > mblock.out
grep "Total Wall time" mblock.out
grep "PASSED" mblock.out
rc=$?; if [[ $rc != 0 ]]; then echo "TEST FAILED";exit $rc; fi
rm -f mblock.out *.h5 data*
echo "All AMD HIP complier based applications ---- PASSED"

fi

echo "---------- Exiting Test Script "
