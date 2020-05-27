#!/bin/bash
set -e
cd ../../../ops/c
#<<COMMENT
source ../../scripts/$SOURCE_INTEL
make clean
make -j
cd -
make clean
rm -f .generated
make IEEE=1 -j
rm -f *.h5

#============================ Test mblock 2D With Intel Compilers==========================================================
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

echo '============> Running MPI_Inline with MPI+OpenMP'
export OMP_NUM_THREADS=2;$MPI_INSTALL_PATH/bin/mpirun -np 10 ./mblock_mpi_inline > mblock.out
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

echo '============> Running MPI_Inline'
export OMP_NUM_THREADS=1;$MPI_INSTALL_PATH/bin/mpirun -np 20 ./mblock_mpi_inline > mblock.out
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


echo '============> Running OpenCL on CPU'
./mblock_opencl OPS_CL_DEVICE=0 OPS_BLOCK_SIZE_X=512 OPS_BLOCK_SIZE_Y=1 > mblock.out
grep "Total Wall time" mblock.out
grep "PASSED" mblock.out
rc=$?; if [[ $rc != 0 ]]; then echo "TEST FAILED";exit $rc; fi
rm -f mblock.out *.h5 data*


echo '============> Running OpenCL on GPU'
./mblock_opencl OPS_CL_DEVICE=1 OPS_BLOCK_SIZE_X=32 OPS_BLOCK_SIZE_Y=4 > mblock.out
./mblock_opencl OPS_CL_DEVICE=1 OPS_BLOCK_SIZE_X=32 OPS_BLOCK_SIZE_Y=4 > mblock.out
grep "Total Wall time" mblock.out
grep "PASSED" mblock.out
rc=$?; if [[ $rc != 0 ]]; then echo "TEST FAILED";exit $rc; fi
rm -f mblock.out *.h5 data*

echo '============> Running MPI+OpenCL on CPU'
$MPI_INSTALL_PATH/bin/mpirun -np 20 ./mblock_mpi_opencl OPS_CL_DEVICE=0  > mblock.out
$MPI_INSTALL_PATH/bin/mpirun -np 20 ./mblock_mpi_opencl OPS_CL_DEVICE=0  > mblock.out
grep "Total Wall time" mblock.out
grep "PASSED" mblock.out
rc=$?; if [[ $rc != 0 ]]; then echo "TEST FAILED";exit $rc; fi
rm -f mblock.out *.h5 data*

echo '============> Running MPI+OpenCL on GPU'
$MPI_INSTALL_PATH/bin/mpirun -np 2 ./mblock_mpi_opencl OPS_CL_DEVICE=1 OPS_BLOCK_SIZE_X=32 OPS_BLOCK_SIZE_Y=4 > mblock.out
$MPI_INSTALL_PATH/bin/mpirun -np 2 ./mblock_mpi_opencl OPS_CL_DEVICE=1 OPS_BLOCK_SIZE_X=32 OPS_BLOCK_SIZE_Y=4 > mblock.out
grep "Total Wall time" mblock.out
grep "PASSED" mblock.out
rc=$?; if [[ $rc != 0 ]]; then echo "TEST FAILED";exit $rc; fi
rm -f mblock.out *.h5 data*

echo "All Intel complied applications PASSED : Moving no to PGI Compiler Tests "
#exit

cd -
#COMMENT
source ../../scripts/$SOURCE_PGI

make clean
make
cd -
make clean
make IEEE=1


#============================ Test mblock 2D With PGI Compilers==========================================================
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


#echo '============> Running OpenCL on CPU'
#./mblock_opencl OPS_CL_DEVICE=0 OPS_BLOCK_SIZE_X=512 OPS_BLOCK_SIZE_Y=1 > mblock.out
#grep "Total Wall time" mblock.out
#grep "PASSED" mblock.out
#rc=$?; if [[ $rc != 0 ]]; then echo "TEST FAILED";exit $rc; fi
#rm -f mblock.out *.h5 data*


echo '============> Running OpenCL on GPU'
./mblock_opencl OPS_CL_DEVICE=1 OPS_BLOCK_SIZE_X=32 OPS_BLOCK_SIZE_Y=4 > mblock.out
./mblock_opencl OPS_CL_DEVICE=1 OPS_BLOCK_SIZE_X=32 OPS_BLOCK_SIZE_Y=4 > mblock.out
grep "Total Wall time" mblock.out
grep "PASSED" mblock.out
rc=$?; if [[ $rc != 0 ]]; then echo "TEST FAILED";exit $rc; fi
rm -f mblock.out *.h5 data*

#echo '============> Running MPI+OpenCL on CPU'
#$MPI_INSTALL_PATH/bin/mpirun -np 20 ./mblock_mpi_opencl OPS_CL_DEVICE=0  > mblock.out
#$MPI_INSTALL_PATH/bin/mpirun -np 20 ./mblock_mpi_opencl OPS_CL_DEVICE=0  > mblock.out
#grep "Total Wall time" mblock.out


echo '============> Running MPI+OpenCL on GPU'
$MPI_INSTALL_PATH/bin/mpirun -np 2 ./mblock_mpi_opencl OPS_CL_DEVICE=1 OPS_BLOCK_SIZE_X=32 OPS_BLOCK_SIZE_Y=4 > mblock.out
$MPI_INSTALL_PATH/bin/mpirun -np 2 ./mblock_mpi_opencl OPS_CL_DEVICE=1 OPS_BLOCK_SIZE_X=32 OPS_BLOCK_SIZE_Y=4 > mblock.out
grep "Total Wall time" mblock.out
grep "PASSED" mblock.out
rc=$?; if [[ $rc != 0 ]]; then echo "TEST FAILED";exit $rc; fi
rm -f mblock.out *.h5 data*

echo '============> Running OpenACC'
./mblock_openacc OPS_BLOCK_SIZE_X=64 OPS_BLOCK_SIZE_Y=4 > mblock.out
grep "Total Wall time" mblock.out
grep "PASSED" mblock.out
rc=$?; if [[ $rc != 0 ]]; then echo "TEST FAILED";exit $rc; fi
rm -f mblock.out *.h5 data*

echo '============> Running MPI+OpenACC'
$MPI_INSTALL_PATH/bin/mpirun -np 2 ./mblock_mpi_openacc OPS_BLOCK_SIZE_X=64 OPS_BLOCK_SIZE_Y=4 > mblock.out
grep "Total Wall time" mblock.out
grep "PASSED" mblock.out
rc=$?; if [[ $rc != 0 ]]; then echo "TEST FAILED";exit $rc; fi
rm -f mblock.out *.h5 data*

echo "All PGI complied applications PASSED : Exiting Test Script "
