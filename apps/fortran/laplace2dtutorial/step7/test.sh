set -e

export SOURCE_INTEL=source_intel_2021.3_pythonenv
export SOURCE_PGI=source_pgi_nvhpc_23_pythonenv
export SOURCE_INTEL_SYCL=source_intel_2021.3_sycl_pythonenv
export SOURCE_AMD_HIP=source_amd_rocm-5.4.3_pythonenv

#export AMOS=TRUE
#export DMOS=TRUE
export TELOS=TRUE
#export KOS=TRUE

if [[ -v TELOS || -v KOS ]]; then

echo '============================ Test laplace2d Intel Compilers=========================================================='
cd $OPS_INSTALL_PATH/fortran
source ../../scripts/$SOURCE_INTEL
make
cd -
make clean
make IEEE=1

echo '============> Running OpenMP'
KMP_AFFINITY=compact OMP_NUM_THREADS=20 ./laplace2d_openmp > perf_out
grep "Total error" perf_out
grep "Max total runtime" perf_out
grep "PASSED" perf_out
rc=$?; if [[ $rc != 0 ]]; then echo "TEST FAILED";exit $rc; fi
rm perf_out

echo '============> Running MPI+OpenMP'
export OMP_NUM_THREADS=2;KMP_AFFINITY=compact $MPI_INSTALL_PATH/bin/mpirun -np 10 ./laplace2d_mpi_openmp > perf_out
grep "Total error" perf_out
grep "Max total runtime" perf_out
grep "PASSED" perf_out
rc=$?; if [[ $rc != 0 ]]; then echo "TEST FAILED";exit $rc; fi
rm perf_out

echo '============> Running MPI'
$MPI_INSTALL_PATH/bin/mpirun -np 20 ./laplace2d_mpi > perf_out
grep "Total error" perf_out
grep "Max total runtime" perf_out
grep "PASSED" perf_out
rc=$?; if [[ $rc != 0 ]]; then echo "TEST FAILED";exit $rc; fi
rm perf_out

fi
echo "All Intel classic complier based applications ---- PASSED"

if [[ -v TELOS ]]; then

cd $OPS_INSTALL_PATH/fortran
#============================ Test with PGI Compilers==========================================
source ../../scripts/$SOURCE_PGI
make clean
make
cd -
make clean
make laplace2d_openmp laplace2d_mpi_openmp laplace2d_mpi laplace2d_cuda laplace2d_mpi_cuda


echo '============================ Test laplace2d PGI Compilers=========================================================='
echo '============> Running OpenMP'
KMP_AFFINITY=compact OMP_NUM_THREADS=20 ./laplace2d_openmp > perf_out
grep "Total error" perf_out
grep "Max total runtime" perf_out
grep "PASSED" perf_out
rc=$?; if [[ $rc != 0 ]]; then echo "TEST FAILED";exit $rc; fi
rm perf_out

echo '============> Running MPI+OpenMP'
export OMP_NUM_THREADS=2;$MPI_INSTALL_PATH/bin/mpirun -np 10 ./laplace2d_mpi_openmp > perf_out
grep "Total error" perf_out
grep "Max total runtime" perf_out
grep "PASSED" perf_out
rc=$?; if [[ $rc != 0 ]]; then echo "TEST FAILED";exit $rc; fi
rm perf_out

echo '============> Running MPI'
$MPI_INSTALL_PATH/bin/mpirun -np 20 ./laplace2d_mpi > perf_out
grep "Total error" perf_out
grep "Max total runtime" perf_out
grep "PASSED" perf_out
rc=$?; if [[ $rc != 0 ]]; then echo "TEST FAILED";exit $rc; fi
rm perf_out

echo '============> Running CUDA'
./laplace2d_cuda OPS_BLOCK_SIZE_X=64 OPS_BLOCK_SIZE_Y=4 > perf_out
grep "Total error" perf_out
grep "Max total runtime" perf_out
grep "PASSED" perf_out
rc=$?; if [[ $rc != 0 ]]; then echo "TEST FAILED";exit $rc; fi
rm perf_out

echo '============> Running MPI+CUDA'
$MPI_INSTALL_PATH/bin/mpirun -np 2 ./laplace2d_mpi_cuda OPS_BLOCK_SIZE_X=64 OPS_BLOCK_SIZE_Y=4 > perf_out
grep "Total error" perf_out
grep "Max total runtime" perf_out
grep "PASSED" perf_out
rc=$?; if [[ $rc != 0 ]]; then echo "TEST FAILED";exit $rc; fi
rm perf_out

#echo '============> Running MPI+CUDA with GPU-Direct'
#MV2_USE_CUDA=1 $MPI_INSTALL_PATH/bin/mpirun -np 2 ./laplace2d_mpi_cuda -gpudirect OPS_BLOCK_SIZE_X=64 OPS_BLOCK_SIZE_Y=4 > perf_out
#grep "Total error" perf_out
#grep "Max total runtime" perf_out
#grep "PASSED" perf_out
#rc=$?; if [[ $rc != 0 ]]; then echo "TEST FAILED";exit $rc; fi
#rm perf_out

#echo '============> Running OMPOFFLOAD'
#./laplace2doad OPS_BLOCK_SIZE_X=64 OPS_BLOCK_SIZE_Y=4 > perf_out
#grep "Total error" perf_out
#grep "Total Wall time" perf_out
#grep "PASSED" perf_out
#rc=$?; if [[ $rc != 0 ]]; then echo "TEST FAILED";exit $rc; fi
#rm perf_out

#echo '============> Running MPI+OMPOFFLOAD'
#$MPI_INSTALL_PATH/bin/mpirun -np 2 ./laplace2d_mpi_ompoffload OPS_BLOCK_SIZE_X=64 OPS_BLOCK_SIZE_Y=4 > perf_out
#grep "Total error" perf_out
#grep "Total Wall time" perf_out
#grep "PASSED" perf_out
#rc=$?; if [[ $rc != 0 ]]; then echo "TEST FAILED";exit $rc; fi
#rm perf_out

#echo '============> Running MPI+OMPOFFLOAD+Tiled'
#$MPI_INSTALL_PATH/bin/mpirun -np 2 ./laplace2d_mpi_ompoffload_tiled OPS_BLOCK_SIZE_X=64 OPS_BLOCK_SIZE_Y=4 > perf_out
#grep "Total error" perf_out
#grep "Total Wall time" perf_out
#grep "PASSED" perf_out
#rc=$?; if [[ $rc != 0 ]]; then echo "TEST FAILED";exit $rc; fi
#rm perf_out

echo "All PGI complier based applications ---- PASSED"
fi

echo "---------- Exiting Test Script "
