set -e

export SOURCE_INTEL=source_oneapi_sycl_pythonenv
export SOURCE_PGI=source_pgi_nvhpc_23_pythonenv
export SOURCE_INTEL_SYCL=source_oneapi_sycl_pythonenv
export SOURCE_AMD_HIP=source_amd_rocm-5.4.3_pythonenv

#export AMOS=TRUE
#export TELOS=TRUE
export DEMOS=TRUE
#export KOS=TRUE

if [[ -v TELOS || -v DEMOS || -v KOS ]]; then

echo '============================ Test lowdim Intel Compilers=========================================================='
cd $OPS_INSTALL_PATH/fortran
source ../../scripts/$SOURCE_INTEL
make
cd -
make cleanall
make IEEE=1

echo '============> Running OpenMP'
KMP_AFFINITY=compact OMP_NUM_THREADS=20 ./lowdim_openmp > perf_out
grep "Max total runtime" perf_out
grep "PASSED" perf_out
rc=$?; if [[ $rc != 0 ]]; then echo "TEST FAILED";exit $rc; fi
rm perf_out

echo '============> Running MPI+OpenMP'
export OMP_NUM_THREADS=2;KMP_AFFINITY=compact $MPI_INSTALL_PATH/bin/mpirun -np 5 ./lowdim_mpi_openmp > perf_out
grep "Max total runtime" perf_out
grep "PASSED" perf_out
rc=$?; if [[ $rc != 0 ]]; then echo "TEST FAILED";exit $rc; fi
rm perf_out

echo '============> Running MPI'
$MPI_INSTALL_PATH/bin/mpirun -np 10 ./lowdim_mpi > perf_out
grep "Max total runtime" perf_out
grep "PASSED" perf_out
rc=$?; if [[ $rc != 0 ]]; then echo "TEST FAILED";exit $rc; fi
rm perf_out

echo '============> Running F2C Sequential'
./lowdim_f2c_seq > perf_out
grep "Max total runtime" perf_out
grep "PASSED" perf_out
rc=$?; if [[ $rc != 0 ]]; then echo "TEST FAILED"; exit $rc; fi
rm perf_out

echo '============> Running F2C MPI'
$MPI_INSTALL_PATH/bin/mpirun -np 2 ./lowdim_f2c_mpi > perf_out
grep "Max total runtime" perf_out
grep "PASSED" perf_out
rc=$?; if [[ $rc != 0 ]]; then echo "TEST FAILED"; exit $rc; fi
rm perf_out

echo '============> Running F2C CUDA'
./lowdim_f2c_cuda OPS_BLOCK_SIZE_X=64 OPS_BLOCK_SIZE_Y=4 > perf_out
grep "Max total runtime" perf_out
grep "PASSED" perf_out
rc=$?; if [[ $rc != 0 ]]; then echo "TEST FAILED"; exit $rc; fi
rm perf_out

echo '============> Running F2C MPI+CUDA'
$MPI_INSTALL_PATH/bin/mpirun -np 2 ./lowdim_f2c_mpi_cuda OPS_BLOCK_SIZE_X=64 OPS_BLOCK_SIZE_Y=4 > perf_out
grep "Max total runtime" perf_out
grep "PASSED" perf_out
rc=$?; if [[ $rc != 0 ]]; then echo "TEST FAILED"; exit $rc; fi
rm perf_out

echo '============> Running F2C MPI+CUDA+Tiled'
$MPI_INSTALL_PATH/bin/mpirun -np 2 ./lowdim_f2c_mpi_cuda_tiled OPS_TILING OPS_TILING_MAXDEPTH=2 OPS_BLOCK_SIZE_X=64 OPS_BLOCK_SIZE_Y=4 > perf_out
grep "Max total runtime" perf_out
grep "PASSED" perf_out
rc=$?; if [[ $rc != 0 ]]; then echo "TEST FAILED"; exit $rc; fi
rm perf_out

echo '============> Running F2C MPI+Tiled'
$MPI_INSTALL_PATH/bin/mpirun -np 2 ./lowdim_f2c_mpi_tiled OPS_TILING OPS_TILING_MAXDEPTH=2 > perf_out
grep "Max total runtime" perf_out
grep "PASSED" perf_out
rc=$?; if [[ $rc != 0 ]]; then echo "TEST FAILED"; exit $rc; fi
rm perf_out

echo '============> Running F2C MPI+SYCL'
$MPI_INSTALL_PATH/bin/mpirun -np 2 ./lowdim_f2c_mpi_sycl OPS_BLOCK_SIZE_X=64 OPS_BLOCK_SIZE_Y=4 > perf_out
grep "Max total runtime" perf_out
grep "PASSED" perf_out
rc=$?; if [[ $rc != 0 ]]; then echo "TEST FAILED"; exit $rc; fi
rm perf_out

echo '============> Running F2C MPI+SYCL+Tiled'
$MPI_INSTALL_PATH/bin/mpirun -np 2 ./lowdim_f2c_mpi_sycl_tiled OPS_TILING OPS_TILING_MAXDEPTH=2 OPS_BLOCK_SIZE_X=64 OPS_BLOCK_SIZE_Y=4 > perf_out
grep "Max total runtime" perf_out
grep "PASSED" perf_out
rc=$?; if [[ $rc != 0 ]]; then echo "TEST FAILED"; exit $rc; fi
rm perf_out

fi
echo "All Intel classic complier based applications ---- PASSED"

if [[ -v TELOS || -v DEMOS ]]; then

cd $OPS_INSTALL_PATH/fortran
#============================ Test with PGI Compilers==========================================
source ../../scripts/$SOURCE_PGI
make clean
make
cd -
make clean
make
#lowdim_openmp lowdim_mpi_openmp lowdim_mpi lowdim_cuda lowdim_mpi_cuda


echo '============================ Test lowdim PGI Compilers=========================================================='
echo '============> Running OpenMP'
KMP_AFFINITY=compact OMP_NUM_THREADS=20 ./lowdim_openmp > perf_out
grep "Max total runtime" perf_out
grep "PASSED" perf_out
rc=$?; if [[ $rc != 0 ]]; then echo "TEST FAILED";exit $rc; fi
rm perf_out

echo '============> Running MPI+OpenMP'
export OMP_NUM_THREADS=2;$MPI_INSTALL_PATH/bin/mpirun -np 5 ./lowdim_mpi_openmp > perf_out
grep "Max total runtime" perf_out
grep "PASSED" perf_out
rc=$?; if [[ $rc != 0 ]]; then echo "TEST FAILED";exit $rc; fi
rm perf_out

echo '============> Running MPI'
$MPI_INSTALL_PATH/bin/mpirun -np 10 ./lowdim_mpi > perf_out
grep "Max total runtime" perf_out
grep "PASSED" perf_out
rc=$?; if [[ $rc != 0 ]]; then echo "TEST FAILED";exit $rc; fi
rm perf_out

echo '============> Running CUDA'
./lowdim_cuda OPS_BLOCK_SIZE_X=64 OPS_BLOCK_SIZE_Y=4 > perf_out
grep "Max total runtime" perf_out
grep "PASSED" perf_out
rc=$?; if [[ $rc != 0 ]]; then echo "TEST FAILED";exit $rc; fi
rm perf_out

echo '============> Running MPI+CUDA'
$MPI_INSTALL_PATH/bin/mpirun -np 2 ./lowdim_mpi_cuda OPS_BLOCK_SIZE_X=64 OPS_BLOCK_SIZE_Y=4 > perf_out
grep "Max total runtime" perf_out
grep "PASSED" perf_out
rc=$?; if [[ $rc != 0 ]]; then echo "TEST FAILED";exit $rc; fi
rm perf_out

#echo '============> Running MPI+CUDA with GPU-Direct'
#MV2_USE_CUDA=1 $MPI_INSTALL_PATH/bin/mpirun -np 2 ./lowdim_mpi_cuda -gpudirect OPS_BLOCK_SIZE_X=64 OPS_BLOCK_SIZE_Y=4 > perf_out

#grep "Max total runtime" perf_out
#grep "PASSED" perf_out
#rc=$?; if [[ $rc != 0 ]]; then echo "TEST FAILED";exit $rc; fi
#rm perf_out

echo '============> Running F2C Sequential'
./lowdim_f2c_seq > perf_out
grep "Max total runtime" perf_out
grep "PASSED" perf_out
rc=$?; if [[ $rc != 0 ]]; then echo "TEST FAILED"; exit $rc; fi
rm perf_out

echo '============> Running F2C MPI'
$MPI_INSTALL_PATH/bin/mpirun -np 2 ./lowdim_f2c_mpi > perf_out
grep "Max total runtime" perf_out
grep "PASSED" perf_out
rc=$?; if [[ $rc != 0 ]]; then echo "TEST FAILED"; exit $rc; fi
rm perf_out

echo '============> Running F2C CUDA'
./lowdim_f2c_cuda OPS_BLOCK_SIZE_X=64 OPS_BLOCK_SIZE_Y=4 > perf_out
grep "Max total runtime" perf_out
grep "PASSED" perf_out
rc=$?; if [[ $rc != 0 ]]; then echo "TEST FAILED"; exit $rc; fi
rm perf_out

echo '============> Running F2C MPI+CUDA'
$MPI_INSTALL_PATH/bin/mpirun -np 2 ./lowdim_f2c_mpi_cuda OPS_BLOCK_SIZE_X=64 OPS_BLOCK_SIZE_Y=4 > perf_out
grep "Max total runtime" perf_out
grep "PASSED" perf_out
rc=$?; if [[ $rc != 0 ]]; then echo "TEST FAILED"; exit $rc; fi
rm perf_out

echo '============> Running F2C MPI+CUDA+Tiled'
$MPI_INSTALL_PATH/bin/mpirun -np 2 ./lowdim_f2c_mpi_cuda_tiled OPS_TILING OPS_TILING_MAXDEPTH=2 OPS_BLOCK_SIZE_X=64 OPS_BLOCK_SIZE_Y=4 > perf_out
grep "Max total runtime" perf_out
grep "PASSED" perf_out
rc=$?; if [[ $rc != 0 ]]; then echo "TEST FAILED"; exit $rc; fi
rm perf_out

echo '============> Running F2C MPI+Tiled'
$MPI_INSTALL_PATH/bin/mpirun -np 2 ./lowdim_f2c_mpi_tiled OPS_TILING OPS_TILING_MAXDEPTH=2 > perf_out
grep "Max total runtime" perf_out
grep "PASSED" perf_out
rc=$?; if [[ $rc != 0 ]]; then echo "TEST FAILED"; exit $rc; fi
rm perf_out

echo '============> Running OMPOFFLOAD'
./lowdim_ompoffload OPS_BLOCK_SIZE_X=64 OPS_BLOCK_SIZE_Y=4 > perf_out
grep "Max total runtime" perf_out
grep "PASSED" perf_out
rc=$?; if [[ $rc != 0 ]]; then echo "TEST FAILED";exit $rc; fi
rm perf_out

echo '============> Running MPI+OMPOFFLOAD'
$MPI_INSTALL_PATH/bin/mpirun -np 2 ./lowdim_mpi_ompoffload OPS_BLOCK_SIZE_X=64 OPS_BLOCK_SIZE_Y=4 > perf_out
grep "Max total runtime" perf_out
grep "PASSED" perf_out
rc=$?; if [[ $rc != 0 ]]; then echo "TEST FAILED";exit $rc; fi
rm perf_out

echo '============> Running MPI+OMPOFFLOAD+Tiled'
$MPI_INSTALL_PATH/bin/mpirun -np 2 ./lowdim_mpi_ompoffload_tiled  OPS_TILING OPS_TILING_MAXDEPTH=2 OPS_BLOCK_SIZE_X=64 OPS_BLOCK_SIZE_Y=4 > perf_out
grep "Max total runtime" perf_out
grep "PASSED" perf_out
rc=$?; if [[ $rc != 0 ]]; then echo "TEST FAILED";exit $rc; fi
rm perf_out

echo "All PGI complier based applications ---- PASSED"
fi

echo "---------- Exiting Test Script "
