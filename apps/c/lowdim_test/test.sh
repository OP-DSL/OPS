#!/bin/bash
set -e
cd ../../../ops/c

export SOURCE_INTEL=source_intel_2021.3_pythonenv
export SOURCE_PGI=source_pgi_nvhpc_23_pythonenv
export SOURCE_INTEL_SYCL=source_intel_2021.3_sycl_pythonenv
export SOURCE_AMD_HIP=source_amd_rocm-5.4.3_pythonenv

#export AMOS=TRUE
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

cd $OPS_INSTALL_PATH/../apps/c/lowdim_test
make clean
make IEEE=1

echo '============> Running SEQ'
./lowdim_dev_mpi > perf_out
mv output.h5 output_seq.h5
grep "PASSED" perf_out
rc=$?; if [[ $rc != 0 ]]; then echo "TEST FAILED";exit $rc; fi
rm -rf perf_out

echo '============> Running OpenMP'
KMP_AFFINITY=compact OMP_NUM_THREADS=12 ./lowdim_openmp > perf_out
#$HDF5_INSTALL_PATH/bin/h5diff output.h5 output_seq.h5
#rc=$?; if [[ $rc != 0 ]]; then echo "TEST FAILED";exit $rc; fi;
grep "PASSED" perf_out
rc=$?; if [[ $rc != 0 ]]; then echo "TEST FAILED";exit $rc; fi
rm -rf perf_out output.h5 output.h5

echo '============> Running MPI+OpenMP'
export OMP_NUM_THREADS=2;$MPI_INSTALL_PATH/bin/mpirun -np 12 ./lowdim_mpi_openmp > perf_out
#$HDF5_INSTALL_PATH/bin/h5diff output.h5 output_seq.h5
#rc=$?; if [[ $rc != 0 ]]; then echo "TEST FAILED";exit $rc; fi;
grep "PASSED" perf_out
rc=$?; if [[ $rc != 0 ]]; then echo "TEST FAILED";exit $rc; fi
rm -rf perf_out output.h5

echo '============> Running DEV_MPI'
$MPI_INSTALL_PATH/bin/mpirun -np 20 ./lowdim_dev_mpi > perf_out
#$HDF5_INSTALL_PATH/bin/h5diff output.h5 output_seq.h5
#rc=$?; if [[ $rc != 0 ]]; then echo "TEST FAILED";exit $rc; fi;
grep "PASSED" perf_out
rc=$?; if [[ $rc != 0 ]]; then echo "TEST FAILED";exit $rc; fi
rm -rf perf_out output.h5

echo '============> Running MPI'
$MPI_INSTALL_PATH/bin/mpirun -np 20 ./lowdim_mpi > perf_out
#$HDF5_INSTALL_PATH/bin/h5diff output.h5 output_seq.h5
#rc=$?; if [[ $rc != 0 ]]; then echo "TEST FAILED";exit $rc; fi;
grep "PASSED" perf_out
rc=$?; if [[ $rc != 0 ]]; then echo "TEST FAILED";exit $rc; fi
rm -rf perf_out output.h5

if [[ -v CUDA_INSTALL_PATH ]]; then
echo '============> Running CUDA'
./lowdim_cuda OPS_BLOCK_SIZE_X=64 OPS_BLOCK_SIZE_Y=4 > perf_out
#$HDF5_INSTALL_PATH/bin/h5diff output.h5 output_seq.h5
#rc=$?; if [[ $rc != 0 ]]; then echo "TEST FAILED";exit $rc; fi;
grep "PASSED" perf_out
rc=$?; if [[ $rc != 0 ]]; then echo "TEST FAILED";exit $rc; fi
rm -rf perf_out output.h5

echo '============> Running MPI+CUDA'
$MPI_INSTALL_PATH/bin/mpirun -np 2 ./lowdim_mpi_cuda OPS_BLOCK_SIZE_X=64 OPS_BLOCK_SIZE_Y=4 > perf_out
#$HDF5_INSTALL_PATH/bin/h5diff output.h5 output_seq.h5
#rc=$?; if [[ $rc != 0 ]]; then echo "TEST FAILED";exit $rc; fi;
grep "PASSED" perf_out
rc=$?; if [[ $rc != 0 ]]; then echo "TEST FAILED";exit $rc; fi
rm -rf perf_out output.h5

#echo '============> Running MPI+CUDA with GPU-Direct'
#MV2_USE_CUDA=1 $MPI_INSTALL_PATH/bin/mpirun -np 2 ./lowdim_mpi_cuda -gpudirect OPS_BLOCK_SIZE_X=64 OPS_BLOCK_SIZE_Y=4 > perf_out

#grep "PASSED" perf_out
#rc=$?; if [[ $rc != 0 ]]; then echo "TEST FAILED";exit $rc; fi
#rm -rf perf_out output.h5

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
cd $OPS_INSTALL_PATH/../apps/c/lowdim_test

make clean
make IEEE=1 lowdim_sycl lowdim_mpi_sycl lowdim_mpi_sycl_tiled

echo '============> Running SYCL on CPU'
./lowdim_sycl OPS_CL_DEVICE=0 OPS_BLOCK_SIZE_X=512 OPS_BLOCK_SIZE_Y=1 > perf_out
#$HDF5_INSTALL_PATH/bin/h5diff output.h5 output_seq.h5
#rc=$?; if [[ $rc != 0 ]]; then echo "TEST FAILED";exit $rc; fi;
grep "PASSED" perf_out
rc=$?; if [[ $rc != 0 ]]; then echo "TEST FAILED";exit $rc; fi
rm -rf perf_out output.h5

echo '============> Running MPI+SYCL on CPU'
$MPI_INSTALL_PATH/bin/mpirun -np 12 ./lowdim_mpi_sycl OPS_CL_DEVICE=0 OPS_BLOCK_SIZE_X=256 OPS_BLOCK_SIZE_Y=1 > perf_out
#$HDF5_INSTALL_PATH/bin/h5diff output.h5 output_seq.h5
#rc=$?; if [[ $rc != 0 ]]; then echo "TEST FAILED";exit $rc; fi;
grep "PASSED" perf_out
rc=$?; if [[ $rc != 0 ]]; then echo "TEST FAILED";exit $rc; fi
rm -rf perf_out output.h5

echo '============> Running MPI+SYCL+Tiled on CPU'
$MPI_INSTALL_PATH/bin/mpirun -np 2 ./lowdim_mpi_sycl_tiled OPS_CL_DEVICE=1 OPS_BLOCK_SIZE_X=32 OPS_BLOCK_SIZE_Y=4 > perf_out
#$HDF5_INSTALL_PATH/bin/h5diff output.h5 output_seq.h5
#rc=$?; if [[ $rc != 0 ]]; then echo "TEST FAILED";exit $rc; fi;
grep "PASSED" perf_out
rc=$?; if [[ $rc != 0 ]]; then echo "TEST FAILED";exit $rc; fi
rm -rf perf_out output.h5

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
cd $OPS_INSTALL_PATH/../apps/c/lowdim_test
make clean
make IEEE=1

echo '============> Running OpenMP'
KMP_AFFINITY=compact OMP_NUM_THREADS=12 ./lowdim_openmp > perf_out
#$HDF5_INSTALL_PATH/bin/h5diff output.h5 output_seq.h5
#rc=$?; if [[ $rc != 0 ]]; then echo "TEST FAILED";exit $rc; fi;
grep "PASSED" perf_out
rc=$?; if [[ $rc != 0 ]]; then echo "TEST FAILED";exit $rc; fi
rm -rf perf_out output.h5

echo '============> Running MPI+OpenMP'
export OMP_NUM_THREADS=2;$MPI_INSTALL_PATH/bin/mpirun -np 12 ./lowdim_mpi_openmp > perf_out
#$HDF5_INSTALL_PATH/bin/h5diff output.h5 output_seq.h5
#rc=$?; if [[ $rc != 0 ]]; then echo "TEST FAILED";exit $rc; fi;
grep "PASSED" perf_out
rc=$?; if [[ $rc != 0 ]]; then echo "TEST FAILED";exit $rc; fi
rm -rf perf_out output.h5

echo '============> Running DEV_MPI'
$MPI_INSTALL_PATH/bin/mpirun -np 20 ./lowdim_dev_mpi > perf_out
#$HDF5_INSTALL_PATH/bin/h5diff output.h5 output_seq.h5
#rc=$?; if [[ $rc != 0 ]]; then echo "TEST FAILED";exit $rc; fi;
grep "PASSED" perf_out
rc=$?; if [[ $rc != 0 ]]; then echo "TEST FAILED";exit $rc; fi
rm -rf perf_out output.h5

echo '============> Running MPI'
$MPI_INSTALL_PATH/bin/mpirun -np 20 ./lowdim_mpi > perf_out
#$HDF5_INSTALL_PATH/bin/h5diff output.h5 output_seq.h5
#rc=$?; if [[ $rc != 0 ]]; then echo "TEST FAILED";exit $rc; fi;
grep "PASSED" perf_out
rc=$?; if [[ $rc != 0 ]]; then echo "TEST FAILED";exit $rc; fi
rm -rf perf_out output.h5

if [[ -v CUDA_INSTALL_PATH ]]; then
echo '============> Running CUDA'
./lowdim_cuda OPS_BLOCK_SIZE_X=64 OPS_BLOCK_SIZE_Y=4 > perf_out
#$HDF5_INSTALL_PATH/bin/h5diff output.h5 output_seq.h5
#rc=$?; if [[ $rc != 0 ]]; then echo "TEST FAILED";exit $rc; fi;
grep "PASSED" perf_out
rc=$?; if [[ $rc != 0 ]]; then echo "TEST FAILED";exit $rc; fi
rm -rf perf_out output.h5

echo '============> Running MPI+CUDA'
$MPI_INSTALL_PATH/bin/mpirun -np 2 ./lowdim_mpi_cuda OPS_BLOCK_SIZE_X=64 OPS_BLOCK_SIZE_Y=4 > perf_out
#$HDF5_INSTALL_PATH/bin/h5diff output.h5 output_seq.h5
#rc=$?; if [[ $rc != 0 ]]; then echo "TEST FAILED";exit $rc; fi;
grep "PASSED" perf_out
rc=$?; if [[ $rc != 0 ]]; then echo "TEST FAILED";exit $rc; fi
rm -rf perf_out output.h5

#echo '============> Running MPI+CUDA with GPU-Direct'
#MV2_USE_CUDA=1 $MPI_INSTALL_PATH/bin/mpirun -np 2 ./lowdim_mpi_cuda -gpudirect OPS_BLOCK_SIZE_X=64 OPS_BLOCK_SIZE_Y=4 > perf_out


#grep "PASSED" perf_out
#rc=$?; if [[ $rc != 0 ]]; then echo "TEST FAILED";exit $rc; fi
#rm -rf perf_out output.h5

#echo '============> Running OpenCL on CPU'
#./lowdim_opencl OPS_CL_DEVICE=0 OPS_BLOCK_SIZE_X=512 OPS_BLOCK_SIZE_Y=1 > perf_out


#grep "PASSED" perf_out
#rc=$?; if [[ $rc != 0 ]]; then echo "TEST FAILED";exit $rc; fi
#rm -rf perf_out output.h5
fi

echo '============> Running OMPOFFLOAD'
./lowdim_ompoffload OPS_BLOCK_SIZE_X=32 OPS_BLOCK_SIZE_Y=4 > perf_out
#$HDF5_INSTALL_PATH/bin/h5diff output.h5 output_seq.h5
#rc=$?; if [[ $rc != 0 ]]; then echo "TEST FAILED";exit $rc; fi;
grep "PASSED" perf_out
rc=$?; if [[ $rc != 0 ]]; then echo "TEST FAILED";exit $rc; fi
rm -rf perf_out output.h5
#COMMENT
echo '============> Running MPI+OMPOFFLOAD'
$MPI_INSTALL_PATH/bin/mpirun -np 2 ./lowdim_mpi_ompoffload numawrap2 OPS_BLOCK_SIZE_X=32 OPS_BLOCK_SIZE_Y=4 > perf_out
#$HDF5_INSTALL_PATH/bin/h5diff output.h5 output_seq.h5
#rc=$?; if [[ $rc != 0 ]]; then echo "TEST FAILED";exit $rc; fi;
grep "PASSED" perf_out
rc=$?; if [[ $rc != 0 ]]; then echo "TEST FAILED";exit $rc; fi
rm -rf perf_out output.h5
echo "All PGI complied applications PASSED : Exiting Test Script "

echo "All PGI complier based applications ---- PASSED"

fi

if [[ -v AMOS ]]; then

echo "Testing AMD HIP complier based applications ---- "
cd $OPS_INSTALL_PATH/c
source ../../scripts/$SOURCE_AMD_HIP
make clean
make
cd $OPS_INSTALL_PATH/../apps/c/lowdim_test

make clean
make lowdim_hip lowdim_mpi_hip

echo '============> Running HIP'
./lowdim_hip OPS_BLOCK_SIZE_X=64 OPS_BLOCK_SIZE_Y=4 > perf_out
#$HDF5_INSTALL_PATH/bin/h5diff output.h5 output_seq.h5
#rc=$?; if [[ $rc != 0 ]]; then echo "TEST FAILED";exit $rc; fi;
grep "PASSED" perf_out
rc=$?; if [[ $rc != 0 ]]; then echo "TEST FAILED";exit $rc; fi
rm -rf perf_out output.h5

echo '============> Running MPI+HIP'
mpirun --allow-run-as-root -np 2 ./lowdim_mpi_hip OPS_BLOCK_SIZE_X=64 OPS_BLOCK_SIZE_Y=4 > perf_out
#$HDF5_INSTALL_PATH/bin/h5diff output.h5 output_seq.h5
#rc=$?; if [[ $rc != 0 ]]; then echo "TEST FAILED";exit $rc; fi;
grep "PASSED" perf_out
rc=$?; if [[ $rc != 0 ]]; then echo "TEST FAILED";exit $rc; fi
rm -rf perf_out output.h5

echo "All AMD HIP complier based applications ---- PASSED"

fi

echo "---------- Exiting Test Script "