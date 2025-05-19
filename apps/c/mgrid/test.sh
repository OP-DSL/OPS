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

if [[ -v TELOS || -v KOS ]]; then

#============================ Test with Intel Classic Compilers==========================================
echo "Testing Intel classic complier based applications ---- "  
cd $OPS_INSTALL_PATH/c
source ../../scripts/$SOURCE_INTEL
#make -j -B
make clean
make
cd $OPS_INSTALL_PATH/../apps/c/mgrid/
make clean
rm -f .generated
make IEEE=1


echo '============> Running SEQ'
./mgrid_seq > perf_out
grep "Total Wall time" perf_out
grep "PASSED" perf_out
rc=$?; if [[ $rc != 0 ]]; then echo "TEST FAILED";exit $rc; fi
rm perf_out
cp data.h5 data_ref.h5

echo '============> Running OpenMP'
KMP_AFFINITY=compact OMP_NUM_THREADS=12 ./mgrid_openmp > perf_out
grep "Total Wall time" perf_out
grep "PASSED" perf_out
rc=$?; if [[ $rc != 0 ]]; then echo "TEST FAILED";exit $rc; fi
$HDF5_INSTALL_PATH/bin/h5diff data.h5 data_ref.h5
rc=$?; if [[ $rc != 0 ]]; then echo "TEST FAILED - HDF5 files comparison";exit $rc; fi;
rm perf_out

echo '============> Running MPI+OpenMP'
export OMP_NUM_THREADS=2;$MPI_INSTALL_PATH/bin/mpirun -np 6 ./mgrid_mpi_openmp > perf_out
grep "Total Wall time" perf_out
grep "PASSED" perf_out
rc=$?; if [[ $rc != 0 ]]; then echo "TEST FAILED";exit $rc; fi
$HDF5_INSTALL_PATH/bin/h5diff data.h5 data_ref.h5
rc=$?; if [[ $rc != 0 ]]; then echo "TEST FAILED - HDF5 files comparison";exit $rc; fi;
rm perf_out

#echo '============> Running DEV_MPI'
#$MPI_INSTALL_PATH/bin/mpirun -np 6 ./mgrid_dev_mpi > perf_out
#grep "Total Wall time" perf_out
#grep "PASSED" perf_out
#rc=$?; if [[ $rc != 0 ]]; then echo "TEST FAILED";exit $rc; fi
#$HDF5_INSTALL_PATH/bin/h5diff data.h5 data_ref.h5
#rc=$?; if [[ $rc != 0 ]]; then echo "TEST FAILED - HDF5 files comparison";exit $rc; fi;
#rm perf_out

echo '============> Running MPI'
$MPI_INSTALL_PATH/bin/mpirun -np 6 ./mgrid_mpi > perf_out
grep "Total Wall time" perf_out
grep "PASSED" perf_out
rc=$?; if [[ $rc != 0 ]]; then echo "TEST FAILED";exit $rc; fi
$HDF5_INSTALL_PATH/bin/h5diff data.h5 data_ref.h5
rc=$?; if [[ $rc != 0 ]]; then echo "TEST FAILED - HDF5 files comparison";exit $rc; fi;
rm perf_out


if [[ -v CUDA_INSTALL_PATH ]]; then

echo '============> Running CUDA'
./mgrid_cuda OPS_BLOCK_SIZE_X=64 OPS_BLOCK_SIZE_Y=4 > perf_out
grep "Total Wall time" perf_out
grep "PASSED" perf_out
rc=$?; if [[ $rc != 0 ]]; then echo "TEST FAILED";exit $rc; fi
$HDF5_INSTALL_PATH/bin/h5diff data.h5 data_ref.h5
rc=$?; if [[ $rc != 0 ]]; then echo "TEST FAILED - HDF5 files comparison";exit $rc; fi;
rm perf_out

echo '============> Running MPI+CUDA'
$MPI_INSTALL_PATH/bin/mpirun -np 2 ./mgrid_mpi_cuda OPS_BLOCK_SIZE_X=64 OPS_BLOCK_SIZE_Y=4 > perf_out
grep "Total Wall time" perf_out
grep "PASSED" perf_out
rc=$?; if [[ $rc != 0 ]]; then echo "TEST FAILED";exit $rc; fi
$HDF5_INSTALL_PATH/bin/h5diff data.h5 data_ref.h5
rc=$?; if [[ $rc != 0 ]]; then echo "TEST FAILED - HDF5 files comparison";exit $rc; fi;
rm perf_out

#echo '============> Running MPI+CUDA with GPU-Direct'
#MV2_USE_CUDA=1 $MPI_INSTALL_PATH/bin/mpirun -np 2 ./mgrid_mpi_cuda -gpudirect OPS_BLOCK_SIZE_X=64 OPS_BLOCK_SIZE_Y=4 > perf_out
#grep "Total Wall time" perf_out
#grep "PASSED" perf_out
#rc=$?; if [[ $rc != 0 ]]; then echo "TEST FAILED";exit $rc; fi
#$HDF5_INSTALL_PATH/bin/h5diff data.h5 data_ref.h5
#rc=$?; if [[ $rc != 0 ]]; then echo "TEST FAILED - HDF5 files comparison";exit $rc; fi;
#rm perf_out
fi

rm -f data.h5 
fi
echo "All Intel classic complier based applications ---- PASSED"

if [[ -v TELOS ]]; then

echo "Testing Intel SYCL complier based applications ---- "
cd $OPS_INSTALL_PATH/c
source ../../scripts/$SOURCE_INTEL_SYCL
#make -j -B
make clean
make
cd $OPS_INSTALL_PATH/../apps/c/mgrid/

make clean
#make IEEE=1 -j
make IEEE=1 mgrid_sycl mgrid_mpi_sycl mgrid_mpi_sycl_tiled


echo '============> Running SYCL'
./mgrid_sycl > perf_out
grep "Total Wall time" perf_out
grep "PASSED" perf_out
rc=$?; if [[ $rc != 0 ]]; then echo "TEST FAILED";exit $rc; fi
$HDF5_INSTALL_PATH/bin/h5diff data.h5 data_ref.h5
rc=$?; if [[ $rc != 0 ]]; then echo "TEST FAILED - HDF5 files comparison";exit $rc; fi;
rm perf_out

echo '============> Running MPI+SYCL'
$MPI_INSTALL_PATH/bin/mpirun -np 6 ./mgrid_mpi_sycl > perf_out
grep "Total Wall time" perf_out
grep "PASSED" perf_out
rc=$?; if [[ $rc != 0 ]]; then echo "TEST FAILED";exit $rc; fi
$HDF5_INSTALL_PATH/bin/h5diff data.h5 data_ref.h5
rc=$?; if [[ $rc != 0 ]]; then echo "TEST FAILED - HDF5 files comparison";exit $rc; fi;
rm perf_out

echo '============> Running MPI+SYCL Tiled'
$MPI_INSTALL_PATH/bin/mpirun -np 6 ./mgrid_mpi_sycl_tiled OPS_TILING > perf_out
grep "Total Wall time" perf_out
grep "PASSED" perf_out
rc=$?; if [[ $rc != 0 ]]; then echo "TEST FAILED";exit $rc; fi
$HDF5_INSTALL_PATH/bin/h5diff data.h5 data_ref.h5
rc=$?; if [[ $rc != 0 ]]; then echo "TEST FAILED - HDF5 files comparison";exit $rc; fi;
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
cd $OPS_INSTALL_PATH/../apps/c/mgrid/
make clean
make

#============================ Test mgrid with PGI Compilers ==========================================
echo '============> Running SEQ'
./mgrid_seq > perf_out
grep "Total Wall time" perf_out
grep "PASSED" perf_out
rc=$?; if [[ $rc != 0 ]]; then echo "TEST FAILED";exit $rc; fi
rm perf_out
cp data.h5 data_ref.h5


echo '============> Running OpenMP'
KMP_AFFINITY=compact OMP_NUM_THREADS=12 ./mgrid_openmp > perf_out
grep "Total Wall time" perf_out
grep "PASSED" perf_out
rc=$?; if [[ $rc != 0 ]]; then echo "TEST FAILED";exit $rc; fi
$HDF5_INSTALL_PATH/bin/h5diff data.h5 data_ref.h5
rc=$?; if [[ $rc != 0 ]]; then echo "TEST FAILED - HDF5 files comparison";exit $rc; fi;
rm perf_out

echo '============> Running MPI+OpenMP'
export OMP_NUM_THREADS=2;$MPI_INSTALL_PATH/bin/mpirun -np 6 ./mgrid_mpi_openmp > perf_out
grep "Total Wall time" perf_out
grep "PASSED" perf_out
rc=$?; if [[ $rc != 0 ]]; then echo "TEST FAILED";exit $rc; fi
$HDF5_INSTALL_PATH/bin/h5diff data.h5 data_ref.h5
rc=$?; if [[ $rc != 0 ]]; then echo "TEST FAILED - HDF5 files comparison";exit $rc; fi;
rm perf_out

echo '============> Running DEV_MPI'
$MPI_INSTALL_PATH/bin/mpirun -np 6 ./mgrid_dev_mpi > perf_out
grep "Total Wall time" perf_out
grep "PASSED" perf_out
rc=$?; if [[ $rc != 0 ]]; then echo "TEST FAILED";exit $rc; fi
$HDF5_INSTALL_PATH/bin/h5diff data.h5 data_ref.h5
rc=$?; if [[ $rc != 0 ]]; then echo "TEST FAILED - HDF5 files comparison";exit $rc; fi;
rm perf_out

echo '============> Running MPI'
$MPI_INSTALL_PATH/bin/mpirun -np 6 ./mgrid_mpi > perf_out
grep "Total Wall time" perf_out
grep "PASSED" perf_out
rc=$?; if [[ $rc != 0 ]]; then echo "TEST FAILED";exit $rc; fi
$HDF5_INSTALL_PATH/bin/h5diff data.h5 data_ref.h5
rc=$?; if [[ $rc != 0 ]]; then echo "TEST FAILED - HDF5 files comparison";exit $rc; fi;
rm perf_out


if [[ -v CUDA_INSTALL_PATH ]]; then
echo '============> Running CUDA'
./mgrid_cuda OPS_BLOCK_SIZE_X=64 OPS_BLOCK_SIZE_Y=4 > perf_out
grep "Total Wall time" perf_out
grep "PASSED" perf_out
rc=$?; if [[ $rc != 0 ]]; then echo "TEST FAILED";exit $rc; fi
$HDF5_INSTALL_PATH/bin/h5diff data.h5 data_ref.h5
rc=$?; if [[ $rc != 0 ]]; then echo "TEST FAILED - HDF5 files comparison";exit $rc; fi;
rm perf_out

echo '============> Running MPI+CUDA'
$MPI_INSTALL_PATH/bin/mpirun -np 2 ./mgrid_mpi_cuda OPS_BLOCK_SIZE_X=64 OPS_BLOCK_SIZE_Y=4 > perf_out
grep "Total Wall time" perf_out
grep "PASSED" perf_out
rc=$?; if [[ $rc != 0 ]]; then echo "TEST FAILED";exit $rc; fi
$HDF5_INSTALL_PATH/bin/h5diff data.h5 data_ref.h5
rc=$?; if [[ $rc != 0 ]]; then echo "TEST FAILED - HDF5 files comparison";exit $rc; fi;
rm perf_out

#echo '============> Running MPI+CUDA with GPU-Direct'
#MV2_USE_CUDA=1 $MPI_INSTALL_PATH/bin/mpirun -np 2 ./mgrid_mpi_cuda -gpudirect OPS_BLOCK_SIZE_X=64 OPS_BLOCK_SIZE_Y=4 > perf_out
#grep "Total Wall time" perf_out
#grep "PASSED" perf_out
#rc=$?; if [[ $rc != 0 ]]; then echo "TEST FAILED";exit $rc; fi
#$HDF5_INSTALL_PATH/bin/h5diff data.h5 data_ref.h5
#rc=$?; if [[ $rc != 0 ]]; then echo "TEST FAILED - HDF5 files comparison";exit $rc; fi;
#rm perf_out

#echo '============> Running OpenACC'
#./mgrid_openacc OPS_BLOCK_SIZE_X=32 OPS_BLOCK_SIZE_Y=4 > perf_out
#grep "Total Wall time" perf_out
#grep "PASSED" perf_out
#rc=$?; if [[ $rc != 0 ]]; then echo "TEST FAILED";exit $rc; fi
#$HDF5_INSTALL_PATH/bin/h5diff data.h5 data_ref.h5
#rc=$?; if [[ $rc != 0 ]]; then echo "TEST FAILED - HDF5 files comparison";exit $rc; fi;
#rm perf_out
#COMMENT
#echo '============> Running MPI+OpenACC'
#$MPI_INSTALL_PATH/bin/mpirun -np 2 ./mgrid_mpi_openacc OPS_BLOCK_SIZE_X=32 OPS_BLOCK_SIZE_Y=4 > perf_out
#grep "Total Wall time" perf_out
#grep "PASSED" perf_out
#rc=$?; if [[ $rc != 0 ]]; then echo "TEST FAILED";exit $rc; fi
#$HDF5_INSTALL_PATH/bin/h5diff data.h5 data_ref.h5
#rc=$?; if [[ $rc != 0 ]]; then echo "TEST FAILED - HDF5 files comparison";exit $rc; fi;
#rm perf_out
fi

echo '============> Running OMPOFFLOAD'
./mgrid_ompoffload OPS_BLOCK_SIZE_X=64 OPS_BLOCK_SIZE_Y=4 > perf_out
grep "Total Wall time" perf_out
grep "PASSED" perf_out
rc=$?; if [[ $rc != 0 ]]; then echo "TEST FAILED";exit $rc; fi
$HDF5_INSTALL_PATH/bin/h5diff data.h5 data_ref.h5
rc=$?; if [[ $rc != 0 ]]; then echo "TEST FAILED - HDF5 files comparison";exit $rc; fi;
rm perf_out

echo '============> Running MPI+OMPOFFLOAD'
$MPI_INSTALL_PATH/bin/mpirun -np 2 ./mgrid_mpi_ompoffload OPS_BLOCK_SIZE_X=64 OPS_BLOCK_SIZE_Y=4 > perf_out
grep "Total Wall time" perf_out
grep "PASSED" perf_out
rc=$?; if [[ $rc != 0 ]]; then echo "TEST FAILED";exit $rc; fi
$HDF5_INSTALL_PATH/bin/h5diff data.h5 data_ref.h5
rc=$?; if [[ $rc != 0 ]]; then echo "TEST FAILED - HDF5 files comparison";exit $rc; fi;
rm perf_out

rm -f data.h5
echo "All PGI complier based applications ---- PASSED"

fi

if [[ -v AMOS ]]; then

echo "Testing AMD HIP complier based applications ---- "
cd $OPS_INSTALL_PATH/c
source ../../scripts/$SOURCE_AMD_HIP
#make -j -B
make clean
make 
cd $OPS_INSTALL_PATH/../apps/c/mgrid

make clean
rm -f .generated
#make IEEE=1 -j
make IEEE=1 mgrid_seq mgrid_hip mgrid_mpi_hip #mgrid_hip_tiled mgrid_mpi_hip_tiled

echo '============> Running SEQ'
./mgrid_seq > perf_out
grep "Total Wall time" perf_out
grep "PASSED" perf_out
rc=$?; if [[ $rc != 0 ]]; then echo "TEST FAILED";exit $rc; fi
rm perf_out
cp data.h5 data_ref.h5

echo '============> Running HIP'
./mgrid_hip OPS_BLOCK_SIZE_X=64 OPS_BLOCK_SIZE_Y=4 > perf_out
grep "Total Wall time" perf_out
grep "PASSED" perf_out
rc=$?; if [[ $rc != 0 ]]; then echo "TEST FAILED";exit $rc; fi
$HDF5_INSTALL_PATH/bin/h5diff data.h5 data_ref.h5
rc=$?; if [[ $rc != 0 ]]; then echo "TEST FAILED - HDF5 files comparison";exit $rc; fi;
rm perf_out

echo '============> Running MPI+HIP'
$MPI_INSTALL_PATH/bin/mpirun -np 2 ./mgrid_mpi_hip OPS_BLOCK_SIZE_X=64 OPS_BLOCK_SIZE_Y=4 > perf_out
grep "Total Wall time" perf_out
grep "PASSED" perf_out
rc=$?; if [[ $rc != 0 ]]; then echo "TEST FAILED";exit $rc; fi
$HDF5_INSTALL_PATH/bin/h5diff data.h5 data_ref.h5
rc=$?; if [[ $rc != 0 ]]; then echo "TEST FAILED - HDF5 files comparison";exit $rc; fi;
rm perf_out

rm -f data.h5 data_ref.h5

echo "All AMD HIP complier based applications ---- PASSED"

fi

echo "---------- Exiting Test Script "

