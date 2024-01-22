#!/bin/bash
set -e
cd ../../../ops/c

export SOURCE_INTEL=source_intel_2021.3_pythonenv
export SOURCE_PGI=source_pgi_nvhpc_23_pythonenv
export SOURCE_INTEL_SYCL=source_intel_2021.3_sycl_pythonenv
export SOURCE_AMD_HIP=source_amd_rocm-5.4.3_pythonenv

export AMOS=TRUE
#export DMOS=TRUE
#export TELOS=TRUE
#export KOS=TRUE

#<<comment

if [[ -v TELOS || -v KOS ]]; then

#============================ Test with Intel Classic Compilers==========================================
echo "Testing Intel classic complier based applications ---- "
cd $OPS_INSTALL_PATH/c
source ../../scripts/$SOURCE_INTEL
make clean
#make -j -B
make 
cd $OPS_INSTALL_PATH/../apps/c/multiDim_HDF5
#<<COMMENT
rm -f .generated
make -f Makefile.write clean
make -f Makefile.write IEEE=1

rm -f .generated
make -f Makefile.read clean
make -f Makefile.read IEEE=1

echo '============> Running OpenMP'
rm -rf write_data.h5 read_data.h5;
KMP_AFFINITY=compact OMP_NUM_THREADS=20 ./write_openmp 
KMP_AFFINITY=compact OMP_NUM_THREADS=20 ./read_openmp 
$HDF5_INSTALL_PATH/bin/h5diff write_data.h5 read_data.h5
rc=$?; if [[ $rc != 0 ]]; then echo "TEST FAILED";exit $rc; else echo "TEST PASSED"; fi

echo '============> Running MPI+OpenMP'
rm -rf write_data.h5 read_data.h5;
export OMP_NUM_THREADS=2;$MPI_INSTALL_PATH/bin/mpirun -np 10 ./write_mpi_openmp
export OMP_NUM_THREADS=2;$MPI_INSTALL_PATH/bin/mpirun -np 10 ./read_mpi_openmp
$HDF5_INSTALL_PATH/bin/h5diff write_data.h5 read_data.h5
rc=$?; if [[ $rc != 0 ]]; then echo "TEST FAILED";exit $rc; else echo "TEST PASSED"; fi

echo '============> Running DEV_MPI'
rm -rf write_data.h5 read_data.h5;
$MPI_INSTALL_PATH/bin/mpirun -np 20 ./write_dev_mpi
$MPI_INSTALL_PATH/bin/mpirun -np 20 ./read_dev_mpi
$HDF5_INSTALL_PATH/bin/h5diff write_data.h5 read_data.h5
rc=$?; if [[ $rc != 0 ]]; then echo "TEST FAILED";exit $rc; else echo "TEST PASSED"; fi


echo '============> Running MPI'
rm -rf write_data.h5 read_data.h5;
$MPI_INSTALL_PATH/bin/mpirun -np 20 ./write_mpi
$MPI_INSTALL_PATH/bin/mpirun -np 20 ./read_mpi
$HDF5_INSTALL_PATH/bin/h5diff write_data.h5 read_data.h5
rc=$?; if [[ $rc != 0 ]]; then echo "TEST FAILED";exit $rc; else echo "TEST PASSED"; fi

if [[ -v CUDA_INSTALL_PATH ]]; then
echo '============> Running CUDA'
rm -rf write_data.h5 read_data.h5;
./write_cuda OPS_BLOCK_SIZE_X=64 OPS_BLOCK_SIZE_Y=4
./read_cuda OPS_BLOCK_SIZE_X=64 OPS_BLOCK_SIZE_Y=4
$HDF5_INSTALL_PATH/bin/h5diff write_data.h5 read_data.h5
rc=$?; if [[ $rc != 0 ]]; then echo "TEST FAILED";exit $rc; else echo "TEST PASSED"; fi


echo '============> Running MPI+CUDA'
rm -rf write_data.h5 read_data.h5;
$MPI_INSTALL_PATH/bin/mpirun -np 2 ./write_mpi_cuda OPS_BLOCK_SIZE_X=64 OPS_BLOCK_SIZE_Y=4
$MPI_INSTALL_PATH/bin/mpirun -np 2 ./read_mpi_cuda OPS_BLOCK_SIZE_X=64 OPS_BLOCK_SIZE_Y=4
$HDF5_INSTALL_PATH/bin/h5diff write_data.h5 read_data.h5
rc=$?; if [[ $rc != 0 ]]; then echo "TEST FAILED";exit $rc; else echo "TEST PASSED"; fi


#echo '============> Running MPI+CUDA with GPU-Direct'
#rm write_data.h5 read_data.h5;
#MV2_USE_CUDA=1 $MPI_INSTALL_PATH/bin/mpirun -np 2 ./write_mpi_cuda -gpudirect OPS_BLOCK_SIZE_X=64 OPS_BLOCK_SIZE_Y=4
#grep "Total error:
#grep "Total Wall time
#grep "PASSED
#$HDF5_INSTALL_PATH/bin/h5diff write_data.h5 read_data.h5
#rc=$?; if [[ $rc != 0 ]]; then echo "TEST FAILED";exit $rc; else echo "TEST PASSED"; fi
#

fi
echo "All Intel classic complier based applications ---- PASSED"

fi

#comment

if [[ -v TELOS ]]; then
#============================ Test with Intel SYCL Compilers==========================================
echo "Testing Intel SYCL complier based applications ---- "
cd $OPS_INSTALL_PATH/c
source ../../scripts/$SOURCE_INTEL_SYCL
make clean
#make -j -B
make 
cd $OPS_INSTALL_PATH/../apps/c/multiDim_HDF5
#<<COMMENT
rm -f .generated
make -f Makefile.write clean
make -f Makefile.write IEEE=1

rm -f .generated
make -f Makefile.read clean
make -f Makefile.read IEEE=1


echo '============> Running SYCL on CPU'
rm -rf write_data.h5 read_data.h5;
KMP_AFFINITY=compact OMP_NUM_THREADS=20 ./write_sycl OPS_CL_DEVICE=0 OPS_BLOCK_SIZE_X=512 OPS_BLOCK_SIZE_Y=1
KMP_AFFINITY=compact OMP_NUM_THREADS=20 ./read_sycl OPS_CL_DEVICE=0 OPS_BLOCK_SIZE_X=512 OPS_BLOCK_SIZE_Y=1
$HDF5_INSTALL_PATH/bin/h5diff write_data.h5 read_data.h5
rc=$?; if [[ $rc != 0 ]]; then echo "TEST FAILED";exit $rc; else echo "TEST PASSED"; fi

echo '============> Running MPI+SYCL on CPU'
rm -rf write_data.h5 read_data.h5;
export OMP_NUM_THREADS=2;$MPI_INSTALL_PATH/bin/mpirun -np 10 ./write_mpi_sycl OPS_CL_DEVICE=1 OPS_BLOCK_SIZE_X=64 OPS_BLOCK_SIZE_Y=4
export OMP_NUM_THREADS=2;$MPI_INSTALL_PATH/bin/mpirun -np 10 ./read_mpi_sycl OPS_CL_DEVICE=1 OPS_BLOCK_SIZE_X=64 OPS_BLOCK_SIZE_Y=4
$HDF5_INSTALL_PATH/bin/h5diff write_data.h5 read_data.h5
rc=$?; if [[ $rc != 0 ]]; then echo "TEST FAILED";exit $rc; else echo "TEST PASSED"; fi


echo '============> Running MPI+SYCL Tiled on CPU'
rm -rf write_data.h5 read_data.h5;
$MPI_INSTALL_PATH/bin/mpirun -np 2 ./write_mpi_sycl_tiled OPS_CL_DEVICE=1 OPS_TILING OPS_BLOCK_SIZE_X=64 OPS_BLOCK_SIZE_Y=4
$MPI_INSTALL_PATH/bin/mpirun -np 2 ./read_mpi_sycl_tiled OPS_CL_DEVICE=1 OPS_TILING OPS_BLOCK_SIZE_X=64 OPS_BLOCK_SIZE_Y=4
$HDF5_INSTALL_PATH/bin/h5diff write_data.h5 read_data.h5
rc=$?; if [[ $rc != 0 ]]; then echo "TEST FAILED";exit $rc; else echo "TEST PASSED"; fi

#cleanup
rm -rf integers.txt*
echo "All Intel SYCL complier based applications ---- PASSED"

fi


if [[ -v TELOS ]]; then
#============================ Test with PGI Compilers==========================================
echo "Testing PGI/NVHPC complier based applications ---- "
cd $OPS_INSTALL_PATH/c
source ../../scripts/$SOURCE_PGI

make clean
make 
cd $OPS_INSTALL_PATH/../apps/c/multiDim_HDF5
rm -f .generated
make -f Makefile.write clean
make -f Makefile.write IEEE=1

rm -f .generated
make -f Makefile.read clean
make -f Makefile.read IEEE=1

echo '============> Running OpenMP'
rm -rf write_data.h5 read_data.h5;
KMP_AFFINITY=compact OMP_NUM_THREADS=20 ./write_openmp
KMP_AFFINITY=compact OMP_NUM_THREADS=20 ./read_openmp
$HDF5_INSTALL_PATH/bin/h5diff write_data.h5 read_data.h5
rc=$?; if [[ $rc != 0 ]]; then echo "TEST FAILED";exit $rc; else echo "TEST PASSED"; fi


echo '============> Running MPI+OpenMP'
rm -rf write_data.h5 read_data.h5;
export OMP_NUM_THREADS=2;$MPI_INSTALL_PATH/bin/mpirun -np 10 ./write_mpi_openmp
export OMP_NUM_THREADS=2;$MPI_INSTALL_PATH/bin/mpirun -np 10 ./read_mpi_openmp
$HDF5_INSTALL_PATH/bin/h5diff write_data.h5 read_data.h5
rc=$?; if [[ $rc != 0 ]]; then echo "TEST FAILED";exit $rc; else echo "TEST PASSED"; fi


echo '============> Running DEV_MPI'
rm -rf write_data.h5 read_data.h5;
$MPI_INSTALL_PATH/bin/mpirun -np 20 ./write_dev_mpi
$MPI_INSTALL_PATH/bin/mpirun -np 20 ./read_dev_mpi
$HDF5_INSTALL_PATH/bin/h5diff write_data.h5 read_data.h5
rc=$?; if [[ $rc != 0 ]]; then echo "TEST FAILED";exit $rc; else echo "TEST PASSED"; fi


echo '============> Running MPI'
rm -rf write_data.h5 read_data.h5;
$MPI_INSTALL_PATH/bin/mpirun -np 20 ./write_mpi
$MPI_INSTALL_PATH/bin/mpirun -np 20 ./read_mpi
$HDF5_INSTALL_PATH/bin/h5diff write_data.h5 read_data.h5
rc=$?; if [[ $rc != 0 ]]; then echo "TEST FAILED";exit $rc; else echo "TEST PASSED"; fi

if [[ -v CUDA_INSTALL_PATH ]]; then
echo '============> Running CUDA'
rm -rf write_data.h5 read_data.h5;
./write_cuda OPS_BLOCK_SIZE_X=64 OPS_BLOCK_SIZE_Y=4
./read_cuda OPS_BLOCK_SIZE_X=64 OPS_BLOCK_SIZE_Y=4
$HDF5_INSTALL_PATH/bin/h5diff write_data.h5 read_data.h5
rc=$?; if [[ $rc != 0 ]]; then echo "TEST FAILED";exit $rc; else echo "TEST PASSED"; fi

echo '============> Running MPI+CUDA'
rm -rf write_data.h5 read_data.h5;
$MPI_INSTALL_PATH/bin/mpirun -np 2 ./write_mpi_cuda OPS_BLOCK_SIZE_X=64 OPS_BLOCK_SIZE_Y=4
$MPI_INSTALL_PATH/bin/mpirun -np 2 ./read_mpi_cuda OPS_BLOCK_SIZE_X=64 OPS_BLOCK_SIZE_Y=4
$HDF5_INSTALL_PATH/bin/h5diff write_data.h5 read_data.h5
rc=$?; if [[ $rc != 0 ]]; then echo "TEST FAILED";exit $rc; else echo "TEST PASSED"; fi


#echo '============> Running MPI+CUDA with GPU-Direct'
#rm write_data.h5 read_data.h5;
#MV2_USE_CUDA=1 $MPI_INSTALL_PATH/bin/mpirun -np 2 ./write_mpi_cuda -gpudirect OPS_BLOCK_SIZE_X=64 OPS_BLOCK_SIZE_Y=4
#grep "Total error:
#grep "Total Wall time
#grep "PASSED
#$HDF5_INSTALL_PATH/bin/h5diff write_data.h5 read_data.h5
#rc=$?; if [[ $rc != 0 ]]; then echo "TEST FAILED";exit $rc; else echo "TEST PASSED"; fi
#

#echo '============> Running OpenACC'
#rm write_data.h5 read_data.h5;
#./write_openacc OPS_BLOCK_SIZE_X=64 OPS_BLOCK_SIZE_Y=4
#./read_openacc OPS_BLOCK_SIZE_X=64 OPS_BLOCK_SIZE_Y=4
#$HDF5_INSTALL_PATH/bin/h5diff write_data.h5 read_data.h5
#rc=$?; if [[ $rc != 0 ]]; then echo "TEST FAILED";exit $rc; else echo "TEST PASSED"; fi


#echo '============> Running MPI+OpenACC'
#rm write_data.h5 read_data.h5;
#$MPI_INSTALL_PATH/bin/mpirun -np 2 ./write_mpi_openacc OPS_BLOCK_SIZE_X=64 OPS_BLOCK_SIZE_Y=4
#$MPI_INSTALL_PATH/bin/mpirun -np 2 ./read_mpi_openacc OPS_BLOCK_SIZE_X=64 OPS_BLOCK_SIZE_Y=4
#$HDF5_INSTALL_PATH/bin/h5diff write_data.h5 read_data.h5
#rc=$?; if [[ $rc != 0 ]]; then echo "TEST FAILED";exit $rc; else echo "TEST PASSED"; fi
fi

echo '============> Running OMPOFFLOAD'
rm -rf write_data.h5 read_data.h5;
./write_ompoffload OPS_BLOCK_SIZE_X=64 OPS_BLOCK_SIZE_Y=4
./read_ompoffload OPS_BLOCK_SIZE_X=64 OPS_BLOCK_SIZE_Y=4
$HDF5_INSTALL_PATH/bin/h5diff write_data.h5 read_data.h5
rc=$?; if [[ $rc != 0 ]]; then echo "TEST FAILED";exit $rc; else echo "TEST PASSED"; fi

echo '============> Running MPI+OMPOFFLOAD'
rm -rf write_data.h5 read_data.h5;
$MPI_INSTALL_PATH/bin/mpirun -np 2 ./write_mpi_ompoffload OPS_BLOCK_SIZE_X=64 OPS_BLOCK_SIZE_Y=4
$MPI_INSTALL_PATH/bin/mpirun -np 2 ./read_mpi_ompoffload OPS_BLOCK_SIZE_X=64 OPS_BLOCK_SIZE_Y=4
$HDF5_INSTALL_PATH/bin/h5diff write_data.h5 read_data.h5
rc=$?; if [[ $rc != 0 ]]; then echo "TEST FAILED";exit $rc; else echo "TEST PASSED"; fi

#cleanup
rm integers.txt*

echo "All PGI complier based applications ---- PASSED"

fi

if [[ -v AMOS ]]; then

echo "Testing AMD HIP complier based applications ---- "
cd $OPS_INSTALL_PATH/c
source ../../scripts/$SOURCE_AMD_HIP


make clean
make 
cd $OPS_INSTALL_PATH/../apps/c/multiDim_HDF5
rm -f .generated
make -f Makefile.write clean
make -f Makefile.write write_hip write_mpi_hip IEEE=1

rm -f .generated
make -f Makefile.read clean
make -f Makefile.read read_hip read_mpi_hipIEEE=1

echo '============> Running HIP'
rm -rf write_data.h5 read_data.h5;
./write_hip OPS_BLOCK_SIZE_X=64 OPS_BLOCK_SIZE_Y=4
./read_hip OPS_BLOCK_SIZE_X=64 OPS_BLOCK_SIZE_Y=4
$HDF5_INSTALL_PATH/bin/h5diff write_data.h5 read_data.h5
rc=$?; if [[ $rc != 0 ]]; then echo "TEST FAILED";exit $rc; else echo "TEST PASSED"; fi

echo '============> Running MPI+HIP'
rm -rf write_data.h5 read_data.h5;
$MPI_INSTALL_PATH/bin/mpirun -np 2 ./write_mpi_hip OPS_BLOCK_SIZE_X=64 OPS_BLOCK_SIZE_Y=4
$MPI_INSTALL_PATH/bin/mpirun -np 2 ./read_mpi_hip OPS_BLOCK_SIZE_X=64 OPS_BLOCK_SIZE_Y=4
$HDF5_INSTALL_PATH/bin/h5diff write_data.h5 read_data.h5
rc=$?; if [[ $rc != 0 ]]; then echo "TEST FAILED";exit $rc; else echo "TEST PASSED"; fi

echo "All AMD HIP complier based applications ---- PASSED"

fi

echo "---------- Exiting Test Script "