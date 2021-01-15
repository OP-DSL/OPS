#!/bin/bash
set -e
cd ../../../ops/c
if [[ -v HIP_INSTALL_PATH ]]; then
  source ../../scripts/$SOURCE_HIP
  make -j
  cd -
  
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

  echo “All HIP complied applications PASSED : Moving no to Intel Compiler Tests ”
  cd -
fi


source ../../scripts/$SOURCE_INTEL
make clean
make 
cd -
#<<COMMENT
rm -f .generated
../../../ops_translator/c/ops.py write.cpp
../../../ops_translator/c/ops.py read.cpp
make -f Makefile.write clean
make -f Makefile clean
make -f Makefile.write IEEE=1
make -f Makefile IEEE=1



#============================ Test write with Intel Compilers==========================================================
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

## OpenCL CPU test removed due to Intel OpenCL bug for long long types #####
#echo '============> Running OpenCL on CPU'
#rm -rf write_data.h5 read_data.h5;
#./write_opencl OPS_CL_DEVICE=0 OPS_BLOCK_SIZE_X=512 OPS_BLOCK_SIZE_Y=1
#./read_opencl OPS_CL_DEVICE=0 OPS_BLOCK_SIZE_X=512 OPS_BLOCK_SIZE_Y=1
#$HDF5_INSTALL_PATH/bin/h5diff write_data.h5 read_data.h5
#rc=$?; if [[ $rc != 0 ]]; then echo "TEST FAILED";exit $rc; else echo "TEST PASSED"; fi


echo '============> Running OpenCL on GPU'
rm -rf write_data.h5 read_data.h5;
./write_opencl OPS_CL_DEVICE=1 OPS_BLOCK_SIZE_X=32 OPS_BLOCK_SIZE_Y=4
./read_opencl OPS_CL_DEVICE=1 OPS_BLOCK_SIZE_X=32 OPS_BLOCK_SIZE_Y=4
$HDF5_INSTALL_PATH/bin/h5diff write_data.h5 read_data.h5
rc=$?; if [[ $rc != 0 ]]; then echo "TEST FAILED";exit $rc; else echo "TEST PASSED"; fi

## OpenCL CPU test removed due to Intel OpenCL bug for long long types #####
#echo '============> Running MPI+OpenCL on CPU'
#rm -rf write_data.h5 read_data.h5;
#$MPI_INSTALL_PATH/bin/mpirun -np 20 ./write_mpi_opencl OPS_CL_DEVICE=0 OPS_BLOCK_SIZE_X=256 OPS_BLOCK_SIZE_Y=1
#$MPI_INSTALL_PATH/bin/mpirun -np 20 ./read_mpi_opencl OPS_CL_DEVICE=0 OPS_BLOCK_SIZE_X=256 OPS_BLOCK_SIZE_Y=1
#$HDF5_INSTALL_PATH/bin/h5diff write_data.h5 read_data.h5
#rc=$?; if [[ $rc != 0 ]]; then echo "TEST FAILED";exit $rc; else echo "TEST PASSED"; fi


echo '============> Running MPI+OpenCL on GPU'
rm -rf write_data.h5 read_data.h5;
$MPI_INSTALL_PATH/bin/mpirun -np 2 ./write_mpi_opencl OPS_CL_DEVICE=1 OPS_BLOCK_SIZE_X=32 OPS_BLOCK_SIZE_Y=4
$MPI_INSTALL_PATH/bin/mpirun -np 2 ./read_mpi_opencl OPS_CL_DEVICE=1 OPS_BLOCK_SIZE_X=32 OPS_BLOCK_SIZE_Y=4
$HDF5_INSTALL_PATH/bin/h5diff write_data.h5 read_data.h5
rc=$?; if [[ $rc != 0 ]]; then echo "TEST FAILED";exit $rc; else echo "TEST PASSED"; fi

#COMMENT

echo "All Intel complied applications PASSED"

#cleanup
rm -rf integers.txt*

cd -
source ../../scripts/$SOURCE_PGI

make clean
make 
cd -
make -f Makefile.write clean
make -f Makefile clean
make -f Makefile.write IEEE=1
make -f Makefile IEEE=1

#============================ Test write with PGI Compilers==========================================================
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

#echo '============> Running OpenCL on CPU'
#rm write_data.h5 read_data.h5;
#./write_opencl OPS_CL_DEVICE=0 OPS_BLOCK_SIZE_X=512 OPS_BLOCK_SIZE_Y=1
#grep "Total error:
#grep "Total Wall time
#grep "PASSED
#rc=$?; if [[ $rc != 0 ]]; then echo "TEST FAILED";exit $rc; else echo "TEST PASSED"; fi
#

echo '============> Running OpenCL on GPU'
rm -rf write_data.h5 read_data.h5;
./write_opencl OPS_CL_DEVICE=1 OPS_BLOCK_SIZE_X=32 OPS_BLOCK_SIZE_Y=4
./read_opencl OPS_CL_DEVICE=1 OPS_BLOCK_SIZE_X=32 OPS_BLOCK_SIZE_Y=4
$HDF5_INSTALL_PATH/bin/h5diff write_data.h5 read_data.h5
rc=$?; if [[ $rc != 0 ]]; then echo "TEST FAILED";exit $rc; else echo "TEST PASSED"; fi


#echo '============> Running MPI+OpenCL on CPU'
#rm write_data.h5 read_data.h5;
#$MPI_INSTALL_PATH/bin/mpirun -np 20 ./write_mpi_opencl OPS_CL_DEVICE=0 OPS_BLOCK_SIZE_X=256 OPS_BLOCK_SIZE_Y=1
#$MPI_INSTALL_PATH/bin/mpirun -np 20 ./write_mpi_opencl OPS_CL_DEVICE=0 OPS_BLOCK_SIZE_X=256 OPS_BLOCK_SIZE_Y=1
#grep "Total error:
#grep "Total Wall time
#grep "PASSED
#$HDF5_INSTALL_PATH/bin/h5diff write_data.h5 read_data.h5
#rc=$?; if [[ $rc != 0 ]]; then echo "TEST FAILED";exit $rc; else echo "TEST PASSED"; fi
#

echo '============> Running MPI+OpenCL on GPU'
rm -rf write_data.h5 read_data.h5;
$MPI_INSTALL_PATH/bin/mpirun -np 2 ./write_mpi_opencl OPS_CL_DEVICE=1 OPS_BLOCK_SIZE_X=32 OPS_BLOCK_SIZE_Y=4
$MPI_INSTALL_PATH/bin/mpirun -np 2 ./read_mpi_opencl OPS_CL_DEVICE=1 OPS_BLOCK_SIZE_X=32 OPS_BLOCK_SIZE_Y=4
$HDF5_INSTALL_PATH/bin/h5diff write_data.h5 read_data.h5
rc=$?; if [[ $rc != 0 ]]; then echo "TEST FAILED";exit $rc; else echo "TEST PASSED"; fi


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

#cleanup
rm integers.txt*

echo "All PGI complied applications PASSED "
echo "All Tests PASSED : Exiting Test Script "
