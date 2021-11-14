#!/bin/bash
set -e
cd $OPS_INSTALL_PATH/c
<<COMMENT
if [ -x "$(command -v enroot)" ]; then
  cd -
  enroot start --root --mount $OPS_INSTALL_PATH/../:/tmp/OPS --rw cuda112hip sh -c 'cd /tmp/OPS/apps/c/mgrid; ./test.sh'
  grep "PASSED" perf_out
  rc=$?; if [[ $rc != 0 ]]; then echo "TEST FAILED";exit $rc; fi
  rm perf_out
  echo "All HIP complied applications PASSED"
fi

if [[ -v HIP_INSTALL_PATH ]]; then
  source ../../scripts/$SOURCE_HIP
  make -j -B
  cd -
  make clean
  rm -f .generated
  make mgrid_seq mgrid_hip mgrid_mpi_hip -j

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
  h5diff data.h5 data_ref.h5
  rc=$?; if [[ $rc != 0 ]]; then echo "TEST FAILED - HDF5 files comparison";exit $rc; fi;
  rm perf_out
  
  echo '============> Running MPI+HIP'
  mpirun --allow-run-as-root -np 2 ./mgrid_mpi_hip OPS_BLOCK_SIZE_X=64 OPS_BLOCK_SIZE_Y=4 > perf_out
  grep "Total Wall time" perf_out
  grep "PASSED" perf_out
  rc=$?; if [[ $rc != 0 ]]; then echo "TEST FAILED";exit $rc; fi
  h5diff data.h5 data_ref.h5
  rc=$?; if [[ $rc != 0 ]]; then echo "TEST FAILED - HDF5 files comparison";exit $rc; fi;
  rm perf_out
  echo "All HIP complied applications PASSED : Moving no to Intel Compiler Tests " > perf_out
  exit 0
fi
COMMENT
cd $OPS_INSTALL_PATH/c
source ../../scripts/$SOURCE_INTEL
make -j -B
cd $OPS_INSTALL_PATH/../apps/c/mgrid/
make clean
rm -f .generated
make IEEE=1


#============================ Test mgrid with Intel Compilers ==========================================
echo '============> Running SEQ'
./mgrid_seq > perf_out
grep "Total Wall time" perf_out
grep "PASSED" perf_out
rc=$?; if [[ $rc != 0 ]]; then echo "TEST FAILED";exit $rc; fi
rm perf_out
cp data.h5 data_ref.h5

#echo '============> Running OpenMP'
#KMP_AFFINITY=compact OMP_NUM_THREADS=12 ./mgrid_openmp > perf_out
#grep "Total Wall time" perf_out
#grep "PASSED" perf_out
#rc=$?; if [[ $rc != 0 ]]; then echo "TEST FAILED";exit $rc; fi
#$HDF5_INSTALL_PATH/bin/h5diff data.h5 data_ref.h5
#rc=$?; if [[ $rc != 0 ]]; then echo "TEST FAILED - HDF5 files comparison";exit $rc; fi;
#rm perf_out

#echo '============> Running MPI+OpenMP'
#export OMP_NUM_THREADS=2;$MPI_INSTALL_PATH/bin/mpirun -np 6 ./mgrid_mpi_openmp > perf_out
#grep "Total Wall time" perf_out
#grep "PASSED" perf_out
#rc=$?; if [[ $rc != 0 ]]; then echo "TEST FAILED";exit $rc; fi
#$HDF5_INSTALL_PATH/bin/h5diff data.h5 data_ref.h5
#rc=$?; if [[ $rc != 0 ]]; then echo "TEST FAILED - HDF5 files comparison";exit $rc; fi;
#rm perf_out

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

echo '============> Running MPI_inline'
$MPI_INSTALL_PATH/bin/mpirun -np 6 ./mgrid_mpi_inline > perf_out
grep "Total Wall time" perf_out
grep "PASSED" perf_out
rc=$?; if [[ $rc != 0 ]]; then echo "TEST FAILED";exit $rc; fi
$HDF5_INSTALL_PATH/bin/h5diff data.h5 data_ref.h5
rc=$?; if [[ $rc != 0 ]]; then echo "TEST FAILED - HDF5 files comparison";exit $rc; fi;
rm perf_out

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

#echo '============> Running OpenCL on CPU'
#./mgrid_opencl OPS_CL_DEVICE=0 OPS_BLOCK_SIZE_X=512 OPS_BLOCK_SIZE_Y=1 > perf_out
#grep "Total Wall time" perf_out
#grep "PASSED" perf_out
#rc=$?; if [[ $rc != 0 ]]; then echo "TEST FAILED";exit $rc; fi
#$HDF5_INSTALL_PATH/bin/h5diff data.h5 data_ref.h5
#rc=$?; if [[ $rc != 0 ]]; then echo "TEST FAILED - HDF5 files comparison";exit $rc; fi;
#rm perf_out

#echo '============> Running OpenCL on GPU'
#./mgrid_opencl OPS_CL_DEVICE=1 OPS_BLOCK_SIZE_X=32 OPS_BLOCK_SIZE_Y=4 > perf_out
#./mgrid_opencl OPS_CL_DEVICE=1 OPS_BLOCK_SIZE_X=32 OPS_BLOCK_SIZE_Y=4 > perf_out
#grep "Total Wall time" perf_out
#grep "PASSED" perf_out
#rc=$?; if [[ $rc != 0 ]]; then echo "TEST FAILED";exit $rc; fi
#$HDF5_INSTALL_PATH/bin/h5diff data.h5 data_ref.h5
#rc=$?; if [[ $rc != 0 ]]; then echo "TEST FAILED - HDF5 files comparison";exit $rc; fi;
#rm perf_out

#echo '============> Running MPI+OpenCL on CPU'
#$MPI_INSTALL_PATH/bin/mpirun -np 6 ./mgrid_mpi_opencl OPS_CL_DEVICE=0 OPS_BLOCK_SIZE_X=256 OPS_BLOCK_SIZE_Y=1 > perf_out
#$MPI_INSTALL_PATH/bin/mpirun -np 6 ./mgrid_mpi_opencl OPS_CL_DEVICE=0 OPS_BLOCK_SIZE_X=256 OPS_BLOCK_SIZE_Y=1 > perf_out
#grep "Total Wall time" perf_out
#grep "PASSED" perf_out
#rc=$?; if [[ $rc != 0 ]]; then echo "TEST FAILED";exit $rc; fi
#$HDF5_INSTALL_PATH/bin/h5diff data.h5 data_ref.h5
#rc=$?; if [[ $rc != 0 ]]; then echo "TEST FAILED - HDF5 files comparison";exit $rc; fi;
#rm perf_out

#echo '============> Running MPI+OpenCL on GPU'
#$MPI_INSTALL_PATH/bin/mpirun -np 2 ./mgrid_mpi_opencl OPS_CL_DEVICE=1 OPS_BLOCK_SIZE_X=32 OPS_BLOCK_SIZE_Y=4 > perf_out
#$MPI_INSTALL_PATH/bin/mpirun -np 2 ./mgrid_mpi_opencl OPS_CL_DEVICE=1 OPS_BLOCK_SIZE_X=32 OPS_BLOCK_SIZE_Y=4 > perf_out
#grep "Total Wall time" perf_out
#grep "PASSED" perf_out
#rc=$?; if [[ $rc != 0 ]]; then echo "TEST FAILED";exit $rc; fi
#$HDF5_INSTALL_PATH/bin/h5diff data.h5 data_ref.h5
#rc=$?; if [[ $rc != 0 ]]; then echo "TEST FAILED - HDF5 files comparison";exit $rc; fi;
#rm perf_out

rm -f data.h5 data_ref.h5
echo "All Intel complied applications PASSED : Moving no to PGI Compiler Tests "


cd -
source ../../scripts/$SOURCE_PGI

make clean
make
cd -
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


#echo '============> Running OpenMP'
#KMP_AFFINITY=compact OMP_NUM_THREADS=12 ./mgrid_openmp > perf_out
#grep "Total Wall time" perf_out
#grep "PASSED" perf_out
#rc=$?; if [[ $rc != 0 ]]; then echo "TEST FAILED";exit $rc; fi
#$HDF5_INSTALL_PATH/bin/h5diff data.h5 data_ref.h5
#rc=$?; if [[ $rc != 0 ]]; then echo "TEST FAILED - HDF5 files comparison";exit $rc; fi;
#rm perf_out

#echo '============> Running MPI+OpenMP'
#export OMP_NUM_THREADS=2;$MPI_INSTALL_PATH/bin/mpirun -np 6 ./mgrid_mpi_openmp > perf_out
#grep "Total Wall time" perf_out
#grep "PASSED" perf_out
#rc=$?; if [[ $rc != 0 ]]; then echo "TEST FAILED";exit $rc; fi
#$HDF5_INSTALL_PATH/bin/h5diff data.h5 data_ref.h5
#rc=$?; if [[ $rc != 0 ]]; then echo "TEST FAILED - HDF5 files comparison";exit $rc; fi;
#rm perf_out

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

echo '============> Running MPI_inline'
$MPI_INSTALL_PATH/bin/mpirun -np 6 ./mgrid_mpi_inline > perf_out
grep "Total Wall time" perf_out
grep "PASSED" perf_out
rc=$?; if [[ $rc != 0 ]]; then echo "TEST FAILED";exit $rc; fi
$HDF5_INSTALL_PATH/bin/h5diff data.h5 data_ref.h5
rc=$?; if [[ $rc != 0 ]]; then echo "TEST FAILED - HDF5 files comparison";exit $rc; fi;
rm perf_out


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

#echo '============> Running OpenCL on CPU'
#./mgrid_opencl OPS_CL_DEVICE=0 OPS_BLOCK_SIZE_X=512 OPS_BLOCK_SIZE_Y=1 > perf_out
#grep "Total Wall time" perf_out
#grep "PASSED" perf_out
#rc=$?; if [[ $rc != 0 ]]; then echo "TEST FAILED";exit $rc; fi
#$HDF5_INSTALL_PATH/bin/h5diff data.h5 data_ref.h5
#rc=$?; if [[ $rc != 0 ]]; then echo "TEST FAILED - HDF5 files comparison";exit $rc; fi;
#rm perf_out

#echo '============> Running OpenCL on GPU'
#./mgrid_opencl OPS_CL_DEVICE=1 OPS_BLOCK_SIZE_X=32 OPS_BLOCK_SIZE_Y=4 > perf_out
#./mgrid_opencl OPS_CL_DEVICE=1 OPS_BLOCK_SIZE_X=32 OPS_BLOCK_SIZE_Y=4 > perf_out
#grep "Total Wall time" perf_out
#grep "PASSED" perf_out
#rc=$?; if [[ $rc != 0 ]]; then echo "TEST FAILED";exit $rc; fi
#$HDF5_INSTALL_PATH/bin/h5diff data.h5 data_ref.h5
#rc=$?; if [[ $rc != 0 ]]; then echo "TEST FAILED - HDF5 files comparison";exit $rc; fi;
#rm perf_out

#echo '============> Running MPI+OpenCL on CPU'
#$MPI_INSTALL_PATH/bin/mpirun -np 6 ./mgrid_mpi_opencl OPS_CL_DEVICE=0 OPS_BLOCK_SIZE_X=256 OPS_BLOCK_SIZE_Y=1 > perf_out
#$MPI_INSTALL_PATH/bin/mpirun -np 6 ./mgrid_mpi_opencl OPS_CL_DEVICE=0 OPS_BLOCK_SIZE_X=256 OPS_BLOCK_SIZE_Y=1 > perf_out
#grep "Total Wall time" perf_out
#grep "PASSED" perf_out
#rc=$?; if [[ $rc != 0 ]]; then echo "TEST FAILED";exit $rc; fi
#$HDF5_INSTALL_PATH/bin/h5diff data.h5 data_ref.h5
#rc=$?; if [[ $rc != 0 ]]; then echo "TEST FAILED - HDF5 files comparison";exit $rc; fi;
#rm perf_out

#echo '============> Running MPI+OpenCL on GPU'
#$MPI_INSTALL_PATH/bin/mpirun -np 2 ./mgrid_mpi_opencl OPS_CL_DEVICE=1 OPS_BLOCK_SIZE_X=32 OPS_BLOCK_SIZE_Y=4 > perf_out
#$MPI_INSTALL_PATH/bin/mpirun -np 2 ./mgrid_mpi_opencl OPS_CL_DEVICE=1 OPS_BLOCK_SIZE_X=32 OPS_BLOCK_SIZE_Y=4 > perf_out
#grep "Total Wall time" perf_out
#grep "PASSED" perf_out
#rc=$?; if [[ $rc != 0 ]]; then echo "TEST FAILED";exit $rc; fi
#$HDF5_INSTALL_PATH/bin/h5diff data.h5 data_ref.h5
#rc=$?; if [[ $rc != 0 ]]; then echo "TEST FAILED - HDF5 files comparison";exit $rc; fi;
#rm perf_out

echo '============> Running OpenACC'
./mgrid_openacc OPS_BLOCK_SIZE_X=32 OPS_BLOCK_SIZE_Y=4 > perf_out
grep "Total Wall time" perf_out
grep "PASSED" perf_out
rc=$?; if [[ $rc != 0 ]]; then echo "TEST FAILED";exit $rc; fi
$HDF5_INSTALL_PATH/bin/h5diff data.h5 data_ref.h5
rc=$?; if [[ $rc != 0 ]]; then echo "TEST FAILED - HDF5 files comparison";exit $rc; fi;
rm perf_out
#COMMENT
echo '============> Running MPI+OpenACC'
$MPI_INSTALL_PATH/bin/mpirun -np 2 ./mgrid_mpi_openacc OPS_BLOCK_SIZE_X=32 OPS_BLOCK_SIZE_Y=4 > perf_out
grep "Total Wall time" perf_out
grep "PASSED" perf_out
rc=$?; if [[ $rc != 0 ]]; then echo "TEST FAILED";exit $rc; fi
$HDF5_INSTALL_PATH/bin/h5diff data.h5 data_ref.h5
rc=$?; if [[ $rc != 0 ]]; then echo "TEST FAILED - HDF5 files comparison";exit $rc; fi;
rm perf_out

rm -f data.h5 data_ref.h5
echo "All PGI complied applications PASSED : Exiting Test Script "

