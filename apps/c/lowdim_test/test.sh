#!/bin/bash
set -e
cd ../../../ops/c
#<<COMMENT
source ../../scripts/source_intel
make
cd -
make clean
rm -f .generated
make IEEE=1


#============================ Test lowdim with Intel Compilers ==========================================
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

echo '============> Running OpenCL on CPU'
./lowdim_opencl OPS_CL_DEVICE=0 OPS_BLOCK_SIZE_X=512 OPS_BLOCK_SIZE_Y=1 > perf_out
#$HDF5_INSTALL_PATH/bin/h5diff output.h5 output_seq.h5
#rc=$?; if [[ $rc != 0 ]]; then echo "TEST FAILED";exit $rc; fi;
grep "PASSED" perf_out
rc=$?; if [[ $rc != 0 ]]; then echo "TEST FAILED";exit $rc; fi
rm -rf perf_out output.h5

echo '============> Running OpenCL on GPU'
./lowdim_opencl OPS_CL_DEVICE=1 OPS_BLOCK_SIZE_X=32 OPS_BLOCK_SIZE_Y=4 > perf_out
./lowdim_opencl OPS_CL_DEVICE=1 OPS_BLOCK_SIZE_X=32 OPS_BLOCK_SIZE_Y=4 > perf_out
#$HDF5_INSTALL_PATH/bin/h5diff output.h5 output_seq.h5
#rc=$?; if [[ $rc != 0 ]]; then echo "TEST FAILED";exit $rc; fi;
grep "PASSED" perf_out
rc=$?; if [[ $rc != 0 ]]; then echo "TEST FAILED";exit $rc; fi
rm -rf perf_out output.h5

echo '============> Running MPI+OpenCL on CPU'
$MPI_INSTALL_PATH/bin/mpirun -np 20 ./lowdim_mpi_opencl OPS_CL_DEVICE=0 OPS_BLOCK_SIZE_X=256 OPS_BLOCK_SIZE_Y=1 > perf_out
$MPI_INSTALL_PATH/bin/mpirun -np 20 ./lowdim_mpi_opencl OPS_CL_DEVICE=0 OPS_BLOCK_SIZE_X=256 OPS_BLOCK_SIZE_Y=1 > perf_out
#$HDF5_INSTALL_PATH/bin/h5diff output.h5 output_seq.h5
#rc=$?; if [[ $rc != 0 ]]; then echo "TEST FAILED";exit $rc; fi;
grep "PASSED" perf_out
rc=$?; if [[ $rc != 0 ]]; then echo "TEST FAILED";exit $rc; fi
rm -rf perf_out output.h5

echo '============> Running MPI+OpenCL on GPU'
$MPI_INSTALL_PATH/bin/mpirun -np 2 ./lowdim_mpi_opencl OPS_CL_DEVICE=1 OPS_BLOCK_SIZE_X=32 OPS_BLOCK_SIZE_Y=4 > perf_out
$MPI_INSTALL_PATH/bin/mpirun -np 2 ./lowdim_mpi_opencl OPS_CL_DEVICE=1 OPS_BLOCK_SIZE_X=32 OPS_BLOCK_SIZE_Y=4 > perf_out
#$HDF5_INSTALL_PATH/bin/h5diff output.h5 output_seq.h5
#rc=$?; if [[ $rc != 0 ]]; then echo "TEST FAILED";exit $rc; fi;
grep "PASSED" perf_out
rc=$?; if [[ $rc != 0 ]]; then echo "TEST FAILED";exit $rc; fi
rm -rf perf_out output.h5

echo "All Intel complied applications PASSED : Moving no to PGI Compiler Tests "


cd -
source ../../scripts/source_pgi_19

make clean
make
cd -
make clean
make

#============================ Test lowdim with PGI Compilers ==========================================
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

echo '============> Running OpenCL on GPU'
./lowdim_opencl OPS_CL_DEVICE=1 OPS_BLOCK_SIZE_X=32 OPS_BLOCK_SIZE_Y=4 > perf_out
./lowdim_opencl OPS_CL_DEVICE=1 OPS_BLOCK_SIZE_X=32 OPS_BLOCK_SIZE_Y=4 > perf_out
#$HDF5_INSTALL_PATH/bin/h5diff output.h5 output_seq.h5
#rc=$?; if [[ $rc != 0 ]]; then echo "TEST FAILED";exit $rc; fi;
grep "PASSED" perf_out
rc=$?; if [[ $rc != 0 ]]; then echo "TEST FAILED";exit $rc; fi
rm -rf perf_out output.h5

#echo '============> Running MPI+OpenCL on CPU'
#$MPI_INSTALL_PATH/bin/mpirun -np 20 ./lowdim_mpi_opencl OPS_CL_DEVICE=0 OPS_BLOCK_SIZE_X=256 OPS_BLOCK_SIZE_Y=1 > perf_out
#$MPI_INSTALL_PATH/bin/mpirun -np 20 ./lowdim_mpi_opencl OPS_CL_DEVICE=0 OPS_BLOCK_SIZE_X=256 OPS_BLOCK_SIZE_Y=1 > perf_out


#grep "PASSED" perf_out
#rc=$?; if [[ $rc != 0 ]]; then echo "TEST FAILED";exit $rc; fi
#rm -rf perf_out output.h5

echo '============> Running MPI+OpenCL on GPU'
$MPI_INSTALL_PATH/bin/mpirun -np 2 ./lowdim_mpi_opencl OPS_CL_DEVICE=1 OPS_BLOCK_SIZE_X=32 OPS_BLOCK_SIZE_Y=4 > perf_out
$MPI_INSTALL_PATH/bin/mpirun -np 2 ./lowdim_mpi_opencl OPS_CL_DEVICE=1 OPS_BLOCK_SIZE_X=32 OPS_BLOCK_SIZE_Y=4 > perf_out
#$HDF5_INSTALL_PATH/bin/h5diff output.h5 output_seq.h5
#rc=$?; if [[ $rc != 0 ]]; then echo "TEST FAILED";exit $rc; fi;
grep "PASSED" perf_out
rc=$?; if [[ $rc != 0 ]]; then echo "TEST FAILED";exit $rc; fi
rm -rf perf_out output.h5

echo '============> Running OpenACC'
./lowdim_openacc OPS_BLOCK_SIZE_X=32 OPS_BLOCK_SIZE_Y=4 > perf_out
#$HDF5_INSTALL_PATH/bin/h5diff output.h5 output_seq.h5
#rc=$?; if [[ $rc != 0 ]]; then echo "TEST FAILED";exit $rc; fi;
grep "PASSED" perf_out
rc=$?; if [[ $rc != 0 ]]; then echo "TEST FAILED";exit $rc; fi
rm -rf perf_out output.h5
#COMMENT
echo '============> Running MPI+OpenACC'
$MPI_INSTALL_PATH/bin/mpirun -np 2 ./lowdim_mpi_openacc numawrap2 OPS_BLOCK_SIZE_X=32 OPS_BLOCK_SIZE_Y=4 > perf_out
#$HDF5_INSTALL_PATH/bin/h5diff output.h5 output_seq.h5
#rc=$?; if [[ $rc != 0 ]]; then echo "TEST FAILED";exit $rc; fi;
grep "PASSED" perf_out
rc=$?; if [[ $rc != 0 ]]; then echo "TEST FAILED";exit $rc; fi
rm -rf perf_out output.h5
echo "All PGI complied applications PASSED : Exiting Test Script "

