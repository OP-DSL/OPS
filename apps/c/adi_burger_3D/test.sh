#!/bin/bash
set -e
cd $OPS_INSTALL_PATH/c

export SOURCE_INTEL=source_oneapi_sycl_pythonenv

#export AMOS=TRUE
export DEMOS=TRUE
#export TELOS=TRUE
#export KOS=TRUE

if [[ -v TELOS || -v DEMOS || -v KOS ]]; then

#============================ Test with Intel Classic Compilers==========================================
echo "Testing Intel classic complier based applications ---- "

source ../../scripts/$SOURCE_INTEL

#<<COMMENT
# Build and copy Referance application from the TDMA Library
# build lib first
cd $TDMA_INSTALL_PATH/../build
rm -rf ./*
cmake .. -DCUDA_cublas_LIBRARY=$CUDA_INSTALL_PATH/lib64/libcublas.so -DCMAKE_BUILD_TYPE=Release -DBUILD_FOR_CPU=ON -DBUILD_FOR_GPU=ON -DBUILD_FOR_SN=ON -DBUILD_FOR_MPI=ON -DCMAKE_INSTALL_PREFIX=$TDMA_INSTALL_PATH/../

make
make install


#build OPS
cd $OPS_INSTALL_PATH/c
make clean
make IEEE=1

#COMMENT

#now build application
cd $OPS_INSTALL_PATH/../apps/c/adi_burger_3D
make clean
make IEEE=1 adi_burger_dev_seq adi_burger_dev_mpi adi_burger_seq adi_burger_mpi adi_burger_openmp adi_burger_mpi_openmp adi_burger_cuda adi_burger_mpi_cuda


rm -rf *.h5

# set Relative Tolarance for solution check -- h5diff check only
export TOL="1.000E-14"

#============== Run refernace (OPS sequential) solution ========================
echo '============> Running Referance (OPS sequential) Solution adi_burger_dev_seq'
pwd
./adi_burger_dev_seq > perf_out
grep "Total Wall time" perf_out
mv Burger3DRes.h5 Burger3DRes_dev_seq.h5
rm perf_out

#============== Run ops adi_burger application ========================

echo '============> Running SEQ'
./adi_burger_seq > perf_out
mv Burger3DRes.h5 Burger3DRes_seq.h5
grep "Total Wall time" perf_out
$HDF5_INSTALL_PATH/bin/h5diff -p $TOL Burger3DRes_dev_seq.h5 Burger3DRes_seq.h5 > diff_out
if [ -s ./diff_out ]
then
    echo "File not empty - Solution Not Valid";exit 1;
else
    echo "PASSED"
fi
rm -rf perf_out diff_out adi_burger_seq.h5


echo '============> Running OpenMP'
KMP_AFFINITY=compact OMP_NUM_THREADS=20 ./adi_burger_openmp > perf_out
mv Burger3DRes.h5 Burger3DRes_omp.h5
grep "Total Wall time" perf_out
$HDF5_INSTALL_PATH/bin/h5diff -p $TOL Burger3DRes_dev_seq.h5 Burger3DRes_omp.h5 > diff_out
if [ -s ./diff_out ]
then
    echo "File not empty - Solution Not Valid";exit 1;
else
    echo "PASSED"
fi
rm -rf perf_out diff_out adi_burger_omp.h5

echo '============> Running CUDA'
./adi_burger_cuda OPS_BLOCK_SIZE_X=64 OPS_BLOCK_SIZE_Y=4 > perf_out
mv Burger3DRes.h5 Burger3DRes_cuda.h5
grep "Total Wall time" perf_out
$HDF5_INSTALL_PATH/bin/h5diff -p $TOL Burger3DRes_dev_seq.h5 Burger3DRes_cuda.h5 > diff_out
if [ -s ./diff_out ]
then
    echo "File not empty - Solution Not Valid";exit 1;
else
    echo "PASSED"
fi
rm -rf perf_out diff_out adi_burger_cuda.h5

echo '============> Running MPI - Gather Scatter'
$MPI_INSTALL_PATH/bin/mpirun -n 8 ./adi_burger_mpi -halo 1 -m 0 > perf_out
mv Burger3DRes.h5 Burger3DRes_mpi.h5
grep "Total Wall time" perf_out
$HDF5_INSTALL_PATH/bin/h5diff -p $TOL Burger3DRes_dev_seq.h5 Burger3DRes_mpi.h5 > diff_out
if [ -s ./diff_out ]
then
    echo "File not empty - Solution Not Valid";exit 1;
else
    echo "PASSED"
fi
rm -rf perf_out diff_out adi_burger_mpi.h5

echo '============> Running MPI - LATENCY HIDING 2 STEP'
$MPI_INSTALL_PATH/bin/mpirun -n 8 ./adi_burger_mpi -halo 1 -m 2 -bx 16384 -by 16384 -bz 16384 > perf_out
mv Burger3DRes.h5 Burger3DRes_mpi.h5
grep "Total Wall time" perf_out
$HDF5_INSTALL_PATH/bin/h5diff -p $TOL Burger3DRes_dev_seq.h5 Burger3DRes_mpi.h5 > diff_out
if [ -s ./diff_out ]
then
    echo "File not empty - Solution Not Valid";exit 1;
else
    echo "PASSED"
fi
rm -rf adi_burger_mpi.h5

echo '============> Running MPI - LATENCY HIDING INTERLEAVED'
$MPI_INSTALL_PATH/bin/mpirun -n 8 ./adi_burger_mpi -halo 1 -m 3 -bx 16384 -by 16384 -bz 16384 > perf_out
mv Burger3DRes.h5 Burger3DRes_mpi.h5
grep "Total Wall time" perf_out
$HDF5_INSTALL_PATH/bin/h5diff -p $TOL Burger3DRes_dev_seq.h5 Burger3DRes_mpi.h5 > diff_out
if [ -s ./diff_out ]
then
    echo "File not empty - Solution Not Valid";exit 1;
else
    echo "PASSED"
fi
rm -rf adi_burger_mpi.h5 perf_out diff_out

echo '============> Running MPI - JACOBI'
$MPI_INSTALL_PATH/bin/mpirun -n 8 ./adi_burger_mpi -halo 1 -m 4 > perf_out
mv Burger3DRes.h5 Burger3DRes_mpi.h5
grep "Total Wall time" perf_out
$HDF5_INSTALL_PATH/bin/h5diff -p $TOL Burger3DRes_dev_seq.h5 Burger3DRes_mpi.h5 > diff_out
if [ -s ./diff_out ]
then
    echo "File not empty - Solution Not Valid";exit 1;
else
    echo "PASSED"
fi
rm -rf adi_burger_mpi.h5 perf_out diff_out

echo '============> Running MPI - PCR'
$MPI_INSTALL_PATH/bin/mpirun -n 8 ./adi_burger_mpi -halo 1 -m 5 > perf_out
mv Burger3DRes.h5 Burger3DRes_mpi.h5
grep "Total Wall time" perf_out
$HDF5_INSTALL_PATH/bin/h5diff -p $TOL Burger3DRes_dev_seq.h5 Burger3DRes_mpi.h5 > diff_out
if [ -s ./diff_out ]
then
    echo "File not empty - Solution Not Valid";exit 1;
else
    echo "PASSED"
fi
rm -rf adi_burger_mpi.h5 perf_out diff_out

echo '============> Running MPI+CUDA - ALLGATHER'
$MPI_INSTALL_PATH/bin/mpirun -n 8 ./adi_burger_mpi_cuda -halo 1 -m 1 > perf_out
mv Burger3DRes.h5 Burger3DRes_mpi_cuda.h5
grep "Total Wall time" perf_out
$HDF5_INSTALL_PATH/bin/h5diff -p $TOL Burger3DRes_dev_seq.h5 Burger3DRes_mpi_cuda.h5 > diff_out
if [ -s ./diff_out ]
then
    echo "File not empty - Solution Not Valid";exit 1;
else
    echo "PASSED"
fi
rm -rf adi_burger_mpi_cuda.h5

echo '============> Running MPI+CUDA - LATENCY HIDING 2 STEP'
$MPI_INSTALL_PATH/bin/mpirun -n 8 ./adi_burger_mpi_cuda -halo 1 -m 2 -bx 16384 -by 16384 -bz 16384 > perf_out
mv Burger3DRes.h5 Burger3DRes_mpi_cuda.h5
grep "Total Wall time" perf_out
$HDF5_INSTALL_PATH/bin/h5diff -p $TOL Burger3DRes_dev_seq.h5 Burger3DRes_mpi_cuda.h5 > diff_out
if [ -s ./diff_out ]
then
    echo "File not empty - Solution Not Valid";exit 1;
else
    echo "PASSED"
fi
rm -rf adi_burger_mpi_cuda.h5

echo '============> Running MPI+CUDA - LATENCY HIDING INTERLEAVED'
$MPI_INSTALL_PATH/bin/mpirun -n 8 ./adi_burger_mpi_cuda -halo 1 -m 3 -bx 16384 -by 16384 -bz 16384 > perf_out
mv Burger3DRes.h5 Burger3DRes_mpi_cuda.h5
grep "Total Wall time" perf_out
$HDF5_INSTALL_PATH/bin/h5diff -p $TOL ./Burger3DRes_dev_seq.h5 Burger3DRes_mpi_cuda.h5 > diff_out
if [ -s ./diff_out ]
then
    echo "File not empty - Solution Not Valid";exit 1;
else
    echo "PASSED"
fi

echo '============> Running MPI+CUDA - JACOBI'
$MPI_INSTALL_PATH/bin/mpirun -n 8 ./adi_burger_mpi_cuda -halo 1 -m 4 > perf_out
mv Burger3DRes.h5 Burger3DRes_mpi_cuda.h5
grep "Total Wall time" perf_out
$HDF5_INSTALL_PATH/bin/h5diff -p $TOL ./Burger3DRes_seq.h5 Burger3DRes_mpi_cuda.h5 > diff_out
if [ -s ./diff_out ]
then
    echo "File not empty - Solution Not Valid";exit 1;
else
    echo "PASSED"
fi

echo '============> Running MPI+CUDA - PCR'
$MPI_INSTALL_PATH/bin/mpirun -n 8 ./adi_burger_mpi_cuda -halo 1 -m 5 > perf_out
mv Burger3DRes.h5 Burger3DRes_mpi_cuda.h5
grep "Total Wall time" perf_out
$HDF5_INSTALL_PATH/bin/h5diff -p $TOL ./Burger3DRes_seq.h5 Burger3DRes_mpi_cuda.h5 > diff_out
if [ -s ./diff_out ]
then
    echo "File not empty - Solution Not Valid";exit 1;
else
    echo "PASSED"
fi

rm -rf *.h5
echo "All Intel classic complier based applications ---- PASSED"

fi

echo "---------- Exiting Test Script "

