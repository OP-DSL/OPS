#!/bin/bash
set -e
cd $OPS_INSTALL_PATH/c

export SOURCE_INTEL=source_intel_2021.3_pythonenv

#export AMOS=TRUE
#export DMOS=TRUE
#export TELOS=TRUE
#export KOS=TRUE

if [[ -v TELOS || -v KOS ]]; then

#============================ Test with Intel Classic Compilers==========================================
echo "Testing Intel classic complier based applications ---- "

source ../../scripts/$SOURCE_INTEL


# Build and copy Referance application from the TDMA Library
# build lib first
cd $TDMA_INSTALL_PATH/../build
rm -rf ./*
cmake .. -DCUDA_cublas_LIBRARY=/opt/cuda/10.2.89/lib64/libcublas.so -DCMAKE_BUILD_TYPE=Release -DBUILD_FOR_CPU=ON -DBUILD_FOR_GPU=ON -DBUILD_FOR_SN=ON -DBUILD_FOR_MPI=ON -DCMAKE_INSTALL_PREFIX=$TDMA_INSTALL_PATH/../

make
make install


#build OPS
cd $OPS_INSTALL_PATH/c
make clean
make IEEE=1


#now build application
cd $TDMA_INSTALL_PATH/../../apps/adi/build/
rm -rf ./*
cmake .. -DCMAKE_BUILD_TYPE=Release -DBUILD_FOR_CPU=ON -DLIBTRID_PATH=$TDMA_INSTALL_PATH -Dlibtrid_DIR=/rr-home/gihan/tridsolver/scalar/libtrid/lib/cmake/
make adi_orig compare
cp compare adi_orig $OPS_INSTALL_PATH/../apps/c/adi
cd -
make IEEE=1
cd $OPS_INSTALL_PATH/../apps/c/adi
make clean
rm -f .generated
make IEEE=1

#COMMENT

rm -rf h_u.dat adi_orig.dat adi_seq.dat adi_dev_seq.dat adi_cuda.dat adi_openmp.dat  *.h5

# set Relative Tolarance for solution check -- h5diff check only
export TOL="1.000E-14"

#============== Run refernace solution ========================
echo '============> Running Referance Solution adi_orig'
./adi_orig > perf_out
rm perf_out

#============== Run ops ADI application ========================

echo '============> Running DEV_SEQ'
./adi_dev_seq > perf_out
mv adi.h5 adi_dev_seq.h5
grep "Total Wall time" perf_out
./compare adi_orig.dat adi_dev_seq.dat > ref_diff
grep exceeded ref_diff || true
grep SumOfDiff ref_diff
$HDF5_INSTALL_PATH/bin/h5diff -p $TOL adi_dev_seq.h5 adi_dev_seq.h5 > diff_out
if [ -s ./diff_out ]
then
    echo "File not empty - Solution Not Valid";exit 1;
else
    echo "PASSED"
fi
rm -rf perf_out diff_out ref_diff

echo '============> Running SEQ'
./adi_seq > perf_out
mv adi.h5 adi_seq.h5
grep "Total Wall time" perf_out
./compare adi_orig.dat adi_seq.dat > ref_diff
grep exceeded ref_diff || true
grep SumOfDiff ref_diff
$HDF5_INSTALL_PATH/bin/h5diff -p $TOL adi_dev_seq.h5 adi_seq.h5 > diff_out
if [ -s ./diff_out ]
then
    echo "File not empty - Solution Not Valid";exit 1;
else
    echo "PASSED"
fi
rm -rf perf_out diff_out adi_dev_seq.h5


echo '============> Running OpenMP'
KMP_AFFINITY=compact OMP_NUM_THREADS=20 ./adi_openmp > perf_out
mv adi.h5 adi_omp.h5
grep "Total Wall time" perf_out
./compare adi_orig.dat adi_openmp.dat > ref_diff
grep exceeded ref_diff || true
grep SumOfDiff ref_diff
$HDF5_INSTALL_PATH/bin/h5diff -p $TOL adi_seq.h5 adi_omp.h5 > diff_out
if [ -s ./diff_out ]
then
    echo "File not empty - Solution Not Valid";exit 1;
else
    echo "PASSED"
fi
rm -rf perf_out diff_out adi_omp.h5

echo '============> Running CUDA'
./adi_cuda OPS_BLOCK_SIZE_X=64 OPS_BLOCK_SIZE_Y=4 > perf_out
mv adi.h5 adi_cuda.h5
grep "Total Wall time" perf_out
./compare adi_orig.dat adi_cuda.dat > ref_diff
grep exceeded ref_diff || true
grep SumOfDiff ref_diff
$HDF5_INSTALL_PATH/bin/h5diff -p $TOL adi_seq.h5 adi_cuda.h5 > diff_out
if [ -s ./diff_out ]
then
    echo "File not empty - Solution Not Valid";exit 1;
else
    echo "PASSED"
fi
rm -rf perf_out diff_out adi_cuda.h5

echo '============> Running MPI - Gather Scatter'
$MPI_INSTALL_PATH/bin/mpirun -n 8 ./adi_mpi -halo 1 -m 0 > perf_out
mv adi.h5 adi_mpi.h5
grep "Total Wall time" perf_out
$HDF5_INSTALL_PATH/bin/h5diff -p $TOL adi_seq.h5 adi_mpi.h5 > diff_out
if [ -s ./diff_out ]
then
    echo "File not empty - Solution Not Valid";exit 1;
else
    echo "PASSED"
fi
rm -rf perf_out diff_out adi_mpi.h5

echo '============> Running MPI - LATENCY HIDING 2 STEP'
$MPI_INSTALL_PATH/bin/mpirun -n 8 ./adi_mpi -halo 1 -m 2 -bx 16384 -by 16384 -bz 16384 > perf_out
mv adi.h5 adi_mpi.h5
grep "Total Wall time" perf_out
$HDF5_INSTALL_PATH/bin/h5diff -p $TOL adi_seq.h5 adi_mpi.h5 > diff_out
if [ -s ./diff_out ]
then
    echo "File not empty - Solution Not Valid";exit 1;
else
    echo "PASSED"
fi
rm -rf adi_mpi.h5

echo '============> Running MPI - LATENCY HIDING INTERLEAVED'
$MPI_INSTALL_PATH/bin/mpirun -n 8 ./adi_mpi -halo 1 -m 3 -bx 16384 -by 16384 -bz 16384 > perf_out
mv adi.h5 adi_mpi.h5
grep "Total Wall time" perf_out
$HDF5_INSTALL_PATH/bin/h5diff -p $TOL adi_seq.h5 adi_mpi.h5 > diff_out
if [ -s ./diff_out ]
then
    echo "File not empty - Solution Not Valid";exit 1;
else
    echo "PASSED"
fi
rm -rf adi_mpi.h5 perf_out diff_out

echo '============> Running MPI - JACOBI'
$MPI_INSTALL_PATH/bin/mpirun -n 8 ./adi_mpi -halo 1 -m 4 > perf_out
mv adi.h5 adi_mpi.h5
grep "Total Wall time" perf_out
$HDF5_INSTALL_PATH/bin/h5diff -p $TOL adi_seq.h5 adi_mpi.h5 > diff_out
if [ -s ./diff_out ]
then
    echo "File not empty - Solution Not Valid";exit 1;
else
    echo "PASSED"
fi
rm -rf adi_mpi.h5 perf_out diff_out

echo '============> Running MPI - PCR'
$MPI_INSTALL_PATH/bin/mpirun -n 8 ./adi_mpi -halo 1 -m 5 > perf_out
mv adi.h5 adi_mpi.h5
grep "Total Wall time" perf_out
$HDF5_INSTALL_PATH/bin/h5diff -p $TOL adi_seq.h5 adi_mpi.h5 > diff_out
if [ -s ./diff_out ]
then
    echo "File not empty - Solution Not Valid";exit 1;
else
    echo "PASSED"
fi
rm -rf adi_mpi.h5 perf_out diff_out

echo '============> Running MPI+CUDA - ALLGATHER'
$MPI_INSTALL_PATH/bin/mpirun -n 8 ./adi_mpi_cuda -halo 1 -m 1 > perf_out
mv adi.h5 adi_mpi_cuda.h5
grep "Total Wall time" perf_out
$HDF5_INSTALL_PATH/bin/h5diff -p $TOL adi_seq.h5 adi_mpi_cuda.h5 > diff_out
if [ -s ./diff_out ]
then
    echo "File not empty - Solution Not Valid";exit 1;
else
    echo "PASSED"
fi
rm -rf adi_mpi_cuda.h5

echo '============> Running MPI+CUDA - LATENCY HIDING 2 STEP'
$MPI_INSTALL_PATH/bin/mpirun -n 8 ./adi_mpi_cuda -halo 1 -m 2 -bx 16384 -by 16384 -bz 16384 > perf_out
mv adi.h5 adi_mpi_cuda.h5
grep "Total Wall time" perf_out
$HDF5_INSTALL_PATH/bin/h5diff -p $TOL adi_seq.h5 adi_mpi_cuda.h5 > diff_out
if [ -s ./diff_out ]
then
    echo "File not empty - Solution Not Valid";exit 1;
else
    echo "PASSED"
fi
rm -rf adi_mpi_cuda.h5

echo '============> Running MPI+CUDA - LATENCY HIDING INTERLEAVED'
$MPI_INSTALL_PATH/bin/mpirun -n 8 ./adi_mpi_cuda -halo 1 -m 3 -bx 16384 -by 16384 -bz 16384 > perf_out
mv adi.h5 adi_mpi_cuda.h5
grep "Total Wall time" perf_out
$HDF5_INSTALL_PATH/bin/h5diff -p $TOL ./adi_seq.h5 adi_mpi_cuda.h5 > diff_out
if [ -s ./diff_out ]
then
    echo "File not empty - Solution Not Valid";exit 1;
else
    echo "PASSED"
fi

echo '============> Running MPI+CUDA - JACOBI'
$MPI_INSTALL_PATH/bin/mpirun -n 8 ./adi_mpi_cuda -halo 1 -m 4 > perf_out
mv adi.h5 adi_mpi_cuda.h5
grep "Total Wall time" perf_out
$HDF5_INSTALL_PATH/bin/h5diff -p $TOL ./adi_seq.h5 adi_mpi_cuda.h5 > diff_out
if [ -s ./diff_out ]
then
    echo "File not empty - Solution Not Valid";exit 1;
else
    echo "PASSED"
fi

echo '============> Running MPI+CUDA - PCR'
$MPI_INSTALL_PATH/bin/mpirun -n 8 ./adi_mpi_cuda -halo 1 -m 5 > perf_out
mv adi.h5 adi_mpi_cuda.h5
grep "Total Wall time" perf_out
$HDF5_INSTALL_PATH/bin/h5diff -p $TOL ./adi_seq.h5 adi_mpi_cuda.h5 > diff_out
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
