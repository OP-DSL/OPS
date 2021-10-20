#!/bin/bash

# Set TRIDSOLVER_INSTALL_PATH to /path/to/trid/repo/scalar/build/libtrid
# Set OPS_APP_INSTALL_PATH path to where the apps where installed with cmake

# Get reference solution
cd $TRIDSOLVER_INSTALL_PATH/../../../apps/adi/build/
cp compare adi_orig $OPS_APP_INSTALL_PATH/adi

rm -rf *.dat *.h5

# set Relative Tolarance for solution check -- h5diff check only
export TOL="1.000E-14"

cd $OPS_APP_INSTALL_PATH/adi

#============== Run refernace solution ========================
echo '============> Running Referance Solution adi_orig'
./adi_orig > perf_out
rm perf_out

#============================ Test adi with Intel Compilers==========================================================
echo '============> Running SEQ - No Halo'
./adi_seq -halo 0 -t > perf_out
mv adi.h5 adi_seq_h0.h5
mv adi_pad.h5 adi_seq_pad.h5
grep "Total Wall time" perf_out
./compare adi_orig.dat adi_seq.dat > ref_diff
grep exceeded ref_diff || true
grep SumOfDiff ref_diff

echo '============> Running SEQ - Halo'
./adi_seq -halo 1 > perf_out
mv adi.h5 adi_seq_h1.h5
grep "Total Wall time" perf_out
grep "Halo Test" perf_out
$HDF5_INSTALL_PATH/bin/h5diff -p $TOL adi_seq_pad.h5 adi_seq_h1.h5 > diff_out
if [ -s ./diff_out ]
then
    echo "File not empty - Solution Not Valid";exit 1;
else
    echo "PASSED"
fi
rm -rf *.dat adi_seq_h1.h5

echo '============> Running CUDA - No Halo'
./adi_cuda -halo 0 > perf_out
mv adi.h5 adi_cuda_h0.h5
grep "Total Wall time" perf_out
$HDF5_INSTALL_PATH/bin/h5diff -p $TOL adi_seq_h0.h5 adi_cuda_h0.h5 > diff_out
if [ -s ./diff_out ]
then
    echo "File not empty - Solution Not Valid";exit 1;
else
    echo "PASSED"
fi
rm -rf adi_seq_h0.h5 adi_cuda_h0.h5

echo '============> Running CUDA - Halo'
./adi_cuda -halo 1 > perf_out
mv adi.h5 adi_cuda_h1.h5
grep "Total Wall time" perf_out
$HDF5_INSTALL_PATH/bin/h5diff -p $TOL adi_seq_pad.h5 adi_cuda_h1.h5 > diff_out
if [ -s ./diff_out ]
then
    echo "File not empty - Solution Not Valid";exit 1;
else
    echo "PASSED"
fi
rm -rf adi_cuda_h1.h5

echo '============> Running MPI - Gather Scatter'
$MPI_INSTALL_PATH/bin/mpirun -n 8 ./adi_mpi -halo 1 -m 0 > perf_out
mv adi.h5 adi_mpi.h5
grep "Total Wall time" perf_out
$HDF5_INSTALL_PATH/bin/h5diff -p $TOL adi_seq_pad.h5 adi_mpi.h5 > diff_out
if [ -s ./diff_out ]
then
    echo "File not empty - Solution Not Valid";exit 1;
else
    echo "PASSED"
fi
rm -rf adi_mpi.h5

echo '============> Running MPI - ALLGATHER'
$MPI_INSTALL_PATH/bin/mpirun -n 8 ./adi_mpi -halo 1 -m 1 > perf_out
mv adi.h5 adi_mpi.h5
grep "Total Wall time" perf_out
$HDF5_INSTALL_PATH/bin/h5diff -p $TOL adi_seq_pad.h5 adi_mpi.h5 > diff_out
if [ -s ./diff_out ]
then
    echo "File not empty - Solution Not Valid";exit 1;
else
    echo "PASSED"
fi
rm -rf adi_mpi.h5

echo '============> Running MPI - LATENCY HIDING 2 STEP'
$MPI_INSTALL_PATH/bin/mpirun -n 8 ./adi_mpi -halo 1 -m 2 -bx 16384 -by 16384 -bz 16384 > perf_out
mv adi.h5 adi_mpi.h5
grep "Total Wall time" perf_out
$HDF5_INSTALL_PATH/bin/h5diff -p $TOL adi_seq_pad.h5 adi_mpi.h5 > diff_out
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
$HDF5_INSTALL_PATH/bin/h5diff -p $TOL adi_seq_pad.h5 adi_mpi.h5 > diff_out
if [ -s ./diff_out ]
then
    echo "File not empty - Solution Not Valid";exit 1;
else
    echo "PASSED"
fi
rm -rf adi_mpi.h5

echo '============> Running MPI+CUDA - ALLGATHER'
$MPI_INSTALL_PATH/bin/mpirun -n 8 ./adi_mpi_cuda -halo 1 -m 1 > perf_out
mv adi.h5 adi_mpi_cuda.h5
grep "Total Wall time" perf_out
$HDF5_INSTALL_PATH/bin/h5diff -p $TOL adi_seq_pad.h5 adi_mpi_cuda.h5 > diff_out
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
$HDF5_INSTALL_PATH/bin/h5diff -p $TOL adi_seq_pad.h5 adi_mpi_cuda.h5 > diff_out
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
$HDF5_INSTALL_PATH/bin/h5diff -p $TOL adi_seq_pad.h5 adi_mpi_cuda.h5 > diff_out
if [ -s ./diff_out ]
then
    echo "File not empty - Solution Not Valid";exit 1;
else
    echo "PASSED"
fi
rm -rf *.h5 *.dat
