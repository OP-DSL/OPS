#!/bin/bash
set -e
cd ../../../ops/c
source ../../scripts/source_intel
make
cd -
make clean
rm -f .generated
make IEEE=1


rm -rf h_u.dat adi_seq.dat adi_dev_seq.dat adi_cuda.dat adi_openmp.dat  *.h5

#============================ Test adi with Intel Compilers==========================================================

echo '============> Running DEV_SEQ'
./adi_dev_seq > perf_out
mv adi.h5 adi_dev_seq.h5
grep "Total Wall time" perf_out
$HDF5_INSTALL_PATH/bin/h5diff -p 1.000E-14 adi_dev_seq.h5 adi_dev_seq.h5 > diff_out
if [ -s ./diff_out ]
then
    echo "File not empty - Solution Not Valid";exit 1;
else
    echo "PASSED"
fi
rm -rf perf_out diff_out

echo '============> Running SEQ'
./adi_seq > perf_out
mv adi.h5 adi_seq.h5
grep "Total Wall time" perf_out
$HDF5_INSTALL_PATH/bin/h5diff -p 1.000E-14 adi_dev_seq.h5 adi_seq.h5 > diff_out
if [ -s ./diff_out ]
then
    echo "File not empty - Solution Not Valid";exit 1;
else
    echo "PASSED"
fi
rm -rf perf_out diff_out


echo '============> Running OpenMP'
KMP_AFFINITY=compact OMP_NUM_THREADS=20 ./adi_openmp > perf_out
mv adi.h5 adi_omp.h5
grep "Total Wall time" perf_out
$HDF5_INSTALL_PATH/bin/h5diff -p 1.000E-14 adi_seq.h5 adi_omp.h5 > diff_out
if [ -s ./diff_out ]
then
    echo "File not empty - Solution Not Valid";exit 1;
else
    echo "PASSED"
fi
rm -rf perf_out diff_out

echo '============> Running CUDA'
./adi_cuda OPS_BLOCK_SIZE_X=64 OPS_BLOCK_SIZE_Y=4 > perf_out
mv adi.h5 adi_cuda.h5
grep "Total Wall time" perf_out
$HDF5_INSTALL_PATH/bin/h5diff -p 1.000E-14 adi_seq.h5 adi_cuda.h5 > diff_out
if [ -s ./diff_out ]
then
    echo "File not empty - Solution Not Valid";exit 1;
else
    echo "PASSED"
fi
rm -rf perf_out diff_out

rm -rf *.h5
echo "All Intel complied applications PASSED : Exiting Test Script"
