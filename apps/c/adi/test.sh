#!/bin/bash
set -e
cd ../../../ops/c
source ../../scripts/$SOURCE_INTEL
make clean
make

#==== Build and copy Referance application from the TDMA Library ====
#build lib first
cd $TDMA_INSTALL_PATH/
#rm -rf ./*
#cmake .. -DCMAKE_BUILD_TYPE=Release -DBUILD_FOR_CPU=ON -DBUILD_FOR_GPU=ON
#make
#make install

#now build application
cd $TDMA_INSTALL_PATH/../../apps/adi/build/
rm -rf ./*
cmake .. -DCMAKE_BUILD_TYPE=Release -DBUILD_FOR_CPU=ON -DBUILD_FOR_GPU=ON
make adi_orig compare
cp compare adi_orig $OPS_INSTALL_PATH/../apps/c/adi
cd -
make
cd $OPS_INSTALL_PATH/../apps/c/adi
make clean
rm -f .generated
make IEEE=1


rm -rf h_u.dat adi_orig.dat adi_seq.dat adi_dev_seq.dat adi_cuda.dat adi_openmp.dat  *.h5

# set Relative Tolarance for solution check -- h5diff check only
export TOL="1.000E-14"

#============== Run refernace solution ========================
echo '============> Running Referance Solution adi_orig'
./adi_orig > perf_out
rm perf_out

#============================ Test adi with Intel Compilers==========================================================

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

rm -rf *.h5
echo "All Intel complied applications PASSED : Exiting Test Script"
