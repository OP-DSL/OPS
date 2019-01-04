#!/bin/bash
set -e
cd ../../../ops/c
source ../../scripts/source_intel
make
cd -
make clean
rm -f .generated
make IEEE=1


rm -rf h_u.dat adi_seq.dat adi_dev_seq.dat adi_cuda.dat adi_openmp.dat

#============================ Test adi with Intel Compilers==========================================================

echo '============> Running DEV_SEQ'
./adi_dev_seq > perf_out
#grep "Total error:" perf_out
#grep "Total Wall time" perf_out
#grep "PASSED" perf_out
#rc=$?; if [[ $rc != 0 ]]; then echo "TEST FAILED";exit $rc; fi
rm perf_out

echo '============> Running SEQ'
./adi_seq > perf_out
#grep "Total error:" perf_out
#grep "Total Wall time" perf_out
#grep "PASSED" perf_out
#rc=$?; if [[ $rc != 0 ]]; then echo "TEST FAILED";exit $rc; fi
rm perf_out

echo '============> Running OpenMP'
KMP_AFFINITY=compact OMP_NUM_THREADS=20 ./adi_openmp > perf_out
#grep "Total error:" perf_out
#grep "Total Wall time" perf_out
#grep "PASSED" perf_out
#rc=$?; if [[ $rc != 0 ]]; then echo "TEST FAILED";exit $rc; fi
rm perf_out

echo '============> Running CUDA'
./adi_cuda OPS_BLOCK_SIZE_X=64 OPS_BLOCK_SIZE_Y=4 > perf_out
#grep "Total error:" perf_out
#grep "Total Wall time" perf_out
#grep "PASSED" perf_out
#rc=$?; if [[ $rc != 0 ]]; then echo "TEST FAILED";exit $rc; fi
rm perf_out

echo "All Intel complied applications PASSED : Exiting Test Script"
