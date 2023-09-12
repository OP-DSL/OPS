#!/bin/bash
set -e
cd ../../../ops/c

#export SOURCE_INTEL=source_intel_2021.3_pythonenv
#export SOURCE_PGI=source_pgi_nvhpc-23-new
#export SOURCE_INTEL_SYCL=source_intel_2021.3_sycl_pythonenv
#export SOURCE_AMD_HIP=source_amd_rocm-5.4.3_pythonenv

#export AMOS=TRUE
#export DMOS=TRUE
export TELOS=TRUE
#export KOS=TRUE

if [[ -v TELOS || -v KOS ]]; then

#============================ Test with Intel Classic Compilers==========================================
echo "Testing Intel classic complier based applications ---- "

cd $OPS_INSTALL_PATH/c
source ../../scripts/$SOURCE_INTEL
#make -j -B
make clean
make
cd $OPS_INSTALL_PATH/../apps/c/TeaLeaf

make clean
rm -f .generated
make IEEE=1 tealeaf_dev_seq tealeaf_dev_mpi tealeaf_seq tealeaf_tiled tealeaf_openmp tealeaf_mpi \
tealeaf_mpi_tiled tealeaf_mpi_openmp tealeaf_mpi_inline

echo '============> Running OpenMP'
KMP_AFFINITY=compact OMP_NUM_THREADS=20 ./tealeaf_openmp > perf_out
grep "Total Wall time" tea.out
#grep -e "step:    86" -e "step:    87" -e "step:    88"  tea.out
grep "PASSED" tea.out
rc=$?; if [[ $rc != 0 ]]; then echo "TEST FAILED";exit $rc; fi
rm -f tea.out

echo '============> Running OpenMP with Tiling'
KMP_AFFINITY=compact OMP_NUM_THREADS=20 ./tealeaf_tiled OPS_TILING > perf_out
grep "Total Wall time" tea.out
#grep -e "step:    86" -e "step:    87" -e "step:    88"  tea.out
grep "PASSED" tea.out
rc=$?; if [[ $rc != 0 ]]; then echo "TEST FAILED";exit $rc; fi
rm -f tea.out

echo '============> Running MPI+OpenMP'
export OMP_NUM_THREADS=2;$MPI_INSTALL_PATH/bin/mpirun -np 10 numawrap10 ./tealeaf_mpi_openmp > perf_out
grep "Total Wall time" tea.out
#grep -e "step:    86" -e "step:    87" -e "step:    88"  tea.out
grep "PASSED" tea.out
rc=$?; if [[ $rc != 0 ]]; then echo "TEST FAILED";exit $rc; fi
rm -f tea.out

echo '============> Running MPI_Inline with MPI+OpenMP'
export OMP_NUM_THREADS=2;$MPI_INSTALL_PATH/bin/mpirun -np 10 ./tealeaf_mpi_inline > perf_out
grep "Total Wall time" tea.out
#grep -e "step:    86" -e "step:    87" -e "step:    88"  tea.out
grep "PASSED" tea.out
rc=$?; if [[ $rc != 0 ]]; then echo "TEST FAILED";exit $rc; fi
rm -f tea.out

echo '============> Running DEV_MPI'
$MPI_INSTALL_PATH/bin/mpirun -np 20 ./tealeaf_dev_mpi > perf_out
grep "Total Wall time" tea.out
#grep -e "step:    86" -e "step:    87" -e "step:    88"  tea.out
grep "PASSED" tea.out
rc=$?; if [[ $rc != 0 ]]; then echo "TEST FAILED";exit $rc; fi
rm -f tea.out

echo '============> Running MPI'
$MPI_INSTALL_PATH/bin/mpirun -np 20 ./tealeaf_mpi > perf_out
grep "Total Wall time" tea.out
#grep -e "step:    86" -e "step:    87" -e "step:    88"  tea.out
grep "PASSED" tea.out
rc=$?; if [[ $rc != 0 ]]; then echo "TEST FAILED";exit $rc; fi
rm -f tea.out

echo '============> Running MPI_Inline'
export OMP_NUM_THREADS=1;$MPI_INSTALL_PATH/bin/mpirun -np 20 ./tealeaf_mpi_inline > perf_out
grep "Total Wall time" tea.out
#grep -e "step:    86" -e "step:    87" -e "step:    88"  tea.out
grep "PASSED" tea.out
rc=$?; if [[ $rc != 0 ]]; then echo "TEST FAILED";exit $rc; fi
rm -f tea.out

echo '============> Running MPI_Tiled'
export OMP_NUM_THREADS=10;$MPI_INSTALL_PATH/bin/mpirun -np 2 ./tealeaf_mpi_tiled OPS_TILING OPS_TILING_MAXDEPTH=6 > perf_out
grep "Total Wall time" tea.out
#grep -e "step:    86" -e "step:    87" -e "step:    88"  tea.out
grep "PASSED" tea.out
rc=$?; if [[ $rc != 0 ]]; then echo "TEST FAILED";exit $rc; fi
rm -f tea.out

if [[ -v CUDA_INSTALL_PATH ]]; then
make IEEE=1 tealeaf_cuda tealeaf_mpi_cuda tealeaf_mpi_cuda_tiled

echo '============> Running CUDA'
./tealeaf_cuda OPS_BLOCK_SIZE_X=64 OPS_BLOCK_SIZE_Y=4 > perf_out
grep "Total Wall time" tea.out
#grep -e "step:    86" -e "step:    87" -e "step:    88"  tea.out
grep "PASSED" tea.out
rc=$?; if [[ $rc != 0 ]]; then echo "TEST FAILED";exit $rc; fi
rm -f tea.out

echo '============> Running MPI+CUDA'
$MPI_INSTALL_PATH/bin/mpirun -np 2 ./tealeaf_mpi_cuda OPS_BLOCK_SIZE_X=64 OPS_BLOCK_SIZE_Y=4 > perf_out
grep "Total Wall time" tea.out
#grep -e "step:    86" -e "step:    87" -e "step:    88"  tea.out
grep "PASSED" tea.out
rc=$?; if [[ $rc != 0 ]]; then echo "TEST FAILED";exit $rc; fi
rm -f tea.out

echo '============> Running MPI+CUDA+Tiled'
$MPI_INSTALL_PATH/bin/mpirun -np 2 ./tealeaf_mpi_cuda_tiled OPS_TILING OPS_TILING_MAXDEPTH=6 OPS_BLOCK_SIZE_X=64 OPS_BLOCK_SIZE_Y=4 > perf_out
grep "Total Wall time" tea.out
#grep -e "step:   2952" -e "step:   2953" -e "step:   2954" -e "step:   2955" tea.out
grep "PASSED" tea.out
rc=$?; if [[ $rc != 0 ]]; then echo "TEST FAILED";exit $rc; fi
rm -f tea.out

#echo '============> Running MPI+CUDA with GPU-Direct'
#MV2_USE_CUDA=1 $MPI_INSTALL_PATH/bin/mpirun -np 2 ./tealeaf_mpi_cuda -gpudirect OPS_BLOCK_SIZE_X=64 OPS_BLOCK_SIZE_Y=4 > perf_out
#grep "Total Wall time" tea.out
#grep -e "step:    86" -e "step:    87" -e "step:    88"  tea.out
#grep "PASSED" tea.out
#rc=$?; if [[ $rc != 0 ]]; then echo "TEST FAILED";exit $rc; fi
#rm -f tea.out

fi
fi

echo "All Intel classic complier based applications ---- PASSED"


if [[ -v TELOS ]]; then

#============================ Test with Intel SYCL Compilers==========================================
echo "Testing Intel SYCL complier based applications ---- "
cd $OPS_INSTALL_PATH/c
source ../../scripts/$SOURCE_INTEL_SYCL
#make -j -B
make clean
make
cd $OPS_INSTALL_PATH/../apps/c/TeaLeaf

make clean
#make IEEE=1 -j
make tealeaf_sycl tealeaf_mpi_sycl tealeaf_mpi_sycl_tiled

echo '============> Running SYCL on CPU'
./tealeaf_sycl OPS_CL_DEVICE=0 OPS_BLOCK_SIZE_X=512 OPS_BLOCK_SIZE_Y=1 > perf_out
grep "Total Wall time" perf_out
grep "PASSED" perf_out
rc=$?; if [[ $rc != 0 ]]; then echo "TEST FAILED";exit $rc; fi
rm perf_out

echo '============> Running MPI+SYCL on CPU'
$MPI_INSTALL_PATH/bin/mpirun -np 20 ./tealeaf_mpi_sycl OPS_CL_DEVICE=0 OPS_BLOCK_SIZE_X=256 OPS_BLOCK_SIZE_Y=1 > perf_out
grep "Total Wall time" perf_out
grep "PASSED" perf_out
rc=$?; if [[ $rc != 0 ]]; then echo "TEST FAILED";exit $rc; fi
rm perf_out

echo '============> Running MPI+SYCL Tiled on CPU'
$MPI_INSTALL_PATH/bin/mpirun -np 2 ./tealeaf_mpi_sycl_tiled OPS_CL_DEVICE=1 OPS_BLOCK_SIZE_X=32 OPS_BLOCK_SIZE_Y=4 > perf_out
grep "Total Wall time" perf_out
grep "PASSED" perf_out
rc=$?; if [[ $rc != 0 ]]; then echo "TEST FAILED";exit $rc; fi
rm perf_out

echo "All Intel SYCL complier based applications ---- PASSED"

fi


if [[ -v TELOS ]]; then

#============================ Test with PGI Compilers==========================================
echo "Testing PGI/NVHPC complier based applications ---- "
cd $OPS_INSTALL_PATH/c
source ../../scripts/$SOURCE_PGI
make clean
#make -j
make
echo "in here "
cd $OPS_INSTALL_PATH/../apps/c/TeaLeaf
make clean
make tealeaf_dev_seq tealeaf_dev_mpi tealeaf_seq tealeaf_tiled tealeaf_openmp tealeaf_mpi tealeaf_mpi_tiled \
tealeaf_mpi_openmp tealeaf_ompoffload tealeaf_mpi_ompoffload tealeaf_mpi_ompoffload_tiled

echo '============> Running OpenMP'
KMP_AFFINITY=compact OMP_NUM_THREADS=20 ./tealeaf_openmp > perf_out
grep "Total Wall time" tea.out
#grep -e "step:    86" -e "step:    87" -e "step:    88"  tea.out
grep "PASSED" tea.out
rc=$?; if [[ $rc != 0 ]]; then echo "TEST FAILED";exit $rc; fi
rm -f tea.out

echo '============> Running MPI+OpenMP'
export OMP_NUM_THREADS=2;$MPI_INSTALL_PATH/bin/mpirun -np 10 ./tealeaf_mpi_openmp > perf_out
grep "Total Wall time" tea.out
#grep -e "step:    86" -e "step:    87" -e "step:    88"  tea.out
grep "PASSED" tea.out
rc=$?; if [[ $rc != 0 ]]; then echo "TEST FAILED";exit $rc; fi
rm -f tea.out

echo '============> Running DEV_MPI'
$MPI_INSTALL_PATH/bin/mpirun -np 20 ./tealeaf_dev_mpi > perf_out
grep "Total Wall time" tea.out
#grep -e "step:    86" -e "step:    87" -e "step:    88"  tea.out
grep "PASSED" tea.out
rc=$?; if [[ $rc != 0 ]]; then echo "TEST FAILED";exit $rc; fi
rm -f tea.out

echo '============> Running MPI'
$MPI_INSTALL_PATH/bin/mpirun -np 20 ./tealeaf_mpi > perf_out
grep "Total Wall time" tea.out
#grep -e "step:    86" -e "step:    87" -e "step:    88"  tea.out
grep "PASSED" tea.out
rc=$?; if [[ $rc != 0 ]]; then echo "TEST FAILED";exit $rc; fi
rm -f tea.out

if [[ -v CUDA_INSTALL_PATH ]]; then
make IEEE=1 tealeaf_cuda tealeaf_mpi_cuda tealeaf_mpi_cuda_tiled tealeaf_openacc tealeaf_mpi_openacc \
tealeaf_mpi_openacc_tiled

echo '============> Running CUDA'
./tealeaf_cuda OPS_BLOCK_SIZE_X=64 OPS_BLOCK_SIZE_Y=4 > perf_out
grep "Total Wall time" tea.out
#grep -e "step:    86" -e "step:    87" -e "step:    88"  tea.out
grep "PASSED" tea.out
rc=$?; if [[ $rc != 0 ]]; then echo "TEST FAILED";exit $rc; fi
rm -f tea.out

echo '============> Running MPI+CUDA'
$MPI_INSTALL_PATH/bin/mpirun -np 2 ./tealeaf_mpi_cuda OPS_BLOCK_SIZE_X=64 OPS_BLOCK_SIZE_Y=4 > perf_out
grep "Total Wall time" tea.out
#grep -e "step:    86" -e "step:    87" -e "step:    88"  tea.out
grep "PASSED" tea.out
rc=$?; if [[ $rc != 0 ]]; then echo "TEST FAILED";exit $rc; fi
rm -f tea.out

echo '============> Running MPI+CUDA+Tiled'
$MPI_INSTALL_PATH/bin/mpirun -np 2 ./tealeaf_mpi_cuda_tiled OPS_TILING OPS_TILING_MAXDEPTH=6 OPS_BLOCK_SIZE_X=64 OPS_BLOCK_SIZE_Y=4 > perf_out
grep "Total Wall time" tea.out
#grep -e "step:   2952" -e "step:   2953" -e "step:   2954" -e "step:   2955" tea.out
grep "PASSED" tea.out
rc=$?; if [[ $rc != 0 ]]; then echo "TEST FAILED";exit $rc; fi
rm -f tea.out

#echo '============> Running MPI+CUDA with GPU-Direct'
#MV2_USE_CUDA=1 $MPI_INSTALL_PATH/bin/mpirun -np 2 ./tealeaf_mpi_cuda -gpudirect OPS_BLOCK_SIZE_X=64 OPS_BLOCK_SIZE_Y=4 > perf_out
#grep "Total Wall time" tea.out
#grep -e "step:    86" -e "step:    87" -e "step:    88"  tea.out
#grep "PASSED" tea.out
#rc=$?; if [[ $rc != 0 ]]; then echo "TEST FAILED";exit $rc; fi
#rm -f tea.out

echo '============> Running OpenACC'
./tealeaf_openacc OPS_BLOCK_SIZE_X=64 OPS_BLOCK_SIZE_Y=4 > perf_out
grep "Total Wall time" tea.out
#grep -e "step:    86" -e "step:    87" -e "step:    88"  tea.out
grep "PASSED" tea.out
rc=$?; if [[ $rc != 0 ]]; then echo "TEST FAILED";exit $rc; fi
rm -f tea.out
rm perf_out

echo '============> Running MPI+OpenACC'
$MPI_INSTALL_PATH/bin/mpirun -np 2 ./tealeaf_mpi_openacc OPS_BLOCK_SIZE_X=64 OPS_BLOCK_SIZE_Y=4 > perf_out
grep "Total Wall time" tea.out
#grep -e "step:    86" -e "step:    87" -e "step:    88"  tea.out
grep "PASSED" tea.out
rc=$?; if [[ $rc != 0 ]]; then echo "TEST FAILED";exit $rc; fi
rm -f tea.out
rm perf_out

fi

echo '============> Running OMPOFFLOAD'
./tealeaf_ompoffload OPS_BLOCK_SIZE_X=64 OPS_BLOCK_SIZE_Y=4 > perf_out
grep "Total Wall time" perf_out
grep "PASSED" perf_out
rc=$?; if [[ $rc != 0 ]]; then echo "TEST FAILED";exit $rc; fi
rm perf_out

echo '============> Running MPI+OMPOFFLOAD'
$MPI_INSTALL_PATH/bin/mpirun -np 2 ./tealeaf_mpi_ompoffload OPS_BLOCK_SIZE_X=64 OPS_BLOCK_SIZE_Y=4 > perf_out
grep "Total Wall time" perf_out
grep "PASSED" perf_out
rc=$?; if [[ $rc != 0 ]]; then echo "TEST FAILED";exit $rc; fi
rm perf_out

echo "All PGI complier based applications ---- PASSED"

fi

if [[ -v AMOS ]]; then

echo "Testing AMD HIP complier based applications ---- "
cd $OPS_INSTALL_PATH/c
source ../../scripts/$SOURCE_AMD_HIP
#make -j -B
make clean
make
cd $OPS_INSTALL_PATH/../apps/c/TeaLeaf

make clean
rm -f .generated
#make IEEE=1 -j
make IEEE=1 tealeaf_hip tealeaf_mpi_hip

echo '============> Running HIP'
./tealeaf_hip OPS_BLOCK_SIZE_X=64 OPS_BLOCK_SIZE_Y=4 > perf_out
grep "Total Wall time" tea.out
#grep -e "step:    86" -e "step:    87" -e "step:    88"  tea.out
grep "PASSED" tea.out
rc=$?; if [[ $rc != 0 ]]; then echo "TEST FAILED";exit $rc; fi
rm -f tea.out

echo '============> Running MPI+HIP'
mpirun --allow-run-as-root -np 2 ./tealeaf_mpi_hip OPS_BLOCK_SIZE_X=64 OPS_BLOCK_SIZE_Y=4 > perf_out
grep "Total Wall time" tea.out
#grep -e "step:    86" -e "step:    87" -e "step:    88"  tea.out
grep "PASSED" tea.out
rc=$?; if [[ $rc != 0 ]]; then echo "TEST FAILED";exit $rc; fi
rm -f tea.out

echo "All AMD HIP complier based applications ---- PASSED"

fi

echo "---------- Exiting Test Script "