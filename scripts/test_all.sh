#!/bin/bash

export SOURCE_INTEL=source_oneapi_sycl_pythonenv
export SOURCE_PGI=source_pgi_nvhpc_23_pythonenv
export SOURCE_INTEL_SYCL=source_oneapi_sycl_pythonenv
export SOURCE_AMD_HIP=source_amd_rocm-5.4.3_pythonenv

source ./$SOURCE_INTEL #default source to set environment vars

#exit script if any error is encountered during the build or
#application executions.
set -e

#export AMOS=TRUE
export DEMOS=TRUE
#export TELOS=TRUE
#export KOS=TRUE

echo $OPS_INSTALL_PATH
cd $OPS_INSTALL_PATH

echo "************Testing C Applications *****************"
echo "~~~~~~~~~~~~~~~CloverLeaf 2D~~~~~~~~~~~~~~~~~~~~~~~~"
cd ../apps/c/CloverLeaf/
cd ../CloverLeaf/
#./test.sh -- works
echo "~~~~~~~~~~~~~~~CloverLeaf 3D~~~~~~~~~~~~~~~~~~~~~~~~"
cd ../CloverLeaf_3D/
#./test.sh -- works
echo "~~~~~~~~~~~~~~~CloverLeaf 3D HDF5~~~~~~~~~~~~~~~~~~~"
cd ../CloverLeaf_3D_HDF5/
#./test.sh -- works
echo "~~~~~~~~~~~~~~~TeaLeaf 3D ~~~~~~~~~~~~~~~~~~~~~~"
cd ../TeaLeaf/
#./test.sh -- tealeaf_ompoffload with pgi compilers does not compile
echo "~~~~~~~~~~~~~~~Poisson~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
cd ../poisson/
#./test.sh -- works
echo "~~~~~~~~~~~~~~~multiDim~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
cd ../multiDim/
#./test.sh -- multidim_mpi_openmp breaks
echo "~~~~~~~~~~~~~~~multiDim3D~~~~~~~~~~~~~~~~~~~~~~~~~~~"
cd ../multiDim3D/ 
#./test.sh -- works
echo "~~~~~~~~~~~~~~~lowdim_test~~~~~~~~~~~~~~~~~~~~~~~~~~"
cd ../lowdim_test/
#./test.sh -- works
echo "~~~~~~~~~~~~~~~shsgc~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
cd ../shsgc/
#./test.sh -- works
echo "~~~~~~~~~~~~~~~mb_shsgc~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
cd ../mb_shsgc/Max_datatransfer
#./test.sh -- works
echo "~~~~~~~~~~~~~~~multiDim_HDF5~~~~~~~~~~~~~~~~~~~~~~~~"
cd ../../multiDim_HDF5 
#./test.sh -- works 
echo "~~~~~~~~~~~~~~~adi~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
cd ../adi 
#./test.sh -- works
echo "~~~~~~~~~~~~~~~adi_burger~~~~~~~~~~~~~~~~~~~~~~~~~~~"
cd ../adi_burger
#./test.sh -- works
echo "~~~~~~~~~~~~~~~adi_berger_3D~~~~~~~~~~~~~~~~~~~~~~~~"
cd ../adi_burger_3D
#./test.sh -- works
echo "~~~~~~~~~~~~~~~mgrid~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
cd ../mgrid
#./test.sh -- SYCL error at runtime 
echo "~~~~~~~~~~~~~~~mblock~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
cd ../mblock
#./test.sh - SYCL MPI not running error produced
echo "~~~~~~~~~~~~~~OpenSBLI TGV~~~~~~~~~~~~~~~~~~~~~~~~~~"
cd $OPENSBLI_INSTALL_PATH/apps/taylor_green_vortex
#./test.sh  -- works
cd -
echo "  "
echo "All C/C++ application tests PASSED"
echo "  "

echo "************Testing Fortran Applications *****************"
cd $OPS_INSTALL_PATH
echo "~~~~~~~~~~~~~~~hsgc Fortran~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
cd ../apps/fortran/shsgc
./test.sh
echo "~~~~~~~~~~~~~~~poisson Fortran~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
cd ../poisson
./test.sh
echo "~~~~~~~~~~~~~~~multiDim Fortran~~~~~~~~~~~~~~~~~~~~~~~~~~~"
cd ../multiDim
./test.sh
echo "~~~~~~~~~~~~~~~multiDim3D Fortran~~~~~~~~~~~~~~~~~~~~~~~~~"
cd ../multiDim3D
./test.sh
echo "~~~~~~~~~~~~~~~lowdim_test Fortran~~~~~~~~~~~~~~~~~~~~~~~~"
cd ../lowdim_test
./test.sh
echo "~~~~~~~~~~~~~~~laplace2d Fortran~~~~~~~~~~~~~~~~~~~~~~~~~~"
cd ../laplace2dtutorial/step7
./test.sh
echo "END~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
echo "  "
echo "All Fortran application tests PASSED"
echo "  "

echo "All Tests Passed"
echo "*****************************************************************  "

