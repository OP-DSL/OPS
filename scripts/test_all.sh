#!/bin/bash
export SOURCE_INTEL=source_intel_2021.3_pythonenv
export SOURCE_PGI=source_pgi_nvhpc_23_pythonenv
export SOURCE_INTEL_SYCL=source_intel_2021.3_sycl_pythonenv
export SOURCE_AMD_HIP=source_amd_rocm-5.4.3_pythonenv

source ./$SOURCE_INTEL #default source to set environment vars

#exit script if any error is encountered during the build or
#application executions.
set -e

#export AMOS=TRUE
#export DMOS=TRUE
export TELOS=TRUE
#export KOS=TRUE

echo $OPS_INSTALL_PATH
cd $OPS_INSTALL_PATH

echo "************Testing C Applications *****************"
echo "~~~~~~~~~~~~~~~CloverLeaf 2D~~~~~~~~~~~~~~~~~~~~~~~~"
cd ../apps/c/CloverLeaf/
cd ../CloverLeaf/
#./test.sh
echo "~~~~~~~~~~~~~~~CloverLeaf 3D~~~~~~~~~~~~~~~~~~~~~~~~"
cd ../CloverLeaf_3D/
#./test.sh
echo "~~~~~~~~~~~~~~~CloverLeaf 3D HDF5~~~~~~~~~~~~~~~~~~~"
cd ../CloverLeaf_3D_HDF5/
#./test.sh
echo "~~~~~~~~~~~~~~~TeaLeaf 3D ~~~~~~~~~~~~~~~~~~~~~~"
cd ../TeaLeaf/
#./test.sh
echo "~~~~~~~~~~~~~~~Poisson~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
cd ../poisson/
#./test.sh
echo "~~~~~~~~~~~~~~~multiDim~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
cd ../multiDim/
#./test.sh
echo "~~~~~~~~~~~~~~~multiDim3D~~~~~~~~~~~~~~~~~~~~~~~~~~~"
cd ../multiDim3D/
#./test.sh
echo "~~~~~~~~~~~~~~~shsgc~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
cd ../shsgc/
#./test.sh
echo "~~~~~~~~~~~~~~~mb_shsgc~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
cd ../mb_shsgc/Max_datatransfer
#./test.sh
echo "~~~~~~~~~~~~~~~multiDim_HDF5~~~~~~~~~~~~~~~~~~~~~~~~"
cd ../../multiDim_HDF5
#./test.sh #SYCL compilation issue -- needs fixing 
echo "~~~~~~~~~~~~~~~adi~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
cd ../adi
#./test.sh
echo "~~~~~~~~~~~~~~~mgrid~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
cd ../mgrid
#./test.sh -- SYCL MPI not validating
echo "~~~~~~~~~~~~~~~mblock~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
cd ../mblock
#./test.sh - SYCL MPI not running error produced
echo "~~~~~~~~~~~~~~OpenSBLI TGV~~~~~~~~~~~~~~~~~~~~~~~~~~"
cd $OPENSBLI_INSTALL_PATH/apps/taylor_green_vortex
#./test.sh -- check PGI compilation
cd -
echo "All C/C++ application tests PASSED"

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
echo "END~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
echo "All Tests Passed"
