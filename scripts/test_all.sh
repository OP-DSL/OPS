#!/bin/bash
source ./source_intel #default source to set environment vars
#exit script if any error is encountered during the build or
#application executions.
set -e
echo $OPS_INSTALL_PATH
cd $OPS_INSTALL_PATH

echo "************Testing C Applications *****************"
echo "~~~~~~~~~~~~~~~CloverLeaf 2D~~~~~~~~~~~~~~~~~~~~~~~~"
cd ../apps/c/CloverLeaf/
cd ../CloverLeaf/
./test.sh
echo "~~~~~~~~~~~~~~~CloverLeaf 3D~~~~~~~~~~~~~~~~~~~~~~~~"
cd ../CloverLeaf_3D/
./test.sh
echo "~~~~~~~~~~~~~~~CloverLeaf 3D HDF5~~~~~~~~~~~~~~~~~~~"
cd ../CloverLeaf_3D_HDF5/
./test.sh
echo "~~~~~~~~~~~~~~~TeaLeaf 3D ~~~~~~~~~~~~~~~~~~~~~~"
cd ../TeaLeaf/
./test.sh
echo "~~~~~~~~~~~~~~~Poisson~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
cd ../poisson/
./test.sh
echo "~~~~~~~~~~~~~~~multiDim~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
cd ../multiDim/
./test.sh
echo "~~~~~~~~~~~~~~~multiDim3D~~~~~~~~~~~~~~~~~~~~~~~~~~~"
cd ../multiDim3D/
./test.sh
echo "~~~~~~~~~~~~~~~shsgc~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
cd ../shsgc/
./test.sh
echo "~~~~~~~~~~~~~~~mb_shsgc~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
cd ../mb_shsgc/Max_datatransfer
./test.sh
echo "~~~~~~~~~~~~~~~multiDim_HDF5~~~~~~~~~~~~~~~~~~~~~~~~"
cd ../../multiDim_HDF5
./test.sh
echo "~~~~~~~~~~~~~~~adi~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
cd ../adi
./test.sh
echo "~~~~~~~~~~~~~~~mgrid~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
cd ../mgrid
./test.sh

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
