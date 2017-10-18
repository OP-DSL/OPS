#!/bin/bash
source ./source_intel #default source to set environment vars
#exit script if any error is encountered during the build or
#application executions.
set -e
echo $OPS_INSTALL_PATH
cd $OPS_INSTALL_PATH

echo "************Testing C Applications *****************"
cd ../apps/c/CloverLeaf/
cd ../CloverLeaf/
#./test.sh
cd ../CloverLeaf_3D/
#./test.sh
cd ../CloverLeaf_3D_HDF5/
#./test.sh
cd ../TeaLeaf/
#./test.sh
cd ../poisson/
#./test.sh
cd ../multiDim/
#./test.sh
cd ../multiDim3D/
#./test.sh
cd ../shsgc/
#./test.sh
cd ../mb_shsgc/Max_datatransfer
#./test.sh
cd ../../multiDim_HDF5
#./test.sh

echo "************Testing Fortran Applications *****************"
cd $OPS_INSTALL_PATH
cd ../apps/fortran/shsgc
#./test.sh
cd ../poisson
./test.sh
cd ../multiDim
./test.sh
cd ../multiDim3D
./test.sh

echo "All Tests Passed"
