## This script will test the minimum install configuration.
## Just enough to generate a sequential version of the code.
## This requires OPS_INSTALL_PATH environment variable.

echo $OPS_INSTALL_PATH

## Generate the ops files
cd $OPS_INSTALL_PATH/../apps/c/poisson
$OPS_INSTALL_PATH/../ops_translator/c/ops.py poissonERROR.cpp










