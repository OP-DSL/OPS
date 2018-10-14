## This script will test the minimum install configuration.
## Just enough to generate a sequential version of the code.
## This requires OPS_INSTALL_PATH environment variable.
## This script will compile a DEV_SEQ version of poisson example and run it.

echo $OPS_INSTALL_PATH

## Generate the ops files
cd $OPS_INSTALL_PATH/../apps/c/poisson
$OPS_INSTALL_PATH/../ops_translator/c/ops.py poisson.cpp

export OPS_COMPILER=gnu
export MPI_INSTALL_PATH=`which mpicc | sed 's/\/bin\/mpicc//g'`

if g++ -O3 -fPIC -DUNIX -Wall -ffloat-store -g -I$OPS_INSTALL_PATH/c/include \
    -L$OPS_INSTALL_PATH/c/lib poisson.cpp  -lops_seq  -o output_dev_seq 
then
    if ./output_dev_seq
    then
        echo "Success"
        exit 0
    else
        echo "Failed to run poission program."
        exit 1
    fi
else
    echo "Compilation failed."
    exit 2
fi


