#!/bin/bash

export OPS_COMPILER=pgi
export OPS_INSTALL_PATH=$(pwd)/ops

export NV_ARCH=Ampere

echo "GPU architecture " $NV_ARCH

module purge
source /opt/rh/gcc-toolset-11/enable
# PGI and MPI compiler
module load nvhpc-nompi/22.11

export CUDA_INSTALL_PATH=/opt/nvidia/hpc_sdk/Linux_x86_64/22.11/cuda
export CUDA_MATH_LIBS=/opt/nvidia/hpc_sdk/Linux_x86_64/22.11/math_libs/lib64
export LD_LIBRARY_PATH=$CUDA_MATH_LIBS:$LD_LIBRARY_PATH

export MPI_INSTALL_PATH=/usr/local
#export PATH=$MPI_INSTALL_PATH/bin:$PATH
#export LD_LIBRARY_PATH=$MPI_INSTALL_PATH/lib:$LD_LIBRARY_PATH

export MPICPP=mpic++
export OMPI_CXX=pgc++
export OMPI_CC=pgcc
export OMPI_F90=pgfortran
export OMPI_FC=pgfortran


# HDF5
unset HDF5_INSTALL_PATH
export HDF5_INSTALL_PATH=/opt/hdf5-parallel
export PATH=$HDF5_INSTALL_PATH/bin:$PATH
export LD_LIBRARY_PATH=$HDF5_INSTALL_PATH/lib:$LD_LIBRARY_PATH

# Python
# Available by defualt >3.8
source $(pwd)/ops_translator/ops_venv/bin/activate
