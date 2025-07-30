#!/bin/bash

# OPS_COMPILER - icx
export OPS_COMPILER=icx

export OPS_INSTALL_PATH=$HOME/OPS/ops

module purge

#Nvidia H100
export NV_ARCH=Hopper

echo "GPU architecture" $NV_ARCH

# CUDA
module load cuda/12.5

export CUDA_INSTALL_PATH=/opt/cuda/12.5
export CUDA_MATH_LIBS=/opt/cuda/12.5/lib64
export LD_LIBRARY_PATH=$CUDA_MATH_LIBS:$LD_LIBRARY_PATH

# OneAPI with Nvidia plug-in from codeplay
source <path to oneapi installation>/setvars.sh

export SYCL_INSTALL_PATH=<path to oneapi installation>/compiler/2025.2
export MPI_INSTALL_PATH=<path to oneapi installation>/mpi/2021.16
export LD_LIBRARY_PATH=$MPI_INSTALL_PATH/lib:$LD_LIBRARY_PATH

export MPICC=mpiicx
export MPICPP=mpiicx
export MPICXX=mpiicpx
export MPIFC=mpiifort
export MPIF90=mpiifort

unset HDF5_INSTALL_PATH
export HDF5_INSTALL_PATH=<path to HDF5 installl dir>
export LD_LIBRARY_PATH=$HDF5_INSTALL_PATH/lib:$LD_LIBRARY_PATH
export PATH=$HDF5_INSTALL_PATH/bin:$PATH

module load python/3.9.7
source $OPS_INSTALL_PATH/../ops_translator/ops_venv/bin/activate
