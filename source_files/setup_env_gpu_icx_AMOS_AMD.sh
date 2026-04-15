#!/bin/bash

# OPS_COMPILER - icx
export OPS_COMPILER=icx

export OPS_INSTALL_PATH=$HOME/OPS/ops

module purge

# AMOS
export AMD_ARCH=MI100

echo "GPU architecture" $AMD_ARCH

# ROCM
module load rocm/5.6.0
export LD_LIBRARY_PATH=/opt/rocm-5.6.0/llvm/lib:$LD_LIBRARY_PATH

# OneAPI with AMD plug-in from codeplay
. <path to oneapi installation>/setvars.sh --include-intel-llvm

# SYCL
export SYCL_INSTALL_PATH=<path to oneapi installation>/compiler/2025.2
export MPI_INSTALL_PATH=<path to oneapi installation>/mpi/2021.16
export LD_LIBRARY_PATH=$MPI_INSTALL_PATH/lib:$LD_LIBRARY_PATH

# MPI setting
export MPICC=mpiicx
export MPICPP=mpiicx
export MPICXX=mpiicpx
export MPIFC=mpiifort
export MPIF90=mpiifort

# HIP
export HIP_INSTALL_PATH=/opt/rocm-5.6.0/hip
export AOMP=/opt/rocm-5.6.0/llvm

# HDF5
unset HDF5_INSTALL_PATH
export HDF5_INSTALL_PATH=<path to HDF5 installl dir>
export PATH=$HDF5_INSTALL_PATH/bin:$PATH
export LD_LIBRARY_PATH=<path to ZLIB installl dir>:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$HDF5_INSTALL_PATH/lib:$LD_LIBRARY_PATH

# Python
module load python/3.9.7

#export LD_LIBRARY_PATH=<softline for libffi.so.6 specific issue of amos>:$LD_LIBRARY_PATH
