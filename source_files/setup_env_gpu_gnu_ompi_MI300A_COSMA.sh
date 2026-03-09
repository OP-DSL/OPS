#!/bin/bash

# OPS_COMPILER - gnu
export OPS_COMPILER=gnu

export OPS_INSTALL_PATH=$HOME/repos/OPS/ops
module purge

# Compiler
module load gnu_comp/14.1.0
module load openmpi/5.0.3
module load parallel_hdf5/1.14.4

# MPI setting
export MPI_INSTALL_PATH=//cosma/local/openmpi/gnu_14.1.0/5.0.3
export LD_LIBRARY_PATH=$MPI_INSTALL_PATH/lib:$LD_LIBRARY_PATH
# export CPLUS_INCLUDE_PATH=$MPI_INSTALL_PATH/include:$CPLUS_INCLUDE_PATH
# export C_INCLUDE_PATH=$MPI_INSTALL_PATH/include:$C_INCLUDE_PATH
# export CPP_INCLUDE_PATH=$MPI_INSTALL_PATH/include:$CPP_INCLUDE_PATH

export MPICC=mpic++
export MPICPP=mpic++
export MPICXX=mpicxx
# HIP
export AMD_ARCH=MI300A

# export HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

export ROCM_PATH=/etc/alternatives/rocm
export LD_LIBRARY_PATH=$ROCM_PATH/llvm/lib:$LD_LIBRARY_PATH
export HIP_INSTALL_PATH=$ROCM_PATH
export AOMP=$ROCM_PATH/llvm

# HDF5
unset HDF5_INSTALL_PATH
export HDF5_INSTALL_PATH=/cosma/local/parallel_hdf5/gnu_14.1.0_ompi_5.0.3/1.14.4/
export LD_LIBRARY_PATH=$HDF5_INSTALL_PATH/lib:$LD_LIBRARY_PATH

# Python
module load python/3.9.19
source $OPS_INSTALL_PATH/../ops_translator/ops_venv/bin/activate
