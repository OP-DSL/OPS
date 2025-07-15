#!/bin/bash

# OPS_COMPILER - icx
export OPS_COMPILER=icx

export OPS_INSTALL_PATH=$HOME/OPS/ops

module purge

# Compiler
module load oneAPI/2024.2.0
module load compiler-rt tbb compiler mpi dpl dpct
module load parallel_hdf5/1.14.4

# MPI setting
export MPI_INSTALL_PATH=/cosma/local/intel/oneAPI_2024.2.0/mpi/2021.13
export LD_LIBRARY_PATH=$MPI_INSTALL_PATH/lib:$LD_LIBRARY_PATH

export MPICC=mpiicc
export MPICPP=mpiicc
export MPICXX=mpiicpc
export MPIFC=mpiifort
export MPIF90=mpiifort

# HIP
export AMD_ARCH=MI300X

export HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

export ROCM_PATH=/opt/rocm-6.3.2
export LD_LIBRARY_PATH=$ROCM_PATH/llvm/lib:$LD_LIBRARY_PATH
export HIP_INSTALL_PATH=$ROCM_PATH
export AOMP=$ROCM_PATH/llvm

# HDF5
unset HDF5_INSTALL_PATH
export HDF5_INSTALL_PATH=/cosma/local/parallel-hdf5/intel_2024.2.0_intel_mpi_2024.2.0/1.14.4
export LD_LIBRARY_PATH=$HDF5_INSTALL_PATH/lib:$LD_LIBRARY_PATH

# Python
module load python/3.9.19
source $OPS_INSTALL_PATH/../ops_translator/ops_venv/bin/activate
