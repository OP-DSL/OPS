#!/bin/bash

# OPS_COMPILER - CRAY
export OPS_COMPILER=cray

export OPS_INSTALL_PATH=$HOME/OPS/ops

module purge

# Compiler
module load cpe/25.03
module load PrgEnv-cray
module load craype-accel-amd-gfx90a
module load craype-x86-milan
module load rocm/6.3.4
module load craype/2.7.35

# CRAY CPU TARGET
export CRAY_CPU_TARGET=x86-milan

# MPI setting
export MPI_INSTALL_PATH=$CRAY_MPICH_DIR
export LD_LIBRARY_PATH=$MPI_INSTALL_PATH/lib:$LD_LIBRARY_PATH

export MPICC=cc
export MPICPP=CC
export MPICXX=CC
export MPIFC=ftn
export MPIF90=ftn

export MPICH_GPU_SUPPORT_ENABLED=1

# HIP
export AMD_ARCH=MI200

export ROCM_PATH=/opt/rocm-6.3.4
export LD_LIBRARY_PATH=$ROCM_PATH/llvm/lib:$LD_LIBRARY_PATH
export HIP_INSTALL_PATH=$ROCM_PATH
export AOMP=$ROCM_PATH/llvm

# HDF5
module load cray-hdf5-parallel/1.12.2.11

unset HDF5_INSTALL_PATH
export HDF5_INSTALL_PATH=/opt/cray/pe/hdf5-parallel/1.12.2.11/crayclang/17.0/include
export LD_LIBRARY_PATH=$HDF5_INSTALL_PATH/lib:$LD_LIBRARY_PATH
