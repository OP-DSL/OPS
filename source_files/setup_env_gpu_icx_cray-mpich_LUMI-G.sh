#!/bin/bash

# OPS_COMPILER - ICX with CRAY MPICH
export OPS_COMPILER=icx_cray

export OPS_INSTALL_PATH=$HOME/OPS/ops

module purge

# Compiler
module load PrgEnv-cray
module load craype-accel-amd-gfx90a
module load craype-x86-milan
module load rocm/6.3.4
module load cpe/25.03
module load craype/2.7.35

# Load ICX compuler
. ~/intel/oneapi/installation/setvars.sh --force --include-intel-llvm

# Remove and Reload Cray-MPICH to take precedence than loaded intel MPI available from oneapi toolkit
module rm cray-mpich/8.1.32
module load cray-mpich/8.1.32

# CRAY CPU TARGET
export CRAY_CPU_TARGET=x86-64

# MPI setting
export MPI_INSTALL_PATH=$CRAY_MPICH_DIR
export LD_LIBRARY_PATH=$MPI_INSTALL_PATH/lib:$LD_LIBRARY_PATH

export MPICC=icpx
export MPICPP=icpx
export MPICXX=icpx
export MPIFC=ifx
export MPIF90=ifx

export MPICH_GPU_SUPPORT_ENABLED=1

# HIP
export AMD_ARCH=MI200

export ROCM_PATH=/opt/rocm-6.3.4
export LD_LIBRARY_PATH=$ROCM_PATH/llvm/lib:$LD_LIBRARY_PATH
export HIP_INSTALL_PATH=$ROCM_PATH
export AOMP=$ROCM_PATH/llvm

# SYCL
#export SYCL_INSTALL_PATH=~/intel/oneapi/installation/compiler/2025.2 - need to install codeplay plugin for AMD/NVIDIA GPUs

# HDF5
module load cray-hdf5-parallel/1.12.2.11

unset HDF5_INSTALL_PATH
export HDF5_INSTALL_PATH=/opt/cray/pe/hdf5-parallel/1.12.2.11/crayclang/17.0
export LD_LIBRARY_PATH=$HDF5_INSTALL_PATH/lib:$LD_LIBRARY_PATH

# Python
module load cray-python/3.11.7
