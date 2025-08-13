#!/bin/bash

# OPS_COMPILER - GNU
export OPS_COMPILER=gnu

export OPS_INSTALL_PATH=$HOME/OPS/ops

module purge

module load /home/y07/shared/tursa-modules/setup-env

# Compiler
module load gcc/12.2.0
module load openmpi/4.1.5-cuda12.3
module load cuda/12.3

# MPI setting
export MPI_INSTALL_PATH=/home/y07/shared/libs/openmpi/4.1.5.4/gcc12-cuda12
export PATH=$MPI_INSTALL_PATH/bin:$PATH
export LD_LIBRARY_PATH=$MPI_INSTALL_PATH/lib:$LD_LIBRARY_PATH

export MPICPP=mpic++
export MPICH_CXX=mpic++
export MPICH_CC=mpicc
export MPICH_F90=mpif90
export MPIF90_F90=mpif90
export MPICH_FC=mpif90

export CC_COMPILER=g++
export MPI_COMPILER=mpicxx

# CUDA
export NV_ARCH=Ampere

export CUDA_INSTALL_PATH=/mnt/lustre/tursafs1/apps/cuda/12.3
export CUDA_MATH_LIBS=/mnt/lustre/tursafs1/apps/cuda/12.3/lib64
export LD_LIBRARY_PATH=$CUDA_MATH_LIBS:$LD_LIBRARY_PATH

# HDF5 - Installed by user
unset HDF5_INSTALL_PATH
export HDF5_INSTALL_PATH=$HOME/hdf5_gnu
export PATH=$HDF5_INSTALL_PATH/bin:$PATH
export LD_LIBRARY_PATH=$HDF5_INSTALL_PATH/lib:$LD_LIBRARY_PATH

# Python
alias python3='python3.11'
source $OPS_INSTALL_PATH/../ops_translator/ops_venv/bin/activate
