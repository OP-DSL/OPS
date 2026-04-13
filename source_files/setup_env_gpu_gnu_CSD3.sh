#!/bin/bash

# OPS_COMPILER - GNU
export OPS_COMPILER=gnu

export OPS_INSTALL_PATH=$HOME/OPS/ops

module purge

# CUDA
export NV_ARCH=Ampere

module load cuda/12.1

export CUDA_INSTALL_PATH=/usr/local/software/cuda/12.1/
export CUDA_MATH_LIBS=/usr/local/software/cuda/12.1/lib64/
export LD_LIBRARY_PATH=$CUDA_MATH_LIBS:$LD_LIBRARY_PATH

# Python
module load python/3.8.1-icl

# HDF5
module load hdf5/openmpi/gcc/9.3/openmpi-4.0.4/1.12.0
module load rhel8/slurm

unset HDF5_INSTALL_PATH
export HDF5_INSTALL_PATH=/usr/local/Cluster-Apps/hdf5/openmpi/gcc/9.3/1.12.0
export LD_LIBRARY_PATH=$HDF5_INSTALL_PATH/lib:$LD_LIBRARY_PATH

# MPI setting
export MPI_INSTALL_PATH=/usr/local/Cluster-Apps/openmpi/gcc/9.3/4.0.4
export PATH=$MPI_INSTALL_PATH/bin:$PATH
export LD_LIBRARY_PATH=$MPI_INSTALL_PATH/lib:$LD_LIBRARY_PATH

export MPICC=mpicc
export MPICPP=mpicxx
export MPICXX=mpicxx
export MPIFC=mpif90
export MPIF90=mpif90
