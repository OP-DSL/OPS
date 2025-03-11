#!/bin/bash

# OPS_COMPILER - PGI
export OPS_COMPILER=pgi

export OPS_INSTALL_PATH=$HOME/OPS/ops

module purge

module use /mnt/lustre/e1000/home/y07/shared/cirrus-modulefiles
module load epcc/setup-env

# Python
module load python/3.9.13
source $OPS_INSTALL_PATH/../ops_translator/ops_venv/bin/activate

# Compiler
module rm --force openmpi/4.1.6 gcc/10.2.0

module load nvidia/nvhpc-nompi/22.11
module load mpt/2.25
module load gcc/8.2.0

# MPI setting
export MPI_INSTALL_PATH=/opt/hpe/hpc/mpt/mpt-2.25
export PATH=$MPI_INSTALL_PATH/bin:$PATH
export LD_LIBRARY_PATH=$MPI_INSTALL_PATH/lib:$LD_LIBRARY_PATH

export MPICC_CC=pgcc
export MPICXX_CXX=pgc++
export MPIF90_F90=pgfortran
export MPICPP_PATH=mpicxx
export MPICXX_PATH=mpicxx

# CUDA
export NV_ARCH=Volta

export CUDA_INSTALL_PATH=/work/y07/shared/cirrus-software/nvidia/hpcsdk-22.11/Linux_x86_64/22.11/cuda/11.8
export CUDA_MATH_LIBS=/work/y07/shared/cirrus-software/nvidia/hpcsdk-22.11/Linux_x86_64/22.11/math_libs/lib64
export LD_LIBRARY_PATH=$CUDA_MATH_LIBS:$LD_LIBRARY_PATH

export LD_LIBRARY_PATH=/work/y07/shared/cirrus-software/gcc/8.2.0/lib64:$LD_LIBRARY_PATH

# HDF5
module load hdf5parallel/1.10.6-gcc8-mpt225

unset HDF5_INSTALL_PATH
export HDF5_INSTALL_PATH=/work/y07/shared/cirrus-software/hdf5parallel/1.10.6-gcc8-mpt225
export LD_LIBRARY_PATH=$HDF5_INSTALL_PATH/lib:$LD_LIBRARY_PATH
