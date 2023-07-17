#!/bin/bash

export OPS_COMPILER=gnu
export OPS_INSTALL_PATH=/mnt/ceph-training/home/as312140/training/OPS/ops

module purge

module load mpi/openmpi-x86_64

export MPI_INSTALL_PATH=/usr/lib64/openmpi
export PATH=$MPI_INSTALL_PATH/bin:$PATH
export LD_LIBRARY_PATH=$MPI_INSTALL_PATH/lib:$LD_LIBRARY_PATH

export OP_AUTO_SOA=1

export MPICPP=mpic++
export MPICH_CXX=mpic++
export MPICH_CC=mpicc
export MPICH_F90=mpif90
export MPIF90_F90=mpif90
export MPICH_FC=mpif90

module load nvhpc-byo-compiler/22.11
export CUDA_INSTALL_PATH=/opt/nvidia/hpc_sdk/Linux_x86_64/22.11/cuda/11.8
export CUDA_MATH_LIBS=/opt/nvidia/hpc_sdk/Linux_x86_64/22.11/cuda/11.8/lib64

export NV_ARCH=Volta
echo "GPU architecture" $NV_ARCH

unset HDF5_INSTALL_PATH
#export HDF5_INSTALL_PATH=/home/gi386003/install/build_hdf5/gnu
#export PATH=$HDF5_INSTALL_PATH/bin:$PATH
#export LD_LIBRARY_PATH=$HDF5_INSTALL_PATH/lib:$LD_LIBRARY_PATH
