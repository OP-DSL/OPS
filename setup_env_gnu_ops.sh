#!/bin/bash

export OPS_COMPILER=gnu
export OPS_INSTALL_PATH=$(pwd)/ops


module purge

module load mpi/openmpi-x86_64

export MPI_INSTALL_PATH=/usr/lib64/openmpi
export PATH=$MPI_INSTALL_PATH/bin:$PATH
export LD_LIBRARY_PATH=$MPI_INSTALL_PATH/lib:$LD_LIBRARY_PATH

export MPICPP=mpic++
export MPICH_CXX=mpic++
export MPICH_CC=mpicc
export MPICH_F90=mpif90
export MPIF90_F90=mpif90
export MPICH_FC=mpif90

# HDF5
unset HDF5_INSTALL_PATH
export HDF5_INSTALL_PATH=/opt/hdf5-parallel
export PATH=$HDF5_INSTALL_PATH/bin:$PATH
export LD_LIBRARY_PATH=$HDF5_INSTALL_PATH/lib:$LD_LIBRARY_PATH

# Python
# Available by defualt >3.8
source $(pwd)/ops_translator/ops_venv/bin/activate
