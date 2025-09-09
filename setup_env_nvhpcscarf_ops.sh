#!/bin/bash

export OPS_COMPILER=pgi
export OPS_COMPILER_VERS=24.9
export OPS_INSTALL_PATH=$(pwd)/ops

export NV_ARCH=Ampere

echo "GPU architecture " $NV_ARCH

#module purge
# PGI and MPI compiler
module load OpenMPI/5.0.3-NVHPC-24.9-CUDA-12.6.0

export CUDA_INSTALL_PATH=$CUDA_HOME
export CUDA_MATH_LIBS=$NVHPC/Linux_x86_64/$OPS_COMPILER_VERS/math_libs/lib64
#export LD_LIBRARY_PATH=$CUDA_MATH_LIBS:$LD_LIBRARY_PATH

export MPI_INSTALL_PATH=$EBROOTOPENMPI
#export PATH=$MPI_INSTALL_PATH/bin:$PATH
#export LD_LIBRARY_PATH=$MPI_INSTALL_PATH/lib:$LD_LIBRARY_PATH

export MPICPP=mpic++
export MPICH_CXX=mpic++
export MPICH_CC=mpicc
export MPICH_F90=mpif90
export MPIF90_F90=mpif90
export MPICH_FC=mpif90

# HDF5
unset HDF5_INSTALL_PATH
#export HDF5_INSTALL_PATH=/usr/lib64/openmpi 
export HDF5_INSTALL_PATH=$HDF5_ROOT

# Python
# Available by defualt >3.8
export osbli_venv_activate=$(pwd)/ops_translator/ops_venv/bin/activate
if [ -f ${osbli_venv_activate} ]; then
  source ${osbli_venv_activate}
else
  echo "** OPS Virtual enviroment does not exist                     **"
  echo "** Please source ops_translator/setup_venv.sh to generate it **"
fi
