#!/bin/bash

export OPS_COMPILER=pgi
export OPS_INSTALL_PATH=$(pwd)/ops

export NV_ARCH=Ampere

echo "GPU architecture " $NV_ARCH

module purge
# PGI and MPI compiler
module load OpenMPI/5.0.8-NVHPC-25.5-CUDA-12.9.0

export CUDA_INSTALL_PATH=$CUDA_HOME
export CUDA_MATH_LIBS=/work4/scd/scarf562/eb/software/NVHPC/25.5-CUDA-12.9.0/Linux_x86_64/25.5/math_libs/lib64
#export LD_LIBRARY_PATH=$CUDA_MATH_LIBS:$LD_LIBRARY_PATH

export MPI_INSTALL_PATH=/work4/scd/scarf562/eb/software/OpenMPI/5.0.8-NVHPC-25.5-CUDA-12.9.0
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
export HDF5_INSTALL_PATH=/work4/scd/scarf924/Software/opt_nvhpc2505
export PATH=$HDF5_INSTALL_PATH/bin:$PATH
export LD_LIBRARY_PATH=$HDF5_INSTALL_PATH/lib:$LD_LIBRARY_PATH

# Python
# Available by defualt >3.8
export osbli_venv_activate=$(pwd)/ops_translator/ops_venv/bin/activate
if [ -f ${osbli_venv_activate} ]; then
  source ${osbli_venv_activate}
else
  echo "** OPS Virtual enviroment does not exist                     **"
  echo "** Please source ops_translator/setup_venv.sh to generate it **"
fi
