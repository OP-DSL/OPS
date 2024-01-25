#!/bin/bash

#####################
# A rough setup of environmental variables for OPS FPGA development
# Author: Beniel.Thileepan@warwick.ac.uk
####################


############# VITIS SPECIFIC SETUP ################

source /tools/Xilinx/Vitis/2021.2/settings64.sh
export XILINXD_LICENSE_FILE=21001@wthflexsr02.kaust.edu.sa
source /opt/xilinx/xrt/setup.sh


module purge
module load cmake/3.21.4/gcc-7.5.0-dzhxqen

############# OPS SPECIFICS ##############

export OPS_COMPILER=gnu
export OPS_INSTALL_PATH=~/repos/OPS/ops/

############# CUDA & NVIDIA OpenCL #############
module load cuda/11.5.0/gcc-7.5.0-syen6pj
export NV_ARCH=Ampere
export CUDA_VISIBLE_DEVICES=0
echo $NV_ARCH

export CUDA_INSTALL_PATH=/sw/workstations/apps/linux-ubuntu18.04-ivybridge/cuda/11.5.0/gcc-7.5.0/syen6pj6ss3cw66zlj4wkfhtixh5i4ei/
export OPENCL_INSTALL_PATH=$CUDA_INSTALL_PATH
export CUDA_MATH_LIBS=$CUDA_INSTALL_PATH/lib64/
export LD_LIBRARY_PATH=$CUDA_INSTALL_PATH/lib64:$LD_LIBRARY_PATH


############# OPEN MPI AND GCC COMPILERS ##############
module load gcc/9.4.0/gcc-7.5.0-rukpabt
module load openmpi/4.1.1/gcc-9.4.0-4gtto5k
export MPI_INSTALL_PATH=/sw/workstations/apps/linux-ubuntu18.04-ivybridge/openmpi/4.1.1/gcc-9.4.0/4gtto5kgvzpst2lii5r7mlcglej4vukz/
export MPICPP=$MPI_INSTALL_PATH/bin/mpicxx


############# HDF5 ############
module load hdf5/1.10.7/intel-20.0.4-srgjmyt
export HDF5_INSTALL_PATH=/sw/workstations/apps/linux-ubuntu18.04-ivybridge/hdf5/1.10.7/intel-20.0.4/srgjmytnafrqpwhitr2rn63jvrpjxnhn/
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HDF5_INSTALL_PATH/lib/


############# Tridiagonal LIB ###############
export TDMA_INSTALL_PATH=~/lib/libtrid
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$TDMA_INSTALL_PATH/lib/
export LIBTRID_PATH=$TDMA_INSTALL_PATH


############# PYTHON SETUP FOR TRANSLATOR V2 ################
module load llvm/12.0.1/gcc-7.5.0-3k62lda
#module load python/3.8.6/gcc-7.5.0-vnppfx3
#export LD_LIBRARY_PATH=/sw/workstations/apps/linux-ubuntu18.04-ivybridge/llvm/12.0.1/gcc-7.5.0/3k62ldarchejedjr56uhfl6gthkrdnxw/lib:$LD_LIBRARY_PATH
#export LIBCLANG_PATH=/sw/workstations/apps/linux-ubuntu18.04-ivybridge/llvm/12.0.1/gcc-7.5.0/3k62ldarchejedjr56uhfl6gthkrdnxw/lib/libclang.so
export VIRTUALENVWRAPPER_PYTHON=/usr/bin/python3
export VIRTUALENVWRAPPER_VIRTUALENV=$HOME/.local/bin/virtualenv
source $HOME/.local/bin/virtualenvwrapper.sh
workon fpga-cg
