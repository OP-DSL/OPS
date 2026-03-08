export OPS_COMPILER=gnu
export OPS_INSTALL_PATH=~/repos/OPS/ops
export OPS_TRANSLATOR=~/repos/OPS/ops_translator

export USE_HDF5=1

module purge

export AMD_ARCH=MI300A

module load rocm/6.4.0
# module load gcc/base
module load openmpi/5.0.7-ucc1.3.0-ucx1.18.0

export MPI_INSTALL_PATH=/opt/rocmplus-6.4.0/openmpi-5.0.7-ucc-1.3.0-ucx-1.18.0
# export LD_LIBRARY_PATH=$MPI_INSTALL_PATH/lib:$LD_LIBRARY_PATH

if [ $AMD_ARCH = "MI300A" ]; then
    export HSA_XNACK=1 # Enable XNACK for MI300A
fi

export ROCM_PATH=/opt/rocm-6.4.0
# export LD_LIBRARY_PATH=$ROCM_PATH/llvm/lib:$LD_LIBRARY_PATH
export HIP_INSTALL_PATH=$ROCM_PATH
export AOMP=$ROCM_PATH/llvm

export MPICC=mpic++
export MPICPP=mpic++
export MPICXX=mpicxx

# export MPICH_GPU_SUPPORT_ENABLED=1

export HDF5_INSTALL_PATH=/opt/hdf5-v1.14.5/HDF_Group/HDF5/1.14.5
export LD_LIBRARY_PATH=$HDF5_INSTALL_PATH/lib:$LD_LIBRARY_PATH

export PYTHONPATH=$PYTHONPATH:~/repos/opensbli/

source $OPS_INSTALL_PATH/../ops_translator/ops_venv/bin/activate



