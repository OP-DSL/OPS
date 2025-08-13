#!/bin/bash --login

#SBATCH --nodes=1                       # Total number of nodes

#SBATCH --time=24:00:00                 # Run time (hh:mm:ss)

#SBATCH --account=                      # Project for billing

#SBATCH --job-name=SENGA                # Job name

#SBATCH --partition=gpu                 # Partition (queue) name
#SBATCH --qos=gpu
#SBATCH --gres=gpu:4                    # number of GPUs per node (gres=gpu:N)
#SBATCH --exclusive                     # no nodes allocation sharing with other running jobs


cd $SLURM_SUBMIT_DIR

source setup_env_gpu_nvhpc_CIRRUS.sh

export OMP_NUM_THREADS=1
export OMP_PLACES=cores
export OMP_PROC_BIND=close

# OPS MPI + CUDA Fortran version
srun --gres=gpu:4 --ntasks=4 --tasks-per-node=4 --hint=nomultithread ./senga2_mpi_cuda -OPS_DIAGS=2 OPS_FORCE_DECOMP_X=2 OPS_FORCE_DECOMP_Y=2 OPS_FORCE_DECOMP_Z=1 2>&1 | tee log_1node_4ranks_nvhpc_cirrus_cuda_fortran.txt

# OPS MPI + CUDA C version
srun --gres=gpu:4 --ntasks=4 --tasks-per-node=4 --hint=nomultithread ./senga2_f2c_mpi_cuda -OPS_DIAGS=2 OPS_FORCE_DECOMP_X=2 OPS_FORCE_DECOMP_Y=2 OPS_FORCE_DECOMP_Z=1 2>&1 | tee log_1node_4ranks_nvhpc_cirrus_cuda_c.txt
