#!/bin/bash -l

#SBATCH --nodes=1                    # Total number of nodes

#SBATCH --time=00:20:00              # Run time (d-hh:mm:ss)

#SBATCH --account=                   # Project for billing

#SBATCH --job-name=SENGA             # Job name

## Shared GPU node - using 2 GPUs on a node
##SBATCH --gpus=2                     # Allocate one gpu per MPI rank
##SBATCH --partition=gpu              # Partition (queue) name
##SBATCH --qos=gpu-shd

## Exclusive GPU node - using all 4 GPUs
#SBATCH --gpus=4
#SBATCH --partition=gpu
#SBATCH --qos=gpu-exc
#SBATCH --exclusive

cd $SLURM_SUBMIT_DIR

source setup_env_gpu_cray_ARCHER2.sh

export OMP_NUM_THREADS=1
export OMP_PLACES=cores
export OMP_PROC_BIND=close

# Enable GPU-aware MPI
export MPICH_GPU_SUPPORT_ENABLED=1

srun --distribution=block:block  --hint=nomultithread --ntasks=4 --cpus-per-task=8 ./senga2_f2c_mpi_hip -gpudirect -OPS_DIAGS=2 OPS_FORCE_DECOMP_X=2 OPS_FORCE_DECOMP_Y=2 OPS_FORCE_DECOMP_Z=1 2>&1 | tee log_1node_4ranks_cray_archer2_hip_c.txt
