#!/bin/bash

#SBATCH --nodes=1                       # Total number of nodes
#SBATCH --ntasks-per-node=4             # MPI ranks per node
#SBATCH --gres=gpu:4                    # Allocate one gpu per MPI rank

#SBATCH --time=24:00:00                 # Run time (hh:mm:ss)

#SBATCH --account=                      # Project for billing

#SBATCH --job-name=SENGA                # Job name

#SBATCH --partition=gpu-a100-80         # Partition (queue) name
#SBATCH --qos=standard


cd $SLURM_SUBMIT_DIR
source setup_env_gpu_gnu_TURSA.sh


export OMP_NUM_THREADS=1
export OMP_PLACES=cores
export OMP_PROC_BIND=close

echo "SLURM_PROCID=${SLURM_PROCID}, SLURM_NTASKS_PER_NODE=${SLURM_NTASKS_PER_NODE}"

export MPICH_GPU_SUPPORT_ENABLED=1

# OPS MPI + CUDA C version
mpirun -np 4 ./senga2_f2c_mpi_cuda -gpudirect -OPS_DIAGS=2 OPS_FORCE_DECOMP_X=2 OPS_FORCE_DECOMP_Y=2 OPS_FORCE_DECOMP_Z=1 2>&1 | tee log_1node_4ranks_gnu_tursa_cuda_c.txt
