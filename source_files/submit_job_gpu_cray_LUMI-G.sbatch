#!/bin/bash -l

#SBATCH --nodes=1                       # Total number of nodes 
#SBATCH --ntasks-per-node=8             # MPI ranks per node
#SBATCH --gpus-per-node=8               # Allocate one gpu per MPI rank

#SBATCH --time=0-24:00:00               # Run time (d-hh:mm:ss)

#SBATCH --account=                      # Project for billing

#SBATCH --job-name=SENGA                # Job name

#SBATCH --partition=standard-g          # Partition (queue) name


cd $SLURM_SUBMIT_DIR

source setup_env_gpu_cray_LUMI-G.sh

export OMP_NUM_THREADS=1
export OMP_PLACES=cores
export OMP_PROC_BIND=close

CPU_BIND="map_cpu:49,57,17,25,1,9,33,41"

export MPICH_GPU_SUPPORT_ENABLED=1

# OPS MPI + HIP C version
srun --cpu-bind=${CPU_BIND} --ntasks=8 ./senga2_f2c_mpi_hip -gpudirect -OPS_DIAGS=2 OPS_FORCE_DECOMP_X=4 OPS_FORCE_DECOMP_Y=2 OPS_FORCE_DECOMP_Z=1 2>&1 | tee log_1node_8ranks_cray_lumi-g_hip_c.txt
