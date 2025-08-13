#!/bin/bash
 
#SBATCH --nodes=1                       # Total number of nodes
#SBATCH --ntasks-per-node=8             # 8 MPI ranks per node

#SBATCH --time=24:00:00                 # Run time (hh:mm:ss)

#SBATCH --account=                      # Project for billing

#SBATCH --job-name=SENGA                # Job name

#SBATCH --partition=mi300x    	        # Partition (queue) name
#SBATCH --exclusive                     # no nodes allocation sharing with other running jobs


cd $SLURM_SUBMIT_DIR

source setup_env_gpu_icx_COSMA.sh

export OMP_NUM_THREADS=1
export OMP_PLACES=cores
export OMP_PROC_BIND=close

export I_MPI_PMI_LIBRARY=/lib64/libpmi.so

# OPS MPI + HIP C version
srun --ntasks=8 ./senga2_f2c_mpi_hip -OPS_DIAGS=2 OPS_FORCE_DECOMP_X=4 OPS_FORCE_DECOMP_Y=2 OPS_FORCE_DECOMP_Z=1 2>&1 | tee log_1node_8ranks_icx_cosma_hip_c.txt
