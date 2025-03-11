#!/bin/bash -l

#SBATCH --nodes=1                       # Total number of nodes 
#SBATCH --ntasks-per-node=128           # MPI ranks per node

#SBATCH --time=0-24:00:00               # Run time (d-hh:mm:ss)

#SBATCH --account=                      # Project for billing

#SBATCH --job-name=SENGA                # Job name

#SBATCH --partition=standard            # Partition (queue) name


cd $SLURM_SUBMIT_DIR

source setup_env_cpu_gnu_LUMI-C.sh 
 
export OMP_NUM_THREADS=1
export OMP_PLACES=cores
export OMP_PROC_BIND=close

export PMI_NO_PREINITIALIZE=y

# Original F90 version
time srun --distribution=block:block --hint=nomultithread --unbuffered --ntasks-per-node=128 --ntasks=128 ./senga2 2>&1 | tee log_1node_128ranks_gnu_lumi-c_f90.txt

# OPS MPI version
time srun --distribution=block:block --hint=nomultithread --unbuffered --ntasks-per-node=128 --ntasks=128 ./senga2_mpi -OPS_DIAGS=2 OPS_FORCE_DECOMP_X=8 OPS_FORCE_DECOMP_Y=4 OPS_FORCE_DECOMP_Z=4 2>&1 | tee log_1node_128ranks_gnu_lumi-c_ops.txt
