#!/bin/bash --login

#SBATCH --nodes=1                       # Total number of nodes

#SBATCH --time=24:00:00                 # Run time (hh:mm:ss)

#SBATCH --account=                      # Project for billing

#SBATCH --job-name=SENGA                # Job name

#SBATCH --partition=ampere              # Partition (queue) name
#SBATCH --ntasks=4
#SBATCH --gres=gpu:4

#! Number of nodes and tasks per node allocated by SLURM (do not change):
numnodes=$SLURM_JOB_NUM_NODES
numtasks=$SLURM_NTASKS
mpi_tasks_per_node=$(echo "$SLURM_TASKS_PER_NODE" | sed -e  's/^\([0-9][0-9]*\).*$/\1/')

. /etc/profile.d/modules.sh                # Leave this line (enables the module command)
module purge                               # Removes all modules still loaded
module load rhel8/default-amp              # REQUIRED - loads the basic environment

source setup_env_gpu_gnu_CSD3.sh

# Are you using OpenMP (NB this is unrelated to OpenMPI)? If so increase this
# safe value to no more than 128:
export OMP_NUM_THREADS=1
export OMP_PLACES=cores
export OMP_PROC_BIND=close

application="./senga2_f2c_mpi_cuda"

# Run options for the application:
options="-OPS_DIAGS=2 OPS_FORCE_DECOMP_X=2 OPS_FORCE_DECOMP_Y=2 OPS_FORCE_DECOMP_Z=1 2>&1 | tee log_1node_4ranks_nvhpc_cirrus_cuda_c.txt"

#! Number of MPI tasks to be started by the application per node and in total (do not change):
np=$[${numnodes}*${mpi_tasks_per_node}]

# Choose this for a pure shared-memory OpenMP parallel program on a single node:
# (OMP_NUM_THREADS threads will be created):
# CMD="$application $options"

# Choose this for a MPI code using OpenMPI:
CMD="mpirun -npernode $mpi_tasks_per_node -np $np $application $options"

###############################################################
### You should not have to change anything below this line ####
###############################################################

workdir="$SLURM_SUBMIT_DIR"
cd $workdir
echo -e "Changed directory to `pwd`.\n"

JOBID=$SLURM_JOB_ID

echo -e "JobID: $JOBID\n======"
echo "Time: `date`"
echo "Running on master node: `hostname`"
echo "Current directory: `pwd`"

if [ "$SLURM_JOB_NODELIST" ]; then
        #! Create a machine file:
        export NODEFILE=`generate_pbs_nodefile`
        cat $NODEFILE | uniq > machine.file.$JOBID
        echo -e "\nNodes allocated:\n================"
        echo `cat machine.file.$JOBID | sed -e 's/\..*$//g'`
fi

echo -e "\nnumtasks=$numtasks, numnodes=$numnodes, mpi_tasks_per_node=$mpi_tasks_per_node (OMP_NUM_THREADS=$OMP_NUM_THREADS)"

echo -e "\nExecuting command:\n==================\n$CMD\n"

eval $CMD 
