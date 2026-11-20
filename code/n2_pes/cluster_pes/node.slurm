#!/bin/bash
#SBATCH -J n2node
#SBATCH -p general
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=128
#SBATCH --mem=0
#SBATCH --exclusive
#SBATCH --time=48:00:00
#SBATCH -A r01859
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=cshrikh@iu.edu
#SBATCH -o slurm_logs/%x_%j.log

# Big Red 200: one whole node (128 cores, all its memory: --mem=0) running
# pool.py, which keeps as many vDBF tasks going as fit in memory.  submit.sh
# starts several of these on the same task file.

hostname
module reset
export PATH=$HOME/.juliaup/bin:$PATH
cd $SLURM_SUBMIT_DIR
echo "job $SLURM_JOB_ID, $SLURM_CPUS_ON_NODE cpus, SLURM_MEM_PER_NODE=${SLURM_MEM_PER_NODE:-unset} MB, tasks ${TASKFILE:-tasks.txt}"
python3 pool.py --tasks "${TASKFILE:-tasks.txt}" ${POOL_ARGS:-}
