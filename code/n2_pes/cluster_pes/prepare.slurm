#!/bin/bash
#SBATCH -J n2prep
#SBATCH -p general
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=06:00:00
#SBATCH -A r01859
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=cshrikh@iu.edu
#SBATCH -o slurm_logs/%x_%A_%a.log

# Big Red 200: one R per array task (line of points.txt): Pauli Hamiltonians
# for the five bases, S^2, and the four singlet prep circuits.  submit.sh makes
# the node jobs wait for all of them.

hostname
module reset
export PATH=$HOME/.juliaup/bin:$PATH
cd $SLURM_SUBMIT_DIR

R=$(sed -n "${SLURM_ARRAY_TASK_ID}p" points.txt)
bash prepare_one.sh "$R" "$SLURM_CPUS_PER_TASK"
