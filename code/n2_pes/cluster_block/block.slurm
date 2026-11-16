#!/bin/bash
#SBATCH -J n2block
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

# Big Red 200: one whole node (all its memory: --mem=0) runs the block vDBF runs
# of config.sh side by side, by default bare H, λ = 0.02 and λ = 0.05 at R = 2.30
# with cutoff 1e-6, each followed by its replay.  Each run is single-threaded;
# memory, not cores, is what the node is for.
#
#     mkdir -p slurm_logs && sbatch block.slurm
#     COEFF=1e-6 LAMBDAS="0.02" sbatch block.slurm     # settings pass through (sbatch exports the environment)

hostname
module reset
cd "$SLURM_SUBMIT_DIR"
source config.sh
echo "job $SLURM_JOB_ID: R $R, cutoff $COEFF, $MAX_ITER iterations, λ = $LAMBDAS, heap $HEAP per run"
pids=()
for lam in $LAMBDAS; do
    bash run_one.sh "$lam" > "slurm_logs/run_${lam}_$SLURM_JOB_ID.log" 2>&1 &
    pids+=($!)
done
fail=0
for p in "${pids[@]}"; do
    wait "$p" || fail=1
done
echo "$(date '+%F %T')  all runs ended (any failed: $fail)"
exit $fail
