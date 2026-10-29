#!/bin/bash
#SBATCH -J n2pes
#SBATCH -p general
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=48:00:00
#SBATCH -A r01859
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=cshrikh@iu.edu
#SBATCH -o slurm_logs/%x_%A_%a.log

# Big Red 200: one line of $TASKFILE (R, variant) per array task, vDBF then
# the exact replay; submit with submit.sh, which sets TASKFILE and overrides
# --mem per group.  vDBF is single-threaded; the second core is for the replay.
# The max RSS (in practice the vDBF process) is appended to
# results/logs/maxrss.txt, as in the SPP showcase.

# Julia project for vDBF: your DBF.jl checkout (as in the old vDBF submit
# script).  Comment out to use the bundled copy instead (see config.sh).
export DBF_ENV=/N/u/cshrikh/BigRed200/Projects/trunc_DBF/DBF.jl

hostname
module reset
export PATH=$HOME/.juliaup/bin:$PATH
cd $SLURM_SUBMIT_DIR

read -r R VARIANT < <(sed -n "${SLURM_ARRAY_TASK_ID}p" ${TASKFILE:-tasks.txt})
TIMEFILE=slurm_logs/time_${R}_${VARIANT}.txt
echo "$R $VARIANT, $SLURM_CPUS_PER_TASK threads, ${SLURM_MEM_PER_NODE} MB"

/usr/bin/time -v bash run_one.sh "$R" "$VARIANT" "$SLURM_CPUS_PER_TASK" 2> >(tee $TIMEFILE >&2)
status=$?
wait

kilobytes=$(awk -F': ' '/Maximum resident set size/ {print $2}' $TIMEFILE)
[ -n "$kilobytes" ] && echo "$R $VARIANT status=$status mem_GB=$((SLURM_MEM_PER_NODE / 1024)) maxrss_GB=$(awk "BEGIN {printf \"%.2f\", $kilobytes / 1048576}") job=${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}" \
    >> DBF.jl/code/n2_pes/frozen_core/results/logs/maxrss.txt
exit $status
