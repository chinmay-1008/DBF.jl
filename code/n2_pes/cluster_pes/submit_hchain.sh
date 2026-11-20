#!/bin/bash
# Submit the H-chain scan (hchain_tasks.txt) to whole nodes running pool.py, with
# its own queue (queue_hchain/), runner (hchain_run_one.sh) and memory table
# (memory_hchain.txt).  The N2 queue is untouched.
#
#     bash submit_hchain.sh                    # NODES (default 1) node jobs
#     NODES=2 bash submit_hchain.sh 'H12'      # only tasks matching a pattern (grep -E)
#     RETRY_FAILED=1 bash submit_hchain.sh
set -euo pipefail
cd "$(dirname "$0")"
Q=queue_hchain
mkdir -p slurm_logs $Q/tasks
f=$Q/tasks/tasks_$(date +%Y%m%d_%H%M%S)_$$.txt
grep -E "${1:-.}" hchain_tasks.txt > "$f" || true
n=$(wc -l < "$f" | tr -d ' ')
[ "$n" -gt 0 ] || { echo "no tasks match '${1:-.}'"; exit 1; }
[ -z "${RETRY_FAILED:-}" ] || { rm -f $Q/failed/*; echo "cleared failed markers"; }
export QUEUE=$Q RUNNER=hchain_run_one.sh MEMFILE=memory_hchain.txt
for ((i = 1; i <= ${NODES:-1}; i++)); do
    J=$(sbatch --parsable -J n2hchain --export=ALL,TASKFILE=$f,POOL_ARGS="--threads 8" node.slurm)
    echo "node job $i: $J"
done
echo "$n tasks in $f, shared by ${NODES:-1} node jobs; progress: tail -f $Q/events.log"
