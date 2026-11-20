#!/bin/bash
# Submit the prepare array (one task per R in points.txt), then NODES whole-node
# jobs (node.slurm -> pool.py) that share the vDBF tasks; they start only once
# every prepare task has succeeded.
#
#     bash submit.sh                     # everything: 23 R x 14 variants
#     bash submit.sh '^2\.30 '           # only tasks.txt lines matching a pattern (grep -E)
#     NODES=6 bash submit.sh             # number of node jobs (default 4)
#     NOPREP=1 bash submit.sh            # skip the prepare array (inputs already built)
#     RETRY_FAILED=1 NOPREP=1 bash submit.sh   # resubmit; also retry tasks marked failed
#     COEFF=1e-5 NODES=1 NOPREP=1 bash submit.sh   # a sweep at another truncation (own queue_c1e-5/)
#
# Rerunning is safe: finished tasks are skipped (queue/done, and run_one.sh
# skips existing outputs), and claims of jobs that are gone are released.
set -euo pipefail
cd "$(dirname "$0")"
Q=queue; JN=n2node
if [ -n "${COEFF:-}" ] && [ "$COEFF" != 1e-6 ]; then Q=queue_c$COEFF; JN=n2node_c$COEFF; fi
export COEFF=${COEFF:-1e-6}
mkdir -p slurm_logs $Q/tasks
PATTERN=${1:-.}
f=$Q/tasks/tasks_$(date +%Y%m%d_%H%M%S)_$$.txt
grep -E "$PATTERN" tasks.txt > "$f" || true
n=$(wc -l < "$f" | tr -d ' ')
[ "$n" -gt 0 ] || { echo "no tasks match '$PATTERN'"; exit 1; }
[ -z "${RETRY_FAILED:-}" ] || { rm -f $Q/failed/*; echo "cleared failed markers"; }
DEP=()
if [ -z "${NOPREP:-}" ]; then
    PJ=$(sbatch --parsable --array=1-$(wc -l < points.txt | tr -d ' ') prepare.slurm)
    echo "prepare array: $PJ"
    DEP=(--dependency=afterok:${PJ%%;*})
fi
for ((i = 1; i <= ${NODES:-4}; i++)); do
    J=$(sbatch --parsable -J $JN "${DEP[@]+"${DEP[@]}"}" --export=ALL,TASKFILE=$f node.slurm)
    echo "node job $i: $J"
done
echo "coeff $COEFF: $n tasks in $f, shared by ${NODES:-4} node jobs; progress: tail -f $Q/events.log"
