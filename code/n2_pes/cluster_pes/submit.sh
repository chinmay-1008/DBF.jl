#!/bin/bash
# Submit the prepare array (one task per R in points.txt), then the vDBF tasks
# as two arrays by memory, both starting only once every prepare task has
# succeeded:
#     small  lno_bare, uno_s2pen     MEM_SMALL (default 32G)
#     big    lno_s2pen, lno_prep     MEM_BIG   (default 64G)
# At 2.3 A the largest operators were ~11M terms (uno_s2pen; 22 GB peak
# footprint without a heap limit) and ~29M terms with 68M in [H, Z] (lno_s2pen).
#
#     bash submit.sh                     # everything
#     bash submit.sh '^2\.30 '           # only tasks.txt lines matching a pattern (grep -E)
#     MAXPAR=10 bash submit.sh           # at most 10 tasks of each array at once (default 20)
#     NOPREP=1 bash submit.sh            # skip the prepare array (inputs already built)
#     MEM_BIG=96G bash submit.sh         # e.g. resubmit with more memory; finished tasks skip
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p slurm_logs
PATTERN=${1:-.}
DEP=()
if [ -z "${NOPREP:-}" ]; then
    PJ=$(sbatch --parsable --array=1-$(wc -l < points.txt | tr -d ' ') prepare.slurm)
    echo "prepare array: $PJ"
    DEP=(--dependency=afterok:${PJ%%;*})
fi
for group in small:lno_bare\|uno_s2pen:${MEM_SMALL:-32G} big:lno_s2pen\|lno_prep:${MEM_BIG:-64G}; do
    IFS=: read -r name variants mem <<< "$group"
    f=slurm_logs/tasks_$name.txt
    grep -E "$PATTERN" tasks.txt | grep -E " ($variants)$" > $f || true
    n=$(wc -l < $f | tr -d ' ')
    [ "$n" -gt 0 ] || { echo "$name: no tasks"; continue; }
    J=$(sbatch --parsable "${DEP[@]+"${DEP[@]}"}" -J n2pes_$name --mem=$mem --export=ALL,TASKFILE=$f \
            --array="1-${n}%${MAXPAR:-20}" run_task.slurm)
    echo "$name array ($n tasks, $mem): $J   task list $f"
done
