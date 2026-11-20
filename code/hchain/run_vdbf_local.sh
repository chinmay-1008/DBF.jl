#!/bin/bash
# vDBF (UNO + 0.05 S^2, as the N2 UNO runs) on H-chain points, locally, LANES at a time.
# A task is "<n> <spacing> <coeff>"; tasks that already have a result are skipped.
#
#     bash run_vdbf_local.sh                         # default task list below
#     LANES=3 HEAP=8G bash run_vdbf_local.sh "4 1.00 1e-6" "10 2.00 1e-5"
#
# Output: data/H<n>/R_<d>/dbf_uno_s2pen0.05[_c<coeff>].json and _ckpt.jld2 (run_dbf_pes.jl),
# log logs/dbf_H<n>_<d>_c<coeff>.out (its per-iteration table has len(H)).
cd "$(dirname "$0")" || exit 1
HC=$(pwd)
J=${JULIA:-$HOME/.juliaup/bin/julia}
HEAP=${HEAP:-10G}
LANES=${LANES:-3}
MAXITER=${MAXITER:-300}

task() {
    local n=$1 d=$2 c=$3
    local stem=dbf_uno_s2pen0.05; [ "$c" = 1e-5 ] || stem=${stem}_c$c
    local log=$HC/logs/dbf_H${n}_${d}_c$c.out
    [ -f "$HC/data/H$n/R_$d/$stem.json" ] && { echo "skip H$n $d $c"; return; }
    (cd "$HC/../n2_pes" && $J --heap-size-hint=$HEAP --project=. run_dbf_pes.jl --bases uno --R $d --s2-penalty 0.05 \
        --coeff $c --max-iter $MAXITER --datadir "$HC/data/H$n" --results "$HC/results/H$n" > "$log" 2>&1) ||
        { echo "H$n $d $c FAILED (see $log)"; return; }
    echo "$(date '+%F %T')  H$n $d $c: $(grep -h '^ R = .* E_DBF' "$log" | tail -1 | cut -c1-140)"
}

if [ $# -gt 0 ]; then TASKS=("$@"); else
    TASKS=()
    for c in 1e-4 1e-5 1e-6; do for n in 4 6 8; do for d in 1.00 2.00; do TASKS+=("$n $d $c"); done; done; done
fi
mkdir -p logs
i=0
for t in "${TASKS[@]}"; do
    task $t &
    i=$((i + 1))
    if [ $((i % LANES)) = 0 ]; then wait; fi
done
wait
echo "$(date '+%F %T')  all tasks finished"
