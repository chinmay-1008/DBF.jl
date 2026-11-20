#!/bin/bash
# One H-chain scan task (pool.py with RUNNER=hchain_run_one.sh): vDBF on
# UNO + 0.05 S^2 at one cutoff, then statevector.jl (true <H> per iteration,
# truncated-state replay of the final circuit).  Both steps skip if their output
# exists, so a task can be rerun.
#
#     bash hchain_run_one.sh <spacing> H<n>_c<coeff> [threads]      e.g. 2.00 H12_c1e-6 8
#
# Inputs (Hamiltonian, S^2, FCI vector) were built locally and staged into
# DBF.jl/code/hchain/data by stage_hchain.sh.  HEAP_HINT as in run_one.sh.
set -euo pipefail
BUNDLE=$(cd "$(dirname "$0")" && pwd)
source "$BUNDLE/config.sh"
D=$1; VARIANT=$2; T=${3:-4}
n=${VARIANT%%_*}; n=${n#H}; C=${VARIANT##*_c}
HC=$BUNDLE/DBF.jl/code/hchain
STEM=dbf_uno_s2pen0.05; [ "$C" = 1e-5 ] || STEM=${STEM}_c$C
LOG=$HC/logs/dbf_H${n}_${D}_c$C.out
mkdir -p "$HC/logs" "$HC/results"
JFLAGS=(); [ -z "${HEAP_HINT:-}" ] || JFLAGS=(--heap-size-hint=$HEAP_HINT)
echo "$(date '+%F %T')  H$n d=$D coeff $C  host $(hostname)  threads $T  ${JFLAGS[*]+"${JFLAGS[*]}"}  env $DBF_ENV"
if [ ! -f "$HC/data/H$n/R_$D/$STEM.json" ]; then
    cd "$BUNDLE/DBF.jl/code/n2_pes"
    "$JULIA" -t 1 ${JFLAGS[@]+"${JFLAGS[@]}"} --project="$DBF_ENV" run_dbf_pes.jl --bases uno --R "$D" --s2-penalty 0.05 \
        --coeff "$C" --max-iter ${MAXITER:-300} --datadir "$HC/data/H$n" --results "$HC/results/H$n" > "$LOG" 2>&1
fi
grep -h "^ R = .* E_DBF\|below FCI" "$LOG" || true
cd "$HC"
OUT=results/sv_H${n}_${D}_c$C.json
if [ ! -f "$OUT" ]; then
    X=(); [ $((2 * n)) -le 24 ] || X=(--no-exact)
    "$JULIA" -t "$T" --project=../integrals statevector.jl data/H$n/R_$D "$STEM" "$LOG" "$OUT" ${X[@]+"${X[@]}"} \
        > logs/sv_H${n}_${D}_c$C.out 2>&1
fi
grep -h "^  iter\|circuit" logs/sv_H${n}_${D}_c$C.out | tail -3 | cut -c1-120 || true
echo "$(date '+%F %T')  done"
