#!/bin/bash
# One block vDBF run with Pauli propagation (run_block_pp.jl) at R with S^2
# penalty λ (0 = bare H), then the exact replay of its circuit (replay_block.jl:
# the true energies, variances and <S^2> at every iteration).
#
#     bash run_one.sh <λ>                  # the run (skipped if finished), then the replay
#     REPLAY_ONLY=1 bash run_one.sh <λ>    # replay what the run JSON holds, e.g. a run
#                                          # stopped at the time limit (saved every SAVE_EVERY iterations)
#
# Output, under DBF.jl/code/n2_pes/frozen_core/:
#     sv_results/uno_<R>_blockpp_c<COEFF>[_s2pen<λ>].json          the run ("finished": true once done)
#     sv_results/uno_<R>_blockpp_c<COEFF>[_s2pen<λ>]_replay.json   its replay
#     logs/<label>.out, logs/<label>_replay.out, logs/maxrss.txt   (peak memory, with GNU time)
set -euo pipefail
BUNDLE=$(cd "$(dirname "$0")" && pwd)
source "$BUNDLE/config.sh"
LAM=$1
N2=$BUNDLE/DBF.jl/code/n2_pes
FC=$N2/frozen_core
LABEL=uno_${R}_blockpp_c$COEFF
ARGS=(--coeff "$COEFF" --max-iter "$MAX_ITER" --save-every "$SAVE_EVERY")
if [ "$LAM" != 0 ]; then LABEL=${LABEL}_s2pen$LAM; ARGS+=(--s2-penalty "$LAM"); fi
OUT=$FC/sv_results/$LABEL.json
mkdir -p "$FC/logs" "$FC/sv_results"

# peak memory and wall time of each step, where /usr/bin/time is GNU time (Linux)
timed() {
    local tag=$1; shift
    if /usr/bin/time --version 2>&1 | grep -q GNU; then
        /usr/bin/time -a -o "$FC/logs/maxrss.txt" -f "$tag: maxrss %M KB, %e s, exit %x" "$@"
    else
        "$@"
    fi
}

echo "$(date '+%F %T')  $LABEL  host $(hostname)  heap $HEAP  env $DBF_ENV"
if [ -n "${REPLAY_ONLY:-}" ]; then
    [ -f "$OUT" ] || { echo "no run JSON $OUT" >&2; exit 1; }
elif grep -q '"finished": *true' "$OUT" 2>/dev/null; then
    echo "finished run exists: $OUT"
else
    cd "$N2"
    timed "$LABEL run" "$JULIA" -t 1 --heap-size-hint="$HEAP" --project="$DBF_ENV" run_block_pp.jl \
        frozen_core/data_unocore/R_$R uno "$LABEL" "${ARGS[@]}" $EXTRA_ARGS > "$FC/logs/$LABEL.out" 2>&1
    tail -1 "$FC/logs/$LABEL.out"
fi

cd "$FC"
REP=sv_results/${LABEL}_replay.json
if [ -f "$REP" ] && [ "$REP" -nt "$OUT" ]; then
    echo "replay is newer than the run: $REP"
else
    timed "$LABEL replay" "$JULIA" -t "$REPLAY_THREADS" --project=../../integrals replay_block.jl \
        data_unocore/R_$R uno "$LABEL" 1 > "logs/${LABEL}_replay.out" 2>&1
    tail -1 "logs/${LABEL}_replay.out" | cut -c1-160
fi
echo "$(date '+%F %T')  done"
