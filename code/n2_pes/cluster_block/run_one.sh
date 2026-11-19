#!/bin/bash
# One block vDBF run with Pauli propagation (run_block_pp.jl) at R, then the exact
# replay of its circuit (replay_block.jl: the true energies, variances, <S^2> and
# fidelity at every iteration).
#
#     bash run_one.sh <λ>                  # S^2 penalty λ (0 = bare H), default ranking
#     bash run_one.sh <λ>/<rank>           # ranking operator: d1c, exact, ds or dz (run_block_pp.jl --rank)
#     REPLAY_ONLY=1 bash run_one.sh <λ>    # replay what the run JSON holds, e.g. a run
#                                          # stopped at the time limit (saved every SAVE_EVERY iterations)
#     KETS=ref,3.4.5.6,3.5 KTAG=k3456_35 bash run_one.sh 0/ds    # other kets (config.sh)
#
# A run still going after RUN_LIMIT (config.sh) is stopped, and its last save is replayed.
#
# Output, under DBF.jl/code/n2_pes/frozen_core/, with label
# uno_<R>_blockpp_c<COEFF>[_<KTAG>][_<rank>][_s2pen<λ>] (rank only when it is not the default:
# d1c for the old kets, exact for KETS; exact on the old kets is "_rexact", as in the local runs):
#     sv_results/<label>.json          the run ("finished": true once done)
#     sv_results/<label>_replay.json   its replay
#     logs/<label>.out, logs/<label>_replay.out, logs/maxrss.txt   (peak memory, with GNU time)
set -euo pipefail
BUNDLE=$(cd "$(dirname "$0")" && pwd)
source "$BUNDLE/config.sh"
SPEC=$1
LAM=${SPEC%%/*}
RANK=
case $SPEC in */*) RANK=${SPEC#*/} ;; esac
N2=$BUNDLE/DBF.jl/code/n2_pes
FC=$N2/frozen_core
LABEL=uno_${R}_blockpp_c$COEFF
ARGS=(--coeff "$COEFF" --max-iter "$MAX_ITER" --save-every "$SAVE_EVERY")
DEFAULT=d1c
if [ -n "$KETS" ]; then
    [ -n "$KTAG" ] || { echo "KETS needs KTAG, the tag for the labels (e.g. k3456_35)" >&2; exit 1; }
    LABEL=${LABEL}_$KTAG
    ARGS+=(--kets "$KETS")
    DEFAULT=exact
fi
if [ -n "$RANK" ]; then
    ARGS+=(--rank "$RANK")
    if [ "$RANK" != "$DEFAULT" ]; then
        if [ "$RANK" = exact ]; then LABEL=${LABEL}_rexact; else LABEL=${LABEL}_$RANK; fi
    fi
fi
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
# the run's wall-clock limit: SIGTERM at RUN_LIMIT, SIGKILL 5 min later (the run JSON is written by rename,
# so a kill never leaves a half-written file)
LIMIT=()
if [ -n "$RUN_LIMIT" ] && command -v timeout > /dev/null; then
    LIMIT=(timeout --signal=TERM --kill-after=5m "$RUN_LIMIT")
fi

echo "$(date '+%F %T')  $LABEL  host $(hostname)  heap $HEAP  limit ${RUN_LIMIT:-none}  env $DBF_ENV"
echo "  run_block_pp.jl ${ARGS[*]} $EXTRA_ARGS"
if [ -n "${REPLAY_ONLY:-}" ]; then
    [ -f "$OUT" ] || { echo "no run JSON $OUT" >&2; exit 1; }
elif grep -q '"finished": *true' "$OUT" 2>/dev/null; then
    echo "finished run exists: $OUT"
else
    cd "$N2"
    set +e
    timed "$LABEL run" ${LIMIT[@]+"${LIMIT[@]}"} "$JULIA" -t 1 --heap-size-hint="$HEAP" --project="$DBF_ENV" \
        run_block_pp.jl frozen_core/data_unocore/R_$R uno "$LABEL" "${ARGS[@]}" $EXTRA_ARGS > "$FC/logs/$LABEL.out" 2>&1
    rc=$?
    set -e
    tail -1 "$FC/logs/$LABEL.out"
    if [ $rc = 124 ] || [ $rc = 137 ]; then
        echo "stopped at RUN_LIMIT $RUN_LIMIT; replaying the last save"
    elif [ $rc != 0 ]; then
        echo "run failed (exit $rc)" >&2
        exit $rc
    fi
    [ -f "$OUT" ] || { echo "no run JSON to replay: the run stopped before its first save" >&2; exit 1; }
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
