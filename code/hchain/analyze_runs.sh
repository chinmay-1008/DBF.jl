#!/bin/bash
# statevector.jl on every finished H-chain vDBF run that has no analysis yet:
# results/sv_H<n>_<d>_c<coeff>.json (true <H> per iteration, truncated-state replay).
#
#     bash analyze_runs.sh                 # all runs
#     EXACT_MAX_N=20 bash analyze_runs.sh  # exact dense replay only up to this many qubits (default 24)
cd "$(dirname "$0")" || exit 1
J=${JULIA:-$HOME/.juliaup/bin/julia}
EXACT_MAX_N=${EXACT_MAX_N:-24}
mkdir -p results logs
for f in data/H*/R_*/dbf_uno_s2pen0.05*.json; do
    [ -e "$f" ] || continue
    dir=$(dirname "$f"); stem=$(basename "$f" .json)
    n=$(echo "$dir" | sed 's#data/H\([0-9]*\)/.*#\1#'); d=${dir##*/R_}
    c=${stem##*_c}; [ "$c" = "$stem" ] && c=1e-5
    out=results/sv_H${n}_${d}_c$c.json
    [ -f "$out" ] && continue
    extra=(); [ $((2 * n)) -le "$EXACT_MAX_N" ] || extra=(--no-exact)
    "$J" -t ${THREADS:-8} --project=../integrals statevector.jl "$dir" "$stem" logs/dbf_H${n}_${d}_c$c.out "$out" "${extra[@]}" \
        > logs/sv_H${n}_${d}_c$c.out 2>&1 || { echo "H$n $d $c: statevector failed (logs/sv_H${n}_${d}_c$c.out)"; continue; }
    echo "H$n $d $c: $(grep -h '^  iter' logs/sv_H${n}_${d}_c$c.out | tail -1 | cut -c1-110)"
done
