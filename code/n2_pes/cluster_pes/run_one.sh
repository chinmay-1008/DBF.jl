#!/bin/bash
# One PES point and variant: truncated vDBF, then the exact replay of its
# circuit (true <H>, Var, <S^2>, fidelity).  Both steps skip if their output
# already exists, so a task can simply be rerun.
#
#     bash run_one.sh <R> <variant> [threads]
#
# variant = <basis>_<kind>
#     basis   mo, lmo   canonical / Boys-localized UHF orbitals      (data_uhfcore)
#             no, lno   UCCSD natural orbitals / localized            (data_nocore)
#             uno       UHF natural orbitals, closed-shell reference  (data_unocore)
#     kind    bare      H
#             s2pen     H + LAMBDA S^2
#             prep      H, from the singlet projection of the reference (prep
#                       circuit folded into H); not for uno, already a singlet
#
# COEFF (default 1e-6, config.sh) is the vDBF truncation; other values get
# their own replay label, <basis>_<R>_<variant>_c<COEFF>.
#
# HEAP_HINT (e.g. 48G), set by pool.py, is passed to Julia's --heap-size-hint;
# otherwise, under Slurm, 80% of the job's memory.
set -euo pipefail
BUNDLE=$(cd "$(dirname "$0")" && pwd)
source "$BUNDLE/config.sh"
R=$1; VARIANT=$2; T=${3:-1}
N2=$BUNDLE/DBF.jl/code/n2_pes

B=${VARIANT%_*}; KIND=${VARIANT##*_}
case $B in
    mo|lmo) TREE=data_uhfcore ;;
    no|lno) TREE=data_nocore ;;
    uno)    TREE=data_unocore ;;
    *) echo "unknown basis in $VARIANT" >&2; exit 1 ;;
esac
case $KIND in
    bare)  ARGS=();                                      STEM=dbf_$B ;;
    s2pen) ARGS=(--s2-penalty $LAMBDA);                  STEM=dbf_${B}_s2pen$LAMBDA ;;
    prep)  [ $B != uno ] || { echo "uno_prep: the UNO reference is already a singlet" >&2; exit 1; }
           ARGS=(--prep $PREP --prep-coeff $PREP_COEFF); STEM=dbf_${B}_prep$PREP ;;
    *) echo "unknown kind in $VARIANT" >&2; exit 1 ;;
esac
[ "$COEFF" = 1e-5 ] || STEM=${STEM}_c$COEFF
LABEL=${B}_${R}_${VARIANT}
[ "$COEFF" = 1e-6 ] || LABEL=${LABEL}_c$COEFF
LOGS=$N2/frozen_core/results/logs
# Left alone, Julia's GC lets garbage pile up to about twice the live
# operators (22 GB peak for uno_s2pen at 2.3 A), so give it a target
JFLAGS=()
if [ -n "${HEAP_HINT:-}" ]; then
    JFLAGS=(--heap-size-hint=$HEAP_HINT)
elif [ -n "${SLURM_MEM_PER_NODE:-}" ] && [ "$SLURM_MEM_PER_NODE" -gt 0 ]; then
    JFLAGS=(--heap-size-hint=$((SLURM_MEM_PER_NODE * 8 / 10))M)
fi
echo "$(date '+%F %T')  R=$R  $VARIANT  stem $STEM  host $(hostname)  threads $T  ${JFLAGS[*]+"${JFLAGS[*]}"}  env $DBF_ENV"
# A reference already within PREP_TOL of its singlet projection (short R,
# where UHF = RHF) has an empty prep circuit: that run would repeat <basis>_bare,
# and collect_pes.py reuses it
if [ "$KIND" = prep ] && grep -q "^prep circuit: 0 rotations" "$LOGS/prep_${B}_$R.out"; then
    echo "empty prep circuit at R=$R (reference within 1 - F < $PREP_TOL of a singlet): same as ${B}_bare, skipped"
    exit 0
fi

cd "$N2"
"$JULIA" -t 1 ${JFLAGS[@]+"${JFLAGS[@]}"} --project="$DBF_ENV" run_dbf_pes.jl --bases $B --R "$R" --coeff $COEFF ${ARGS[@]+"${ARGS[@]}"} $EXTRA_ARGS \
    --datadir frozen_core/$TREE --results frozen_core/results > "$LOGS/dbf_$LABEL.out" 2>&1
grep -h "^ R = .* E_DBF\|below FCI" "$LOGS/dbf_$LABEL.out" || true

cd "$N2/frozen_core"
if [ -f "replay_results/$LABEL.json" ]; then
    echo "replay exists: replay_results/$LABEL.json"
else
    "$JULIA" -t "$T" ${JFLAGS[@]+"${JFLAGS[@]}"} --project=../../integrals replay_fc.jl $TREE/R_$R $B $STEM $LABEL > "$LOGS/replay_$LABEL.out" 2>&1
    grep -v "^ *\[\|@ " "$LOGS/replay_$LABEL.out" | tail -3 | cut -c1-120
fi
echo "$(date '+%F %T')  done"
