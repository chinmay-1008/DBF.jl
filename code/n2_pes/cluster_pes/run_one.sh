#!/bin/bash
# One PES point and variant: truncated vDBF, then the exact replay of its
# circuit (true <H>, Var, <S^2>, fidelity).  Both steps skip if their output
# already exists, so a task can simply be resubmitted.
#
#     bash run_one.sh <R> <variant> [threads]
#
#     lno_bare    LNO, UHF-like reference, H
#     lno_s2pen   LNO, H + LAMBDA S^2
#     lno_prep    LNO, singlet-projected start (prep circuit folded into H)
#     uno_s2pen   UNO closed-shell reference, H + LAMBDA S^2
set -euo pipefail
BUNDLE=$(cd "$(dirname "$0")" && pwd)
source "$BUNDLE/config.sh"
R=$1; VARIANT=$2; T=${3:-1}
N2=$BUNDLE/DBF.jl/code/n2_pes

case $VARIANT in
    lno_bare)  TREE=data_nocore;  B=lno; ARGS=();                              STEM=dbf_lno ;;
    lno_s2pen) TREE=data_nocore;  B=lno; ARGS=(--s2-penalty $LAMBDA);          STEM=dbf_lno_s2pen$LAMBDA ;;
    lno_prep)  TREE=data_nocore;  B=lno; ARGS=(--prep $PREP --prep-coeff $PREP_COEFF); STEM=dbf_lno_prep$PREP ;;
    uno_s2pen) TREE=data_unocore; B=uno; ARGS=(--s2-penalty $LAMBDA);          STEM=dbf_uno_s2pen$LAMBDA ;;
    *) echo "unknown variant $VARIANT" >&2; exit 1 ;;
esac
[ "$COEFF" = 1e-5 ] || STEM=${STEM}_c$COEFF
LABEL=${B}_${R}_${VARIANT}
LOGS=$N2/frozen_core/results/logs
# Under Slurm, ask Julia's GC to stay within 80% of the job's memory: left
# alone it lets garbage pile up to about twice the live operators (22 GB peak
# for uno_s2pen at 2.3 A) and the job would be killed for memory
JFLAGS=()
[ -n "${SLURM_MEM_PER_NODE:-}" ] && JFLAGS=(--heap-size-hint=$((SLURM_MEM_PER_NODE * 8 / 10))M)
echo "$(date '+%F %T')  R=$R  $VARIANT  stem $STEM  host $(hostname)  threads $T  ${JFLAGS[*]+"${JFLAGS[*]}"}  env $DBF_ENV"
# a reference already within PREP_TOL of its singlet projection (R <= 1.2 A,
# where UHF ~ RHF) has an empty prep circuit: that run would repeat lno_bare,
# and collect_pes.py reuses it
if [ "$VARIANT" = lno_prep ] && grep -q "^prep circuit: 0 rotations" "$LOGS/prepare_$R.out"; then
    echo "empty prep circuit at R=$R (reference within 1 - F < $PREP_TOL of a singlet): same as lno_bare, skipped"
    exit 0
fi

cd "$N2"
"$JULIA" -t "$T" ${JFLAGS[@]+"${JFLAGS[@]}"} --project="$DBF_ENV" run_dbf_pes.jl --bases $B --R "$R" --coeff $COEFF ${ARGS[@]+"${ARGS[@]}"} $EXTRA_ARGS \
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
