#!/bin/bash
# UNO + 0.05 S^2 vDBF (operator cutoff COEFF, default 1e-6) along the whole
# frozen-core PES, locally, with the exact replay of each circuit: the
# cluster_pes uno_s2pen task, run in this tree.  LANES lanes (default 2), each
# from the stretched end inward; points that already have a result or replay
# are skipped, so the script can be rerun.
#
#     caffeinate -i bash run_uno_pes_local.sh                    # from frozen_core/
#     COEFF=1e-5 LANES=4 caffeinate -i bash run_uno_pes_local.sh
#
# Per point: UNO Pauli Hamiltonian (converter, ED = FCI check) and S^2
# (build_s2.jl) if missing, then run_dbf_pes.jl, then replay_fc.jl.  Julia's
# heap is capped (HEAP, default 12G) so two runs fit in 32 GB: uncapped, the
# 2.3 A run's footprint peaked at 22 GB.
cd "$(dirname "$0")" || exit 1
J=${JULIA:-$HOME/.juliaup/bin/julia}
HEAP=${HEAP:-12G}
COEFF=${COEFF:-1e-6}
LANES=${LANES:-2}
L=results/logs
# run_dbf_pes.jl adds _c<coeff> to its file names unless the cutoff is its default, 1e-5
if [ "$COEFF" = 1e-5 ]; then STEM=dbf_uno_s2pen0.05; else STEM=dbf_uno_s2pen0.05_c$COEFF; fi
TAG=s2pen0.05_c$COEFF

point() {
    R=$1
    D=data_unocore/R_$R
    [ -f $D/hamiltonian_uno.jld2 ] ||
        $J --project=../../integrals ../../integrals/uhf_integrals_to_pauli.jl $D --basis uno --ed-max-dim 4000 \
            > $L/prepare_uno_$R.out 2>&1 || { echo "R=$R: converter failed"; return 1; }
    [ -f $D/s2_uno.jld2 ] ||
        $J --project=../../integrals build_s2.jl R_$R >> $L/prepare_uno_$R.out 2>&1 || { echo "R=$R: S^2 failed"; return 1; }
    if [ ! -f $D/$STEM.json ]; then
        (cd .. && $J --heap-size-hint=$HEAP --project=. run_dbf_pes.jl --bases uno --R $R --s2-penalty 0.05 --coeff $COEFF \
            --datadir frozen_core/data_unocore --results frozen_core/results \
            > frozen_core/$L/dbf_uno_${R}_$TAG.out 2>&1) || { echo "R=$R: vDBF failed"; return 1; }
    fi
    grep -h "^ R = .* E_DBF\|below FCI" $L/dbf_uno_${R}_$TAG.out
    [ -f replay_results/uno_${R}_trunc_$TAG.json ] ||
        $J -t 2 --heap-size-hint=$HEAP --project=../../integrals replay_fc.jl $D uno $STEM uno_${R}_trunc_$TAG \
            > $L/replay_uno_${R}_trunc_$TAG.out 2>&1 || { echo "R=$R: replay failed"; return 1; }
    echo "$(date '+%F %T')  R=$R done: $(grep -v '^ *\[\|@ ' $L/replay_uno_${R}_trunc_$TAG.out | tail -1 | cut -c1-70)"
}

lane() { for R in "$@"; do point $R; done; }

ALL=(3.00 2.90 2.80 2.70 2.60 2.50 2.40 2.30 2.20 2.10 2.00 1.90 1.80 1.70 1.60 1.50 1.40 1.30 1.20 1.10 1.00 0.90 0.80)
for ((k = 0; k < LANES; k++)); do
    lane $(for ((i = k; i < ${#ALL[@]}; i += LANES)); do echo ${ALL[i]}; done) &
done
wait
echo "$(date '+%F %T')  all lanes finished"
