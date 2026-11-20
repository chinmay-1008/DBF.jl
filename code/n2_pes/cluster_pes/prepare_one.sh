#!/bin/bash
# Build one point's Pauli inputs from its integrals (run before run_one.sh):
#     hamiltonian_<b>.jld2   converter, b = mo lmo (data_uhfcore), no lno (data_nocore),
#                            uno (data_unocore); exact diagonalization = FCI check
#     s2_<b>.jld2            build_s2.jl; checked against the state-vector S^2
#     prep_<PREP>_<b>.jld2   build_singlet_prep.jl for mo lmo no lno, 1 - F below PREP_TOL;
#                            log in results/logs/prep_<b>_<R>.out
# Each file is skipped if it already exists.
#
#     bash prepare_one.sh <R> [threads]
set -euo pipefail
BUNDLE=$(cd "$(dirname "$0")" && pwd)
source "$BUNDLE/config.sh"
R=$1; T=${2:-1}
cd "$BUNDLE/DBF.jl/code/n2_pes/frozen_core"
LOG=results/logs/prepare_$R.out
TB="data_uhfcore:mo data_uhfcore:lmo data_nocore:no data_nocore:lno data_unocore:uno"
echo "$(date '+%F %T')  prepare R=$R  host $(hostname)" | tee -a "$LOG"
for tb in $TB; do
    tree=${tb%:*}; b=${tb#*:}
    [ -f $tree/R_$R/hamiltonian_$b.jld2 ] ||
        "$JULIA" --project=../../integrals ../../integrals/uhf_integrals_to_pauli.jl $tree/R_$R --basis $b --ed-max-dim 4000 >> "$LOG" 2>&1
done
missing=0
for tb in $TB; do [ -f ${tb%:*}/R_$R/s2_${tb#*:}.jld2 ] || missing=1; done
[ $missing = 0 ] || "$JULIA" --project=../../integrals build_s2.jl R_$R >> "$LOG" 2>&1
for tb in data_uhfcore:mo data_uhfcore:lmo data_nocore:no data_nocore:lno; do
    tree=${tb%:*}; b=${tb#*:}
    [ -f $tree/R_$R/prep_${PREP}_$b.jld2 ] ||
        "$JULIA" -t "$T" --project=../../integrals build_singlet_prep.jl $tree/R_$R $b \
            --tol $PREP_TOL --name $PREP --fold-coeff none > results/logs/prep_${b}_$R.out 2>&1
    echo "  $b: $(grep -h '^prep circuit' results/logs/prep_${b}_$R.out | cut -c1-100)" | tee -a "$LOG"
done
grep -h "PySCF FCI\|<ref|S^2|ref>" "$LOG" | cut -c1-150 || true
echo "$(date '+%F %T')  prepare R=$R done" | tee -a "$LOG"
