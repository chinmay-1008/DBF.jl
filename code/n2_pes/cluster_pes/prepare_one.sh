#!/bin/bash
# Build one point's Pauli inputs from its integrals (run before run_one.sh):
#     hamiltonian_lno.jld2, hamiltonian_uno.jld2   converter; exact diagonalization = FCI check
#     s2_lno.jld2, s2_uno.jld2                     build_s2.jl; checked against the state-vector S^2
#     prep_<PREP>_lno.jld2                         build_singlet_prep.jl, 1 - F below PREP_TOL
# Each file is skipped if it already exists.
#
#     bash prepare_one.sh <R> [threads]
set -euo pipefail
BUNDLE=$(cd "$(dirname "$0")" && pwd)
source "$BUNDLE/config.sh"
R=$1; T=${2:-1}
cd "$BUNDLE/DBF.jl/code/n2_pes/frozen_core"
LOG=results/logs/prepare_$R.out
echo "$(date '+%F %T')  prepare R=$R  host $(hostname)" | tee -a "$LOG"
for tb in data_nocore:lno data_unocore:uno; do
    tree=${tb%:*}; b=${tb#*:}
    [ -f $tree/R_$R/hamiltonian_$b.jld2 ] ||
        "$JULIA" --project=../../integrals ../../integrals/uhf_integrals_to_pauli.jl $tree/R_$R --basis $b --ed-max-dim 4000 >> "$LOG" 2>&1
done
[ -f data_nocore/R_$R/s2_lno.jld2 ] && [ -f data_unocore/R_$R/s2_uno.jld2 ] ||
    "$JULIA" --project=../../integrals build_s2.jl R_$R >> "$LOG" 2>&1
[ -f data_nocore/R_$R/prep_${PREP}_lno.jld2 ] ||
    "$JULIA" -t "$T" --project=../../integrals build_singlet_prep.jl data_nocore/R_$R lno \
        --tol $PREP_TOL --name $PREP --fold-coeff none >> "$LOG" 2>&1
grep -h "PySCF FCI\|<ref|S^2|ref>\|^prep circuit" "$LOG" | cut -c1-150 || true
echo "$(date '+%F %T')  prepare R=$R done" | tee -a "$LOG"
