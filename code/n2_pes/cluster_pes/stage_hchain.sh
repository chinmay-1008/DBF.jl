#!/bin/bash
# Run LOCALLY: add the H-chain scan to the staged bundle (DBF.jl/code/hchain:
# scripts and the H10/H12 inputs built by make_hchain.py, the converter and
# build_s2_hchain.jl), without touching the rest of DBF.jl/ (stage.sh would wipe it).
#
#     bash stage_hchain.sh
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
SRC=$HERE/../../hchain
DST=$HERE/DBF.jl/code/hchain
mkdir -p "$DST/logs" "$DST/results"
cp "$SRC/"{statevector.jl,make_hchain.py,schrodinger_bestk.py,scan_plot.py,build_s2_hchain.jl} "$DST/"
for n in ${HCHAIN_N:-10 12}; do
    for d in "$SRC/data/H$n"/R_*; do
        mkdir -p "$DST/data/H$n/$(basename "$d")"
        cp "$d/"{metadata.json,hamiltonian_uno.jld2,s2_uno.jld2,uno_ab_overlap.npy,bestk.json} "$DST/data/H$n/$(basename "$d")/"
    done
done
echo "staged $(ls -d "$DST"/data/H*/R_* | wc -l | tr -d ' ') H-chain points; $(du -sh "$DST" | cut -f1)"
