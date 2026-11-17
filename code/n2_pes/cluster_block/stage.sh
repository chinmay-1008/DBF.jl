#!/bin/bash
# Run LOCALLY (not on the cluster): copy the DBF package, the two pinned Julia
# environments, the block scripts and the Pauli inputs of each point in POINTS
# into ./DBF.jl, laid out as in the repo so every script runs unchanged.
#
#     bash stage.sh                    # R = 2.30
#     POINTS="1.10 2.30" bash stage.sh
#
# The inputs (hamiltonian_uno.jld2, s2_uno.jld2) are copied as built here, so the
# cluster runs start from the same H and S^2 as the local ones.  No results are copied.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
SRC=$(cd "$HERE/../../.." && pwd)          # the DBF.jl repo
FC=$SRC/code/n2_pes/frozen_core
DST=$HERE/DBF.jl
# outside the repo (e.g. on the cluster) the rm below would delete the staged bundle and copy nothing back
if [ ! -f "$SRC/code/n2_pes/run_block_pp.jl" ] || [ ! -d "$SRC/src" ]; then
    echo "stage.sh runs only from code/n2_pes/cluster_block inside the local DBF.jl repo; nothing changed" >&2
    exit 1
fi
rm -rf "$DST"
mkdir -p "$DST/code/integrals" "$DST/code/n2_pes/frozen_core/sv_results" "$DST/code/n2_pes/frozen_core/logs"

# DBF package; its [sources] entry points at a local PauliOperators checkout that
# does not exist on the cluster, and both environments pin PauliOperators from
# GitHub anyway, so drop it (as ../cluster_pes/stage.sh does)
cp -R "$SRC/src" "$SRC/ext" "$SRC/LICENSE" "$DST/"
awk '/^\[sources\]/{skip=1; next} skip && /^\[/{skip=0} !skip' "$SRC/Project.toml" > "$DST/Project.toml"

cp "$SRC/code/integrals/"{Project.toml,Manifest.toml} "$DST/code/integrals/"
cp "$SRC/code/n2_pes/"{Project.toml,Manifest.toml,run_block_pp.jl} "$DST/code/n2_pes/"
cp "$FC/"{replay_block.jl,sv_block.jl,sv_dbf.jl} "$DST/code/n2_pes/frozen_core/"

for R in ${POINTS:-2.30}; do
    d=data_unocore/R_$R
    mkdir -p "$DST/code/n2_pes/frozen_core/$d"
    for f in metadata.json hamiltonian_uno.jld2 s2_uno.jld2 uno_ab_overlap.npy; do
        [ -f "$FC/$d/$f" ] || { echo "missing frozen_core/$d/$f: build it locally first" >&2; exit 1; }
        cp "$FC/$d/$f" "$DST/code/n2_pes/frozen_core/$d/"
    done
done
# reference for setup.sh's check: the first 3 iterations of the local bare-H run at 1e-5
LOCALRUN=$FC/sv_results/uno_2.30_blockpp_c1e-5.json
if [ -f "$LOCALRUN" ]; then
    python3 - "$LOCALRUN" "$DST/check_reference.json" <<'EOF'
import json, sys
r, n = json.load(open(sys.argv[1])), 3
k = sum(r["rots_per_iter"][:n + 1])
json.dump({"from": r["label"], "iterations": n, "lam": r["history"]["lam"][:n + 1],
           "generators": r["generators"][:k], "angles": r["angles"][:k]}, open(sys.argv[2], "w"))
EOF
else
    echo "no $LOCALRUN: setup.sh's reproduction check will be skipped"
fi
echo "staged R = ${POINTS:-2.30} in $DST ($(du -sh "$DST" | cut -f1))"
