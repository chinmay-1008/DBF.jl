#!/bin/bash
# Run LOCALLY (not on the cluster): copy the code, the two pinned Julia
# environments and each point's integrals into ./DBF.jl, mirroring the repo's
# relative layout so every script runs unchanged, then write points.txt and
# tasks.txt.
#
#     bash stage.sh
#
# The integrals need PySCF, so they are built here first (in ../frozen_core):
#     python make_fc_integrals.py --R <all>                    -> data_uhfcore (mo, lmo), data_nocore (no, lno)
#     python make_uno_integrals.py --R <all> --bases uno       -> data_unocore
# Everything after that (Pauli Hamiltonians, S^2, prep circuits) is built on
# the cluster by prepare_one.sh.  No Pauli inputs or run results are copied.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
SRC=$(cd "$HERE/../../.." && pwd)          # the DBF.jl repo
FC=$SRC/code/n2_pes/frozen_core
DST=$HERE/DBF.jl
rm -rf "$DST"
mkdir -p "$DST/code/integrals" "$DST/code/n2_pes/results" \
         "$DST/code/n2_pes/frozen_core/results/logs" "$DST/code/n2_pes/frozen_core/replay_results"

# DBF package; its [sources] entry points at a local PauliOperators checkout
# that does not exist on the cluster, and the environments below pin
# PauliOperators from GitHub anyway, so drop it
cp -R "$SRC/src" "$SRC/ext" "$SRC/LICENSE" "$DST/"
awk '/^\[sources\]/{skip=1; next} skip && /^\[/{skip=0} !skip' "$SRC/Project.toml" > "$DST/Project.toml"

cp "$SRC/code/integrals/"{Project.toml,Manifest.toml,uhf_integrals_to_pauli.jl} "$DST/code/integrals/"
cp "$SRC/code/n2_pes/"{Project.toml,Manifest.toml,run_dbf_pes.jl,extrapolate.py} "$DST/code/n2_pes/"
cp "$FC/results/fc_cc.csv" "$DST/code/n2_pes/frozen_core/results/"
cp "$SRC/code/n2_pes/results/references.csv" "$DST/code/n2_pes/results/"
cp "$FC/"{replay_fc.jl,sv_dbf.jl,build_s2.jl,build_singlet_prep.jl} "$DST/code/n2_pes/frozen_core/"

# every basis bare and with the S^2 penalty; the singlet-projected start for
# the four spin-broken references (UNO's is already a singlet)
VARIANTS="mo_bare lmo_bare no_bare lno_bare uno_bare mo_s2pen lmo_s2pen no_s2pen lno_s2pen uno_s2pen
          mo_prep lmo_prep no_prep lno_prep"
: > "$HERE/points.txt"; : > "$HERE/tasks.txt"
for d in $(ls "$FC/data_nocore" | grep '^R_'); do
    R=${d#R_}
    for tb in data_uhfcore:mo data_uhfcore:lmo data_nocore:no data_nocore:lno data_unocore:uno; do
        tree=${tb%:*}; b=${tb#*:}
        mkdir -p "$DST/code/n2_pes/frozen_core/$tree/$d"
        for f in metadata.json ${b}_h1_a.npy ${b}_h1_b.npy ${b}_eri_aa.npy ${b}_eri_ab.npy ${b}_eri_bb.npy ${b}_ab_overlap.npy; do
            [ -f "$FC/$tree/$d/$f" ] || { echo "missing $tree/$d/$f" >&2; exit 1; }
            cp "$FC/$tree/$d/$f" "$DST/code/n2_pes/frozen_core/$tree/$d/"
        done
    done
    echo "$R" >> "$HERE/points.txt"
    for v in $VARIANTS; do echo "$R $v" >> "$HERE/tasks.txt"; done
done
echo "staged $(wc -l < "$HERE/points.txt" | tr -d ' ') points, $(wc -l < "$HERE/tasks.txt" | tr -d ' ') vDBF tasks; $(du -sh "$DST" | cut -f1)"
