#!/bin/bash
# Run ONCE on a login node (needs internet).
#  1. install and precompile the Julia environments:
#        run     DBF_ENV from config.sh (the DBF clone, or the bundled DBF.jl/code/n2_pes)
#        replay  the bundled DBF.jl/code/integrals
#  2. check that the run environment reproduces the local runs: 3 iterations at cutoff 1e-5
#     must give the energies and rotations of each local 2.3 A run in DBF.jl/checks/ (made by
#     stage.sh: the old kets with D1c, and the new kets with the projector, D_S and dz), run with
#     that run's options.  This catches a PauliOperators clone or a DBF src/ that computes
#     differently from the local one.
set -euo pipefail
BUNDLE=$(cd "$(dirname "$0")" && pwd)
source "$BUNDLE/config.sh"
"$JULIA" --version
echo "== run environment: $DBF_ENV"
"$JULIA" --project="$DBF_ENV" -e 'using Pkg; Pkg.instantiate(); Pkg.precompile()'
"$JULIA" --project="$DBF_ENV" -e '
    using DBF, PauliOperators, JLD2
    try
        @eval using JSON
    catch
        println("MISSING: JSON. Install it in the default environment: julia -e \"using Pkg; Pkg.add(\\\"JSON\\\")\"")
        exit(1)
    end
    for (m, f) in ((DBF, :commutator_clipped), (DBF, :matvec), (PauliOperators, :SparsePauliVector),
                   (PauliOperators, :CoeffTruncation))
        isdefined(m, f) || (println("MISSING: ", m, ".", f); exit(1))
    end
    println("ok: run environment has what run_block_pp.jl uses")'
[ -d "$DBF_ENV/../PauliOperators.jl/.git" ] &&
    echo "   PauliOperators.jl next to it: $(git -C "$DBF_ENV/../PauliOperators.jl" log -1 --format='%h %ci')"
echo "== replay environment: DBF.jl/code/integrals"
"$JULIA" --project="$BUNDLE/DBF.jl/code/integrals" -e 'using Pkg; Pkg.instantiate(); Pkg.precompile()'
"$JULIA" --project="$BUNDLE/DBF.jl/code/integrals" -e 'using PauliOperators, JLD2, JSON, NPZ; println("ok: replay environment")'

shopt -s nullglob
REFS=("$BUNDLE"/DBF.jl/checks/*.json)
if [ ${#REFS[@]} = 0 ]; then
    echo "no DBF.jl/checks/*.json (stage.sh makes them from the local 2.3 A runs at 1e-5): reproduction checks skipped"
    exit 0
fi
CHK=$BUNDLE/DBF.jl/setup_check
cd "$BUNDLE/DBF.jl/code/n2_pes"
for REF in "${REFS[@]}"; do
    ARGS=$("$JULIA" --project="$DBF_ENV" -e 'using JSON; print(join(JSON.parsefile(ARGS[1])["args"], " "))' "$REF")
    echo "== reproduction check: 3 iterations at cutoff 1e-5 against the local $(basename "$REF" .json) ($ARGS)"
    rm -rf "$CHK"
    "$JULIA" -t 1 --project="$DBF_ENV" run_block_pp.jl frozen_core/data_unocore/R_2.30 uno check \
        --coeff 1e-5 --max-iter 3 --save-every 0 --outdir "$CHK" $ARGS > "$CHK.out" 2>&1 ||
        { tail -20 "$CHK.out"; echo "FAILED: the check run did not finish (log $CHK.out)"; exit 1; }
    "$JULIA" --project="$DBF_ENV" -e '
        using JSON
        a, b = JSON.parsefile(ARGS[1]), JSON.parsefile(ARGS[2])
        dl = maximum(abs(x - y) for (u, v) in zip(a["history"]["lam"], b["lam"]) for (x, y) in zip(u, v))
        ng = count(x == y for (x, y) in zip(a["generators"], b["generators"]))
        println("   largest energy difference ", dl, " Ha; ", ng, " of ", length(b["generators"]), " rotations identical")
        if dl < 1e-9 && ng == length(a["generators"]) == length(b["generators"])
            println("ok: the cluster run reproduces the local run")
        else
            println("DIFFERENT from the local run: check the PauliOperators.jl clone and src/ of DBF_ENV")
            exit(1)
        end' "$CHK/check.json" "$REF"
    rm -rf "$CHK" "$CHK.out"
done
