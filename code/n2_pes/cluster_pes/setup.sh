#!/bin/bash
# Run ONCE on a login node (needs internet): install the two pinned Julia
# environments into $JULIA_DEPOT_PATH and precompile them, then check that
# both load.
set -euo pipefail
BUNDLE=$(cd "$(dirname "$0")" && pwd)
source "$BUNDLE/config.sh"
"$JULIA" --version
for env in DBF.jl/code/n2_pes DBF.jl/code/integrals; do
    echo "== $env"
    "$JULIA" --project="$BUNDLE/$env" -e 'using Pkg; Pkg.instantiate(); Pkg.precompile()'
done
"$JULIA" --project="$BUNDLE/DBF.jl/code/n2_pes" -e 'using DBF, PauliOperators, JLD2, JSON; println("ok: bundled vDBF environment")'
# the DBF_ENV from config.sh, if it is not the bundled copy: does it have what run_dbf_pes.jl uses?
ENV_SLURM=$DBF_ENV
if [ "$ENV_SLURM" != "$BUNDLE/DBF.jl/code/n2_pes" ]; then
    echo "== config.sh: DBF_ENV=$ENV_SLURM"
    "$JULIA" --project="$ENV_SLURM" -e 'using Pkg; Pkg.instantiate()'
    "$JULIA" --project="$ENV_SLURM" -e '
        using DBF, PauliOperators, JLD2, Printf
        let ok = true
        try; @eval using JSON; catch; ok = false
            println("MISSING: JSON. Install it in the default environment: julia -e \"using Pkg; Pkg.add(\\\"JSON\\\")\""); end
        isdefined(PauliOperators, :SparsePauliVector) || (ok = false; println("MISSING: PauliOperators.SparsePauliVector"))
        kw = Base.kwarg_decl(first(methods(DBF.dbf_groundstate)))
        for k in (:operator_truncation, :gradient_truncation, :energy_lowering_thresh, :max_rots_per_grad, :compute_var_error,
                  :checkfile, :checkpoint_interval, :checkpoint_mode)
            k in kw || (ok = false; println("MISSING: dbf_groundstate keyword ", k))
        end
        println(ok ? "ok: DBF_ENV has everything run_dbf_pes.jl uses" :
                     "DBF_ENV is not compatible: fix it, or set DBF_ENV in config.sh to the bundled copy")
        end'
fi
"$JULIA" --project="$BUNDLE/DBF.jl/code/integrals" -e 'using PauliOperators, JLD2, JSON, NPZ; println("ok: prepare/replay environment")'
