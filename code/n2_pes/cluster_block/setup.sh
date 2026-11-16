#!/bin/bash
# Run ONCE on a login node (needs internet): install and precompile the two
# pinned Julia environments, then check that both load.
#     run     DBF.jl/code/n2_pes     run_block_pp.jl (DBF = the bundled src/)
#     replay  DBF.jl/code/integrals  replay_block.jl
set -euo pipefail
BUNDLE=$(cd "$(dirname "$0")" && pwd)
source "$BUNDLE/config.sh"
"$JULIA" --version
for env in DBF.jl/code/n2_pes DBF.jl/code/integrals; do
    echo "== $env"
    "$JULIA" --project="$BUNDLE/$env" -e 'using Pkg; Pkg.instantiate(); Pkg.precompile()'
done
"$JULIA" --project="$BUNDLE/DBF.jl/code/n2_pes" -e 'using DBF, PauliOperators, JLD2, JSON; println("ok: run environment")'
"$JULIA" --project="$BUNDLE/DBF.jl/code/integrals" -e 'using PauliOperators, JLD2, JSON, NPZ; println("ok: replay environment")'
