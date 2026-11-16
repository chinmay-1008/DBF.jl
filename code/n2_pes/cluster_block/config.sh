# Settings shared by setup.sh, run_one.sh and block.slurm.  The SLURM resources
# (partition general, account r01859, time, whole node) are in block.slurm.
# Every setting can be overridden from the environment, e.g. COEFF=1e-5 bash run_one.sh 0.05

# Julia from juliaup, as for the PES bundle (../cluster_pes); packages go to ~/.julia
export PATH=$HOME/.juliaup/bin:$PATH
JULIA=${JULIA:-julia}

# Julia project for the run step: the DBF clone the PES runs use (chinmay-1008/DBF.jl,
# branch qual_exam, with nmayhall/PauliOperators.jl next to it; ../cluster_pes/README.md,
# step 3).  Its src/ is the same as excited_state's; run_block_pp.jl itself comes with this
# bundle.  JSON comes from the default environment, as for the PES runs.  If that
# directory does not exist, the bundled copy (DBF.jl/code/n2_pes, pinned) is used.
# The replay step always uses the bundled DBF.jl/code/integrals environment.
DBF_ENV=${DBF_ENV:-/N/u/cshrikh/BigRed200/Projects/n2_dbf/DBF.jl}
[ -d "$DBF_ENV" ] || DBF_ENV=$BUNDLE/DBF.jl/code/n2_pes

R=${R:-2.30}                        # the point; its inputs are staged by stage.sh (POINTS)
COEFF=${COEFF:-1e-6}                # operator cutoff, CoeffTruncation
MAX_ITER=${MAX_ITER:-500}           # as in the local 2.3 A runs at 1e-5
SAVE_EVERY=${SAVE_EVERY:-50}        # the run JSON is also written every SAVE_EVERY iterations
LAMBDAS=${LAMBDAS:-"0 0.02 0.05"}   # S^2 penalties, one run each (0 = bare H); block.slurm runs them side by side
HEAP=${HEAP:-70G}                   # Julia --heap-size-hint per run: 3 x 70G fits a 256 GB node
REPLAY_THREADS=${REPLAY_THREADS:-4}
EXTRA_ARGS=${EXTRA_ARGS:-}          # passed through to run_block_pp.jl
