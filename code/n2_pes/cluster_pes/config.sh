# Settings shared by setup.sh, prepare_one.sh, run_one.sh, pool.py and the
# SLURM jobs.  The SLURM resources (partition general, account r01859, time,
# memory, cores) are in the #SBATCH lines of prepare.slurm and node.slurm.

# Julia from juliaup, as for the SPP showcase (>= 1.11; the environments were
# resolved with 1.12.7).  Packages go to the default depot, ~/.julia.
export PATH=$HOME/.juliaup/bin:$PATH
JULIA=${JULIA:-julia}

# Julia project for the vDBF step: a clone of chinmay-1008/DBF.jl, branch
# qual_exam, with nmayhall/PauliOperators.jl next to it (README, step 3).  If
# that directory does not exist, the bundled copy of the same code is used.
# The prepare and replay steps always use the bundled DBF.jl/code/integrals env.
DBF_ENV=${DBF_ENV:-/N/u/cshrikh/BigRed200/Projects/n2_dbf/DBF.jl}
[ -d "$DBF_ENV" ] || DBF_ENV=$BUNDLE/DBF.jl/code/n2_pes

# vDBF settings (as in the 2.3 A runs)
# operator truncation (CoeffTruncation); submit with COEFF=1e-5 for a second sweep,
# which gets its own queue (queue_c1e-5/) and replay labels (..._c1e-5)
COEFF=${COEFF:-1e-6}
LAMBDA=0.05         # S^2 penalty of the *_s2pen variants
PREP=singlet5       # name of the singlet-projection prep circuit (*_prep variants)
PREP_TOL=1e-5       # its infidelity, 1 - F
PREP_COEFF=1e-6     # cutoff while folding the prep circuit into H
EXTRA_ARGS=${EXTRA_ARGS:-}   # passed through to run_dbf_pes.jl, e.g. "--max-iter 3" for a smoke test
