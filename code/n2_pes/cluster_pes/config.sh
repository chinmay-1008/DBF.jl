# Settings shared by setup.sh, prepare_one.sh, run_one.sh and the SLURM jobs.
# The SLURM resources (Big Red 200: partition general, account r01859, time,
# memory, cores) are in the #SBATCH lines of prepare.slurm and run_task.slurm.

# Julia from juliaup, as for the SPP showcase (>= 1.11; the environments were
# resolved with 1.12.7).  Packages go to the default depot, ~/.julia.
export PATH=$HOME/.juliaup/bin:$PATH
JULIA=${JULIA:-julia}

# Julia project for the vDBF step.  Default: the bundled DBF copy (the code
# that produced the 2.3 A runs) via DBF.jl/code/n2_pes.  run_task.slurm may
# set DBF_ENV to another DBF.jl checkout; it then needs JSON loadable (in that
# project or the default ~/.julia environment) -- setup.sh checks.  The
# prepare and replay steps always use the bundled DBF.jl/code/integrals env.
DBF_ENV=${DBF_ENV:-$BUNDLE/DBF.jl/code/n2_pes}

# vDBF settings (as in the 2.3 A runs)
COEFF=1e-6          # operator truncation, CoeffTruncation
LAMBDA=0.05         # S^2 penalty for lno_s2pen and uno_s2pen
PREP=singlet5       # name of the singlet-projection prep circuit
PREP_TOL=1e-5       # its infidelity, 1 - F
PREP_COEFF=1e-6     # cutoff while folding the prep circuit into H
EXTRA_ARGS=${EXTRA_ARGS:-}   # passed through to run_dbf_pes.jl, e.g. "--max-iter 3" for a smoke test
