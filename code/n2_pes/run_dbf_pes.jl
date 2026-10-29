# vDBF (`dbf_groundstate`) along the N2 curve, in each orbital basis.
#
# For every `data/R_<R>/hamiltonian_<basis>.jld2` (built by
# ../integrals/uhf_integrals_to_pauli.jl) the flow starts from the stored
# reference determinant.  Per point it writes `data/R_<R>/dbf_<basis>.json`
# (final numbers plus the per-iteration energy / variance / accumulated
# truncation-error history) and a :compact checkpoint every
# --checkpoint-interval iterations; `results/dbf_<basis>.csv` is rebuilt from
# all JSON files at the end, so partial reruns are fine.  The variance
# extrapolation (the paper's tail-fit protocol) is done afterwards by
# extrapolate.py from the stored histories.
#
#     julia --project=. run_dbf_pes.jl --bases mo,lmo,no,lno            # every R
#     julia --project=. run_dbf_pes.jl --bases no --R 1.10,2.50 --overwrite
#
# Defaults: CoeffTruncation(1e-5) on the operator, CoeffTruncation(1e-6) on the
# gradient, conv_thresh 1e-5, max_iter 200, max_rots_per_grad 50; PT2 off
# (compute_pt2 = compute_pt2_error = false), variance correction on.
# --max-flips k adds XWeightTruncation(k) to the operator truncation (drop
# terms flipping more than k qubits relative to |0...0>); --plateau-window w
# stops once <H> drops by less than --plateau-thresh over w iterations.
# --s2-penalty λ runs the flow on H + λ S^2, with S^2 read from s2_<basis>.jld2
# in the point directory (frozen_core/build_s2.jl writes it); the singlet
# ground state is unchanged and every energy in the output is of H + λ S^2.
# Those runs are saved as dbf_<basis>_s2pen<λ>.json / _ckpt / .csv, and a
# non-default --coeff adds _c<coeff> to the name.
# --prep <name> folds a fixed state-preparation circuit into the operator before
# the flow, O_0 = P_M' ... P_1' H P_1 ... P_M, read from prep_<name>_<basis>.jld2
# in the point directory (frozen_core/build_singlet_prep.jl writes `singlet`).
# The flow still starts from |0...0>, so the state is P_1 ... P_M U_1 ... U_K |0>.
# Terms below --prep-coeff are dropped while folding; <0|O_0|0> is checked
# against the stored circuit energy and the difference (the folding error) is
# saved.  Those runs get _prep<name> in the name.
#

using DBF, PauliOperators, JLD2, JSON, Printf

function parse_cli(args)
    opts = Dict{String,String}(
        "bases" => "mo,lmo,no,lno", "R" => "all",
        "coeff" => "1e-5", "grad" => "1e-6", "conv" => "1e-5",
        "energy-lowering" => "1e-6", "max-iter" => "200", "max-rots" => "50",
        "checkpoint-interval" => "10", "max-flips" => "none",
        "plateau-window" => "0", "plateau-thresh" => "1e-6", "s2-penalty" => "0",
        "prep" => "none", "prep-coeff" => "1e-8",
        "datadir" => "data", "results" => "results", "spv" => "true",
        "overwrite" => "false")
    i = 1
    while i <= length(args)
        key = lstrip(args[i], '-')
        haskey(opts, key) || error("unknown option $(args[i])")
        if key == "overwrite" && (i == length(args) || startswith(args[i+1], "--"))
            opts[key] = "true"; i += 1; continue
        end
        opts[key] = args[i+1]
        i += 2
    end
    return opts
end

"""
The flow's generator pool is [P, H] with P = I + sum_i Z_i, the projector
built around |0...0>.  For a number-conserving H and P written around any
other determinant that commutator vanishes, so -- as run_lno.jl did -- conjugate
H by X on every occupied qubit and start from |0...0>.  X Z X = -Z and
X Y X = -Y, so each term just picks up (-1)^popcount(z & occupied).
"""
function to_vacuum_reference(H::PauliSum{N,T}, ψ::Ket{N}) where {N,T}
    Hx = PauliSum(N, T)
    for (p, c) in H
        Hx[p] = isodd(count_ones(p.z & ψ.v)) ? -c : c
    end
    return Hx, Ket{N}(0)
end

point_dirs(datadir) = sort(filter(d -> startswith(d, "R_") && isdir(joinpath(datadir, d)),
                                  readdir(datadir)))
r_of(dir) = parse(Float64, dir[3:end])
penalty(opts) = parse(Float64, opts["s2-penalty"])
# Output name: dbf_<basis>, plus _prep<name> for a folded preparation circuit,
# _s2pen<λ> for penalty runs and _c<coeff> when the operator cutoff is not the
# default 1e-5, so variants never overwrite each other
function stem(basis, opts)
    s = "dbf_$(basis)"
    opts["prep"] == "none" || (s *= "_prep$(opts["prep"])")
    penalty(opts) == 0 || (s *= "_s2pen$(opts["s2-penalty"])")
    opts["coeff"] == "1e-5" || (s *= "_c$(opts["coeff"])")
    return s
end

function run_point(dir, basis, opts)
    hfile = joinpath(dir, "hamiltonian_$(basis).jld2")
    isfile(hfile) || error("missing $hfile -- run uhf_integrals_to_pauli.jl first")
    meta = JSON.parsefile(joinpath(dir, "metadata.json"))
    e_fci = meta["E_FCI_with_mu"] + meta["constant"]
    data = load(hfile)
    H, ψ = to_vacuum_reference(data["H"], data["reference_ket"])
    e_ref = data["reference_energy"]
    abs(real(expectation_value(H, ψ)) - e_ref) < 1e-8 ||
        error("<0|X H X|0> does not reproduce the reference energy")
    λ = penalty(opts)
    s2_ref = nothing
    if λ != 0
        S2, _ = to_vacuum_reference(load(joinpath(dir, "s2_$(basis).jld2"), "S2"), data["reference_ket"])
        s2_ref = real(expectation_value(S2, ψ))
        for (p, c) in S2
            H[p] = get(H, p, zero(valtype(H))) + λ * c
        end
        @printf(" Penalty: H + %g S^2   (<ref|S^2|ref> = %.6f, %d terms in total)\n", λ, s2_ref, length(H))
    end
    prep_info = nothing
    if opts["prep"] != "none"
        pf = load(joinpath(dir, "prep_$(opts["prep"])_$(basis).jld2"))
        gens, angs = pf["generators"], pf["angles"]
        pc = parse(Float64, opts["prep-coeff"])
        t = @elapsed for (g, θ) in zip(gens, angs)
            H = PauliOperators.evolve(H, g, θ)
            coeff_clip!(H, pc)
        end
        e_target = pf["E_prep"] + λ * pf["S2_prep"]
        e_fold = real(expectation_value(H, ψ))
        @printf(" Prep %s: %d rotations folded in %.0f s (cutoff %g), %d terms; <0|O_0|0> - circuit energy = %.1e\n",
                opts["prep"], length(gens), t, pc, length(H), e_fold - e_target)
        # The cutoff makes O_0 approximate; the error is recorded, and replay_fc.jl
        # measures the true energy of prep + flow
        abs(e_fold - e_target) < 1e-2 || error("folded operator is off by more than 10 mHa")
        prep_info = Dict("name" => opts["prep"], "rotations" => length(gens), "fidelity" => pf["fidelity"],
                         "E_prep" => pf["E_prep"], "S2_prep" => pf["S2_prep"], "fold_error" => e_fold - e_target)
    end

    # Complex storage: commutators of Hermitian operators carry factors of i.
    Hop = opts["spv"] == "true" ? SparsePauliVector(H) : H
    max_iter = parse(Int, opts["max-iter"])
    println("="^72)
    @printf(" R = %.2f A   basis %s   %d Pauli terms   E_ref = %.10f   E_FCI = %.10f\n",
            meta["R_angstrom"], basis, length(H), e_ref, e_fci)
    println("="^72)
    op_trunc = CoeffTruncation(parse(Float64, opts["coeff"]))
    if opts["max-flips"] != "none"
        op_trunc = CompositeTruncation(op_trunc, XWeightTruncation(parse(Int, opts["max-flips"])))
    end
    # The plateau stop is only passed when used: DBF builds without it (e.g. the
    # reviewer_2 branch) reject the keywords
    plateau = parse(Int, opts["plateau-window"]) == 0 ? (;) :
        (plateau_window=parse(Int, opts["plateau-window"]),
         plateau_thresh=parse(Float64, opts["plateau-thresh"]))
    t = @elapsed res = dbf_groundstate(Hop, ψ;
        operator_truncation=op_trunc,
        gradient_truncation=CoeffTruncation(parse(Float64, opts["grad"])),
        energy_lowering_thresh=parse(Float64, opts["energy-lowering"]),
        conv_thresh=parse(Float64, opts["conv"]),
        max_iter=max_iter,
        max_rots_per_grad=parse(Int, opts["max-rots"]),
        plateau...,
        compute_pt2=false, compute_pt2_error=false, compute_var_error=true,
        checkfile=joinpath(dir, "$(stem(basis, opts))_ckpt"),
        checkpoint_interval=parse(Int, opts["checkpoint-interval"]),
        checkpoint_mode=:compact)

    e_dbf = res["energies"][end]
    acc_err = res["accumulated_error"][end]
    var = res["variances"][end]
    acc_var_err = res["accumulated_var_error"][end]
    n_iter = length(res["energies_per_grad"]) - 1
    summary = Dict(
        "R" => meta["R_angstrom"], "basis" => basis,
        "E_ref" => e_ref, "E_FCI" => e_fci, "E_UHF" => meta["references"]["E_UHF"],
        "E_DBF" => e_dbf, "accumulated_error" => acc_err,
        "E_DBF_minus_error" => e_dbf - acc_err,
        "variance" => var, "accumulated_var_error" => acc_var_err,
        "variance_minus_error" => var - acc_var_err,
        "error_vs_FCI" => e_dbf - e_fci,
        "n_iterations" => n_iter, "n_rotations" => length(res["angles"]),
        "hit_max_iter" => n_iter >= max_iter,
        "final_terms" => length(res["hamiltonian"]), "wall_time_s" => t,
        "settings" => Dict(k => opts[k] for k in
            ("coeff", "grad", "conv", "energy-lowering", "max-iter", "max-rots",
             "checkpoint-interval", "spv", "max-flips", "plateau-window",
             "plateau-thresh", "s2-penalty", "prep", "prep-coeff")),
        "prep" => prep_info,
        "objective" => λ == 0 ? "H" : "H + $(λ) S^2", "S2_ref" => s2_ref,
        "history" => Dict(k => res[k] for k in
            ("energies_per_grad", "variance_per_grad",
             "accumulated_error_per_grad", "accumulated_var_error_per_grad")))
    open(joinpath(dir, "$(stem(basis, opts)).json"), "w") do io
        JSON.print(io, summary, 2)
    end
    @printf(" R = %.2f  %-3s  E_DBF = %.10f  err-corr = %.10f  var-corr = %.3e  E-E_FCI = %+.2e  (%d iter, %d rot, %.0f s)\n",
            meta["R_angstrom"], basis, e_dbf, e_dbf - acc_err, var - acc_var_err,
            e_dbf - e_fci, n_iter, length(res["angles"]), t)
    if e_dbf - acc_err < e_fci - 1e-8
        @printf(" !! below FCI by %.2e after removing the truncation error\n",
                e_fci - (e_dbf - acc_err))
    end
    return summary
end

const CSV_COLUMNS = ["R", "E_ref", "E_UHF", "E_FCI", "E_DBF", "accumulated_error",
                     "E_DBF_minus_error", "variance", "accumulated_var_error",
                     "variance_minus_error", "error_vs_FCI",
                     "n_iterations", "n_rotations", "hit_max_iter", "final_terms",
                     "wall_time_s"]

function write_csv(datadir, results, name)
    rows = []
    for d in point_dirs(datadir)
        f = joinpath(datadir, d, "$(name).json")
        isfile(f) && push!(rows, JSON.parsefile(f))
    end
    isempty(rows) && return
    path = joinpath(results, "$(name).csv")
    open(path, "w") do io
        println(io, join(CSV_COLUMNS, ","))
        for r in sort(rows; by=r -> r["R"])
            println(io, join([r[c] === nothing ? "NaN" : string(r[c]) for c in CSV_COLUMNS], ","))
        end
    end
    println("Wrote $path ($(length(rows)) points)")
end

function main(args)
    opts = parse_cli(args)
    datadir, results = opts["datadir"], opts["results"]
    mkpath(results)
    bases = split(opts["bases"], ",")
    dirs = point_dirs(datadir)
    if opts["R"] != "all"
        wanted = parse.(Float64, split(opts["R"], ","))
        dirs = filter(d -> any(isapprox(r_of(d), w; atol=1e-6) for w in wanted), dirs)
    end
    isempty(dirs) && error("no data/R_* points selected")
    for basis in bases, d in dirs
        dir = joinpath(datadir, d)
        if isfile(joinpath(dir, "$(stem(basis, opts)).json")) && opts["overwrite"] != "true"
            println("skip $d $basis (done; pass --overwrite to redo)")
            continue
        end
        run_point(dir, basis, opts)
        flush(stdout)
    end
    foreach(b -> write_csv(datadir, results, stem(b, opts)), bases)
end

main(ARGS)
