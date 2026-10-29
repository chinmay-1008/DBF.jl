# Exact replay of a truncated dbf_groundstate run on the frozen-core problem:
# apply its stored generators and angles to |0> with no truncation and report
# the true <H>, Var(H), <S^2> and fidelity with the exact ground state.
#
# DBF evolves O <- U' O U with the newest rotation acting first on |0>, so after
# K rotations psi = U_1 U_2 ... U_K |0> with U_k = exp(-i θ_k G_k / 2); each
# checkpoint is replayed from |0>.  The first iteration must reproduce DBF's own
# energy (of H + λ S^2 for penalty runs) to 1e-5, or the script stops.
#
#     julia -t 4 --project=../../integrals replay_fc.jl <point dir> <basis> <run stem> <label> [λ ckpt]
#     e.g. replay_fc.jl data_nocore/R_2.30 lno dbf_lno_s2pen0.05 lno_2.30_trunc_s2pen0.05
# For a run still in progress (no <run stem>.json yet) pass λ and a copy of its
# latest checkpoint; the replay then covers the iterations saved so far.
# Runs with --prep <name> (settings "prep", or a 7th argument for runs in
# progress) put the stored preparation circuit in front of DBF's rotations, so
# psi = P_1 ... P_M U_1 ... U_K |0>; iteration 0 is then the prepared state.
include(joinpath(@__DIR__, "sv_dbf.jl"))

function apply_rotation!(ψ, G, θ, states, pos)
    c, s = cos(θ / 2), sin(θ / 2)
    x = UInt64(G.x)
    out = c .* ψ
    for (j, b) in enumerate(states)
        a = ψ[j]
        a == 0 && continue
        ph, k = G * Ket{N}(b)
        i = pos[UInt64(k.v)+1]
        i == 0 && error("generator x string $(x) leaves the sector")
        out[i] += -1im * s * ph * a
    end
    ψ .= out
    return ψ
end

function replay(gens, angs, n, states, pos)
    ψ = zeros(ComplexF64, length(states))
    ψ[1] = 1
    for r in n:-1:1
        apply_rotation!(ψ, gens[r], angs[r], states, pos)
    end
    return ψ
end

function main_replay(args)
    dir, basis, runstem, label = args[1:4]
    meta = JSON.parsefile(joinpath(dir, "metadata.json"))
    d = load(joinpath(dir, "hamiltonian_$(basis).jld2"))
    ref = UInt64(d["reference_ket"].v)
    e_fci = meta["E_FCI_with_mu"] + meta["constant"]
    H = to_vacuum_reference(d["H"], ref)
    Ovl = npzread(joinpath(dir, "$(basis)_ab_overlap.npy"))
    Ovl[abs.(Ovl) .< 1e-6] .= 0

    gb = gf2_basis(unique(UInt64(p.x) for p in keys(H)))
    states = enumerate_span(gb)
    pos = zeros(Int32, 1 << N)
    for (k, b) in enumerate(states)
        pos[b+1] = k
    end
    Hs = sector_matrix(H, states, pos)
    S2s = s2_matrix(states, pos, ref, Ovl)
    amask = UInt64(1) << NORB - 1
    blk = [k for (k, b) in enumerate(states)
           if count_ones((b ⊻ ref) & amask) == meta["n_alpha"] && count_ones((b ⊻ ref) >> NORB) == meta["n_beta"]]
    F = eigen(Symmetric(Matrix(Hs[blk, blk])))
    Φ0 = zeros(length(states)); Φ0[blk] = F.vectors[:, 1]
    abs(F.values[1] - e_fci) < 1e-8 || error("sector ground state is not FCI")

    if length(args) >= 6
        λ, ckpt = parse(Float64, args[5]), args[6]
        prep = length(args) >= 7 ? args[7] : "none"
        fold_err = 0.0
    else
        run = JSON.parsefile(joinpath(dir, "$(runstem).json"))
        λ = parse(Float64, get(run["settings"], "s2-penalty", "0"))
        prep = get(run["settings"], "prep", "none")
        fold_err = prep == "none" ? 0.0 : run["prep"]["fold_error"]
        ckpt = joinpath(dir, "$(runstem)_ckpt.jld2")
    end
    o = load(ckpt)["out"]
    gens, angs = o["generators"], o["angles"]
    niter = length(o["energies_per_grad"]) - 1
    per = length(gens) ÷ niter
    per * niter == length(gens) || error("rotations per iteration not constant")
    m = 0
    if prep != "none"
        pf = load(joinpath(dir, "prep_$(prep)_$(basis).jld2"))
        m = length(pf["generators"])
        gens, angs = vcat(pf["generators"], gens), vcat(pf["angles"], angs)
        @printf("%s: prep circuit %s with %d rotations in front\n", label, prep, m)
    end

    measure(ψ) = begin
        hψ = Hs * ψ
        e = real(dot(ψ, hψ))
        (e, real(dot(hψ, hψ)) - e^2, real(dot(ψ, S2s * ψ)), abs2(dot(Φ0, ψ)))
    end
    # Convention check on the first iteration.  The wrong order or angle sign is
    # off by ~0.1 Ha; what is left is the first iteration's truncation, which
    # E - err accounts for to first order.
    e1, _, s1, _ = measure(replay(gens, angs, m + per, states, pos))
    # A folded prep circuit was truncated, so DBF's operator is off by about the
    # folding error; take it out and allow 1e-3 instead of 1e-4
    dev = abs(e1 + λ * s1 + fold_err - (o["energies_per_grad"][2] - o["accumulated_error_per_grad"][2]))
    @printf("%s: λ = %g, %d iterations x %d rotations; iteration-1 check |E_replay - (E - err)_dbf| = %.1e\n",
            label, λ, niter, per, dev)
    dev < (prep == "none" ? 1e-4 : 1e-3) || error("replay does not reproduce DBF's first iteration")

    iters = filter(<=(niter), sort(unique(vcat([0, 1, 2, 5], collect(10:10:niter), [niter]))))
    rows = []
    @printf("%5s %11s %10s %8s %8s | %11s %11s\n", "iter", "true E-FCI", "true var", "<S^2>", "fidelity",
            "DBF E-FCI", "DBF E-err")
    for it in iters
        e, v, s2, f = measure(replay(gens, angs, m + it * per, states, pos))
        push!(rows, Dict("iter" => it, "E" => e, "var" => v, "S2" => s2, "fidelity" => f))
        E = o["energies_per_grad"][it+1]; err = o["accumulated_error_per_grad"][it+1]
        @printf("%5d %+11.3f %10.3e %8.4f %8.4f | %+11.3f %+11.3f\n", it, 1e3 * (e - e_fci), v, s2, f,
                1e3 * (E - e_fci), 1e3 * (E - err - e_fci))
        flush(stdout)
    end
    outdir = joinpath(@__DIR__, "replay_results")
    mkpath(outdir)
    open(joinpath(outdir, "$(label).json"), "w") do io
        JSON.print(io, Dict("label" => label, "E_FCI" => e_fci, "s2_penalty" => λ, "prep" => prep, "rows" => rows,
                            "dbf" => Dict(k => o[k] for k in ("energies_per_grad", "accumulated_error_per_grad",
                                                               "variance_per_grad", "accumulated_var_error_per_grad"))), 1)
    end
end

main_replay(ARGS)
