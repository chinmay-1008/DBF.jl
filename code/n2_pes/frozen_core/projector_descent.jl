# Discrete projector flow: one rotation per gradient evaluation.
#
# With P = |0><0| the double-bracket generator in the dressed frame is
#     eta = [O, P] = |r><0| - |0><r|,   |r> = (O - E)|0>  (the residual / energy gradient),
# rank 2, so exp(θ eta) rotates |0> towards r^ and leaves everything else alone.
# In the lab frame that is psi <- cos θ psi + sin θ r^ with r = (H' - E')psi,
# H' = H + λ S^2 (the same objective as sv_dbf.jl's penalty runs), at the θ
# that minimizes <H'> on span{psi, r^} (2x2 problem): steepest descent with an
# exact line search.  Each step costs one gradient, as one sv_dbf.jl iteration
# does (which then spends it on up to 50 single-Pauli rotations).
#
# E, Var, <S^2> and fidelity are of H.  Also recorded: how many directions x
# (basis states) the step's rotation mixes, i.e. nonzero components of r.
#
#     julia --project=../../integrals projector_descent.jl <point dir> <basis> <label> [λ] [steps]
include(joinpath(@__DIR__, "sv_dbf.jl"))
using SparseArrays

function descent(args)
    dir, basis, label = args[1:3]
    λ = length(args) >= 4 ? parse(Float64, args[4]) : 0.0
    nsteps = length(args) >= 5 ? parse(Int, args[5]) : 1000
    meta = JSON.parsefile(joinpath(dir, "metadata.json"))
    d = load(joinpath(dir, "hamiltonian_$(basis).jld2"))
    ref = UInt64(d["reference_ket"].v)
    e_fci = meta["E_FCI_with_mu"] + meta["constant"]
    Ovl = npzread(joinpath(dir, "$(basis)_ab_overlap.npy"))
    Ovl[abs.(Ovl) .< 1e-6] .= 0
    H = to_vacuum_reference(d["H"], ref)
    gb = gf2_basis(unique(UInt64(p.x) for p in keys(H)))
    states = enumerate_span(gb)
    D = length(states)
    pos = zeros(Int32, 1 << N)
    for (k, b) in enumerate(states)
        pos[b+1] = k
    end
    Hs = sector_matrix(H, states, pos)
    S2s = s2_matrix(states, pos, ref, Ovl)
    Op = λ == 0 ? Hs : Hs + λ * S2s
    amask = UInt64(1) << NORB - 1
    blk = [k for (k, b) in enumerate(states)
           if count_ones((b ⊻ ref) & amask) == meta["n_alpha"] && count_ones((b ⊻ ref) >> NORB) == meta["n_beta"]]
    Fb = eigen(Symmetric(Matrix(Hs[blk, blk])))
    abs(Fb.values[1] - e_fci) < 1e-8 || error("sector ground state is not FCI")
    Φ0 = zeros(D); Φ0[blk] = Fb.vectors[:, 1]
    measure(ψ) = (h = Hs * ψ; e = dot(ψ, h); (e, dot(h, h) - e^2, dot(ψ, S2s * ψ), abs2(dot(Φ0, ψ))))

    ψ = zeros(D); ψ[pos[1]] = 1.0
    rows = Dict{String,Any}[]
    push!(rows, Dict(zip(("E", "var", "S2", "fidelity"), measure(ψ))))
    rows[end]["n_dirs"] = 0
    for k in 1:nsteps
        o = Op * ψ
        a = dot(ψ, o)
        r = o - a * ψ
        b = norm(r)
        b < 1e-12 && break
        rh = r / b
        c = dot(rh, Op * rh)
        # lowest eigenvector of [[a, b], [b, c]] = (cos θ, sin θ)
        θ = 0.5 * atan(2b, a - c)          # angle of the eigenvector pair
        v1 = (cos(θ), sin(θ)); v2 = (-sin(θ), cos(θ))
        q(v) = a * v[1]^2 + 2b * v[1] * v[2] + c * v[2]^2
        v = q(v1) < q(v2) ? v1 : v2
        ψ = v[1] * ψ + v[2] * rh
        ψ /= norm(ψ)
        row = Dict(zip(("E", "var", "S2", "fidelity"), measure(ψ)))
        row["n_dirs"] = count(x -> abs(x) > 1e-12, r)
        push!(rows, row)
    end
    hit(t) = (i = findfirst(r -> r["E"] - e_fci < t, rows); i === nothing ? -1 : i - 1)
    @printf("R = %.2f %s, λ = %g: %d steps; gradients to reach E - FCI < 1.6e-3: %d  1e-4: %d  1e-6: %d;  after 200: %.2e  final %.2e;  dirs per step ~%d\n",
            meta["R_angstrom"], basis, λ, length(rows) - 1, hit(1.6e-3), hit(1e-4), hit(1e-6),
            rows[min(201, end)]["E"] - e_fci, rows[end]["E"] - e_fci, rows[min(end, 2)]["n_dirs"])
    out = joinpath(@__DIR__, "sv_results")
    mkpath(out)
    open(joinpath(out, "$(label).json"), "w") do io
        JSON.print(io, Dict("label" => label, "R" => meta["R_angstrom"], "basis" => basis, "s2_penalty" => λ,
                            "E_FCI" => e_fci, "sector_dim" => D, "rows" => rows), 1)
    end
end

descent(ARGS)
