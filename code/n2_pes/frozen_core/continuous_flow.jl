# Continuous double-bracket flow with the full projector generator,
#
#     dO/ds = [eta, O],   eta = [O, P],   P = |0><0|,   O(0) = H + λ S^2,
#
# on the 16-qubit frozen-core problems, with nothing truncated.  Writing
# O(s) = W(s)' H W(s), dW/ds = -W eta, the state psi(s) = W(s)|0> obeys
#
#     dpsi/ds = -(O' - E'(s)) psi   in the lab frame, with O' = H + λ S^2 and
#                                   E'(s) = <psi|O'|psi>,
#
# because eta|0> = O|0> - E'|0> and W O = O' W.  That is normalized imaginary-
# time evolution, so the flow is integrated exactly from the eigenvectors of O'
# on the sector reachable from |0>:
#
#     psi(s) = sum_k c_k exp(-s (e_k - e_0)) v_k / norm,   c_k = <v_k|0>.
#
# No step size enters; s is sampled on a log grid.  As a check that the
# closed form is the flow, the ODE is also integrated with RK4 (ds = DS) up to
# s = S_CHECK and compared.  E, Var, <S^2> and the fidelity with FCI are of H.
#
#     julia --project=../../integrals continuous_flow.jl <point dir> <basis> <label> [λ]
include(joinpath(@__DIR__, "sv_dbf.jl"))
using SparseArrays

const DS, S_CHECK = 0.01, 20.0

function flow(args)
    dir, basis, label = args[1:3]
    λ = length(args) >= 4 ? parse(Float64, args[4]) : 0.0
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

    F = eigen(Symmetric(Matrix(Op)))
    ψ0 = zeros(D); ψ0[pos[1]] = 1.0                      # |0>, the reference
    c = F.vectors' * ψ0
    state(s) = (a = c .* exp.(-s .* (F.values .- F.values[1])); v = F.vectors * a; v / norm(v))
    measure(ψ) = (h = Hs * ψ; e = dot(ψ, h); (e, dot(h, h) - e^2, dot(ψ, S2s * ψ), abs2(dot(Φ0, ψ))))

    # RK4 on dpsi/ds = -(O' - E') psi, against the closed form
    rhs(ψ) = (o = Op * ψ; -(o - dot(ψ, o) * ψ))
    ψ = copy(ψ0); dev = 0.0
    for k in 1:round(Int, S_CHECK / DS)
        k1 = rhs(ψ); k2 = rhs(ψ + DS / 2 * k1); k3 = rhs(ψ + DS / 2 * k2); k4 = rhs(ψ + DS * k3)
        ψ += DS / 6 * (k1 + 2k2 + 2k3 + k4)
        k % 100 == 0 && (dev = max(dev, norm(ψ - state(k * DS))))
    end

    svals = vcat(0.0, 10 .^ range(-2, 5, length=351))
    rows = Dict{String,Any}[]
    @printf("R = %.2f %s, λ = %g: sector %d, overlap of |0> with the ground state %.4f, RK4 (ds %g, s <= %g) vs closed form %.1e\n",
            meta["R_angstrom"], basis, λ, D, abs2(c[1]), DS, S_CHECK, dev)
    for s in svals
        e, v, s2, f = measure(state(s))
        push!(rows, Dict("s" => s, "E" => e, "var" => v, "S2" => s2, "fidelity" => f))
    end
    # first s where E - FCI drops below a few targets (Ha)
    hit(t) = (i = findfirst(r -> r["E"] - e_fci < t, rows); i === nothing ? NaN : rows[i]["s"])
    @printf("   s to reach E - FCI < 1.6e-3: %.3g   1e-4: %.3g   1e-6: %.3g   final E - FCI %.2e\n",
            hit(1.6e-3), hit(1e-4), hit(1e-6), rows[end]["E"] - e_fci)
    out = joinpath(@__DIR__, "sv_results")
    mkpath(out)
    open(joinpath(out, "$(label).json"), "w") do io
        JSON.print(io, Dict("label" => label, "R" => meta["R_angstrom"], "basis" => basis, "s2_penalty" => λ,
                            "E_FCI" => e_fci, "sector_dim" => D, "ref_overlap" => abs2(c[1]),
                            "rk4_check" => Dict("ds" => DS, "s_max" => S_CHECK, "max_dev" => dev),
                            "eigs_above_ground_mHa" => [(x - F.values[1]) * 1e3 for x in F.values[2:min(D, 11)]],
                            "rows" => rows), 1)
    end
end

flow(ARGS)
