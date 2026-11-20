# Continuous double-bracket flow with the one-body (default) or two-body generator,
#
#     dO/ds = [G, O],   G = [O, Λ],   Λ = ρ1 = I + Σ_i Z_i  or  ρ2 = ρ1 + Σ_{i<j} Z_i Z_j
#     (in the reference frame Σ Z_i |x> = (n - 2|x|)|x>, Σ Z_i Z_j |x> = (C(n,2) - 2|x|(n - |x|))|x>),
#     O(0) = H + λ S^2,
#
# the flow of the report's notebook (rk4_flow there), on the 16-qubit frozen-core
# problems, exactly (no truncation).  dE/ds = 2 Σ_k |O_0k|^2 (Λ_k - Λ_0) <= 0.
# Unlike Λ = |0><0| (continuous_flow.jl) it has no closed form on the state, so
# the whole operator is integrated.  O conserves N_alpha and N_beta and Λ is
# diagonal, so the flow stays in the (n_alpha, n_beta) block of the sector
# reachable from |0>.
#
# Integrator: exponential midpoint, O <- e^{ds G_m} O e^{-ds G_m} with G_m = G at
# the predicted half step; every step is exactly unitary (the spectrum cannot
# drift), second order in ds.  The flow is stiff (couplings to high-energy states
# decay ~1000x faster than the slowest mode), so ds is adapted by step doubling:
# a step is accepted when one full step and two half steps agree to tol (max
# element of O, Ha).  Run with two tolerances to check that the curve is the
# continuous one.  The dressing W (O = W' O(0) W) is carried along for the
# fidelity and <S^2> of psi = W|0>.
#
#     julia -t 8 --project=../../integrals cflow_onebody.jl <point dir> <basis> <label> [λ] [ds0] [s_max] [tol] [onebody|twobody]
include(joinpath(@__DIR__, "sv_dbf.jl"))
using SparseArrays

function main_flow(args)
    dir, basis, label = args[1:3]
    λ = length(args) >= 4 ? parse(Float64, args[4]) : 0.0
    ds = length(args) >= 5 ? parse(Float64, args[5]) : 0.02
    smax = length(args) >= 6 ? parse(Float64, args[6]) : 300.0
    tol = length(args) >= 7 ? parse(Float64, args[7]) : 1e-9
    gen = length(args) >= 8 ? args[8] : "onebody"
    gen in ("onebody", "twobody") || error("generator must be onebody or twobody")
    meta = JSON.parsefile(joinpath(dir, "metadata.json"))
    d = load(joinpath(dir, "hamiltonian_$(basis).jld2"))
    ref = UInt64(d["reference_ket"].v)
    e_fci = meta["E_FCI_with_mu"] + meta["constant"]
    Ovl = npzread(joinpath(dir, "$(basis)_ab_overlap.npy"))
    Ovl[abs.(Ovl) .< 1e-6] .= 0
    H = to_vacuum_reference(d["H"], ref)
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
    blk[1] == pos[1] || error("reference is not the first state of the block")
    Hb = Matrix(Hs[blk, blk]); Sb = Matrix(S2s[blk, blk])
    O = λ == 0 ? copy(Hb) : Hb + λ * Sb
    M = size(O, 1)
    # paper convention, unnormalized: ρ1 = I + Σ Z_i, ρ2 = ρ1 + Σ_{i<j} Z_i Z_j (the I drops out of the flow)
    Λ = [let k = count_ones(states[j])
             z1 = N - 2k
             gen == "onebody" ? Float64(z1) : Float64(z1 + (N * (N - 1)) ÷ 2 - 2k * (N - k))
         end for j in blk]
    Fb = eigen(Symmetric(Hb))
    abs(Fb.values[1] - e_fci) < 1e-8 || error("block ground state is not FCI")
    Φ0 = Fb.vectors[:, 1]
    spec0 = eigvals(Symmetric(O))
    G(O) = O .* (Λ' .- Λ)                        # [O, Λ]_jk = O_jk (Λ_k - Λ_j)
    W = Matrix{Float64}(I, M, M)
    rows = Dict{String,Any}[]
    rec(s) = begin
        ψ = W[:, 1]; e = dot(ψ, Hb * ψ)
        push!(rows, Dict("s" => s, "E" => e, "obj" => O[1, 1], "S2" => dot(ψ, Sb * ψ), "fidelity" => dot(Φ0, ψ)^2,
                         "var" => dot(Hb * ψ, Hb * ψ) - e^2))
    end
    @printf("R = %.2f %s, λ = %g, generator %s: block %d, initial ds = %g, s_max = %g, tol = %g\n", meta["R_angstrom"], basis, λ, gen, M, ds, smax, tol)
    rec(0.0)
    # one exponential-midpoint step and its unitary
    step(O, h) = (Eh = exp(0.5 * h * G(O)); Oh = Eh * O * Eh'; E = exp(h * G(Oh)); (E * O * E', E))
    s, t0, nacc, nrej = 0.0, time(), 0, 0
    nextrec = 1e-5
    while s < smax
        h = min(ds, smax - s)
        O1, E1 = step(O, h)                                  # one full step
        Oa, Ea = step(O, h / 2); O2, Eb = step(Oa, h / 2)    # two half steps
        err = maximum(abs, O1 - O2) / 3                      # second order: error of the half-step pair
        if err > tol
            ds = h * max(0.2, 0.9 * (tol / err)^(1 / 3)); nrej += 1
            continue
        end
        O = (O2 + O2') / 2; W = W * (Eb * Ea)'; s += h; nacc += 1
        ds = h * min(2.0, 0.9 * (tol / max(err, 1e-300))^(1 / 3))
        if s >= nextrec || s >= smax
            rec(s); nextrec = s * 10^(1 / 60)
            r = rows[end]
            @printf("s %9.4f  ds %.2e  E - FCI %.3e  obj - FCI %.3e  fidelity %.6f  steps %d (+%d rejected)  (%.0f s)\n",
                    s, ds, r["E"] - e_fci, r["obj"] - e_fci, r["fidelity"], nacc, nrej, time() - t0)
            flush(stdout)
            r["obj"] - e_fci < 1e-11 && break
        end
    end
    drift = maximum(abs, eigvals(Symmetric(O)) - spec0)
    @printf("spectrum drift %.1e\n", drift)
    open(joinpath(@__DIR__, "sv_results", "$(label).json"), "w") do io
        JSON.print(io, Dict("label" => label, "R" => meta["R_angstrom"], "basis" => basis, "s2_penalty" => λ,
                            "ds" => ds, "generator" => gen, "E_FCI" => e_fci, "block_dim" => M, "spectrum_drift" => drift, "rows" => rows), 1)
    end
end

main_flow(ARGS)
