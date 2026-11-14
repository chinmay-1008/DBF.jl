# Pauli-rotation circuit that prepares the singlet projection of the reference
# determinant, written to <point dir>/prep_singlet_<basis>.jld2 for
# run_dbf_pes.jl --prep singlet (which folds it into H before the flow).
#
# The state phi is built exactly on the parity sector (sv_dbf.jl's
# singlet_projection, vacuum frame).  It is then disentangled greedily: each
# step picks the largest component phi_x (x != 0) and applies R = exp(-iθG/2),
# G = X on the bits of x with Y on the lowest, at the angle that moves phi_x
# entirely into |0>.  R only mixes |0> with |x> in that pair, so the weight on
# |0> rises by exactly phi_x^2 per step.  After M steps R_M ... R_1 phi = |0>
# up to the residual, so the prep circuit is
#
#     psi_prep = P_1 P_2 ... P_M |0>,   P_j = exp(-i (-θ_j) G_j / 2)
#
# in the same newest-first convention as dbf_groundstate and replay_fc.jl, and
# 1 - |<phi|psi_prep>|^2 is the residual 1 - phi_0^2 at step M.  The circuit
# is cut at the first M with that residual below --tol.
#
# Check: an exact replay of the stored circuit reproduces the fidelity.  The
# circuit is then folded into H with PauliOperators.evolve, dropping terms below
# --fold-coeff, as a diagnostic: size of O_0 and <0|O_0|0> - <psi_prep|H|psi_prep>.
#
#     julia -t 4 --project=../../integrals build_singlet_prep.jl <point dir> <basis> [--tol 1e-6] [--fold-coeff 1e-6] [--name singlet]
# --name sets the file name, prep_<name>_<basis>.jld2, so circuits at different
# tolerances can sit side by side.  --fold-coeff none skips the folding diagnostic.
include(joinpath(@__DIR__, "sv_dbf.jl"))

# psi <- R psi for R = exp(-iθG/2): (R psi)_b = c psi_b + s mu(b xor x) psi_{b xor x}
function rotate_vec!(ψ, G, θ, states, pos)
    c, s = cos(θ / 2), sin(θ / 2)
    x = UInt64(G.x)
    hb = UInt64(1) << (63 - leading_zeros(x))
    for (j, b) in enumerate(states)
        b & hb == 0 || continue
        k = pos[(b ⊻ x)+1]
        a, bk = ψ[j], ψ[k]
        ψ[j] = c * a + s * μ(G, b ⊻ x) * bk
        ψ[k] = c * bk + s * μ(G, b) * a
    end
    return ψ
end

function main_prep(args)
    dir, basis = args[1], args[2]
    opt = Dict("tol" => "1e-6", "fold-coeff" => "1e-6", "name" => "singlet")
    for i in 3:2:length(args)
        opt[args[i][3:end]] = args[i+1]
    end
    tol = parse(Float64, opt["tol"])
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
    amask = UInt64(1) << NORB - 1
    nel = meta["n_alpha"] + meta["n_beta"]
    blk = [k for (k, b) in enumerate(states)
           if count_ones((b ⊻ ref) & amask) == meta["n_alpha"] && count_ones((b ⊻ ref) >> NORB) == meta["n_beta"]]
    φ = singlet_projection(S2s, blk, D)
    @printf("phi: <H> - FCI = %+.4f mHa, components above 1e-3: %d, above 1e-5: %d\n",
            1e3 * (dot(φ, Hs * φ) - e_fci), count(abs.(φ) .> 1e-3), count(abs.(φ) .> 1e-5))

    # greedy disentangling of phi into |0>
    ψ = copy(φ)
    gens, angs = PauliBasis{N}[], Float64[]
    report = Set([1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000])
    while 1 - ψ[1]^2 > tol
        k = 1 + argmax(abs.(view(ψ, 2:D)))
        G = generator(states[k], "low")
        θ = 2 * atan(-ψ[k], μ(G, UInt64(0)) * ψ[1])
        rotate_vec!(ψ, G, θ, states, pos)
        abs(ψ[k]) < 1e-12 || error("rotation did not clear component $k")
        push!(gens, G); push!(angs, θ)
        length(gens) in report && @printf("  %5d rotations: 1 - F = %.3e\n", length(gens), 1 - ψ[1]^2)
        length(gens) > 20000 && error("no convergence after 20000 rotations")
    end
    M = length(gens)
    pgens, pangs = gens, -angs            # P_j = R_j'

    # exact replay of the prep circuit: P_M acts first
    χ = zeros(D); χ[1] = 1
    for j in M:-1:1
        rotate_vec!(χ, pgens[j], pangs[j], states, pos)
    end
    F = dot(φ, χ)^2
    E = dot(χ, Hs * χ); S2 = dot(χ, S2s * χ)
    lk = leak(χ, states, ref, nel)
    @printf("prep circuit: %d rotations, 1 - F = %.3e, <H> - FCI = %+.4f mHa (phi %+.4f), var %.3e (phi %.3e), <S^2> = %.2e, leak %.1e\n",
            M, 1 - F, 1e3 * (E - e_fci), 1e3 * (dot(φ, Hs * φ) - e_fci), dot(Hs * χ, Hs * χ) - E^2,
            dot(Hs * φ, Hs * φ) - dot(φ, Hs * φ)^2, S2, lk)
    abs((1 - F) - (1 - ψ[1]^2)) < 1e-9 || error("replayed fidelity differs from the greedy residual")

    out = joinpath(dir, "prep_$(opt["name"])_$(basis).jld2")
    jldsave(out; generators=pgens, angles=pangs, tol=tol, fidelity=F, E_prep=E, S2_prep=S2,
            E_phi=dot(φ, Hs * φ), leak=lk, var_prep=dot(Hs * χ, Hs * χ) - E^2)
    println("wrote ", out)

    opt["fold-coeff"] == "none" && return
    fc = parse(Float64, opt["fold-coeff"])
    # fold into H, dropping terms below fc
    O = PauliSum(N, ComplexF64)
    for (p, c) in H
        O[p] = c
    end
    t0 = time()
    for j in 1:M
        O = PauliOperators.evolve(O, pgens[j], pangs[j])
        coeff_clip!(O, fc)
        (j % 50 == 0 || j == M) && (@printf("  folded %5d / %d: %d terms, <0|O|0> - FCI = %+.4f mHa (%.0f s)\n",
                                            j, M, length(O), 1e3 * (real(expectation_value(O, Ket{N}(0))) - e_fci),
                                            time() - t0); flush(stdout))
    end
    t = time() - t0
    e0 = real(expectation_value(O, Ket{N}(0)))
    @printf("folded H (cutoff %g): %d Pauli terms (H had %d), <0|O_0|0> - <psi|H|psi> = %+.4f mHa  (%.1f s)\n",
            fc, length(O), length(H), 1e3 * (e0 - E), t)
end

main_prep(ARGS)
