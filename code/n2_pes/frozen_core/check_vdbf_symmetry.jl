# vDBF symmetry-breaking diagnostic: largest sigma-pi (and pi_x-pi_y) element of each vDBF
# state's 1-RDM in the fixed-pi orbitals, the pi_x / pi_y occupation mismatch, and the
# electron count.  Why bond_magic.jl uses the exact state's NOs for the vDBF states.
#
#     julia --project=../../integrals check_vdbf_symmetry.jl
include(joinpath(@__DIR__, "bond_magic.jl"))
base = joinpath(@__DIR__, "data_unocore")
for d in ["R_0.80", "R_1.60", "R_2.10", "R_3.00"]
    dir = joinpath(base, d); meta = JSON.parsefile(joinpath(dir, "metadata.json"))
    h = load(joinpath(dir, "hamiltonian_uno.jld2")); ref = UInt64(h["reference_ket"].v)
    states = enumerate_span(spin_parity_basis()); pos = zeros(Int32, 1 << N); for (k, b) in enumerate(states); pos[b+1] = k; end
    K = pi_generator(dir, meta["n_alpha"]); Kmat = onebody_matrix(K, states, pos, ref)
    C = npzread(joinpath(dir, "uno_coeff.npy"))[:, NCORE_ORB+1:end] * exp(K)
    πx = findall(>(0.5), vec(sum(abs2, C[PX, :]; dims=1))); πy = findall(>(0.5), vec(sum(abs2, C[PY, :]; dims=1)))
    σ = setdiff(1:NORB, πx, πy); lab = zeros(Int, NORB); lab[σ] .= 1; lab[πx] .= 2; lab[πy] .= 3
    for (c, stem) in STEMS
        o = load(joinpath(dir, "$(stem)_ckpt.jld2"))["out"]
        ψ = to_fixed_pi(replay(o["generators"], o["angles"], length(states), states, pos), K, Kmat, states, pos, ref)
        γ = rdm1(ψ, states, pos, ref)
        off = maximum(abs(γ[p, q]) for p in 1:NORB, q in 1:NORB if lab[p] != lab[q])
        dx = maximum(abs.(sort(eigvals(Symmetric(γ[πx, πx]))) - sort(eigvals(Symmetric(γ[πy, πy])))))
        @printf("%s %s  max off-block |γ| %.2e   πx vs πy occupation diff %.2e   trace %.6f\n", d, c, off, dx, tr(γ))
    end
end
