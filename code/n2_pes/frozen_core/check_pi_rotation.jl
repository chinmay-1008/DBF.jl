# Independent check of to_fixed_pi: build H' from pi-rotated integrals with the
# converter's own Jordan-Wigner code; the rotated FCI and vDBF states must give the
# same energies with H' as the originals with H, and the rotated FCI state must be
# H''s ground state.
const FC = @__DIR__
include(joinpath(FC, "resources.jl"))
include(joinpath(FC, "..", "..", "integrals", "uhf_integrals_to_pauli.jl"))

dir = joinpath(FC, "data_unocore", get(ARGS, 1, "R_2.10"))
meta = JSON.parsefile(joinpath(dir, "metadata.json"))
h = load(joinpath(dir, "hamiltonian_uno.jld2"))
ref = UInt64(h["reference_ket"].v)
H = to_vacuum_reference(h["H"], ref)
gb = spin_parity_basis()
states = enumerate_span(gb)
pos = zeros(Int32, 1 << N); for (k, b) in enumerate(states); pos[b+1] = k; end
Hs = sector_matrix(H, states, pos)
amask = UInt64(1) << NORB - 1
blk = [k for (k, b) in enumerate(states)
       if count_ones((b ⊻ ref) & amask) == meta["n_alpha"] && count_ones((b ⊻ ref) >> NORB) == meta["n_beta"]]
F = eigen(Symmetric(Matrix(Hs[blk, blk])))
Φ0 = zeros(ComplexF64, length(states)); Φ0[blk] = F.vectors[:, 1]

K = pi_generator(dir, meta["n_alpha"]); R = exp(K)
println("pi angles: ", [K[q, p] for p in 1:NORB for q in p+1:NORB if K[q, p] != 0])
rot1(A) = R' * A * R
function rot2(g)
    n = size(g, 1); out = copy(g)
    for ax in 1:4
        perm = [ax; setdiff(1:4, ax)]
        t = reshape(permutedims(out, perm), n, :)
        out = permutedims(reshape(R' * t, n, n, n, n), invperm(perm))
    end
    out
end
ld(k) = npzread(joinpath(dir, "uno_$k.npy"))
Hp = uhf_hamiltonian(rot1(ld("h1_a")), rot1(ld("h1_b")), rot2(ld("eri_aa")), rot2(ld("eri_bb")), rot2(ld("eri_ab")))
sum!(Hp, Pauli(N) * Float64(meta["constant"]))
Hps = sector_matrix(to_vacuum_reference(Dict(Hp), ref), states, pos)

Kmat = onebody_matrix(K, states, pos, ref)
Φp = to_fixed_pi(Φ0, K, Kmat, states, pos, ref)
Fp = eigen(Symmetric(Matrix(Hps[blk, blk])))
gp = zeros(ComplexF64, length(states)); gp[blk] = Fp.vectors[:, 1]
@printf("E_FCI %.10f   lowest of H' %.10f   <Φ'|H'|Φ'> %.10f   |<Φ'|ground H'>|^2 %.12f\n",
        F.values[1], Fp.values[1], real(dot(Φp, Hps * Φp)), abs2(dot(gp, Φp)))
for stem in ("dbf_uno_s2pen0.05", "dbf_uno_s2pen0.05_c1e-6")
    o = load(joinpath(dir, "$(stem)_ckpt.jld2"))["out"]
    ψ = replay(o["generators"], o["angles"], length(states), states, pos)
    ψp = to_fixed_pi(ψ, K, Kmat, states, pos, ref)
    @printf("%-26s <ψ|H|ψ> %.10f   <ψ'|H'|ψ'> %.10f\n", stem, real(dot(ψ, Hs * ψ)), real(dot(ψp, Hps * ψp)))
end
