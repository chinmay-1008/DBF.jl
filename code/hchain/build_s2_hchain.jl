# Physical S^2 as a Jordan-Wigner PauliSum for an H-chain point, written as
# s2_uno.jld2 next to hamiltonian_uno.jld2, so run_dbf_pes.jl --s2-penalty can
# use it.  Same construction as n2_pes/frozen_core/build_s2.jl (s2_paulisum),
# with the identity alpha-beta overlap of the restricted UNOs.
#
# Checks: real coefficients; <ref|S^2|ref> = 0 for the closed-shell reference;
# S^2 commutes with H (max coefficient of [H, S^2] below 1e-7: the converter
# drops H's terms below 1e-8, which breaks the commutator at that level).
#
#     julia --project=../integrals build_s2_hchain.jl data/H8/R_2.00 [...]
include(joinpath(@__DIR__, "..", "n2_pes", "frozen_core", "build_s2.jl"))

for dir in ARGS
    h = load(joinpath(dir, "hamiltonian_uno.jld2"))
    n = h["n_orbitals"]
    O = npzread(joinpath(dir, "uno_ab_overlap.npy"))
    maximum(abs, O - I) < 1e-12 || error("$dir: alpha-beta overlap is not the identity")
    S2 = s2_paulisum(O, n)
    maximum(abs ∘ imag, values(S2)) < 1e-12 || error("$dir: S^2 has complex coefficients")
    e_ref = real(expectation_value(S2, h["reference_ket"]))
    abs(e_ref) < 1e-10 || error("$dir: <ref|S^2|ref> = $e_ref, not 0")
    C = h["H"] * S2 - S2 * h["H"]
    comm = isempty(C) ? 0.0 : maximum(abs, values(C))
    comm < 1e-7 || error("$dir: [H, S^2] has a coefficient of $comm")
    S2r = PauliSum(2n, Float64)
    for (p, c) in S2
        S2r[p] = real(c)
    end
    jldsave(joinpath(dir, "s2_uno.jld2"); S2=S2r)
    @printf("%s  S^2: %d Pauli terms, <ref|S^2|ref> = %.1e, max |[H, S^2]| = %.1e\n", dir, length(S2r), e_ref, comm)
end
