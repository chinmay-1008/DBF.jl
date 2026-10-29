# Physical S^2 as a Jordan-Wigner PauliSum for each frozen-core point and basis,
# written next to the Hamiltonian as s2_<basis>.jld2 (key `S2`), so
# run_dbf_pes.jl --s2-penalty λ can run DBF on H + λ S^2.
#
#     S^2 = S- S+ + Sz^2 + Sz,   Sz = 1/2 sum_p (n_{p,alpha} - n_{p,beta})
#     S+  = sum_pq O_pq a+_{p,alpha} a_{q,beta},   S- = S+'
#
# with O_pq = <phi^alpha_p | phi^beta_q> from <basis>_ab_overlap.npy (alpha and
# beta orbitals differ in all four bases).  Overlaps below 1e-6 are
# symmetry-forbidden localization noise and are dropped, as in sv_dbf.jl.
#
# Checks: S^2 is Hermitian with real coefficients, <ref|S^2|ref> equals the
# single-determinant formula, and in the vacuum frame its matrix on the sector
# reachable from |0> equals sv_dbf.jl's s2_matrix to 1e-10.
#
#     julia --project=../../integrals build_s2.jl [R_2.30 ...]
include(joinpath(@__DIR__, "..", "..", "integrals", "uhf_integrals_to_pauli.jl"))
include(joinpath(@__DIR__, "sv_dbf.jl"))

const TREES = (("data_uhfcore", ["mo", "lmo"]), ("data_nocore", ["no", "lno"]), ("data_unocore", ["uno", "luno"]))

# val * A B C D for four (two-Pauli) Jordan-Wigner operators
function add_quad!(S, val, A, B, C, D)
    for a in A, b in B, c in C, d in D
        p = a * b * c * d
        key = PauliBasis(p)
        S[key] = get(S, key, zero(ComplexF64)) + val * coeff(p)
    end
end

function s2_paulisum(O, n)
    N = 2n
    dag = [jw_operator(N, i, dagger=true) for i in 1:N]
    col = [jw_operator(N, i, dagger=false) for i in 1:N]
    S = PauliSum(N, ComplexF64)
    α(p) = p            # 1-based qubit of alpha orbital p (1-based)
    β(p) = n + p
    # S- S+ = sum O_rs O_pq a+_{s,beta} a_{r,alpha} a+_{p,alpha} a_{q,beta}
    for p in 1:n, q in 1:n, r in 1:n, s in 1:n
        v = O[p, q] * O[r, s]
        v == 0 && continue
        add_quad!(S, v, dag[β(s)], col[α(r)], dag[α(p)], col[β(q)])
    end
    # Sz^2 + Sz with Sz = 1/2 sum (n_alpha - n_beta)
    sz = [(α(p), 0.5) for p in 1:n] ∪ [(β(p), -0.5) for p in 1:n]
    for (i, a) in sz, (j, b) in sz
        add_quad!(S, a * b, dag[i], col[i], dag[j], col[j])
    end
    for (i, a) in sz
        for x in dag[i], y in col[i]
            pr = x * y
            key = PauliBasis(pr)
            S[key] = get(S, key, zero(ComplexF64)) + a * coeff(pr)
        end
    end
    coeff_clip!(S, 1e-14)
    return S
end

function check_against_sv(S2, H, ref, O)
    Hx = to_vacuum_reference(Dict(H), ref)
    Sx = to_vacuum_reference(Dict(S2), ref)
    gb = gf2_basis(unique(UInt64(p.x) for p in keys(Hx)))
    states = enumerate_span(gb)
    pos = zeros(Int32, 1 << N)
    for (k, b) in enumerate(states)
        pos[b+1] = k
    end
    A = sector_matrix(Sx, states, pos)
    B = s2_matrix(states, pos, ref, O)
    return norm(A - B), A[1, 1]
end

dirs = isempty(ARGS) ? nothing : Set(ARGS)
for (tree, bases) in TREES
    root = joinpath(@__DIR__, tree)
    isdir(root) || continue
    for d in sort(filter(startswith("R_"), readdir(root))), b in bases
        dirs === nothing || d in dirs || continue
        dir = joinpath(root, d)
        isfile(joinpath(dir, "hamiltonian_$(b).jld2")) || continue   # basis not built at this point
        O = npzread(joinpath(dir, "$(b)_ab_overlap.npy"))
        O[abs.(O) .< 1e-6] .= 0
        h = load(joinpath(dir, "hamiltonian_$(b).jld2"))
        n = h["n_orbitals"]
        S2 = s2_paulisum(O, n)
        maximum(abs ∘ imag, values(S2)) < 1e-12 || error("$d $b: S^2 has complex coefficients")
        ref = h["reference_ket"]
        e_ref = real(expectation_value(S2, ref))
        occ_a = [p for p in 1:n if (ref.v >> (p - 1)) & 1 == 1]
        occ_b = [p for p in 1:n if (ref.v >> (n + p - 1)) & 1 == 1]
        sz = 0.5 * (length(occ_a) - length(occ_b))
        # <S- S+> = sum over occupied beta q, empty alpha p of O_pq^2 (the active
        # alpha orbitals do not span the active beta ones exactly, so this is
        # not N_beta - sum_occ |O|^2)
        virt_a = setdiff(1:n, occ_a)
        e_det = sz^2 + sz + sum(abs2, O[virt_a, occ_b])
        abs(e_ref - e_det) < 1e-10 || error("$d $b: <ref|S^2|ref> $e_ref != determinant formula $e_det")
        diff, s00 = check_against_sv(S2, h["H"], UInt64(ref.v), O)
        diff < 1e-10 || error("$d $b: S^2 sector matrix differs from sv_dbf.jl by $diff")
        S2r = PauliSum(2n, Float64)
        for (p, c) in S2
            S2r[p] = real(c)
        end
        jldsave(joinpath(dir, "s2_$(b).jld2"); S2=S2r)
        @printf("%s %-3s  %4d Pauli terms   <ref|S^2|ref> = %.6f (determinant formula %.6f)   |S2 - sv S2|_sector = %.1e\n",
                d, b, length(S2r), e_ref, e_det, diff)
    end
end
