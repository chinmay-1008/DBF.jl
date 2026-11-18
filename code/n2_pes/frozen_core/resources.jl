# Configuration entropy and magic of the final vDBF states along the UNO + 0.05 S^2
# PES (cutoffs 1e-5 and 1e-6), and of the exact ground state, at every R.
#
#   S   = -sum_b |c_b|^2 ln |c_b|^2, the Shannon entropy of the state's coefficients
#         in the UNO determinant basis (as DBF's entropy(O) does for Pauli coefficients)
#   M_2 = -ln( sum_P <P>^4 / 2^N ), the stabilizer Renyi entropy (Leone et al. 2022);
#         0 for |0> and any other stabilizer state
#
# The final state psi = U_1 ... U_K |0> is replayed exactly from the stored rotations,
# as in replay_fc.jl, on the space V of vacuum-frame states with even weight in each
# spin block (2^14 states).  V holds every (n_alpha, n_beta) determinant and is closed
# under H, the DBF generators and orbital rotations (H's own x-string span, 2^12, is not
# closed under the pi rotation below).  For P = i^{|x&z|} Z^z X^x, <psi|P|psi> is nonzero
# only for x in V, and |<P>| = |sum_b conj(psi_{b+x}) psi_b (-1)^{z.b}| depends on z only
# through its character on V, so sum_P <P>^4 = 2^(N - dim V) sum_{x in V} sum_y |F_x(y)|^4
# with F_x the Walsh-Hadamard transform of conj(psi_{c+x}) psi_c in V's coordinates c.
# Checked against sum_P <P>^2 = 2^N.
#
# Both S and M_2 depend on the orbital basis.  N2's pi orbitals come in degenerate
# pairs, which the UNO step mixes arbitrarily (and differently at every R), so each
# state is first rewritten in orbitals where every pi pair is pure px / py:
# psi' = exp(-kappa) psi, kappa = sum_pq K_pq a+_p a_q over both spins, exp(K) = R the
# rotation within each degenerate pair (occupied and virtual pairs separately, so the
# reference determinant is unchanged).  Orbital sign flips and swaps act as Cliffords
# and leave S and M_2 alone, so only the mixing angle matters.  Checked on every state:
# its spin-summed 1-RDM must become R' gamma R.  Both bases are written.
#
#     julia -t 4 --project=../../integrals resources.jl          # writes results/resources_uno.json
include(joinpath(@__DIR__, "sv_dbf.jl"))

const NCORE_ORB = 2                       # frozen 1s orbitals ahead of the active ones in uno_coeff.npy
const PX, PY = [3, 8], [4, 9]             # STO-3G AOs per N: 1s 2s 2px 2py 2pz

# basis of V: pairs of flips within one spin block
spin_parity_basis() = gf2_basis([UInt64(1) << (off + i) | UInt64(1) << off for off in (0, NORB) for i in 1:NORB-1])

# Generator K (active x active, antisymmetric) that turns each degenerate pi pair
# into one pure-px and one pure-py orbital: phi'_p = c phi_p + s phi_q.
function pi_generator(dir, nocc)
    C = npzread(joinpath(dir, "uno_coeff.npy"))[:, NCORE_ORB+1:end]
    w = vec(sum(abs2, C[vcat(PX, PY), :]; dims=1))
    pis = findall(>(0.5), w)
    length(pis) == 4 || error("$dir: expected 4 pi orbitals, found $pis")
    K = zeros(size(C, 2), size(C, 2))
    for (p, q) in ((pis[1], pis[2]), (pis[3], pis[4]))
        (q <= nocc) == (p <= nocc) || error("$dir: pi pair ($p, $q) straddles occupied and virtual")
        u = svd(C[PY, [p, q]]).V[:, end]                 # combination with no py weight
        norm(C[PY, [p, q]] * u) < 1e-6 || error("$dir: pi pair ($p, $q) cannot be made pure px")
        φ = atan(u[2], u[1])
        K[q, p], K[p, q] = φ, -φ
    end
    K
end

# one-body operator sum_pq A_pq (a+_pα a_qα + a+_pβ a_qβ) on the sector
function onebody_matrix(A, states, pos, ref)
    I, J, V = Int[], Int[], Float64[]
    for (j, b) in enumerate(states), off in (0, NORB), q in 0:NORB-1, p in 0:NORB-1
        A[p+1, q+1] == 0 && continue
        s, o = hop(b ⊻ ref, UInt64(1) << (q + off), UInt64(1) << (p + off))
        s == 0 && continue
        k = pos[(o⊻ref)+1]
        k == 0 && error("one-body operator leaves the sector")
        push!(I, k); push!(J, j); push!(V, A[p+1, q+1] * s)
    end
    sparse(I, J, V, length(states), length(states))
end

# exp(A) ψ by its Taylor series (A sparse, ‖A‖ of order one here)
function expmv(A, ψ)
    out, term = copy(ψ), copy(ψ)
    for n in 1:200
        term = A * term / n
        out .+= term
        norm(term) < 1e-15 && return out
    end
    error("exp(A) ψ did not converge")
end

function rdm1(ψ, states, pos, ref)
    γ = zeros(NORB, NORB)
    for p in 1:NORB, q in 1:NORB
        E = zeros(NORB, NORB); E[p, q] = 1
        γ[p, q] = real(dot(ψ, onebody_matrix(E, states, pos, ref) * ψ))
    end
    γ
end

# ψ in the fixed-pi orbitals, with the 1-RDM check
function to_fixed_pi(ψ, K, Kmat, states, pos, ref)
    ψp = expmv(-Kmat, ψ)
    R = exp(K)
    γ, γp = rdm1(ψ, states, pos, ref), rdm1(ψp, states, pos, ref)
    maximum(abs.(γp - R' * γ * R)) < 1e-9 || error("orbital rotation of the state is inconsistent")
    ψp
end

const STEMS = ["1e-5" => "dbf_uno_s2pen0.05", "1e-6" => "dbf_uno_s2pen0.05_c1e-6"]

function apply_rotation!(ψ, G, θ, states, pos)
    c, s = cos(θ / 2), sin(θ / 2)
    out = c .* ψ
    for (j, b) in enumerate(states)
        a = ψ[j]
        a == 0 && continue
        ph, k = G * Ket{N}(b)
        i = pos[UInt64(k.v)+1]
        i == 0 && error("generator leaves the sector")
        out[i] += -1im * s * ph * a
    end
    ψ .= out
end

function replay(gens, angs, nstates, states, pos)
    ψ = zeros(ComplexF64, nstates)
    ψ[1] = 1
    for r in length(gens):-1:1
        apply_rotation!(ψ, gens[r], angs[r], states, pos)
    end
    ψ
end

# in-place unnormalized Walsh-Hadamard transform
function fwht!(f)
    h = 1
    while h < length(f)
        for i in 0:2h:length(f)-1, j in i+1:i+h
            a, b = f[j], f[j+h]
            f[j], f[j+h] = a + b, a - b
        end
        h *= 2
    end
    f
end

conf_entropy(ψ) = -sum(p * log(p) for p in abs2.(ψ) if p > 0)

# ψ indexed by V's coordinates c (states[c+1]), so b ⊻ x <-> c ⊻ xc
function magic_m2(ψ, dimV)
    n = length(ψ)
    chunks = collect(Iterators.partition(0:n-1, cld(n, 4nthreads())))
    s4 = zeros(length(chunks)); s2 = zeros(length(chunks))
    @threads for ci in eachindex(chunks)
        for xc in chunks[ci]
            f = [conj(ψ[(c⊻xc)+1]) * ψ[c+1] for c in 0:n-1]
            fwht!(f)
            s4[ci] += sum(abs2(v)^2 for v in f)
            s2[ci] += sum(abs2, f)
        end
    end
    mult = 2.0^(N - dimV)
    abs(mult * sum(s2) - 2.0^N) < 1e-8 * 2.0^N || error("Pauli weights do not sum to 2^N")
    -log(mult * sum(s4) / 2.0^N)
end

function main_resources()
    base = joinpath(@__DIR__, "data_unocore")
    rows = []
    @printf("%5s %8s %8s | %8s %8s %8s | %8s %8s %8s\n", "R", "S fci", "M2 fci", "S 1e-5", "M2 1e-5", "dE 1e-5",
            "S 1e-6", "M2 1e-6", "dE 1e-6")
    for d in sort(filter(x -> startswith(x, "R_"), readdir(base)))
        dir = joinpath(base, d)
        all(isfile(joinpath(dir, "$(s)_ckpt.jld2")) for (_, s) in STEMS) || continue
        meta = JSON.parsefile(joinpath(dir, "metadata.json"))
        h = load(joinpath(dir, "hamiltonian_uno.jld2"))
        ref = UInt64(h["reference_ket"].v)
        e_fci = meta["E_FCI_with_mu"] + meta["constant"]
        H = to_vacuum_reference(h["H"], ref)
        gb = spin_parity_basis()
        states = enumerate_span(gb)
        pos = zeros(Int32, 1 << N)
        for (k, b) in enumerate(states)
            pos[b+1] = k
        end
        Hs = sector_matrix(H, states, pos)
        amask = UInt64(1) << NORB - 1
        blk = [k for (k, b) in enumerate(states)
               if count_ones((b ⊻ ref) & amask) == meta["n_alpha"] && count_ones((b ⊻ ref) >> NORB) == meta["n_beta"]]
        F = eigen(Symmetric(Matrix(Hs[blk, blk])))
        abs(F.values[1] - e_fci) < 1e-6 || error("$d: sector ground state is not FCI")
        Φ0 = zeros(ComplexF64, length(states)); Φ0[blk] = F.vectors[:, 1]
        K = pi_generator(dir, meta["n_alpha"])
        Kmat = onebody_matrix(K, states, pos, ref)
        # S, M2 in the fixed-pi orbitals; S_uno, M2_uno in the UNOs as computed
        resources(ψ) = (ψp = to_fixed_pi(ψ, K, Kmat, states, pos, ref);
                        Dict("S" => conf_entropy(ψp), "M2" => magic_m2(ψp, length(gb)),
                             "S_uno" => conf_entropy(ψ), "M2_uno" => magic_m2(ψ, length(gb))))
        row = Dict{String,Any}("R" => meta["R_angstrom"], "E_FCI" => e_fci, "fci" => resources(Φ0),
                               "pi_angles" => [K[q, p] for p in 1:NORB for q in p+1:NORB if K[q, p] != 0])
        for (c, stem) in STEMS
            o = load(joinpath(dir, "$(stem)_ckpt.jld2"))["out"]
            ψ = replay(o["generators"], o["angles"], length(states), states, pos)
            e = real(dot(ψ, Hs * ψ))
            row[c] = merge(resources(ψ), Dict("E_true" => e, "fidelity" => abs2(dot(Φ0, ψ)),
                                              "rotations" => length(o["angles"])))
        end
        push!(rows, row)
        @printf("%5.2f %8.4f %8.4f | %8.4f %8.4f %+8.3f | %8.4f %8.4f %+8.3f\n", row["R"], row["fci"]["S"], row["fci"]["M2"],
                row["1e-5"]["S"], row["1e-5"]["M2"], 1e3 * (row["1e-5"]["E_true"] - e_fci),
                row["1e-6"]["S"], row["1e-6"]["M2"], 1e3 * (row["1e-6"]["E_true"] - e_fci))
        flush(stdout)
    end
    out = joinpath(@__DIR__, "results", "resources_uno.json")
    open(io -> JSON.print(io, rows, 1), out, "w")
    println(out)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main_resources()
end
