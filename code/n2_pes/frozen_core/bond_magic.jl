# Does magic track the breaking of the individual bonds of N2?  For the exact ground
# state at every R, and for the final vDBF states (cutoffs 1e-5 and 1e-6, replayed
# exactly from their stored rotations as in resources.jl): the natural occupations of
# the sigma and pi bonding / antibonding pairs (when each bond breaks), and S and M_2
# in the exact state's natural orbitals, the basis in which a broken bond is a
# two-determinant pair.
#
# The vDBF states break the symmetry slightly (sigma-pi elements of their 1-RDM up to
# 1e-2 at 1e-5 and 2e-4 at 1e-6, pi_x and pi_y occupations apart by up to 0.014, and
# the electron count off by up to 4e-4, since single Pauli rotations do not conserve
# it), so their own natural orbitals would mix pi_x and pi_y again.  They are written
# in the exact state's NOs instead, one basis for all three states at each R; their
# occupations are the eigenvalues of their own sigma, pi_x and pi_y blocks.
#
# Group magic: the 16 qubits split into the sigma orbitals (8 qubits) and the pi_x + pi_y
# orbitals (8 qubits), both spins.  For each group A, the reduced state rho_A (256 x 256,
# traced over the other group) gives the mixed-state stabilizer Renyi entropy
# M_2(rho_A) = -ln( sum_{P in A} tr(P rho_A)^4 / sum_{P in A} tr(P rho_A)^2 ) (Leone et al.
# 2022): zero for a stabilizer or maximally mixed group, and equal to the pair magic of
# its bonds if the groups were independent.  The purity tr(rho_A^2) is kept with it.
#
# Natural orbitals are taken per symmetry block of the fixed-pi orbitals (sigma, pi_x,
# pi_y; resources.jl), so the degenerate pi NOs stay pure px / py, and the state is
# rotated into them as in to_fixed_pi.  Written to results/bond_magic_uno.json.
#
#     julia -t 4 --project=../../integrals bond_magic.jl
include(joinpath(@__DIR__, "resources.jl"))

# rotation matrix whose columns are the NOs of each block, close to the identity
# (each NO in the slot of the orbital it overlaps most, sign so that overlap is positive)
function block_nos(γ, blocks)
    U = zeros(NORB, NORB)
    occ = zeros(NORB)
    for b in blocks
        F = eigen(Symmetric(γ[b, b]))
        V = F.vectors
        order = zeros(Int, length(b))
        for j in sortperm(F.values; rev=true)          # greedy slot assignment
            free = [i for i in eachindex(b) if !(i in order)]
            order[j] = free[argmax(abs.(V[free, j]))]
        end
        for (j, i) in enumerate(order)
            U[b, b[i]] = V[:, j] * sign(V[i, j])
            occ[b[i]] = F.values[j]
        end
    end
    U, occ
end

# M_2 and purity of the reduced state of qubits qs (0-based)
function group_magic(ψ, states, qs)
    d = 1 << length(qs)
    amask = reduce(|, UInt64(1) << q for q in qs)
    abits(b) = sum(((b >> q) & 1) << (i - 1) for (i, q) in enumerate(qs))
    rest = Dict{UInt64,Vector{Tuple{Int,ComplexF64}}}()
    for (k, b) in enumerate(states)
        ψ[k] == 0 && continue
        push!(get!(rest, b & ~amask, Tuple{Int,ComplexF64}[]), (Int(abits(b)), ψ[k]))
    end
    ρ = zeros(ComplexF64, d, d)
    for v in values(rest), (a, x) in v, (a2, y) in v
        ρ[a+1, a2+1] += x * conj(y)
    end
    s4, s2 = 0.0, 0.0
    for xm in 0:d-1
        f = [ρ[(a⊻xm)+1, a+1] for a in 0:d-1]       # |tr(P rho)| for P = Z^z X^x is |WHT(f)(z)|
        fwht!(f)
        s4 += sum(abs2(v)^2 for v in f)
        s2 += sum(abs2, f)
    end
    purity = real(tr(ρ * ρ))
    abs(s2 - d * purity) < 1e-8 * d || error("group Pauli weights do not match the purity")
    -log(s4 / s2), purity
end

function main_bond()
    base = joinpath(@__DIR__, "data_unocore")
    rows = []
    @printf("%5s %8s %8s %8s %8s %8s | %8s %8s | %8s %8s\n", "R", "n σ*", "n π*", "M2", "M2 σ", "M2 π",
            "M2σ 1e-5", "M2π 1e-5", "M2σ 1e-6", "M2π 1e-6")
    for d in sort(filter(x -> startswith(x, "R_"), readdir(base)))
        dir = joinpath(base, d)
        meta = JSON.parsefile(joinpath(dir, "metadata.json"))
        h = load(joinpath(dir, "hamiltonian_uno.jld2"))
        ref = UInt64(h["reference_ket"].v)
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
        Φ0 = zeros(ComplexF64, length(states)); Φ0[blk] = F.vectors[:, 1]

        K = pi_generator(dir, meta["n_alpha"])
        Kmat = onebody_matrix(K, states, pos, ref)
        # symmetry blocks in the fixed-pi orbitals
        C = npzread(joinpath(dir, "uno_coeff.npy"))[:, NCORE_ORB+1:end] * exp(K)
        px = vec(sum(abs2, C[PX, :]; dims=1)); py = vec(sum(abs2, C[PY, :]; dims=1))
        πx, πy = findall(>(0.5), px), findall(>(0.5), py)
        σ = setdiff(1:NORB, πx, πy)
        (length(πx), length(πy), length(σ)) == (2, 2, 4) || error("$d: unexpected orbital symmetries")

        blocks = [σ, πx, πy]
        occs(Φ) = (γ = rdm1(Φ, states, pos, ref);
                   Dict("occ_sigma" => sort(eigvals(Symmetric(γ[σ, σ]))),     # 3σu*, 3σg, 2σu, 2σg from the bottom
                        "occ_pix" => sort(eigvals(Symmetric(γ[πx, πx]))),     # 1πg*, 1πu
                        "occ_piy" => sort(eigvals(Symmetric(γ[πy, πy])))))

        # the exact state's NOs, and the rotation into them
        Φ = to_fixed_pi(Φ0, K, Kmat, states, pos, ref)
        U, occ = block_nos(rdm1(Φ, states, pos, ref), blocks)
        K2 = real(log(U))
        maximum(abs.(K2 + K2')) < 1e-8 || error("$d: NO rotation has no real antisymmetric log")
        K2mat = onebody_matrix(K2, states, pos, ref)
        ψ = to_fixed_pi(Φ, K2, K2mat, states, pos, ref)          # same check, NO rotation
        maximum(abs.(rdm1(ψ, states, pos, ref) - Diagonal(occ))) < 1e-8 || error("$d: state is not in its natural orbitals")
        qubits(orbs) = vcat([o - 1 for o in orbs], [o - 1 + NORB for o in orbs])
        function in_nos(ψ)
            mσ, pσ = group_magic(ψ, states, qubits(σ))
            mπ, pπ = group_magic(ψ, states, qubits(vcat(πx, πy)))
            Dict("S_NO" => conf_entropy(ψ), "M2_NO" => magic_m2(ψ, length(gb)),
                 "M2_sigma" => mσ, "purity_sigma" => pσ, "M2_pi" => mπ, "purity_pi" => pπ)
        end

        row = merge(Dict{String,Any}("R" => meta["R_angstrom"]), occs(Φ), in_nos(ψ))   # exact state at the top level
        for (c, stem) in STEMS
            o = load(joinpath(dir, "$(stem)_ckpt.jld2"))["out"]
            Φv = to_fixed_pi(replay(o["generators"], o["angles"], length(states), states, pos), K, Kmat, states, pos, ref)
            row[c] = merge(occs(Φv), in_nos(to_fixed_pi(Φv, K2, K2mat, states, pos, ref)))
        end
        push!(rows, row)
        @printf("%5.2f %8.4f %8.4f %8.4f %8.4f %8.4f | %8.4f %8.4f | %8.4f %8.4f\n", row["R"], row["occ_sigma"][1],
                row["occ_pix"][1], row["M2_NO"], row["M2_sigma"], row["M2_pi"], row["1e-5"]["M2_sigma"], row["1e-5"]["M2_pi"],
                row["1e-6"]["M2_sigma"], row["1e-6"]["M2_pi"])
        flush(stdout)
    end
    out = joinpath(@__DIR__, "results", "bond_magic_uno.json")
    open(io -> JSON.print(io, rows, 1), out, "w")
    println(out)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main_bond()
end
