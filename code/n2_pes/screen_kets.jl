# Screen for the block states of sv_block.jl / run_block_pp.jl with a cheap, loose
# full-diagonalising flow, before the real block run.
#
# The block flow can only bring a block state onto a level in the same symmetry
# sector, so the kets must be chosen for the levels wanted (code/vdbf_excited_states_runs.md,
# section 11).  Here:
#
#  1. A full-diagonalising flow (dbf_diag's step from src/diagonalization.jl: generator
#     = the largest term of [diag O, O], angle maximising |diag O|) runs on
#     H (or H + λ S^2) + κ [(N_α - n_α)^2 + (N_β - n_β)^2] with a loose coefficient cutoff
#     (--coeff, e.g. 1e-3).  It pushes all of H toward diagonal, not only a block.  The
#     number penalty (--n-penalty κ, Ha) changes no level with n_α, n_β electrons; it
#     keeps the flow from swapping them with other electron numbers (single Pauli
#     rotations do not conserve the electron number).  dbf_diag itself is not called: it
#     stops when truncation makes a generator repeat; here up to --per-iter generators
#     are applied per iteration and the loop runs --max-iter iterations or until the
#     largest term of [diag O, O] is below --grad.  dbf_diag's angle lies in (0, π); θ and
#     θ - π give the same |diag O|, but θ > π/2 also swaps the determinants each rotation
#     pairs, so here the angle is taken in (-π/2, π/2]: every determinant keeps its label.
#  2. Every determinant y with n_alpha, n_beta electrons gets its dressed energy
#     <y|O|y>, O the rotated operator: the Z-only terms of O evaluated on y.  It is a
#     rough energy of the eigenstate the flow maps onto y.
#  3. The determinants are sorted by dressed energy.  The flow never leaves a sector
#     (its rotations are products of H's Pauli strings, and truncation only drops
#     terms), so a determinant near a low level is in that level's sector, however
#     loose the cutoff.
#  4. Suggested kets: the --n-kets lowest, skipping the alpha<->beta mirror of a ket
#     already picked (such a pair spans one singlet and one triplet of the same
#     orbitals: the spin trap at 1.1 A).
#
# The plain diagonal <y|H|y> is printed as well: sorting by it is the no-flow baseline.
#
# Result on N2 (code/vdbf_excited_states_runs.md, section 11, "Can a loose vDBF pick the kets?"):
# it does not work.  After 25,000-50,000 rotations at 1e-3 or 1e-4 the dressed energies are still
# 30-250 mHa off, the sectors picked change with the cutoff and the penalty, and at 2.3 A they
# are never right.
# Energies in mHa above E_FCI.  Kets are written as for sv_block.jl --kets: "ref", or
# the qubits flipped from the reference (vacuum frame) joined by dots.
#
#     julia --project=. screen_kets.jl <point dir> <basis>
#         [--coeff 1e-3] [--max-iter 1000] [--per-iter 50] [--grad 1e-6] [--n-top 100]
#         [--n-penalty 1.0] [--s2-penalty 0] [--n-kets 3] [--show 12] [--out file.json]
#     e.g. julia --project=. screen_kets.jl frozen_core/data_unocore/R_2.30 uno --coeff 1e-4 --n-penalty 0
using DBF, PauliOperators, JLD2, JSON, Printf, LinearAlgebra

const N_Q = 16
const NORB = 8

function parse_screen_cli(args)
    opts = Dict("coeff" => "1e-3", "max-iter" => "1000", "per-iter" => "50", "grad" => "1e-6",
                "n-top" => "100", "n-penalty" => "1.0", "s2-penalty" => "0",
                "n-kets" => "3", "show" => "12", "out" => "")
    pos = String[]
    i = 1
    while i <= length(args)
        if startswith(args[i], "--")
            k = args[i][3:end]
            haskey(opts, k) || error("unknown option --$k")
            opts[k] = args[i+1]
            i += 2
        else
            push!(pos, args[i])
            i += 1
        end
    end
    length(pos) == 2 || error("usage: screen_kets.jl <point dir> <basis> [options]")
    return pos..., opts
end

# as in run_block_pp.jl: conjugate by X on the occupied qubits so the reference is |0...0>
function to_vacuum_reference(H::PauliSum{N,T}, ψ::Ket{N}) where {N,T}
    Hx = PauliSum(N, T)
    for (p, c) in H
        Hx[p] = isodd(count_ones(p.z & ψ.v)) ? -c : c
    end
    return Hx
end

# GF(2) span of the x strings (as in frozen_core/sv_dbf.jl) and a vector's coset label
function gf2_basis(xs)
    basis = UInt64[]
    for x in xs
        v = x
        for b in basis
            v = min(v, v ⊻ b)
        end
        v == 0 && continue
        push!(basis, v)
        sort!(basis; rev=true)
    end
    return basis
end
reduce_span(v, basis) = foldl((u, b) -> min(u, u ⊻ b), basis; init=v)

# <y|O|y> for every y from the Z-only terms of O (vacuum frame: Z_q|y> = (-1)^{y_q}|y>)
function diagonal_energies(O, ys)
    zs = [(UInt64(p.z), real(c)) for (p, c) in O if p.x == 0]
    return [sum(c * (isodd(count_ones(z & y)) ? -1 : 1) for (z, c) in zs) for y in ys]
end

# κ [(N_α - na)^2 + (N_β - nb)^2] in the vacuum frame: n_q = (1 - s_q Z_q)/2 with s_q = -1 on the
# qubits occupied in the reference, so N_σ - n = c0 - (1/2) Σ s_q Z_q with c0 = NORB/2 - n
function number_penalty(ref, na, nb, κ)
    P = PauliSum(N_Q, ComplexF64)
    add!(z, c) = (k = PauliBasis{N_Q}(Int128(z), Int128(0)); P[k] = get(P, k, 0.0im) + c)
    s(q) = (ref >> q) & 1 == 1 ? -1.0 : 1.0
    for (off, n) in ((0, na), (NORB, nb))
        c0 = NORB / 2 - n
        add!(0, κ * (c0^2 + NORB / 4))
        for q in off:off+NORB-1
            add!(UInt64(1) << q, -κ * c0 * s(q))
            for r in q+1:off+NORB-1
                add!((UInt64(1) << q) | (UInt64(1) << r), κ * s(q) * s(r) / 2)
            end
        end
    end
    return P
end

# |diag(U O U')|^2 for U = exp(-i θ G/2) as a function of θ, from the same six inner products as
# DBF.optimize_theta_diagonalization, maximised on a grid over (-π/2, π/2] and refined by golden
# section (that routine's local search on (0, π) misses the maximum once the number penalty makes
# the cost multimodal)
function best_angle(O, G)
    ip(a, b) = PauliOperators.inner_product(a, b)
    Od, GOGd, comd = diag(O), DBF.diag_GOG(G, O), DBF.diag_commutator(G, O)
    A, B, C, D, E, F = ip(Od, Od), ip(GOGd, GOGd), ip(comd, comd), ip(Od, GOGd), ip(Od, comd), ip(GOGd, comd)
    cost(θ) = (c = cos(θ / 2); s = sin(θ / 2);
               real(c^4 * A + s^4 * B + s^2 * c^2 * (C + 2D) + 2im * c^3 * s * E + 2im * c * s^3 * F))
    θs = range(-π / 2, π / 2; length=361)
    k = argmax(cost.(θs))
    a, b = θs[max(k - 1, 1)], θs[min(k + 1, end)]
    g = (sqrt(5) - 1) / 2
    for _ in 1:60
        x1, x2 = b - g * (b - a), a + g * (b - a)
        cost(x1) > cost(x2) ? (b = x2) : (a = x1)
    end
    θ = (a + b) / 2
    return cost(θ) > cost(0.0) ? θ : 0.0
end

# dbf_diag's iteration (src/diagonalization.jl) with several generators per iteration, no stop on a
# repeated generator, and angles in (-π/2, π/2]
# The generators are ranked with diag O - P (P the number penalty): with P in, the largest terms of
# [diag O, O] are the ones that change the electron number, whose best angle is ~0.
function diag_flow(O, P, coeff, max_iter, per_iter, gthresh, n_top)
    nrot, gmax = 0, Inf
    for it in 1:max_iter
        SO_OS = DBF.max_of_commutator2(isempty(P) ? diag(O) : diag(O) - P, O; n_top=n_top)
        isempty(SO_OS) && break
        top = sort!(collect(SO_OS); by=kv -> -abs(kv[2]))
        gmax = abs(top[1][2])
        gmax < gthresh && break
        for (G, _) in top[1:min(per_iter, end)]
            θ = best_angle(O, G)
            abs(θ) < 1e-10 && continue
            O = PauliOperators.evolve(O, G, θ)
            truncate!(O, CoeffTruncation(coeff))
            nrot += 1
        end
    end
    return O, nrot, gmax
end

ket_label(y) = y == 0 ? "ref" : join([q for q in 0:N_Q-1 if (y >> q) & 1 == 1], ".")
mirror(y) = ((y & ((UInt64(1) << NORB) - 1)) << NORB) | (y >> NORB)

# "alpha 4->5" style description, from the reference occupation
function describe(y, ref)
    y == 0 && return "reference"
    parts = String[]
    for (s, off) in (("α", 0), ("β", NORB))
        holes = [q - off for q in off:off+NORB-1 if (y >> q) & 1 == 1 && (ref >> q) & 1 == 1]
        parts_ = [q - off for q in off:off+NORB-1 if (y >> q) & 1 == 1 && (ref >> q) & 1 == 0]
        isempty(holes) || push!(parts, "$s $(join(holes, ","))→$(join(parts_, ","))")
    end
    return join(parts, ", ")
end

function screen_main(args)
    dir, basis, opts = parse_screen_cli(args)
    coeff = parse(Float64, opts["coeff"])
    max_iter, n_top = parse(Int, opts["max-iter"]), parse(Int, opts["n-top"])
    λ = parse(Float64, opts["s2-penalty"])
    nk, nshow = parse(Int, opts["n-kets"]), parse(Int, opts["show"])

    meta = JSON.parsefile(joinpath(dir, "metadata.json"))
    na, nb = meta["n_alpha"], meta["n_beta"]
    e_fci = meta["E_FCI_with_mu"] + meta["constant"]
    data = load(joinpath(dir, "hamiltonian_$(basis).jld2"))
    ψ0 = data["reference_ket"]
    ref = UInt64(ψ0.v)
    H = to_vacuum_reference(data["H"], ψ0)
    Hopt = H
    if λ != 0
        S2 = to_vacuum_reference(load(joinpath(dir, "s2_$(basis).jld2"), "S2"), ψ0)
        Hopt = H + λ * S2
    end
    V0 = gf2_basis(unique(UInt64(p.x) for (p, _) in H))

    # the determinants with na alpha and nb beta electrons, in the vacuum frame
    amask = (UInt64(1) << NORB) - 1
    ys = [o ⊻ ref for o in UInt64(0):UInt64((1 << N_Q) - 1)
          if count_ones(o & amask) == na && count_ones(o >> NORB) == nb]

    κ = parse(Float64, opts["n-penalty"])
    P = PauliSum(N_Q, ComplexF64)
    if κ != 0
        P = number_penalty(ref, na, nb, κ)
        pen = diagonal_energies(P, UInt64(0):UInt64((1 << N_Q) - 1))
        all(abs(pen[y+1]) < 1e-12 for y in ys) && minimum(pen) > -1e-12 || error("number penalty is wrong")
        Hopt = Hopt + P
    end
    per_iter, gthresh = parse(Int, opts["per-iter"]), parse(Float64, opts["grad"])
    t = @elapsed O, nrot, gmax = diag_flow(Hopt, P, coeff, max_iter, per_iter, gthresh, n_top)
    e0 = diagonal_energies(Hopt, ys)
    e1 = diagonal_energies(O, ys)
    offd = sqrt(sum(abs2(c) for (p, c) in O if p.x != 0; init=0.0))
    @printf("R = %.2f %s, flow on H + %.3g S^2 + %.3g (N-n)^2, cutoff %.0e: %d rotations (%.1f s), %d terms, off-diagonal norm %.3e, largest [diag O, O] term %.2e\n",
            meta["R_angstrom"], basis, λ, κ, coeff, nrot, t, length(O), offd, gmax)
    @printf("H has %d terms, %d of them at or above the cutoff\n", length(Hopt), count(c -> abs(c) >= coeff, values(Hopt)))
    k0 = findfirst(==(UInt64(0)), ys)
    @printf("%d determinants with (%d,%d) electrons; energies in mHa above E_FCI; reference: plain %.3f, dressed %.3f\n",
            length(ys), na, nb, 1e3 * (diagonal_energies(Hopt, [UInt64(0)])[1] - e_fci), 1e3 * (diagonal_energies(O, [UInt64(0)])[1] - e_fci))

    rank_of(e) = sortperm(e)
    p1, p0 = rank_of(e1), rank_of(e0)
    r0 = invperm(p0)
    @printf("%4s %10s %10s %6s  %-14s %-22s %s\n", "rank", "dressed", "plain", "(rank)", "ket", "", "sector")
    for (i, k) in enumerate(p1[1:min(nshow, end)])
        y = ys[k]
        @printf("%4d %10.3f %10.3f %6d  %-14s %-22s %d\n", i, 1e3 * (e1[k] - e_fci), 1e3 * (e0[k] - e_fci), r0[k],
                ket_label(y), describe(y, ref), reduce_span(y, V0))
    end

    pick(p) = (S = UInt64[]; for k in p
                   y = ys[k]
                   any(mirror(y) == s for s in S) && continue
                   push!(S, y); length(S) == nk && break
               end; S)
    S1, S0 = pick(p1), pick(p0)
    @printf("suggested kets (dressed): --kets %s   sectors %s\n", join(ket_label.(S1), ","),
            join([reduce_span(y, V0) for y in S1], ","))
    @printf("plain-diagonal baseline:   --kets %s   sectors %s\n", join(ket_label.(S0), ","),
            join([reduce_span(y, V0) for y in S0], ","))

    if !isempty(opts["out"])
        rows = [Dict("ket" => ket_label(ys[k]), "describe" => describe(ys[k], ref), "dressed" => e1[k],
                     "plain" => e0[k], "sector" => reduce_span(ys[k], V0)) for k in p1[1:min(50, end)]]
        open(opts["out"], "w") do f
            JSON.print(f, Dict("R" => meta["R_angstrom"], "basis" => basis, "coeff" => coeff, "s2_penalty" => λ,
                               "n_penalty" => κ, "rotations" => nrot, "terms" => length(O), "seconds" => t, "e_fci" => e_fci,
                               "kets" => ket_label.(S1), "kets_plain" => ket_label.(S0), "lowest" => rows), 1)
        end
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    screen_main(ARGS)
end
