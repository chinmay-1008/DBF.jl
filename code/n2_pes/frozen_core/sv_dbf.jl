# Exact (untruncated) DBF in state-vector form, for the 16-qubit frozen-core N2
# Hamiltonians.
#
# dbf_groundstate works in the Heisenberg picture, O_K = U_K' ... U_1' H U_1 ... U_K,
# and truncates O after every rotation.  Here nothing is truncated: the dressing
# W = U_1 U_2 ... U_K is stored as a dense real orthogonal matrix on the parity
# sector reachable from |0> (the GF(2) span of H's x-strings), so that
#
#     psi = W|0>,     <x|O|0> = (W' H psi)_x,     <x|O|x> = (W e_x)' H (W e_x)
#
# are exact.  A generator for direction x is the Pauli with X on the flipped
# qubits and Y on the lowest one (odd Y count -> real rotation), and it acts on
# |0> like any DBF generator with that x string; only its action on other basis
# states (the dressing of the frame) depends on that choice of z.
#
# Each macro-iteration computes every O_x0 once, ranks the directions, and
# applies the top --max-rots of them in order, each at its exact optimal angle
# (a 2x2 minimisation on {|0>, |x>} of the current O), like dbf_groundstate's
# max_rots_per_grad batches.  Ranking (--rank):
#   onebody    |x| * O_x0^2   what P = I + sum_i Z_i gives: <x|[P,O]|0> = -2|x| O_x0
#   projector  O_x0^2         what the exact projector P = |0><0| gives
#
# --s2-penalty λ runs the flow on H + λ S^2 instead of H (the singlet ground
# state is unchanged; a spin-S state moves up by λ S(S+1)); E and the variance
# are always reported for H.  --ypos low|high puts the generator's Y on the
# lowest or highest flipped qubit, which changes only the frame dressing.
#
# --init singlet starts from the singlet projection of the UHF determinant
# instead of |0>: S^2 is diagonalised in the (n_alpha, n_beta) block, |0> is
# projected onto its S^2 ~ 0 eigenvectors, and W starts as the Householder
# reflection I - 2uu' that maps e_0 to that state.  This is DBF on
# O_0 = W' H W, i.e. the projected state folded into H as a fixed preparation.
#
# Diagnostics per iteration: E, variance, fidelity with the exact ground state,
# physical <S^2> (alpha/beta orbitals differ, so S+ uses <basis>_ab_overlap.npy),
# and the weight outside the 10-electron sector.
#
#     julia -t 8 --project=../../integrals sv_dbf.jl <point dir> <basis> <label> [--rank onebody]
#         [--max-rots 50] [--max-iter 200] [--grad-thresh 1e-6] [--s2-penalty 0] [--ypos low] [--init uhf]
using PauliOperators, JLD2, JSON, NPZ, Printf, LinearAlgebra
using Base.Threads

const N = 16
const NORB = 8

function parse_cli(args)
    opts = Dict("rank" => "onebody", "max-rots" => "50", "max-iter" => "200",
                "grad-thresh" => "1e-6", "s2-penalty" => "0", "ypos" => "low", "init" => "uhf",
                "outdir" => joinpath(@__DIR__, "sv_results"))
    pos = String[]
    i = 1
    while i <= length(args)
        if startswith(args[i], "--")
            key = args[i][3:end]
            haskey(opts, key) || error("unknown option $(args[i])")
            opts[key] = args[i+1]
            i += 2
        else
            push!(pos, args[i])
            i += 1
        end
    end
    length(pos) == 3 || error("usage: sv_dbf.jl <point dir> <basis> <label> [options]")
    return pos..., opts
end

function to_vacuum_reference(H::Dict{PauliBasis{N},T}, ref) where {N,T}
    Hx = Dict{PauliBasis{N},T}()
    for (p, c) in H
        Hx[p] = isodd(count_ones(p.z & ref)) ? -c : c
    end
    return Hx
end

# --- reachable sector: GF(2) span of the x strings ----------------------------
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

function enumerate_span(basis)
    states = UInt64[0]
    for b in basis
        states = vcat(states, states .⊻ b)
    end
    return states
end

# --- operators as real sparse matrices on the sector ---------------------------
function sector_matrix(terms, states, pos)
    D = length(states)
    groups = Dict{UInt64,Vector{Tuple{PauliBasis{N},ComplexF64}}}()
    for (p, c) in terms
        push!(get!(groups, UInt64(p.x), []), (p, ComplexF64(c)))
    end
    I, J, V = Int[], Int[], Float64[]
    for (x, ps) in groups, (j, b) in enumerate(states)
        k = pos[(b ⊻ x)+1]
        k == 0 && error("x string $x leaves the sector")
        val = 0.0im
        for (p, c) in ps
            ph, _ = p * Ket{N}(b)
            val += c * ph
        end
        abs(imag(val)) < 1e-10 || error("complex matrix element $val")
        abs(val) > 1e-14 || continue
        push!(I, k); push!(J, j); push!(V, real(val))
    end
    return sparse(I, J, V, D, D)
end

# Physical S^2 = Sz^2 + Sz + S- S+ with S+ = sum_pq O_pq a+_{p,alpha} a_{q,beta}
# and S- = S+'.  Occupations are b xor ref; alpha orbital p is bit p, beta
# orbital q is bit NORB+q; Jordan-Wigner signs count occupied bits below.
jwsign(occ, bit) = isodd(count_ones(occ & (bit - 1))) ? -1.0 : 1.0

# a+_{to} a_{from} |occ>, with `from` occupied and `to` empty after the removal
function hop(occ, from, to)
    occ & from == 0 && return (0.0, occ)
    s1 = jwsign(occ, from)
    o1 = occ ⊻ from
    o1 & to != 0 && return (0.0, occ)
    return (s1 * jwsign(o1, to), o1 | to)
end

function s2_matrix(states, pos, ref, O)
    D = length(states)
    amask = UInt64(1) << NORB - 1
    I, J, V = Int[], Int[], Float64[]
    for (j, b) in enumerate(states)
        occ = b ⊻ ref
        sz = 0.5 * (count_ones(occ & amask) - count_ones(occ >> NORB))
        col = Dict{Int,Float64}(j => sz^2 + sz)
        for q in 0:NORB-1, p in 0:NORB-1                 # S+: beta q -> alpha p
            O[p+1, q+1] == 0 && continue
            s1, o1 = hop(occ, UInt64(1) << (NORB + q), UInt64(1) << p)
            s1 == 0 && continue
            for p2 in 0:NORB-1, q2 in 0:NORB-1           # S-: alpha p2 -> beta q2
                O[p2+1, q2+1] == 0 && continue
                s2, o2 = hop(o1, UInt64(1) << p2, UInt64(1) << (NORB + q2))
                s2 == 0 && continue
                k = pos[(o2 ⊻ ref)+1]
                k == 0 && error("S^2 leaves the sector")
                col[k] = get(col, k, 0.0) + O[p+1, q+1] * O[p2+1, q2+1] * s1 * s2
            end
        end
        for (k, v) in col
            abs(v) > 1e-14 || continue
            push!(I, k); push!(J, j); push!(V, v)
        end
    end
    return sparse(I, J, V, D, D)
end

leak(ψ, states, ref, nel) = sum(abs2(ψ[k]) for (k, b) in enumerate(states) if count_ones(b ⊻ ref) != nel)

# Singlet projection of |0> (the reference determinant): diagonalise S^2 on the
# (n_alpha, n_beta) block and keep the S^2 ~ 0 eigenvectors.  Normalised, phi[1] > 0.
function singlet_projection(S2s, blk, D)
    Fs = eigen(Symmetric(Matrix(S2s[blk, blk])))
    keep = Fs.values .< 0.5
    r = findfirst(==(1), blk)
    φ = zeros(D)
    φ[blk] = Fs.vectors[:, keep] * Fs.vectors[r, keep]
    w = dot(φ, φ)
    φ ./= sqrt(w) * sign(φ[1])
    @printf("singlet projection: %d S^2 ~ 0 states (max |S^2| %.1e), weight of |0> on them %.4f, next S^2 eigenvalue %.3f\n",
            count(keep), maximum(abs, Fs.values[keep]), w, minimum(Fs.values[.!keep]))
    return φ
end

# --- W <- W R for R = exp(-i θ/2 G), G = Y on one flipped qubit, X on the rest -
# R e_b = c e_b + s μ(b) e_{b⊻x},  μ(b) = -i g(b) where G|b> = g(b)|b⊻x>.
function generator(x::UInt64, ypos)
    y = ypos == "low" ? x & (~x + 1) : UInt64(1) << (63 - leading_zeros(x))
    return PauliBasis{N}(Int128(y), Int128(x))
end

μ(G, b) = (ph = (G * Ket{N}(b))[1]; r = real(-1im * ph); @assert abs(imag(-1im * ph)) < 1e-12; r)

function rotate_W!(W, G, θ, states, pos, x)
    c, s = cos(θ / 2), sin(θ / 2)
    hb = UInt64(1) << (63 - leading_zeros(x))
    D = size(W, 1)
    @threads for j in eachindex(states)
        b = states[j]
        b & hb == 0 || continue
        k = pos[(b ⊻ x)+1]
        mj, mk = μ(G, b), μ(G, b ⊻ x)
        @inbounds @simd for r in 1:D
            wj, wk = W[r, j], W[r, k]
            W[r, j] = c * wj + s * mj * wk
            W[r, k] = c * wk + s * mk * wj
        end
    end
    return W
end

function main(args)
    dir, basis, label, opts = parse_cli(args)
    rank = opts["rank"]
    rank in ("onebody", "projector") || error("--rank must be onebody or projector")
    max_rots, max_iter = parse(Int, opts["max-rots"]), parse(Int, opts["max-iter"])
    gthresh = parse(Float64, opts["grad-thresh"])
    λ = parse(Float64, opts["s2-penalty"])
    ypos = opts["ypos"]
    ypos in ("low", "high") || error("--ypos must be low or high")
    init = opts["init"]
    init in ("uhf", "singlet") || error("--init must be uhf or singlet")

    meta = JSON.parsefile(joinpath(dir, "metadata.json"))
    d = load(joinpath(dir, "hamiltonian_$(basis).jld2"))
    ref = UInt64(d["reference_ket"].v)
    nel = meta["n_alpha"] + meta["n_beta"]
    e_fci = meta["E_FCI_with_mu"] + meta["constant"]
    Ovl = npzread(joinpath(dir, "$(basis)_ab_overlap.npy"))
    # Symmetry-forbidden overlaps are localization noise (up to ~4e-8 in lno);
    # keep them out of S^2.  s2_matrix still errors if a real one survives.
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
    @printf("R = %.2f %s: %d Pauli terms, sector dimension %d (rank %d), nnz %d, threads %d\n",
            meta["R_angstrom"], basis, length(H), D, length(gb), nnz(Hs), nthreads())
    norm(Hs - Hs') < 1e-10 || error("sector H is not symmetric")
    S2s = s2_matrix(states, pos, ref, Ovl)
    norm(S2s - S2s') < 1e-10 || error("sector S^2 is not symmetric")
    norm(Hs * S2s - S2s * Hs) / norm(Hs) < 1e-8 ||
        @printf("note: [H, S^2] / |H| = %.1e (S^2 is approximate: frozen-core alpha/beta overlap)\n",
                norm(Hs * S2s - S2s * Hs) / norm(Hs))
    spin_s2(ψ) = dot(ψ, S2s * ψ)
    Hopt = λ == 0 ? Hs : Hs + λ * S2s

    # exact ground state in the (5a,5b) block of the sector
    amask = UInt64(1) << NORB - 1
    blk = [k for (k, b) in enumerate(states)
           if count_ones((b ⊻ ref) & amask) == meta["n_alpha"] && count_ones((b ⊻ ref) >> NORB) == meta["n_beta"]]
    F = eigen(Symmetric(Matrix(Hs[blk, blk])))
    Φ0 = zeros(D); Φ0[blk] = F.vectors[:, 1]
    @printf("exact ground state in the (%d,%d) block (dim %d): %.10f   FCI %.10f   diff %.1e   <S^2> %.4f\n",
            meta["n_alpha"], meta["n_beta"], length(blk), F.values[1], e_fci, F.values[1] - e_fci,
            spin_s2(Φ0))
    s2_low = [dot(F.vectors[:, i], S2s[blk, blk] * F.vectors[:, i]) for i in 2:6]
    println("their <S^2>: ", join([@sprintf("%.3f", s) for s in s2_low], ", "))
    abs(F.values[1] - e_fci) < 1e-8 || error("sector ground state is not the FCI energy")
    gaps = [(v - F.values[1]) * 1e3 for v in F.values[2:6]]
    println("next eigenvalues in that block (mHa above FCI): ", join([@sprintf("%.2f", g) for g in gaps], ", "))

    # h = Hopt ψ drives ranking and angles; E and the variance are always of H
    W = Matrix{Float64}(I, D, D)
    if init == "singlet"
        φ = singlet_projection(S2s, blk, D)
        u = -φ; u[1] += 1; u ./= norm(u)
        W .-= 2 .* u .* u'
        maximum(abs, W[:, 1] - φ) < 1e-12 || error("Householder W does not map e_0 to the projected state")
    end
    ψ = W[:, 1]
    h = Hopt * ψ
    E = dot(ψ, h)
    flips = [count_ones(b) for b in states]
    hist = Dict(k => Float64[] for k in ("E", "var", "fidelity", "S2", "leak", "max_grad", "time"))
    rot_E = Float64[E]
    record!(ψ, g, t) = begin
        hH = Hs * ψ
        eH = dot(ψ, hH)
        push!(hist["E"], eH); push!(hist["var"], dot(hH, hH) - eH^2)
        push!(hist["fidelity"], dot(Φ0, ψ)^2); push!(hist["S2"], spin_s2(ψ))
        push!(hist["leak"], leak(ψ, states, ref, nel)); push!(hist["max_grad"], g)
        push!(hist["time"], t)
    end
    record!(ψ, 0.0, 0.0)
    @printf("%5s %12s %10s %10s %8s %7s %9s %9s %6s\n", "iter", "E-FCI mHa", "variance", "max|O_x0|",
            "fidelity", "<S^2>", "leak", "drift", "time")
    @printf("%5d %+12.4f %10.3e %10s %8.4f %7.4f %9.2e %9s %6.1f\n", 0, 1e3 * (hist["E"][end] - e_fci),
            hist["var"][end], "", hist["fidelity"][end], hist["S2"][end], hist["leak"][end], "", 0.0)
    t0 = time()
    for iter in 1:max_iter
        o = W' * h
        o[1] = 0
        gmax = maximum(abs, o)
        if gmax < gthresh
            println("converged: max |O_x0| = $gmax < $gthresh")
            break
        end
        cand = [k for k in 2:D if abs(o[k]) > gthresh]
        score = rank == "onebody" ? [flips[k] * o[k]^2 for k in cand] : [o[k]^2 for k in cand]
        order = cand[sortperm(score; rev=true)][1:min(max_rots, length(cand))]
        for k in order
            x = states[k]
            G = generator(x, ypos)
            v = W[:, k]
            hv = Hopt * v
            a, bb, m = E, dot(v, hv), dot(v, h)
            m0 = μ(G, UInt64(0))
            # E(θ) = (a+b)/2 + (a-b)/2 cos θ + m0 m sin θ
            θ = atan(-m0 * m, -(a - bb) / 2)
            c, s = cos(θ / 2), sin(θ / 2)
            rotate_W!(W, G, θ, states, pos, x)
            ψ .= c .* ψ .+ s * m0 .* v
            h .= c .* h .+ s * m0 .* hv
            E = dot(ψ, h)
            push!(rot_E, E)
        end
        # recompute from scratch to keep the recurrences honest
        ψ .= view(W, :, 1)
        hnew = Hopt * ψ
        drift = maximum(abs, hnew .- h)
        h .= hnew
        E = dot(ψ, h)
        record!(ψ, gmax, time() - t0)
        @printf("%5d %+12.4f %10.3e %10.3e %8.4f %7.4f %9.2e %9.1e %6.1f\n", iter, 1e3 * (hist["E"][end] - e_fci),
                hist["var"][end], gmax, hist["fidelity"][end], hist["S2"][end], hist["leak"][end],
                drift, hist["time"][end])
        flush(stdout)
    end
    orth = maximum(abs, W[:, 1:min(D, 64)]' * W[:, 1:min(D, 64)] - I)
    @printf("orthogonality of W (first 64 columns): %.1e\n", orth)

    mkpath(opts["outdir"])
    open(joinpath(opts["outdir"], "$(label).json"), "w") do io
        JSON.print(io, Dict("label" => label, "R" => meta["R_angstrom"], "basis" => basis,
                            "rank" => rank, "max_rots" => max_rots, "E_FCI" => e_fci,
                            "s2_penalty" => λ, "ypos" => ypos, "init" => init,
                            "sector_dim" => D, "gaps_mHa" => gaps, "history" => hist,
                            "rotation_energies" => rot_E, "orthogonality" => orth), 1)
    end
end

using SparseArrays
if abspath(PROGRAM_FILE) == @__FILE__
    main(ARGS)
end
