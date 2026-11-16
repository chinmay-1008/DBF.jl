# Block vDBF with Pauli propagation: idea 1c of code/vdbf_excited_states.md on the
# frozen-core N2 Hamiltonians, the truncated counterpart of frozen_core/sv_block.jl.
#
# dbf_groundstate's loop (src/groundstate.jl) with the plan's five swaps, written
# here so src/ is unchanged:
#
#  1. ranking   G = [D, O] with the 1c operator
#               D = sum_rest Z_k + Z_H Z_L + Z_Hb Z_Lb + c (Z_H + Z_Hb - Z_H Z_Hb)
#               in place of P = I + sum_i Z_i; its top eigenspace is the block
#               {|0>, |a>, |b>} (checked).
#  2. gradient  each Pauli p of G scores sum_y 2 Re(<p y|O|y> c_p ph_y) over the
#               three block states, i.e. the coefficient of p in [D, O] times the
#               exact gradient of the block sum (dbf_groundstate's score is the
#               single-state version).  In-block rotations cancel exactly.
#  3. angle     theta minimises C(theta) = sum_y <y|U' O U|y> = a + b cos + c sin,
#               the sum of optimize_theta_expval's three per-state costs; solved
#               in closed form.
#  4. truncation correction  for the 3x3 energy matrix M = <y|O|y'> and the 3x3
#               variance matrix V = <y|O(1 - Π_S)O|y'> = A - M², A = <y|O²|y'>.
#               Coefficient truncation only drops terms (O_before = O_kept + B), so
#               the exact changes follow from the dropped terms alone, as in
#               PauliOperators' fast path for EnergyVarianceCorrection:
#                   ΔM = -<y|B|y'>,   ΔA = -<O y|B y'> - <B y|O y'> - <B y|B y'>,
#               needing B|y> (on #dropped kets) and O|y> on that support.  The
#               changes accumulate; corrected M = M - ΔM_acc and V = V - ΔV_acc,
#               the block version of "E - accumulated error", "Var - var error".
#  5. convergence and output  each eigenstate φ_i of M has energy λ_i and its own
#               energy variance var_i = φ_i' V φ_i (raw from M, V; corrected from
#               the corrected M, V); they sum to the block variance V_S = tr V.
#
# Truncation is applied after every rotation (evolve!, then the drop-based
# correction, then truncate! with the same cutoff).
#
#     julia --project=. run_block_pp.jl <point dir> <basis> <label>
#         [--coeff 1e-5] [--grad 1e-6] [--max-iter 200] [--max-rots 50]
#         [--conv 1e-5] [--c 1.0] [--outdir frozen_core/sv_results]
#     e.g. julia --project=. run_block_pp.jl frozen_core/data_unocore/R_1.10 uno uno_1.10_blockpp_c1e-5
using DBF, PauliOperators, JLD2, JSON, Printf, LinearAlgebra
using DBF: commutator_clipped, matvec

const N_Q = 16
const NORB = 8

function parse_cli(args)
    opts = Dict("coeff" => "1e-5", "grad" => "1e-6", "max-iter" => "200", "max-rots" => "50",
                "conv" => "1e-5", "c" => "1.0", "outdir" => joinpath(@__DIR__, "frozen_core", "sv_results"))
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
    length(pos) == 3 || error("usage: run_block_pp.jl <point dir> <basis> <label> [options]")
    return pos..., opts
end

# as in run_dbf_pes.jl: conjugate by X on the occupied qubits so the reference is |0...0>
function to_vacuum_reference(H::PauliSum{N,T}, ψ::Ket{N}) where {N,T}
    Hx = PauliSum(N, T)
    for (p, c) in H
        Hx[p] = isodd(count_ones(p.z & ψ.v)) ? -c : c
    end
    return Hx, Ket{N}(0)
end

# ---------------------------------------------------------------- block pieces
# <u|v> for ket amplitudes stored by ket bits
ip(u::AbstractDict, v::AbstractDict) = sum((conj(get(u, s, zero(ComplexF64))) * x for (s, x) in v); init=zero(ComplexF64))

# A_ij = <k_i|O²|k_j> = <O k_i|O k_j> from the vectors O|k>
function block_A(σvs)
    n = length(σvs)
    A = [real(ip(σvs[i], σvs[j])) for i in 1:n, j in 1:n]
    return (A + A') / 2
end

# swap 4: drop |c| <= thresh and accumulate the exact change of M and V = A - M²
mutable struct BlockAccumulator
    M::Matrix{Float64}
    V::Matrix{Float64}
    dropped::Int
    a::Matrix{ComplexF64}       # O_kept|k_j> as dense columns over the 2^N kets (zero between uses)
    b::Matrix{ComplexF64}       # B|k_j>, likewise
    inU::BitVector              # kets where B|k_j> lives or that are block kets
end
# Dense scratch columns over the 2^N kets only for small N; beyond that the correction
# keeps the amplitudes in hash maps (truncate_block_sparse!).  BLOCK_DENSE_MAX overrides.
const DENSE_MAX = parse(Int, get(ENV, "BLOCK_DENSE_MAX", "22"))
BlockAccumulator(n, N) = N <= DENSE_MAX ?
    BlockAccumulator(zeros(n, n), zeros(n, n), 0, zeros(ComplexF64, 1 << N, n), zeros(ComplexF64, 1 << N, n), falses(1 << N)) :
    BlockAccumulator(zeros(n, n), zeros(n, n), 0, zeros(ComplexF64, 0, n), zeros(ComplexF64, 0, n), falses(0))

# the shared last step: from O_kept|k_j> (a) and B|k_j> (b) on the touched kets, the changes of M and V
function accumulate_block!(acc::BlockAccumulator, n, Mafter, ΔM, ip)
    ΔA = [real(-ip(:a, i, :b, j) - ip(:b, i, :a, j) - ip(:b, i, :b, j)) for i in 1:n, j in 1:n]
    Mbefore = Mafter - ΔM
    ΔV = ΔA - (Mafter * Mafter - Mbefore * Mbefore)
    acc.M .+= (ΔM + ΔM') / 2
    acc.V .+= (ΔV + ΔV') / 2
end

# Large N: the same correction with amplitudes in hash maps keyed by ket bits.
function truncate_block_sparse!(O::SparsePauliVector{N,W}, kets::Vector{Ket{N}}, strategy::CoeffTruncation,
                                acc::BlockAccumulator) where {N,W}
    n, thr = length(kets), strategy.thresh
    kv = [k.v % Int128 for k in kets]
    b = [Dict{Int128,ComplexF64}() for _ in 1:n]
    nd = 0
    @inbounds for i in 1:O.n
        c = O.c[i]
        abs(c) <= thr || continue
        nd += 1
        z, x = O.z[i] % Int128, O.x[i] % Int128
        for j in 1:n
            ph, σ = PauliOperators._ket_action(z, x, kv[j])
            b[j][σ] = get(b[j], σ, zero(ComplexF64)) + ph * c
        end
    end
    truncate!(O, strategy)
    nd == 0 && return O
    acc.dropped += nd
    U = Set{Int128}(kv)
    foreach(bj -> union!(U, keys(bj)), b)
    a = [Dict{Int128,ComplexF64}() for _ in 1:n]
    @inbounds for i in 1:O.n
        x = O.x[i] % Int128
        for j in 1:n
            σ = kv[j] ⊻ x
            σ in U || continue
            ph, _ = PauliOperators._ket_action(O.z[i] % Int128, x, kv[j])
            a[j][σ] = get(a[j], σ, zero(ComplexF64)) + ph * O.c[i]
        end
    end
    vecs = Dict(:a => a, :b => b)
    ip(u, i, v, j) = (du = vecs[u][i]; sum((conj(get(du, s, zero(ComplexF64))) * x for (s, x) in vecs[v][j]);
                                          init=zero(ComplexF64)))
    Mafter = [real(get(a[j], kv[i], zero(ComplexF64))) for i in 1:n, j in 1:n]
    ΔM = [-real(get(b[j], kv[i], zero(ComplexF64))) for i in 1:n, j in 1:n]
    accumulate_block!(acc, n, Mafter, ΔM, ip)
    return O
end

# Both passes read the SparsePauliVector's flat sorted arrays (z, x, c, n) and use
# PauliOperators' raw-word ket action, so the work per term is a few bit operations;
# amplitudes go into dense columns indexed by ket bits (16 qubits: 2^16 entries).
function truncate_block!(O::SparsePauliVector{N,W}, kets::Vector{Ket{N}}, strategy::CoeffTruncation,
                         acc::BlockAccumulator) where {N,W}
    N > DENSE_MAX && return truncate_block_sparse!(O, kets, strategy, acc)
    n, thr = length(kets), strategy.thresh
    kv = [k.v % Int128 for k in kets]
    a, b, inU = acc.a, acc.b, acc.inU
    U = Int[]                                                 # touched kets (0-based bits)
    for k in kv
        inU[k+1] || (inU[k+1] = true; push!(U, k))
    end
    nd = 0
    @inbounds for i in 1:O.n
        c = O.c[i]
        abs(c) <= thr || continue
        nd += 1
        z, x = O.z[i] % Int128, O.x[i] % Int128
        for j in 1:n
            ph, σ = PauliOperators._ket_action(z, x, kv[j])
            b[σ+1, j] += ph * c
            inU[σ+1] || (inU[σ+1] = true; push!(U, σ))
        end
    end
    truncate!(O, strategy)                                    # same criterion, PauliOperators' fast path
    if nd > 0
        acc.dropped += nd
        @inbounds for i in 1:O.n
            x = O.x[i] % Int128
            for j in 1:n
                σ = kv[j] ⊻ x
                inU[σ+1] || continue
                ph, _ = PauliOperators._ket_action(O.z[i] % Int128, x, kv[j])
                a[σ+1, j] += ph * O.c[i]
            end
        end
        cols = Dict(:a => a, :b => b)
        ip(u, i, v, j) = sum(conj(cols[u][s+1, i]) * cols[v][s+1, j] for s in U)    # <u_i|v_j> over the touched kets
        Mafter = [real(a[kv[i]+1, j]) for i in 1:n, j in 1:n]
        ΔM = [-real(b[kv[i]+1, j]) for i in 1:n, j in 1:n]
        accumulate_block!(acc, n, Mafter, ΔM, ip)
    end
    @inbounds for s in U                                      # reset the scratch columns
        inU[s+1] = false
        for j in 1:n
            a[s+1, j] = 0
            b[s+1, j] = 0
        end
    end
    return O
end

# eigenvalues of M and each eigenvector's variance under V
function states_of(M, V)
    F = eigen(Symmetric(M))
    return F.values, [dot(F.vectors[:, i], V * F.vectors[:, i]) for i in axes(F.vectors, 2)]
end

# swap 2: score of each Pauli in G, summed over the block states
function block_gradient(G, σvs, kets::Vector{Ket{N}}, thresh) where N
    grad_vec, grad_ops = Float64[], PauliBasis{N}[]
    for (p, c) in G
        g = 0.0
        for (σv, k) in zip(σvs, kets)
            ph, σ = p * k
            g += 2 * real(get(σv, σ, zero(ComplexF64)) * c * ph)
        end
        if abs(g) > thresh
            push!(grad_vec, g)
            push!(grad_ops, p)
        end
    end
    return grad_vec, grad_ops
end

# swap 3: the block sum after rotating by exp(-i θ G/2) is a + b cos θ + c sin θ
function optimize_theta_block(O, G::PauliBasis{N}, kets::Vector{Ket{N}}) where N
    a, b, c = 0.0, 0.0, 0.0
    for k in kets
        cg, kg = G * k
        A = real(expectation_value(O, k))
        B = real(expectation_value(O, kg)) * abs2(cg)
        OG = matrix_element(k', O, kg) * cg
        # optimize_theta_expval's cost: cos²(θ/2) A + sin²(θ/2) B - 2i cos(θ/2) sin(θ/2) OG
        a += (A + B) / 2
        b += (A - B) / 2
        c += real(-1im * OG)
    end
    θ = atan(-c, -b)
    θ < 0 && (θ += 2π)
    return θ, a + b * cos(θ) + c * sin(θ), a + b
end

# the plan's 1c operator, as a PauliSum (Pauli(N, Z=[q+1]) acts on bit q)
function d1c_operator(N, qH, qL, qHb, qLb, cpar)
    D = PauliSum(N)
    for q in 0:N-1
        q in (qH, qL, qHb, qLb) && continue
        D += Pauli(N, Z=[q + 1])
    end
    D += Pauli(N, Z=[qH + 1, qL + 1]) + Pauli(N, Z=[qHb + 1, qLb + 1])
    D += cpar * (Pauli(N, Z=[qH + 1]) + Pauli(N, Z=[qHb + 1]) - Pauli(N, Z=[qH + 1, qHb + 1]))
    return D
end

# ---------------------------------------------------------------- the flow
function dbf_block(Oin, kets::Vector{Ket{N}}, D; operator_truncation, gradient_truncation, max_iter, max_rots_per_grad,
                   conv_thresh, grad_thresh=1e-6, e_ref=0.0, verbose=1, groups=nothing, on_iter=nothing) where N
    # groups: block states that H connects, e.g. per symmetry sector; M and V are diagonalised group by
    # group and the eigenvalues listed in group order.  Default: one group (eigenvalues sorted).
    groups = groups === nothing ? [collect(eachindex(kets))] : groups
    O = deepcopy(Oin)
    acc = BlockAccumulator(length(kets), N)
    hist = Dict{String,Any}(k => Any[] for k in ("lam", "lam_corr", "var", "var_corr", "VS", "VS_corr", "C", "C_corr",
                                                  "terms", "dropped", "grad_norm", "len_G", "n_rots", "time"))
    generators, angles = PauliBasis{N}[], Float64[]
    σvs = Any[]                                               # O|k> for the block kets, reused by the next gradient
    function record!(gnorm, lenG, nrot, t)
        empty!(σvs); append!(σvs, [matvec(O, k) for k in kets])
        sv = [Dict{Int128,ComplexF64}(s.v => x for (s, x) in v) for v in σvs]
        M = [real(get(sv[j], kets[i].v, zero(ComplexF64))) for i in eachindex(kets), j in eachindex(kets)]
        M = (M + M') / 2
        V = block_A(sv) - M * M
        Mc, Vc = M - acc.M, V - acc.V
        per(Mx, Vx) = (r = [states_of(Mx[g, g], Vx[g, g]) for g in groups]; (vcat(first.(r)...), vcat(last.(r)...)))
        lam, var = per(M, V)
        lamc, varc = per(Mc, Vc)
        push!(hist["lam"], lam); push!(hist["var"], var)
        push!(hist["lam_corr"], lamc); push!(hist["var_corr"], varc)
        push!(hist["VS"], tr(V)); push!(hist["VS_corr"], tr(Vc))
        push!(hist["C"], tr(M)); push!(hist["C_corr"], tr(Mc))
        push!(hist["terms"], length(O)); push!(hist["dropped"], acc.dropped); push!(hist["grad_norm"], gnorm)
        push!(hist["len_G"], lenG); push!(hist["n_rots"], nrot); push!(hist["time"], t)
        if verbose > 0
            @printf("%5d  λ-E_ref(mHa) %s  corrected %s  var %s  V_S %.3e  |g| %.2e  len(O) %8d  rots %3d  %7.1f s\n",
                    length(hist["lam"]) - 1, join([@sprintf("%9.3f", 1e3 * (x - e_ref)) for x in lam], " "),
                    join([@sprintf("%9.3f", 1e3 * (x - e_ref)) for x in lamc], " "),
                    join([@sprintf("%.2e", x) for x in varc], " "), tr(V), gnorm, length(O), nrot, t)
            flush(stdout)
        end
    end
    Dop = O isa SparsePauliVector ? SparsePauliVector(D; T=ComplexF64) : D
    record!(0.0, 0, 0, 0.0)
    t0 = time()
    for iter in 1:max_iter
        G = commutator_clipped(Dop, O)
        truncate!(G, gradient_truncation)
        grad_vec, grad_ops = block_gradient(G, σvs, kets, grad_thresh)
        order = sort(collect(eachindex(grad_vec)), by=i -> (-abs(grad_vec[i]), grad_ops[i].z, grad_ops[i].x))
        n_rots = 0
        for gi in order
            Gi = grad_ops[gi]
            θi, _, _ = optimize_theta_block(O, Gi, kets)
            evolve!(O, Gi, θi)
            truncate_block!(O, kets, operator_truncation, acc)
            push!(generators, Gi); push!(angles, θi)
            n_rots += 1
            n_rots >= max_rots_per_grad && break
        end
        gnorm = norm(grad_vec)
        record!(gnorm, length(grad_vec), n_rots, time() - t0)
        on_iter === nothing || on_iter(iter, hist, generators, angles, acc, O)
        if gnorm < conv_thresh
            verbose < 1 || println(" Converged.")
            break
        end
        if n_rots == 0
            @warn "no search directions found"
            break
        end
    end
    return O, hist, generators, angles, acc
end

function main(args)
    dir, basis, label, opts = parse_cli(args)
    meta = JSON.parsefile(joinpath(dir, "metadata.json"))
    e_fci = meta["E_FCI_with_mu"] + meta["constant"]
    data = load(joinpath(dir, "hamiltonian_$(basis).jld2"))
    H, ψ0 = to_vacuum_reference(data["H"], data["reference_ket"])
    abs(real(expectation_value(H, ψ0)) - data["reference_energy"]) < 1e-8 ||
        error("<0|X H X|0> does not reproduce the reference energy")
    na = meta["n_alpha"]
    qH, qL, qHb, qLb = na - 1, na, NORB + na - 1, NORB + na
    bitk(qs...) = Ket{N_Q}(reduce(|, (Int128(1) << q for q in qs); init=Int128(0)))
    kets = [Ket{N_Q}(0), bitk(qH, qL), bitk(qHb, qLb)]
    cpar = parse(Float64, opts["c"])
    D = d1c_operator(N_Q, qH, qL, qHb, qLb, cpar)
    # the block must be D's top eigenspace: check the block kets and every single and double flip
    dtop = real(expectation_value(D, kets[1]))
    all(abs(real(expectation_value(D, k)) - dtop) < 1e-12 for k in kets) || error("block kets differ in D")
    others = [bitk(q) for q in 0:N_Q-1]
    append!(others, [bitk(q1, q2) for q1 in 0:N_Q-1 for q2 in q1+1:N_Q-1])
    filter!(k -> !(k in kets), others)
    maximum(real(expectation_value(D, k)) for k in others) < dtop - 1e-9 || error("block is not D's top eigenspace")

    Hop = SparsePauliVector(H)
    op_trunc = CoeffTruncation(parse(Float64, opts["coeff"]))
    grad_trunc = CoeffTruncation(parse(Float64, opts["grad"]))
    println("="^100)
    @printf(" R = %.2f %s  %d Pauli terms  E_FCI = %.10f  block = |0>, |%s>, |%s>  D top value %.1f\n",
            meta["R_angstrom"], basis, length(H), e_fci, string(kets[2].v, base=2), string(kets[3].v, base=2), dtop)
    @printf(" operator cutoff %s, gradient cutoff %s, %s rotations per gradient, max %s iterations; energies in mHa above E_FCI\n",
            opts["coeff"], opts["grad"], opts["max-rots"], opts["max-iter"])
    println("="^100)
    t = @elapsed O, hist, gens, angs, acc = dbf_block(Hop, kets, D;
        operator_truncation=op_trunc, gradient_truncation=grad_trunc,
        max_iter=parse(Int, opts["max-iter"]), max_rots_per_grad=parse(Int, opts["max-rots"]),
        conv_thresh=parse(Float64, opts["conv"]), e_ref=e_fci)
    mkpath(opts["outdir"])
    open(joinpath(opts["outdir"], "$(label).json"), "w") do io
        JSON.print(io, Dict("label" => label, "R" => meta["R_angstrom"], "basis" => basis, "E_FCI" => e_fci,
                            "settings" => opts, "n_rotations" => length(angs), "wall_time_s" => t,
                            "final_terms" => length(O), "accumulated_M" => acc.M, "accumulated_V" => acc.V,
                            "rots_per_iter" => hist["n_rots"],
                            "generators" => [[Int(g.z), Int(g.x)] for g in gens], "angles" => angs,
                            "history" => hist), 1)
    end
    @printf(" done: %d rotations in %.0f s; wrote %s.json\n", length(angs), t, label)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main(ARGS)
end
