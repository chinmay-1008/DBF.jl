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
#  4. truncation correction  BlockCorrection, a PauliOperators CorrectionAccumulator
#               (the documented _measure / _accumulate! extension point), records
#               the 3x3 block matrix M before and after every truncation and
#               accumulates the change.  Corrected block = M - accumulated, the
#               block version of dbf_groundstate's "E - accumulated error".
#  5. convergence and output  the block variance V_S (from PauliOperators.variance
#               per block state) and the eigenvalues of M, raw and corrected.
#
# Truncation is applied after every rotation (evolve! then truncate!, the
# unfused path of dbf_groundstate, since the correction is a custom one).
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
# M_ij = <k_i|O|k_j>; only Paulis whose x string is k_i xor k_j contribute
function block_matrix(O, kets::Vector{Ket{N}}) where N
    n = length(kets)
    xs = unique([kets[i].v ⊻ kets[j].v for i in 1:n, j in 1:n])
    M = zeros(ComplexF64, n, n)
    for (p, c) in O
        p.x in xs || continue
        for i in 1:n, j in 1:n
            (kets[i].v ⊻ kets[j].v) == p.x || continue
            M[i, j] += c * matrix_element(kets[i]', p, kets[j])
        end
    end
    maximum(abs, imag(M)) < 1e-10 || @warn "complex block matrix" maximum(abs, imag(M))
    R = real(M)
    return (R + R') / 2
end

# swap 4: the change truncation makes to the whole 3x3 block
mutable struct BlockCorrection{N} <: PauliOperators.CorrectionAccumulator
    kets::Vector{Ket{N}}
    accumulated::Matrix{Float64}
end
BlockCorrection(kets::Vector{Ket{N}}) where N = BlockCorrection{N}(kets, zeros(length(kets), length(kets)))
PauliOperators._measure(O::PauliOperators.AnyPauliSum{N}, corr::BlockCorrection{N}) where N = block_matrix(O, corr.kets)
PauliOperators._accumulate!(corr::BlockCorrection, before, after) = (corr.accumulated .+= after .- before; nothing)

# V_S = sum_y <y|O^2|y> - sum_{y,y'} |M_y'y|^2, with <y|O^2|y> = Var_y + M_yy^2
block_variance(O, kets, M) = sum(real(variance(O, k)) + M[i, i]^2 for (i, k) in enumerate(kets)) - sum(abs2, M)

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
                   conv_thresh, grad_thresh=1e-6, e_ref=0.0, verbose=1) where N
    O = deepcopy(Oin)
    corr = BlockCorrection(kets)
    hist = Dict{String,Any}(k => Any[] for k in ("lam", "lam_corr", "VS", "C", "C_corr", "terms", "grad_norm",
                                                  "len_G", "n_rots", "time"))
    generators, angles = PauliBasis{N}[], Float64[]
    function record!(gnorm, lenG, nrot, t)
        M = block_matrix(O, kets)
        push!(hist["lam"], eigvals(Symmetric(M)))
        push!(hist["lam_corr"], eigvals(Symmetric(M - corr.accumulated)))
        push!(hist["VS"], block_variance(O, kets, M))
        push!(hist["C"], tr(M)); push!(hist["C_corr"], tr(M - corr.accumulated))
        push!(hist["terms"], length(O)); push!(hist["grad_norm"], gnorm)
        push!(hist["len_G"], lenG); push!(hist["n_rots"], nrot); push!(hist["time"], t)
        if verbose > 0
            @printf("%5d  λ-E_ref(mHa) %s  corrected %s  V_S %.3e  |g| %.2e  len(O) %8d  len(G) %7d  rots %3d  %7.1f s\n",
                    length(hist["lam"]) - 1, join([@sprintf("%9.3f", 1e3 * (x - e_ref)) for x in hist["lam"][end]], " "),
                    join([@sprintf("%9.3f", 1e3 * (x - e_ref)) for x in hist["lam_corr"][end]], " "),
                    hist["VS"][end], gnorm, length(O), lenG, nrot, t)
            flush(stdout)
        end
    end
    Dop = O isa SparsePauliVector ? SparsePauliVector(D; T=ComplexF64) : D
    record!(0.0, 0, 0, 0.0)
    t0 = time()
    for iter in 1:max_iter
        G = commutator_clipped(Dop, O)
        truncate!(G, gradient_truncation)
        σvs = [matvec(O, k) for k in kets]
        grad_vec, grad_ops = block_gradient(G, σvs, kets, grad_thresh)
        order = sort(collect(eachindex(grad_vec)), by=i -> (-abs(grad_vec[i]), grad_ops[i].z, grad_ops[i].x))
        n_rots = 0
        for gi in order
            Gi = grad_ops[gi]
            θi, _, _ = optimize_theta_block(O, Gi, kets)
            evolve!(O, Gi, θi)
            truncate!(O, operator_truncation, corr)
            push!(generators, Gi); push!(angles, θi)
            n_rots += 1
            n_rots >= max_rots_per_grad && break
        end
        gnorm = norm(grad_vec)
        record!(gnorm, length(grad_vec), n_rots, time() - t0)
        if gnorm < conv_thresh
            verbose < 1 || println(" Converged.")
            break
        end
        if n_rots == 0
            @warn "no search directions found"
            break
        end
    end
    return O, hist, generators, angles, corr
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
    t = @elapsed O, hist, gens, angs, corr = dbf_block(Hop, kets, D;
        operator_truncation=op_trunc, gradient_truncation=grad_trunc,
        max_iter=parse(Int, opts["max-iter"]), max_rots_per_grad=parse(Int, opts["max-rots"]),
        conv_thresh=parse(Float64, opts["conv"]), e_ref=e_fci)
    mkpath(opts["outdir"])
    open(joinpath(opts["outdir"], "$(label).json"), "w") do io
        JSON.print(io, Dict("label" => label, "R" => meta["R_angstrom"], "basis" => basis, "E_FCI" => e_fci,
                            "settings" => opts, "n_rotations" => length(angs), "wall_time_s" => t,
                            "final_terms" => length(O), "accumulated_block" => corr.accumulated,
                            "rots_per_iter" => hist["n_rots"],
                            "generators" => [[Int(g.z), Int(g.x)] for g in gens], "angles" => angs,
                            "history" => hist), 1)
    end
    @printf(" done: %d rotations in %.0f s; wrote %s.json\n", length(angs), t, label)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main(ARGS)
end
