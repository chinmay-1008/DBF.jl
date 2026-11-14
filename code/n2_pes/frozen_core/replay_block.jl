# Exact replay of a truncated block vDBF run (../run_block_pp.jl): apply its stored
# generators and angles to the three block states with no truncation, and report
# the true eigenvalues of the 3x3 block of H and the true block variance V_S at
# the end of each macro-iteration.  The block analogue of replay_fc.jl.
#
# The run evolves O <- U' O U with U_k = exp(-i θ_k G_k / 2), so after K rotations
# the block columns are U_1 U_2 ... U_K |y>: U_K acts first.  Each checkpoint is
# replayed from the bare block states.  Every generator has an odd number of Y's
# (it comes from [D, O] with D diagonal and O real), so the rotation is real:
# U|b> = cos(θ/2)|b> + sin(θ/2) μ(b) |b xor x>, μ(b) = μ(0) (-1)^{z.b}.
#
#     julia -t 4 --project=../../integrals replay_block.jl <point dir> <basis> <run label> [every]
#     e.g. replay_block.jl data_unocore/R_1.10 uno uno_1.10_blockpp_c1e-5 5
include(joinpath(@__DIR__, "sv_block.jl"))   # sector machinery; block_main() does not run

function replay_main(args)
    dir, basis, label = args[1:3]
    every = length(args) >= 4 ? parse(Int, args[4]) : 1
    run = JSON.parsefile(joinpath(@__DIR__, "sv_results", "$(label).json"))
    meta = JSON.parsefile(joinpath(dir, "metadata.json"))
    e_fci = meta["E_FCI_with_mu"] + meta["constant"]
    d = load(joinpath(dir, "hamiltonian_$(basis).jld2"))
    ref = UInt64(d["reference_ket"].v)
    H = to_vacuum_reference(d["H"], ref)
    na = meta["n_alpha"]
    bit(q) = UInt64(1) << q
    S = UInt64[0, bit(na - 1) | bit(na), bit(NORB + na - 1) | bit(NORB + na)]
    hx = unique(UInt64(p.x) for p in keys(H))
    states = enumerate_span(gf2_basis(vcat(hx, S)))
    D = length(states)
    pos = zeros(Int32, 1 << N)
    for (k, b) in enumerate(states)
        pos[b+1] = k
    end
    Hs = sector_matrix(H, states, pos)
    gens = [PauliBasis{N}(Int128(g[1]), Int128(g[2])) for g in run["generators"]]
    angs = Float64.(run["angles"])
    nper = Int.(run["rots_per_iter"])                       # entry 1 is iteration 0 (no rotations)
    ends = cumsum(nper)                                     # rotations applied by the end of each iteration
    # per generator: x, μ(0), and the index map b -> b xor x on the sector
    xs = [UInt64(g.x) for g in gens]
    zs = [UInt64(g.z) for g in gens]
    μ0 = [μ(g, UInt64(0)) for g in gens]
    for (g, z, m) in zip(gens[1:min(end, 20)], zs, μ0)       # check μ(b) = μ(0)(-1)^{z.b} on a few states
        for b in states[1:min(end, 64)]
            abs(μ(g, b) - m * (isodd(count_ones(z & b)) ? -1 : 1)) < 1e-12 || error("phase rule fails")
        end
    end
    @printf("%s: %d rotations over %d iterations, sector %d; replaying every %d iteration(s)\n",
            label, length(gens), length(nper) - 1, D, every)
    flip = Vector{Int32}(undef, D)
    sgn = Vector{Float64}(undef, D)
    function apply!(V, k)                                   # V <- U_k V, all three columns
        x, z, m = xs[k], zs[k], μ0[k]
        c, s = cos(angs[k] / 2), sin(angs[k] / 2)
        @inbounds for (j, b) in enumerate(states)
            flip[j] = pos[(b ⊻ x)+1]
            sgn[j] = m * (isodd(count_ones(z & b)) ? -1.0 : 1.0)    # μ(b)
        end
        W = similar(V)
        @inbounds for col in axes(V, 2), j in 1:D
            jb = flip[j]                                    # (U v)_j = c v_j + s μ(b_j xor x) v_{j xor x}
            W[j, col] = c * V[j, col] + s * sgn[jb] * V[jb, col]
        end
        V .= W
    end
    rows = []
    its = [i for i in 0:length(nper)-1 if i % every == 0 || i == length(nper) - 1]
    t0 = time()
    for it in its
        K = ends[it+1]
        V = zeros(D, 3)
        for (c, y) in enumerate(S)
            V[pos[y+1], c] = 1.0
        end
        for k in K:-1:1
            apply!(V, k)
        end
        HV = Hs * V
        M = Symmetric(V' * HV)
        lam = eigvals(M)
        vs = sum(abs2, HV - V * Matrix(M))
        orth = maximum(abs, V' * V - I)
        push!(rows, Dict("iter" => it, "rotations" => K, "lam" => lam, "VS" => vs))
        @printf("%5d  %6d rot  true λ - E_FCI (mHa) %s  true V_S %.3e  orth %.1e  (%.0f s)\n", it, K,
                join([@sprintf("%10.4f", 1e3 * (l - e_fci)) for l in lam], " "), vs, orth, time() - t0)
        flush(stdout)
    end
    open(joinpath(@__DIR__, "sv_results", "$(label)_replay.json"), "w") do io
        JSON.print(io, Dict("label" => label, "E_FCI" => e_fci, "rows" => rows), 1)
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    replay_main(ARGS)
end
