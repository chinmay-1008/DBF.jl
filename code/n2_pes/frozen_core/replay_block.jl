# Exact replay of a truncated block vDBF run (../run_block_pp.jl): apply its stored
# generators and angles to the block states (the run's "kets", default |0>, |a>, |b>) with no truncation, and report
# the true eigenvalues of the 3x3 block of H and the true block variance V_S at
# the end of each macro-iteration.  For a run on H + λ S^2 the block is that of H + λ S^2,
# and lam_H, var_H, VS_H and S2 give H and S^2 on its eigenvectors.  The block analogue of replay_fc.jl.
# fid = |P_low P_block|^2 / |S| is the fidelity of the block with the |S| lowest levels of the flow operator
# over all symmetry sectors (sv_block.jl's lowest_levels); fid_states is each level's weight in the block.
#
# The run evolves O <- U' O U with U_k = exp(-i θ_k G_k / 2), so after K rotations
# the block columns are U_1 U_2 ... U_K |y>: U_K acts first.  Each checkpoint is
# replayed from the bare block states.  Every generator has an odd number of Y's
# (it comes from [D, O] with D diagonal and O real, or from --rank exact's GF(2) choice), so the rotation is real:
# U|b> = cos(θ/2)|b> + sin(θ/2) μ(b) |b xor x>, μ(b) = μ(0) (-1)^{z.b}.
#
#     julia -t 4 --project=../../integrals replay_block.jl <point dir> <basis> <run label> [every] [output label]
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
    knames = ["|0>", "|a>", "|b>"]
    if haskey(run, "kets")                                  # run_block_pp.jl stores its block (--kets)
        S, knames = UInt64.(run["kets"]), String.(run["ket_names"])
    end
    hx = unique(UInt64(p.x) for p in keys(H))
    V0 = gf2_basis(hx)                                      # symmetry sectors of H
    ovlfile = joinpath(dir, "$(basis)_ab_overlap.npy")
    Ovl = isfile(ovlfile) ? npzread(ovlfile) : nothing
    Ovl === nothing || (Ovl[abs.(Ovl) .< 1e-6] .= 0)       # as in sv_dbf.jl
    # a run on H + λ S^2 (run_block_pp.jl --s2-penalty): replay the same flow operator, and report the
    # H energies, <S^2> and H variances of its block eigenvectors as well
    λs2 = parse(Float64, string(get(run["settings"], "s2-penalty", "0")))
    s2file = joinpath(dir, "s2_$(basis).jld2")              # <S^2> is reported whenever build_s2.jl has run
    (λs2 != 0 || isfile(s2file)) && append!(hx, unique(UInt64(p.x) for p in keys(load(s2file, "S2"))))
    states = enumerate_span(gf2_basis(vcat(hx, S)))
    D = length(states)
    pos = zeros(Int32, 1 << N)
    for (k, b) in enumerate(states)
        pos[b+1] = k
    end
    Hs = sector_matrix(H, states, pos)
    S2s = nothing
    if λs2 != 0 || isfile(s2file)
        S2s = s2_matrix(states, pos, ref, Ovl)
        norm(S2s - S2s') < 1e-10 || error("sector S^2 is not symmetric")
    end
    Os = λs2 == 0 ? Hs : Hs + λs2 * S2s
    @printf("flow operator H + %g S^2%s\n", λs2, S2s === nothing ? "" : "; <S^2> reported")
    glev = lowest_levels(H, ref, Ovl, na, meta["n_beta"], λs2, length(S), V0)
    Φg = embed_levels(glev, states, pos)
    grecs, gbest = report_levels(glev, Φg, λs2, e_fci, S, V0; names=knames)
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
    function apply!(V, k)                                   # V <- U_k V, every block column
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
        V = zeros(D, length(S))
        for (c, y) in enumerate(S)
            V[pos[y+1], c] = 1.0
        end
        for k in K:-1:1
            apply!(V, k)
        end
        HV = Os * V                                             # the flow operator (H, or H + λ S^2)
        M = Symmetric(V' * HV)
        F = eigen(M)
        lam = F.values
        Vmat = HV' * HV - Matrix(M) * Matrix(M)                 # <y|O(1 - Π_S)O|y'>
        var = [dot(F.vectors[:, i], Vmat * F.vectors[:, i]) for i in eachindex(S)]
        vs = sum(abs2, HV - V * Matrix(M))
        orth = maximum(abs, V' * V - I)
        ov = vec(sum(abs2, Φg' * V; dims=2))                   # weight of each lowest level inside the block
        row = Dict("iter" => it, "rotations" => K, "lam" => lam, "var" => var, "VS" => vs,
                   "fid" => sum(ov) / length(S), "fid_states" => ov)
        if S2s !== nothing                                      # H and S^2 on the same eigenvectors of O's block
            HHV = Hs * V
            MH = V' * HHV
            VH = HHV' * HHV - MH * MH
            S2b = V' * (S2s * V)
            φ = F.vectors
            row["lam_H"] = [dot(φ[:, i], MH * φ[:, i]) for i in eachindex(S)]
            row["var_H"] = [dot(φ[:, i], VH * φ[:, i]) for i in eachindex(S)]
            row["S2"] = [dot(φ[:, i], S2b * φ[:, i]) for i in eachindex(S)]
            row["VS_H"] = sum(abs2, HHV - V * MH)
        end
        push!(rows, row)
        @printf("%5d  %6d rot  true λ - E_FCI (mHa) %s  true var %s  V_S %.3e  F_low %.4f  orth %.1e  (%.0f s)\n",
                it, K, join([@sprintf("%10.4f", 1e3 * (l - e_fci)) for l in lam], " "),
                join([@sprintf("%.3e", x) for x in var], " "), vs, row["fid"], orth, time() - t0)
        flush(stdout)
    end
    out = length(args) >= 5 ? args[5] : "$(label)_replay"
    open(joinpath(@__DIR__, "sv_results", "$(out).json"), "w") do io
        JSON.print(io, Dict("label" => label, "E_FCI" => e_fci, "s2_penalty" => λs2, "fid_targets" => grecs, "fid_best" => gbest,
                            "rows" => rows), 1)
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    replay_main(ARGS)
end
