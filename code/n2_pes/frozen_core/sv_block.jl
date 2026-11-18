# Exact (untruncated) block DBF for the 16-qubit frozen-core N2 Hamiltonians:
# idea 1c of code/vdbf_excited_states.md.
#
# sv_dbf.jl decouples |0>.  Here the block S = {|0>, |a>, |b>} is decoupled
# instead, with a = HOMO_alpha -> LUMO_alpha and b = HOMO_beta -> LUMO_beta
# (vacuum frame: a flipped occupied bit is a hole, a flipped virtual bit a
# particle).  The flow minimises the block cost
#
#     C_S = sum_{y in S} <y|O|y>,      O = W' H W,
#
# whose minimum is the sum of the |S| lowest eigenvalues (Ky Fan).  The 3x3
# block matrix M_yy' = <y|O|y'> is diagonalised at every iteration; its
# eigenvalues are the ground state and two excited states once V_S = 0.
#
# Same machinery as sv_dbf.jl (included below): W dense on a GF(2) sector, one
# Pauli rotation per direction x at its exact optimal angle.  Differences:
#
# * Sector.  |a> and |b> need not have the reference's spatial symmetry, so the
#   sector is the GF(2) span of H's x strings and of a, b.  H is block diagonal
#   over the cosets of the x-string span (symmetry sectors); rotations across
#   cosets have zero gradient and are never picked.  The target is therefore,
#   per coset, the sum of as many lowest eigenvalues as S has states in it.
# * Angle.  For direction x, W <- W exp(-i θ G/2) gives
#       C_S(θ) = const + B cos θ + Cg sin θ,
#       B = sum_y (O_yy - O_{y^x,y^x})/2,     Cg = sum_y μ(y) O_{y^x,y},
#   summed over y in S with y^x outside S (in-block pairs cancel exactly).
# * Generator.  μ(y) = ±1 is G's sign on |y>, which depends on G's z string
#   through (-1)^{z.y}.  z is chosen by a GF(2) solve so that every term of Cg
#   has the same sign, |Cg| = sum_y |O_{y^x,y}|, with an odd number of Y's so
#   the rotation is real (as in sv_dbf.jl).
# * Ranking (--rank).
#     exact  Cg^2, the exact block gradient squared (generalises 'projector')
#     d1c    |Cg| * sum_y |O_{y^x,y}| (d(y) - d(y^x))/2, with d the eigenvalues
#            of the 1c operator D = sum_rest Z_k + Z_H Z_L + Z_Hb Z_Lb
#            + c (Z_H + Z_Hb - Z_H Z_Hb): the <y^x|[D,O]|y> proxy restricted
#            to the block (generalises 'onebody').  The block is checked to be
#            D's top eigenspace.
#     dz     as d1c, with D built from the block's kets the same way (../zblock.jl): Z on
#            every qubit no ket flips, plus one- and two-qubit Z terms on the qubits the kets
#            flip, chosen so the kets are exactly D's top eigenspace (gap 2).  Works for any
#            block, --kets included; for |0>, |a>, |b> it finds an operator of D1c's shape.
#
# --s2-penalty λ runs the flow on H + λ S^2 (as in sv_dbf.jl): cost, gradient,
# angles and the Ky Fan target are of H + λ S^2, so a spin-S level moves up by
# λ S(S+1).  M, its eigenvalues and V_S^H are always reported for H.
#
# Diagnostics per iteration (all exact): block cost minus its Ky Fan target,
# eigenvalues of M minus the target eigenvalues, block variance
# V_S = sum_y |(1 - Π_S) O e_y|^2 of the flow operator and of H,
# subspace infidelity 1 - |P_target P_block|^2/|S| (targets: the Ky Fan levels of the block's own
# sectors), F_low = |P_low P_block|^2/|S| with P_low the |S| lowest levels of the flow operator over
# ALL symmetry sectors (below 1 at convergence: the block states are the wrong determinants for the
# lowest |S| levels), and <S^2> and the weight outside 10 electrons for each eigenvector of M.
#
# --kets replaces the block {|0>, |a>, |b>} by any determinants: comma-separated, "ref" for |0>,
# otherwise the qubits to flip in the vacuum frame joined by dots (alpha 0-7, beta 8-15; an occupied
# qubit becomes a hole, a virtual one a particle), e.g. --kets ref,4.5,4.7 = |0>, HOMO -> LUMO alpha
# and HOMO -> the other 1pi_g alpha.  Each must keep n_alpha and n_beta.  Only with --rank exact or dz.
#
#     julia -t 8 --project=../../integrals sv_block.jl <point dir> <basis> <label>
#         [--homo n_alpha-1] [--lumo n_alpha] [--rank exact|d1c|dz] [--c 1.0] [--s2-penalty 0] [--kets ...]
#         [--max-rots 50] [--max-iter 200] [--grad-thresh 1e-6]
include(joinpath(@__DIR__, "sv_dbf.jl"))   # sector, S^2, rotate_W!, μ; its main() does not run
include(joinpath(@__DIR__, "..", "zblock.jl"))   # z_block_operator: the Z operator of any block (--rank dz)

function parse_block_cli(args)
    opts = Dict("homo" => "", "lumo" => "", "kets" => "", "rank" => "exact", "c" => "1.0", "s2-penalty" => "0", "max-rots" => "50",
                "max-iter" => "200", "grad-thresh" => "1e-6", "outdir" => joinpath(@__DIR__, "sv_results"))
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
    length(pos) == 3 || error("usage: sv_block.jl <point dir> <basis> <label> [options]")
    return pos..., opts
end

# canonical representative of v modulo the span of `basis` (gf2_basis output)
reduce_span(v, basis) = foldl((u, b) -> min(u, u ⊻ b), basis; init=v)

# z with parity(z & rows[i]) == rhs[i] for every i (GF(2) elimination), or nothing
function gf2_solve(rows, rhs)
    piv = Tuple{UInt64,UInt64,Bool}[]          # (row, pivot bit, rhs); a pivot bit is in no other row
    for (r, t) in zip(rows, rhs)
        for (pr, pb, pt) in piv
            r & pb == 0 || (r ⊻= pr; t ⊻= pt)
        end
        if r == 0
            t && return nothing
            continue
        end
        b = r & (~r + 1)
        for i in eachindex(piv)
            pr, pb, pt = piv[i]
            pr & b == 0 || (piv[i] = (pr ⊻ r, pb, pt ⊻ t))
        end
        push!(piv, (r, b, t))
    end
    return reduce(|, (pb for (_, pb, pt) in piv if pt); init=UInt64(0))
end

# sector_matrix for a list of determinants that is closed under particle number but not under
# every x string: a term that leaves the list must have zero amplitude on it
function number_matrix(terms, states, pos)
    D = length(states)
    groups = Dict{UInt64,Vector{Tuple{PauliBasis{N},ComplexF64}}}()
    for (p, c) in terms
        push!(get!(groups, UInt64(p.x), []), (p, ComplexF64(c)))
    end
    I, J, V = Int[], Int[], Float64[]
    for (x, ps) in groups, (j, b) in enumerate(states)
        val = 0.0im
        for (p, c) in ps
            ph, _ = p * Ket{N}(b)
            val += c * ph
        end
        k = pos[(b ⊻ x)+1]
        if k == 0
            abs(val) < 1e-10 || error("x string $x changes the particle numbers")
            continue
        end
        abs(imag(val)) < 1e-10 || error("complex matrix element $val")
        abs(val) > 1e-14 || continue
        push!(I, k); push!(J, j); push!(V, real(val))
    end
    return sparse(I, J, V, D, D)
end

# The subspace a block of nS states should end up spanning, for the fidelity check of the choice of
# block states: the nS lowest levels of H + λ S^2 among all determinants with na alpha and nb beta
# electrons, in every symmetry sector (coset of V0, the span of H's x strings), not only the sectors
# of the block states, plus any level within 1e-6 Ha of the last.  Each sector is diagonalised on its
# own, so degenerate levels of different sectors stay apart.  Sorted by eigenvalue of H + λ S^2; each
# level carries its determinants and coefficients.  Ovl === nothing: no S^2 (λ must be 0; S2 = NaN).
function lowest_levels(H, ref, Ovl, na, nb, λ, nS, V0; tol=1e-6)
    amask = (UInt64(1) << NORB) - 1
    full = [o ⊻ ref for o in UInt64(0):UInt64((1 << N) - 1)
            if count_ones(o & amask) == na && count_ones(o >> NORB) == nb]
    fpos = zeros(Int32, 1 << N)
    for (k, b) in enumerate(full)
        fpos[b+1] = k
    end
    Hf = number_matrix(H, full, fpos)
    (Ovl === nothing && λ != 0) && error("the S^2 penalty needs the alpha-beta overlap")
    S2f = Ovl === nothing ? spzeros(length(full), length(full)) : s2_matrix(full, fpos, ref, Ovl)
    sec = [reduce_span(b, V0) for b in full]
    all(sec[i] == sec[j] for (i, j) in zip(findnz(S2f)[1:2]...)) || error("S^2 couples symmetry sectors")
    lv = []
    for c in unique(sec)
        ks = findall(==(c), sec)
        Hc, S2c = Hf[ks, ks], S2f[ks, ks]
        F = eigen(Symmetric(Matrix(Hc + λ * S2c)))
        for i in 1:min(nS + 4, length(ks))
            v = F.vectors[:, i]
            push!(lv, (E=dot(v, Hc * v), Eopt=F.values[i], S2=Ovl === nothing ? NaN : dot(v, S2c * v),
                       sector=c, dets=full[ks], v=v))
        end
    end
    sort!(lv; by=l -> l.Eopt)
    return [l for l in lv if l.Eopt < lv[nS].Eopt + tol]
end

# the levels as columns over a sector's determinants; parts outside the sector are dropped, since
# the block, which stays in its sector, has no overlap with them
function embed_levels(levels, states, pos)
    Φ = zeros(length(states), length(levels))
    for (i, l) in enumerate(levels), (b, x) in zip(l.dets, l.v)
        k = pos[b+1]
        k == 0 || (Φ[k, i] = x)
    end
    return Φ
end

# Print the levels of lowest_levels with the symmetry sector each lies in.  The flow never leaves a
# sector, so sector by sector the block can hold at most min(its states there, lowest levels there):
# the sum over sectors / |S| is the best fidelity these block states allow (below 1: the wrong
# determinants for the lowest levels, however long the flow runs).  Returns (JSON records, best).
function report_levels(levels, Φ, λ, e_fci, S, V0; names=nothing)
    nS = length(S)
    names = names !== nothing ? names : nS == 3 ? ["|0>", "|a>", "|b>"] : ["block state $j" for j in 1:nS]
    bsec = [reduce_span(y, V0) for y in S]
    where(c) = c in bsec ? "sector of " * join(names[bsec .== c], ", ") : "a sector with no block state"
    @printf("fidelity target: the %d lowest levels of H + %.3g S^2 in all symmetry sectors (E of H, mHa above FCI)\n",
            nS, λ)
    recs = []
    for (i, l) in enumerate(levels)
        @printf("   %10.3f   <S^2> %6.3f   %s\n", 1e3 * (l.E - e_fci), l.S2, where(l.sector))
        push!(recs, Dict("E" => l.E, "Eopt" => l.Eopt, "S2" => l.S2, "sector" => l.sector,
                         "block_states_in_sector" => count(==(l.sector), bsec)))
    end
    lsec = [l.sector for l in levels]
    best = min(nS, sum(min(count(==(c), bsec), count(==(c), lsec)) for c in unique(bsec))) / nS
    @printf("   best fidelity these block states allow: %.4f%s\n", best,
            best < 0.9 ? "  (below 0.9: the block states are in the wrong symmetry sectors for these levels)" : "")
    return recs, best
end

function block_main(args)
    dir, basis, label, opts = parse_block_cli(args)
    rank = opts["rank"]
    rank in ("exact", "d1c", "dz") || error("--rank must be exact, d1c or dz")
    cpar = parse(Float64, opts["c"])
    max_rots, max_iter = parse(Int, opts["max-rots"]), parse(Int, opts["max-iter"])
    gthresh = parse(Float64, opts["grad-thresh"])
    λ = parse(Float64, opts["s2-penalty"])

    meta = JSON.parsefile(joinpath(dir, "metadata.json"))
    na0, nb0 = meta["n_alpha"], meta["n_beta"]
    homo = isempty(opts["homo"]) ? na0 - 1 : parse(Int, opts["homo"])
    lumo = isempty(opts["lumo"]) ? na0 : parse(Int, opts["lumo"])
    d = load(joinpath(dir, "hamiltonian_$(basis).jld2"))
    ref = UInt64(d["reference_ket"].v)
    nel = na0 + nb0
    e_fci = meta["E_FCI_with_mu"] + meta["constant"]
    Ovl = npzread(joinpath(dir, "$(basis)_ab_overlap.npy"))
    Ovl[abs.(Ovl) .< 1e-6] .= 0                # as in sv_dbf.jl
    H = to_vacuum_reference(d["H"], ref)

    # block: |0>, HOMO->LUMO alpha, HOMO->LUMO beta
    bit(q) = UInt64(1) << q
    qH, qL, qHb, qLb = homo, lumo, NORB + homo, NORB + lumo
    ref & bit(qH) != 0 && ref & bit(qHb) != 0 || error("orbital $homo is not occupied in the reference")
    ref & bit(qL) == 0 && ref & bit(qLb) == 0 || error("orbital $lumo is occupied in the reference")
    S = UInt64[0, bit(qH) | bit(qL), bit(qHb) | bit(qLb)]
    knames = ["|0>", "|a>", "|b>"]
    if !isempty(opts["kets"])                  # any determinants instead (see the header)
        rank in ("exact", "dz") || error("--kets needs --rank exact or dz (the 1c operator is built for |0>, |a>, |b>)")
        S, knames = UInt64[], String[]
        for k in split(opts["kets"], ",")
            y = k == "ref" ? UInt64(0) : reduce(|, (bit(parse(Int, q)) for q in split(k, ".")); init=UInt64(0))
            o = y ⊻ ref
            count_ones(o & (bit(NORB) - 1)) == na0 && count_ones(o >> NORB) == nb0 ||
                error("ket $k changes the number of alpha or beta electrons")
            push!(S, y); push!(knames, k == "ref" ? "|0>" : "|$k>")
        end
        allunique(S) || error("--kets lists a determinant twice")
        @printf("block: %s\n", join(knames, ", "))
    end
    nS = length(S)

    # sector: span of H's x strings and of the block states
    hx = unique(UInt64(p.x) for p in keys(H))
    V0 = gf2_basis(hx)
    states = enumerate_span(gf2_basis(vcat(hx, S)))
    D = length(states)
    pos = zeros(Int32, 1 << N)
    for (k, b) in enumerate(states)
        pos[b+1] = k
    end
    kS = [Int(pos[y+1]) for y in S]
    Hs = sector_matrix(H, states, pos)
    norm(Hs - Hs') < 1e-10 || error("sector H is not symmetric")
    S2s = s2_matrix(states, pos, ref, Ovl)
    norm(S2s - S2s') < 1e-10 || error("sector S^2 is not symmetric")
    Hopt = λ == 0 ? Hs : Hs + λ * S2s          # drives the flow; reported energies are of Hs
    cos_S = [reduce_span(y, V0) for y in S]
    cosets = unique(cos_S)
    @printf("R = %.2f %s: HOMO %d, LUMO %d (qubits %d,%d and %d,%d); sector dim %d, %d of its cosets hold block states; nnz(H) %d, threads %d\n",
            meta["R_angstrom"], basis, homo, lumo, qH, qL, qHb, qLb, D, length(cosets), nnz(Hs), nthreads())
    @printf("flow on H + %.3g S^2; [H, S^2] / |H| = %.1e\n", λ, norm(Hs * S2s - S2s * Hs) / norm(Hs))

    # exact levels per coset, from its (n_alpha, n_beta) blocks
    amask = bit(NORB) - 1
    nab(b) = (count_ones((b ⊻ ref) & amask), count_ones((b ⊻ ref) >> NORB))
    classes = Dict{Tuple{UInt64,Int,Int},Vector{Int}}()
    for (k, b) in enumerate(states)
        c = reduce_span(b, V0)
        c in cosets && push!(get!(classes, (c, nab(b)...), Int[]), k)
    end
    # levels of the flow operator; E is the energy for H, Eopt the eigenvalue of H + λ S^2
    levels = Dict(c => Any[] for c in cosets)
    for ((c, a_, b_), ks) in classes
        F = eigen(Symmetric(Matrix(Hopt[ks, ks])))
        Hb, S2b = Hs[ks, ks], S2s[ks, ks]
        for i in 1:min(8, length(ks))
            v = F.vectors[:, i]
            push!(levels[c], (E=dot(v, Hb * v), Eopt=F.values[i], S2=dot(v, S2b * v), na=a_, nb=b_, ks=ks, v=v))
        end
    end
    targets, edge = Any[], Any[]
    for c in cosets
        sort!(levels[c]; by=l -> l.Eopt)
        nc = count(==(c), cos_S)
        @printf("coset of %s (%d block state%s): lowest levels of H + %.3g S^2; E of H in mHa above FCI\n",
                join(knames[cos_S .== c], ", "), nc, nc == 1 ? "" : "s", λ)
        for (i, l) in enumerate(levels[c][1:min(8, end)])
            @printf("   %s %10.3f   <S^2> %6.3f   (%d,%d)\n", i <= nc ? "*" : " ", 1e3 * (l.E - e_fci), l.S2, l.na, l.nb)
        end
        @printf("   gap above the targets (H + λ S^2): %.3f mHa\n", 1e3 * (levels[c][nc+1].Eopt - levels[c][nc].Eopt))
        append!(targets, levels[c][1:nc])
        # target space for the fidelity: the targets plus any level degenerate with the last one
        append!(edge, [l for l in levels[c] if l.Eopt < levels[c][nc].Eopt + 1e-6])
    end
    UInt64(0) in cosets && (abs(levels[UInt64(0)][1].E - e_fci) < 1e-8 || error("lowest level of the |0> coset is not FCI"))
    E_tgt = [l.E for l in targets]             # coset by coset, ascending within each
    kyfan = sum(l.Eopt for l in targets)
    Φ = zeros(D, length(edge))
    for (i, l) in enumerate(edge)
        Φ[l.ks, i] = l.v
    end
    blk_idx = [findall(==(c), cos_S) for c in cosets]
    # fidelity against the lowest levels over all sectors: below 1 at convergence means the block states
    # sit in the wrong symmetry sectors for the lowest nS levels
    glev = lowest_levels(H, ref, Ovl, na0, nb0, λ, nS, V0)
    Φg = embed_levels(glev, states, pos)
    grecs, gbest = report_levels(glev, Φg, λ, e_fci, S, V0; names=knames)

    # eigenvalues of D on the sector states: the 1c operator (--rank d1c) or the block's own Z operator (dz)
    zq(b, q) = (b >> q) & 1 == 0 ? 1.0 : -1.0
    act = (qH, qL, qHb, qLb)
    if rank == "dz"
        zr = z_block_operator(S, N)
        zr === nothing && error("no one- and two-qubit Z operator has exactly these kets as its top eigenspace")
        zterms, zA = zr
        @printf("D = Z on every other qubit %s\n", z_terms_string(zterms, zA))
        dval = b -> z_value(zterms, b)
    else
        dval = b -> sum(zq(b, q) for q in 0:N-1 if !(q in act)) + zq(b, qH) * zq(b, qL) + zq(b, qHb) * zq(b, qLb) +
                    cpar * (zq(b, qH) + zq(b, qHb) - zq(b, qH) * zq(b, qHb))
    end
    dvec = dval.(states)
    dtop = dvec[kS[1]]
    rank in ("d1c", "dz") && !(all(dvec[kS] .== dtop) && count(dvec .>= dtop - 1e-12) == nS) &&
        error("the block is not the top eigenspace of D (rank $rank)")

    # flow
    W = Matrix{Float64}(I, D, D)
    hS = [Hopt * W[:, k] for k in kS]          # (H + λ S^2) w_y for the block columns
    hist = Dict{String,Any}(k => Any[] for k in ("C", "lam", "var", "VS", "VS_H", "infid", "fid", "fid_states", "S2",
                                                  "leak", "cross", "max_grad", "time"))
    function record!(g, t)
        Wb = W[:, kS]
        Mopt = Wb' * hcat(hS...)
        hH = [Hs * Wb[:, j] for j in 1:nS]
        HH = hcat(hH...)
        M = Wb' * HH
        M = (M + M') / 2
        # variance matrix <y|H(1 - Π_S)H|y'> = A - M², A = <y|H²|y'>; each block
        # eigenstate's energy variance is its quadratic form, and they sum to V_S
        Vmat = HH' * HH - M * M
        # M is block diagonal over cosets; diagonalise per coset so lam pairs with E_tgt
        lam, var, φ = Float64[], Float64[], zeros(D, 0)
        for idx in blk_idx
            F = eigen(Symmetric(M[idx, idx]))
            append!(lam, F.values)
            append!(var, [dot(F.vectors[:, i], Vmat[idx, idx] * F.vectors[:, i]) for i in axes(F.vectors, 2)])
            φ = hcat(φ, Wb[:, idx] * F.vectors)
        end
        push!(hist["var"], var)
        push!(hist["cross"], maximum((abs(M[i, j]) for i in 1:nS, j in 1:nS if cos_S[i] != cos_S[j]); init=0.0))
        push!(hist["C"], tr(Mopt)); push!(hist["lam"], lam)
        push!(hist["VS"], sum(dot(h, h) for h in hS) - sum(abs2, Mopt))
        push!(hist["VS_H"], sum(dot(h, h) for h in hH) - sum(abs2, M))
        push!(hist["infid"], 1 - sum(abs2, Φ' * Wb) / nS)
        ov = vec(sum(abs2, Φg' * Wb; dims=2))       # weight of each lowest level inside the block
        push!(hist["fid"], sum(ov) / nS); push!(hist["fid_states"], ov)
        push!(hist["S2"], [dot(φ[:, i], S2s * φ[:, i]) for i in 1:nS])
        push!(hist["leak"], [leak(φ[:, i], states, ref, nel) for i in 1:nS])
        push!(hist["max_grad"], g); push!(hist["time"], t)
    end
    function report(iter, drift)
        @printf("%5d %+11.4f %s %10.3e %10.3e %10.3e %9.2e %6.4f %s %8.1e %8s %6.1f\n", iter, 1e3 * (hist["C"][end] - kyfan),
                join([@sprintf("%+10.4f", 1e3 * x) for x in hist["lam"][end] .- E_tgt], " "),
                hist["VS"][end], hist["VS_H"][end], hist["max_grad"][end], hist["infid"][end], hist["fid"][end],
                join([@sprintf("%6.3f", s) for s in hist["S2"][end]], " "), maximum(hist["leak"][end]),
                drift === nothing ? "" : @sprintf("%8.1e", drift), hist["time"][end])
        flush(stdout)
    end
    @printf("target: %s mHa above FCI (Ky Fan sum of H + λ S^2: %.10f)\n",
            join([@sprintf("%.3f", 1e3 * (e - e_fci)) for e in E_tgt], ", "), kyfan)
    @printf("%5s %11s %32s %10s %10s %10s %9s %6s %20s %8s %8s %6s\n", "iter", "C-target", "lambda_i - target_i (mHa)",
            "V_S", "V_S^H", "max|grad|", "1-F_S", "F_low", "<S^2> of M's states", "leak", "drift", "time")
    record!(0.0, 0.0)
    report(0, nothing)
    t0 = time()
    for iter in 1:max_iter
        o = [W' * h for h in hS]               # o[j][k] = <states[k]|O|S[j]>
        cand, score = Int[], Float64[]
        gmax = 0.0
        for k in 2:D
            x = states[k]
            g, dw = 0.0, 0.0
            for j in 1:nS
                yx = S[j] ⊻ x
                yx in S && continue
                k2 = pos[yx+1]
                t = abs(o[j][k2])
                g += t
                dw += t * (dvec[kS[j]] - dvec[k2]) / 2
            end
            gmax = max(gmax, g)
            g > gthresh || continue
            push!(cand, k); push!(score, rank == "exact" ? g^2 : g * dw)
        end
        if gmax < gthresh
            println("converged: max |block gradient| = $gmax < $gthresh")
            break
        end
        for k in cand[sortperm(score; rev=true)][1:min(max_rots, length(cand))]
            x = states[k]
            kx = [Int(pos[(y ⊻ x)+1]) for y in S]
            out = [j for j in 1:nS if !((S[j] ⊻ x) in S)]
            V = [W[:, kx[j]] for j in 1:nS]
            Mv = [dot(V[j], hS[j]) for j in 1:nS]        # O_{y^x, y}
            jr = out[1]
            rows, rhs = UInt64[x], Bool[true]           # odd Y count
            for j in out[2:end]                          # all terms of Cg with one sign
                push!(rows, S[j] ⊻ S[jr]); push!(rhs, (Mv[j] < 0) != (Mv[jr] < 0))
            end
            z = gf2_solve(rows, rhs)
            z === nothing && error("no generator with the wanted signs for x = $x")
            G = PauliBasis{N}(Int128(z), Int128(x))
            μs = [μ(G, y) for y in S]
            HV = [Hopt * V[j] for j in 1:nS]
            B = sum((dot(view(W, :, kS[j]), hS[j]) - dot(V[j], HV[j])) / 2 for j in out)
            Cg = sum(μs[j] * Mv[j] for j in out)
            tsum = sum(abs(Mv[j]) for j in out)
            abs(abs(Cg) - tsum) <= 1e-10 * max(1.0, tsum) || error("sign choice failed for x = $x")
            θ = atan(-Cg, -B)
            c, s = cos(θ / 2), sin(θ / 2)
            rotate_W!(W, G, θ, states, pos, x)
            for j in 1:nS
                hS[j] = c .* hS[j] .+ (s * μs[j]) .* HV[j]
            end
        end
        # recompute from scratch to keep the recurrences honest
        drift = 0.0
        for j in 1:nS
            hn = Hopt * W[:, kS[j]]
            drift = max(drift, maximum(abs, hn .- hS[j]))
            hS[j] = hn
        end
        record!(gmax, time() - t0)
        report(iter, drift)
    end
    orth = maximum(abs, W[:, kS]' * W[:, kS] - I)
    @printf("orthogonality of the block columns of W: %.1e; max cross-coset |M_yy'|: %.1e\n", orth, maximum(hist["cross"]))
    lam = hist["lam"][end]
    @printf("block eigenvalues - FCI (mHa): %s\n", join([@sprintf("%.4f", 1e3 * (e - e_fci)) for e in lam], ", "))
    @printf("exact              (mHa): %s\n", join([@sprintf("%.4f", 1e3 * (e - e_fci)) for e in E_tgt], ", "))
    # The flow can settle on a symmetry-protected saddle (e.g. a singlet kept in the block
    # where Ky Fan wants a second triplet component), so also compare each block state with
    # the nearest exact level of its coset and spin.
    S2f, i = hist["S2"][end], 0
    for (ci, c) in enumerate(cosets), _ in blk_idx[ci]
        i += 1
        same = [l for l in levels[c] if abs(l.S2 - S2f[i]) < 0.5]
        if isempty(same)
            @printf("   block state %d: %.4f mHa, <S^2> %.3f: no exact level with that spin (spin-contaminated)\n",
                    i, 1e3 * (lam[i] - e_fci), S2f[i])
        else
            l = same[argmin([abs(l.E - lam[i]) for l in same])]
            @printf("   block state %d: %.4f mHa, <S^2> %.3f; nearest exact level of that spin %.4f mHa: error %+.4f mHa\n",
                    i, 1e3 * (lam[i] - e_fci), S2f[i], 1e3 * (l.E - e_fci), 1e3 * (lam[i] - l.E))
        end
    end

    mkpath(opts["outdir"])
    open(joinpath(opts["outdir"], "$(label).json"), "w") do io
        JSON.print(io, Dict("label" => label, "R" => meta["R_angstrom"], "basis" => basis,
                            "homo" => homo, "lumo" => lumo, "kets" => knames, "rank" => rank, "c" => cpar, "s2_penalty" => λ,
                            "max_rots" => max_rots, "E_FCI" => e_fci, "sector_dim" => D,
                            "targets" => [Dict("E" => l.E, "Eopt" => l.Eopt, "S2" => l.S2, "n_alpha" => l.na, "n_beta" => l.nb)
                                          for l in targets],
                            "levels" => Dict(string(c) => [Dict("E" => l.E, "Eopt" => l.Eopt, "S2" => l.S2,
                                                                "n_alpha" => l.na, "n_beta" => l.nb)
                                                           for l in levels[c]] for c in cosets),
                            "fid_targets" => grecs, "fid_best" => gbest, "history" => hist, "orthogonality" => orth), 1)
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    block_main(ARGS)
end
