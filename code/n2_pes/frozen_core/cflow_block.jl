# The full (continuous) double-bracket flow for the 1c block, exactly, as the
# reference for the greedy block vDBF of sv_block.jl:
#
#     dO/ds = [G, O],   G = [O, Λ],   O(0) = H + λ S^2,   Λ diagonal,
#
# which drives O to commute with Λ and puts the lowest energies on the largest
# values of Λ (Brockett).  Two choices of Λ:
#
#   projector  Λ = Π_S, the projector onto the block {|0>, |a>, |b>}.  This is the
#              gradient flow of the block cost C_S = Tr(Π_S O), the continuous
#              counterpart of sv_block.jl --rank exact.  It has a closed form: the
#              block columns V = W[:, S] obey dV/ds = -(1 - V V') H V, so
#              span V(s) = span e^{-sH} V(0) (subspace imaginary-time evolution);
#              computed from the eigenvectors of H, no integration.
#   d1c        Λ = D_1c = sum_rest Z_k + Z_H Z_L + Z_Hb Z_Lb + c (Z_H + Z_Hb - Z_H Z_Hb),
#              the plan's operator, counterpart of sv_block.jl --rank d1c.  Integrated
#              with the exponential-midpoint scheme of cflow_onebody.jl (unitary,
#              second order, step size adapted by step doubling to tol).
#
# H and Λ conserve N_alpha, N_beta and the symmetry coset, so the flow stays in the
# (n_alpha, n_beta) block of each coset that holds block states: the coset of |0>
# (block state |0>) and the coset of |a> (block states |a>, |b>).  The two flow
# independently; --coset 0 or --coset a integrates one of them (so the d1c run can
# use two processes), --coset all does both, and --merge <label> joins the two
# per-coset outputs.  All runs record at the same flow times s, so the merge is
# row by row.
#
# Recorded at each s, in the format of sv_block.jl's history: the eigenvalues of
# the block matrix of H (per coset), the block variance V_S of the flow operator
# and of H, and <S^2> of each block eigenvector.
#
#     julia -t 8 --project=../../integrals cflow_block.jl <point dir> <basis> <label>
#         [--gen projector|d1c] [--c 1.0] [--s2-penalty 0] [--coset all|0|a]
#         [--s-max 60] [--tol 1e-9] [--ds 1e-4] [--vs-stop 1e-13]
#     julia --project=../../integrals cflow_block.jl --merge <label>
include(joinpath(@__DIR__, "sv_block.jl"))   # sector, cosets, S^2; its block_main() does not run

const S_GRID = [0.0; [1e-4 * 10^(k / 40) for k in 0:400]]   # shared record times, 1e-4 .. 1e6

function parse_flow_cli(args)
    opts = Dict("gen" => "projector", "c" => "1.0", "s2-penalty" => "0", "coset" => "all", "s-max" => "60",
                "tol" => "1e-9", "ds" => "1e-4", "vs-stop" => "1e-13", "outdir" => joinpath(@__DIR__, "sv_results"))
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
    length(pos) == 3 || error("usage: cflow_block.jl <point dir> <basis> <label> [options]")
    return pos..., opts
end

# one coset's (n_alpha, n_beta) block: H, S^2, Λ, local indices of the block states
function coset_problem(dir, basis, c_par, gen)
    meta = JSON.parsefile(joinpath(dir, "metadata.json"))
    na0, nb0 = meta["n_alpha"], meta["n_beta"]
    homo, lumo = na0 - 1, na0
    d = load(joinpath(dir, "hamiltonian_$(basis).jld2"))
    ref = UInt64(d["reference_ket"].v)
    Ovl = npzread(joinpath(dir, "$(basis)_ab_overlap.npy"))
    Ovl[abs.(Ovl) .< 1e-6] .= 0
    H = to_vacuum_reference(d["H"], ref)
    bit(q) = UInt64(1) << q
    qH, qL, qHb, qLb = homo, lumo, NORB + homo, NORB + lumo
    S = UInt64[0, bit(qH) | bit(qL), bit(qHb) | bit(qLb)]
    hx = unique(UInt64(p.x) for p in keys(H))
    V0 = gf2_basis(hx)
    states = enumerate_span(gf2_basis(vcat(hx, S)))
    pos = zeros(Int32, 1 << N)
    for (k, b) in enumerate(states)
        pos[b+1] = k
    end
    Hs = sector_matrix(H, states, pos)
    S2s = s2_matrix(states, pos, ref, Ovl)
    amask = bit(NORB) - 1
    nab(b) = (count_ones((b ⊻ ref) & amask), count_ones((b ⊻ ref) >> NORB))
    zq(b, q) = (b >> q) & 1 == 0 ? 1.0 : -1.0
    act = (qH, qL, qHb, qLb)
    d1c(b) = sum(zq(b, q) for q in 0:N-1 if !(q in act)) + zq(b, qH) * zq(b, qL) + zq(b, qHb) * zq(b, qLb) +
             c_par * (zq(b, qH) + zq(b, qHb) - zq(b, qH) * zq(b, qHb))
    probs = Dict{String,Any}()
    for (name, y) in (("0", S[1]), ("a", S[2]))
        c = reduce_span(y, V0)
        ks = [k for (k, b) in enumerate(states) if reduce_span(b, V0) == c && nab(b) == (na0, nb0)]
        loc = [findfirst(==(Int(pos[s+1])), ks) for s in S if reduce_span(s, V0) == c]
        Λ = gen == "projector" ? [Float64(k in loc) for k in eachindex(ks)] : [d1c(states[k]) for k in ks]
        probs[name] = (H=Matrix(Hs[ks, ks]), S2=Matrix(S2s[ks, ks]), Λ=Λ, B=loc)
    end
    e_fci = meta["E_FCI_with_mu"] + meta["constant"]
    return probs, e_fci, meta
end

# block quantities for block columns V of one coset (H is the physical Hamiltonian)
function block_record(V, Hb, Sb, Hopt)
    HV, OV = Hb * V, Hopt * V
    M = Symmetric(V' * HV)
    F = eigen(M)
    φ = V * F.vectors
    Mo = V' * OV
    # V_S = |(1 - V V') O V|^2, from the residual itself so it stays >= 0 near convergence
    return (lam=F.values, S2=[dot(φ[:, i], Sb * φ[:, i]) for i in axes(φ, 2)],
            VS=sum(abs2, OV - V * Mo), VS_H=sum(abs2, HV - V * Matrix(M)), C=tr(Mo))
end

function levels_of(Hopt, Hb, Sb, e_fci)
    F = eigen(Symmetric(Hopt))
    [Dict("E" => dot(F.vectors[:, i], Hb * F.vectors[:, i]), "Eopt" => F.values[i],
          "S2" => dot(F.vectors[:, i], Sb * F.vectors[:, i]), "n_alpha" => 5, "n_beta" => 5) for i in 1:min(8, length(F.values))]
end

# projector flow, closed form: span V(s) = span e^{-s Hopt} V(0)
function flow_projector(p, λ, smax)
    Hopt = λ == 0 ? p.H : p.H + λ * p.S2
    F = eigen(Symmetric(Hopt))
    C0 = F.vectors[p.B, :]'                          # eigen-coefficients of the block states
    rows = []
    for s in S_GRID
        s > smax && break
        Q = Matrix(qr(exp.(-s .* (F.values .- F.values[1])) .* C0).Q)
        push!(rows, merge((s=s,), block_record(F.vectors * Q, p.H, p.S2, Hopt)))
        rows[end].VS < 1e-20 && break
    end
    return rows
end

# d1c (or any diagonal Λ) flow, integrated: exponential midpoint with step doubling
function flow_integrate(p, λ, smax, tol, ds0, vs_stop)
    Hopt = λ == 0 ? copy(p.H) : p.H + λ * p.S2
    O = copy(Hopt)
    n = size(O, 1)
    W = Matrix{Float64}(I, n, n)
    Λ = p.Λ
    G(O) = O .* (Λ' .- Λ)                            # [O, Λ]_jk = O_jk (Λ_k - Λ_j)
    step(O, h) = (Eh = exp(0.5 * h * G(O)); Oh = Eh * O * Eh'; E = exp(h * G(Oh)); (E * O * E', E))
    rows = [merge((s=0.0,), block_record(W[:, p.B], p.H, p.S2, Hopt))]
    s, ds, nacc, nrej, t0 = 0.0, ds0, 0, 0, time()
    for target in S_GRID[2:end]
        target > smax && break
        while s < target - 1e-15
            h = min(ds, target - s)
            O1, _ = step(O, h)
            Oa, Ea = step(O, h / 2); O2, Eb = step(Oa, h / 2)
            err = maximum(abs, O1 - O2) / 3
            if err > tol
                ds = h * max(0.2, 0.9 * (tol / err)^(1 / 3)); nrej += 1
                continue
            end
            O = (O2 + O2') / 2; W = W * (Eb * Ea)'; s += h; nacc += 1
            h == ds && (ds = h * min(2.0, 0.9 * (tol / max(err, 1e-300))^(1 / 3)))
        end
        r = merge((s=s,), block_record(W[:, p.B], p.H, p.S2, Hopt))
        push!(rows, r)
        @printf("s %10.5f  ds %.2e  V_S %.3e  lam %s  steps %d (+%d)  (%.0f s)\n", s, ds, r.VS,
                join([@sprintf("%.6f", x) for x in r.lam], " "), nacc, nrej, time() - t0)
        flush(stdout)
        r.VS < vs_stop && break
    end
    return rows
end

function write_out(path, label, meta, e_fci, gen, λ, cpar, parts)
    # parts: ordered [(coset name, rows, levels)]; rows of all parts share s.  A part that
    # stopped early (converged) is padded with its last row up to the longest part.
    n = maximum(length(r) for (_, r, _) in parts)
    longest = parts[argmax([length(r) for (_, r, _) in parts])][2]
    at(r, i) = r[min(i, length(r))]
    hist = Dict{String,Any}("s" => [], "lam" => [], "S2" => [], "VS" => [], "VS_H" => [], "C" => [])
    for i in 1:n
        push!(hist["s"], longest[i].s)
        push!(hist["lam"], vcat([at(r, i).lam for (_, r, _) in parts]...))
        push!(hist["S2"], vcat([at(r, i).S2 for (_, r, _) in parts]...))
        for k in ("VS", "VS_H", "C")
            push!(hist[k], sum(getproperty(at(r, i), Symbol(k)) for (_, r, _) in parts))
        end
    end
    targets = []
    for (name, _, lv) in parts
        nb = name == "0" ? 1 : 2
        append!(targets, sort(lv; by=l -> l["Eopt"])[1:nb])
    end
    open(path, "w") do io
        JSON.print(io, Dict("label" => label, "R" => meta["R_angstrom"], "generator" => gen, "s2_penalty" => λ, "c" => cpar,
                            "E_FCI" => e_fci, "targets" => targets,
                            "levels" => Dict(name => lv for (name, _, lv) in parts), "history" => hist), 1)
    end
end

function flow_main(args)
    if args[1] == "--merge"
        label = args[2]
        dir = joinpath(@__DIR__, "sv_results")
        a, b = (JSON.parsefile(joinpath(dir, "$(label)_coset$(c).json")) for c in ("0", "a"))
        to_rows(d) = [(s=d["history"]["s"][i], lam=d["history"]["lam"][i], S2=d["history"]["S2"][i],
                       VS=d["history"]["VS"][i], VS_H=d["history"]["VS_H"][i], C=d["history"]["C"][i])
                      for i in eachindex(d["history"]["s"])]
        meta = Dict("R_angstrom" => a["R"])
        write_out(joinpath(dir, "$(label).json"), label, meta, a["E_FCI"], a["generator"], a["s2_penalty"], a["c"],
                  [("0", to_rows(a), a["levels"]["0"]), ("a", to_rows(b), b["levels"]["a"])])
        println("merged -> ", label, ".json")
        return
    end
    dir, basis, label, opts = parse_flow_cli(args)
    gen = opts["gen"]
    gen in ("projector", "d1c") || error("--gen must be projector or d1c")
    λ, cpar = parse(Float64, opts["s2-penalty"]), parse(Float64, opts["c"])
    smax, tol, ds0, vs_stop = (parse(Float64, opts[k]) for k in ("s-max", "tol", "ds", "vs-stop"))
    which = opts["coset"] == "all" ? ["0", "a"] : [opts["coset"]]
    probs, e_fci, meta = coset_problem(dir, basis, cpar, gen)
    parts = []
    for name in which
        p = probs[name]
        @printf("R = %.2f %s, generator %s, coset of |%s>: dim %d, block states %s, Λ on them %s\n", meta["R_angstrom"], basis,
                gen, name, size(p.H, 1), p.B, p.Λ[p.B])
        rows = gen == "projector" ? flow_projector(p, λ, smax) : flow_integrate(p, λ, smax, tol, ds0, vs_stop)
        Hopt = λ == 0 ? p.H : p.H + λ * p.S2
        push!(parts, (name, rows, levels_of(Hopt, p.H, p.S2, e_fci)))
        r = rows[end]
        @printf("   end: s %.4g, V_S %.2e, block eigenvalues - FCI (mHa) %s, <S^2> %s\n", r.s, r.VS,
                join([@sprintf("%.4f", 1e3 * (x - e_fci)) for x in r.lam], ", "), join([@sprintf("%.3f", x) for x in r.S2], ", "))
    end
    mkpath(opts["outdir"])
    out = length(which) == 2 ? "$(label).json" : "$(label)_coset$(which[1]).json"
    write_out(joinpath(opts["outdir"], out), label, meta, e_fci, gen, λ, cpar, parts)
    println("wrote ", out)
end

if abspath(PROGRAM_FILE) == @__FILE__
    flow_main(ARGS)
end
