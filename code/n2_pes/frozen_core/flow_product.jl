# The double-bracket flow integrated by the product formula (report Eq. 22, as in
# notebooks/hw_vdbf_vs_flow.ipynb) on the 16-qubit frozen-core problems, exactly, in
# sv_dbf.jl's dressed-frame representation, so it is directly comparable with
# sv_dbf.jl (exact vDBF) rotation by rotation.
#
# Generator: the one-body Λ = Σ_i Z_i of vDBF's ranking.  At the start of a step,
# for every direction x (a basis state of the sector reachable from |0>) the
# coupling to the reference is O_x0 = <x|O|0>, and G = [O, Λ] couples |0> and |x>
# with weight (Λ_0 - Λ_x) O_x0 = 2|x| O_x0.  One step rotates about every
# direction in turn (sv_dbf.jl's Pauli for x: X on the flipped qubits, Y on the
# lowest), each by the angle exp(G δs) assigns it, θ_x = -4|x| μ0 O_x0 δs, so to
# first order the energy falls by 4|x| O_x0^2 δs per direction.  Directions with
# |2|x| O_x0| < 1e-6 are skipped, as in the notebook.  A step costs one rotation
# per direction kept.
#
# One difference from the notebook's Pauli-string version: there a direction x can
# appear as several strings of G (same X/Y pattern, different Z) that act alike on
# |0> but differently elsewhere; here each direction is one rotation, as in vDBF.
# That only makes the flow cheaper to count.
#
#     --ds <value>     fixed step
#     --ds line        step chosen by a line search on the objective of the whole
#                      product step: 60-point log grid on (0, --ds-max], golden-section
#                      refinement around the best point, never a step that raises it
#
#     julia -t 8 --project=../../integrals flow_product.jl <point dir> <basis> <label>
#           [--ds 0.02|line] [--ds-max 2] [--s2-penalty 0.05] [--max-rots 300000] [--tol 1e-10]
include(joinpath(@__DIR__, "sv_dbf.jl"))
using SparseArrays

# sign of mu(G, b) relative to mu(G, 0): G has a single Z (on the Y qubit y)
function mu_table(G, x, states, y)
    m0 = μ(G, UInt64(0))
    for b in states[1:min(end, 64)]            # spot-check the closed form
        @assert μ(G, b) ≈ m0 * (b & y == 0 ? 1 : -1)
    end
    return m0
end

# u <- R u for R e_b = c e_b + s mu(b) e_{b⊻x}, mu(b) = m0 (-1)^{b&y}
function rotate_vec!(u, x, y, m0, θ, states, pos)
    c, s = cos(θ / 2), sin(θ / 2)
    hb = UInt64(1) << (63 - leading_zeros(x))
    @inbounds for j in eachindex(states)
        b = states[j]
        b & hb == 0 || continue
        k = pos[(b ⊻ x)+1]
        mj = b & y == 0 ? m0 : -m0
        mk = (b ⊻ x) & y == 0 ? m0 : -m0
        uj, uk = u[j], u[k]
        u[j] = c * uj + s * mk * uk
        u[k] = c * uk + s * mj * uj
    end
    return u
end

function run_flow(args)
    dir, basis, label = args[1:3]
    opts = Dict("ds" => "0.02", "ds-max" => "2", "s2-penalty" => "0", "max-rots" => "300000", "tol" => "1e-10")
    i = 4
    while i <= length(args)
        opts[args[i][3:end]] = args[i+1]; i += 2
    end
    λ = parse(Float64, opts["s2-penalty"])
    maxrot, tol = parse(Int, opts["max-rots"]), parse(Float64, opts["tol"])
    meta = JSON.parsefile(joinpath(dir, "metadata.json"))
    d = load(joinpath(dir, "hamiltonian_$(basis).jld2"))
    ref = UInt64(d["reference_ket"].v)
    e_fci = meta["E_FCI_with_mu"] + meta["constant"]
    Ovl = npzread(joinpath(dir, "$(basis)_ab_overlap.npy"))
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
    S2s = sparse(s2_matrix(states, pos, ref, Ovl))
    Op = λ == 0 ? Hs : Hs + λ * S2s
    flips = [count_ones(b) for b in states]
    gens = [k == 1 ? nothing : generator(states[k], "low") for k in 1:D]
    ys = [k == 1 ? UInt64(0) : states[k] & (~states[k] + 1) for k in 1:D]
    m0s = [k == 1 ? 0.0 : mu_table(gens[k], states[k], states, ys[k]) for k in 1:D]

    W = Matrix{Float64}(I, D, D)
    ψ = W[:, 1]; h = Op * ψ
    Eh(ψ) = dot(ψ, Hs * ψ)
    rotE = [Eh(ψ)]                                # energy of H after every rotation
    rotObj = [dot(ψ, Op * ψ)]                     # and of the objective H + λ S^2 (what vDBF's rotation_energies hold)
    stepinfo = Dict{String,Any}[]
    t0 = time()
    @printf("R = %.2f %s, λ = %g, ds = %s: sector %d\n", meta["R_angstrom"], basis, λ, opts["ds"], D)
    while length(rotE) - 1 < maxrot
        o = W' * h; o[1] = 0
        dirs = [k for k in 2:D if abs(2 * flips[k] * o[k]) > 1e-6]
        isempty(dirs) && break
        sort!(dirs; by=k -> -flips[k] * o[k]^2)
        coef = [-4 * flips[k] * m0s[k] * o[k] for k in dirs]   # θ_k = coef_k δs
        trial(ds) = begin                                        # objective after the whole step
            u = zeros(D); u[1] = 1
            for j in length(dirs):-1:1                           # psi' = W R_1 ... R_n e_0
                k = dirs[j]; rotate_vec!(u, states[k], ys[k], m0s[k], coef[j] * ds, states, pos)
            end
            v = W * u; dot(v, Op * v)
        end
        pred = NaN
        ds = if opts["ds"] == "line"
            # as in the notebook: a grid over (0, ds_max], then a golden-section refinement
            # around the best grid point; never a step that raises the objective
            dmax = parse(Float64, opts["ds-max"])
            grid = dmax .* (10 .^ range(-4, 0, length=60))
            fg = [trial(x) for x in grid]
            ib = argmin(fg)
            a = ib == 1 ? 0.0 : grid[ib-1]; b = ib == length(grid) ? grid[end] : grid[ib+1]
            g = (sqrt(5) - 1) / 2
            c1, c2 = b - g * (b - a), a + g * (b - a); f1, f2 = trial(c1), trial(c2)
            for _ in 1:30
                if f1 < f2
                    b, c2, f2 = c2, c1, f1; c1 = b - g * (b - a); f1 = trial(c1)
                else
                    a, c1, f1 = c1, c2, f2; c2 = a + g * (b - a); f2 = trial(c2)
                end
            end
            x = (a + b) / 2; fx = trial(x)
            fx < fg[ib] || (x = grid[ib]; fx = fg[ib])
            fx < dot(ψ, h) || (println("line search found no lower energy; stopping"); break)
            pred = fx
            x
        else
            parse(Float64, opts["ds"])
        end
        for (j, k) in enumerate(dirs)
            θ = coef[j] * ds
            v = W[:, k]
            rotate_W!(W, gens[k], θ, states, pos, states[k])
            c, s = cos(θ / 2), sin(θ / 2)
            ψ .= c .* ψ .+ s * m0s[k] .* v
            hs = Hs * ψ; eh = dot(ψ, hs)
            push!(rotE, eh); push!(rotObj, λ == 0 ? eh : eh + λ * dot(ψ, S2s * ψ))
        end
        ψ .= view(W, :, 1); h = Op * ψ
        obj = dot(ψ, h)
        isnan(pred) || abs(pred - obj) < 1e-8 || error("line search predicted $pred, step reached $obj")
        push!(stepinfo, Dict("ds" => ds, "n_dirs" => length(dirs), "rotations" => length(rotE) - 1,
                             "E" => Eh(ψ), "obj" => obj, "time" => time() - t0))
        @printf("step %4d  ds %.4f  dirs %5d  rotations %7d  E - FCI %.3e  obj - FCI %.3e  (%.0f s)\n", length(stepinfo), ds,
                length(dirs), length(rotE) - 1, Eh(ψ) - e_fci, obj - e_fci, time() - t0)
        flush(stdout)
        Eh(ψ) - e_fci < tol && break
        if !isfinite(Eh(ψ)) || Eh(ψ) - e_fci > 10
            println("diverged"); break
        end
    end
    open(joinpath(@__DIR__, "sv_results", "$(label).json"), "w") do io
        JSON.print(io, Dict("label" => label, "R" => meta["R_angstrom"], "basis" => basis, "s2_penalty" => λ,
                            "ds" => opts["ds"], "E_FCI" => e_fci, "rotation_energies" => rotE, "rotation_objective" => rotObj, "steps" => stepinfo), 1)
    end
end

run_flow(ARGS)
