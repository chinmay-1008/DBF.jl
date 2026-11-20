# Schrödinger-side measurements of a vDBF run, for any qubit count up to ~24:
#
#   heisenberg  the true <H> of the circuit after selected iterations (exact
#               replay, dense state over all 2^N basis states), next to the
#               operator size len(H) vDBF carried then (from the run log) and its
#               own estimate E - err
#   circuit     the final circuit applied to |0> with amplitudes |c| < tau dropped
#               after every rotation (sparse state): energy and how many
#               amplitudes were kept (final and peak)
#
# Conventions as in n2_pes/frozen_core/replay_fc.jl: H is taken to the vacuum
# frame (X on the reference's occupied qubits), psi = U_1 ... U_K |0> with
# U_k = exp(-i θ_k G_k / 2), the newest rotation acting first.  A Pauli with bit
# masks (z, x) acts as P|b> = i^{|x&z|} (-1)^{|z&b|} |b ⊻ x>; that formula is
# checked against PauliOperators on every generator before use, and the first
# iteration's replayed energy must match vDBF's E - err to 1e-4.
#
#     julia -t 8 --project=../integrals statevector.jl <point dir> <run stem> <dbf log> <out.json>
#         [--iters 0,1,2,5,10,...] [--taus 1e-2,1e-3,...] [--no-exact] [--lambda 0.05]
# A run stopped before the end (time limit) has no <run stem>.json; its latest
# checkpoint is used, with λ from --lambda.
using PauliOperators, JLD2, JSON, LinearAlgebra, Printf

function vacuum_terms(H, ref)
    terms = Dict{UInt64,Vector{Tuple{UInt64,ComplexF64}}}()      # x => [(z, c i^{|x&z|})]
    for (p, c) in H
        z, x = UInt64(p.z), UInt64(p.x)
        cc = isodd(count_ones(z & ref)) ? -c : c
        push!(get!(terms, x, Tuple{UInt64,ComplexF64}[]), (z, cc * im^(count_ones(x & z) % 4)))
    end
    terms
end

@inline sgn(z, b) = isodd(count_ones(z & b)) ? -1.0 : 1.0

# <psi|H|psi>, dense psi over all 2^N states
function energy_dense(ψ, terms)
    groups = collect(terms)
    chunks = collect(Iterators.partition(0:length(ψ)-1, cld(length(ψ), 4Threads.nthreads())))
    tot = zeros(ComplexF64, length(chunks))
    Threads.@threads for ci in eachindex(chunks)
        acc = zero(ComplexF64)
        for (x, zs) in groups, i in chunks[ci]
            a = ψ[i+1]
            a == 0 && continue
            b = UInt64(i)
            f = zero(ComplexF64)
            for (z, c) in zs
                f += c * sgn(z, b)
            end
            acc += conj(ψ[(b⊻x)+1]) * f * a
        end
        tot[ci] = acc
    end
    real(sum(tot))
end

function energy_sparse(ψ::Dict{UInt64,ComplexF64}, terms)
    tot = zero(ComplexF64)
    for (x, zs) in terms, (b, a) in ψ
        t = get(ψ, b ⊻ x, zero(ComplexF64))
        t == 0 && continue
        f = zero(ComplexF64)
        for (z, c) in zs
            f += c * sgn(z, b)
        end
        tot += conj(t) * f * a
    end
    real(tot)
end

# psi <- exp(-i θ G / 2) psi for the Pauli (z, x, phase i^{|x&z|})
function rotate_dense!(ψ, z, x, θ)
    c, s = cos(θ / 2), sin(θ / 2)
    w = im^(count_ones(x & z) % 4)
    hb = UInt64(1) << (63 - leading_zeros(x))
    Threads.@threads :static for i in 0:length(ψ)-1
        b = UInt64(i)
        b & hb == 0 || continue
        k = b ⊻ x
        a, t = ψ[b+1], ψ[k+1]
        ψ[b+1] = c * a - im * s * w * sgn(z, k) * t        # <b|G|k> = w (-1)^{|z&k|}
        ψ[k+1] = c * t - im * s * w * sgn(z, b) * a
    end
    ψ
end

function rotate_sparse(ψ::Dict{UInt64,ComplexF64}, z, x, θ, tau)
    c, s = cos(θ / 2), sin(θ / 2)
    w = im^(count_ones(x & z) % 4)
    out = Dict{UInt64,ComplexF64}()
    sizehint!(out, 2length(ψ))
    for (b, a) in ψ
        out[b] = get(out, b, zero(ComplexF64)) + c * a
        k = b ⊻ x
        out[k] = get(out, k, zero(ComplexF64)) - im * s * w * sgn(z, b) * a
    end
    tau > 0 && filter!(p -> abs(p.second) >= tau, out)
    out
end

function check_convention(gens, N)
    for G in gens[1:min(end, 200)], b in rand(UInt64(0):UInt64(2)^N-1, 4)
        ph, k = G * Ket{N}(b)
        z, x = UInt64(G.z), UInt64(G.x)
        UInt64(k.v) == b ⊻ x && ph ≈ im^(count_ones(x & z) % 4) * sgn(z, b) || error("Pauli phase convention mismatch")
    end
end

function main(args)
    dir, stem, logf, outf = args[1:4]
    opt(k, d) = (i = findfirst(==("--$k"), args); i === nothing ? d : args[i+1])
    meta = JSON.parsefile(joinpath(dir, "metadata.json"))
    e_fci = meta["E_FCI_with_mu"] + meta["constant"]
    h = load(joinpath(dir, "hamiltonian_uno.jld2"))
    N = 2h["n_orbitals"]
    ref = UInt64(h["reference_ket"].v)
    terms = vacuum_terms(h["H"], ref)
    s2terms = vacuum_terms(load(joinpath(dir, "s2_uno.jld2"), "S2"), ref)
    jf = joinpath(dir, "$stem.json")
    run = isfile(jf) ? JSON.parsefile(jf) : nothing
    o = load(joinpath(dir, "$(stem)_ckpt.jld2"))["out"]
    gens, angs = o["generators"], o["angles"]
    check_convention(gens, N)
    zx = [(UInt64(G.z), UInt64(G.x)) for G in gens]
    ends = [0]
    for e in o["energies_per_grad"][2:end]
        j = findnext(==(e), o["energies"], ends[end] + 1)
        j === nothing && error("cannot place the iteration boundaries")
        push!(ends, j - 1)
    end
    ends[end] == length(gens) || error("iteration boundaries do not cover all rotations")
    niter = length(ends) - 1
    lenH = Dict{Int,Int}(0 => length(h["H"]))
    for l in eachline(logf)
        f = split(l)
        length(f) >= 13 && f[1] == "*" && (lenH[parse(Int, f[2])] = parse(Int, f[9]))
    end
    λ = run === nothing ? parse(Float64, opt("lambda", "0.05")) : parse(Float64, run["settings"]["s2-penalty"])
    Eg, Errg = o["energies_per_grad"], o["accumulated_error_per_grad"]
    @printf("%s %s: N = %d, %d iterations, %d rotations, %d H terms in %d x-groups\n", dir, stem, N, niter,
            length(gens), length(h["H"]), length(terms))

    replay_dense(n) = (ψ = zeros(ComplexF64, 1 << N); ψ[1] = 1; for r in n:-1:1; rotate_dense!(ψ, zx[r]..., angs[r]); end; ψ)
    heis = []
    if !("--no-exact" in args)
        grid = parse.(Int, split(opt("iters", "0,1,2,3,5,7,10,15,20,30,40,50,70,100,130,160,200,250,300"), ","))
        for it in filter(<=(niter), unique(vcat(grid, niter)))
            t = @elapsed (ψ = replay_dense(ends[it+1]); e = energy_dense(ψ, terms))
            if it == 1
                # vDBF's energies are of H + λ S^2: what is left is the first iteration's
                # truncation, which E - err accounts for to first order (replay_fc.jl's check)
                o1 = e + λ * energy_dense(ψ, s2terms)
                abs(o1 - (Eg[2] - Errg[2])) < 1e-4 || error("iteration-1 replay $o1 != DBF $(Eg[2] - Errg[2])")
            end
            push!(heis, Dict("iter" => it, "rotations" => ends[it+1], "len_H" => get(lenH, it, -1),
                             "E_true_err" => e - e_fci, "E_dbf_err" => Eg[it+1] - e_fci,
                             "E_minus_err_err" => Eg[it+1] - Errg[it+1] - e_fci,
                             "nnz" => count(a -> abs2(a) > 1e-20, ψ)))
            @printf("  iter %4d  rot %6d  len(H) %10d  true E-FCI %.3e  (E-err)-FCI %+.3e  nnz %8d  (%.0f s)\n", it,
                    ends[it+1], get(lenH, it, -1), e - e_fci, Eg[it+1] - Errg[it+1] - e_fci, heis[end]["nnz"], t)
            flush(stdout)
        end
    end
    circ = []
    for tau in parse.(Float64, split(opt("taus", "1e-2,3e-3,1e-3,3e-4,1e-4,3e-5,1e-5,1e-6"), ","))
        t = @elapsed begin
            ψ = Dict{UInt64,ComplexF64}(UInt64(0) => 1.0 + 0im)
            peak = 1
            for r in length(gens):-1:1
                ψ = rotate_sparse(ψ, zx[r]..., angs[r], tau)
                peak = max(peak, length(ψ))
            end
            nrm = sqrt(sum(abs2, values(ψ)))
            for k in keys(ψ); ψ[k] /= nrm; end
            e = energy_sparse(ψ, terms)
        end
        push!(circ, Dict("tau" => tau, "E_err" => e - e_fci, "nnz_final" => length(ψ), "nnz_peak" => peak, "norm" => nrm))
        @printf("  circuit tau %.0e  E-FCI %.3e  nnz final %8d  peak %8d  norm %.6f  (%.0f s)\n", tau, e - e_fci,
                length(ψ), peak, nrm, t)
        flush(stdout)
    end
    open(outf, "w") do io
        JSON.print(io, Dict("dir" => dir, "stem" => stem, "N" => N, "E_FCI" => e_fci, "coeff" => run === nothing ? "checkpoint" : run["settings"]["coeff"], "finished" => run !== nothing,
                            "len_H_all" => sort([[k, v] for (k, v) in lenH]), "heisenberg" => heis, "circuit" => circ), 1)
    end
end

main(ARGS)
