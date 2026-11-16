# Why the block sits at the centre of the open chain: where along the chain do the target states live?
# DMRG for S0 (Sz = 0) and the lowest Sz = 1 state (the triplet's Ms = +1 component), then
#   <S^z_i> in the Sz = 1 state, and its smooth part (sums over site pairs 1-2, 3-4, ...),
#   |<f_b|S0>|^2    for the flip-flop determinant on every bond b, b+1 (Néel with that pair exchanged),
#   |<s_i|T+1>|^2   for the single-flip determinant on every Dn site i (Néel with site i flipped Dn -> Up),
# in dmrg_heisenberg.jl's convention (Néel = Up on odd sites).  run_block_heisenberg.jl uses bond N/2, N/2+1
# and the single flip on site N/2.  Writes results/position_N<N>.json; plot_heisenberg.py plots it.
#
#     julia --project=. position_check.jl [N 100] [maxdim 300]
using ITensors, ITensorMPS, JSON, Printf

function main(args)
    N = length(args) >= 1 ? parse(Int, args[1]) : 100
    maxdim = length(args) >= 2 ? parse(Int, args[2]) : 300
    sites = siteinds("S=1/2", N; conserve_qns=true)
    os = OpSum()
    for i in 1:N-1
        os += 0.5, "S+", i, "S-", i + 1
        os += 0.5, "S-", i, "S+", i + 1
        os += 1.0, "Sz", i, "Sz", i + 1
    end
    H = MPO(os, sites)
    neel = [isodd(i) ? "Up" : "Dn" for i in 1:N]
    dims = [min(maxdim, d) for d in (20, 40, 80, 160, 240, 300, 300, 300, 300, 300, 300, 300)]
    sw = (nsweeps=12, maxdim=dims, cutoff=1e-11, noise=[1e-6, 1e-7, 1e-8, 0.0], outputlevel=0)
    e0, psi0 = dmrg(H, random_mps(sites, neel; linkdims=10); sw...)
    up1 = copy(neel)
    up1[N÷2] = "Up"
    e1, psi1 = dmrg(H, random_mps(sites, up1; linkdims=10); sw...)
    @printf("N = %d  E0 = %.8f  E(Sz=1) = %.8f  gap %.5f\n", N, e0, e1, e1 - e0)

    m = expect(psi1, "Sz")
    pair = [m[i] + m[i+1] for i in 1:2:N-1]
    @printf("sum <Sz> = %.6f; smooth part per site pair: ends %.4f / %.4f, centre %.4f, min %.4f\n",
            sum(m), pair[1], pair[end], pair[length(pair)÷2], minimum(pair))

    ov(psi, cfg) = inner(MPS(sites, cfg), psi)
    flipflop = Float64[]
    for b in 1:N-1
        f = copy(neel)
        f[b], f[b+1] = neel[b+1], neel[b]
        push!(flipflop, abs2(ov(psi0, f)))
    end
    single = Union{Float64,Nothing}[]
    for i in 1:N
        if neel[i] == "Dn"
            s = copy(neel)
            s[i] = "Up"
            push!(single, abs2(ov(psi1, s)))
        else
            push!(single, nothing)
        end
    end
    c = N ÷ 2
    @printf("|<Néel|S0>|^2 = %.3e\n", abs2(ov(psi0, neel)))
    @printf("flip-flop on bond %d-%d: %.3e (bond 1-2: %.3e, bulk median %.3e)\n", c, c + 1, flipflop[c], flipflop[1],
            sort(flipflop)[length(flipflop)÷2])
    sv = [x for x in single if x !== nothing]
    imax = argmax([x === nothing ? -1.0 : x for x in single])
    @printf("single flip on site %d: %.3e (best: site %d, %.3e; site 2: %.3e)\n", c, single[c], imax, single[imax], single[2])

    mkpath(joinpath(@__DIR__, "results"))
    open(joinpath(@__DIR__, "results", "position_N$(N).json"), "w") do io
        JSON.print(io, Dict("N" => N, "E0" => e0, "E1" => e1, "Sz" => m, "Sz_pairs" => pair,
                            "neel_S0" => abs2(ov(psi0, neel)), "flipflop_S0" => flipflop, "single_T1" => single,
                            "centre" => c), 1)
    end
end

main(ARGS)
