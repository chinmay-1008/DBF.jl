# DMRG reference energies for the open spin-1/2 Heisenberg chain H = J sum_i S_i . S_{i+1}
# (DBF.af_heisenberg(N, J, J, J; periodic=false), i.e. (J/4) sum sigma.sigma), for the block vDBF runs.
#
#   E0      ground state, Sz = 0 sector (a singlet for even N)
#   ET_m1   lowest state with Sz = 1: the lowest triplet, Ms = +1 component
#   ET_m0   first excited state with Sz = 0 (orthogonal to the ground state): the same
#           triplet's Ms = 0 component; it must equal ET_m1, which checks the DMRG
#   E_neel  <Neel|H|Neel> = -J (N-1)/4, a check of the Hamiltonian convention
#   block_weights  overlaps of these states with the block determinants of run_block_heisenberg.jl
#
#     julia --project=. dmrg_heisenberg.jl <N> [maxdim 400] [J 1.0]
using ITensors, ITensorMPS, JSON, Printf

function main(args)
    N = parse(Int, args[1])
    maxdim = length(args) >= 2 ? parse(Int, args[2]) : 400
    J = length(args) >= 3 ? parse(Float64, args[3]) : 1.0
    sites = siteinds("S=1/2", N; conserve_qns=true)
    os = OpSum()
    for i in 1:N-1
        os += J / 2, "S+", i, "S-", i + 1
        os += J / 2, "S-", i, "S+", i + 1
        os += J, "Sz", i, "Sz", i + 1
    end
    H = MPO(os, sites)
    neel = [isodd(i) ? "Up" : "Dn" for i in 1:N]
    nsweeps = 14
    dims = [min(maxdim, d) for d in (20, 40, 80, 160, 240, 320, 400, 400, 400, 400, 400, 400, 400, 400)]
    sw = (nsweeps=nsweeps, maxdim=dims, cutoff=1e-11, noise=[1e-6, 1e-7, 1e-8, 0.0], outputlevel=0)

    psi_neel = MPS(sites, neel)
    e_neel = inner(psi_neel', H, psi_neel)
    t = @elapsed e0, psi0 = dmrg(H, random_mps(sites, neel; linkdims=10); sw...)
    @printf("N = %d  E_neel = %.10f (expected %.10f)\n", N, e_neel, -J * (N - 1) / 4)
    @printf("E0     = %.10f   maxlinkdim %d   (%.0f s)\n", e0, maxlinkdim(psi0), t)

    up1 = copy(neel)
    up1[N ÷ 2 + (iseven(N ÷ 2) ? 0 : 1)] = "Up"              # flip one Dn spin near the centre: Sz = 1
    t = @elapsed e1, psi1 = dmrg(H, random_mps(sites, up1; linkdims=10); sw...)
    @printf("ET_m1  = %.10f   gap %.10f   (%.0f s)\n", e1, e1 - e0, t)

    t = @elapsed e2, psi2 = dmrg(H, [psi0], random_mps(sites, neel; linkdims=10); sw..., weight=20.0)
    @printf("ET_m0  = %.10f   gap %.10f   overlap with ground %.1e   (%.0f s)\n", e2, e2 - e0, abs(inner(psi0, psi2)), t)

    # weights of the exact (DMRG) states on the block determinants of run_block_heisenberg.jl, in this
    # script's convention (Néel = Up on odd sites; centre sites c = N/2 and c + 1):
    #   |0> Néel, |f> centre pair exchanged (Sz = 0), |s> site c flipped Dn -> Up (Sz = +1)
    c = N ÷ 2
    fl = copy(neel); fl[c], fl[c+1] = neel[c+1], neel[c]
    sf = copy(neel); sf[c] = neel[c] == "Up" ? "Dn" : "Up"
    ov(psi, cfg) = inner(MPS(sites, cfg), psi)
    w = Dict{String,Any}()
    for (name, psi) in (("S0", psi0), ("T_m0", psi2))
        a0, af = ov(psi, neel), ov(psi, fl)
        w[name] = Dict("neel" => a0, "flipflop" => af, "span" => abs2(a0) + abs2(af),
                       "plus" => abs2(a0 + af) / 2, "minus" => abs2(a0 - af) / 2)
        @printf("%-5s <Néel|psi> = %+.4e   <f|psi> = %+.4e   weight on {Néel, f} %.3e   on (Néel ± f)/√2: %.3e / %.3e\n",
                name, a0, af, w[name]["span"], w[name]["plus"], w[name]["minus"])
    end
    as = ov(psi1, sf)
    w["T_m1"] = Dict("s" => as, "span" => abs2(as))
    @printf("T_m1  <s|psi> = %+.4e   weight %.3e\n", as, abs2(as))

    mkpath(joinpath(@__DIR__, "results"))
    open(joinpath(@__DIR__, "results", "dmrg_N$(N).json"), "w") do io
        JSON.print(io, Dict("N" => N, "J" => J, "boundary" => "open", "maxdim" => maxdim, "E_neel" => e_neel,
                            "E0" => e0, "ET_m1" => e1, "ET_m0" => e2, "gap_m1" => e1 - e0, "gap_m0" => e2 - e0,
                            "maxlinkdim" => [maxlinkdim(psi0), maxlinkdim(psi1), maxlinkdim(psi2)],
                            "block_weights" => w), 2)
    end
end

main(ARGS)
