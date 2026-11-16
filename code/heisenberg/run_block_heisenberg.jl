# Block vDBF with Pauli propagation on the open spin-1/2 Heisenberg chain H = J sum S_i.S_{i+1}
# (DBF.af_heisenberg(N, J, J, J; periodic=false)), in the site basis with a Néel reference,
# the way the paper runs lattices.  The loop is dbf_block from ../n2_pes/run_block_pp.jl.
#
# Frame: X on every other site (0-based even bits) maps the Néel state to |0...0>.
# Block (c = N/2, the centre bond c, c+1 with 0-based bits qc = c-1, qc1 = c):
#     |0>                 Néel                                        Sz = 0
#     |f> = bits qc, qc1  the antiparallel centre pair flipped (flip-flop)   Sz = 0
#     |s> = bit qc        one centre spin flipped                     Sz = ±1
# H conserves Sz and every term flips an even number of bits, so {|0>, |f>} and {|s>} sit
# in different sectors: the flow targets S0 and the lowest triplet's Ms = 0 component in the
# first, the same triplet's Ms = ±1 component in the second.  The two triplet estimates
# must agree, which checks the method itself.
# D (top eigenspace exactly the block, checked against every single and double flip):
#     D = sum_{k not in centre} Z_k + a (Z_qc Z_qc1 - Z_qc + Z_qc1)
# the centre term is +a on 00, 11, 10 and -3a on 01.
#
#     julia --project=../n2_pes run_block_heisenberg.jl <N> <label> [--coeff 1e-5] [--grad 1e-6]
#         [--max-iter 200] [--max-rots 50] [--conv 1e-5] [--a 1.0] [--save-every 10]
# The JSON is rewritten every --save-every iterations ("finished": false) and at the end.
include(joinpath(@__DIR__, "..", "n2_pes", "run_block_pp.jl"))   # dbf_block etc.; its main() does not run

function heis_main(args)
    N, label = parse(Int, args[1]), args[2]
    opts = Dict("coeff" => "1e-5", "grad" => "1e-6", "max-iter" => "200", "max-rots" => "50", "conv" => "1e-5",
                "a" => "1.0", "save-every" => "10", "outdir" => joinpath(@__DIR__, "results"))
    i = 3
    while i <= length(args)
        key = args[i][3:end]
        haskey(opts, key) || error("unknown option $(args[i])")
        opts[key] = args[i+1]
        i += 2
    end
    H0 = DBF.af_heisenberg(N, 1.0, 1.0, 1.0; periodic=false)
    neel = reduce(|, (Int128(1) << q for q in 0:2:N-1); init=Int128(0))
    H, ψ0 = to_vacuum_reference(H0, Ket{N}(neel))
    e_neel = real(expectation_value(H, ψ0))
    abs(e_neel + (N - 1) / 4) < 1e-10 || error("Néel energy $e_neel, expected $(-(N - 1) / 4)")
    qc, qc1 = N ÷ 2 - 1, N ÷ 2
    bitk(qs...) = Ket{N}(reduce(|, (Int128(1) << q for q in qs); init=Int128(0)))
    kets = [Ket{N}(0), bitk(qc, qc1), bitk(qc)]
    a = parse(Float64, opts["a"])
    D = PauliSum(N)
    for q in 0:N-1
        q in (qc, qc1) || (D += Pauli(N, Z=[q + 1]))
    end
    D += a * (Pauli(N, Z=[qc + 1, qc1 + 1]) - Pauli(N, Z=[qc + 1]) + Pauli(N, Z=[qc1 + 1]))
    dv(k) = real(expectation_value(D, k))
    dtop = dv(kets[1])
    all(abs(dv(k) - dtop) < 1e-12 for k in kets) || error("block kets differ in D")
    others = [bitk(q) for q in 0:N-1]
    append!(others, [bitk(q1, q2) for q1 in 0:N-1 for q2 in q1+1:N-1])
    filter!(k -> !(k in kets), others)
    maximum(dv, others) < dtop - 1e-9 || error("block is not D's top eigenspace")

    dmrg_file = joinpath(@__DIR__, "results", "dmrg_N$(N).json")
    ref = isfile(dmrg_file) ? JSON.parsefile(dmrg_file) : nothing
    e_ref = ref === nothing ? 0.0 : ref["E0"]
    println("="^100)
    @printf(" Heisenberg chain N = %d (open), %d Pauli terms, Néel energy %.6f; block: Néel, centre flip-flop (bits %d,%d), centre flip (bit %d)\n",
            N, length(H), e_neel, qc, qc1, qc)
    ref === nothing || @printf(" DMRG: E0 = %.8f, lowest triplet %.8f (Ms=+1) / %.8f (Ms=0), gap %.6f\n",
                               ref["E0"], ref["ET_m1"], ref["ET_m0"], ref["gap_m1"])
    @printf(" cutoff %s, gradient cutoff %s, %s rotations per gradient, max %s iterations; λ in mHa above %s\n",
            opts["coeff"], opts["grad"], opts["max-rots"], opts["max-iter"], ref === nothing ? "0" : "DMRG E0")
    println("="^100)
    function save(hist, gens, angs, acc, O, t, finished)
        mkpath(opts["outdir"])
        open(joinpath(opts["outdir"], "$(label).json"), "w") do io
            JSON.print(io, Dict("label" => label, "N" => N, "model" => "af_heisenberg open, J = 1", "settings" => opts,
                                "states" => ["S0 (Sz=0)", "T Ms=0 (Sz=0)", "T Ms=+1 (Sz=1)"], "dmrg" => ref,
                                "E_neel" => e_neel, "n_rotations" => length(angs), "wall_time_s" => t, "finished" => finished,
                                "final_terms" => length(O), "accumulated_M" => acc.M, "accumulated_V" => acc.V,
                                "rots_per_iter" => hist["n_rots"],
                                "generators" => [[string(g.z, base=16), string(g.x, base=16)] for g in gens], "angles" => angs,
                                "history" => hist), 1)
        end
    end
    t0 = time()
    every = parse(Int, opts["save-every"])
    t = @elapsed O, hist, gens, angs, acc = dbf_block(SparsePauliVector(H), kets, D;
        on_iter=(it, h, g, a_, acc_, O_) -> (it % every == 0 && save(h, g, a_, acc_, O_, time() - t0, false)),
        operator_truncation=CoeffTruncation(parse(Float64, opts["coeff"])),
        gradient_truncation=CoeffTruncation(parse(Float64, opts["grad"])),
        max_iter=parse(Int, opts["max-iter"]), max_rots_per_grad=parse(Int, opts["max-rots"]),
        conv_thresh=parse(Float64, opts["conv"]), e_ref=e_ref, groups=[[1, 2], [3]])
    save(hist, gens, angs, acc, O, t, true)
    @printf(" done: %d rotations in %.0f s; wrote %s.json\n", length(angs), t, label)
end

heis_main(ARGS)
