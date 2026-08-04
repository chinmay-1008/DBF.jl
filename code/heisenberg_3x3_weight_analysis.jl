using DBF
using PauliOperators
using Printf

# Change these two values to control the analysis.
const ANALYZE_WEIGHTS = true
const WEIGHT_ANALYSIS_INTERVAL = :iteration  # use :rotation for every rotation

function run_3x3_heisenberg(;
        analyze_weights::Bool=ANALYZE_WEIGHTS,
        weight_analysis_interval::Symbol=WEIGHT_ANALYSIS_INTERVAL)
    Lx, Ly = 3, 3
    N = Lx * Ly

    # Antiferromagnetic H = (1/4) Σ_<ij> (XᵢXⱼ + YᵢYⱼ + ZᵢZⱼ)
    # on an open 3×3 square lattice with snake-like qubit ordering.
    H = DBF.heisenberg_2D_zigzag(
        Lx, Ly, -1 / 8, -1 / 8, -1 / 8; periodic=false)

    # Rotate the Néel reference to |000...0>, as expected by this DBF setup.
    generators, angles = DBF.get_1d_neel_state_sequence(N)
    for (generator, angle) in zip(generators, angles)
        H = evolve(H, generator, angle)
    end
    ψ = Ket{N}(0)

    output_dir = joinpath(@__DIR__, "results", "heisenberg_3x3_weight_analysis")
    mkpath(output_dir)
    csv_file = analyze_weights ? joinpath(
        output_dir, "weight_history_$(weight_analysis_interval).csv") : nothing
    checkpoint_file = joinpath(output_dir, "dbf_run_$(weight_analysis_interval)")

    @printf("3×3 Heisenberg reference energy: %.10f\n", expectation_value(H, ψ))
    println("Weight analysis enabled: ", analyze_weights)
    analyze_weights && println("Sampling interval: ", weight_analysis_interval)

    result = DBF.dbf_groundstate(
        SparsePauliVector(H), ψ;
        max_iter=50,
        max_rots_per_grad=100,
        conv_thresh=1e-4,
        operator_truncation=CoeffTruncation(1e-3),
        gradient_truncation=CoeffTruncation(1e-6),
        compute_var_error=false,
        analyze_weights,
        weight_analysis_interval,
        weight_analysis_file=csv_file,
        checkfile=checkpoint_file,
        checkpoint_interval=0,
        checkpoint_mode=:compact,
    )

    if analyze_weights
        history = result["weight_analysis"]
        println("Stored samples: ", length(history["iterations"]))
        println("Python-ready CSV: ", history["csv_file"])
    end
    println("Compact DBF result: ", checkpoint_file * ".jld2")
    return result
end

result = run_3x3_heisenberg()
