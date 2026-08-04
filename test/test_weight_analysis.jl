using DBF
using PauliOperators
using LinearAlgebra
using Test

@testset "weight distributions" begin
    N = 3
    H = PauliSum(N)
    H += 1.0 * Pauli(N)
    H += 2.0 * Pauli(N, X=[1])
    H += 3.0 * Pauli(N, Z=[1, 2])
    H += 4.0 * Pauli(N, Y=[1, 2, 3])

    snapshot = weight_distribution(H)
    @test snapshot.pauli_counts == get_weight_counts(H)
    @test snapshot.majorana_counts == get_majorana_weight_counts(H)
    @test snapshot.pauli_l2_squared ≈ get_weight_probs(H)
    @test snapshot.majorana_l2_squared ≈ get_majorana_weight_probs(H)
    @test snapshot.pauli_l2_norms .^ 2 ≈ get_weight_probs(H)
    @test snapshot.majorana_l2_norms .^ 2 ≈ get_majorana_weight_probs(H)
    @test sum(snapshot.pauli_count_percent) ≈ 100
    @test sum(snapshot.majorana_count_percent) ≈ 100
    @test sum(snapshot.pauli_l2_percent) ≈ 100
    @test sum(snapshot.majorana_l2_percent) ≈ 100
    @test snapshot.pauli_count_percent ≈ 100 .* get_weight_counts(H) ./ length(H)
    @test snapshot.pauli_l2_percent ≈ 100 .* get_weight_probs(H) ./ norm(H)^2

    spv_snapshot = weight_distribution(SparsePauliVector(H))
    @test spv_snapshot.pauli_count_percent == snapshot.pauli_count_percent
    @test spv_snapshot.majorana_count_percent == snapshot.majorana_count_percent
    @test spv_snapshot.pauli_l2_percent ≈ snapshot.pauli_l2_percent
    @test spv_snapshot.majorana_l2_percent ≈ snapshot.majorana_l2_percent

    empty_snapshot = weight_distribution(PauliSum(3))
    @test iszero(sum(empty_snapshot.pauli_count_percent))
    @test iszero(sum(empty_snapshot.majorana_l2_percent))
end

@testset "ground-state weight history" begin
    N = 3
    H = DBF.heisenberg_1D(N, 1.0, 2.0, 3.0; z=0.1)
    ψ = Ket{N}(0)
    kwargs = (
        max_iter=2,
        max_rots_per_grad=2,
        verbose=0,
        operator_truncation=CoeffTruncation(1e-8),
        gradient_truncation=CoeffTruncation(1e-8),
        energy_lowering_thresh=1e-10,
        analyze_weights=true,
    )

    per_rotation = dbf_groundstate(H, ψ;
        weight_analysis_interval=:rotation, kwargs...)
    rotation_history = per_rotation["weight_analysis"]
    @test rotation_history["interval"] === :rotation
    @test rotation_history["rotations"] == collect(0:length(per_rotation["angles"]))
    @test length(rotation_history["iterations"]) == length(per_rotation["angles"]) + 1
    @test all(x -> isapprox(sum(x), 100), rotation_history["pauli_count_percent"])
    @test all(x -> isapprox(sum(x), 100), rotation_history["pauli_l2_percent"])
    final_snapshot = weight_distribution(per_rotation["hamiltonian"])
    @test rotation_history["majorana_count_percent"][end] ≈
          final_snapshot.majorana_count_percent
    @test rotation_history["majorana_l2_percent"][end] ≈
          final_snapshot.majorana_l2_percent

    csv_file = joinpath(mktempdir(), "weight_history")
    per_iteration = dbf_groundstate(H, ψ;
        weight_analysis_interval=:iteration,
        weight_analysis_file=csv_file,
        kwargs...)
    iteration_history = per_iteration["weight_analysis"]
    @test iteration_history["interval"] === :iteration
    @test iteration_history["iterations"] ==
          collect(0:length(per_iteration["energies_per_grad"]) - 1)
    @test length(iteration_history["rotations"]) ==
          length(per_iteration["energies_per_grad"])
    @test iteration_history["pauli_counts"][end] ==
          weight_distribution(per_iteration["hamiltonian"]).pauli_counts
    @test iteration_history["majorana_l2_squared"][end] ≈
          weight_distribution(per_iteration["hamiltonian"]).majorana_l2_squared

    csv_path = csv_file * ".csv"
    @test iteration_history["csv_file"] == csv_path
    @test isfile(csv_path)
    csv_lines = readlines(csv_path)
    @test startswith(csv_lines[1],
        "sample_index,iteration,rotation,interval,n_qubits")
    @test length(csv_lines) == 1 + length(iteration_history["iterations"]) * (3N + 2)
    @test occursin(",pauli,", csv_lines[2])
    @test any(line -> occursin(",majorana,", line), csv_lines)

    without_analysis = dbf_groundstate(H, ψ;
        max_iter=1, max_rots_per_grad=1, verbose=0,
        operator_truncation=CoeffTruncation(1e-8),
        gradient_truncation=CoeffTruncation(1e-8),
        analyze_weights=false)
    @test !haskey(without_analysis, "weight_analysis")

    @test_throws ArgumentError dbf_groundstate(H, ψ;
        max_iter=0, verbose=0, weight_analysis_interval=:bad)
    @test_throws ArgumentError dbf_groundstate(H, ψ;
        max_iter=0, verbose=0, analyze_weights=false,
        weight_analysis_file="unused.csv")
end
