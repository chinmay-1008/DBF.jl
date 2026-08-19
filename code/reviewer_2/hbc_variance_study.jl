using DBF
using JLD2
using PauliOperators
using Printf

# Replay an existing HBC vDBF trajectory without rebuilding gradients:
#
#   julia --project=. code/reviewer_2/hbc_variance_study.jl \
#       t0.00016_hbc.jld2 1.6e-4 [max_iterations]
#
# A relative trajectory filename is resolved beside this script as well as
# from the current working directory.

const DEFAULT_TRAJECTORY = "t0.00016_hbc.jld2"
const INITIAL_HAMILTONIAN_NAME = "hammy_pauli_hbc_transformed.jld2"


function parse_positive(::Type{T}, value::AbstractString, name::AbstractString) where T
    parsed = parse(T, value)
    parsed > zero(T) || throw(ArgumentError("$name must be positive"))
    return parsed
end


"Resolve a file supplied either as a path or as a filename beside this script."
function resolve_trajectory(selector::AbstractString)
    candidates = unique(abspath.((
        selector,
        joinpath(@__DIR__, selector),
        joinpath(@__DIR__, "..", selector),
        joinpath(pwd(), selector),
    )))
    index = findfirst(isfile, candidates)
    isnothing(index) && error(
        "Could not find trajectory '$selector'. Checked:\n  " *
        join(candidates, "\n  "),
    )
    return candidates[index]
end


"Read the original truncation threshold from a t<value>_hbc.jld2 filename."
function trajectory_threshold(path::AbstractString)
    result = match(r"^t(.+)_hbc\.jld2$", basename(path))
    isnothing(result) && throw(ArgumentError(
        "Cannot infer the threshold from $(basename(path)); pass it as the " *
        "second command-line argument.",
    ))
    return parse_positive(Float64, only(result.captures), "threshold")
end


"Find the initial Hamiltonian, allowing an explicit environment override."
function resolve_initial_hamiltonian(trajectory_file::AbstractString)
    candidates = String[]
    if haskey(ENV, "DBF_HBC_INITIAL_HAMILTONIAN")
        push!(candidates, abspath(ENV["DBF_HBC_INITIAL_HAMILTONIAN"]))
    end
    append!(candidates, abspath.((
        joinpath(dirname(trajectory_file), INITIAL_HAMILTONIAN_NAME),
        joinpath(@__DIR__, INITIAL_HAMILTONIAN_NAME),
        joinpath(@__DIR__, "..", INITIAL_HAMILTONIAN_NAME),
        joinpath(pwd(), INITIAL_HAMILTONIAN_NAME),
        joinpath(pwd(), "code", INITIAL_HAMILTONIAN_NAME),
    )))
    unique!(candidates)
    index = findfirst(isfile, candidates)
    isnothing(index) && error(
        "The saved trajectory does not contain H0, and the initial HBC " *
        "Hamiltonian was not found. Put $INITIAL_HAMILTONIAN_NAME beside " *
        "the trajectory or set DBF_HBC_INITIAL_HAMILTONIAN. Checked:\n  " *
        join(candidates, "\n  "),
    )
    return candidates[index]
end


function load_initial_hamiltonian(saved::AbstractDict, trajectory_file::AbstractString)
    if haskey(saved, "H0")
        println("Using H0 stored in the trajectory")
        return saved["H0"]
    end

    path = resolve_initial_hamiltonian(trajectory_file)
    @printf("Loading initial Hamiltonian from %s\n", path)
    return JLD2.load(path, "H_transformed")
end


"Infer macro-iteration boundaries from the saved per-gradient energies."
function iteration_boundaries(saved::AbstractDict, number_of_rotations::Int)
    haskey(saved, "energies") || error("Trajectory has no per-rotation energies")
    haskey(saved, "energies_per_grad") || error(
        "Trajectory has no energies_per_grad iteration metadata",
    )
    energies = saved["energies"]
    per_gradient = saved["energies_per_grad"]
    length(energies) == number_of_rotations + 1 || error(
        "Expected $(number_of_rotations + 1) saved energies, got " *
        "$(length(energies))",
    )
    length(per_gradient) >= 2 || error("Trajectory contains no completed iterations")

    boundaries = Int[]
    previous = 0
    for target in @view per_gradient[2:end]
        match_index = findfirst(previous + 1:number_of_rotations) do rotation
            isapprox(energies[rotation + 1], target; rtol=1e-10, atol=1e-10)
        end
        if isnothing(match_index)
            empty!(boundaries)
            break
        end
        previous += match_index
        push!(boundaries, previous)
    end

    if !isempty(boundaries) && last(boundaries) == number_of_rotations
        return boundaries
    end

    number_of_iterations = length(per_gradient) - 1
    rotations_per_iteration, remainder =
        divrem(number_of_rotations, number_of_iterations)
    remainder == 0 || error(
        "Could not match per-gradient energies to rotation boundaries, and " *
        "$number_of_rotations rotations do not divide evenly into " *
        "$number_of_iterations iterations.",
    )
    @warn "Could not recover variable iteration boundaries from the saved " *
          "energies; assuming $rotations_per_iteration rotations per iteration"
    return collect(rotations_per_iteration:rotations_per_iteration:number_of_rotations)
end


function rotation_metadata(boundaries::Vector{Int}, iterations_to_run::Int)
    cutoff = boundaries[iterations_to_run]
    macro_iterations = Vector{Int}(undef, cutoff)
    rotations_in_iteration = Vector{Int}(undef, cutoff)
    first_rotation = 1
    for iteration in 1:iterations_to_run
        last_rotation = boundaries[iteration]
        for rotation in first_rotation:last_rotation
            macro_iterations[rotation] = iteration
            rotations_in_iteration[rotation] = rotation - first_rotation + 1
        end
        first_rotation = last_rotation + 1
    end
    return macro_iterations, rotations_in_iteration
end


function write_diagnostics(path, result, threshold, max_rots_per_grad)
    records = result["variance_truncation_records"]
    iterations = result["variance_truncation_iteration"]
    rotations_in_iteration =
        result["variance_truncation_rotation_in_iteration"]
    nrecords = length(records)

    nrecords == length(iterations) == length(rotations_in_iteration) ||
        error("Variance-component records and rotation metadata are misaligned")
    length(result["energies"]) == nrecords + 1 ||
        error("Energy history does not align with variance-component records")
    length(result["variances"]) == nrecords + 1 ||
        error("Variance history does not align with variance-component records")

    cumulative_var_B = 0.0
    cumulative_covariance = 0.0
    reconstructed_error = result["accumulated_var_error"][1]
    maximum_residual = 0.0

    open(path, "w") do io
        println(io, join((
            "global_rotation",
            "macro_iteration",
            "rotation_within_iteration",
            "threshold",
            "max_rots_per_grad",
            "energy",
            "raw_variance",
            "var_B",
            "cov_A_B",
            "two_cov_A_B",
            "minus_var_B",
            "minus_two_cov_A_B",
            "delta_variance",
            "cumulative_var_B_contribution",
            "cumulative_covariance_contribution",
            "reconstructed_accumulated_variance_error",
            "recorded_accumulated_variance_error",
            "reconstruction_residual",
            "delta_energy",
            "accumulated_energy_error",
            "covariance_to_var_B_ratio",
        ), '\t'))

        for index in eachindex(records)
            record = records[index]
            minus_var_B = -record.var_B
            minus_two_covariance = -record.two_cov_A_B
            cumulative_var_B += minus_var_B
            cumulative_covariance += minus_two_covariance
            reconstructed_error += record.delta_variance

            history_index = index + 1
            recorded_error = result["accumulated_var_error"][history_index]
            residual = reconstructed_error - recorded_error
            maximum_residual = max(maximum_residual, abs(residual))
            ratio = abs(cumulative_var_B) > eps(Float64) ?
                abs(cumulative_covariance) / abs(cumulative_var_B) : NaN

            values = (
                index,
                iterations[index],
                rotations_in_iteration[index],
                threshold,
                max_rots_per_grad,
                result["energies"][history_index],
                result["variances"][history_index],
                record.var_B,
                record.cov_A_B,
                record.two_cov_A_B,
                minus_var_B,
                minus_two_covariance,
                record.delta_variance,
                cumulative_var_B,
                cumulative_covariance,
                reconstructed_error,
                recorded_error,
                residual,
                record.delta_energy,
                result["accumulated_error"][history_index],
                ratio,
            )
            println(io, join(values, '\t'))
        end
    end

    return (
        rotations=nrecords,
        cumulative_var_B=cumulative_var_B,
        cumulative_covariance=cumulative_covariance,
        reconstructed_error=reconstructed_error,
        recorded_error=result["accumulated_var_error"][end],
        maximum_residual=maximum_residual,
    )
end


function replay_trajectory(
    trajectory_file::AbstractString,
    threshold::Float64;
    max_iterations::Union{Nothing,Int}=nothing,
)
    @printf("Loading saved trajectory from %s\n", trajectory_file)
    saved = JLD2.load(trajectory_file, "out")
    generators = saved["generators"]
    angles = saved["angles"]
    number_of_rotations = length(angles)
    length(generators) == number_of_rotations || error(
        "Generator/angle length mismatch: $(length(generators)) generators " *
        "and $number_of_rotations angles",
    )

    boundaries = iteration_boundaries(saved, number_of_rotations)
    available_iterations = length(boundaries)
    if !isnothing(max_iterations)
        max_iterations > 0 || throw(ArgumentError("max_iterations must be positive"))
    end
    iterations_to_run = isnothing(max_iterations) ?
        available_iterations : min(max_iterations, available_iterations)
    macro_iterations, rotations_in_iteration =
        rotation_metadata(boundaries, iterations_to_run)
    rotations_to_run = length(macro_iterations)
    maximum_rotations_per_iteration = maximum(rotations_in_iteration)

    initial = load_initial_hamiltonian(saved, trajectory_file)
    hamiltonian = initial isa SparsePauliVector ?
        deepcopy(initial) : SparsePauliVector(initial)
    initial = nothing
    reference = Ket{84}(0)
    truncation = CoeffTruncation(threshold)
    correction = EnergyVarianceCorrection(reference; record_components=true)

    initial_energy_error = haskey(saved, "accumulated_error") ?
        real(saved["accumulated_error"][1]) : 0.0
    initial_variance_error = haskey(saved, "accumulated_var_error") ?
        real(saved["accumulated_var_error"][1]) : 0.0
    correction.accumulated_energy = initial_energy_error
    correction.accumulated_variance = initial_variance_error

    energies = Vector{Float64}(undef, rotations_to_run + 1)
    variances = Vector{Float64}(undef, rotations_to_run + 1)
    accumulated_error = Vector{Float64}(undef, rotations_to_run + 1)
    accumulated_var_error = Vector{Float64}(undef, rotations_to_run + 1)
    energies[1] = real(expectation_value(hamiltonian, reference))
    variances[1] = real(variance(hamiltonian, reference))
    accumulated_error[1] = initial_energy_error
    accumulated_var_error[1] = initial_variance_error

    @printf(
        "Replaying %d/%d macro-iterations (%d rotations; maximum %d per iteration)\n",
        iterations_to_run,
        available_iterations,
        rotations_to_run,
        maximum_rotations_per_iteration,
    )
    @printf("No gradients, commutators, or angles will be recomputed.\n")

    for rotation in 1:rotations_to_run
        record_count_before = length(correction.records)
        PauliOperators.evolve!(
            hamiltonian,
            [generators[rotation]],
            [angles[rotation]];
            window=1,
            truncation=truncation,
            correction=correction,
        )
        length(correction.records) == record_count_before + 1 || error(
            "Expected exactly one truncation record at rotation $rotation; got " *
            "$(length(correction.records) - record_count_before)",
        )

        # Energies were saved per rotation. Variance was generally saved only
        # per macro-iteration, so calculate it from the replayed Hamiltonian.
        energies[rotation + 1] = real(saved["energies"][rotation + 1])
        variances[rotation + 1] = real(variance(hamiltonian, reference))
        accumulated_error[rotation + 1] = real(correction.accumulated_energy)
        accumulated_var_error[rotation + 1] = real(correction.accumulated_variance)

        iteration = macro_iterations[rotation]
        if rotation == boundaries[iteration] &&
           (iteration == 1 || iteration == iterations_to_run || iteration % 10 == 0)
            @printf(
                "  iteration %4d/%4d, rotation %6d, terms %d\n",
                iteration,
                iterations_to_run,
                rotation,
                length(hamiltonian),
            )
            flush(stdout)
        end
    end

    result = Dict{String,Any}(
        "variance_truncation_records" => correction.records,
        "variance_truncation_iteration" => macro_iterations,
        "variance_truncation_rotation_in_iteration" => rotations_in_iteration,
        "iteration_boundaries" => boundaries[1:iterations_to_run],
        "energies" => energies,
        "variances" => variances,
        "accumulated_error" => accumulated_error,
        "accumulated_var_error" => accumulated_var_error,
    )
    return result, iterations_to_run, maximum_rotations_per_iteration
end


function compare_saved_variance_checkpoints(saved, result)
    haskey(saved, "variance_per_grad") || return nothing
    boundaries = result["iteration_boundaries"]
    saved_variances = saved["variance_per_grad"]
    length(saved_variances) >= length(boundaries) + 1 || return nothing
    replayed_variances = result["variances"][boundaries .+ 1]
    return maximum(abs.(
        real.(saved_variances[2:length(boundaries) + 1]) .- replayed_variances,
    ))
end


function compare_saved_errors(saved, result, rotations_to_run)
    comparisons = Pair{String,Float64}[]
    for (name, replayed_name) in (
        "accumulated_error" => "accumulated_error",
        "accumulated_var_error" => "accumulated_var_error",
    )
        haskey(saved, name) || continue
        saved_history = saved[name]
        length(saved_history) >= rotations_to_run + 1 || continue
        replayed = result[replayed_name]
        # Older HBC trajectories were generated with compute_var_error=false,
        # leaving an all-zero placeholder rather than a measured history.
        name == "accumulated_var_error" && all(iszero, saved_history) &&
            any(x -> !iszero(x), replayed) && continue
        difference = maximum(abs.(
            real.(saved_history[1:rotations_to_run + 1]) .- replayed,
        ))
        push!(comparisons, name => difference)
    end
    return comparisons
end


function run(args=ARGS)
    length(args) <= 3 || throw(ArgumentError(
        "Usage: hbc_variance_study.jl [trajectory.jld2] [threshold] " *
        "[max_iterations]",
    ))
    trajectory = resolve_trajectory(
        length(args) >= 1 ? args[1] : DEFAULT_TRAJECTORY,
    )
    threshold = length(args) >= 2 ?
        parse_positive(Float64, args[2], "threshold") :
        trajectory_threshold(trajectory)
    max_iterations = length(args) >= 3 ?
        parse_positive(Int, args[3], "max_iterations") : nothing

    filename_threshold = try
        trajectory_threshold(trajectory)
    catch error
        error isa ArgumentError || rethrow()
        nothing
    end
    if !isnothing(filename_threshold) &&
       !isapprox(threshold, filename_threshold; rtol=1e-12, atol=0.0)
        throw(ArgumentError(
            "Replay threshold $threshold does not match the trajectory " *
            "threshold $filename_threshold encoded in $(basename(trajectory)). " *
            "Use the original threshold so the saved rotations reproduce the run.",
        ))
    end

    @printf("HBC saved-trajectory variance decomposition\n")
    @printf(" trajectory           = %s\n", trajectory)
    @printf(" replay threshold     = %.6e\n", threshold)
    result, iterations_run, max_rots_per_grad = replay_trajectory(
        trajectory,
        threshold;
        max_iterations=max_iterations,
    )

    output_directory = joinpath(
        @__DIR__, "results", "variance_decomposition")
    mkpath(output_directory)
    threshold_label = replace(@sprintf("%.1e", threshold), "+" => "")
    output_file = joinpath(
        output_directory,
        "hbc_variance_decomposition_eps$(threshold_label)_" *
        "iter$(iterations_run)_rots$(max_rots_per_grad).tsv",
    )
    summary = write_diagnostics(
        output_file,
        result,
        threshold,
        max_rots_per_grad,
    )

    saved = JLD2.load(trajectory, "out")
    comparisons = compare_saved_errors(saved, result, summary.rotations)
    variance_checkpoint_difference =
        compare_saved_variance_checkpoints(saved, result)
    @printf("\nSaved %d rotation records to:\n%s\n", summary.rotations, output_file)
    @printf(" cumulative -Var(B)       = %.12e\n", summary.cumulative_var_B)
    @printf(" cumulative -2Cov(A,B)    = %.12e\n", summary.cumulative_covariance)
    @printf(" reconstructed var error  = %.12e\n", summary.reconstructed_error)
    @printf(" recorded var error       = %.12e\n", summary.recorded_error)
    @printf(" maximum residual         = %.3e\n", summary.maximum_residual)
    if haskey(saved, "accumulated_var_error") &&
       all(iszero, saved["accumulated_var_error"]) &&
       any(x -> !iszero(x), result["accumulated_var_error"])
        println(" saved accumulated_var_error was an unrecorded all-zero " *
                "placeholder; the replay generated this history")
    end
    if !isnothing(variance_checkpoint_difference)
        @printf(" max replay/saved variance checkpoint difference = %.3e\n",
                variance_checkpoint_difference)
    end
    for (name, difference) in comparisons
        @printf(" max replay/saved %-21s difference = %.3e\n", name, difference)
    end
    return output_file
end


if abspath(PROGRAM_FILE) == abspath(@__FILE__)
    run()
end
