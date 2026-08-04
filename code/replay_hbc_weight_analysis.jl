using DBF
using JLD2
using PauliOperators
using Printf

# `submit.sh` supplies `--project=/path/to/DBF.jl`. Slurm normally preserves
# the submission directory in SLURM_SUBMIT_DIR, so the large JLD2 files do not
# need to live in the cloned package repository or beside this script.
const DEFAULT_RUN_DIR = get(ENV, "SLURM_SUBMIT_DIR", pwd())
const DATA_DIR = abspath(get(ENV, "DBF_HBC_DATA_DIR", DEFAULT_RUN_DIR))
const INITIAL_HAMILTONIAN_FILE = joinpath(DATA_DIR, "hammy_pauli_hbc_transformed.jld2")
const OUTPUT_DIR = abspath(get(ENV, "DBF_HBC_OUTPUT_DIR", joinpath(DATA_DIR, "results")))
const REPLAY_THRESHOLD = parse(Float64, get(ENV, "DBF_HBC_REPLAY_THRESHOLD", "1e-5"))


"""Read the numeric truncation threshold from `t<value>_hbc.jld2`."""
function trajectory_threshold(path::AbstractString)
    result = match(r"^t(.+)_hbc\.jld2$", basename(path))
    isnothing(result) && error("Cannot read a threshold from $(basename(path))")
    return parse(Float64, only(result.captures))
end


"""Find all saved HBC trajectories, or resolve one trajectory selected on the CLI."""
function trajectory_files(selector::AbstractString="all")
    isdir(DATA_DIR) || error("Data directory does not exist: $DATA_DIR")

    if lowercase(selector) == "all"
        files = filter(readdir(DATA_DIR; join=true)) do path
            occursin(r"^t.+_hbc\.jld2$", basename(path))
        end
        isempty(files) && error("No t*_hbc.jld2 files found in $DATA_DIR")
        return sort(files; by=trajectory_threshold)
    end

    candidates = String[
        selector,
        joinpath(DATA_DIR, selector),
        joinpath(DATA_DIR, "t$(selector)_hbc.jld2"),
    ]
    index = findfirst(isfile, candidates)
    isnothing(index) && error(
        "Could not find trajectory '$selector'. Use 'all', a threshold such as " *
        "'0.001', or a t*_hbc.jld2 filename.",
    )
    return [abspath(candidates[index])]
end


"""
Infer the exact iteration boundaries recorded by DBF. The supplied HBC files
contain one initial entry in `energies_per_grad`, followed by one entry per
iteration.
"""
function iteration_layout(saved::AbstractDict, number_of_rotations::Int)
    haskey(saved, "energies_per_grad") || error(
        "The trajectory has no energies_per_grad metadata, so its iteration " *
        "boundaries cannot be inferred.",
    )
    number_of_iterations = length(saved["energies_per_grad"]) - 1
    number_of_iterations > 0 || error("The trajectory contains no completed iterations")
    rotations_per_iteration, remainder = divrem(number_of_rotations, number_of_iterations)
    remainder == 0 || error(
        "$number_of_rotations rotations do not divide evenly into " *
        "$number_of_iterations saved iterations.",
    )
    return number_of_iterations, rotations_per_iteration
end


function write_csv_header(io::IO)
    println(
        io,
        "sample_index,iteration,rotation,interval,n_qubits,weight_kind,weight," *
        "term_count,subspace_l2_norm,subspace_l2_squared,count_percent," *
        "l2_percent,total_term_count,total_l2_norm,total_l2_squared",
    )
end


"""Append one Pauli- and Majorana-weight snapshot in DBF's usual CSV schema."""
function write_snapshot(io::IO, snapshot; sample_index::Int, iteration::Int, rotation::Int)
    for (kind, weights, counts, l2_norms, l2_squared, count_percent, l2_percent) in (
        (
            "pauli",
            snapshot.pauli_weights,
            snapshot.pauli_counts,
            snapshot.pauli_l2_norms,
            snapshot.pauli_l2_squared,
            snapshot.pauli_count_percent,
            snapshot.pauli_l2_percent,
        ),
        (
            "majorana",
            snapshot.majorana_weights,
            snapshot.majorana_counts,
            snapshot.majorana_l2_norms,
            snapshot.majorana_l2_squared,
            snapshot.majorana_count_percent,
            snapshot.majorana_l2_percent,
        ),
    )
        for index in eachindex(weights)
            @printf(
                io,
                "%d,%d,%d,iteration,%d,%s,%d,%d,%.17g,%.17g,%.17g,%.17g,%d,%.17g,%.17g\n",
                sample_index,
                iteration,
                rotation,
                snapshot.n_qubits,
                kind,
                weights[index],
                counts[index],
                l2_norms[index],
                l2_squared[index],
                count_percent[index],
                l2_percent[index],
                snapshot.term_count,
                snapshot.l2_norm,
                snapshot.l2_squared,
            )
        end
    end
    # Keep every completed iteration available even if a long replay is stopped.
    flush(io)
end


"""Replay one saved DBF trajectory and save weight data after every iteration."""
function replay_trajectory(
    trajectory_file::AbstractString;
    max_iterations::Union{Nothing,Int}=nothing,
)
    isfile(INITIAL_HAMILTONIAN_FILE) || error(
        "Initial Hamiltonian not found: $INITIAL_HAMILTONIAN_FILE",
    )
    if !isnothing(max_iterations)
        max_iterations >= 0 || error("max_iterations must be nonnegative")
    end

    @printf("Loading rotations from %s\n", trajectory_file)
    saved = JLD2.load(trajectory_file, "out")
    generators = saved["generators"]
    angles = saved["angles"]
    length(generators) == length(angles) || error(
        "Generator/angle length mismatch in $(basename(trajectory_file))",
    )

    available_iterations, rotations_per_iteration = iteration_layout(saved, length(angles))
    iterations_to_run = isnothing(max_iterations) ?
        available_iterations : min(max_iterations, available_iterations)
    cutoff = trajectory_threshold(trajectory_file)

    @printf("Loading initial Hamiltonian from %s\n", INITIAL_HAMILTONIAN_FILE)
    initial_hamiltonian = JLD2.load(INITIAL_HAMILTONIAN_FILE, "H_transformed")
    hamiltonian = initial_hamiltonian isa SparsePauliVector ?
        deepcopy(initial_hamiltonian) : SparsePauliVector(initial_hamiltonian)
    initial_hamiltonian = nothing

    initial_snapshot = DBF.weight_distribution(hamiltonian)
    reference = Ket{initial_snapshot.n_qubits}(0)
    correction = EnergyCorrection(reference)
    truncation = CoeffTruncation(REPLAY_THRESHOLD)

    mkpath(OUTPUT_DIR)
    stem = splitext(basename(trajectory_file))[1]
    output_file = joinpath(OUTPUT_DIR, "weight_distribution_$(stem).csv")

    @printf(
        "Replaying %d/%d iterations (%d rotations each); trajectory %.6e, replay cutoff %.6e\n",
        iterations_to_run,
        available_iterations,
        rotations_per_iteration,
        cutoff,
        REPLAY_THRESHOLD,
    )
    @printf("Writing each completed iteration to %s\n", output_file)

    open(output_file, "w") do io
        write_csv_header(io)
        write_snapshot(io, initial_snapshot; sample_index=0, iteration=0, rotation=0)

        for iteration in 1:iterations_to_run
            first_rotation = (iteration - 1) * rotations_per_iteration + 1
            last_rotation = iteration * rotations_per_iteration

            PauliOperators.evolve!(
                hamiltonian,
                generators[first_rotation:last_rotation],
                angles[first_rotation:last_rotation];
                window=1,
                truncation=truncation,
                correction=correction,
            )

            snapshot = DBF.weight_distribution(hamiltonian)
            write_snapshot(
                io,
                snapshot;
                sample_index=iteration,
                iteration=iteration,
                rotation=last_rotation,
            )

            if iteration == 1 || iteration == iterations_to_run || iteration % 10 == 0
                @printf(
                    "  iteration %4d/%4d, rotation %6d, terms %d\n",
                    iteration,
                    iterations_to_run,
                    last_rotation,
                    snapshot.term_count,
                )
                flush(stdout)
            end
        end
    end

    @printf("Finished: %s\n", output_file)
    return output_file
end


function run(; selector::AbstractString="all", max_iterations::Union{Nothing,Int}=nothing)
    files = trajectory_files(selector)
    outputs = String[]
    for file in files
        push!(outputs, replay_trajectory(file; max_iterations=max_iterations))
        GC.gc()
    end
    println("\nSaved $(length(outputs)) CSV file(s) in $OUTPUT_DIR")
    return outputs
end


if abspath(PROGRAM_FILE) == @__FILE__
    selector = isempty(ARGS) ? "all" : ARGS[1]
    max_iterations = length(ARGS) < 2 ? nothing : parse(Int, ARGS[2])
    length(ARGS) <= 2 || error(
        "Usage: julia replay_hbc_weight_analysis.jl [all|threshold|filename] [max_iterations]",
    )
    run(; selector=selector, max_iterations=max_iterations)
end
