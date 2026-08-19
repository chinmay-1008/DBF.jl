using DBF
using PauliOperators
using Printf
using JLD2


function parse_positive(::Type{T}, value::AbstractString, name::AbstractString) where T
    parsed = parse(T, value)
    parsed > zero(T) || throw(ArgumentError("$name must be positive"))
    return parsed
end

function write_diagnostics(path, result, threshold, max_rots_per_grad)
    records = result["variance_truncation_records"]
    global_rotations = result["variance_truncation_global_rotation"]
    iterations = result["variance_truncation_iteration"]
    rotations_in_iteration =
        result["variance_truncation_rotation_in_iteration"]
    nrecords = length(records)

    nrecords == length(global_rotations) == length(iterations) ==
        length(rotations_in_iteration) ||
        error("Variance-component records and rotation metadata are misaligned")
    length(result["energies"]) == nrecords + 1 ||
        error("Energy history does not align with variance-component records")
    length(result["variances"]) == nrecords + 1 ||
        error("Variance history does not align with variance-component records")

    cumulative_var_B = 0.0
    cumulative_covariance = 0.0
    reconstructed_error = 0.0
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

            history_index = index + 1  # index 1 is the initial Hamiltonian
            recorded_error =
                result["accumulated_var_error"][history_index]
            residual = reconstructed_error - recorded_error
            maximum_residual = max(maximum_residual, abs(residual))
            ratio = abs(cumulative_var_B) > eps(Float64) ?
                abs(cumulative_covariance) / abs(cumulative_var_B) : NaN

            values = (
                global_rotations[index],
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

function run(args=ARGS)
    threshold = length(args) >= 1 ?
        parse_positive(Float64, args[1], "threshold") : 8e-5
    max_iter = length(args) >= 2 ?
        parse_positive(Int, args[2], "max_iter") : 1000
    max_rots_per_grad = length(args) >= 3 ?
        parse_positive(Int, args[3], "max_rots_per_grad") : 1
    length(args) <= 3 || throw(ArgumentError(
        "Usage: hbc_variance_decomposition.jl " *
        "[threshold] [max_iter] [max_rots_per_grad]"))

    hamiltonian_file = normpath(joinpath(
        @__DIR__, "..", "hammy_pauli_hbc_transformed.jld2"))
    isfile(hamiltonian_file) || error("Missing HBC Hamiltonian: $hamiltonian_file")
    H_transformed = JLD2.load(hamiltonian_file, "H_transformed")
    ψ = Ket{84}(0)

    @printf("HBC variance decomposition\n")
    @printf(" threshold            = %.6e\n", threshold)
    @printf(" max_iter             = %d\n", max_iter)
    @printf(" max_rots_per_grad    = %d\n", max_rots_per_grad)

    result = DBF.dbf_groundstate(
        SparsePauliVector(H_transformed),
        ψ;
        max_iter=max_iter,
        max_rots_per_grad=max_rots_per_grad,
        conv_thresh=1e-6,
        operator_truncation=CoeffTruncation(threshold),
        gradient_truncation=CoeffTruncation(1e-6),
        compute_var_error=true,
        record_variance_components=true,
    )

    output_directory = joinpath(
        @__DIR__, "results", "variance_decomposition")
    mkpath(output_directory)
    threshold_label = replace(@sprintf("%.1e", threshold), "+" => "")
    output_file = joinpath(
        output_directory,
        "hbc_variance_decomposition_eps$(threshold_label)_" *
        "iter$(max_iter)_rots$(max_rots_per_grad).tsv",
    )
    summary = write_diagnostics(
        output_file, result, threshold, max_rots_per_grad)

    @printf("\nSaved %d rotation records to:\n%s\n",
            summary.rotations, output_file)
    @printf(" cumulative -Var(B)       = %.12e\n",
            summary.cumulative_var_B)
    @printf(" cumulative -2Cov(A,B)    = %.12e\n",
            summary.cumulative_covariance)
    @printf(" reconstructed var error  = %.12e\n",
            summary.reconstructed_error)
    @printf(" recorded var error       = %.12e\n",
            summary.recorded_error)
    @printf(" maximum residual         = %.3e\n",
            summary.maximum_residual)

    return output_file
end

if abspath(PROGRAM_FILE) == abspath(@__FILE__)
    run()
end
