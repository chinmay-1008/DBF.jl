"""
    weight_distribution(O::AnyPauliSum)

Compute the Pauli- and Majorana-weight distributions of `O`. The returned
named tuple contains term counts, coefficient L2 norms, and two percentage
distributions for each weight definition.

`*_count_percent` is the percentage of represented Pauli terms in each
weight subspace. `*_l2_percent` is the percentage of the squared coefficient
L2 norm in each subspace,

```math
100 \\frac{\\lVert O_w \\rVert_2^2}{\\lVert O \\rVert_2^2},
```

so both kinds of percentage distribution sum to 100 for a nonempty,
nonzero operator.
"""
function weight_distribution(O::AnyPauliSum{N}) where N
    pauli_counts = zeros(Int, N + 1)
    pauli_l2_squared = zeros(Float64, N + 1)
    majorana_counts = zeros(Int, 2N + 1)
    majorana_l2_squared = zeros(Float64, 2N + 1)

    for (p, c) in O
        pw_idx = weight(p) + 1
        mw_idx = majorana_weight(p) + 1
        coefficient_norm = Float64(abs2(c))

        pauli_counts[pw_idx] += 1
        pauli_l2_squared[pw_idx] += coefficient_norm
        majorana_counts[mw_idx] += 1
        majorana_l2_squared[mw_idx] += coefficient_norm
    end

    count_total = sum(pauli_counts)
    l2_squared_total = sum(pauli_l2_squared)

    count_scale = iszero(count_total) ? 0.0 : 100.0 / count_total
    l2_scale = iszero(l2_squared_total) ? 0.0 : 100.0 / l2_squared_total

    return (
        n_qubits=N,
        pauli_weights=collect(0:N),
        majorana_weights=collect(0:2N),
        pauli_counts=pauli_counts,
        majorana_counts=majorana_counts,
        pauli_l2_squared=pauli_l2_squared,
        majorana_l2_squared=majorana_l2_squared,
        pauli_l2_norms=sqrt.(pauli_l2_squared),
        majorana_l2_norms=sqrt.(majorana_l2_squared),
        pauli_count_percent=count_scale .* pauli_counts,
        majorana_count_percent=count_scale .* majorana_counts,
        pauli_l2_percent=l2_scale .* pauli_l2_squared,
        majorana_l2_percent=l2_scale .* majorana_l2_squared,
        term_count=count_total,
        l2_norm=sqrt(l2_squared_total),
        l2_squared=l2_squared_total,
    )
end

function _initialize_weight_analysis(O::AnyPauliSum, interval::Symbol)
    snapshot = weight_distribution(O)
    return Dict{String,Any}(
        "interval" => interval,
        "n_qubits" => snapshot.n_qubits,
        "iterations" => Int[],
        "rotations" => Int[],
        "pauli_weights" => snapshot.pauli_weights,
        "majorana_weights" => snapshot.majorana_weights,
        "term_counts" => Int[],
        "l2_norms" => Float64[],
        "l2_squared" => Float64[],
        "pauli_counts" => Vector{Vector{Int}}(),
        "majorana_counts" => Vector{Vector{Int}}(),
        "pauli_l2_norms" => Vector{Vector{Float64}}(),
        "majorana_l2_norms" => Vector{Vector{Float64}}(),
        "pauli_l2_squared" => Vector{Vector{Float64}}(),
        "majorana_l2_squared" => Vector{Vector{Float64}}(),
        "pauli_count_percent" => Vector{Vector{Float64}}(),
        "pauli_l2_percent" => Vector{Vector{Float64}}(),
        "majorana_count_percent" => Vector{Vector{Float64}}(),
        "majorana_l2_percent" => Vector{Vector{Float64}}(),
    )
end

function _record_weight_distribution!(history::AbstractDict, O::AnyPauliSum;
                                      iteration::Int, rotation::Int)
    snapshot = weight_distribution(O)
    push!(history["iterations"], iteration)
    push!(history["rotations"], rotation)
    push!(history["term_counts"], snapshot.term_count)
    push!(history["l2_norms"], snapshot.l2_norm)
    push!(history["l2_squared"], snapshot.l2_squared)
    push!(history["pauli_counts"], snapshot.pauli_counts)
    push!(history["majorana_counts"], snapshot.majorana_counts)
    push!(history["pauli_l2_norms"], snapshot.pauli_l2_norms)
    push!(history["majorana_l2_norms"], snapshot.majorana_l2_norms)
    push!(history["pauli_l2_squared"], snapshot.pauli_l2_squared)
    push!(history["majorana_l2_squared"], snapshot.majorana_l2_squared)
    push!(history["pauli_count_percent"], snapshot.pauli_count_percent)
    push!(history["pauli_l2_percent"], snapshot.pauli_l2_percent)
    push!(history["majorana_count_percent"], snapshot.majorana_count_percent)
    push!(history["majorana_l2_percent"], snapshot.majorana_l2_percent)
    return history
end

"""
    _save_weight_analysis_csv(file, history)

Save a weight-analysis history in a long-form CSV suitable for Python, R, or
spreadsheet post-processing. Returns the final path; `.csv` is appended when
the supplied filename has no CSV extension.
"""
function _save_weight_analysis_csv(file, history::AbstractDict)
    path = string(file)
    endswith(lowercase(path), ".csv") || (path *= ".csv")
    path = abspath(path)
    parent = dirname(path)
    mkpath(parent)

    open(path, "w") do io
        println(io,
            "sample_index,iteration,rotation,interval,n_qubits,weight_kind,weight," *
            "term_count,subspace_l2_norm,subspace_l2_squared,count_percent," *
            "l2_percent,total_term_count,total_l2_norm,total_l2_squared")

        interval = string(history["interval"])
        for sample in eachindex(history["iterations"])
            for (kind, weights, counts, l2_norms, l2_squared,
                 count_percent, l2_percent) in (
                    ("pauli", history["pauli_weights"],
                     history["pauli_counts"][sample],
                     history["pauli_l2_norms"][sample],
                     history["pauli_l2_squared"][sample],
                     history["pauli_count_percent"][sample],
                     history["pauli_l2_percent"][sample]),
                    ("majorana", history["majorana_weights"],
                     history["majorana_counts"][sample],
                     history["majorana_l2_norms"][sample],
                     history["majorana_l2_squared"][sample],
                     history["majorana_count_percent"][sample],
                     history["majorana_l2_percent"][sample]),
                )
                for i in eachindex(weights)
                    @printf(io,
                        "%d,%d,%d,%s,%d,%s,%d,%d,%.17g,%.17g,%.17g,%.17g,%d,%.17g,%.17g\n",
                        sample - 1,
                        history["iterations"][sample],
                        history["rotations"][sample],
                        interval,
                        history["n_qubits"],
                        kind,
                        weights[i],
                        counts[i],
                        l2_norms[i],
                        l2_squared[i],
                        count_percent[i],
                        l2_percent[i],
                        history["term_counts"][sample],
                        history["l2_norms"][sample],
                        history["l2_squared"][sample])
                end
            end
        end
    end
    return path
end
