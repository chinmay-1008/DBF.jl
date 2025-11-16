using DBF
using PauliOperators
using Printf
using Random
using LinearAlgebra
using JLD2

function run(L)
    println("\n SYSTEM SIZE: ", L)
    N = L

    H = DBF.heisenberg_1D(L, -1.0, -1.0, -1.0)
    DBF.coeff_clip!(H)

    # Transform H to make |000> the most stable bitstring
    for i in 1:N
        if i % 2 == 0
            H = Pauli(N, X=[i]) * H * Pauli(N, X=[i])
        end
    end

    ψ = Ket([0 for i in 1:N])
    e0 = expectation_value(H, ψ)
    @printf(" Reference = %12.8f\n", e0)

    println("\n ########################")
    @time res = DBF.dbf_groundstate(
        H, ψ;
        verbose=1,
        max_iter=500,
        conv_thresh=1e-4,
        evolve_coeff_thresh=1e-4,
        grad_coeff_thresh=1e-6,
        energy_lowering_thresh=1e-6,
        max_rots_per_grad=100,
        # checkfile="/N/u/cshrikh/BigRed200/final_results/var_N/heisenberg/1D/thresh_1e-4/checkfiles/t1e-4grad_$(N)"
    )

    energy = res["energies"][end] / N
    energy_corr = (res["energies"][end] - res["accumulated_error"][end]) / N

    println("\n===========FINAL================")
    println("DBF Energy/N = ", energy)
    println("Corrected Energy/N = ", energy_corr)

    # @save "/N/u/cshrikh/BigRed200/final_results/var_N/heisenberg/1D/thresh_1e-4/checkfiles/energy_gradL$(L).jld2" energy energy_corr res
end

# # -------- MAIN -------------
# if length(ARGS) == 0
#     error("Usage: julia var_N_1e-4.jl <L>")
# end

# L = parse(Int, ARGS[1])
run(100)
