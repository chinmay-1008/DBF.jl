using DBF
using PauliOperators
using Printf
using Random
using LinearAlgebra
using JLD2

function run()
    N = 6
    # k = -1.0
    J = -1.0
    ratios = sort(unique(vcat(10 .^ range(-2, 2, length=20), 1.0)))
    # k_fixed = -1.0            # keep k constant (you can flip sign)
    # ratios = collect(-3.0:0.05:3.0)
    # ratios = range(-2, 2, length=51)
    # ratios = sort(unique(vcat(range(-2, 2, length=50), [-1.0, 1])))

    # ratios = 10 .^ range(-2, 1, length=20)  # covers k/J = 0.01 → 10
    # ratios = collect(ratios)

    energies = Float64[]
    energies_corr = Float64[]

    exact_energies = Float64[]
    display(ratios)
    for r in ratios
        k = r * J
        # J = r * k
        println("J: ", J, " k: ", k)
        H = DBF.heisenberg_1D(N, J, J, k)
        DBF.coeff_clip!(H)
    
        # Transform H to make |000> the most stable bitstring
        for i in 1:N
            if i%2 == 0
                H = Pauli(N, X=[i]) * H * Pauli(N, X=[i])
            end
        end
        
        H0 = deepcopy(H)
        
        ψ = Ket([0 for i in 1:N])
        # display(ψ)
        e0 = expectation_value(H,ψ)

        exact = minimum(real(eigvals(Matrix(H))))
        @show exact
        push!(exact_energies, real(exact)/N)

        @printf(" Reference = %12.8f\n", e0)
        
        println("\n ########################")
        @time res = DBF.dbf_groundstate(H, ψ,
                                        verbose=1,
                                        max_iter=200, 
                                        conv_thresh=1e-4,
                                        evolve_coeff_thresh=1e-3,
                                        grad_coeff_thresh=1e-6,
                                        energy_lowering_thresh=1e-6,
                                        max_rots_per_grad=50, 
                                        #checkfile = "test/plots/t1e-4"
                                        )
                                        
        push!(energies, res["energies"][end]/N)
        push!(energies_corr, (res["energies"][end] - res["accumulated_error"][end] + res["pt2_per_grad"][end])/N)
        println("Exact")
        display(exact_energies)
        println("DBF")
        display(energies)
        println("Corrected Energies")
        display(energies_corr)
    end
    println("===========FINAL================")
    println("Exact")
    display(exact_energies)
    println("DBF")
    display(energies)
    println("Corrected Energies")
    display(energies_corr)

    return
end


#@profilehtml run(4)
@time run()