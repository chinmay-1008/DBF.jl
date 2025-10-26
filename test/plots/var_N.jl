using DBF
using PauliOperators
using Printf
using Random
using LinearAlgebra
using JLD2

function run()

    energies = Float64[]
    energies_corr = Float64[]

    for N in 2:100
        println("\n SYSTEM SIZE: ", N)
        H = DBF.heisenberg_1D(N, -1.0, -1.0, -1.0)
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
        if N % 10 ==0
            println("DBF")
            display(energies)
            println("Corrected Energies")
            display(energies_corr)
        end
    end
    println("===========FINAL================")

    println("DBF")
    display(energies)
    println("Corrected Energies")
    display(energies_corr)

    return
end


#@profilehtml run(4)
run()