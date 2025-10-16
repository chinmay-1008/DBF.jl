using DBF
using PauliOperators
using Printf
using Random
using LinearAlgebra
using JLD2

function run()
    N = 12 
    Random.seed!(2)
    H = DBF.heisenberg_1D(N, -1, -1, -1)
    DBF.coeff_clip!(H)
   
    # Transform H to make |000> the most stable bitstring
    for i in 1:N
        if i%2 == 0
            H = Pauli(N, X=[i]) * H * Pauli(N, X=[i])
        end
    end 
    
    H0 = deepcopy(H)
    # display(H)
    
    ψ = Ket([0 for i in 1:N])
    display(ψ)
    e0 = expectation_value(H,ψ)
    
    @printf(" Reference = %12.8f\n", e0)
    
    g = Vector{PauliBasis{N}}([]) 
    θ = Vector{Float64}([]) 

    for i in 1:5

        println("\n ########################")
        println(" ", i)
        @time H, g2, θ2 = DBF.dbf_groundstate(H, ψ, n_body=1, 
                                    verbose=1, 
                                    max_iter=120, conv_thresh=1e-3, 
                                    evolve_coeff_thresh=1e-3,
                                    grad_coeff_thresh=1e-4,
                                    search_n_top=1*10^i)
        g = vcat(g,g2)
        θ = vcat(θ,θ2)
       
        @save "out_$(i).jld2" N ψ H0 H g θ
    end
    
    
    return


    println("\n Now reroptimize with higher accuracy:")
    @show length(θ)
    Ht = deepcopy(H0)
    err = 0
    ecurr = expectation_value(Ht,ψ)
    @printf(" Initial energy: %12.8f %8i\n", ecurr, length(Ht))
    for (i,gi) in enumerate(g)
            
        θj, costi = DBF.optimize_theta_expval(Ht, gi, ψ, verbose=0)
        Ht = DBF.evolve(Ht, gi, θj)
        θ[i] = θj
        
        e1 = expectation_value(Ht,ψ)
        DBF.coeff_clip!(Ht, thresh=1e-5)
        e2 = expectation_value(Ht,ψ)

        err += e2 - e1
        if i%100 == 0
            @printf(" Error: %12.8f\n", err)
            e0, e2 = DBF.pt2(Ht, ψ)
            @printf(" E0 = %12.8f E2 = %12.8f EPT2 = %12.8f \n", e0, e2, e0+e2)
            e0, e, v, basis = DBF.cepa(Ht, ψ, thresh=1e-6, tol=1e-2, verbose=0)
            e0, e, v, basis = DBF.fois_ci(Ht, ψ, thresh=1e-6, tol=1e-2, verbose=0)
        end
    end    
    ecurr = expectation_value(Ht,ψ)
    @printf(" ecurr %12.8f err %12.8f %8i\n", ecurr, err, length(Ht))
   

end


run()