using LinearAlgebra
using DBF
using PauliOperators

function run()
    L = 8
    H = DBF.hubbard_model_1D(L, 1.0, 1.0)
    N = 2 * L
    bitstring = "0110010110011001"
    # bitstring = "0110011001"

    # @show ((eigvals(Matrix(H)))[1:5])

    for i in 1:N
        if bitstring[i] == '1'
            H = Pauli(N, X=[i]) * H * Pauli(N, X=[i])
        end
    end 

    ψ = Ket(N, 0)

    @time res = DBF.dbf_groundstate(H, ψ, 
                                    verbose=1, 
                                    max_iter=120, conv_thresh=1e-3, 
                                    evolve_coeff_thresh=1e-4,
                                    grad_coeff_thresh=1e-5,
                                    energy_lowering_thresh=1e-5,
                                    # clifford_check=true,
                                    # compute_pt2_error=true,
                                    max_rots_per_grad=50)
    g = []
    θ = []
    g = vcat(g, res["generators"])
    θ = vcat(θ, res["angles"])
    H = res["hamiltonian"]


    println("Evolution")
    # display(H*ψ)
    display(expectation_value(H,ψ))

    # Number operator evolution

    num_ops = PauliSum(N)
    for i in 1:N
        num_ops += 0.5*(Pauli( 0, 0, N) - Pauli(N, Z = [i]))
    end
    # display(num_ops)

    for i in 1:N
        if bitstring[i] == '1'
            num_ops = Pauli(N, X=[i]) * num_ops * Pauli(N, X=[i])
        end
    end 

    # return
    for (gi,θi) in zip(g,θ)
        num_ops = DBF.evolve(num_ops, gi, θi)
        DBF.coeff_clip!(num_ops, thresh=1e-5)
    end  
    display(expectation_value(num_ops, ψ))  
end

run()