using PauliOperators
using DBF
using Printf
using Random
using LinearAlgebra
using JLD2

function run()
    N = 50
    t=0.1
    U=0.001
    Random.seed!(2)
    H = DBF.hubbard_model_1D(N, t, U)
    DBF.coeff_clip!(H)
    N *= 2
    # Transform H to make |0101...> -> |000...> the most stable bitstring
    # for i in 1:2*N
    #     if i%2 == 0
    #         P=Pauli(2*N, X=[i])
    #         H = P * H * P
    #     end
    # end

    H0 = deepcopy(H)
    ψ = Ket([i%2 for i in 1:N])


    # ψ = Ket([0 for i in 1:2*N])
    display(ψ)
    e0 = expectation_value(H,ψ)

    @printf(" Reference = %12.8f\n", e0)

    g = Vector{PauliBasis{2*N}}([])
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

        #  @save "out_$(i).jld2" N ψ H0 H g θ
    end
end

run()