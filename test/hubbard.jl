using DBF
using PauliOperators
using LinearAlgebra

function run()
    L = 3
    H = DBF.hubbard_model_1D(L, 1.0, 2.0)
    coeff_clip!(H)
    N = 2*L

    for i in 1:N
        if i%2 == 0
            H = Pauli(N, X=[i]) * H * Pauli(N, X=[i])
        end
    end 

    eigvals, eigvecs = eigen(Matrix(H))
    idx_min = argmin(eigvals)

    min_eig = eigvals[idx_min]
    min_vec = eigvecs[:, idx_min]

    for i in 0:2^N - 1
        amp = min_vec[i+1]
        if abs(amp) > 1e-10
            k = Ket(N, i)
            bits = replace(string(k), ['|','>'] => "")  # get bitstring like "0110"
            occupied = findall(x -> x == '1', bits)     # indices (1-based) of occupied sites
            println("State: ", string(k), 
                    "  occupied sites: ", occupied, 
                    "  amplitude: ", amp)
        end
    end

    # ψ = Ket([(i ÷ 2) % 2 for i in 1:N])
    # ψ = Ket([i%2 for i in 1:N])

    ψ = Ket([0 for i in 1:N])
    display(ψ)
    # return
    @time res = DBF.dbf_groundstate(H, ψ, 
                                verbose=1, 
                                max_iter=120, conv_thresh=1e-3, 
                                evolve_coeff_thresh=1e-4,
                                grad_coeff_thresh=1e-5,
                                energy_lowering_thresh=1e-5,
                                clifford_check=true,
                                compute_pt2_error=true,
                                max_rots_per_grad=50)

    return
end


run()