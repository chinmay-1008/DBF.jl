using ITensors
using ITensorMPS
using PauliOperators
using DBF
using LinearAlgebra

function run()
    N = 25
    # k = -1.0
    J = -1.0
    ratios = sort(unique(vcat(10 .^ range(-2, 2, length=20), 1.0)))
    # k_fixed = -1.0            # keep k constant (you can flip sign)
    # ratios = collect(-3.0:0.05:3.0)
    # ratios = range(-2, 2, length=51)
    # ratios = sort(unique(vcat(range(-2, 2, length=50), [-1.0, 1])))

    # ratios = 10 .^ range(-2, 1, length=20)  # covers k/J = 0.01 → 10
    # ratios = collect(ratios)

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

    
        sites = siteinds("S=1/2", N; conserve_qns=false)
        # return
        os = OpSum()
        
        for (pstr, c) in H
            opsites = []
    
            for (i, p) in enumerate(string(pstr))
                if p != 'I'
                    push!(opsites, (string(p), i))
                end
            end
        
            if length(opsites) > 0
                # Flatten operator-site pairs and splat the tuple for OpSum
                flat_opsites = Tuple([x for pair in opsites for x in pair])
                # println("Adding term: ", c, " * ", opsites)
                os += c, flat_opsites...
            end
        end 

        H = MPO(os, sites)
        psi0 = random_mps(sites)
        # nsweeps = 70
        # maxdim = [10, 20, 50,100, 100, 200,200,300,300,400,500,500,500,500,500,750,750,750,750,750,900,900,900,900,900,1000,1000,1000,1000,1000,1000,1000,1000,1000,1000,1000,1000,1000,1000,1000,1000,1000,1000,1000,1000,1000,1000,1000,1000,1000,1000,1000,1000,1000,1000,1000,1000,1000,1000,1000,1000,1000,1000,1000,1000,1000,1000,1000,1000,1000]
    
        # nsweeps = 62
        # maxdim = [10, 20, 50,100, 100, 200,200,300,300,1000,1000,1000,1000,1000,1000,1000,1000,1000,1000,1000,1000,1000,1000,1000,1000,1000,1000,1000,1000,1000,1000,1000,1000,1000,1000,1000,1000,1000,1000,1000,1000,1000,1000,1000,1500,1500,1500,1500,1500,1500,1500,1500,1500,1500,1500,1500,1500,1500,1500,1500,1500,1500]
    
        nsweeps = 20
        maxdim = [10, 20, 50,100, 100, 200,200,300,300,1000,1000,1000,1500,1500,1500,1500,1500,1500,1500,1500]
        cutoff = 1E-10
        energy, psi = dmrg(H, psi0; nsweeps, maxdim, cutoff)
        println("Final energy = $energy")
        push!(exact_energies, real(energy)/N)
    end

    display(exact_energies)

    return
end


#@profilehtml run(4)
run()