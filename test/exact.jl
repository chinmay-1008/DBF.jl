using ITensors
using ITensorMPS
using PauliOperators
using DBF



# println("Number of terms in Hamiltonian: ", length(coeffs))

function run()
    Lx = 3
    Ly = 3
    N = Lx * Ly
    H = DBF.heisenberg_2D(Lx, Ly, -1, -1, -1, x=.0, periodic = false)
#     display(H)
# return
    # N = 50
    # H = DBF.heisenberg_1D(N, -0, -1, -1, x=.0)

    DBF.coeff_clip!(H)
    display(H)
    # return
    
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
            println("Adding term: ", c, " * ", opsites)
            os += c, flat_opsites...
        end
    end 
    display(os)

    
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
end

function temp_run()
    Nx = 3
    Ny = 4
    coord_to_index(i, j) = i + j * Nx + 1
    # index(i, j) = (i - 1) * Ny + j

    index(i, j) = isodd(i) ? (i - 1) * Ny + j : i * Ny - j + 1
    m = [1 2 3;
        6 5 4;
        7 8 9]
    for x in 1:Nx
        for y in 1:Ny
            println("i ", x, " j ", y)
            display(index(x, y))
            # display(m[x, y])
        end
    end
end

run()
