using ITensors
using ITensorMPS


# Snake-like mapping function
function siteid_snake(x, y)
    if isodd(y)
        return (y-1)*Lx + x
    else
        return y*Lx - x + 1
    end
end

function run()
    
    # Parameters
    Lx, Ly = 2, 2
    N_sites = Lx * Ly
    U, t = 1.0, 1.0
    N_e = N_sites             # Specify electron number


    sites = siteinds("Electron", N_sites)

    ampo = OpSum()
    # Hopping: x-direction
    for y in 1:Ly
        for x in 1:(Lx-1)
            i = siteid_snake(x, y)
            j = siteid_snake(x+1, y)
            ampo += -t, "Cdagup", i, "Cup", j
            ampo += -t, "Cdagup", j, "Cup", i
            ampo += -t, "Cdagdn", i, "Cdn", j
            ampo += -t, "Cdagdn", j, "Cdn", i
        end
    end
    # Hopping: y-direction
    for x in 1:Lx
        for y in 1:(Ly-1)
            i = siteid_snake(x, y)
            j = siteid_snake(x, y+1)
            ampo += -t, "Cdagup", i, "Cup", j
            ampo += -t, "Cdagup", j, "Cup", i
            ampo += -t, "Cdagdn", i, "Cdn", j
            ampo += -t, "Cdagdn", j, "Cdn", i
        end
    end
    # On-site term
    for x in 1:Lx
        for y in 1:Ly
            i = siteid_snake(x, y)
            ampo += U, "Nupdn", i
        end
    end
    H = MPO(ampo, sites)

    psi0 = randomMPS(sites, N_e)   # With fixed electron number

    sweeps = Sweeps(10)
    maxdim!(sweeps, 20,40,80,150,300,500,800,1200,1600,2000)
    cutoff!(sweeps, 1E-10)

    energy, psi = dmrg(H, psi0, sweeps; outputlevel=2)

    println("Ground state energy = $energy")

end

run()
