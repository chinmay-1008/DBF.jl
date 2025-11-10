using PauliOperators
using DBF
using JLD2
using NPZ
using LinearAlgebra
using Printf
Lx=4
Ly=4
println()
H = DBF.heisenberg_2D_zigzag(Lx,Ly, -1.0,-1.0,-1.0,x=.0,periodic=false)
display(H)
A = DBF.graph_adjacency(H)
L = DBF.graph_laplacian(H)
l, v = eigen(L)
fidx = 2
for i in 1:length(l)
    @printf(" %i L eigval: %12.8f\n", i, l[i])
    if l[i] > 1e-12
        global fidx = i
            break
    end
end
@show fidx
perm = sortperm(v[:, fidx])
A = A[perm,:][:,perm]
npzwrite("adjacency_Lx_$(Lx)_Ly_$(Ly)_matrix",A)
pauli_strings=[]
coeffs=[]
for (c,p) in H
   println(string(c))
   push!(pauli_strings,string(c))
   push!(coeffs,real(p))
end
println(perm)
# reorder H using perm
for i in 1:length(pauli_strings)
     ps = pauli_strings[i]
    ps_perm = join(ps[perm]) # permute the string according to perm
    pauli_strings[i] = ps_perm
end
using ITensors
using ITensorMPS
let
        N = length(pauli_strings[1])
        sites = siteinds("S=1/2", N; conserve_qns=false)
        os = OpSum()
        for (pstr, c) in zip(pauli_strings, coeffs)
                opsites = []
                for (i, p) in enumerate(pstr)
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
        H = MPO(os, sites)
        psi0 = random_mps(sites)
        nsweeps = 25
        maxdim = [10, 20, 50,100,200,400,400,600,600,800,800,1000,1000,1200,1200,1500,1500,1800,1800,2000,2000,2200,2200]
        cutoff = 1E-10
        energy, psi = dmrg(H, psi0; nsweeps, maxdim, cutoff)
        println("Final energy = $energy")
        @save "final_state_heisenberg_10_10_blas_16.jld2" energy psi
end