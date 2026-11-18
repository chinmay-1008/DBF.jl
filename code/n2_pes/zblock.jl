# The Z operator of a block of kets, built the way D1c is built for |0>, |a>, |b>: one Z for every qubit
# no ket flips, and on the qubits A that some ket flips, one- and two-qubit Z terms (Z_q, Z_q Z_r) chosen
# so that the kets are exactly the top eigenspace.  D1c replaces Z_H + Z_L by Z_H Z_L (and adds the c
# terms) for the same reason.  Used by frozen_core/sv_block.jl and run_block_pp.jl (--rank dz).
#
#     D = sum_{q not in A} Z_q + f,   f = sum_{q in A} a_q Z_q + sum_{q<r in A} b_qr Z_q Z_r
#
# Z_q|y> = (1 - 2 y_q)|y> in the vacuum frame, so every ket (and the reference) sits at the top of the
# first sum.  The coefficients are integers in -cmax..cmax, found by exhaustive search: on the 2^|A|
# patterns of the qubits in A, f must take one value T on every ket's pattern and at most T - 2 on all
# others, the same gap as flipping one qubit outside A.  Among those, the fewest nonzero terms win, then
# the smallest sum of |coefficients|.  Returns (terms, A) with terms = [(z mask, coefficient)], or nothing
# when no one- and two-qubit f exists (|A| > 4 is not searched).

function z_block_operator(kets::AbstractVector{UInt64}, N::Integer; cmax::Int=2)
    A = [q for q in 0:N-1 if any((y >> q) & 1 == 1 for y in kets)]
    nA = length(A)
    nA <= 4 || return nothing
    masks = UInt64[UInt64(1) << q for q in A]
    for i in 1:nA, j in i+1:nA
        push!(masks, (UInt64(1) << A[i]) | (UInt64(1) << A[j]))
    end
    nT = length(masks)
    # pattern p (bits over A) -> the ket bits it stands for; its value of each term
    pat(p) = reduce(|, (UInt64(1) << A[k] for k in 1:nA if (p >> (k - 1)) & 1 == 1); init=UInt64(0))
    P = [pat(p) for p in 0:(1 << nA)-1]
    V = [isodd(count_ones(m & b)) ? -1 : 1 for b in P, m in masks]
    top = [b in kets for b in P]
    best, bestkey = nothing, (typemax(Int), typemax(Int))
    c = fill(-cmax, nT)
    f = zeros(Int, length(P))
    while true
        nz = count(!=(0), c)
        l1 = sum(abs, c)
        if (nz, l1) < bestkey
            mul!(f, V, c)
            T = f[findfirst(top)]
            if all(f[k] == T for k in eachindex(f) if top[k]) && all(f[k] <= T - 2 for k in eachindex(f) if !top[k])
                best, bestkey = copy(c), (nz, l1)
            end
        end
        k = 1                                     # odometer over -cmax..cmax
        while k <= nT && c[k] == cmax
            c[k] = -cmax
            k += 1
        end
        k > nT && break
        c[k] += 1
    end
    best === nothing && return nothing
    terms = Tuple{UInt64,Float64}[(UInt64(1) << q, 1.0) for q in 0:N-1 if !(q in A)]
    append!(terms, [(m, Float64(x)) for (m, x) in zip(masks, best) if x != 0])
    return terms, A
end

# D on a computational-basis state (vacuum frame bits b)
z_value(terms, b::UInt64) = sum(x * (isodd(count_ones(m & b)) ? -1.0 : 1.0) for (m, x) in terms)

# "-Z4 + Z5 + Z4 Z5" style text for the terms on A
function z_terms_string(terms, A)
    s = String[]
    for (m, x) in terms
        qs = [q for q in 0:63 if (m >> q) & 1 == 1]
        all(q in A for q in qs) || continue
        push!(s, (x < 0 ? "-" : "+") * (abs(x) == 1 ? "" : string(Int(abs(x)))) * join(["Z$q" for q in qs], " "))
    end
    return join(s, " ")
end
