"""
UHF integrals (.npy) -> Jordan-Wigner PauliSum (.jld2).

Reads the metadata.json written by hubbard_uhf_integrals.py, so the number of
orbitals, electron counts and file names come from the integral run rather than
being typed into the script.

    H = constant + Σ_pq h^α_pq a†_pα a_qα + Σ_pq h^β_pq a†_pβ a_qβ
      + ½ Σ_pqrs (pq|rs)_αα a†_pα a†_rα a_sα a_qα
      + ½ Σ_pqrs (pq|rs)_ββ a†_pβ a†_rβ a_sβ a_qβ
      +   Σ_pqrs (pq|rs)_αβ a†_pα a†_rβ a_sβ a_qα

Qubits: alpha orbitals 1..n, beta orbitals n+1..2n (block ordering, as in
code/int_pauli.jl).  Pauli strings are Int128, so 2n ≤ 128.

Checks before anything is written:
  * Hermiticity: every coefficient must be real to 1e-10.
  * Reference determinant: <ref|H|ref> must equal the Slater-Condon energy the
    Python script stored for this basis (E_UHF in the mo/lmo bases, a
    different number in no/lno -- see the Python docstring).
  * Exact diagonalization, when the (n_alpha, n_beta) sector has at most
    --ed-max-dim states and the Python run stored an FCI energy: the lowest
    eigenvalue of H in that sector must equal E_FCI.
Both energy checks run on the unclipped H at 1e-8; the shift that --clip
causes is reported separately.

Usage:
    julia --project=code/integrals code/integrals/uhf_integrals_to_pauli.jl \\
        <input_dir> [--basis lno] [--output file.jld2] [--tol 1e-10] \\
        [--clip 1e-8] [--ed-max-dim 2000]

The .jld2 holds `H` (load with `@load file H` as before) plus the metadata
needed to use it: basis, n_orbitals, n_alpha, n_beta, constant, mu,
energy_offset (add to energies of H to undo -mu*N), reference_ket,
reference_energy, e_uhf and (when available) e_fci.
"""

using PauliOperators
using LinearAlgebra
using NPZ
using JLD2
using JSON
using Printf

"""
    jw_operator(N, i; dagger)

The two Pauli terms of a†_i (or a_i) under Jordan-Wigner, as a tuple:
a†_i = ½(X_i - iY_i) Z_{<i},   a_i = ½(X_i + iY_i) Z_{<i}.
"""
function jw_operator(N::Int, i::Int; dagger::Bool)
    bit = Int128(1) << (i - 1)
    z_string = bit - Int128(1)
    x_term = Pauli(PauliBasis(Pauli{N}(1, z_string, bit)))
    y_term = Pauli(PauliBasis(Pauli{N}(1, z_string | bit, bit)))
    return (0.5 * x_term, (dagger ? -0.5im : 0.5im) * y_term)
end

"""
    pair_cache(ops_left, ops_right)

All products ops_left[i] * ops_right[j], each a 4-tuple of Paulis.  Caching
a†_p a†_r and a_s a_q turns every two-body term into 16 single Pauli
products instead of three PauliSum multiplications.
"""
function pair_cache(ops_left, ops_right)
    M = length(ops_left)
    return [Tuple(a * b for a in ops_left[i] for b in ops_right[j]) for i in 1:M, j in 1:M]
end

@inline function add_products!(H::PauliSum{N,ComplexF64}, val, A, B) where N
    for a in A, b in B
        p = a * b
        key = PauliBasis(p)
        H[key] = get(H, key, zero(ComplexF64)) + val * coeff(p)
    end
end

function check_pair_symmetry(eri, label)
    err = maximum(abs.(eri .- permutedims(eri, (3, 4, 1, 2))))
    err < 1e-12 || error("$label is not symmetric under (pq|rs) = (rs|pq) ($err)")
end

function uhf_hamiltonian(h1_a, h1_b, eri_aa, eri_bb, eri_ab; tol=1e-10)
    n = size(h1_a, 1)
    N = 2n
    N <= 128 || error("$N spin orbitals exceed the 128-qubit Int128 Pauli limit")
    println("Building Hamiltonian from separate spin blocks...")
    println("  -> System: $n spatial orbitals ($N spin orbitals)")

    ops_dag = [jw_operator(N, i, dagger=true) for i in 1:N]
    ops_col = [jw_operator(N, i, dagger=false) for i in 1:N]

    H = PauliSum(N, ComplexF64)

    # --- PART 1: ONE-BODY TERMS ---
    println("  -> Processing 1-body terms (alpha & beta)...")
    for q in 1:n, p in 1:n
        abs(h1_a[p, q]) > tol && add_products!(H, h1_a[p, q], ops_dag[p], ops_col[q])
        abs(h1_b[p, q]) > tol && add_products!(H, h1_b[p, q], ops_dag[p+n], ops_col[q+n])
    end

    # --- PART 2: TWO-BODY TERMS ---
    # Same spin: a†_r a†_p a_q a_s = a†_p a†_r a_s a_q and (rs|pq) = (pq|rs), so
    # each unordered (pq),(rs) pair is visited once and the ½ cancels the 2.
    # Terms with p == r or q == s vanish (a†_p a†_p = 0) and are skipped; that
    # includes every pq == rs term, so no diagonal ½ is left over.
    println("  -> Caching pair operators...")
    dag_pairs = pair_cache(ops_dag, ops_dag)   # a†_i a†_j
    col_pairs = pair_cache(ops_col, ops_col)   # a_i a_j

    check_pair_symmetry(eri_aa, "eri_aa")
    check_pair_symmetry(eri_bb, "eri_bb")
    println("  -> Processing 2-body terms (AA, BB)...")
    for s in 1:n, r in 1:n, q in 1:n, p in 1:n
        pq, rs = (q - 1) * n + p, (s - 1) * n + r
        (pq > rs || p == r || q == s) && continue
        val = eri_aa[p, q, r, s]
        abs(val) > tol && add_products!(H, val, dag_pairs[p, r], col_pairs[s, q])
        val = eri_bb[p, q, r, s]
        abs(val) > tol && add_products!(H, val, dag_pairs[p+n, r+n], col_pairs[s+n, q+n])
    end

    # Opposite spin: no ½, the αβ and βα orderings are the same operator.
    println("  -> Processing 2-body terms (AB)...")
    for s in 1:n, r in 1:n, q in 1:n, p in 1:n
        val = eri_ab[p, q, r, s]
        abs(val) > tol && add_products!(H, val, dag_pairs[p, r+n], col_pairs[s+n, q])
    end

    coeff_clip!(H, 1e-14)
    println("  -> Hamiltonian construction complete: $(length(H)) Pauli terms.")
    return H
end

function reference_ket(N, n, occ_alpha, occ_beta)
    v = Int128(0)
    for i in occ_alpha
        v |= Int128(1) << i            # 0-based alpha orbital i -> qubit i+1
    end
    for i in occ_beta
        v |= Int128(1) << (n + i)      # 0-based beta orbital i -> qubit n+i+1
    end
    return Ket{N}(v)
end

"""
    sector_ground_energy(H, n, n_alpha, n_beta)

Lowest eigenvalue of H among determinants with n_alpha electrons in qubits
1..n and n_beta in qubits n+1..2n.  Errors if H connects the sector to
anything outside it (a number- or Sz-nonconserving term).
"""
function sector_ground_energy(H::PauliSum{N,T}, n, n_alpha, n_beta) where {N,T}
    alpha = [v for v in Int128(0):(Int128(1) << n) - 1 if count_ones(v) == n_alpha]
    beta = [v for v in Int128(0):(Int128(1) << n) - 1 if count_ones(v) == n_beta]
    states = [a | (b << n) for b in beta for a in alpha]
    index = Dict(v => k for (k, v) in enumerate(states))

    M = zeros(ComplexF64, length(states), length(states))
    for (col, v) in enumerate(states)
        for (ket, c) in H * Ket{N}(v)
            abs(c) < 1e-12 && continue
            row = get(index, ket.v, 0)
            row == 0 && error("H leaves the (n_alpha, n_beta) sector")
            M[row, col] += c
        end
    end
    herm_err = maximum(abs.(M - M'))
    herm_err < 1e-10 || error("sector matrix is not Hermitian ($herm_err)")
    return eigvals(Hermitian(M))[1], length(states)
end

function parse_cli(args)
    length(args) >= 1 || error("usage: uhf_integrals_to_pauli.jl <input_dir> [--basis lno] ...")
    opts = Dict{String,String}("basis" => "lno", "tol" => "1e-10", "clip" => "1e-8",
                               "ed-max-dim" => "2000", "output" => "")
    input_dir = args[1]
    i = 2
    while i <= length(args)
        key = lstrip(args[i], '-')
        haskey(opts, key) || error("unknown option $(args[i])")
        opts[key] = args[i+1]
        i += 2
    end
    return input_dir, opts
end

function run(args)
    input_dir, opts = parse_cli(args)
    basis = opts["basis"]
    meta = JSON.parsefile(joinpath(input_dir, "metadata.json"))
    haskey(meta["bases"], basis) ||
        error("basis '$basis' not in metadata; available: $(join(keys(meta["bases"]), ", "))")
    entry = meta["bases"][basis]
    n = meta["n_orbitals"]
    N = 2n
    n_alpha, n_beta = meta["n_alpha"], meta["n_beta"]
    constant = Float64(meta["constant"])
    output = isempty(opts["output"]) ? joinpath(input_dir, "hamiltonian_$(basis).jld2") : opts["output"]

    println("Input:  $input_dir  (basis $basis)")
    println("System: $(meta["model"]), $n orbitals, n_alpha = $n_alpha, n_beta = $n_beta")

    load(key) = npzread(joinpath(input_dir, entry["files"][key]))
    @time H = uhf_hamiltonian(load("h1_alpha"), load("h1_beta"),
                              load("eri_aa"), load("eri_bb"), load("eri_ab");
                              tol=parse(Float64, opts["tol"]))
    sum!(H, Pauli(N) * constant)

    # --- CHECKS ---
    imag_err = maximum(abs ∘ imag, values(H))
    @printf(" max |imag(coeff)|         = %.3e\n", imag_err)
    imag_err < 1e-10 || error("H is not Hermitian")

    ψ = reference_ket(N, n, entry["occupied_alpha"], entry["occupied_beta"])
    e_ref = real(expectation_value(H, ψ))
    e_ref_py = entry["reference_determinant_energy"] + constant
    println("------------------------------------------------")
    @printf(" Reference determinant  <ref|H|ref> = %16.10f\n", e_ref)
    @printf("                        Python      = %16.10f   (diff %.2e)\n", e_ref_py, e_ref - e_ref_py)
    @printf(" E_UHF (with mu)                    = %16.10f\n", meta["E_UHF_with_mu"] + constant)
    abs(e_ref - e_ref_py) < 1e-8 || error("reference determinant energy does not match Python")

    e_fci = nothing
    if haskey(entry, "fci_energy")
        dim = binomial(n, n_alpha) * binomial(n, n_beta)
        if dim <= parse(Int, opts["ed-max-dim"])
            e_ed, dim = sector_ground_energy(H, n, n_alpha, n_beta)
            e_fci = entry["fci_energy"] + constant
            @printf(" Sector ED ground state (dim %5d) = %16.10f\n", dim, e_ed)
            @printf("                        PySCF FCI   = %16.10f   (diff %.2e)\n", e_fci, e_ed - e_fci)
            abs(e_ed - e_fci) < 1e-8 || error("sector ground state does not match FCI")
        else
            println(" (sector dimension $dim > --ed-max-dim; skipping exact diagonalization)")
        end
    end

    # --- CLIP AND SAVE ---
    n_terms = length(H)
    coeff_clip!(H, parse(Float64, opts["clip"]))
    e_ref_clipped = real(expectation_value(H, ψ))
    @printf(" Clipped at %s: %d -> %d terms, <ref|H|ref> shift %.2e\n",
            opts["clip"], n_terms, length(H), e_ref_clipped - e_ref)
    println("------------------------------------------------")

    jldsave(output; H, basis, n_orbitals=n, n_alpha, n_beta, constant,
            mu=meta["mu"], energy_offset=meta["energy_offset"],
            reference_ket=ψ, reference_energy=e_ref_py,
            e_uhf=meta["E_UHF_with_mu"] + constant, e_fci)
    println("Wrote $output")
end

if abspath(PROGRAM_FILE) == @__FILE__
    @time run(ARGS)
end
