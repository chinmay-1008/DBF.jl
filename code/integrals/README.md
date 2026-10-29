# Integrals → PauliSum

Two steps: Python (PySCF) builds one- and two-body integrals in a chosen
orbital basis, then Julia maps them to a Jordan-Wigner `PauliSum` and saves
it as `.jld2`.

| File | What it does |
|---|---|
| `hubbard_uhf_integrals.py` | 2D Hubbard → UHF → (UCCSD NOs) → localization → `.npy` integrals + `metadata.json` |
| `uhf_integrals_to_pauli.jl` | `.npy` + `metadata.json` → `PauliSum` → `hamiltonian_<basis>.jld2` |
| `Project.toml` / `Manifest.toml` | Julia environment for the converter (PauliOperators 3.1, NPZ, JLD2, JSON). It doesn't depend on DBF. |

## Hubbard (UHF)

```bash
cd code/integrals

# 8x8, t=1, U=4, mu=1.17, N=56, PBC (the run_lno.py defaults), LNO basis
python -u hubbard_uhf_integrals.py --basis lno > lno.out 2>&1
julia --project=. uhf_integrals_to_pauli.jl data/8x8_pp_N56_U4_mu1.17 --basis lno

# small test, every basis, with FCI / exact-diagonalization checks
python -u hubbard_uhf_integrals.py --nx 2 --ny 3 --U 4 --mu 0 --nelec 6 \
    --open-x --open-y --basis all
julia --project=. uhf_integrals_to_pauli.jl data/2x3_oo_N6_U4_mu0 --basis lno
```

Bases (`--basis`, several allowed, or `all`):
`site`, `mo` (canonical UHF), `lmo` (localized UHF), `no` (UCCSD natural
orbitals), `lno` (localized NOs). The Python script prints the Julia command
for each basis it wrote.

## Conventions

- Integrals are in chemist notation `(pq|rs)`: `h1_a`, `h1_b`, `eri_aa`,
  `eri_bb`, `eri_ab`.
- Qubit order: alpha orbitals `1..n`, then beta orbitals `n+1..2n`. Lattice
  sites are numbered in snake order.
- `H` includes the `-mu*N` term. Add `energy_offset` (= `mu*N`, stored in
  `metadata.json` and the `.jld2`) to get the mu = 0 energy.
- The `.jld2` holds `H` (so `@load file H` still works) plus `n_alpha`,
  `n_beta`, `reference_ket`, `reference_energy`, `e_uhf`, `e_fci`, `mu` and
  `energy_offset`.
- `dbf_groundstate` needs the reference to be `|0...0>`: its generator pool
  `[I + Σ Z_i, H]` is zero for a number-conserving H otherwise. Conjugate H by
  X on the occupied qubits first (`to_vacuum_reference` in
  `../n2_pes/run_dbf_pes.jl`).
- `uhf_integrals_to_pauli.jl` only runs `run(ARGS)` when executed as a script,
  so it can also be `include`d and `run([...])` called in a loop
  (`../n2_pes/build_hamiltonians.jl`).

## Checks (any failure stops the run before files are written)

- Python: the integrals reproduce E_UHF in every basis, and localization stays
  inside the occupied and virtual blocks. On small lattices (at most
  `--fci-max-sites`), FCI gives the same energy in every basis.
- Julia: `H` is Hermitian, and `<ref|H|ref>` equals the reference-determinant
  energy from Python. When the sector dimension is at most `--ed-max-dim`, the
  lowest eigenvalue in the `(n_alpha, n_beta)` sector equals E_FCI.
- In the `no` and `lno` bases the reference determinant (first `n_occ` natural
  orbitals) is **not** the UHF determinant, so `<ref|H|ref>` is higher than
  E_UHF. The script compares against the right number for each basis.

## Cost

The converter loops over `n^4` integrals per spin block. For the 8x8 case
(`n = 64`, 128 qubits) with dense rotated integrals, a test build took about
80 s and 7.4 GB of RAM and gave about 16.8M Pauli terms. `Pauli` strings are
`Int128`, so 128 spin orbitals is the maximum.
