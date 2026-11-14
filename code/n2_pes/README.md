# N2 potential energy curve: references vs vDBF

N2 in STO-3G with all electrons: 10 orbitals, 7α + 7β electrons, 20 qubits.
R runs from 0.8 to 3.0 Å in steps of 0.1 (23 points). vDBF (`dbf_groundstate`)
runs in four orbital bases and is compared with RHF, UHF, RCCSD, RCCSD(T),
UCCSD, UCCSD(T) and FCI.

| File | What it does |
|---|---|
| `n2_pes.py` | SCF scans, CC and FCI references, UCCSD NOs, Boys localization, integrals for each R |
| `build_hamiltonians.jl` | Turns every `data/R_*` into PauliSum `.jld2` files with `../integrals/uhf_integrals_to_pauli.jl`, all in one Julia session |
| `run_dbf_pes.jl` | Runs vDBF for each R and basis and saves the results |
| `extrapolate.py` | Variance extrapolation of every trajectory, with the vdbf-paper tail-fit rule |
| `plot_pes.py` | Draws the PES (vDBF final and extrapolated), the error vs FCI and ⟨S²⟩, and prints the non-parallelity error |
| `Project.toml` | Julia environment for `run_dbf_pes.jl` (DBF from `../..`, PauliOperators 3.1 from the registry) |

## Run

```bash
cd code/n2_pes
python -u n2_pes.py > results/logs/n2_pes.out 2>&1                       # ~5 min
julia --project=../integrals build_hamiltonians.jl > results/logs/build_hamiltonians.out 2>&1
for b in mo lmo no lno; do julia --project=. run_dbf_pes.jl --bases $b > results/logs/dbf_$b.out 2>&1 & done; wait
python extrapolate.py
python plot_pes.py
```

To rerun some points: `run_dbf_pes.jl --bases no --R 1.10,2.50 --overwrite`.
A point that already has `dbf_<basis>.json` is skipped otherwise.
`results/dbf_<basis>.csv` is rebuilt from all the JSON files each time.

vDBF settings (the script defaults):
- `operator_truncation = CoeffTruncation(1e-5)` (`--coeff`)
- `gradient_truncation = CoeffTruncation(1e-6)` (`--grad`)
- `energy_lowering_thresh = 1e-6` (`--energy-lowering`)
- `conv_thresh = 1e-5` (`--conv`), `max_iter = 200` (`--max-iter`)
- `max_rots_per_grad = 50` (`--max-rots`)
- `compute_pt2 = false`, `compute_pt2_error = false`, `compute_var_error = true`
- a `:compact` checkpoint every 10 iterations (`--checkpoint-interval`) and at the end
- `SparsePauliVector` storage (`--spv false` uses `PauliSum` instead)

PauliOperators comes from https://github.com/nmayhall/PauliOperators.jl `main`
(pinned in `Manifest.toml`). It is about 3.5x faster than the registered
v3.1.0 here and gives identical energies.

## Variance extrapolation

`extrapolate.py` applies the vdbf-paper protocol
(`corrected_energy_vs_corrected_variance.ipynb` in
https://github.com/chinmay-1008/data-vdbf1) to each R and basis.

1. Use one point per macro-iteration, iterations 1..n.
2. Corrected energy = E − accumulated truncation error. Corrected variance = Var − accumulated variance error.
3. For every final window of at least 10 points, fit corrected E against corrected Var with a line and a quadratic.
4. Keep the window that minimizes ½|E0_lin − E0_quad| + (1 − R²_lin).
5. E0 = mean of the two intercepts, and its uncertainty = half their difference.

Points that stop with fewer than 10 iterations get NaN and a note.

## Orbital bases

| Basis | Orbitals | Reference determinant |
|---|---|---|
| `mo` | canonical UHF | the UHF determinant |
| `lmo` | UHF; 1s core kept canonical; valence-occupied and virtual blocks Boys-localized separately | the UHF determinant |
| `no` | UCCSD natural orbitals, in order of decreasing occupation | first 7 NOs per spin, energy just above E_UHF |
| `lno` | NOs; the 2 core-like NOs kept; the rest of the occupied-like and virtual-like blocks localized separately | same space as `no` |

Localization rotates orbitals only within each block. So `mo`/`lmo` share one
reference determinant and `no`/`lno` share another. FCI is identical in all
four bases, and the script checks that.

## SCF: continuation and stability

- **Convergence:** no SCF is accepted unless it reports converged, has an orbital gradient below 1e-6, and has aufbau occupations. A failed attempt is retried from the previous geometry's density matrix: DIIS, then level shift 0.2 and 0.5, then a Fermi-smearing anneal, then Newton. If all of them fail, the script stops and names R.
- **RHF:** swept in increasing R, each point starting from the previous density. Internal instabilities are followed until the solution is stable. Each point's RHF→UHF instability is recorded in the `RHF_external_unstable` column.
- **UHF:** a forward sweep (seeded from the RHF density) and a backward sweep (seeded from the 3.0 Å solution), each starting from the previous geometry's density. Internal instabilities are followed at every point.
  - PySCF's UHF stability check starts from α = β vectors, so it never sees the spin-symmetry-breaking direction. Where the UHF is still spin-symmetric and the RHF is RHF→UHF unstable, the script follows the RHF external eigenvector instead. That is what breaks symmetry at 1.2 Å.
  - At each R the lower of the two sweeps is kept. In this run they agree at every point, so there is only one UHF solution.
- **FCI:** the lowest state of the (7α, 7β) sector, the same sector vDBF works in. ⟨S²⟩ is stored as well.
- **CCSD:** if it doesn't converge, one retry uses a level shift. A point that still fails is recorded as `converged = False` rather than dropped. RCCSD fails from about 1.7 Å onward, which is the failure being shown. The plot draws these points as open markers.

## Checks

- **Python:**
  - The integrals in every basis reproduce E_UHF to 1e-8.
  - The reference determinant energy equals E_UHF in `mo` and `lmo`.
  - FCI is the same in every basis and matches PySCF FCI on the RHF orbitals to 1e-8.
  - Localization keeps each block's projector and orthonormality, and the Boys optimum is confirmed by at least two starts reaching it.
- **Julia converter:** ⟨ref|H|ref⟩ matches Python (≤ 1.1e-12 at every point). Exact diagonalization of the (7,7) sector (dimension 14400) equals FCI at 1.1 and 2.5 Å, to 6e-13 and 1e-13. Exact diagonalization is off by default because of the matrix size; to run it, pass `--ed-max-dim 15000` (it builds a 3.3 GB dense matrix).
- **`run_dbf_pes.jl`:** ⟨0|XHX|0⟩ equals the stored reference energy. A warning is printed if E_DBF minus the accumulated truncation error falls below FCI.

## Reference must be |0…0⟩

`dbf_groundstate` builds its generators from `[P, H]` with `P = I + Σ Z_i`.
Up to constants, that P is the particle-number operator. A molecular
Hamiltonian conserves particle number, so starting from any other
determinant the commutator is exactly zero and the flow has nothing to do.

`run_dbf_pes.jl` therefore conjugates H by X on every occupied qubit, as the
old `run_lno.jl` did, and starts from `Ket(0)`. Each term only changes sign,
by (−1)^popcount(z & occupied). The stored `.jld2` keeps the original H and
`reference_ket`. The same applies to the Hubbard Hamiltonians in
`../integrals`.

## Outputs

- `data/R_<R>/`
  - integrals: `<basis>_*.npy`
  - `metadata.json`: every reference energy, NO occupations and localization diagnostics
  - `hamiltonian_<basis>.jld2`
  - `dbf_<basis>.json`: final numbers plus the energy, variance and accumulated-error history per macro-iteration
  - `dbf_<basis>_ckpt.jld2`: compact DBF checkpoint
- `results/references.csv`: R, E_nuc, E_RHF, the RHF→UHF instability flag, E_UHF, ⟨S²⟩_UHF, which sweep was kept, E for RCCSD, RCCSD(T), UCCSD and UCCSD(T) with convergence flags, E_FCI and ⟨S²⟩_FCI
- `results/dbf_<basis>.csv`:
  - energies: E_DBF, the accumulated truncation error, E_DBF − error, variance and accumulated variance error, error vs FCI
  - run statistics: number of iterations and rotations, final number of terms, wall time
- `results/extrapolation_<basis>.csv`: the chosen window, E0 (linear, quadratic, average), uncertainty, R², error vs FCI; `_scan.csv` has every candidate window
- `results/n2_pes.png` and `.pdf`; logs in `results/logs/`
