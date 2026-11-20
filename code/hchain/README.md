# H-chain size scan: Heisenberg vs Schrödinger

How many stored coefficients does each picture need for a given energy error, and how does that grow with system size?

- **Systems:** linear H_n chains in STO-3G, n = 4–12 (8–24 qubits).
- **Spacings:** 1.0 Å (weakly correlated) and 2.0 Å (strongly correlated; UHF breaks spin symmetry).
- **Method:** vDBF on UNO + 0.05 S², as in the N2 UNO runs.

## Pipeline

```bash
python make_hchain.py --n 4,6,8,10,12                  # PySCF: UNO integrals, FCI (+ vector) -> data/H<n>/R_<d>/
julia --project=../integrals ../integrals/uhf_integrals_to_pauli.jl data/H8/R_2.00 --basis uno --ed-max-dim 5000
julia --project=../integrals build_s2_hchain.jl data/H8/R_2.00
python schrodinger_bestk.py data/H8/R_2.00             # best energy with k determinants -> bestk.json
bash run_vdbf_local.sh "8 2.00 1e-6"                   # vDBF (n2_pes/run_dbf_pes.jl) -> dbf_uno_s2pen0.05[_c<coeff>].json
bash analyze_runs.sh                                   # statevector.jl on every finished run -> results/sv_H<n>_<d>_c<coeff>.json
python scan_plot.py                                    # results/hchain_scan.png
```

H10 and H12 at tight cutoffs run on the cluster. They are added to the N2 bundle in `n2_pes/cluster_pes`: `stage_hchain.sh`, `submit_hchain.sh`, `hchain_run_one.sh` and `memory_hchain.txt`.

## What is measured

| Picture | Quantity | Source |
|---|---|---|
| Heisenberg | Pauli terms vDBF carries per iteration (`len(H)`) vs the true ⟨H⟩ of the circuit then | run log; exact dense replay in `statevector.jl` |
| Schrödinger, ideal | best energy with the k largest FCI determinants (H diagonalized in that space) | `schrodinger_bestk.py` |
| Schrödinger, same circuit | vDBF's final circuit applied to the state, amplitudes below τₛ dropped after every rotation: energy and peak amplitudes kept | `statevector.jl` |

`statevector.jl` reproduces the N2 2.3 Å replay exactly. It gives the same true ⟨H⟩ at every iteration, and the same truncated-state energies and amplitude counts.
