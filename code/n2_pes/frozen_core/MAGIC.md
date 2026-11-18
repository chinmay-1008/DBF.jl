# Magic and N₂ bond breaking

Does magic (non-stabilizerness) show a signature as the N₂ triple bond breaks, and
does vDBF reproduce it?

**Write-up for the PI (Claude Doc, with all figures):**
[N₂ bond breaking seen through magic](https://claude.ai/artifact/H1NXjC9nYs4ie3fMa1tTfQ)

Related: the vDBF results these states come from, including 10⁻⁵ vs 10⁻⁶ accuracy
and runtimes, are in [DBF Flows on N₂](https://claude.ai/artifact/3fxShjpwAYzXn3is1JQKum)
(section 7).

## Short answer

- Magic follows the bonds as they break: first the two π bonds, then σ. Total M₂ peaks
  at 2.1 Å, when σ is half broken. After that it stays high as a collective property
  of the separated atoms.
- The total M₂ does not separate the π break from the σ break. The magic of the σ and
  π orbital groups taken separately does:
  - the π group flattens at the π half-break (1.72 Å);
  - the σ group peaks at the σ half-break (2.09 Å).
- vDBF at cutoff 10⁻⁶ reproduces every curve to within 0.01 nats. At 10⁻⁵ it
  overshoots by up to 0.08 nats near 2.0–2.1 Å.

## System and states

- N₂, STO-3G, frozen 1s core: 8 active orbitals, 5α + 5β electrons, 16 qubits
  (Jordan–Wigner: α orbital i is qubit i, β is qubit 8+i). R = 0.8–3.0 Å in steps
  of 0.1 Å (23 points).
- Orbitals are the UHF natural orbitals (UNO, `data_unocore/`), the same orbitals
  the vDBF runs used.
- States:
  - **Exact (FCI):** lowest eigenvector of H in the (5α, 5β) block, 3136
    determinants. It is checked against E_FCI to 10⁻⁶.
  - **vDBF:** the final states of the UNO + 0.05 S² runs at cutoffs 10⁻⁵
    (`dbf_uno_s2pen0.05`) and 10⁻⁶ (`dbf_uno_s2pen0.05_c1e-6`).
    - Each state is replayed exactly from its stored rotations in `<stem>_ckpt.jld2`:
      ψ = U₁U₂…U_K|0⟩ with U_k = exp(−iθ_k G_k/2). This is a statevector, with no
      truncation, i.e. the state the circuit would prepare.
    - The newest rotation acts first.
- Working space V:
  - V is the vacuum-frame states with even weight in each spin block, 2¹⁴ states
    (`spin_parity_basis`).
  - V is closed under H, the DBF generators and orbital rotations. H's own x-span
    (2¹²) is *not* closed under the π rotation, which is why the larger space is
    needed.

## Quantities

- **Configuration entropy:** S = −Σ_b |c_b|² ln |c_b|² over determinants b.
- **Magic, the stabilizer Rényi entropy:** M₂ = −ln(Σ_P ⟨P⟩⁴ / 2¹⁶), summed over
  all 4¹⁶ Pauli strings.
  - For each X pattern x in V, the expectation values over all Z patterns are one
    Walsh–Hadamard transform, so the whole sum takes seconds. A multiplier of
    2^(N − dim V) accounts for Z patterns that look the same on V.
  - Checked on every state: Σ_P ⟨P⟩² = 2¹⁶.
- **Bond breaking:**
  - Natural orbitals are computed per symmetry block (σ 4×4, πx 2×2, πy 2×2).
  - The antibonding occupation n goes from 0 (bond intact) to 1 (fully broken).
  - "Half broken" means n = 0.5: π at 1.72 Å, σ at 2.09 Å.
- **Group magic:**
  - The qubits are split into the σ orbitals (8 qubits, both spins) and the πx + πy
    orbitals (8 qubits).
  - The reduced state ρ_A (256×256) gives the mixed-state
    M₂(ρ_A) = −ln(Σ_P tr(Pρ_A)⁴ / Σ_P tr(Pρ_A)²), with P running over Paulis in the
    group.
  - Checked against the purity: Σ_P tr(Pρ_A)² = 2⁸ tr ρ_A².
  - It is *not* additive once the groups are entangled, so compare where each curve
    changes, not the sum.
- **Independent-bond model** (the shaded and dashed curves):
  - Each bond is treated as an isolated pair √(1−w)|bond²⟩ − √w|anti²⟩ with w = n/2.
    That pair is Clifford-equivalent to one qubit with Bloch vector z = 1−n,
    x = √(n(2−n)).
  - Pair magic is −ln((1 + x⁴ + z⁴)/2): 0 when the bond is intact, 0 when it is fully
    broken, at most ln 4/3 at n = 1 − 1/√2.
  - Pair entropy is −w ln w − (1−w) ln(1−w), which rises to ln 2.
  - The model sums σ + 2π, using the measured occupations.

## Orbital-basis fixes

S and M₂ depend on the orbital basis; energies do not.

1. **π orientation.**
   - The UNO π pairs are degenerate and come out arbitrarily mixed, differently at
     every R. For example, at 2.1 Å the π* pair was a 30/73 px/py mix. This made the
     first S/M₂ curves jagged (`uno_resources_raw.png`).
   - Fix: rotate each π pair (occupied and virtual separately) to pure px/py. The
     angle comes from an SVD of the py-AO coefficient block (`pi_generator`).
   - The state is then rewritten as ψ′ = exp(−κ̂)ψ with κ̂ = Σ K_pq (a†_pα a_qα + a†_pβ a_qβ),
     using a Taylor expmv (`to_fixed_pi`).
   - Checks:
     - every state's 1-RDM must become RᵀγR;
     - `check_pi_rotation.jl` rebuilds H from π-rotated integrals with the
       converter's own `uhf_hamiltonian`. At 2.1 Å the rotated FCI state is its
       ground state (overlap 1.000000000000), and the vDBF energies agree to
       10 digits.
2. **Natural orbitals (bond_magic.jl).**
   - The state is rotated into the exact state's NOs, block by block, via
     K₂ = log U, which is checked to be antisymmetric.
   - vDBF states use the *exact* state's NOs too, because they break symmetry
     slightly and their own NOs would mix πx and πy again. `check_vdbf_symmetry.jl`
     shows:
     - σ–π 1-RDM elements up to 10⁻² at 10⁻⁵ and 2×10⁻⁴ at 10⁻⁶;
     - πx/πy occupations differing by up to 0.014;
     - electron count off by up to 4×10⁻⁴.
   - Their occupations are the eigenvalues of their own blocks.

## Results (exact state, natural orbitals)

The Coulson–Fischer point (UHF first below RHF) is at 1.2 Å.

| R (Å) | n σ* | n π* | M₂ | S | M₂ σ group | M₂ π group | group purity |
|---|---|---|---|---|---|---|---|
| 0.8 | 0.006 | 0.031 | 0.13 | 0.24 | 0.03 | 0.11 | 0.98 |
| 1.2 | 0.028 | 0.104 | 0.44 | 0.67 | 0.07 | 0.37 | 0.94 |
| 1.6 | 0.102 | 0.356 | 1.19 | 1.58 | 0.20 | 0.99 | 0.89 |
| 1.7 | 0.148 | 0.474 | 1.38 | 1.92 | 0.28 | 1.09 | 0.87 |
| 2.0 | 0.409 | 0.795 | 1.65 | 2.85 | 0.58 | 1.10 | 0.71 |
| 2.1 | 0.508 | 0.855 | 1.67 | 3.03 | 0.60 | 1.12 | 0.65 |
| 3.0 | 0.928 | 0.991 | 1.58 | 3.40 | 0.48 | 1.21 | 0.49 |

- **Total M₂** peaks at 1.67 nats at 2.1 Å.
  - Its slope has a single maximum, about 2.2 nats/Å at 1.5–1.6 Å, so the total
    alone can't separate π from σ.
- **Independent-bond model:**
  - M₂ peaks at 0.73 near 1.6 Å and is about 0 by 3.0 Å.
  - S levels off at 3 ln 2 = 2.08.
  - Up to about 1.3 Å, magic is the sum of the bonds. Beyond that the excess
    dominates: at 3.0 Å the bonds alone give 0.005 nats, but the state has 1.58.
- **Group magic:**
  - The π group follows its own bond model to about 1.2 Å. It flattens at
    1.09–1.10 nats from 1.7 to 2.0 Å, with a small maximum at 1.8 Å, then creeps
    to 1.21 by 3.0 Å.
  - The σ group follows its model to about 1.6 Å, peaks at 0.60 at 2.10 Å, and
    falls to 0.48.
  - The purity, which is the same for both groups, falls from 0.98 to 0.49, so σ
    and π become entangled.
  - The working hypothesis for the leftover magic is spin recoupling: each N atom
    couples its σ and π electrons into one spin-3/2 state, and the two atoms couple
    to S = 0. This is **not yet tested**.

**In simple words, for Figure 3.**
- Split N₂'s electrons into a σ team and a π team.
- The π team's magic flattens when the π bonds are half broken (1.7 Å).
- The σ team's magic peaks when the σ bond is half broken (2.1 Å).
- The dashed lines are what isolated bonds would give. The real curves stay higher
  because the σ and π electrons become linked as each atom lines up its spins.

## Files (all in this folder)

| File | What it does |
|---|---|
| `resources.jl` | Replays the FCI and vDBF states, fixes π, and computes S and M₂ in the fixed-π UNOs and the raw UNOs → `results/resources_uno.json` |
| `bond_magic.jl` | Block NOs, occupations, and S, M₂ and group magic (σ, π) in the exact state's NOs → `results/bond_magic_uno.json` |
| `plot_resources.py` | `uno_resources.png` (S, M₂ and the PES; add `--raw` for the jagged `uno_resources_raw.png`) |
| `plot_bond_magic.py` | `uno_bond_magic.png` (M₂, S and occupations, with the independent-bond shading) |
| `plot_group_magic.py` | `uno_group_magic.png` (σ and π group magic, with occupations) |
| `check_pi_rotation.jl` | Independent check of the π rotation, using a Hamiltonian rebuilt from rotated integrals |
| `check_vdbf_symmetry.jl` | The vDBF symmetry-breaking diagnostic |
| `results/logs/{resources,bond_magic}_uno.out` | Run logs |

Run from this folder:

```bash
julia -t 4 --project=../../integrals resources.jl
julia -t 4 --project=../../integrals bond_magic.jl
python plot_resources.py && python plot_resources.py --raw
python plot_bond_magic.py
python plot_group_magic.py
```

**Inputs not on this branch.** These are shared with the PES work and are left
untracked, so switching branches does not delete them:
- `data_unocore/R_*/hamiltonian_uno.jld2`;
- the vDBF checkpoints `data_unocore/R_*/dbf_uno_s2pen0.05{,_c1e-6}_ckpt.jld2`;
- `results/fc_cc.csv`, the RHF/UHF energies used by `plot_resources.py`.

`uno_coeff.npy` and `metadata.json` are tracked. `resources.jl` includes `sv_dbf.jl`
for the sector, replay and one-body helpers.

## Open ideas

- Test the spin-recoupling hypothesis:
  - project the state onto atomic spin states;
  - or compare with a model where two spin-3/2 atoms couple to S = 0, which should
    carry the leftover magic at 3.0 Å.
- Per-bond magic beyond groups: πx alone vs πy alone; σ split further.
- Larger basis (cc-pVDZ) or a finer R grid near 1.6–2.2 Å.
- Magic along the vDBF flow (state after each rotation) instead of only the final state.
- Does the 10⁻⁵ overshoot in M₂ near 2.0–2.1 Å come from the same rotations that
  cause its energy error?
