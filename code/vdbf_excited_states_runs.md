# What the first excited-state runs do (idea 1c, exact DBF, N2)

Companion to `vdbf_excited_states.md` (the plan). Written 14 Nov 2026 to explain the `sv_block.jl` runs. Energies in mHa (1 mHa = 0.0272 eV; chemical accuracy = 1.6 mHa).

## In one paragraph

Ground-state vDBF rotates H until the reference $|0\rangle$ decouples from everything else. Here we decouple a block of three determinants instead: the reference, and the HOMO→LUMO single excitation in α and in β. Diagonalising the resulting 3×3 block gives a ground state and two excited states. We do this first **without Pauli propagation and without truncation**, using the exact state-vector version of DBF that already exists for frozen-core N2 (`sv_dbf.jl`), so any error is the flow's own. **At 1.1 Å it works:** after 200 iterations the block gives S0, a triplet (³Πg) and a singlet (¹Πg), all within 0.04 mHa of exact, with excitation energies within 0.02 mHa. **At 2.3 Å it does not converge without help:** the ground-state column mixes with a quintet 9 mHa away, the same spin contamination the ground-state flow shows there. **With a spin penalty $H + 0.05\,S^2$ it converges** (S0 +0.010, T1 +0.030 mHa after 500 iterations). A larger penalty, λ = 0.1, loses T1: both excited columns end up as singlets.

## 1. System and why the exact code

- N2, STO-3G, 1s cores frozen: 8 spatial orbitals, 16 qubits, 5α + 5β electrons. UNO basis (natural orbitals of UHF; closed-shell reference). Same Hamiltonians as the qual-exam PES.
- Two geometries: **1.1 Å** (near equilibrium, weakly correlated) and **2.3 Å** (stretched, strongly correlated).
- **Exact code.** `sv_dbf.jl` stores the product of all rotations so far, $W = U_1 U_2 \cdots U_K$, as a dense orthogonal matrix. Column $y$ of $W$ is the current, rotated version of determinant $|y\rangle$. The rotated Hamiltonian is $O = W^\top H W$; we never build it, only the few elements we need ($\langle x|O|y\rangle = w_x^\top H w_y$). Nothing is truncated, so this isolates the *method* (block cost, ranking, greedy rotations) from Pauli-propagation error. Once it works here, the same five changes go into `dbf_groundstate` with truncation (plan, Stage 4).
- **Orbitals.** HOMO = active orbital 4, LUMO = active orbital 5 at both geometries (checked against the Fock diagonal). At 1.1 Å that is σg → πg. At 2.3 Å it is πu → πg. The π orbitals come in degenerate pairs (x, y).

## 2. What the ground-state flow does (sv_dbf.jl, for reference)

Each macro-iteration:

1. For every determinant $|x\rangle$, compute the coupling $\langle x|O|0\rangle$ to the reference.
2. Rank the directions $x$ by that coupling and take the top 50.
3. For each, apply one Pauli rotation $e^{-i\theta G/2}$ that mixes $|0\rangle$ with $|x\rangle$. $G$ has X/Y on the bits where $x$ is 1. $\theta$ is chosen exactly to minimise $\langle 0|O|0\rangle$; the energy as a function of $\theta$ is $a + b\cos\theta + c\sin\theta$.

Converged when every $\langle x|O|0\rangle \approx 0$, i.e. $|0\rangle$ is decoupled and $\langle 0|O|0\rangle = E_0$.

## 3. What the block flow does (sv_block.jl)

**The block.** $S = \{|0\rangle, |a\rangle, |b\rangle\}$, with $|a\rangle$ = HOMO→LUMO in α and $|b\rangle$ = HOMO→LUMO in β. Their combinations $|a\rangle \mp |b\rangle$ are the $M_s=0$ triplet and the singlet of that excitation. This is variant 1c of the plan.

**The cost.** Instead of $\langle 0|O|0\rangle$ we minimise the block's diagonal sum

$$C_S = \langle 0|O|0\rangle + \langle a|O|a\rangle + \langle b|O|b\rangle .$$

By the Ky Fan theorem its minimum over all rotations is the sum of the 3 lowest eigenvalues. When the block is decoupled ($\langle x|O|y\rangle = 0$ for $y$ in the block and $x$ outside), the 3×3 matrix $M_{yy'} = \langle y|O|y'\rangle$ has exact eigenvalues.

**The angle.** Same structure as the ground state: for a rotation in direction $x$, $C_S(\theta) = \text{const} + B\cos\theta + C\sin\theta$, summed over the three block states. So the optimal angle is closed-form, exactly as before. Rotations inside the block (e.g. $|0\rangle \leftrightarrow |a\rangle$) cancel out of $C_S$, as the plan said.

**One new detail: the sign of the generator.** With one block state, a Pauli generator's sign on $|0\rangle$ doesn't matter. With three, the gradient is a sum of three terms, $\sum_y \mu(y)\,\langle y\oplus x|O|y\rangle$, where $\mu(y)=\pm1$ is the sign the Pauli picks up on $|y\rangle$. That sign depends on where the Pauli has Z's. So for each direction $x$ we pick the Z-pattern (a small linear solve over bits) that makes all three terms add instead of cancel. It is still one ordinary Pauli rotation, with an odd number of Y's so it stays real.

**Ranking, two options.**

- `exact`: rank by the exact gradient of $C_S$ (the block version of ranking by $|\langle x|O|0\rangle|^2$).
- `d1c`: rank by the plan's operator $D_{1c} = \sum_{\text{rest}} Z_k + Z_H Z_L + Z_{\bar H} Z_{\bar L} + c\,(Z_H + Z_{\bar H} - Z_H Z_{\bar H})$, i.e. by $\langle y\oplus x|[D,O]|y\rangle$ restricted to the block. This is the proxy the Pauli-propagation code would use (plan §6.4). The code checks that the block is exactly the top eigenspace of $D_{1c}$, as the plan claims.

## 4. Symmetry: why the sector had to change, and what the targets are

`sv_dbf.jl` only works with determinants reachable from $|0\rangle$ by H's own excitations (the span of H's Pauli x-strings). That set only contains determinants with the **same spatial symmetry as the reference** (4096 of them). The HOMO→LUMO excitation has a different symmetry (σg→πg is Πg; πu→πg is ungerade), so $|a\rangle, |b\rangle$ are not in it. `sv_block.jl` adds them, which doubles the space to 8192: two symmetry sectors.

H does not couple the two sectors (checked: those block elements stay exactly 0 throughout). So the problem splits:

- Sector of $|0\rangle$: one block state → the flow targets **the lowest state of that symmetry** (S0).
- Sector of $|a\rangle$, $|b\rangle$: two block states → the flow targets **the two lowest states of that symmetry**.

Exact low-lying levels (from diagonalising each sector), mHa above the ground state; ★ = Ky Fan target:

| | 1.1 Å, sector of \|0⟩ | 1.1 Å, sector of \|a⟩ | 2.3 Å, sector of \|0⟩ | 2.3 Å, sector of \|a⟩ |
|---|---|---|---|---|
| 1 | ★ 0 (S0) | ★ 297.17 triplet | ★ 0 (S0) | ★ 2.84 triplet |
| 2 | 622.6 quintet | ★ 297.17 triplet (degenerate partner) | 9.00 quintet | ★ 20.14 septet |
| 3 | | 347.38 singlet ×2 | | 120.6 triplet ×2 |
| | | | | singlet: not in the lowest 8 |

What this means physically:

- **1.1 Å.** In the HOMO→LUMO sector the lowest states are the two degenerate components of a triplet, ³Πg (x and y), with the singlet ¹Πg 50 mHa above. The π orbitals are not symmetry-adapted, so x and y share a sector. So the Ky Fan minimum is (S0, ³Πg x, ³Πg y), **not** the (S0, T1, S1) the plan assumed.
- **2.3 Å.** The stretched molecule has the spin ladder of two N atoms: singlet, triplet (2.8 mHa), quintet (9.0), septet (20.1). Singlet and quintet share the reference's symmetry; triplet and septet are in the HOMO→LUMO sector. So the Ky Fan minimum there is (S0, triplet, septet).
- The overall lowest triplet of N2 can sit in yet another symmetry; this block cannot see it. A block only finds the lowest states of the symmetries its determinants carry.
- Labels like ³Πg are assigned from degeneracy, ⟨S²⟩ and orbital character; the code does not compute irreps.

## 5. Results without the penalty (200 iterations × 50 rotations, about 2 min each)

Block eigenvalues compared with the exact level of the same symmetry and spin, mHa above the exact ground state. "Start" = the bare 3×3 block before any rotation.

**1.1 Å**

| state | start | after flow (exact ranking) | exact | error |
|---|---|---|---|---|
| S0 | 157.3 | 0.019 | 0 | +0.019 |
| ³Πg | 435.8 | 297.196 | 297.174 | +0.022 |
| ¹Πg | 512.2 | 347.421 | 347.384 | +0.037 |

Excitation energies: ³Πg 297.177 vs 297.174 (8.087 eV), ¹Πg 347.402 vs 347.384 (9.453 eV). Errors partly cancel in the differences. The d1c ranking gives nearly identical numbers (+0.021, +0.028, +0.038). Still improving at 200 iterations.

**Why it lands on (S0, ³Πg, ¹Πg), not the Ky Fan minimum.** To swap the singlet for the second triplet component, the flow would need a coupling between a singlet column and a triplet state. H conserves spin exactly ($[H, S^2]$ = 5e-16 here), so for a spin-pure column that coupling is zero and the gradient pointing there is zero. The block is decoupled (any set of eigenstates is a stationary point), just not the lowest one. The plan calls this a saddle (§6.2). Here it is welcome: it is exactly the plan's intended (S0, T1, S1). In the log this shows as `C-target` stuck at +50.2 mHa (the ¹Πg–³Πg gap) and `1-F_S` at exactly 1/3 (one of three block directions is outside the Ky Fan target space).

**2.3 Å**

| state | start | after flow (exact ranking) | comment |
|---|---|---|---|
| S0 | 744.7 | 1.56 | ⟨S²⟩ = 0.76: mixed with the quintet 9 mHa above |
| triplet | 543.5 | 3.68 (exact 2.84) | +0.84 |
| 3rd | 607.1 | ~120.5 | wanders between a triplet at 120.6 and the septet |

Not converged, and the d1c ranking is worse (S0 at +3.3 with ⟨S²⟩ 1.4, third state drifting toward the septet). Cause: a single Pauli rotation is not spin-adapted. When a state of different spin sits close by (the quintet, 9 mHa above S0, in the same symmetry), the greedy rotations mix them and the flow is slow to undo it. The ground-state-only flow from the same start has the same problem (`uno_2.30_onebody`, ⟨S²⟩ 0.32 after 200 iterations). There the fix was flowing on $H + 0.05\,S^2$.

## 6. The spin penalty

`--s2-penalty λ` makes the flow minimise the block cost of $H + \lambda S^2$; reported energies are still of $H$. Because $[H, S^2] = 0$, the eigenstates are unchanged; each spin-$S$ level just moves up by $\lambda S(S+1)$ in the cost (triplet +2λ, quintet +6λ, septet +12λ). Two effects:

1. The quintet is pushed far from S0, so the |0⟩ column should stop mixing.
2. The Ky Fan target in the HOMO→LUMO sector changes. At 2.3 Å the HOMO→LUMO singlet is at 227.5 mHa:
   - **λ = 0.05:** target = triplet 2.84 + one of the triplet pair at 120.6 (degenerate at the block edge).
   - **λ = 0.1:** the 120.6 triplets move above the singlet (they rise by 2λ, the singlet doesn't), so the target becomes exactly **(S0, T1 at 2.84, S1 at 227.5)**, with a 29 mHa gap to the next level. Any λ above ≈ 0.054 does this.

Both were run for 500 iterations with exact ranking (`uno_2.30_block1c_exact_s2pen0.05`, `uno_2.30_block1c_exact_s2pen0.1`).

**λ = 0.05: converged.**

| state | after flow | exact | error | ⟨S²⟩ |
|---|---|---|---|---|
| S0 | 0.0095 | 0 | +0.0095 | 0.000 |
| T1 (³Σu⁺) | 2.873 | 2.842 | +0.030 | 2.000 |
| triplet | 120.974 | 120.614 | +0.36 | 1.995 |

The spin contamination is gone, and the T1 excitation energy is 2.863 vs 2.842 mHa. The block reaches its Ky Fan target (C − target 0.13 mHa and falling, $V_S^H$ = 1.2e-4).

**λ = 0.1: the triplet is lost.**

| state | after flow | exact | error | ⟨S²⟩ |
|---|---|---|---|---|
| S0 | 0.0445 | 0 | +0.045 | 0.000 |
| S1 (HOMO→LUMO singlet) | 228.013 | 227.512 | +0.50 | 0.000 |
| second singlet | 263.991 | 263.281 | +0.71 | 0.002 |

The target was exactly (S0, T1, S1), but the flow ended on (S0, S1, a second singlet), with C − target stuck at 62 mHa. With a large penalty, triplet character costs 2λ = 200 mHa, so early in the flow the greedy rotations push both HOMO→LUMO columns into singlet space. Once a column is a pure singlet, spin conservation means nothing pulls it back toward T1. It is the same protection that kept ¹Πg in the block at 1.1 Å, now working against us. The penalty decides each column's spin early; getting (S0, T1, S1) reliably at 2.3 Å needs the spin of each column steered, not just penalised. λ = 0.05 is the setting that works.

## 6b. Watching the chosen states converge

Figures from `plot_block.py` (written to `frozen_core/sv_results/plots/`, not in git):

- `energy_vs_variance.png`: each block eigenvalue minus the exact level it ends on, against the block variance $V_S$ of H (log; converging to the right), one panel per run. At 1.1 Å all three states lie on straight, parallel lines: each error is roughly proportional to $V_S$. At 2.3 Å with bare H the lines fold back and stall. With +0.05 S² they converge after a rearrangement near $V_S \approx 10^{-2}$. With +0.1 S² the second state starts as the triplet, ~100 mHa below the singlet it ends on, and then jumps to the singlet; that jump is where T1 is lost.
- `tail_extrapolation.png`: the last 40% of iterations on linear axes, with a straight-line fit extrapolated to $V_S = 0$.
- `block_variance.png`: $V_S$ against iteration for all runs.
- `levels.png`: the exact levels per symmetry sector with the Ky Fan targets.

**Does it extrapolate linearly in $V_S$?** (Open question 1 of the plan.) Error at the last iteration → straight-line extrapolation to $V_S = 0$, mHa:

| run | state | last | extrapolated |
|---|---|---|---|
| 1.1 Å | S0 / ³Πg / ¹Πg | +0.019 / +0.023 / +0.037 | −0.002 / −0.0002 / −0.001 |
| 2.3 Å, +0.05 S² | S0 / T1 / triplet 120.6 | +0.010 / +0.030 / +0.36 | −0.005 / −0.022 / +0.25 |
| 2.3 Å, +0.1 S² | S0 / S1 / second singlet | +0.045 / +0.50 / +0.71 | −0.011 / +0.25 / +0.51 |

At 1.1 Å, yes: the tails are straight and extrapolation removes almost all of the error. At 2.3 Å it helps S0 and T1, but the higher states' tails still curve. $V_S$ is one number for the whole block; a per-state variance would be a cleaner x-axis and is easy to add.

## 7. How to read a log line

```
 iter    C-target   lambda_i - target_i (mHa)   V_S   V_S^H   max|grad|   1-F_S   <S^2> of M's states   leak   drift   time
```

- **C-target**: block cost minus its Ky Fan minimum (of $H + \lambda S^2$). Goes to 0 only at the true minimum; plateaus on a saddle.
- **λᵢ − targetᵢ**: eigenvalues of the 3×3 block of $H$ minus the target levels, per symmetry sector (S0 first). On a saddle, compare instead with the matching exact level, which the end-of-run printout does.
- **V_S**: block variance, $\sum_{y\in S} \|(1-\Pi_S)\,O\,|y\rangle\|^2$: how strongly the block still couples to the outside. 0 = decoupled. `V_S` is for the flow operator, `V_S^H` for H (equal when λ = 0).
- **max|grad|**: largest block gradient over all directions; convergence when below 1e-6.
- **1−F_S**: how much of the block lies outside the target eigenvectors (0 = block spans exactly the targets; 1/3 = one block direction is a different eigenstate, as on the 1.1 Å saddle).
- **⟨S²⟩**: of each eigenvector of the 3×3 block (0 singlet, 2 triplet, 6 quintet, 12 septet).
- **leak**: weight outside 10 electrons. The Hamiltonian has no chemical potential and the sector contains other electron numbers, so this should go to 0.
- **drift**: check that the running updates match a from-scratch recomputation (~1e-13).

## 8. Caveats

- Exact arithmetic only so far. Nothing here says how truncation will behave.
- 200 iterations was not enough to converge fully even at 1.1 Å ($V_S$ ≈ 2.5e-4 and falling).
- The target depends on the symmetry of the block determinants and on the orbital basis (here the degenerate π pairs are not symmetry-adapted).
- The d1c ranking is the plan's $[H, D]$ proxy evaluated only on matrix elements touching the block, as `sv_dbf.jl`'s "onebody" ranking is for the ground state; the Pauli-propagation version sees the whole operator.

## 9. Reproduce

From `code/n2_pes/frozen_core/` (input data in `data_unocore/R_*`, not in git; copy from the qual_exam checkout):

```
julia -t 8 --project=../../integrals sv_block.jl data_unocore/R_1.10 uno uno_1.10_block1c_exact
julia -t 8 --project=../../integrals sv_block.jl data_unocore/R_2.30 uno uno_2.30_block1c_exact
julia -t 8 --project=../../integrals sv_block.jl data_unocore/R_2.30 uno uno_2.30_block1c_d1c --rank d1c
julia -t 8 --project=../../integrals sv_block.jl data_unocore/R_2.30 uno uno_2.30_block1c_exact_s2pen0.05 --s2-penalty 0.05 --max-iter 500
julia -t 8 --project=../../integrals sv_block.jl data_unocore/R_2.30 uno uno_2.30_block1c_exact_s2pen0.1 --s2-penalty 0.1 --max-iter 500
python plot_block.py
```

Results go to `sv_results/<label>.json`; the logs above were saved to `sv_results/logs/<label>.out`.
