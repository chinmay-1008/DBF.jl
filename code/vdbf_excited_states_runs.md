# Excited states of N₂ with block DBF. Part 1: R = 1.1 Å

Energies in mHa (1 mHa = 0.0272 eV). Exact numbers come from diagonalising the same Hamiltonian. Part 2 (stretched N₂, 2.3 Å) comes after this one.

## 1. What we want

vDBF finds the ground state by rotating H until the reference determinant $|0\rangle$ is decoupled from everything else; then $\langle 0|\bar H|0\rangle = E_0$.

We want a few **excited states** from the same machinery. The idea: instead of decoupling one determinant, decouple a small **block** of determinants, then diagonalise that block. Here the block has 3 determinants, so we get 3 states: the ground state and two excited states.

For this first test we use the **exact** version of DBF (state vector, no Pauli truncation), so whatever error we see belongs to the method itself.

## 2. The molecule: N₂ at 1.1 Å

**Model.** STO-3G basis, the two 1s cores frozen. That leaves 8 spatial orbitals (16 spin orbitals = 16 qubits: α on qubits 0–7, β on 8–15) and 10 electrons.

**Orbitals** (UNO basis; at 1.1 Å these are the ordinary RHF orbitals):

| qubit (α / β) | orbital | Fock energy (Ha) | in the reference |
|---|---|---|---|
| 0 / 8 | 2σg | −1.440 | occupied |
| 1 / 9 | 2σu | −0.723 | occupied |
| 2, 3 / 10, 11 | 1πu (x, y) | −0.571 | occupied |
| **4 / 12** | **3σg = HOMO** | −0.539 | occupied |
| **5 / 13** | **1πg = LUMO** | +0.280 | empty |
| 6 / 14 | 3σu | +1.118 | empty |
| 7 / 15 | 1πg (other component) | +0.280 | empty |

**Exact ground state.** E₀ = −107.6538 Ha. Written as a sum of determinants, $|\Psi_0\rangle = c_0|0\rangle + \sum_k c_k|k\rangle$, the exact wavefunction has **92% of its weight on the closed-shell determinant** (2σg)²(2σu)²(1πu)⁴(3σg)², which we call $|0\rangle$: $|c_0|^2 = 0.92$. The other 8% is correlation, mostly 1πu² → 1πg² double excitations at about 2% each. Because $|0\rangle$ is only part of the true state, its energy alone is 157.3 mHa above E₀; that is the correlation energy the flow has to recover.

**Excited states we target.** The lowest excitation in this picture is HOMO → LUMO, 3σg → 1πg. Promoting one electron gives two determinants:

- $|a\rangle$: 3σg → 1πg in α (qubits 4 and 5 flipped)
- $|b\rangle$: 3σg → 1πg in β (qubits 12 and 13 flipped)

Their two combinations are the triplet and the singlet of this excitation:

| exact state | exact energy above E₀ | weight of the exact wavefunction on the block |
|---|---|---|
| S0 | 0 | 0.92 on $\lvert 0\rangle$ |
| ³Πg | 297.17 mHa (8.09 eV) | 0.92 on $(\lvert a\rangle - \lvert b\rangle)/\sqrt2$ |
| ¹Πg | 347.38 mHa (9.45 eV) | 0.89 on $(\lvert a\rangle + \lvert b\rangle)/\sqrt2$ |

These weights describe the exact states themselves, not how well DBF does. DBF still converges to the exact energies (section 6); the weights say how much of each exact state the bare determinants already contain, and so how much the flow has to build in.

(Π because 1πg is doubly degenerate, so each of these has a second, degenerate component made from the other πg orbital.)

So the three states we want are each mostly one combination of the three determinants $\{|0\rangle, |a\rangle, |b\rangle\}$. The remaining ~10% is correlation, spread over many other determinants. **The job of the flow is to rotate that correlation into the three determinants**, so that the 3×3 block alone gives the exact energies.

(If we also kept the double excitation 3σg² → 1πg², the block would have 4 states; that is variant 1b of the plan. Here we use the 3-state variant 1c.)

## 3. What ordinary DBF does

**Frame.** We work in the frame where the reference is $|00\ldots0\rangle$. A 1 on qubit $i$ means orbital $i$'s occupation is flipped: a hole if the orbital is occupied in the reference, a particle if it is empty.

**The diagonal operator.** Ordinary DBF uses

$$D = \sum_{i=1}^{16} Z_i .$$

$Z_i = +1$ if qubit $i$ is not flipped and $-1$ if it is. On a determinant with $w$ flipped qubits, $D = 16 - 2w$. So $D$ is largest (= 16) only on $|0\rangle$, and every other determinant sits lower.

**The flow.** A double-bracket flow with a diagonal $D$ rotates H until it commutes with $D$, and it puts the lowest energies on the largest values of $D$. Since only $|0\rangle$ has the largest value, $|0\rangle$ ends up decoupled, holding $E_0$. Section 5 writes the flow out; section 6 describes vDBF, the greedy version of it.

## 4. What we change: the HOMO and LUMO terms of D

We keep $Z_k$ on the 12 other qubits and replace the four HOMO/LUMO terms $Z_H + Z_L + Z_{\bar H} + Z_{\bar L}$ (qubits 4, 5, 12, 13; bar = β) with

$$D_\text{active} = Z_H Z_L + Z_{\bar H} Z_{\bar L} + c\,(Z_H + Z_{\bar H} - Z_H Z_{\bar H}), \qquad c = 1 .$$

What each piece does:

- **$Z_H Z_L$** is +1 when HOMO-α and LUMO-α are both unflipped or both flipped, i.e. the α electron of that pair is either still in the HOMO or has moved to the LUMO. $Z_{\bar H} Z_{\bar L}$ does the same for β. These two alone favour 4 configurations: reference, $|a\rangle$, $|b\rangle$ and the double.
- **The $c$ term** is +1 on the reference, $|a\rangle$ and $|b\rangle$, and −3 on the double, so it pushes the double down.

Values on the four HOMO/LUMO qubits (order: HOMO-α, HOMO-β, LUMO-α, LUMO-β):

| configuration | meaning | $D_\text{active}$ |
|---|---|---|
| 0000 | reference $\lvert 0\rangle$ | 2 + c |
| 1010 | $\lvert a\rangle$, HOMO→LUMO in α | 2 + c |
| 0101 | $\lvert b\rangle$, HOMO→LUMO in β | 2 + c |
| 1111 | double, HOMO² → LUMO² | 2 − 3c |
| any other (e.g. 1000: electron removed from HOMO-α only) | breaks a pair | c or lower |

Flipping any of the other 12 qubits lowers $D$ by 2 as before. So **the largest value of $D$ now belongs to exactly three determinants, $\{|0\rangle, |a\rangle, |b\rangle\}$**, and the flow decouples this block as a whole instead of $|0\rangle$ alone.

What this means for any flow with this $D$:

- **What goes down** is the sum of the block's three diagonal elements, $C = \langle 0|\bar H|0\rangle + \langle a|\bar H|a\rangle + \langle b|\bar H|b\rangle$. Its minimum is the sum of the three lowest eigenvalues the block can reach (Ky Fan).
- **At the end** we diagonalise the 3×3 matrix $\langle y|\bar H|y'\rangle$, $y, y' \in \{0, a, b\}$, to get the three energies.
- **How decoupled the block is** is measured by the **block variance** $V_S = \sum_{y} \lVert (1 - \Pi_S)\,\bar H\,|y\rangle\rVert^2$, the weight with which the block still couples to the outside ($V_S = 0$: fully decoupled). All plots below use it as the x-axis, the way ground-state plots use the energy variance.

## 5. Exact evolution: the full double-bracket flow

**Formalism.** The flow is a continuous rotation of H, with flow time $s$:

$$\frac{d\bar H}{ds} = [\,\eta, \bar H\,], \qquad \eta = [\,\bar H, D\,], \qquad \bar H(0) = H .$$

Every instant is an exact unitary rotation, $\bar H(s) = W(s)^\dagger H\, W(s)$, so the spectrum never changes. $\mathrm{Tr}(\bar H D)$ decreases monotonically, and the flow stops when $[\bar H, D] = 0$, with the lowest energies on the top eigenspace of $D$, i.e. on the block. Here it is computed exactly, with no truncation.

**Which $D$.** Section 4's operator, $D_{1c}$. The flow is integrated numerically (exponential midpoint, step size adapted to keep every matrix element accurate to $10^{-9}$ Ha), separately in the two symmetry sectors since H couples neither; about 3,000 steps and 40 minutes per sector.

Only the top eigenspace of $D$ decides which states end up in the block. The simplest $D$ with the same top eigenspace is the **block projector** $\Pi_S = |0\rangle\langle 0| + |a\rangle\langle a| + |b\rangle\langle b|$; with it the flow is exactly the gradient flow of the block sum $C$ and has a closed form (the block is spanned by $e^{-sH}|0\rangle$, $e^{-sH}|a\rangle$, $e^{-sH}|b\rangle$, subspace imaginary-time evolution). It traces almost the same energy-against-$V_S$ curve as $D_{1c}$ (section 7), so only $D_{1c}$ is shown.

**Result.** Same start as everything else (the bare 3×3 block, $V_S = 0.93$):

| state | start | after the flow | exact | error |
|---|---|---|---|---|
| S0 | 157.3 | 0.0000 | 0 | < 10⁻⁶ |
| ³Πg | 435.8 | 297.1739 | 297.1739 | < 10⁻⁶ |
| ¹Πg | 512.2 | 347.3837 | 347.3837 | < 10⁻⁶ |

All three are within 1.6 mHa by $s = 0.28$ ($V_S = 2.2\times10^{-3}$), within 0.01 mHa by $s = 1.3$ ($V_S = 5\times10^{-6}$), and exact to $10^{-6}$ mHa by $s = 3.4$; the run stops at $V_S = 10^{-13}$. They stay spin-pure throughout (⟨S²⟩ = 0, 2, 0).

![Full flow: block eigenvalues against block variance at 1.1 Å](n2_pes/frozen_core/sv_results/plots/n2_110_flow_d1c_energies.png)

The flow starts on the right (large $V_S$, hollow dots) and moves left as the block decouples; the three energies settle on the exact energies (dashed).

![Full flow: zoom on each state at 1.1 Å](n2_pes/frozen_core/sv_results/plots/n2_110_flow_d1c_zoom.png)

Zoomed onto each exact line, with the same windows as the greedy zoom in section 6. The flow lands exactly on all three.

## 6. Greedy vDBF, exact

**Formalism.** vDBF does the flow greedily, with single Pauli rotations instead of the continuous $\eta$. Here too everything is exact (state vector, no truncation):

1. **Start.** Nothing is rotated. The 3×3 block of the bare H gives 157.3, 435.8 and 512.2 mHa above E₀: the reference energy and the bare HOMO→LUMO triplet and singlet.
2. **Each macro-iteration.** For every determinant outside the block, compute how strongly it couples to the block. Rank them, take the top 50, and apply one Pauli rotation for each, with the angle chosen exactly to lower the block sum $C$.
3. **Track.** After each macro-iteration, diagonalise the 3×3 block and record its three eigenvalues and $V_S$.
4. **Stop** after 200 macro-iterations (10 000 rotations, about 2 minutes).

The ranking can use $D$ (the $[H, D]$ proxy, as in vDBF) or the exact gradient of $C$; here both give the same numbers.

**Result.**

| state | start | after 200 iterations | exact | error |
|---|---|---|---|---|
| S0 | 157.3 | 0.019 | 0 | +0.019 |
| ³Πg | 435.8 | 297.196 | 297.174 | +0.022 |
| ¹Πg | 512.2 | 347.421 | 347.384 | +0.037 |

Excitation energies: ³Πg 297.177 vs 297.174 mHa, ¹Πg 347.402 vs 347.384 mHa. All three stay spin-pure (⟨S²⟩ = 0, 2, 0). With $D$-ranking: +0.021, +0.028, +0.038.

![Greedy vDBF: block eigenvalues against block variance at 1.1 Å](n2_pes/frozen_core/sv_results/plots/n2_110_energies.png)

![Greedy vDBF: zoom on each state at 1.1 Å](n2_pes/frozen_core/sv_results/plots/n2_110_zoom.png)

All three are still approaching their exact lines at the last iteration, so the next step is to extrapolate, as in ground-state vDBF.

### Extrapolating in the variance

In ground-state vDBF the energy is extrapolated against its variance, $\langle 0|\bar H^2|0\rangle - \langle 0|\bar H|0\rangle^2$, which measures how strongly $|0\rangle$ still couples to everything else. The block has the same quantity for each state. Each eigenvector $\phi_i$ of the 3×3 block (energy $\lambda_i$) is an ordinary state with energy variance

$$\mathrm{var}_i = \langle \phi_i|\bar H^2|\phi_i\rangle - \lambda_i^2 = \lVert (1 - \Pi_S)\,\bar H\,\phi_i \rVert^2 .$$

Inside the block $\bar H\phi_i = \lambda_i \phi_i$ exactly, so only the coupling to the outside contributes. The three variances add up to the block variance $V_S$; for a one-state block this is the usual ground-state variance.

Near convergence the energy error of each state is proportional to its own variance (if $\phi = \sqrt{1-\epsilon^2}\,|\Psi\rangle + \epsilon\,|\chi\rangle$, both are of order $\epsilon^2$). So each state is extrapolated on its own: energy against its own variance, fitted with a straight line over the window $[v, 2v]$, where $v$ is that state's variance at the last iteration, and read off at zero variance. The uncertainty is half the difference from a quadratic fit over the same window.

![Per-state extrapolation, exact greedy, 1.1 Å](n2_pes/frozen_core/sv_results/plots/n2_110_extrap_exact.png)

| state | error at the last iteration | extrapolated error |
|---|---|---|
| S0 | +0.019 mHa | +0.0009 ± 0.0014 mHa |
| ³Πg | +0.022 mHa | −0.0015 ± 0.0010 mHa |
| ¹Πg | +0.037 mHa | +0.0000 ± 0.0038 mHa |

The tails are straight lines (about 40 points in each window), and extrapolation cuts the errors by more than 10×, to about 0.001 mHa. Excitation energies from the extrapolated values: ³Πg 297.172 vs 297.174 mHa, ¹Πg 347.383 vs 347.384 mHa.

One difference from the ground state: an excited state can also mix with states below it, so it is not guaranteed to approach its exact energy from above (only the block sum is variational). Here all three approach from above.

## 7. Both on the same plots

![Greedy vDBF and the full flow on the same axes, 1.1 Å](n2_pes/frozen_core/sv_results/plots/n2_110_compare_energies.png)

![Greedy vDBF and the full flow, zoom on each state](n2_pes/frozen_core/sv_results/plots/n2_110_compare_zoom.png)

- **Same states.** Both end on S0, ³Πg and ¹Πg.
- **At the same $V_S$, greedy is lower.** Where greedy stopped ($V_S = 2.5\times10^{-4}$) its errors are 0.019, 0.022, 0.037 mHa; the full flow at that $V_S$ has 0.035, 0.072, 0.29 mHa (0.042, 0.072, 0.31 with the block projector instead of $D_{1c}$). Likewise greedy has all three within 1.6 mHa by $V_S = 9.6\times10^{-3}$ (iteration 37), the full flow only by $V_S = 2.2\times10^{-3}$. So the greedy single-Pauli path loses nothing against the true flow; per unit of decoupling it does better.
- **The full flow goes all the way.** Greedy stopped at 200 iterations; the full flow continues to $V_S \approx 10^{-20}$ and the exact energies. In both, the error falls roughly in proportion to $V_S$.

## 8. Greedy vDBF with Pauli propagation (cutoff 10⁻⁵)

**Formalism.** The same greedy flow as section 6, now run the way vDBF runs on large systems: $\bar H$ is stored as a sum of Pauli strings, and after every rotation each string with coefficient $|c| \le 10^{-5}$ is dropped. Relative to ground-state vDBF only the block parts change: candidate rotations come from $[\bar H, D_{1c}]$, they are scored and their angles chosen on the block sum $C$, and at the end the 3×3 block is diagonalised.

Truncation means the block read off the truncated $\bar H$ is no longer exactly the block of the rotations that were applied. Three numbers are therefore shown for each state:

- **raw**: the eigenvalues of the 3×3 block of the truncated $\bar H$, what the run sees;
- **corrected**: raw minus the accumulated change that the truncations made to the 3×3 block, for both the energies and the variances. This is the block version of ground-state vDBF's "E − accumulated error" and "Var − accumulated variance error";
- **true**: the generated rotations applied exactly to the three determinants. That is the actual energy of the circuit, possible here because the system is small.

200 iterations × 50 rotations took 27 minutes; $\bar H$ grew to 1.1 million Pauli strings.

**Result** (errors in mHa at the last iteration):

| state | raw | corrected | true |
|---|---|---|---|
| S0 | +1.08 | −0.18 | +0.14 |
| ³Πg | +0.85 | +0.03 | +0.15 |
| ¹Πg | +1.04 | −0.46 | +0.20 |

![Pauli propagation 10⁻⁵: block eigenvalues against block variance at 1.1 Å](n2_pes/frozen_core/sv_results/plots/n2_110_pp_c1e-5_energies.png)

Same layout as the earlier energy figures, with the run's own $V_S$ on the x-axis: corrected (solid), raw (thin), true (dots).

![Pauli propagation 10⁻⁵: zoom on each state, against the exact greedy run](n2_pes/frozen_core/sv_results/plots/n2_110_pp_c1e-5_zoom.png)

- **The circuit is as good as the exact one.** The true energies (dots) lie on the exact greedy curve (grey): at this cutoff truncation does not spoil the rotations.
- **What truncation spoils is the estimate.** Raw sits about 1 mHa too high. The corrected values overshoot below exact. The correction counts each dropped term's effect at the moment it is dropped, not after the later rotations, and over 10,000 truncations that difference adds up. The corrected ¹Πg variance even ends slightly negative.

**Extrapolation**, per state over $[v, 2v]$ as in section 6:

![Pauli propagation 10⁻⁵: per-state extrapolation, true / corrected / raw](n2_pes/frozen_core/sv_results/plots/n2_110_pp_c1e-5_extrap.png)

| state | true | corrected | raw |
|---|---|---|---|
| S0 | −0.013 ± 0.009 | −0.187 ± 0.001 | +0.860 ± 0.073 |
| ³Πg | −0.012 ± 0.011 | −0.083 ± 0.026 | +0.863 ± 0.116 |
| ¹Πg | −0.038 ± 0.095 | −0.443 ± 0.001 * | +0.760 ± 0.036 |
| ³Πg excitation | +0.001 | +0.104 | +0.003 |
| ¹Πg excitation | −0.025 | −0.255 | −0.100 |

- The true energies extrapolate to within 0.04 mHa.
- The raw energy is flat inside its window (it has stopped improving while the variance still falls), so all three land about 0.8 mHa high. The offset is nearly the same for all three, so the raw *excitation energies* are good: +0.003 and −0.10 mHa.
- The corrected values are not reliable at this cutoff.

\* The corrected ¹Πg variance turns negative from iteration 188 on, so $[v, 2v]$ does not exist; that series is fitted over $[0, 0.2\times10^{-3}]$ Ha² instead (46 points, iterations 142–187).

The same run at cutoff 10⁻⁶ is in progress.

## 9. Two things worth knowing

- **Symmetry.** $|a\rangle$ and $|b\rangle$ have a different spatial symmetry (Πg) from the ground state (Σg⁺), so H never couples them to $|0\rangle$. In practice both flows decouple $|0\rangle$ on its own and $\{|a\rangle, |b\rangle\}$ on their own.
- **Which excited states come out.** Because 1πg is degenerate, ³Πg has two components at 297.17. Strictly, the lowest possible block sum would take both ³Πg components instead of ³Πg + ¹Πg. Neither flow goes there: H conserves spin, so nothing pulls the singlet toward a triplet, and it stays a singlet. The result is exactly the set we wanted: S0, one triplet and one singlet of the same excitation.

---

Code in `code/n2_pes/frozen_core/`: greedy `sv_block.jl`, full flow `cflow_block.jl`, plots `plot_block.py` (written to `sv_results/plots/`, not in git).

```
julia -t 8 --project=../../integrals sv_block.jl data_unocore/R_1.10 uno uno_1.10_block1c_exact
julia --project=../../integrals cflow_block.jl data_unocore/R_1.10 uno uno_1.10_flow1c_d1c --gen d1c --coset 0 --s-max 60
julia --project=../../integrals cflow_block.jl data_unocore/R_1.10 uno uno_1.10_flow1c_d1c --gen d1c --coset a --s-max 60
julia --project=../../integrals cflow_block.jl --merge uno_1.10_flow1c_d1c
python plot_block.py
```
