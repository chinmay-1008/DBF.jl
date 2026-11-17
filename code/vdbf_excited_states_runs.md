# Excited states of N₂ with block DBF. Part 1: R = 1.1 Å

Energies in mHa (1 mHa = 0.0272 eV). Exact numbers come from diagonalising the same Hamiltonian. Part 2 (stretched N₂, 2.3 Å) comes after this one.

## 1. What we want

vDBF finds the ground state by rotating H until the reference determinant $|0\rangle$ is decoupled from everything else; then $\langle 0|\bar H|0\rangle = E_0$.

We want a few **excited states** from the same machinery. The idea: instead of decoupling one determinant, decouple a small **block** of determinants, then diagonalise that block. Here the block has 3 determinants, so we get 3 states: the ground state and two excited states.

For this first test we use the **exact** version of DBF (state vector, no Pauli truncation), so whatever error we see belongs to the method itself.

### In pictures: a random 4-qubit matrix

The same idea on a random 4-qubit Hamiltonian (16 × 16), drawn as heatmaps of $|H_{ij}|$. The double-bracket flow rotates $H$; the generator $D$ decides what ends up decoupled.

![Toy model: what each generator D decouples](toy_block/results/toy_generators.png)

- $D = \mathrm{diag}\,H$: everything is decoupled, so $H$ ends diagonal.
- $D = |0\rangle\langle 0|$ (ground-state vDBF): only row and column 0 are decoupled.
- $D = \sum_i Z_i$: blocks by Hamming weight (1 + 4 + 6 + 4 + 1).
- $D = \sum_{y \in S} |y\rangle\langle y|$ (block DBF): exactly the chosen kets $S$ (marked) are decoupled.

![Toy model: the block flow for two different choices of kets](toy_block/results/toy_block.png)

- Rows and columns are ordered with the chosen kets first. As the flow runs, the coupling between the block and the rest goes to zero.
- Whichever three kets we choose, the block ends with the **3 lowest eigenvalues** of $H$. Diagonalising the 3 × 3 block then gives the three states.

![Toy model: with a symmetry, the kets must sit in the right sectors](toy_block/results/toy_symmetry.png)

- The catch is symmetry. This $H$ conserves the Hamming weight, so it is block diagonal from the start, and the flow never moves weight between sectors.
- So the block can only collect the lowest levels **of the sectors its kets are in**.
- Top row: the 3 kets with the lowest diagonal energies sit in sectors of weight 0, 2, 1, but the 3 lowest levels are in sectors 2, 1, 2. The block ends with −1.38, −1.02, −0.51 instead of −1.38, −1.02, −0.84. Its fidelity with the 3 lowest states is $F = 2/3$.
- Bottom row: one ket per lowest level, in that level's sector, gives $F = 1$.
- Section 11 runs this check on N₂.

Script: `code/toy_block/block_heatmaps.py`.

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

## 8. Greedy vDBF with Pauli propagation (cutoffs 10⁻⁵ and 10⁻⁶)

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

### Cutoff 10⁻⁶

This is the same run with a cutoff 10 times smaller. 200 iterations × 50 rotations took 2.25 h, 5 times longer, and $\bar H$ grew to 6.8 million Pauli strings, 6 times more.

**Result** (errors in mHa at the last iteration):

| state | raw | corrected | true |
|---|---|---|---|
| S0 | +0.048 | +0.105 | +0.087 |
| ³Πg | +0.029 | +0.085 | +0.097 |
| ¹Πg | +0.256 | +0.085 | +0.115 |

![Pauli propagation 10⁻⁶: block eigenvalues against block variance at 1.1 Å](n2_pes/frozen_core/sv_results/plots/n2_110_pp_c1e-6_energies.png)

![Pauli propagation 10⁻⁶: zoom on each state, against the exact greedy run](n2_pes/frozen_core/sv_results/plots/n2_110_pp_c1e-6_zoom.png)

- **The circuit is better at the same iteration:** the true errors are 0.09–0.12 mHa, against 0.14–0.20 at 10⁻⁵. Against $V_S$ it still lies on the exact greedy curve.
- **The estimates now follow the true energies.** Corrected is within 0.03 mHa of true for all three states, and the corrected variances stay positive. Raw is within 0.07 mHa for S0 and ³Πg but still 0.14 mHa high for ¹Πg.

**Extrapolation**, per state over $[v, 2v]$:

![Pauli propagation 10⁻⁶: per-state extrapolation, true / corrected / raw](n2_pes/frozen_core/sv_results/plots/n2_110_pp_c1e-6_extrap.png)

| state | true | corrected | raw |
|---|---|---|---|
| S0 | −0.008 ± 0.005 | −0.001 ± 0.007 | +0.013 ± 0.041 |
| ³Πg | −0.011 ± 0.010 | −0.021 ± 0.009 | −0.054 ± 0.011 |
| ¹Πg | −0.032 ± 0.039 | −0.041 ± 0.036 | +0.170 ± 0.030 |
| ³Πg excitation | −0.003 | −0.020 | −0.067 |
| ¹Πg excitation | −0.023 | −0.040 | +0.157 |

- **At 10⁻⁶ the corrected estimate can be trusted.** It extrapolates to within 0.04 mHa of exact for all three states and both excitation energies, about as well as the true energies. That matters for large systems, where there is no exact replay and the corrected estimate is all we have.
- **Raw is still biased for ¹Πg,** by +0.17 mHa.

**Both cutoffs together:**

![Pauli propagation, cutoffs 10⁻⁵ and 10⁻⁶ against the exact greedy run](n2_pes/frozen_core/sv_results/plots/n2_110_pp_compare_zoom.png)

Going from 10⁻⁵ to 10⁻⁶ costs 5 times the time and 6 times the operator size. What it buys is mainly a reliable *estimate*: the corrected values go from overshooting by up to 0.44 mHa to tracking the true energies within 0.03 mHa. The circuit itself improves less, from 0.14–0.20 mHa above exact to 0.09–0.12 mHa at iteration 200.

## 9. Two things worth knowing

- **Symmetry.** $|a\rangle$ and $|b\rangle$ have a different spatial symmetry (Πg) from the ground state (Σg⁺), so H never couples them to $|0\rangle$. In practice both flows decouple $|0\rangle$ on its own and $\{|a\rangle, |b\rangle\}$ on their own.
- **Which excited states come out.** Because 1πg is degenerate, ³Πg has two components at 297.17. Strictly, the lowest possible block sum would take both ³Πg components instead of ³Πg + ¹Πg. Neither flow goes there: H conserves spin, so nothing pulls the singlet toward a triplet, and it stays a singlet. The result is exactly the set we wanted: S0, one triplet and one singlet of the same excitation.

## 10. Does an S² penalty speed up convergence?

**Idea.** Run the flow on $H + \lambda S^2$ instead of $H$: the cost, the gradient and the angles all use $H + \lambda S^2$, while energies and $V_S$ are still reported for $H$. H conserves spin, so $H + \lambda S^2$ has the same eigenstates as H, with every spin-$S$ level raised by $\lambda S(S+1)$. If the flow spends rotations on spin-contaminated directions, the penalty should cut them out and converge faster.

**1.1 Å.** This is the greedy exact flow of section 6 (200 iterations × 50 rotations) at four penalties, compared with bare H.

- **The target.** At λ = 0.01 and 0.02 Ha the lowest-sum target is unchanged: S0 plus the two ³Πg components. At λ = 0.05 and 0.1, the 2λ = 100–200 mHa shift lifts the triplet above ¹Πg, which is only 50.2 mHa higher, so the target becomes the two ¹Πg components.
- **Where the runs end.** All four still end on S0, ³Πg and ¹Πg, as the bare run does.

![S² penalty at 1.1 Å: error against iteration, and its ratio to bare H](n2_pes/frozen_core/sv_results/plots/n2_110_s2pen_iter.png)

![S² penalty at 1.1 Å: error against the block variance of H](n2_pes/frozen_core/sv_results/plots/n2_110_s2pen_vs.png)

| | S0 | ³Πg | ¹Πg | below 0.1 mHa from iteration (S0 / ³Πg / ¹Πg) |
|---|---|---|---|---|
| bare H | 0.0194 | 0.0225 | 0.0369 | 116 / 121 / 146 |
| λ = 0.01 | 0.0185 | 0.0232 | 0.0361 | 116 / 123 / 144 |
| λ = 0.02 | 0.0182 | 0.0232 | 0.0354 | 112 / 122 / 144 |
| λ = 0.05 | 0.0183 | 0.0243 | 0.0355 | 115 / 124 / 145 |
| λ = 0.1 | 0.0195 | 0.0252 | 0.0359 | 117 / 125 / 145 |

(The first three columns are errors in mHa at iteration 200.)

- **There is no speedup at 1.1 Å.**
  - At every iteration, every run is within 15% of bare H.
  - At the end, S0 and ¹Πg are 2–6% ahead and ³Πg is 3–13% behind, with the lag growing with λ.
  - Against $V_S$ the curves coincide.
- **Why.**
  - Near equilibrium the states are already spin-clean: in the bare run, the block states' ⟨S²⟩ stays within 0.011 of 0, 2 and 0 throughout.
  - The nearest state of a different spin is far away: a quintet 623 mHa above S0 in the |0⟩ sector.
  - So the penalty has nothing to remove. What sets the pace is dynamical correlation, which the penalty does not touch.

**2.3 Å, where it does help.** These are the stretched-bond runs from earlier: bare H to 200 iterations, and λ = 0.05 and 0.1 to 500. Here states of different spin are close:

- a quintet 9.0 mHa above S0 in the |0⟩ sector;
- T1 2.8 mHa above S0, and a septet 20 mHa above S0, in the |a⟩, |b⟩ sector.

![S² penalty at 2.3 Å: S0 energy and ⟨S²⟩ against iteration](n2_pes/frozen_core/sv_results/plots/n2_230_s2pen_s0.png)

- **Bare H mixes spins.** The S0 block state reaches ⟨S²⟩ = 2.9 at iteration 9, mixing with the nearby quintet, and is still at 0.76 at iteration 200. S0 is then 1.56 mHa above FCI.
- **λ = 0.05 is the right strength.**
  - The quintet moves to 309 mHa above S0, and S0's ⟨S²⟩ stays below 0.01 from iteration 31 on.
  - It overtakes bare H at iteration 24. At iteration 200 it is 0.33 mHa above FCI, 5× closer than bare H, and 0.010 mHa at iteration 500.
  - T1 ends 0.03 mHa above its exact level, and the third state (a triplet at 120.6) 0.36 mHa.
- **λ = 0.1 is too strong.** S0 is just as clean, but the second block state is pulled off the triplet and ends on singlets (228.0 and 264.0 mHa above FCI), so T1 is lost.

**So far:** in the exact runs the penalty pays off when states of a different spin sit close to the target, as with stretched bonds and open shells. It speeds things up there by turning a gap of a few mHa into hundreds. Near equilibrium it does nothing. λ has to stay small enough that the wanted spin states remain the lowest in their sector; here that means 0.05, not 0.1.

### With Pauli propagation (1.1 Å, cutoff 10⁻⁵)

This is the run of section 8 repeated on $H + \lambda S^2$ with λ = 0.02 and 0.05; it took 14 minutes per run.

- **Estimates.** Raw and corrected come from the eigenvalues of $H + \lambda S^2$ minus $\lambda S(S+1)$, which removes 2λ from ³Πg and nothing from the singlets.
- **True energies.** Every circuit is replayed exactly at every iteration. That gives the true energies and ⟨S²⟩, and the triplet is identified by its ⟨S²⟩.
- **Variances** are those of $H + \lambda S^2$.
- **The baseline.** The no-penalty run was also replayed at every iteration, so its true numbers differ slightly from section 8, which replayed every 5.

![Pauli propagation with the S² penalty: each state's energy against its own variance, true / corrected / raw](n2_pes/frozen_core/sv_results/plots/n2_110_pp_s2pen_traj.png)

True energy of the circuit, error in mHa:

| iteration | 50 | 100 | 150 | 200 (S0 / ³Πg / ¹Πg) |
|---|---|---|---|---|
| bare H | 0.79 | 0.28 | 0.18 | 0.140 / 0.149 / 0.205 |
| λ = 0.02 | 0.70 | 0.26 | 0.16 | 0.120 / 0.125 / 0.176 |
| λ = 0.05 | 0.68 | 0.24 | 0.16 | 0.121 / 0.142 / 0.181 |

(The columns for iterations 50–150 are S0 only.)

![Pauli propagation with λ = 0.05: per-state extrapolation, true / corrected / raw](n2_pes/frozen_core/sv_results/plots/n2_110_pp_c1e-5_s2pen0.05_extrap.png)

Extrapolated over $[v, 2v]$, in mHa from exact:

| state | series | bare H | λ = 0.02 | λ = 0.05 |
|---|---|---|---|---|
| S0 | true | −0.014 ± 0.013 | −0.019 ± 0.025 | −0.011 ± 0.015 |
| | corrected | −0.187 ± 0.001 | −0.144 ± 0.005 | −0.141 ± 0.006 |
| | raw | +0.860 ± 0.073 | +0.257 ± 0.004 | +1.665 ± 0.035 |
| ³Πg | true | −0.013 ± 0.013 | −0.018 ± 0.021 | −0.014 ± 0.028 |
| | corrected | −0.083 ± 0.026 | −0.070 ± 0.007 | −0.065 ± 0.002 |
| | raw | +0.863 ± 0.116 | +0.174 ± 0.076 | −0.018 ± 0.220 |
| ¹Πg | true | −0.042 ± 0.097 | −0.017 ± 0.039 | −0.033 ± 0.055 |
| | corrected | −0.443 ± 0.001 * | −0.270 ± 0.015 | −0.227 ± 0.001 |
| | raw | +0.760 ± 0.036 | +1.884 ± 0.120 | +1.663 ± 0.184 |

\* The corrected variance turned negative, so this fit uses the $[0, 0.2\times10^{-3}]$ Ha² window, as in section 8.

- **The circuit gets somewhat better.** With the penalty, the true energies are 10–17% closer to exact at every iteration. In the energy-against-variance plane (top row) they lie on the same line as without the penalty: the run gets further along it, not onto a different trajectory. Spin contamination was already small without the penalty: the mean |⟨S²⟩ − S(S+1)| goes from 0.0007, 0.0002 and 0.0015 to 0.0005, 0.0002 and 0.0011 at λ = 0.02.
- **The corrected estimates get better.**
  - The overshoot below exact roughly halves for ¹Πg: −0.44 → −0.27 and −0.23 mHa extrapolated.
  - The corrected variances stay positive, so the fallback window is not needed.
  - The corrected ¹Πg excitation energy improves from −0.26 to −0.13 (λ = 0.02) and −0.09 mHa (λ = 0.05). The corrected values still come out too low, by 0.07–0.27 mHa.
- **The raw estimates get worse.** The truncation offset moves between states rather than shrinking:
  - λ = 0.02 brings S0 and ³Πg down to about 0.2 mHa but pushes ¹Πg up to 1.9.
  - λ = 0.05 makes ³Πg almost exact and leaves S0 and ¹Πg 1.7 mHa high.
  - Without the penalty the offset was nearly equal for all three states, so the raw excitation energies were good (+0.003 and −0.10 mHa). With the penalty they are off by up to 1.7 mHa.
- **Extrapolating the true energies works equally well with or without the penalty.** All of them land within 0.04 mHa of exact. The ¹Πg error bar shrinks from ±0.097 to ±0.04–0.06.

**So:** with truncation, the S² penalty helps a little. The circuit improves by 10–17%, and the corrected estimates and their extrapolation improve clearly. The raw estimate becomes less useful for excitation energies. The large gain is still the one seen in the exact runs at 2.3 Å, where states of another spin are close.

### With Pauli propagation (2.3 Å, cutoff 10⁻⁵)

The same comparison at 2.3 Å: bare H, λ = 0.02 and λ = 0.05, 500 iterations each, taking 1.7, 2.7 and 3.0 hours. Every iteration is replayed exactly.

- **Order of the states.** The three block states are kept in the run's own energy order.
- **Targets.** For bare H they are S0, T1 and the septet at 20.1 mHa. With the penalty the septet moves up by 12λ (240 or 600 mHa), so the third target becomes the triplet at 120.6.
- **Grey line.** This is the exact λ = 0.05 run from above, with no truncation.

![Pauli propagation at 2.3 Å with and without the S² penalty: true energy, corrected estimate and ⟨S²⟩ against iteration](n2_pes/frozen_core/sv_results/plots/n2_230_pp_s2pen.png)

True energy of the circuit, in mHa from each run's own target:

| | S0 | T1 | third state |
|---|---|---|---|
| bare H, best (at iteration) | +57 (164) | +148 (27) | +229 (150) |
| bare H, iteration 500 | +97 | +194 | +246 |
| λ = 0.02, best | +32 (119) | +30 (179) | +38 (147) |
| λ = 0.02, iteration 500 | +78 | +73 | +105 |
| λ = 0.05, best | +17 (133) | +134 (108) | +127 (366) |
| λ = 0.05, iteration 500 | +44 | +211 | +127 |

- **Bare H fails.** S0 mixes with the quintet 9 mHa above it: its ⟨S²⟩ is about 5 for most of the run. Its best is 57 mHa above FCI.
- **The penalty keeps S0 clean and brings it closer.**
  - λ = 0.05: ⟨S²⟩ stays below 0.2 and S0 gets to 17 mHa, 3× closer than bare H.
  - λ = 0.02: ⟨S²⟩ is about 0.7 and S0 gets to 32 mHa.
- **Only λ = 0.02 reaches all three targets.**
  - Around iteration 100 it makes the same drop as the exact run, a little later, and gets within 30–40 mHa of all three.
  - λ = 0.05 never makes that drop. Its second state stays on a triplet about 135 mHa up, where the exact run also waits until iteration 90. Its third state is a singlet.
- **No run converges.** Every run is at its best around iterations 110–180 and gets worse after that.
- **The corrected estimate cannot be trusted here.** It keeps going down while the true energy goes up. At iteration 500, λ = 0.05 has corrected S0 and T1 within 8 mHa of exact (−2.8 and −8.2), while the real circuit is 44 and 211 mHa away.

**Extrapolating in the variance.** This is the same [v, 2v] fit as at 1.1 Å, one figure per run. The true rows use the energy and variance of H. The corrected and raw rows use H + λS², as before.

![2.3 Å, bare H: per-state extrapolation, true / corrected / raw](n2_pes/frozen_core/sv_results/plots/n2_230_pp_c1e-5_extrap.png)

![2.3 Å, λ = 0.02: per-state extrapolation, true / corrected / raw](n2_pes/frozen_core/sv_results/plots/n2_230_pp_c1e-5_s2pen0.02_extrap.png)

![2.3 Å, λ = 0.05: per-state extrapolation, true / corrected / raw](n2_pes/frozen_core/sv_results/plots/n2_230_pp_c1e-5_s2pen0.05_extrap.png)

Extrapolated, in mHa from each run's own target ("no window" means fewer than 3 points in [v, 2v]):

| run | series | S0 | T1 | third state |
|---|---|---|---|---|
| bare H | true | no window | +24 ± 746 | −228 ± 1668 |
| | corrected | +78 ± 431 | +81 ± 147 | −56 ± 212 |
| λ = 0.02 | true | no window | 0 ± 3047 | no window |
| | corrected | +26.3 ± 1.3 | −57 ± 48 | −309 ± 301 |
| λ = 0.05 | true | −416 ± 2829 | no window | no window |
| | corrected | +0.003 ± 0.36 * | +76 ± 75 | −90 ± 228 |

\* The corrected variance turned negative, so this fit uses the $[0, 0.2\times10^{-3}]$ Ha² window.

- **The extrapolation does not work at 2.3 Å.**
  - The window is set by the variance at the last iteration. In these runs the variance grows after iteration 100–150, so the last point sits near the largest variance of the whole run.
  - So the window often holds fewer than 3 points. Where it holds more, they come from different stretches of the run, and the error bars are tens to thousands of mHa.
- **Why.** Extrapolation assumes the run moves down a line towards zero variance. Here the true points turn around after their best iteration and move back up: energy and variance grow together (top rows).
- **Two corrected values look good but are not.**
  - λ = 0.02, S0: +26.3 ± 1.3 is a tight fit, but to the corrected estimate, which by iteration 500 is 46 mHa below the true circuit.
  - λ = 0.05, S0: +0.003 ± 0.36 comes from the fallback window after the corrected variance went negative. The circuit itself is 44 mHa off.

**So:** at 2.3 Å the penalty helps the truncated runs as well: S0 is 2–3× closer and stays spin-clean. But cutoff 10⁻⁵ is far too coarse for the stretched bond, with or without the penalty. The best errors are tens of mHa, against 0.01 mHa without truncation and 0.1–0.2 mHa at 1.1 Å with the same cutoff.

A tighter cutoff is the obvious next step. At 1.1 Å, going from 10⁻⁵ to 10⁻⁶ multiplied the operator size by 6 and the run time by 5. At 2.3 Å that would mean about 25–30 million Pauli terms and about 13 hours per run. That is close to this machine's 32 GB of memory.

## 11. Does the block span the right states?

For N₂ we can diagonalise the whole 10-electron space, so we can check directly whether the block ends up spanning the 3 lowest exact states:

$$F = \|P_{\rm low}\, P_{\rm block}\|^2 / 3 .$$

- $P_{\rm low}$ projects on the 3 lowest levels of the flow operator ($H$, or $H + \lambda S^2$) over **all** symmetry sectors, plus any level degenerate with the third.
- $F = 1$ means the block spans exactly those states. $F$ below 0.9 means we chose the wrong kets (the third picture of section 1).
- The flow never leaves a ket's symmetry sector. So in each sector the block holds at most min(kets there, lowest levels there), and $F$ is capped by that count. `sv_block.jl` and `replay_block.jl` print the cap at the start.

![Fidelity of the block with the 3 lowest exact levels, all N₂ runs](n2_pes/frozen_core/sv_results/plots/n2_fidelity.png)

| run | 3 lowest levels (mHa above S0) | cap | exact greedy | Pauli prop. 10⁻⁵ (end / best) |
|---|---|---|---|---|
| 1.1 Å, H | S0, ³Πg 297.17 ×2 | 1 | 0.667 | 0.667 / 0.667 |
| 1.1 Å, H + 0.02 S² | S0, ³Πg 297.17 ×2 | 1 | 0.667 | 0.667 / 0.667 |
| 1.1 Å, H + 0.05 S² | S0, ¹Πg 347.38 ×2 | 1 | 0.667 | 0.667 / 0.667 |
| 2.3 Å, H | S0, T1 2.84, quintet 8.99 | 0.667 | 0.657 | 0.091 / 0.228 |
| 2.3 Å, H + 0.02 S² | S0, T1 2.84, quintet 8.99 | 0.667 | 0.667 | 0.527 / 0.595 |
| 2.3 Å, H + 0.05 S² | S0, T1 2.84, triplet 120.61 ×2 | 1 | **0.999** | 0.507 / 0.625 |

(×2: two degenerate levels; the block needs only one of them. 1.1 Å at cutoff 10⁻⁶: 0.667. The 2.3 Å exact run with λ = 0.02 is new; the others are the runs of sections 6 and 10.)

- **Only one run passes: 2.3 Å with λ = 0.05, without truncation.** The penalty pushes the quintet above the second triplet, so the 3 lowest levels of H + 0.05 S² are the ones our kets can reach.
- **2.3 Å, bare H and λ = 0.02: wrong sectors.**
  - The 3 lowest levels are S0 and the quintet, both in the |0⟩ sector, and T1, in the |a⟩, |b⟩ sector. Our kets are split 1 + 2, the other way round.
  - So F stays at or below 2/3 however long the flow runs. In both runs the third block state ends on the triplet at 120.6 mHa instead of the quintet.
  - Better kets: two in the |0⟩ sector and one in the |a⟩, |b⟩ sector (tested below).
- **1.1 Å: right sectors, wrong pair.**
  - The two lowest excited levels are the two components of ³Πg: 3σg → 1πg and 3σg → 1πg (other component), qubits 4 → 5 and 4 → 7.
  - |a⟩ and |b⟩ are both 4 → 5, in α and β. So they make one triplet and one singlet of the same orbitals, ³Πg and ¹Πg, and never reach the other ³Πg component. F is 2/3 from the first iterations.
  - The energies of section 6 are right; they are just not the 3 lowest levels. The cap (1) misses this because it counts sectors, not spin.
  - Better kets: |0⟩, 4 → 5 and 4 → 7, in the same spin (tested below). With λ = 0.05 the singlet pair is lowest, and the same thing happens with singlet and triplet swapped.
- **Pauli propagation at 2.3 Å loses fidelity as the run goes on.** Every run ends below its best, as its energies did (section 10).

### Testing better kets

Same exact greedy runs, with kets chosen from the sectors (and spins) of the 3 lowest levels. Each ket is written as the qubits flipped from the reference, all in α (`sv_block.jl --kets`):

- 1.1 Å: |0⟩, 4 → 5 and 4 → 7, i.e. 3σg → 1πg into each of the two πg orbitals (`--kets ref,4.5,4.7`);
- 2.3 Å: |0⟩, the double excitation 3,4 → 5,6 (the kind of determinant the quintet is made of) and the single excitation 3 → 5 (part of T1) (`--kets ref,3.4.5.6,3.5`).

![Fidelity of the old kets against the new kets, exact greedy runs](n2_pes/frozen_core/sv_results/plots/n2_fidelity_kets.png)

| run | F at the end (old kets) | F > 0.9 from iteration | error at the end (mHa) |
|---|---|---|---|
| 1.1 Å, H | **0.992** (0.667) | 45 | S0 +0.14, ³Πg +0.41, ³Πg +1.37 |
| 1.1 Å, H + 0.05 S² | **0.991** (0.667) | 51 | S0 +0.15, ¹Πg −0.80, ¹Πg +0.15 |
| 2.3 Å, H | **0.998** (0.657) | 14 | S0 +0.009, T1 +0.031, quintet +0.004 |
| 2.3 Å, H + 0.02 S² | **0.998** (0.667) | 16 | S0 +0.001, T1 +0.001, quintet +0.022 |

(1.1 Å: 200 iterations; 2.3 Å: 500. Errors are from each state's own exact level.)

- **Both pass.** Choosing kets by the sectors and spins of the lowest levels gives the right subspace.
- **2.3 Å now works with bare H, no penalty needed.**
  - At iteration 200, S0 is 0.07 mHa off and spin-clean. With the old kets it was 1.56 mHa off with ⟨S²⟩ = 0.76.
  - That old spin mixing was the quintet, which the old block had no room for. Now the quintet is one of the block states.
- **1.1 Å converges more slowly than with the old kets.**
  - At iteration 200: S0 0.14 mHa against 0.02, and the two ³Πg components 0.4 and 1.4 mHa.
  - Each new ket (one α excitation) is half triplet and half singlet, so the flow also has to separate the spins.
  - With λ = 0.05 one ¹Πg ends 0.8 mHa below its exact level, because it still holds ⟨S²⟩ = 0.05 of triplet.

## 12. Scaling up: the 1×100 Heisenberg chain

**System.** Open spin-½ chain, $H = J\sum_i \mathbf S_i\cdot\mathbf S_{i+1}$, $N = 100$, $J = 1$. Energies are in units of $J$ and errors in mJ ($10^{-3} J$). At 100 qubits nothing can be checked exactly, so the references come from DMRG (ITensors, bond dimension up to 400, $S_z$ conserved):

| | DMRG energy |
|---|---|
| S0, the ground state (a singlet) | −44.12774 |
| T, the lowest triplet ($M_s = 0$ and $M_s = +1$ components) | −44.08730 |
| spin gap T − S0 | 40.44 mJ |

**Sites and qubits.** Site $i$ (1–100) is qubit $i - 1$. The reference is the Néel state: ↑ on odd sites, ↓ on even sites. Every bond is antiparallel and contributes $-J/4$, so its energy is $-(N-1)/4 = -24.75$, which is 19.4 J above the ground state. Flipping every other qubit maps Néel to $|00\cdots0\rangle$. So, as for N₂, a 1 on a qubit means that site's spin is flipped relative to Néel.

**The block: three determinants at the centre.** Only sites 50 and 51 (qubits 49 and 50) differ between them; the other 98 sites stay Néel.

| | … 48 | 49 | **50** | **51** | 52 | 53 … | $S_z$ | energy |
|---|---|---|---|---|---|---|---|---|
| $\lvert 0\rangle$, Néel | ↓ | ↑ | ↓ | ↑ | ↓ | ↑ | 0 | −24.75 |
| $\lvert f\rangle$, flip-flop | ↓ | ↑ | **↑** | **↓** | ↓ | ↑ | 0 | −23.75 |
| $\lvert s\rangle$, single flip | ↓ | ↑ | **↑** | ↑ | ↓ | ↑ | +1 | −23.75 |

- **$|f\rangle$** exchanges the antiparallel centre pair. Bond 50–51 stays antiparallel, but bonds 49–50 and 51–52 become parallel, which costs $J/2$ each.
- **$|s\rangle$** flips site 50 alone, which makes bonds 49–50 and 50–51 parallel.
- **Coupling.** H connects $|0\rangle$ and $|f\rangle$ directly: the flip-flop term on bond 50–51 has matrix element $J/2$. $|s\rangle$ is in a different $S_z$ sector.

**Exact states we target.**

| exact state | $S_z$ sector | DMRG energy | grows from | weight of the exact state on those determinants |
|---|---|---|---|---|
| S0 | 0 | −44.12774 | $\lvert 0\rangle, \lvert f\rangle$ | 5.6 × 10⁻⁹ |
| T, $M_s = 0$ | 0 | −44.08730 | $\lvert 0\rangle, \lvert f\rangle$ | 1.1 × 10⁻⁸ |
| T, $M_s = +1$ | +1 | −44.08730 | $\lvert s\rangle$ | 6.4 × 10⁻¹² |

- **Which state comes from where.** H conserves $S_z$, so S0 and T($M_s = 0$) are the two eigenvalues of the 2×2 $S_z = 0$ block $\{|0\rangle, |f\rangle\}$, and T($M_s = +1$) comes from $|s\rangle$ alone. The two triplet components have the same exact energy, which gives a built-in check.
- **The start.** The $S_z = 0$ block is $\begin{pmatrix} -24.75 & 0.5 \\ 0.5 & -23.75 \end{pmatrix}$, with eigenvalues −24.96 and −23.54; $|s\rangle$ gives −23.75. These are 19.2, 20.5 and 20.3 J above their DMRG levels.
- **The big difference from N₂ is the weight column.** In N₂ the three determinants already held about 90% of each target state. Here they hold about 10⁻⁸, because the exact 100-site states are spread over an enormous number of determinants. The flow has to rotate essentially all of the weight into the block.

**What we change in D.** Ordinary DBF on the chain uses $D = \sum_k Z_k$ over all 100 qubits; its top value (100) belongs to Néel alone. We keep $Z_k$ on the other 98 sites and replace $Z_c + Z_{c+1}$ (sites 50 and 51) with

$$D_\text{centre} = Z_c Z_{c+1} - Z_c + Z_{c+1} .$$

| sites 50, 51 flipped? | spins | meaning | $D_\text{centre}$ |
|---|---|---|---|
| no, no | ↓↑ | $\lvert 0\rangle$, Néel | +1 |
| yes, yes | ↑↓ | $\lvert f\rangle$ | +1 |
| yes, no | ↑↑ | $\lvert s\rangle$, $S_z = +1$ | +1 |
| no, yes | ↓↓ | the $S_z = -1$ partner of $\lvert s\rangle$ | −3 |

Flipping any other site lowers $D$ by 2. So **the top value of $D$ (99) belongs to exactly $|0\rangle$, $|f\rangle$ and $|s\rangle$**, checked against every single and double flip. The −3 on ↓↓ keeps only one triplet component, $M_s = +1$, in the block. Without it, the block would get a fourth state, the $M_s = -1$ component, which would only repeat the triplet energy.

**Why the centre, and is it the best place?** We put the block at the centre for two reasons. The ends of an open chain are special, because an end site has only one neighbour. And the chain is mirror-symmetric about bond 50–51, so the centre treats both halves alike. To check whether the centre is also where the target states overlap the block most, DMRG gives the weight of each exact state on the block determinant placed at every position:

![1×100 Heisenberg: weight of the exact states on the block determinants at every position](heisenberg/results/plots/heis_N100_position.png)

| determinant | at the centre (used) | best position | worst position |
|---|---|---|---|
| $\lvert f\rangle$ on S0 | 7.1 × 10⁻¹⁰ (bond 50–51) | 2.0 × 10⁻⁹ (end bonds 1–2, 99–100) | 4.7 × 10⁻¹⁰ (bonds 2–3, 98–99) |
| $\lvert s\rangle$ on T, $M_s = +1$ | 6.4 × 10⁻¹² (site 50) | 5.9 × 10⁻¹¹ (site 100) | 3.4 × 10⁻¹⁶ (site 2) |

- **The overlaps do not favour the centre.**
  - The flip-flop does about equally well on every bulk bond, and 3× better on the end bonds.
  - The single flip gets steadily better from the left end to the right end. The centre is in the middle of that range, and the last site is 9× better.
- **Why the single flip is lopsided.** In T($M_s = +1$) the staggered magnetisation (right panel) changes sign at the centre: the left half follows our Néel pattern (↑ on odd sites) and the right half the opposite one. A ↓ → ↑ flip on the right half agrees with the triplet's local pattern; on the left half it goes against it. With the other Néel state, or $M_s = -1$, the picture is mirrored.
- **Whether it matters for convergence is untested.** All of these weights are tiny (10⁻⁹ to 10⁻¹¹), and the flow has to build almost the whole state either way, so a 3–9× better start may or may not converge faster. Testing it needs a block at the end of the chain, with the flip-flop on bond 99–100 and the single flip on site 100. That has not been run.

So the centre was a symmetry choice. By starting overlap, the right end would have been better.

**Run.** The same greedy Pauli-propagation loop as section 8, at cutoff $10^{-5}$ with 100 rotations per iteration. It was stopped at iteration 360: 36,000 rotations in 8.1 h, $\bar H$ at 16.5 million Pauli strings, and the last iterations taking about 140 s each.

A run at cutoff $10^{-4}$ is left out. From about iteration 120 on, its truncation correction runs away: by iteration 160 corrected and raw S0 differ by 300 mJ, the corrected S0 variance turns negative at iteration 511, and the run ends with the corrected S0 853 mJ *below* the exact ground state.

**Target: 1% of each state's energy**, $|E - E_\text{DMRG}| \le 0.01\,|E_\text{DMRG}| \approx 441$ mJ. It is the grey band in the figures.

![1×100 Heisenberg: block eigenvalues against block variance](heisenberg/results/plots/heis_N100_energies.png)

![1×100 Heisenberg: zoom on each state](heisenberg/results/plots/heis_N100_zoom.png)

**Result** at iteration 360, in mJ from each state's own DMRG energy (relative error in brackets):

| state | raw | corrected | inside 1% (corrected) since |
|---|---|---|---|
| S0 | +235 (0.53%) | +211 (0.48%) | iteration 44 |
| T, $M_s = 0$ | +446 (1.01%) | +426 (0.97%) | iteration 326 |
| T, $M_s = +1$ | +378 (0.86%) | +341 (0.77%) | iteration 156 |

- **All three corrected energies are inside the 1% target**, T($M_s = 0$) only just. Raw T($M_s = 0$) is still 0.01% outside.
- **Truncation is under control at $10^{-5}$:** raw and corrected differ by 20–40 mJ, and the corrected variances stay positive.

**Extrapolation**, per state over $[v, 2v]$ as in sections 6 and 8:

![1×100 Heisenberg: per-state extrapolation, corrected and raw](heisenberg/results/plots/heis_N100_extrap.png)

| state | corrected | raw |
|---|---|---|
| S0 | −11 ± 41 (0.02%) | +63 ± 63 (0.14%) |
| T, $M_s = 0$ | −102 ± 30 (0.23%) | −98 ± 69 (0.22%) |
| T, $M_s = +1$ | −22 ± 28 (0.05%) | +58 ± 56 (0.13%) |

- **Energy against variance is a straight line** over the last 300 iterations, as in ground-state vDBF, so the extrapolation is well defined.
- **The errors shrink by about 10×.** All six extrapolated values are within 0.25% of DMRG. S0 and T($M_s = +1$) land within their error bars.
- **T($M_s = 0$) overshoots**, to about −100 mJ, outside its error bar. Its line is steeper than the other two. It is the second state of the two-state $S_z = 0$ block, while T($M_s = +1$) is the lowest state of its own sector.

![1×100 Heisenberg: spin-gap estimates against block variance](heisenberg/results/plots/heis_N100_gap.png)

- **The spin gap is not resolved.** 40 mJ is 0.09% of the energy, well inside the 1% target.
  - Directly at iteration 360, the estimates are 171 mJ ($M_s = +1$) and 256 mJ ($M_s = 0$).
  - From the extrapolated corrected energies, they are 29 ± 49 mJ and −51 ± 50 mJ.
  - Resolving it needs errors around 1–5 mJ, ten times below what extrapolation gives now.
- **Why this is harder than N₂.** The Néel start is 19.4 J above the ground state, 480 times the gap. The block determinants hold about 10⁻⁸ of the target states, against about 90% in N₂. In N₂ the start was 0.16 Ha above, about half the excitation energies.

---

Code in `code/n2_pes/frozen_core/`: greedy `sv_block.jl`, full flow `cflow_block.jl`, plots `plot_block.py` (written to `sv_results/plots/`, not in git).

```
julia -t 8 --project=../../integrals sv_block.jl data_unocore/R_1.10 uno uno_1.10_block1c_exact
julia --project=../../integrals cflow_block.jl data_unocore/R_1.10 uno uno_1.10_flow1c_d1c --gen d1c --coset 0 --s-max 60
julia --project=../../integrals cflow_block.jl data_unocore/R_1.10 uno uno_1.10_flow1c_d1c --gen d1c --coset a --s-max 60
julia --project=../../integrals cflow_block.jl --merge uno_1.10_flow1c_d1c
python plot_block.py
julia -t 8 --project=../../integrals sv_block.jl data_unocore/R_1.10 uno uno_1.10_block1c_exact_s2pen0.05 --s2-penalty 0.05
cd .. && julia --project=. run_block_pp.jl frozen_core/data_unocore/R_1.10 uno uno_1.10_blockpp_c1e-5_s2pen0.05 --coeff 1e-5 --s2-penalty 0.05
cd frozen_core && julia -t 4 --project=../../integrals replay_block.jl data_unocore/R_1.10 uno uno_1.10_blockpp_c1e-5_s2pen0.05 1
cd .. && julia --project=. run_block_pp.jl frozen_core/data_unocore/R_2.30 uno uno_2.30_blockpp_c1e-5_s2pen0.02 --coeff 1e-5 --s2-penalty 0.02 --max-iter 500
cd frozen_core && julia -t 3 --project=../../integrals replay_block.jl data_unocore/R_2.30 uno uno_2.30_blockpp_c1e-5_s2pen0.02 1
python plot_block.py --penalty          # section 10 figures only
```

Heisenberg chain, in `code/heisenberg/` (plots in `results/plots/`, not in git):

```
julia --project=. dmrg_heisenberg.jl 100
julia --project=. position_check.jl 100        # weights of the DMRG states on the block determinants at every position
julia --project=../n2_pes run_block_heisenberg.jl 100 heis_N100_block_c1e-5 --coeff 1e-5 --max-rots 100 --max-iter 600
python plot_heisenberg.py 100
```
