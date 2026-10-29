# From Hartree–Fock to vDBF: a study guide built around N₂ bond breaking

A self-contained syllabus for exam preparation. It covers mean-field theory
(HF, RHF, UHF), spin, electron correlation (CI, CC), density matrices and
natural orbitals (NO, UNO), localization (LMO, LNO), and how every one of
these ideas shows up in the double-bracket-flow ground-state method (DBF /
vDBF) used in this project. Every concept is anchored to real numbers from
our N₂ / STO-3G calculations at R = 2.3 Å (stretched, nearly dissociated).

---

## 0. How to use this file with Claude

Paste this file into a new conversation (or attach it) and start with one of
these prompts:

- **Tutor mode.** "You are my tutor for a PhD qualifying exam in electronic
  structure and quantum algorithms. Teach me this guide one section at a
  time. For each section: explain it at board level, derive the key
  equations with me, then ask me 2–3 questions and wait for my answers
  before moving on. Correct me precisely. Use the N₂ numbers in the guide
  as examples."
- **Quiz mode.** "Quiz me on sections X–Y. Mix conceptual, derivation and
  'explain the plot' questions. Do not give answers until I try."
- **Oral-exam mode.** "Play a committee member. Ask me open questions about
  the N₂ results in section 13, interrupt with follow-ups, and push on
  anything vague."
- **Gap check.** "Here is my explanation of <topic>. Grade it like an
  examiner and list what is missing or wrong."

Suggested order: 1 → 2 → 3 → 4 → 5 → 6 → 7 → 8 → 9 → 10 → 11 → 12 → 13.
Each section ends with **Check yourself** questions; answers are folded
underneath.

---

## 1. The electronic structure problem

### 1.1 The Hamiltonian
Born–Oppenheimer approximation: nuclei are frozen; for each geometry R we
solve for electrons only. The electronic Hamiltonian (atomic units):

$$
H = -\sum_i \tfrac12\nabla_i^2 - \sum_{i,A}\frac{Z_A}{r_{iA}} + \sum_{i<j}\frac{1}{r_{ij}} + E_{\rm nuc}
$$

The potential energy surface (PES) is the ground-state energy E(R) as a
function of geometry. For N₂ we scan the bond length R.

### 1.2 Basis sets and orbitals
- We expand orbitals in a finite **basis set** of atom-centred functions.
  STO-3G = minimal basis: one function per occupied atomic shell (N: 1s, 2s,
  2pₓ, 2p_y, 2p_z → 5 per atom, 10 for N₂).
- A **molecular orbital** (MO) is a linear combination of basis functions,
  φ_p = Σ_μ C_μp χ_μ. The basis functions are not orthogonal (overlap
  matrix S), so orthonormal MOs satisfy CᵀSC = 1.
- A **spin-orbital** is a spatial orbital times a spin function (α or β).
  10 spatial orbitals → 20 spin-orbitals → 20 qubits.

### 1.3 Slater determinants
Electrons are fermions, so the wavefunction must be antisymmetric. The
simplest antisymmetric N-electron state is a **Slater determinant**: pick N
spin-orbitals and antisymmetrize their product. Notation |φ₁φ₂…φ_N⟩, or as
an occupation bit string |11111000…⟩.

All determinants built from a fixed orbital set form a complete basis of the
N-electron space (in that one-electron basis). Expanding the exact state in
all of them is **Full CI (FCI)**: the exact answer in the given basis.

### 1.4 Second quantization
Creation/annihilation operators a†_p, a_p with {a_p, a†_q} = δ_pq. Then

$$
H = \sum_{pq} h_{pq}\, a^\dagger_p a_q + \tfrac12 \sum_{pqrs} g_{pqrs}\, a^\dagger_p a^\dagger_q a_s a_r + E_{\rm nuc}
$$

- h_pq: one-electron integrals (kinetic + nuclear attraction).
- g_pqrs = (pr|qs): two-electron repulsion integrals.
- Changing orbitals (a unitary rotation C → CU) changes the numbers h and g
  but not the physics. FCI energy is invariant; what changes is how
  *compact* the wavefunction looks and how *sparse* H looks.

**Slater–Condon rules.** H couples determinants that differ by at most two
spin-orbitals (H has at most two-body terms). So ⟨Φ|H|Φ'⟩ = 0 if Φ' is a
triple or higher excitation of Φ. *Keep this in mind for vDBF: a state that
needs large weight on a 6-fold excitation can only be reached through
chains of smaller steps.*

### 1.5 From fermions to qubits: Jordan–Wigner
Each spin-orbital becomes a qubit, |1⟩ = occupied:

$$
a^\dagger_p = \Big(\prod_{q<p} Z_q\Big)\frac{X_p - iY_p}{2}, \qquad n_p = a^\dagger_p a_p = \frac{I - Z_p}{2}
$$

H becomes a sum of Pauli strings: H = Σ_k c_k P_k, P_k ∈ {I,X,Y,Z}^⊗n.
Every Pauli string can be written as (phase) · X^x Z^z with bit strings x, z.
The **x-string** says which qubits it flips. Acting on a basis state, a
Pauli string maps |b⟩ → (phase)|b ⊕ x⟩. So terms with x = 0 are diagonal
(energies of determinants); terms with x ≠ 0 connect determinant b to
determinant b ⊕ x (an excitation).

Our frozen-core N₂ (Section 10): 8 spatial orbitals, 5α + 5β electrons → 16
qubits. Number of Pauli terms in H depends on the orbitals: 1177 (UNO) vs
2465 (LNO) vs 4397 (localized UNO).

<details><summary>Check yourself</summary>

1. Why is FCI basis-dependent but orbital-invariant? → It depends on the
   one-electron basis set (STO-3G), but within that space, any unitary
   rotation of orbitals gives the same set of N-electron states, just in a
   different determinant basis.
2. Why can a Pauli-string Hamiltonian have more terms in one orbital basis
   than another? → Rotating orbitals mixes h and g; symmetry-adapted
   orbitals make many integrals exactly zero by symmetry, localized or
   symmetry-broken orbitals make them small but nonzero.
3. What does ⟨x|H|0⟩ mean physically? → The Hamiltonian coupling between the
   reference determinant and the determinant reached by flipping the bits
   in x (an excitation).
</details>

---

## 2. Hartree–Fock (HF)

### 2.1 The idea
Find the single determinant with the lowest energy (variational principle:
E[Φ] ≥ E₀ for any normalized trial Φ). Each electron sees the *average*
field of the others: a mean-field theory.

### 2.2 The equations
Minimizing ⟨Φ|H|Φ⟩ over orthonormal orbitals gives the Fock eigenvalue
problem

$$
F\,C = S\,C\,\varepsilon, \qquad F_{\mu\nu} = h_{\mu\nu} + \sum_{\lambda\sigma} P_{\lambda\sigma}\big[(\mu\nu|\lambda\sigma) - \tfrac12(\mu\lambda|\nu\sigma)\big]\ \text{(RHF form)}
$$

- J (Coulomb): classical repulsion from the average charge.
- K (exchange): purely quantum, from antisymmetry; only between same-spin
  electrons.
- F depends on the density P built from the occupied orbitals → solve
  iteratively: **self-consistent field (SCF)**. Convergence aids: DIIS,
  level shifting, smearing, second-order (Newton) steps.

### 2.3 Key properties
- **Canonical orbitals**: eigenvectors of F, with orbital energies ε. They
  are usually delocalized over the molecule.
- **Invariance**: rotating occupied orbitals among themselves (or virtuals
  among themselves) does not change the determinant or its energy. This is
  why we can **localize** orbitals freely (Section 9).
- **Brillouin's theorem**: at a converged HF solution, ⟨Φ_i^a|H|Φ⟩ = 0 for
  every single excitation i → a. (The HF determinant is stationary with
  respect to orbital rotations.)
- **Koopmans' theorem**: −ε_i ≈ ionization energy (frozen orbitals, no
  correlation).
- **Stability**: the SCF solution is a stationary point, not necessarily a
  minimum. Stability analysis checks the orbital Hessian. *Internal*
  instability: a lower solution of the same type exists. *External*
  instability: a lower solution exists if a constraint is relaxed
  (e.g. RHF → UHF).

### 2.4 Correlation energy
E_corr = E_FCI − E_HF. Everything HF misses because each electron only feels
the average of the others.

<details><summary>Check yourself</summary>

1. Why is HF variational? → It minimizes ⟨Φ|H|Φ⟩ over a subset of normalized
   states (single determinants), so E_HF ≥ E_exact in the same basis.
2. What does Brillouin's theorem imply for CI with singles only (CIS) on the
   ground state? → Singles don't couple directly to Φ, so CIS doesn't improve
   the ground-state energy.
3. Why can we localize HF orbitals without changing the energy? →
   Occupied–occupied rotations leave the determinant unchanged (up to a
   phase).
</details>

---

## 3. Restricted Hartree–Fock (RHF)

### 3.1 Definition
α and β electrons share the *same spatial orbitals*; each occupied orbital
is doubly occupied (closed shell). Constraint: C_α = C_β.

### 3.2 Good: spin purity
A closed-shell determinant is an exact singlet (S = 0, ⟨S²⟩ = 0). Reason: in
each doubly occupied orbital the two electrons are forced into the
antisymmetric spin pairing.

### 3.3 Bad: bond breaking
**H₂ in a minimal basis.** σ_g = (1s_A + 1s_B)/√2. The RHF determinant
|σ_g α σ_g β⟩ expands into

$$
|\sigma_g^2\rangle = \tfrac12\big(\underbrace{|A\bar A\rangle + |B\bar B\rangle}_{\text{ionic: H}^-\text{H}^+} + \underbrace{|A\bar B\rangle + |B\bar A\rangle}_{\text{covalent: H·H·}}\big)
$$

It is 50% ionic at *every* distance. At dissociation the true state is 100%
covalent (two H atoms), so RHF energy is far too high and RHF is **not
size-consistent** for bond breaking.

The exact (FCI) minimal-basis H₂ ground state is

$$
\Psi = c_1 |\sigma_g^2\rangle - c_2 |\sigma_u^2\rangle, \qquad c_1 = c_2 = 1/\sqrt2 \text{ at dissociation}
$$

The doubly excited determinant |σ_u²⟩ cancels the ionic part. This is
**static (nondynamic) correlation**: a second determinant becomes as
important as the first.

**N₂ is three of these at once** (σ, πₓ, π_y bonds). At R = 2.3 Å:

| Method | E (Ha) | E − E_FCI |
|---|---|---|
| RHF | −106.9804 | **+462 mHa** |
| UHF | −107.4362 | +6.4 mHa |
| FCI | −107.4427 | 0 |

RHF is externally (RHF → UHF) unstable here.

<details><summary>Check yourself</summary>

1. Why is |σ_g²⟩ half ionic? → Expand the product (A+B)(A+B): AA and BB terms
   put both electrons on one atom.
2. Why is FCI for H₂ only two determinants (by symmetry) in minimal basis? →
   Only σ_g and σ_u exist; the singlet Σ_g⁺ ground state can only contain
   σ_g² and σ_u² (σ_gσ_u open-shell is u-symmetry).
3. What is an external instability? → The RHF solution is a saddle point in
   the larger UHF space: relaxing α = β lowers the energy.
</details>

---

## 4. Unrestricted Hartree–Fock (UHF)

### 4.1 Definition
α and β electrons get *different* spatial orbitals (C_α ≠ C_β, "different
orbitals for different spins"). Two coupled Fock equations F_α, F_β.

### 4.2 Symmetry breaking at stretched bonds
Past the **Coulson–Fischer point** UHF finds a lower solution in which α
electrons localize on one atom and β on the other. For H₂: α on A, β on B →
|A B̄⟩, purely covalent, correct energy at dissociation. Size-consistent
energy, but…

### 4.3 Spin contamination
A UHF determinant is *not* an S² eigenstate. Its expectation value:

$$
\langle S^2\rangle_{\rm UHF} = S_z(S_z+1) + N_\beta - \sum_{i,j}^{\rm occ} |\langle \phi_i^\alpha|\phi_j^\beta\rangle|^2
$$

For N₂ at dissociation each N atom is a quartet (three unpaired 2p
electrons, Hund's rule, S_atom = 3/2). UHF puts the α spins on atom A and β
on atom B:

$$
|\text{UHF}\rangle \approx |S{=}\tfrac32, M{=}{+}\tfrac32\rangle_A\,|S{=}\tfrac32, M{=}{-}\tfrac32\rangle_B
$$

Coupling two spin-3/2 (Section 5) gives total S = 0, 1, 2, 3 with
Clebsch–Gordan weights **1/4, 9/20, 1/4, 1/20**. Measured on our 2.3 Å UHF
state: 0.254, 0.449, 0.242, 0.047. Hence ⟨S²⟩ = 0·¼ + 2·(9/20) + 6·¼ + 12·(1/20)
= 3.0 (measured 2.939). The singlet is only 25% of the UHF state.

### 4.4 Why this hurts at dissociation
At large R the spin multiplets become nearly degenerate (the atoms stop
talking to each other). At 2.3 Å, above the singlet ground state:

| State | Gap |
|---|---|
| triplet | 2.84 mHa |
| quintet | 9.0 mHa |
| septet | 20.1 mHa |

So any method that starts from UHF and lowers ⟨H⟩ has very little energetic
reason to remove the triplet/quintet components: they cost only a few mHa.
The weighted average of the UHF mixture lies ≈ 4.5 mHa above the singlet,
exactly where our truncated vDBF runs stalled.

### 4.5 Fixes for spin contamination
- **Löwdin projection**: P_S = Π_{k≠S} [S² − k(k+1)] / [S(S+1) − k(k+1)].
  For the singlet out of S = 0..3: P₀ = (S²−2)(S²−6)(S²−12)/(−144).
  Projected UHF (PUHF) = P₀|UHF⟩ normalized.
- **Spin penalty**: work with H + λS². The singlet is unchanged; a spin-S
  state is raised by λS(S+1). Natural choice λ* = (E_singlet,1 −
  E_triplet)/2 (so the lowest triplet moves above the first excited
  singlet); at 2.3 Å λ* = 0.112, we used λ = 0.05 with the same results.
- **Restricted reference** (RHF, UNO) with an exact singlet start.
- Spin-adapted methods (configuration state functions, CSFs).

<details><summary>Check yourself</summary>

1. Why does UHF give the right dissociation energy but the wrong state? →
   It describes each atom correctly, but couples their spins as a mixture of
   all total S instead of the singlet.
2. Derive ⟨S²⟩ for |A B̄⟩ in H₂ at dissociation. → S_z = 0, N_β = 1, overlap
   ⟨A|B⟩ = 0 → ⟨S²⟩ = 1 = half singlet (0) + half triplet (2).
3. Why is λ* tied to the triplet–excited-singlet gap? → λ must push every
   spin state above the target, but making λ huge makes the problem stiff;
   the minimal useful λ lifts the lowest triplet past the first excited
   singlet.
</details>

---

## 5. Spin: the algebra you must know cold

- S_z = ½ Σ_p (n_pα − n_pβ). S₊ = Σ_pq O_pq a†_pα a_qβ with O_pq = ⟨φ_pα|φ_qβ⟩
  (O = 1 for restricted orbitals). S² = S₋S₊ + S_z² + S_z.
- Eigenvalues S(S+1): singlet 0, triplet 2, quintet 6, septet 12.
- A single determinant is always an S_z eigenstate, but an S² eigenstate
  only if it is closed-shell (or high-spin). Open-shell singlets need
  **two** determinants: (|aᾱ b̄β| − |āβ bα|)/√2-type combinations (CSFs).
- **Addition of angular momentum.** Two spins j₁, j₂ combine to J = |j₁−j₂|
  … j₁+j₂. Count states: 3/2 ⊗ 3/2 has 16 states; M = −3..3 multiplicities
  1,2,3,4,3,2,1 → peel off J = 3 (7 states), J = 2 (5), J = 1 (3), J = 0 (1).
- **Clebsch–Gordan weights** for |3/2,+3/2⟩|3/2,−3/2⟩ on J = 0,1,2,3:
  1/4, 9/20, 1/4, 1/20.
- **Corresponding orbitals** (Amos–Hall): SVD of the occupied α–β overlap
  gives paired orbitals with overlaps d_i = cos θ_i. Then
  ⟨S²⟩ = S_z(S_z+1) + Σ_i (1 − d_i²). Paired orbitals with d = 1 contribute
  nothing; fully split pairs (d = 0) contribute 1 each.

<details><summary>Check yourself</summary>

1. Why is a closed-shell determinant a singlet? → Each doubly occupied
   orbital contributes a spin-paired, antisymmetric spin function with S = 0.
2. Why does a single Pauli rotation in vDBF typically break spin? → It adds
   amplitude on one determinant; an open-shell singlet needs two
   determinants with equal amplitude, so one alone is half singlet, half
   triplet.
</details>

---

## 6. Electron correlation and Full CI

- **Dynamic correlation**: electrons avoiding each other at short range;
  many small contributions from many determinants.
- **Static (nondynamic) correlation**: a few determinants become nearly
  degenerate (bond breaking, diradicals). The HF reference is qualitatively
  wrong. Diagnostic: natural occupation numbers far from 2/0 (Section 8).
- **Configuration interaction**: Ψ = c₀Φ₀ + Σ c_i^a Φ_i^a + Σ c_ij^ab Φ_ij^ab + …
  Truncated CI (CISD) is variational but **not size-extensive** (for two
  noninteracting molecules, CISD ≠ 2 × CISD).
- **FCI**: all determinants; exact in the basis; cost grows combinatorially.
  Our frozen-core (5α,5β) in 8 orbitals: (8 choose 5)² = 3136 determinants.
- **Excitation-level structure** of the exact state matters for any method
  that builds it from a reference (CC and DBF): see Section 12.4.

<details><summary>Check yourself</summary>

1. Why is truncated CI not size-extensive? → For A + B far apart, the exact
   state contains products of A's doubles and B's doubles = quadruples,
   which CISD leaves out.
2. Static vs dynamic in one sentence each. → Static: a few configurations
   are nearly degenerate and must be treated equally. Dynamic: many small
   corrections for short-range electron avoidance.
</details>

---

## 7. Coupled cluster (CCSD, CCSD(T))

### 7.1 Ansatz
Ψ = e^T Φ₀, T = T₁ + T₂ + …, T₂ = ¼ Σ t_ij^ab a†_a a†_b a_j a_i.
The exponential generates products (T₂²/2 gives quadruples) → **size
extensive**.

### 7.2 Equations
Project the similarity-transformed Hamiltonian H̄ = e^{−T} H e^{T}:
E = ⟨Φ₀|H̄|Φ₀⟩, 0 = ⟨Φ_μ|H̄|Φ₀⟩ for each excitation μ in T.
H̄ is not Hermitian → CC is **not variational** (can go below FCI).

- CCSD: T₁ + T₂. Cost N⁶.
- CCSD(T): perturbative triples, N⁷. "Gold standard" near equilibrium.
- Λ equations: left-hand amplitudes needed for densities and properties
  (we used them to build the UCCSD one-particle density for NOs).

### 7.3 At stretched bonds
- **RCCSD** (on RHF) fails: the reference is 50% ionic per bond, T must be
  huge, equations become ill-conditioned or diverge, CCSD(T) can drop below
  FCI. At 2.3 Å our RCCSD did **not converge** (last energy +241 mHa).
- **UCCSD** (on UHF) converges and is close (+1.45 mHa at 2.3 Å) but
  inherits spin contamination.

### 7.4 Link to DBF
CC builds e^T|Φ₀⟩ with commuting excitation operators. DBF builds a product
of unitary rotations U₁U₂…|0⟩. Both naturally generate products of
excitations (disconnected quadruples, hextuples) from lower-order pieces.
That is why in a basis where the exact state is "a product of pair
excitations" both work well (Section 12).

<details><summary>Check yourself</summary>

1. Why is CC size-extensive but CISD not? → e^{T_A + T_B} = e^{T_A} e^{T_B}
   factorizes for noninteracting fragments.
2. Why is CC not variational? → Energy is a projection with a non-Hermitian
   H̄, not an expectation value.
3. Why does RCCSD diverge for stretched N₂? → The closed-shell reference has
   tiny weight; the needed amplitudes are O(1), outside the regime where the
   projected equations behave.
</details>

---

## 8. Density matrices and natural orbitals (NO, UNO)

### 8.1 One-particle reduced density matrix (1-RDM)
γ_pq = ⟨Ψ|a†_p a_q|Ψ⟩. Hermitian, eigenvalues in [0, 1] per spin-orbital
(or [0, 2] for spatial orbitals with both spins). Trace = N.

### 8.2 Natural orbitals
Eigenvectors of γ; eigenvalues = **natural occupation numbers**.
- For a single determinant: occupations exactly 1 or 0. Any rotation of the
  occupied space is a valid set of NOs → **NOs of a determinant carry no
  information**.
- For a correlated state: fractional occupations. Near 2/0 = weakly
  correlated; pairs drifting to 1/1 = bond breaking (static correlation).
- **Löwdin's result**: NOs give the most rapidly converging CI expansion;
  the determinant of the highest-occupation NOs has (near-)maximal overlap
  with Ψ.

### 8.3 What we built
- **NO**: diagonalize the UCCSD (unrelaxed, from Λ) density *separately for
  α and β*. Unrestricted: α and β NOs differ. Reference = top 5 per spin.
  Its energy is above E_UHF (UHF is already the lowest determinant) but it
  is "closer" in shape to the correlated state. Still spin-contaminated.
- **UNO** (UHF natural orbitals, "charge-density NOs", Pulay–Hamilton):
  diagonalize the *spin-summed* UHF density S^{1/2}(P_α + P_β)S^{1/2}.
  - Each spin density alone is idempotent (0/1) and useless, but because
    α ≠ β the sum is not, and its eigenvalues lie in [0, 2].
  - One set of spatial orbitals for both spins → **restricted**.
  - Corresponding-orbital pairs with overlap d_i give UNO occupations
    1 ± d_i.
  - The doubly occupied closed-shell UNO determinant is an exact singlet.

**Worked check at 2.3 Å.** UNO occupations: 2, 2, 2, 2, **1.237**, **1.047**,
**1.047**, 0.953, 0.953, 0.763. So d_σ = 0.237, d_π = 0.047 (×2). Then
Σ(1 − d²) = (1 − 0.056) + 2(1 − 0.0022) = 2.940 = ⟨S²⟩_UHF (2.939) ✓.
The π bonds are almost fully broken in UHF (occupations ≈ 1/1).

### 8.4 Diagnostics of multireference character
Natural occupations far from 2/0, small reference weight in FCI, large
T₁/D₁ diagnostics in CC, large ⟨S²⟩ in UHF.

<details><summary>Check yourself</summary>

1. Why do NOs of a single determinant say nothing? → Its 1-RDM is the
   projector on the occupied space: all occupations 1, any basis of that
   space diagonalizes it.
2. Derive UNO occupations 1 ± d for one corresponding pair. → Spin-summed
   density of |φ_α⟩⟨φ_α| + |φ_β⟩⟨φ_β| with ⟨φ_α|φ_β⟩ = d: in the 2D span
   the matrix has eigenvalues 1 ± d.
3. Why is the closed-shell UNO determinant much higher in energy than UHF?
   → It puts both electrons of each broken bond into one delocalized
   bonding orbital: RHF-like, ionic-contaminated (+745 mHa at 2.3 Å vs RHF
   +462).
</details>

---

## 9. Localized orbitals (LMO, LNO)

- **Localization** = rotation within the occupied block (and separately
  within the virtual block) to make orbitals spatially compact.
  - Boys: maximize Σ_i |⟨i|r|i⟩|² (spread of centroids).
  - Pipek–Mezey: maximize atomic Mulliken populations.
- Determinant and energy unchanged (Section 2.3), but integrals change:
  local orbitals → many small, short-range interactions.
- **What we built**: LMO = UHF orbitals with valence-occupied and virtual
  blocks Boys-localized (core kept canonical). LNO = same for the UCCSD NOs
  (top-5 block and bottom-3 block separately). LUNO = same for UNO.
- **Why localize for vDBF**: the hope is that Pauli coefficients decay
  faster, so coefficient truncation drops more with less error.
- **When it backfires**: for restricted orbitals at stretched N₂,
  localization mixes the bonding σ/π orbitals with the lone pairs into
  atom-centred hybrids. The exact state, which in UNO is a product of three
  pair excitations, becomes spread over many determinants (99% of FCI
  weight: 64 determinants in UNO vs 448 in LUNO), symmetry no longer zeroes
  integrals (1177 → 4397 Pauli terms), and the reachable sector grows from
  4096 to 16384 states. The exact flow from the same singlet start reached
  FCI to 0.2 µHa in UNO but was still 7.6 mHa off in LUNO after 200
  iterations.

<details><summary>Check yourself</summary>

1. Why does localization not change ⟨ref|H|ref⟩? → It is an
   occupied–occupied rotation; the determinant is invariant.
2. Give one reason localization helps and one reason it hurts DBF. → Helps:
   faster coefficient decay with distance. Hurts: destroys symmetry-adapted
   pairing structure, spreading the target state over many directions.
</details>

---

## 10. Practical details: frozen core, sectors, symmetry

- **Frozen core**: the N 1s orbitals (ε ≈ −15.3 Ha) are kept doubly occupied
  and folded into an effective one-body term and a constant:
  h_eff = h + J[core] − K[core], E_core = ½ tr_core(h + h_eff).
  14 electrons/10 orbitals → 10 electrons/8 orbitals, 20 → 16 qubits.
  Different bases freeze slightly different 1s orbitals, so frozen-core FCI
  differs by a few µHa between "core groups"; always compare against the
  FCI of the same group.
- **Symmetry sectors**: H conserves N_α, N_β (particle number and S_z) and
  point-group symmetry. Starting from one determinant, the flow can only
  reach states connected by H's x-strings: the GF(2) span of those strings
  (the "parity sector"). Symmetry-adapted orbitals (UNO, RHF) make this
  sector smaller (UNO: 4096 states, and the lowest triplet is not even in
  it); localized/broken-symmetry orbitals make it larger (16384).

---

## 11. Double-bracket flow (DBF) and vDBF

### 11.1 Continuous double-bracket flows
Brockett / Wegner / Głazek–Wilson flow: dH/dℓ = [η(ℓ), H(ℓ)] with
η = [D, H], D a fixed "diagonal" operator. H(ℓ) flows toward a form that
commutes with D (block-diagonal). Each step is a unitary rotation of H; the
spectrum never changes.

### 11.2 Ground-state DBF in DBF.jl (Heisenberg picture)
Goal: rotate H so that a chosen basis state |ψ⟩ (our reference determinant)
becomes its ground state.

- **Heisenberg picture**: never store a state vector. Store the operator
  O_K = U_K† … U₁† H U₁ … U_K as a Pauli sum. The energy of the prepared
  state U₁…U_K|ψ⟩ is ⟨ψ|O_K|ψ⟩.
- **Vacuum frame**: apply X on the occupied qubits so the reference is
  |0…0⟩. Then ⟨x|O|0⟩ = "coupling from the reference to determinant x".
- **Generators** are single Pauli strings G; rotation U = exp(−iθG/2). For
  a Pauli G, conjugation is cheap: terms commuting with G are untouched,
  anticommuting terms P → cos θ P + (phase) sin θ GP. One rotation at most
  doubles the number of terms.
- **Which generators**: from the commutator [D, O] with a "source" operator
  D built from Z's (DBF.jl: n-body Z-projector approximation; our analysis
  code compares the one-body pool P = I + Σ Z_i, which weights direction x
  by |x|·O_x0², and the exact projector |0⟩⟨0|, which weights by O_x0²).
  The energy gradient of a rotation along x is ∝ O_x0 (the coupling to
  determinant x). Directions are ranked by gradient; the top ones (we used
  50 per macro-iteration) are applied, each at its **exactly optimal angle**
  (the energy along one Pauli rotation is a sinusoid in θ, minimized in
  closed form / by Brent).
- **Convergence**: all O_x0 → 0 means |0⟩ is an eigenvector of O. This is a
  *generalized Brillouin condition* (HF only zeroes singles; DBF zeroes
  couplings to every determinant). Equivalently the **variance**
  ⟨0|O²|0⟩ − ⟨0|O|0⟩² → 0.

### 11.3 vDBF: truncation and its bookkeeping
- Without truncation the Pauli sum grows exponentially. **Coefficient
  truncation** (CoeffTruncation(τ)) drops terms with |c| < τ after every
  rotation. τ = 1e-5 or 1e-6 in our runs (10⁶–10⁷ terms at 1e-6).
- The dropped pieces change ⟨0|O|0⟩ and the variance. The code accumulates
  an **energy error** and **variance error** estimate per rotation; we
  report "E − err" and "Var − var err" (corrected values).
- **Replay** (our check): take the stored circuit (generators + angles),
  apply it exactly to a state vector, and compute the *true* ⟨H⟩. The
  truncated operator's energy is not variational; the replay is. Example at
  2.3 Å, LNO, τ = 1e-6, UHF start: reported E − err = +0.59 mHa, true
  ⟨H⟩ = +2.69 mHa (⟨S²⟩ = 0.84).
- **Energy–variance extrapolation**: for a state close to an eigenstate,
  E − E_exact ≈ a·Var (+ b·Var²). Fit the tail of the (Var, E) trajectory
  and read off E₀ at Var = 0 (adaptive tail selection). It works only when
  the tail is in the linear regime and the state is converging to the
  right eigenstate. A spin-contaminated tail can extrapolate *below* FCI
  (−0.94 mHa for the exact UHF-start flow at 2.3 Å).
- **PT2**: second-order perturbative correction from the remaining
  couplings O_x0 (like Epstein–Nesbet), optionally computed each iteration.

### 11.4 Symmetry in DBF
- The exact imaginary-time-like flow would preserve S² if the start were a
  singlet, but DBF's steps are **single Pauli rotations**, each adding one
  determinant. Open-shell singlets need two determinants with equal
  amplitude, so individual rotations generate triplet admixture. Observed:
  the bare UNO start (S² = 0 exactly) reaches ⟨S²⟩ = 1.9 after one
  iteration and 2.4 after five.
- Pure *pair* excitations (both electrons of a spatial orbital move
  together, α and β in the same orbitals) lead to closed-shell determinants,
  which are singlets: rotations into them keep S² = 0.
- Single Pauli rotations can also leak out of the fixed-particle-number
  sector ("leak" diagnostic), because a Pauli string with a given x-string
  also acts on other basis states.

### 11.5 Ways to fix the reference/spin problem in vDBF
1. **S² penalty**: run on H + λS². Cheap (S² has ~345 Pauli terms), keeps the
   singlet ground state, punishes spin-breaking directions.
2. **Singlet-projected start**: build P₀|UHF⟩ exactly, then a greedy
   Pauli-rotation **prep circuit** that maps |0⟩ to it (each rotation moves
   the largest remaining component into |0⟩), and **fold** the circuit into
   H (O₀ = P_M† … P₁† H P₁ … P_M) before the flow. Cost: 2219 rotations for
   1 − F = 10⁻⁵ (LNO), folded H with 9.4M terms at cutoff 1e-6, fold error
   +0.21 mHa.
3. **Restricted reference** (UNO or RHF closed shell): exact singlet with
   no circuit, plus the S² penalty to stop the flow re-breaking symmetry.

<details><summary>Check yourself</summary>

1. Why work in the Heisenberg picture? → The state on 16+ qubits is never
   stored; cost is set by the number of Pauli terms in O, which truncation
   controls.
2. Why is the energy along one Pauli rotation a sinusoid? → G² = 1, so
   U = cos(θ/2) − i sin(θ/2)G; ⟨U†OU⟩ = A + B cos θ + C sin θ.
3. Why can the reported E − err be below the true energy? → The error
   estimate is first-order and the truncated operator is no longer unitarily
   equivalent to H; only the replayed circuit energy is variational.
4. What does variance → 0 mean? → The reference is an eigenstate of the
   rotated operator (not necessarily the ground state!).
</details>

---

## 12. Why the reference matters so much for vDBF (the core insight)

### 12.1 Energy distance is not difficulty
The UNO start is 745 mHa above FCI, the UHF start only 6.4 mHa (LMO) or
17.2 mHa (LNO). Yet the exact flow with the S² penalty converges far faster
from UNO. What matters is (a) how the exact state *looks* from the
reference, and (b) how strongly H couples the reference to what is missing.

### 12.2 FCI seen from each reference (2.3 Å)

| | UHF start (LMO) | UNO closed-shell start |
|---|---|---|
| E − E_FCI at start | +6.4 mHa | +745 mHa |
| overlap with FCI | 0.254 | 0.129 |
| FCI weight by excitation rank 0 / 2 / 4 / 6 | 0.25 / 0.25 / 0.24 / **0.21** (+0.05 odd) | 0.13 / **0.52** / 0.32 / 0.03 |
| second-largest FCI determinant | **6-fold** excitation | **pair (2-fold)** excitation |
| determinants for 99.9% of FCI | 192 | 87 |
| Pauli terms / reachable sector | 2485 / 16384 | 1177 / 4096 |

### 12.3 UNO: high but on a steep straight slope
In UNO orbitals the exact state is essentially a product of three two-level
bond problems, Π_bonds (c₁|bond²⟩ − c₂|antibond²⟩) (generalized
valence-bond / perfect-pairing structure, the H₂ story three times). Each
bond needs one pair-excitation rotation with a large coupling and a large
angle; DBF takes the exact optimal angle in one step; rotations on
different bonds commute, so 4- and 6-fold products appear for free (like
T₂² in CC). Energy: 745 → 98 → 7 → 1.3 mHa after iterations 1, 5, 10 (with
penalty). Pair excitations are closed-shell → singlets → no spin damage.
The penalty only has to police the small open-shell part.

### 12.4 UHF: low but in the wrong place
UHF got its energy by breaking spin symmetry. The singlet needs the
reference **plus its spin-flipped partner** (α ↔ β swapped between atoms)
with almost equal weight (0.25 vs 0.21). That partner is a 6-fold
excitation, coupled to the reference only weakly (exchange-type couplings
between stretched atoms) and, by Slater–Condon, only through intermediate
steps. The flow must build it while removing the 75% of weight in S = 1,
2, 3, which costs only 3–20 mHa in H: slow, and without a penalty it
stalls.

### 12.5 Exact (untruncated) flows at 2.3 Å, 200 iterations

| Start | iter 20 | iter 200 | E₀ (tail fit) |
|---|---|---|---|
| LNO UHF, H | +2.55 | +0.89 | −0.94 |
| LNO UHF, H + 0.05 S² | +1.36 | +0.016 | +0.000 |
| LNO singlet-projected UHF, H | +0.13 | +0.001 | −0.001 |
| UNO closed shell, H | +3.40 | +0.52 | −0.20 |
| **UNO closed shell, H + 0.05 S²** | **+0.33** | **+0.0002** | **+0.000** |
| LUNO closed shell, H | +81.7 | +10.6 | +3.1 |
| LUNO closed shell, H + 0.05 S² | +89.1 | +7.6 | +0.004 |

(mHa above the FCI of the same frozen-core group. λ = 0.112 gives the same
as 0.05 for UNO.)

Take-home: **restricted, delocalized, symmetry-adapted orbitals + S² penalty**
is the best reference so far. The singlet-projected UHF start is equally good
in the exact flow (its overlap with FCI is already 0.978) but needs a deep
prep circuit. Whether UNO keeps its advantage under truncation is the open
question: its early large-angle rotations will grow the operator fast.

### 12.6 Predictions to test next
- Canonical **RHF** orbitals ≈ UNO in STO-3G (π orbitals fixed by symmetry;
  only σ_g/σ_u 2×2 blocks can differ), lower start energy (+462 vs +745
  mHa). Expect UNO-like behaviour.
- **RHF-based LMO**: expect LUNO-like failure (localization).
- **RHF-based LNO**: no reliable parent (RCCSD diverges at 2.3 Å).

---

## 13. The N₂ project story (for an oral exam)

1. **Goal**: N₂ PES (STO-3G) with vDBF vs RHF/UHF/RCCSD(T)/UCCSD(T)/FCI.
   Bases tried: MO, LMO, NO, LNO from UHF / UCCSD.
2. **Problem**: at stretched geometries vDBF trajectories stall a few mHa
   above FCI and extrapolations are unreliable.
3. **Ruled out**: the generator pool (one-body Z pool vs exact projector
   behave the same); truncation alone (the exact, untruncated flow from UHF
   also stalls: +0.9 mHa after 200 iterations).
4. **Diagnosis**: the UHF reference is a mixture of S = 0, 1, 2, 3
   (1/4, 9/20, 1/4, 1/20) and these states are within 3–20 mHa of each
   other at 2.3 Å. The flow lowers ⟨H⟩ but has almost no energetic reason
   to remove the spin contamination; the stall energy (~4.5 mHa) matches
   the weighted average of the spin ladder.
5. **Fix 1, S² penalty**: H + λS². Exact flow reaches +0.016 mHa; truncated
   (τ = 1e-6) LNO true energy +0.59 mHa vs +2.69 for the bare run.
6. **Fix 2, singlet-projected start**: exact flow reaches FCI in ~25
   iterations, but the prep circuit is deep (thousands of rotations) and
   folding it into H adds millions of terms and a fold error.
7. **Fix 3, UNO closed-shell reference** (PI's suggestion): an exact singlet
   with no circuit. Alone it re-breaks spin; with the penalty it is the best
   exact result (0.2 µHa) and has the sparsest H. Localizing it destroys the
   advantage.
8. **Conclusion so far**: at bond breaking, vDBF needs (a) a reference with
   the right spin, (b) a basis in which the target state is compact (pair
   structure), and (c) a tight enough threshold, because truncation error
   and spin leakage compound.

---

## 14. Exam question bank

Conceptual
1. Why does RHF fail for bond dissociation while UHF gets the energy right?
   What does UHF get wrong instead?
2. Define the Coulson–Fischer point. What happens to ⟨S²⟩ past it?
3. Explain static vs dynamic correlation with H₂ and N₂ examples.
4. State and prove Brillouin's theorem. What is its DBF analogue?
5. Why is CC size-extensive and CISD not? Why is CC not variational?
6. Why does RCCSD fail for stretched N₂ while UCCSD converges?
7. What are natural orbitals, and why do NOs of a single determinant carry
   no information? How do UNOs get around this?
8. Show that UNO occupations are 1 ± d for corresponding orbitals and relate
   them to ⟨S²⟩_UHF.
9. What does orbital localization change and not change? When can it hurt?
10. Explain the Jordan–Wigner mapping and what an x-string means for a
    Hamiltonian term.

vDBF-specific
11. Describe one macro-iteration of ground-state DBF: what is computed, how
    generators are chosen, how angles are set.
12. What is the Heisenberg picture buying you, and what does truncation
    cost?
13. Why can the reported (truncated) energy fall below FCI? How do you
    check the true energy?
14. Explain energy–variance extrapolation and when it fails.
15. Why does a singlet start not stay a singlet under single Pauli
    rotations? Which rotations are spin-safe?
16. Why does the UNO start (745 mHa above FCI) converge faster than the UHF
    start (6 mHa above)?
17. Compare three ways to fix spin contamination in vDBF (penalty,
    projected start + prep circuit, restricted reference): cost, accuracy,
    drawbacks.
18. How would you choose λ for the S² penalty?
19. Why does localizing the UNOs make the flow worse? Use the numbers
    (determinant count, Pauli terms, sector size).
20. Predict what happens with a canonical RHF reference. Justify with
    symmetry in a minimal basis.

Derivations to practice
- HF energy expression and the Fock operator from the variational
  principle.
- ⟨S²⟩ of a UHF determinant.
- Clebsch–Gordan decomposition of 3/2 ⊗ 3/2 (state counting).
- Löwdin projector for the singlet out of S = 0..3.
- Energy along a single Pauli rotation: E(θ) = A + B cos θ + C sin θ and its
  minimum.
- Minimal-basis H₂ FCI (2×2 problem) and the dissociation limit.

---

## 15. Glossary

| Term | Meaning |
|---|---|
| STO-3G | minimal Gaussian basis (one contracted function per atomic orbital) |
| Determinant | antisymmetrized product of spin-orbitals; a bit string |
| FCI | exact solution within the basis (all determinants) |
| RHF / UHF | HF with shared / separate spatial orbitals for α and β |
| Canonical MO | Fock eigenvector |
| Coulson–Fischer point | bond length where UHF breaks spin symmetry below RHF |
| Spin contamination | admixture of other S into a state meant to be one S |
| ⟨S²⟩ | S(S+1) for a pure state; 0 singlet, 2 triplet, 6 quintet, 12 septet |
| PUHF | spin-projected UHF (Löwdin projector) |
| CSF | spin-adapted linear combination of determinants |
| CCSD(T) | coupled cluster singles + doubles with perturbative triples |
| 1-RDM | one-particle reduced density matrix γ_pq = ⟨a†_p a_q⟩ |
| NO | eigenvector of the 1-RDM |
| UNO | eigenvector of the spin-summed UHF density (restricted) |
| Corresponding orbitals | α/β pairs from SVD of their overlap |
| LMO / LNO / LUNO | Boys-localized MO / NO / UNO (within occ and virt blocks) |
| Frozen core | 1s orbitals kept doubly occupied and folded into constants |
| Jordan–Wigner | fermion → qubit mapping with Z strings |
| x-string | which qubits a Pauli term flips |
| Vacuum frame | reference determinant mapped to |0…0⟩ |
| DBF | double-bracket flow; unitary rotations that drive H toward a target form |
| vDBF | DBF with operator truncation (Pauli propagation) |
| Generator | Pauli string G in a rotation exp(−iθG/2) |
| Fold | conjugate H by a fixed circuit before the flow |
| Replay | exact state-vector evaluation of a stored circuit |
| E₀ | energy extrapolated to zero variance |
| Penalty λ | coefficient of S² added to H |
</content>
</invoke>
