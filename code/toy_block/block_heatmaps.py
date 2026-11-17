"""Heatmaps of the double-bracket flow on a random 4-qubit Hamiltonian: the picture behind block DBF.

The flow rotates H by small exact unitary steps,

    H <- E H E^dag,   E = exp(ds W),   W = [H, D]       (dH/ds = [[H, D], H])

which lowers tr(H D): the lowest eigenvalues move onto the kets where D is largest, and every coupling
between kets with different D values goes to zero.  (D = diag(H), re-evaluated each step, uses
W = [diag H, H] instead: Wegner's flow, which diagonalises H completely.)  Each panel shows |H_ij|
on one colour scale, in the style of the qualifying-exam report's DBF heatmaps.

Figures (in results/):
  1. toy_generators.png  one random H and four choices of D, basis in Hamming-weight order:
                         D = diag H (diagonal) | |0><0| (row 0 decoupled) | sum Z_i (Hamming blocks)
                         | |0><0| + |a><a| + |b><b| (the three chosen kets decoupled)
  2. toy_block.png       D = projector on three chosen kets, two different choices: snapshots at
                         the start, at 50 % and 15 % of the initial block-rest coupling, at the end,
                         and after diagonalising the 3x3 block.  Rows and columns: chosen kets first.
  3. toy_symmetry.png    H that conserves Hamming weight (sectors 1 + 4 + 6 + 4 + 1).  The flow never
                         mixes sectors, so the block holds the lowest levels of the sectors its kets
                         are in.  Kets picked by lowest diagonal energy land in the wrong sectors
                         (fidelity with the 3 lowest levels F < 1); kets placed one per lowest level
                         in its sector give F = 1.

    python block_heatmaps.py          # from code/toy_block/
"""
import itertools
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

N = 4
DIM = 2 ** N
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results")
# the report's palette: azure ramp, charcoal lines, tangerine accent
CMAP = LinearSegmentedColormap.from_list("azure", [(0, "white"), (0.3, "#E0F1FF"), (0.65, "#2E9CF0"), (1, "#0B58A0")])
CHARCOAL, ACCENT, INK, MUTED = "#2B2F36", "#E8731A", "#15171c", "#6a6f7d"
plt.rcParams.update({"font.family": ["Helvetica", "Arial", "DejaVu Sans"], "font.size": 9})

P1 = {"I": np.eye(2), "X": np.array([[0, 1], [1, 0]]), "Y": np.array([[0, -1j], [1j, 0]]), "Z": np.diag([1.0, -1.0])}


def pauli(s):
    """matrix of a Pauli string; s[q] acts on qubit q = bit q of the basis index"""
    m = np.array([[1.0]])
    for ch in reversed(s):
        m = np.kron(m, P1[ch])
    return m


def ket(b):
    return r"$|" + format(b, f"0{N}b") + r"\rangle$"


def bits(S):
    return "[" + ", ".join(format(b, f"0{N}b") for b in S) + "]"


def hw(b):
    return bin(b).count("1")


HW_ORDER = sorted(range(DIM), key=lambda b: (hw(b), b))
HW_LINES = list(np.cumsum([len([b for b in range(DIM) if hw(b) == w]) for w in range(N)]))


def random_h(seed, n_terms=60):
    """sum of random Pauli strings with normal coefficients (Hermitian), scaled to max |H_ij| = 1"""
    rng = np.random.default_rng(seed)
    H = sum(rng.normal() * pauli("".join(rng.choice(list("IXYZ"), N))) for _ in range(n_terms))
    return H / np.abs(H).max()


def hamming_h(seed):
    """random hopping (XX + YY)/2, fields Z and couplings ZZ: conserves the Hamming weight"""
    rng = np.random.default_rng(seed)
    H = np.zeros((DIM, DIM), dtype=complex)
    for i, j in itertools.combinations(range(N), 2):
        s = ["I"] * N
        s[i] = s[j] = "X"
        xx = pauli("".join(s))
        s[i] = s[j] = "Y"
        yy = pauli("".join(s))
        s[i] = s[j] = "Z"
        H += rng.normal() * (xx + yy) / 2 + 0.5 * rng.normal() * pauli("".join(s))
    for i in range(N):
        s = ["I"] * N
        s[i] = "Z"
        H += 0.7 * rng.normal() * pauli("".join(s))
    return H / np.abs(H).max()


def unitary_step(W, ds):
    """exp(ds W) for anti-Hermitian W"""
    k, v = np.linalg.eigh(1j * W)
    return (v * np.exp(-1j * ds * k)) @ v.conj().T


def flow(H, D=None, coupling=None, ds=0.01, tol=2e-3, max_steps=200_000, fracs=(0.5, 0.15)):
    """run the flow until coupling(H) < tol * initial; snapshots [(label, H, U)] at the start, when the coupling
    first drops below each fraction, and at the end.  U is the accumulated rotation, H(s) = U H U^dag."""
    U = np.eye(DIM, dtype=complex)
    r0 = coupling(H)
    snaps, todo = [("start", H, U)], list(fracs)
    for step in range(1, max_steps + 1):
        if D is None:                                    # Wegner: D = diag(H) now, W = [diag H, H]
            Dm = np.diag(np.diag(H))
            W = Dm @ H - H @ Dm
        else:
            W = H @ D - D @ H
        E = unitary_step(W, ds)
        H = E @ H @ E.conj().T
        U = E @ U
        c = coupling(H)
        while todo and c <= todo[0] * r0:
            snaps.append((f"{round(100 * todo.pop(0))} % coupling", H, U))
        if c < tol * r0:
            break
    snaps.append(("end", H, U))
    return snaps, step


def off_block(sel):
    sel = np.asarray(sel)
    rest = np.setdiff1d(np.arange(DIM), sel)
    return lambda M: np.linalg.norm(M[np.ix_(sel, rest)])


def between_sectors(M):
    w = np.array([hw(b) for b in range(DIM)])
    return np.sqrt(sum(abs(M[i, j]) ** 2 for i in range(DIM) for j in range(DIM) if w[i] != w[j]))


def heat(ax, M, cmax, order, lines, marks=(), title=None, note=None, labels=False):
    """|M| in the given basis order, lines at the block boundaries, chosen kets marked on the edges"""
    A = np.abs(M)[np.ix_(order, order)]
    ax.imshow(A, cmap=CMAP, vmin=0, vmax=cmax, interpolation="nearest")
    for b in lines:
        ax.axhline(b - 0.5, color=CHARCOAL, lw=0.9, alpha=0.45)
        ax.axvline(b - 0.5, color=CHARCOAL, lw=0.9, alpha=0.45)
    for k, b in enumerate(order):
        if b in marks:
            ax.plot([-0.9], [k], marker=">", ms=4.5, color=ACCENT, clip_on=False)
            ax.plot([k], [-0.9], marker="v", ms=4.5, color=ACCENT, clip_on=False)
            if labels:
                ax.text(-1.6, k, ket(b), ha="right", va="center", fontsize=7, color=INK)
    ax.set_xlim(-0.5, DIM - 0.5)
    ax.set_ylim(DIM - 0.5, -0.5)
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_color(CHARCOAL)
        s.set_linewidth(1.0)
    if title:
        ax.set_title(title, fontsize=8.5, color=INK, pad=8)
    if note:
        ax.text(0.5, -0.05, note, transform=ax.transAxes, ha="center", va="top", fontsize=7.5, color=MUTED)


def projector(sel):
    D = np.zeros((DIM, DIM))
    for b in sel:
        D[b, b] = 1.0
    return D


def block_eigs(H, sel):
    return np.linalg.eigvalsh(H[np.ix_(sel, sel)])


def fidelity(U, sel, H0, k=3):
    """|P_low P_block|^2 / k: P_low the k lowest eigenvectors of H0, P_block = span U^dag |y>, y in sel"""
    _, v = np.linalg.eigh(H0)
    B = U.conj().T[:, sel]
    return np.sum(np.abs(v[:, :k].conj().T @ B) ** 2) / k


def fmt(e):
    return ", ".join(f"{x:.2f}" for x in e)


# ---------------------------------------------------------------- figure 1: four generators
def fig_generators(H0, S):
    cmax = np.abs(H0).max()
    runs = [("$D$ = diag $H$ (Wegner)", None, lambda M: np.linalg.norm(M - np.diag(np.diag(M))), "diagonal", ()),
            (r"$D = |0000\rangle\langle 0000|$", projector([0]), off_block([0]), "row and column 0 decoupled", (0,)),
            (r"$D = \Sigma_i\, Z_i$", sum(pauli("".join("Z" if q == i else "I" for q in range(N))).real for i in range(N)),
             between_sectors, "Hamming-weight blocks 1+4+6+4+1", ()),
            (r"$D = \Sigma_{y \in S}\, |y\rangle\langle y|$", projector(S), off_block(S),
             "S = " + ", ".join(ket(b) for b in S) + ": decoupled", tuple(S))]
    fig, axes = plt.subplots(1, 5, figsize=(12.5, 3.2))
    heat(axes[0], H0, cmax, HW_ORDER, HW_LINES, title="random H (4 qubits)", note="basis in Hamming-weight order")
    for ax, (name, D, cp, what, marks) in zip(axes[1:], runs):
        snaps, steps = flow(H0, D, cp, tol=1e-3)
        heat(ax, snaps[-1][1], cmax, HW_ORDER, HW_LINES, marks=marks, title=f"after the flow, {name}", note=what)
        print(f"  {name.replace('$', '')}: {steps} steps")
    fig.suptitle("The generator $D$ decides what the double-bracket flow decouples ($|H_{ij}|$, one colour scale; "
                 "lines at the Hamming-weight sectors)", x=0.01, ha="left", fontsize=10, color=INK)
    fig.subplots_adjust(left=0.01, right=0.99, bottom=0.1, top=0.8, wspace=0.12)
    return fig


# ---------------------------------------------------------------- figure 2: the block, two choices of kets
def fig_block(H0, choices):
    cmax = np.abs(H0).max()
    low = np.linalg.eigvalsh(H0)[:3]
    fig, axes = plt.subplots(len(choices), 5, figsize=(12.5, 3.25 * len(choices)), squeeze=False)
    for r, S in enumerate(choices):
        order = list(S) + [b for b in range(DIM) if b not in S]
        snaps, steps = flow(H0, projector(S), off_block(S), tol=1e-3)
        for c, (lab, M, U) in enumerate(snaps):
            note = None
            if lab == "end":
                note = f"block eigenvalues {fmt(block_eigs(M, S))}"
            heat(axes[r, c], M, cmax, order, [len(S)], marks=S, title=lab, note=note, labels=(c == 0))
        # last panel: diagonalise the 3x3 block (vDBF does this classically)
        M, U = snaps[-1][1], snaps[-1][2]
        e, v = np.linalg.eigh(M[np.ix_(S, S)])
        R = np.eye(DIM, dtype=complex)
        R[np.ix_(S, S)] = v.conj().T
        heat(axes[r, 4], R @ M @ R.conj().T, cmax, order, [len(S)], marks=S, title="3×3 block diagonalised",
             note=f"F = {fidelity(U, S, H0):.3f} with the 3 lowest levels")
        print(f"  block {bits(S)}: {steps} steps, block eigenvalues {fmt(e)}, F = {fidelity(U, S, H0):.4f}")
        axes[r, 0].text(-0.3, 1.2, f"{'first' if r == 0 else 'second'} choice: S = " + ", ".join(ket(b) for b in S),
                        transform=axes[r, 0].transAxes, ha="left", va="bottom", fontsize=9.5, color=INK, weight="bold")
    fig.suptitle(r"$D = \Sigma_{y \in S}\, |y\rangle\langle y|$, the projector on three chosen kets (marked): the flow "
                 "decouples exactly those kets, whichever they are."
                 f"\nRows and columns: the chosen kets first.  Both choices end with the 3 lowest eigenvalues of H "
                 f"({fmt(low)}) in the block.", x=0.01, ha="left", fontsize=10, color=INK)
    fig.subplots_adjust(left=0.07, right=0.99, bottom=0.06, top=0.8, wspace=0.15, hspace=0.42)
    return fig


# ---------------------------------------------------------------- figure 3: a symmetry, right and wrong kets
def sector_levels(H0, k=3):
    """(eigenvalue, Hamming weight) of the k lowest levels; H0 conserves the Hamming weight"""
    lv = []
    for w in range(N + 1):
        idx = [b for b in range(DIM) if hw(b) == w]
        lv += [(e, w) for e in np.linalg.eigvalsh(H0[np.ix_(idx, idx)])]
    return sorted(lv)[:k]


def kets_by_diagonal(H0, k=3):
    return sorted(sorted(range(DIM), key=lambda b: H0[b, b].real)[:k])


def kets_by_sector(H0, k=3):
    """per lowest level, the lowest-diagonal ket of its sector not used yet"""
    out = []
    for _, w in sector_levels(H0, k):
        out.append(min((b for b in range(DIM) if hw(b) == w and b not in out), key=lambda b: H0[b, b].real))
    return sorted(out)


def fig_symmetry(H0):
    cmax = np.abs(H0).max()
    lv = sector_levels(H0)
    rows = [("kets with the lowest diagonal energies", kets_by_diagonal(H0)),
            ("one ket per lowest level, in its sector", kets_by_sector(H0))]
    fig, axes = plt.subplots(2, 4, figsize=(10.4, 6.6), squeeze=False)
    for r, (name, S) in enumerate(rows):
        snaps, steps = flow(H0, projector(S), off_block(S), tol=1e-3)
        for c, (lab, M, U) in enumerate(snaps):
            note = None
            if lab == "end":
                F = fidelity(U, S, H0)
                note = f"block {fmt(block_eigs(M, S))};  F = {F:.2f}"
            heat(axes[r, c], M, cmax, HW_ORDER, HW_LINES, marks=S, title=lab, note=note, labels=(c == 0))
        sec = [hw(b) for b in S]
        axes[r, 0].text(-0.3, 1.2, f"{name}: S = " + ", ".join(ket(b) for b in S) + f"  (Hamming weights {sec})",
                        transform=axes[r, 0].transAxes, ha="left", va="bottom", fontsize=9.5, color=INK, weight="bold")
        print(f"  {name}: {bits(S)} (weights {sec}), {steps} steps, "
              f"block {fmt(block_eigs(snaps[-1][1], S))}, F = {fidelity(snaps[-1][2], S, H0):.4f}")
    fig.suptitle("H that conserves the Hamming weight: already block diagonal (sectors 1+4+6+4+1), and the flow never "
                 f"mixes sectors.\nThe 3 lowest levels are {fmt([e for e, _ in lv])} in sectors of weight "
                 f"{[w for _, w in lv]}; the block can only hold the lowest levels of the sectors its kets are in.",
                 x=0.01, ha="left", fontsize=10, color=INK)
    fig.subplots_adjust(left=0.09, right=0.99, bottom=0.05, top=0.83, wspace=0.12, hspace=0.42)
    return fig


def symmetry_seed(start=0):
    """first seed whose lowest-diagonal kets sit in the wrong sectors for the 3 lowest levels"""
    for seed in range(start, start + 500):
        H0 = hamming_h(seed)
        want = sorted(w for _, w in sector_levels(H0))
        if sorted(hw(b) for b in kets_by_diagonal(H0)) != want and len(set(want)) > 1:
            return seed, H0
    raise RuntimeError("no seed found")


def main():
    os.makedirs(OUT, exist_ok=True)
    H0 = random_h(12)
    S1 = [0b0000, 0b0011, 0b1100]                       # |0>, and one "excitation" in each half (like |a>, |b>)
    S2 = [0b0101, 0b1001, 0b1110]                       # any other three kets
    print("figure 1: four generators")
    fig_generators(H0, S1).savefig(os.path.join(OUT, "toy_generators.png"), dpi=170)
    print("figure 2: the block, two choices of kets; 3 lowest eigenvalues of H:", fmt(np.linalg.eigvalsh(H0)[:3]))
    fig_block(H0, [S1, S2]).savefig(os.path.join(OUT, "toy_block.png"), dpi=170)
    seed, Hs = symmetry_seed()
    print(f"figure 3: Hamming-weight-conserving H (seed {seed}); 3 lowest levels (energy, weight):",
          [(round(float(e), 3), w) for e, w in sector_levels(Hs)])
    fig_symmetry(Hs).savefig(os.path.join(OUT, "toy_symmetry.png"), dpi=170)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
