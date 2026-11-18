"""N2 / STO-3G frozen-core bond breaking: does magic follow the individual bonds?

    python plot_bond_magic.py [out_stem]     # from frozen_core/; after bond_magic.jl

Top: M_2 of the exact ground state in its natural orbitals, and what independent
bonds would give.  A bond whose antibonding NO holds n electrons is, on its own, the
pair sqrt(1 - n/2)|bond^2> - sqrt(n/2)|anti^2>, a Clifford encoding of one qubit with
Bloch vector (x, 0, z), z = 1 - n, x = sqrt(n(2 - n)); its magic is
-ln((1 + x^4 + z^4) / 2), largest (ln 4/3) at n = 1 - 1/sqrt(2) = 0.29 and zero both
at equilibrium and when the bond is fully broken (n = 1).  The model sums the sigma
and the two pi pairs.
Middle: the configuration entropy S in the same natural orbitals, with the same
independent-bond model: a lone pair contributes -w ln w - (1 - w) ln(1 - w), w = n/2,
which keeps growing to ln 2 when the bond is fully broken.
Bottom: antibonding NO occupations, sigma* and pi* (pi_x and pi_y are degenerate and
break together); dotted lines where each bond is half broken (n = 0.5).
Lines: exact ground state (the model uses its occupations).  Markers: the final vDBF
states at cutoff 1e-5 (open) and 1e-6 (filled), replayed from their rotations; their
M_2 and S are taken in the exact state's natural orbitals, their occupations are their own
(bond_magic.jl)."""
import json
import math
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
out = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "uno_bond_magic")

rows = sorted(json.load(open(os.path.join(HERE, "results", "bond_magic_uno.json"))), key=lambda r: r["R"])
R = np.array([r["R"] for r in rows])
m2 = np.array([r["M2_NO"] for r in rows])
sc = np.array([r["S_NO"] for r in rows])
n_sig = np.array([r["occ_sigma"][0] for r in rows])
n_pi = np.array([r["occ_pix"][0] for r in rows])
CUT = {"1e-5": ("D", "white", 4, "vDBF, $10^{-5}$"), "1e-6": ("o", None, 4.5, "vDBF, $10^{-6}$")}
vd = {c: {"M2": np.array([r[c]["M2_NO"] for r in rows]), "S": np.array([r[c]["S_NO"] for r in rows]), "sig": np.array([r[c]["occ_sigma"][0] for r in rows]),
          "pi": np.array([r[c]["occ_pix"][0] for r in rows])} for c in CUT}


def markers(ax, y, color, cut):
    m, mfc, ms, _ = CUT[cut]
    ax.plot(R, y, m, color=color, mfc=mfc or color, mec=color, mew=1, ms=ms, zorder=4)


def pair_m2(n):
    z, x = 1 - n, np.sqrt(n * (2 - n))
    return -np.log((1 + x**4 + z**4) / 2)


def pair_s(n):
    w = n / 2
    return -w * np.log(w) - (1 - w) * np.log(1 - w)


def half_broken(n):
    return float(np.interp(0.5, n, R))         # n rises monotonically with R


m_sig, m_pi = pair_m2(n_sig), 2 * pair_m2(n_pi)
s_sig, s_pi = pair_s(n_sig), 2 * pair_s(n_pi)
r_pi, r_sig = half_broken(n_pi), half_broken(n_sig)
C_SIG, C_PI, C_M2, C_S = "C0", "C1", "C3", "C4"

plt.rcParams.update({"font.family": ["Arial", "Helvetica", "DejaVu Sans"], "font.size": 9,
                     "mathtext.fontset": "custom", "mathtext.rm": "Arial", "mathtext.it": "Arial:italic",
                     "axes.linewidth": 1.2, "xtick.major.width": 1.2, "ytick.major.width": 1.2,
                     "xtick.major.size": 5, "ytick.major.size": 5, "legend.fontsize": 7.5})
fig, (at, am, ab) = plt.subplots(3, 1, figsize=(4.2, 7.8), sharex=True)

at.fill_between(R, 0, m_pi, color=C_PI, alpha=0.25, lw=0, label=r"$\pi_x + \pi_y$ pairs")
at.fill_between(R, m_pi, m_pi + m_sig, color=C_SIG, alpha=0.25, lw=0, label=r"$\sigma$ pair")
at.plot(R, m_pi + m_sig, "--", color="0.3", lw=1.4, label="independent bonds")
at.plot(R, m2, "-", color=C_M2, lw=2)
for c in CUT:
    markers(at, vd[c]["M2"], C_M2, c)
ip = int(np.argmax(m2))
at.annotate(f"peak {R[ip]:.1f} Å", (R[ip], m2[ip]), xytext=(6, -18), textcoords="offset points", color=C_M2, fontsize=8)
at.set_ylabel("Magic $M_2$ (nats)")
at.set_ylim(0, 2.6)   # headroom for the legend
from matplotlib.lines import Line2D  # noqa: E402
h, l = at.get_legend_handles_labels()
h.append(Line2D([], [], color="0.3", lw=2))
l.append("exact ground state")
for c, (m, mfc, ms, lab) in CUT.items():
    h.append(Line2D([], [], ls="none", marker=m, color="0.3", mfc=mfc or "0.3", mec="0.3", ms=ms))
    l.append(lab)
at.legend(h, l, loc="upper left", ncol=2, fontsize=6.5, columnspacing=1.0, handlelength=1.6,
          borderpad=0.4, labelspacing=0.3, fancybox=True, framealpha=1, edgecolor="#cccccc")

am.fill_between(R, 0, s_pi, color=C_PI, alpha=0.25, lw=0)
am.fill_between(R, s_pi, s_pi + s_sig, color=C_SIG, alpha=0.25, lw=0)
am.plot(R, s_pi + s_sig, "--", color="0.3", lw=1.4)
am.plot(R, sc, "-", color=C_S, lw=2)
for c in CUT:
    markers(am, vd[c]["S"], C_S, c)
am.set_ylabel("Configuration entropy $S$ (nats)")
am.set_ylim(0, 3.9)

ab.axhline(0.5, color="0.6", lw=0.9, ls=":")
ab.plot(R, n_pi, "-", color=C_PI, lw=1.8, label=r"$\pi^*$ (each)")
ab.plot(R, n_sig, "-", color=C_SIG, lw=1.8, label=r"$\sigma^*$")
for c in CUT:
    markers(ab, vd[c]["pi"], C_PI, c)
    markers(ab, vd[c]["sig"], C_SIG, c)
ab.set_ylabel("antibonding NO occupation")
ab.set_xlabel("Bond length, $R$ (Å)")
ab.set_ylim(0, 1.05)
ab.legend(loc="lower right", bbox_to_anchor=(1, 0.2), fancybox=True, framealpha=1, edgecolor="#cccccc")

for a in (at, am, ab):
    for r, c in ((r_pi, C_PI), (r_sig, C_SIG)):
        a.axvline(r, color=c, lw=1, ls=":", zorder=1)
    a.grid(color="#e5e5e5", lw=0.8)
    a.set_axisbelow(True)
    a.set_xlim(0.7, 3.1)
ab.text(r_pi - 0.03, 0.98, f"$\\pi$ half broken\n{r_pi:.2f} Å", color=C_PI, fontsize=7.5, ha="right", va="top")
ab.text(r_sig + 0.03, 0.04, f"$\\sigma$ half broken\n{r_sig:.2f} Å", color=C_SIG, fontsize=7.5, ha="left", va="bottom")
at.set_title("N$_2$: magic, entropy and bond breaking", fontsize=10)
fig.tight_layout(h_pad=0.8)
for ext in ("png", "pdf"):
    fig.savefig(f"{out}.{ext}", dpi=200)
print(f"{out}.png, {out}.pdf   (half broken: pi {r_pi:.2f} A, sigma {r_sig:.2f} A)")
print(f"{'R':>5} {'M2':>7} {'model':>7} {'excess':>7} {'M2 1e-5':>8} {'M2 1e-6':>8} {'σ* 1e-5':>8} {'σ* 1e-6':>8}")
for i, r in enumerate(R):
    print(f"{r:5.2f} {m2[i]:7.3f} {m_pi[i] + m_sig[i]:7.3f} {m2[i] - m_pi[i] - m_sig[i]:7.3f}"
          f" {vd['1e-5']['M2'][i]:8.3f} {vd['1e-6']['M2'][i]:8.3f} {vd['1e-5']['sig'][i]:8.4f} {vd['1e-6']['sig'][i]:8.4f}")
