"""N2 / STO-3G frozen-core bond breaking: magic of the sigma and of the pi orbitals.

    python plot_group_magic.py [out_stem]    # from frozen_core/; after bond_magic.jl

Top: mixed-state M_2 of the reduced state of each orbital group (sigma: 8 qubits;
pi_x + pi_y: 8 qubits), in the exact state's natural orbitals.  Lines: exact state;
markers: vDBF at 1e-5 (open) and 1e-6 (filled).  Dashed: what the group's own bonds
would give as independent pairs (one sigma pair; two pi pairs), the model of
plot_bond_magic.py.
Bottom: antibonding NO occupations of the exact state; dotted lines where each bond
is half broken."""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
out = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "uno_group_magic")

rows = sorted(json.load(open(os.path.join(HERE, "results", "bond_magic_uno.json"))), key=lambda r: r["R"])
R = np.array([r["R"] for r in rows])
n_sig = np.array([r["occ_sigma"][0] for r in rows])
n_pi = np.array([r["occ_pix"][0] for r in rows])
ex = {k: np.array([r[k] for r in rows]) for k in ("M2_sigma", "M2_pi", "purity_sigma", "purity_pi")}
CUT = {"1e-5": ("D", "white", 4, "vDBF, $10^{-5}$"), "1e-6": ("o", None, 4.5, "vDBF, $10^{-6}$")}
vd = {c: {k: np.array([r[c][k] for r in rows]) for k in ("M2_sigma", "M2_pi")} for c in CUT}


def pair_m2(n):
    z, x = 1 - n, np.sqrt(n * (2 - n))
    return -np.log((1 + x**4 + z**4) / 2)


r_pi, r_sig = (float(np.interp(0.5, n, R)) for n in (n_pi, n_sig))
C_SIG, C_PI = "C0", "C1"

plt.rcParams.update({"font.family": ["Arial", "Helvetica", "DejaVu Sans"], "font.size": 9,
                     "mathtext.fontset": "custom", "mathtext.rm": "Arial", "mathtext.it": "Arial:italic",
                     "axes.linewidth": 1.2, "xtick.major.width": 1.2, "ytick.major.width": 1.2,
                     "xtick.major.size": 5, "ytick.major.size": 5, "legend.fontsize": 7})
fig, (at, ab) = plt.subplots(2, 1, figsize=(4.2, 5.4), sharex=True)

for key, model, c, lab in (("M2_pi", 2 * pair_m2(n_pi), C_PI, r"$\pi$ orbitals"),
                           ("M2_sigma", pair_m2(n_sig), C_SIG, r"$\sigma$ orbitals")):
    at.plot(R, ex[key], "-", color=c, lw=2, label=lab)
    at.plot(R, model, "--", color=c, lw=1.2)
    for cut, (m, mfc, ms, _) in CUT.items():
        at.plot(R, vd[cut][key], m, color=c, mfc=mfc or c, mec=c, mew=1, ms=ms, zorder=4)
at.set_ylabel("group magic $M_2(\\rho_A)$ (nats)")
h, l = at.get_legend_handles_labels()
h += [Line2D([], [], color="0.3", lw=1.2, ls="--")] + [
    Line2D([], [], ls="none", marker=m, color="0.3", mfc=mfc or "0.3", mec="0.3", ms=ms) for m, mfc, ms, _ in CUT.values()]
l += ["independent bonds"] + [v[3] for v in CUT.values()]
at.legend(h, l, loc="upper left", ncol=2, fontsize=6.5, columnspacing=1.0, handlelength=1.6,
          borderpad=0.4, labelspacing=0.3, fancybox=True, framealpha=1, edgecolor="#cccccc")
top = max(ex["M2_sigma"].max(), ex["M2_pi"].max(), max(v[k].max() for v in vd.values() for k in v))
at.set_ylim(0, 1.45 * top)   # headroom for the legend

ab.axhline(0.5, color="0.6", lw=0.9, ls=":")
ab.plot(R, n_pi, "-", color=C_PI, lw=1.8, label=r"$\pi^*$ (each)")
ab.plot(R, n_sig, "-", color=C_SIG, lw=1.8, label=r"$\sigma^*$")
ab.set_ylabel("antibonding NO occupation")
ab.set_xlabel("Bond length, $R$ (Å)")
ab.set_ylim(0, 1.05)
ab.legend(loc="lower right", bbox_to_anchor=(1, 0.2), fancybox=True, framealpha=1, edgecolor="#cccccc", fontsize=7.5)
ab.text(r_pi - 0.03, 0.98, f"$\\pi$ half broken\n{r_pi:.2f} Å", color=C_PI, fontsize=7.5, ha="right", va="top")
ab.text(r_sig + 0.03, 0.04, f"$\\sigma$ half broken\n{r_sig:.2f} Å", color=C_SIG, fontsize=7.5, ha="left", va="bottom")

for a in (at, ab):
    for r, c in ((r_pi, C_PI), (r_sig, C_SIG)):
        a.axvline(r, color=c, lw=1, ls=":", zorder=1)
    a.grid(color="#e5e5e5", lw=0.8)
    a.set_axisbelow(True)
    a.set_xlim(0.7, 3.1)
at.set_title("N$_2$: magic of the $\\sigma$ and $\\pi$ orbitals", fontsize=10)
fig.tight_layout(h_pad=0.8)
for ext in ("png", "pdf"):
    fig.savefig(f"{out}.{ext}", dpi=200)
print(f"{out}.png, {out}.pdf")
for name, key in (("sigma", "M2_sigma"), ("pi", "M2_pi")):
    i = int(np.argmax(ex[key]))
    print(f"{name}: peak {ex[key][i]:.3f} nats at {R[i]:.2f} A")
print(f"{'R':>5} {'M2 σ':>7} {'model σ':>8} {'pur σ':>7} {'M2 π':>7} {'model π':>8} {'pur π':>7}")
for i, r in enumerate(R):
    print(f"{r:5.2f} {ex['M2_sigma'][i]:7.3f} {pair_m2(n_sig[i]):8.3f} {ex['purity_sigma'][i]:7.3f}"
          f" {ex['M2_pi'][i]:7.3f} {2 * pair_m2(n_pi[i]):8.3f} {ex['purity_pi'][i]:7.3f}")
