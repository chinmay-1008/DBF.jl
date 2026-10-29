"""Frozen-core N2: truncated dbf_groundstate vs exact state-vector DBF.

One row per (basis, R).  Left: energy vs iteration.  Middle: fidelity with the
exact ground state and <S^2> of the exact runs.  Right: energy vs variance,
the plane the paper extrapolates in.  Truncated DBF is shown as reported
(E - accumulated error vs Var - accumulated variance error); the exact runs
are exact.  Exact runs apply 50 distinct directions per iteration, while
dbf_groundstate's 50 generators per iteration are mostly copies of a few
excitations, so the iteration axes are not equal work."""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROWS = [("lno", "1.70", "data_nocore"), ("lno", "2.00", "data_nocore"),
        ("lno", "2.30", "data_nocore"), ("mo", "2.30", "data_uhfcore")]
EXACT = [("onebody", "exact DBF, one-body P (as dbf_groundstate)", "#2a78d6", "-"),
         ("projector", "exact DBF, projector |0⟩⟨0|", "#1baf7a", "--"),
         ("onebody_s2pen0.05", "exact DBF on H + 0.05 S²", "#eb6834", "-")]
TRUNC = "#6b6a64"
INK, MUTED, GRID = "#1f1f1e", "#6b6a64", "#e4e3de"
MHA = 1e3


def style(ax, ylog=True):
    if ylog:
        ax.set_yscale("symlog", linthresh=1.0, linscale=0.6)
        ax.axhline(0, color=INK, lw=0.9, ls=":")
    ax.grid(color=GRID, lw=0.5)
    ax.tick_params(colors=MUTED, labelsize=8)
    for s in ax.spines.values():
        s.set_color(GRID)


fig, axes = plt.subplots(len(ROWS), 3, figsize=(15, 3.6 * len(ROWS)))
for row, (basis, R, tree) in enumerate(ROWS):
    eax, fax, vax = axes[row]
    tpath = os.path.join(HERE, tree, f"R_{R}", f"dbf_{basis}.json")
    title = f"{basis.upper()} {float(R):.1f} Å (frozen core)"
    if os.path.exists(tpath):
        t = json.load(open(tpath))
        h, fci = t["history"], t["E_FCI"]
        E = np.array(h["energies_per_grad"]); err = np.array(h["accumulated_error_per_grad"])
        V = np.array(h["variance_per_grad"]); verr = np.array(h["accumulated_var_error_per_grad"])
        it = np.arange(len(E))
        eax.plot(it, (E - err - fci) * MHA, color=TRUNC, lw=1.4)
        eax.plot(it, (E - fci) * MHA, color=TRUNC, lw=0.8, ls=":")
        vax.plot((V - verr)[1:], (E - err - fci)[1:] * MHA, "s", color=TRUNC, ms=2.2, alpha=0.6)
    else:
        title += "  (truncated run not finished)"
    for key, _, colour, ls in EXACT:
        p = os.path.join(HERE, "sv_results", f"{basis}_{R}_{key}.json")
        if not os.path.exists(p):
            continue
        d = json.load(open(p))
        hh, fci = d["history"], d["E_FCI"]
        E = np.array(hh["E"]); it = np.arange(len(E))
        eax.plot(it, (E - fci) * MHA, color=colour, lw=1.6, ls=ls)
        fax.plot(it, hh["fidelity"], color=colour, lw=1.6, ls=ls)
        fax.plot(it, np.array(hh["S2"]) / 3, color=colour, lw=0.9, ls=":")
        vax.plot(np.array(hh["var"])[1:], (E - fci)[1:] * MHA, color=colour, lw=1.4, ls=ls)
    eax.set_title(title, color=INK, fontsize=10, loc="left")
    eax.set_xlabel("iteration", color=MUTED, fontsize=8)
    eax.set_ylabel("E − E_FCI, mHa (symlog)", color=MUTED, fontsize=8)
    style(eax)
    fax.set_ylim(0, 1.02)
    fax.set_xlabel("iteration", color=MUTED, fontsize=8)
    fax.set_ylabel("fidelity (solid), ⟨S²⟩/3 (dotted)", color=MUTED, fontsize=8)
    style(fax, ylog=False)
    vax.set_xscale("log")
    vax.set_xlabel("variance, Ha² (log)", color=MUTED, fontsize=8)
    vax.set_ylabel("E − E_FCI, mHa (symlog)", color=MUTED, fontsize=8)
    style(vax)

handles = [Line2D([], [], color=c, ls=ls, lw=1.6, label=lab) for _, lab, c, ls in EXACT]
handles += [Line2D([], [], color=TRUNC, lw=1.4, label="dbf_groundstate, coeff 1e-5: E − err (right: vs Var − var err)"),
            Line2D([], [], color=TRUNC, lw=0.8, ls=":", label="dbf_groundstate: raw ⟨0|H̃|0⟩"),
            Line2D([], [], color=INK, ls=":", lw=0.9, label="FCI")]
fig.legend(handles=handles, loc="upper center", ncol=3, frameon=False, fontsize=9, labelcolor=INK)
fig.tight_layout(rect=(0, 0, 1, 0.955))
out = os.path.join(HERE, "sv_vs_truncated.png")
fig.savefig(out, dpi=140)
print(out)
