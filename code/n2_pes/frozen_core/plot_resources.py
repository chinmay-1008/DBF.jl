"""N2 / STO-3G frozen-core bond breaking: configuration entropy and magic of the
ground state, next to the PES.

    python plot_resources.py [out_stem]      # from frozen_core/; after resources.jl
    python plot_resources.py --raw           # S and M_2 in the UNOs as computed (jagged): uno_resources_raw

Top: S (Shannon entropy of the state's coefficients in the determinant basis)
and M_2 (stabilizer Renyi entropy) of the exact ground state (lines) and of the
final vDBF states, UNO start, H + 0.05 S^2, cutoff 1e-5 and 1e-6 (markers), all
in the UNOs with each degenerate pi pair turned to pure px / py (resources.jl);
the arbitrary pi mixing of the raw UNOs makes both curves jump from one R to the next.
Bottom: RHF, UHF and FCI, and the true <H> of the same vDBF states.  The grey line
is the Coulson-Fischer point, where UHF first drops below RHF."""
import csv
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
RAW = "--raw" in sys.argv
args = [a for a in sys.argv[1:] if a != "--raw"]
out = args[0] if args else os.path.join(HERE, "uno_resources_raw" if RAW else "uno_resources")

rows = sorted(json.load(open(os.path.join(HERE, "results", "resources_uno.json"))), key=lambda r: r["R"])
R = np.array([r["R"] for r in rows])
fci = np.array([r["E_FCI"] for r in rows])
get = lambda src, k: np.array([r[src][k + ("_uno" if RAW and k in ("S", "M2") else "")] for r in rows])
cc = {round(float(r["R"]), 2): r for r in csv.DictReader(open(os.path.join(HERE, "results", "fc_cc.csv")))}
rhf = np.array([float(cc[round(r, 2)]["E_RHF"]) for r in R])
uhf = np.array([float(cc[round(r, 2)]["E_UHF"]) for r in R])
r_cf = R[np.argmax(uhf < rhf - 1e-6)]

QTY = {"S": ("C4", "$S$, configuration entropy"), "M2": ("C3", "$M_2$, magic (stabilizer Rényi)")}
CUT = {"1e-5": ("D", "vDBF, $10^{-5}$"), "1e-6": ("o", "vDBF, $10^{-6}$")}

plt.rcParams.update({"font.family": ["Arial", "Helvetica", "DejaVu Sans"], "font.size": 9,
                     "mathtext.fontset": "custom", "mathtext.rm": "Arial", "mathtext.it": "Arial:italic",
                     "axes.linewidth": 1.2, "xtick.major.width": 1.2, "ytick.major.width": 1.2,
                     "xtick.major.size": 5, "ytick.major.size": 5, "legend.fontsize": 7.5})
fig, (at, ab) = plt.subplots(2, 1, figsize=(4.2, 5.6), sharex=True)

# resources
for k, (c, lab) in QTY.items():
    at.plot(R, get("fci", k), "-", color=c, lw=2, label=lab, zorder=2)
    for cut, (m, _) in CUT.items():
        at.plot(R, get(cut, k), m, color=c, ms=4.5 if m == "o" else 4, mfc="white" if cut == "1e-5" else c,
                mec=c, mew=1, zorder=3)
if not RAW:   # the raw curves jump with the pi mixing, so their maximum means nothing
    m2 = get("fci", "M2")
    ip = int(np.argmax(m2))
    at.plot(R[ip], m2[ip], "o", color=QTY["M2"][0], ms=8, mec="white", mew=1, zorder=4)
    at.annotate(f"peak  $R$ = {R[ip]:.1f} Å", (R[ip], m2[ip]), xytext=(6, 12), textcoords="offset points",
                color=QTY["M2"][0], fontsize=8)
at.set_ylabel("Quantum resource (nats)")
at.set_ylim(0, 5.3)   # headroom for the legend
from matplotlib.lines import Line2D  # noqa: E402
hq, lq = at.get_legend_handles_labels()
hq += [Line2D([], [], color="0.3", lw=2), Line2D([], [], ls="none", marker="D", mfc="white", mec="0.3", ms=4),
       Line2D([], [], ls="none", marker="o", color="0.3", ms=4.5)]
lq += ["exact ground state", CUT["1e-5"][1], CUT["1e-6"][1]]
at.legend(hq, lq, loc="upper left", fancybox=True, framealpha=1, edgecolor="#cccccc", fontsize=7)

# PES
ab.plot(R, rhf, "-", color="C0", lw=1.8, label="RHF")
ab.plot(R, uhf, "-", color="C1", lw=1.8, label="UHF")
ab.plot(R, fci, "-", color="C2", lw=2, label="FCI")
for cut, (m, lab) in CUT.items():
    ab.plot(R, get(cut, "E_true"), m, color="C2", ms=4.5 if m == "o" else 4, mfc="white" if cut == "1e-5" else "C2",
            mec="C2", mew=1, label=lab, zorder=3)
i = int(np.where(R == r_cf)[0][0])
ab.plot(R[i], uhf[i], "o", color="black", ms=6, zorder=4)
ab.annotate("Coulson–Fischer\npoint", (R[i], uhf[i]), xytext=(0.9, -107.12), textcoords="data", fontsize=8,
            arrowprops=dict(arrowstyle="-", color="0.4", lw=0.8))
ab.set_ylabel("Energy (Ha)")
ab.set_xlabel("Bond length, $R$ (Å)")
ab.legend(loc="upper center", ncol=2, fancybox=True, framealpha=1, edgecolor="#cccccc", fontsize=7)

for a in (at, ab):
    a.axvline(r_cf, color="0.55", lw=1.2, zorder=1)
    a.grid(color="#e5e5e5", lw=0.8)
    a.set_axisbelow(True)
    a.set_xlim(0.7, 3.1)
at.set_title("N$_2$: magic vs entropy in the raw UNOs" if RAW else "N$_2$ dissociation: magic vs configuration entropy",
             fontsize=10)
fig.tight_layout(h_pad=0.8)
for ext in ("png", "pdf"):
    fig.savefig(f"{out}.{ext}", dpi=200)
print(f"{out}.png, {out}.pdf   (Coulson-Fischer point {r_cf:.2f} A)")
print(f"{'R':>5} {'S fci':>7} {'S 1e-5':>7} {'S 1e-6':>7} {'M2 fci':>7} {'M2 1e-5':>7} {'M2 1e-6':>7}")
for j, r in enumerate(R):
    print(f"{r:5.2f} " + " ".join(f"{get(s, k)[j]:7.3f}" for k in ("S", "M2") for s in ("fci", "1e-5", "1e-6")))
