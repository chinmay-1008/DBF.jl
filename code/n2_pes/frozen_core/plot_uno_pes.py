"""N2 / STO-3G frozen-core PES: UHF, UCCSD, UCCSD(T), vDBF (UNO start, H + 0.05 S^2,
variance-extrapolated E0, at coeff 1e-5 and 1e-6) and FCI.  Left: the PES; right: error vs FCI.

    python plot_uno_pes.py [out_stem]        # from frozen_core/; writes <out_stem>.png and .pdf

UHF and CC from results/fc_cc.csv (frozen canonical 1s core); FCI and vDBF from
data_unocore (frozen UNO 1s core).  The two frozen-core FCIs differ by a few
uHa, far below what the figure resolves."""
import csv
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
from extrapolate import MIN_FIT_POINTS, select_adaptive_tail  # noqa: E402

STEMS = {"1e-5": "dbf_uno_s2pen0.05", "1e-6": "dbf_uno_s2pen0.05_c1e-6"}   # coeff -> run stem
out = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "uno_pes_c1e-5_c1e-6")
INK, MUTED, GRID, BAND = "#1f1f1e", "#6b6a64", "#e4e3de", "#efeee9"

# geometries with a finished run at every cutoff
dirs = [d for d in sorted(os.listdir(os.path.join(HERE, "data_unocore"))) if d.startswith("R_")
        and all(os.path.exists(os.path.join(HERE, "data_unocore", d, f"{s}.json")) for s in STEMS.values())]
R, fci = [], []
e0 = {c: [] for c in STEMS}
e0u = {c: [] for c in STEMS}
for d in dirs:
    meta = json.load(open(os.path.join(HERE, "data_unocore", d, "metadata.json")))
    R.append(meta["R_angstrom"]); fci.append(meta["constant"] + meta["E_FCI_with_mu"])
    for c, stem in STEMS.items():
        h = json.load(open(os.path.join(HERE, "data_unocore", d, f"{stem}.json")))["history"]
        E = (np.array(h["energies_per_grad"]) - np.array(h["accumulated_error_per_grad"]))[1:]
        V = (np.array(h["variance_per_grad"]) - np.array(h["accumulated_var_error_per_grad"]))[1:]
        best, _ = select_adaptive_tail(V, E, MIN_FIT_POINTS)
        e0[c].append(best["E0_average"]); e0u[c].append(best["fit_half_difference"])
R, fci = np.array(R), np.array(fci)
e0 = {c: np.array(v) for c, v in e0.items()}
e0u = {c: np.array(v) for c, v in e0u.items()}
cc = {round(float(r["R"]), 2): r for r in csv.DictReader(open(os.path.join(HERE, "results", "fc_cc.csv")))}
col = lambda k: np.array([float(cc[round(r, 2)][k]) for r in R])

PAL = ["C0", "C1", "C2", "C3", "C4"]   # matplotlib's default colour cycle: blue, orange, green, red, purple
methods = [  # label, energies, colour, line style, marker
    ("UHF", col("E_UHF"), PAL[0], "-", "o"),
    ("UCCSD", col("E_UCCSD"), PAL[1], "-", "s"),
    ("UCCSD(T)", col("E_UCCSD_T"), PAL[2], "--", "^"),
]
VDBF = {"1e-5": ("vDBF, $10^{-5}$", PAL[4], "D"), "1e-6": ("vDBF, $10^{-6}$", PAL[3], "o")}

plt.rcParams.update({"font.family": ["Arial", "Helvetica", "DejaVu Sans"], "font.size": 9,
                     "mathtext.fontset": "custom", "mathtext.rm": "Arial", "mathtext.it": "Arial:italic",
                     "axes.linewidth": 1.2, "xtick.major.width": 1.2, "ytick.major.width": 1.2,
                     "xtick.major.size": 5, "ytick.major.size": 5, "legend.fontsize": 8})
fig, (ap, ae) = plt.subplots(1, 2, figsize=(6.5, 2.8))

# PES
ap.plot(R, fci, "-", color="black", lw=2.2, label="FCI", zorder=2)
for lab, E, c, ls, m in methods:
    ap.plot(R, E, ls=ls, marker=m, color=c, lw=1.6, ms=4, label=lab, zorder=3)
for c, (lab, colr, m) in VDBF.items():
    ap.plot(R, e0[c], m, color=colr, ms=5.5 if m == "o" else 4.5, mec="white", mew=0.7, label=lab, zorder=5)
ap.set_ylabel("Energy (Ha)")

# error vs FCI
ae.axhspan(-1.6, 1.6, color="#ececec", lw=0, zorder=0, label="chemical accuracy")
ae.axhline(0, color="black", lw=0.8, zorder=1)
for lab, E, c, ls, m in methods:
    ae.plot(R, (E - fci) * 1e3, ls=ls, marker=m, color=c, lw=1.6, ms=4, label=lab, zorder=3)
for c, (lab, colr, m) in VDBF.items():
    ae.errorbar(R, (e0[c] - fci) * 1e3, yerr=e0u[c] * 1e3, fmt=f"{m}-", color=colr, lw=1.6, ms=5 if m == "o" else 4,
                mec="white", mew=0.7, capsize=2, label=lab, zorder=5)
ae.set_yscale("symlog", linthresh=0.1, linscale=0.7)
ae.set_ylim(-10, 500)
ae.set_yticks([-10, -1, -0.1, 0, 0.1, 1, 10, 100])
ae.set_yticklabels(["\u221210", "\u22121", "\u22120.1", "0", "0.1", "1", "10", "100"])
ae.set_ylabel("$E - E_{\\mathrm{FCI}}$ (mHa)")
# ae.legend(handles=[ae.get_legend_handles_labels()[0][0]], loc="lower left", fancybox=True, framealpha=1, edgecolor="#cccccc")  # band explained in the report caption

ap.legend(loc="upper right", fancybox=True, framealpha=1, edgecolor="#cccccc")
for a in (ae, ap):
    a.grid(color="#e5e5e5", lw=0.8)
    a.set_axisbelow(True)
    a.set_xlim(0.7, 3.1)
    a.set_xticks([1.0, 1.5, 2.0, 2.5, 3.0])
    a.set_xlabel("$R$ (\u00c5)")
fig.tight_layout(w_pad=3)
for ext in ("png", "pdf"):
    fig.savefig(f"{out}.{ext}", dpi=200)
print(f"{out}.png, {out}.pdf")
print(f"{'R':>5} " + " ".join(f"{m[0]:>9}" for m in methods) + "".join(f" {'E0 ' + c:>9}" for c in STEMS) + "   (mHa vs FCI)")
for i, r in enumerate(R):
    print(f"{r:5.2f} " + " ".join(f"{(m[1][i] - fci[i]) * 1e3:+9.3f}" for m in methods)
          + "".join(f" {(e0[c][i] - fci[i]) * 1e3:+9.3f}" for c in STEMS))
