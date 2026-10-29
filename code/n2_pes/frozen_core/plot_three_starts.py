"""LNO 2.3 A, frozen core: the same three flows exact (left) and as truncated vDBF
(right, coeff 1e-5 or, with --coeff 1e-6, 1e-6):

    UHF determinant on H            (the bare trajectory)
    UHF determinant on H + 0.05 S^2 (spin penalty)
    singlet projection of the UHF determinant on H
        (exact: sv_dbf.jl --init singlet; vDBF: a prep circuit folded into H,
        run_dbf_pes.jl --prep singlet at 1e-5 (1 - F = 1e-3, 496 rotations) or
        --prep singlet5 at 1e-6 (1 - F = 1e-5, 2219 rotations))

Rows: E - E_FCI vs iteration; the extrapolation plane zoomed in near Var = 0
(zoom.py: tail fits, E0, true energy of the final vDBF circuit); the full
trajectory in the energy-variance plane.  vDBF energies and variances are the
corrected ones the run reports (E - err, Var - var err); for the penalty run
they are of H + 0.05 S^2.  The dashed lines with dots in the top right are the
true <H> of the vDBF circuits from exact replay (replay_fc.jl).

    python plot_three_starts.py                 # -> three_starts_lno_2.30.png
    python plot_three_starts.py --coeff 1e-6    # -> three_starts_lno_2.30_c1e-6.png
    python plot_three_starts.py --exact-only    # -> three_starts_lno_2.30_exact.pdf (left column only)
    python plot_three_starts.py --exact-only --uno
        # -> three_starts_lno_2.30_exact_uno.pdf, adding the closed-shell UHF
        #    natural orbital (UNO) start on H and on H + 0.05 S^2
        #    (data_unocore, make_uno_integrals.py)"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from extrapolate import MIN_FIT_POINTS, select_adaptive_tail  # noqa: E402
from zoom import draw_zoom, zoom_limits  # noqa: E402

COEFF = sys.argv[sys.argv.index("--coeff") + 1] if "--coeff" in sys.argv else "1e-5"
TAG = "" if COEFF == "1e-5" else f"_c{COEFF}"
PREP = "prepsinglet" if COEFF == "1e-5" else "prepsinglet5"
RUNS = [  # label, colour, exact key, vDBF key
    ("UHF start, H", "#2a78d6", "lno_2.30_onebody", f"trunc{TAG}"),
    ("UHF start, H + 0.05 S²", "#eb6834", "lno_2.30_onebody_s2pen0.05", f"trunc_s2pen0.05{TAG}"),
    ("singlet-projected UHF start, H", "#1baf7a", "lno_2.30_onebody_singlet", f"trunc_{PREP}{TAG}"),
]
EXACT = "--exact-only" in sys.argv
UNO = "--uno" in sys.argv
if UNO:
    assert EXACT, "--uno has exact flows only"
    RUNS += [("UNO closed-shell start, H", "#8a5cd1", "uno_2.30_onebody", None),
             ("UNO closed-shell start, H + 0.05 S²", "#c03b8a", "uno_2.30_onebody_s2pen0.05", None)]
COLS = (0,) if EXACT else (0, 1)
prep = None if EXACT else json.load(open(os.path.join(HERE, "data_nocore", "R_2.30", f"dbf_lno_{PREP}{TAG}.json")))["prep"]
INK, MUTED, GRID = "#1f1f1e", "#6b6a64", "#e4e3de"
MHA = 1e3


def style(ax):
    ax.grid(color=GRID, lw=0.5)
    ax.tick_params(colors=MUTED, labelsize=8)
    for s in ax.spines.values():
        s.set_color(GRID)


fig, axes = plt.subplots(3, len(COLS), figsize=(7.5 if EXACT else 14, 14), squeeze=False)
zoom_items = {"exact": [], "vdbf": []}
for label, colour, ekey, tkey in RUNS:
    # exact state-vector flow
    d = json.load(open(os.path.join(HERE, "sv_results", f"{ekey}.json")))
    fci, h = d["E_FCI"], d["history"]
    E, V = np.array(h["E"]), np.array(h["var"])
    y = (E - fci) * MHA
    axes[0, 0].plot(np.arange(len(E)), y, color=colour, lw=1.5)
    axes[2, 0].plot(V, y, "o-", color=colour, ms=2.2, lw=1.0)
    axes[2, 0].plot(V[0], y[0], "o", color=colour, ms=7, mec=INK, mew=0.8)
    best, _ = select_adaptive_tail(V[1:], E[1:], MIN_FIT_POINTS)
    zoom_items["exact"].append(dict(colour=colour, x=V[1:], y=y[1:], k=best["num_take"],
                                    e0=(best["E0_average"] - fci) * MHA, final_e=y[-1]))
    if EXACT:
        print(f"{label}: exact E0 {zoom_items['exact'][-1]['e0']:+.3f}, final {y[-1]:+.3f} mHa")
        continue

    # truncated vDBF, with the exact replay of its circuit
    d = json.load(open(os.path.join(HERE, "replay_results", f"lno_2.30_{tkey}.json")))
    fci, db, rows = d["E_FCI"], d["dbf"], d["rows"]
    E = np.array(db["energies_per_grad"]) - np.array(db["accumulated_error_per_grad"])
    V = np.array(db["variance_per_grad"]) - np.array(db["accumulated_var_error_per_grad"])
    y = (E - fci) * MHA
    axes[0, 1].plot(np.arange(len(E)), y, color=colour, lw=1.5)
    ti = [r["iter"] for r in rows]
    yt = [(r["E"] - fci) * MHA for r in rows]
    axes[0, 1].plot(ti, yt, "o--", color=colour, ms=3.5, lw=1.0)
    axes[2, 1].plot(V, y, "o-", color=colour, ms=2.2, lw=1.0)
    axes[2, 1].plot(V[0], y[0], "o", color=colour, ms=7, mec=INK, mew=0.8)
    best, _ = select_adaptive_tail(V[1:], E[1:], MIN_FIT_POINTS)
    zoom_items["vdbf"].append(dict(colour=colour, x=V[1:], y=y[1:], k=best["num_take"],
                                   e0=(best["E0_average"] - fci) * MHA,
                                   true_var=rows[-1]["var"], true_e=yt[-1]))
    print(f"{label}: exact E0 {zoom_items['exact'][-1]['e0']:+.3f}, final {zoom_items['exact'][-1]['final_e']:+.3f}; "
          f"vDBF E0 {zoom_items['vdbf'][-1]['e0']:+.2f}, reported {y[-1]:+.2f}, true {yt[-1]:+.2f} mHa")

# Row 1: convergence
for c, title in ((0, "exact DBF (no truncation)"), (1, f"vDBF (Pauli propagation, coeff {COEFF})"))[:len(COLS)]:
    ax = axes[0, c]
    ax.set_yscale("symlog", linthresh=1.0, linscale=0.6)
    ax.axhline(0, color=INK, lw=0.9, ls=":")
    ax.set_title(f"{title}: energy vs iteration", color=INK, fontsize=11, loc="left")
    ax.set_xlabel("iteration", color=MUTED, fontsize=9)
    ax.set_ylabel("E − E_FCI, mHa (symlog)", color=MUTED, fontsize=9)
    style(ax)
lo = min(axes[0, c].get_ylim()[0] for c in COLS); hi = max(axes[0, c].get_ylim()[1] for c in COLS)
for c in COLS:
    axes[0, c].set_ylim(lo, hi)

# Row 2: extrapolation, zoomed (each column its own window: exact variances are ~100x smaller)
for c, key, xz in ((0, "exact", None), (1, "vdbf", 6e-3))[:len(COLS)]:
    xzc, yz = zoom_limits([zoom_items[key]], xz=xz)
    draw_zoom(axes[1, c], zoom_items[key], xzc, yz, INK, MUTED)
    style(axes[1, c])
    axes[1, c].set_title("energy vs variance near Var = 0 (tail fits and E0)", color=INK, fontsize=11, loc="left")
    axes[1, c].set_xlabel("variance" + (" (corrected: Var − var err)" if c else "") + ", Ha²", color=MUTED, fontsize=9)
    axes[1, c].set_ylabel("E − E_FCI" + (" (corrected: E − err)" if c else "") + ", mHa", color=MUTED, fontsize=9)

# Row 3: full trajectory
for c in COLS:
    ax = axes[2, c]
    ax.set_xscale("log")
    ax.set_yscale("symlog", linthresh=1.0, linscale=0.6)
    ax.axhline(0, color=INK, lw=0.9, ls=":")
    ax.set_title("full trajectory: energy vs variance", color=INK, fontsize=11, loc="left")
    ax.set_xlabel("variance" + (" (corrected)" if c else "") + ", Ha² (log)", color=MUTED, fontsize=9)
    ax.set_ylabel("E − E_FCI" + (" (corrected)" if c else "") + ", mHa (symlog)", color=MUTED, fontsize=9)
    style(ax)
xl = [axes[2, c].get_xlim() for c in COLS]; yl = [axes[2, c].get_ylim() for c in COLS]
for c in COLS:
    axes[2, c].set_xlim(min(l[0] for l in xl), max(l[1] for l in xl))
    axes[2, c].set_ylim(min(l[0] for l in yl), max(l[1] for l in yl))

handles = [Line2D([], [], color=col, lw=2, label=l) for l, col, _, _ in RUNS]
if not EXACT:
    handles += [Line2D([], [], color=MUTED, marker="o", ms=3.5, ls="--", lw=1,
                       label="top right: true ⟨H⟩ of the vDBF circuit (exact replay)")]
handles += [Line2D([], [], color=MUTED, marker="o", ms=8, mec=INK, ls="none",
                   label="middle: E0 from the tail fit  ·  bottom: iteration 0")]
if not EXACT:
    handles += [Line2D([], [], color=MUTED, marker="*", ms=10, mfc="white", ls="none",
                       label="middle right: true ⟨H⟩ of the final vDBF circuit")]
handles += [Line2D([], [], color="#efeee9", lw=6, label="±1.6 mHa around FCI")]
fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.975), ncol=2 if EXACT else 3,
           frameon=False, fontsize=9.5, labelcolor=INK)
fig.suptitle(("LNO and UNO" if UNO else "LNO") + ", R = 2.3 Å, frozen core (16 qubits)", color=INK, fontsize=12, x=0.01, y=0.995, ha="left")
UNO and fig.text(0.01, 0.003, "UNO runs freeze the UNO 1s orbitals, so their FCI differs from LNO's by 3 µHa; "
                 "each run is measured from its own FCI.", color=MUTED, fontsize=8)
EXACT or fig.text(0.01, 0.003, "vDBF penalty run: energies and variances are of H + 0.05 S².  "
         f"vDBF singlet start: {prep['rotations']}-rotation prep circuit folded into H "
         f"(fold error {prep['fold_error'] * MHA:+.2f} mHa).",
         color=MUTED, fontsize=8.5)
fig.tight_layout(rect=(0, 0.01, 1, 0.94 if EXACT else 0.92))
out = os.path.join(HERE, f"three_starts_lno_2.30_exact{'_uno' if UNO else ''}.pdf" if EXACT else f"three_starts_lno_2.30{TAG}.png")
fig.savefig(out, dpi=140)
print(out)
