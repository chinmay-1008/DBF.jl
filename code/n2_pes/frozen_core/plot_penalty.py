"""Frozen-core N2 (2.3 A unless --R): truncated dbf_groundstate on H and on H + λ S^2,
with the true energy of each circuit (exact replay) and the exact
(untruncated) state-vector DBF for reference.

Left:   E - E_FCI vs iteration.  Solid = what the run reports (truncated: E -
        accumulated error, of H or of H + λS^2; exact: <H>).  Dashed with
        markers = true <H> of the truncated circuit from replay_fc.jl.
Middle: true <S^2> (dotted) and fidelity with the exact ground state (solid).
Third:  the extrapolation plane on linear axes, as each truncated run reports
        it (E - err vs Var - var err): filled = the tail window the vdbf-paper
        rule picked, hollow = earlier iterations, solid/dashed = its linear and
        quadratic fits, big marker = E0 (their average) at Var = 0.  Exact runs
        plotted as <H> vs Var(H).
Fourth: the same plane zoomed in near Var = 0 (zoom.py), with each run's E0,
        the true <H> of the final truncated circuit (hollow star), the exact
        runs' own tail fits, and a table of E0 and the true energy.

    python plot_penalty.py [lno,mo]        # one row per basis -> penalty_2.30_<bases>.png
    python plot_penalty.py lmo,lno prep    # singlet-prep runs  -> prep_2.30_<bases>.png
    python plot_penalty.py lmo,lno curve --R 1.70 --lam 0.077
                                           # penalty at coeff 1e-6 with exact references,
                                           # any R and λ -> curve_<R>_<bases>.png"""
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

pos = [a for i, a in enumerate(sys.argv[1:], 1) if not a.startswith("--") and not sys.argv[i - 1].startswith("--")]
opt = {sys.argv[i][2:]: sys.argv[i + 1] for i in range(1, len(sys.argv) - 1) if sys.argv[i].startswith("--")}
R, LAM = opt.get("R", "2.30"), opt.get("lam", "0.05")
SET = pos[1] if len(pos) > 1 else "penalty"
RUNS = [  # key, label, colour, kind
    ("trunc", "truncated DBF on H, coeff 1e-5", "#6b6a64", "trunc"),
    (f"trunc_s2pen{LAM}", f"truncated DBF on H + {LAM} S², coeff 1e-5", "#1baf7a", "trunc"),
    (f"trunc_s2pen{LAM}_c1e-6", f"truncated DBF on H + {LAM} S², coeff 1e-6", "#eda100", "trunc"),
    ("onebody", "exact DBF on H", "#2a78d6", "exact"),
    (f"onebody_s2pen{LAM}", f"exact DBF on H + {LAM} S²", "#eb6834", "exact"),
]
if SET == "prep":
    RUNS = [
        ("trunc", "truncated DBF on H from UHF, coeff 1e-5", "#6b6a64", "trunc"),
        (f"trunc_s2pen{LAM}", f"truncated DBF on H + {LAM} S² from UHF, coeff 1e-5", "#1baf7a", "trunc"),
        ("trunc_prepsinglet", "truncated DBF on H from the singlet prep circuit (1−F = 1e-3), coeff 1e-5", "#9b59d0", "trunc"),
        ("trunc_prepsinglet5", "truncated DBF on H from the singlet prep circuit (1−F = 1e-5), coeff 1e-5", "#c03b8a", "trunc"),
        ("onebody", "exact DBF on H from UHF", "#2a78d6", "exact"),
        ("onebody_singlet", "exact DBF on H from the singlet projection", "#eb6834", "exact"),
    ]
elif SET == "curve":
    RUNS = [
        ("trunc", "truncated DBF on H from UHF, coeff 1e-5", "#6b6a64", "trunc"),
        (f"trunc_s2pen{LAM}_c1e-6", f"truncated DBF on H + {LAM} S² from UHF, coeff 1e-6", "#eda100", "trunc"),
        ("onebody", "exact DBF on H from UHF", "#2a78d6", "exact"),
        (f"onebody_s2pen{LAM}", f"exact DBF on H + {LAM} S² from UHF", "#eb6834", "exact"),
        ("onebody_singlet", "exact DBF on H from the singlet projection", "#1baf7a", "exact"),
    ]
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


BASES = pos[0].split(",") if pos else ["lno", "mo"]
fig, axes = plt.subplots(len(BASES), 4, figsize=(21, 4.7 * len(BASES)), squeeze=False)
notes, zoom_rows = [], []
for row, basis in enumerate(BASES):
    eax, sax, vax, _ = axes[row]
    zoom_rows.append([])
    for key, label, colour, kind in RUNS:
        if kind == "trunc":
            p = os.path.join(HERE, "replay_results", f"{basis}_{R}_{key}.json")
            if not os.path.exists(p):
                continue
            d = json.load(open(p))
            fci, db = d["E_FCI"], d["dbf"]
            E = np.array(db["energies_per_grad"]); err = np.array(db["accumulated_error_per_grad"])
            V = np.array(db["variance_per_grad"]); verr = np.array(db["accumulated_var_error_per_grad"])
            it = np.arange(len(E))
            eax.plot(it, (E - err - fci) * MHA, color=colour, lw=1.5)
            r = d["rows"]
            ti = np.array([x["iter"] for x in r]); Et = np.array([x["E"] for x in r])
            eax.plot(ti, (Et - fci) * MHA, "o--", color=colour, ms=3.5, lw=1.1)
            sax.plot(ti, [x["fidelity"] for x in r], "o-", color=colour, ms=3, lw=1.3)
            sax.plot(ti, np.array([x["S2"] for x in r]) / 3, ":", color=colour, lw=1.3)
            x, y = (V - verr)[1:], (E - err)[1:]
            best, _ = select_adaptive_tail(x, y, MIN_FIT_POINTS)
            k = best["num_take"]
            vax.plot(x[:-k], (y[:-k] - fci) * MHA, "s", color=colour, ms=2.4, mfc="none", alpha=0.5)
            vax.plot(x[-k:], (y[-k:] - fci) * MHA, "s", color=colour, ms=2.6)
            xs = np.linspace(0, x[-k:].max(), 100)
            vax.plot(xs, (np.polyval(np.polyfit(x[-k:], y[-k:], 1), xs) - fci) * MHA, color=colour, lw=1.3)
            vax.plot(xs, (np.polyval(np.polyfit(x[-k:], y[-k:], 2), xs) - fci) * MHA, color=colour, lw=1.1, ls="--")
            e0 = (best["E0_average"] - fci) * MHA
            vax.plot([0], [e0], "o", ms=8, color=colour, mec=INK, mew=0.8, zorder=5, clip_on=False)
            zoom_rows[-1].append(dict(colour=colour, x=x, y=(y - fci) * MHA, k=k, e0=e0,
                                      true_var=r[-1]["var"], true_e=(Et[-1] - fci) * MHA))
            kbest = 1 + int(np.argmin(Et[1:]))  # iteration 0 is the reference
            n = len(E) - 1
            notes.append(f"{basis.upper()} {label}: reported E−err at {n} {(E[-1] - err[-1] - fci) * MHA:+.2f} mHa, "
                         f"paper E0 {e0:+.2f}; true ⟨H⟩ at {n} {(Et[-1] - fci) * MHA:+.2f}, "
                         f"best true {(Et[kbest] - fci) * MHA:+.2f} at it {ti[kbest]}, "
                         f"⟨S²⟩ at 200 {r[-1]['S2']:.3f}, fidelity {r[-1]['fidelity']:.3f}")
        else:
            p = os.path.join(HERE, "sv_results", f"{basis}_{R}_{key}.json")
            if not os.path.exists(p):
                continue
            d = json.load(open(p))
            fci, h = d["E_FCI"], d["history"]
            E = np.array(h["E"]); it = np.arange(len(E))
            eax.plot(it, (E - fci) * MHA, color=colour, lw=1.3)
            sax.plot(it, h["fidelity"], color=colour, lw=1.1)
            sax.plot(it, np.array(h["S2"]) / 3, ":", color=colour, lw=1.1)
            vax.plot(np.array(h["var"])[1:], (E - fci)[1:] * MHA, color=colour, lw=1.3)
            x, y = np.array(h["var"])[1:], (E - fci)[1:] * MHA
            best, _ = select_adaptive_tail(x, y, MIN_FIT_POINTS)
            zoom_rows[-1].append(dict(colour=colour, x=x, y=y, k=best["num_take"], e0=best["E0_average"],
                                      final_e=y[-1]))
    eax.set_title(f"{basis.upper()} {float(R):.1f} Å, frozen core: energy", color=INK, fontsize=10, loc="left")
    eax.set_xlabel("iteration", color=MUTED, fontsize=8)
    eax.set_ylabel("E − E_FCI, mHa (symlog)", color=MUTED, fontsize=8)
    style(eax)
    sax.set_title("fidelity (solid) and ⟨S²⟩/3 (dotted)", color=INK, fontsize=10, loc="left")
    sax.set_xlabel("iteration", color=MUTED, fontsize=8)
    sax.set_ylim(0, 1.02)
    style(sax, ylog=False)
    vax.set_title("extrapolation plane (linear axes)", color=INK, fontsize=10, loc="left")
    vax.set_xlim(left=0)
    vax.set_xlabel("variance (truncated: Var − var err), Ha²", color=MUTED, fontsize=8)
    vax.set_ylabel("E − E_FCI, mHa", color=MUTED, fontsize=8)
    vax.axhline(0, color=INK, lw=0.9, ls=":")
    style(vax, ylog=False)

# Same limits down each column so the bases compare directly
for col in (0, 2):
    lims = [axes[r, col].get_ylim() for r in range(len(BASES))]
    for r in range(len(BASES)):
        axes[r, col].set_ylim(min(l[0] for l in lims), max(l[1] for l in lims))
xlims = [axes[r, 2].get_xlim() for r in range(len(BASES))]
for r in range(len(BASES)):
    axes[r, 2].set_xlim(0, max(l[1] for l in xlims))
# Truncated tails sit at 1e-3 - 1e-2 Ha^2, the exact ones below 1e-4: show the former
xz, yz = zoom_limits(zoom_rows, xz=6e-3)
for r in range(len(BASES)):
    draw_zoom(axes[r, 3], zoom_rows[r], xz, yz, INK, MUTED)
    style(axes[r, 3], ylog=False)

handles = [Line2D([], [], color=c, lw=1.5, label=l) for _, l, c, _ in RUNS]
handles += [Line2D([], [], color=MUTED, lw=1.5, label="as reported by the run (truncated: E − err)"),
            Line2D([], [], color=MUTED, marker="o", ms=3.5, ls="--", label="true ⟨H⟩ of the truncated circuit (exact replay)"),
            Line2D([], [], color=MUTED, marker="o", ms=8, mec=INK, ls="none", label="paper-rule E0 (tail fit)"),
            Line2D([], [], color=MUTED, marker="s", ms=4, lw=1.3, label="right: tail window and linear fit"),
            Line2D([], [], color=MUTED, ls="--", lw=1.1, label="right: quadratic fit"),
            Line2D([], [], color=MUTED, marker="*", ms=10, mfc="white", ls="none",
                   label="zoom: true ⟨H⟩ of the final truncated circuit"),
            Line2D([], [], color="#efeee9", lw=6, label="zoom: ±1.6 mHa around FCI")]
fig.legend(handles=handles, loc="upper center", ncol=4, frameon=False, fontsize=9, labelcolor=INK)
fig.tight_layout(rect=(0, 0, 1, 0.91))
out = os.path.join(HERE, f"{SET}_{R}_{'_'.join(BASES)}.png")
fig.savefig(out, dpi=140)
print(out)
print("\n".join(notes))
