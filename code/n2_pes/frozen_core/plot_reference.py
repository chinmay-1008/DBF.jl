"""Frozen-core N2 at 2.3 A, exact (untruncated) state-vector DBF from three starts:
the UHF determinant on H, the UHF determinant on H + 0.05 S^2, and the singlet
projection of the UHF determinant on H (sv_dbf.jl --init singlet).

Left:   <H> - E_FCI vs iteration (symlog).
Middle: fidelity with the exact ground state (solid) and <S^2>/3 (dotted).
Third:  the extrapolation plane on linear axes, <H> vs Var(H): filled = the
        tail window the vdbf-paper rule picks, hollow = earlier iterations,
        solid/dashed = its linear and quadratic fits, big marker = E0 at Var = 0.
Fourth: the same plane zoomed in near Var = 0 (zoom.py), with a table of each
        run's E0 and final <H>.

    python plot_reference.py [lmo,lno]     # -> reference_2.30_<bases>.png"""
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

RUNS = [  # key, label, colour
    ("onebody", "UHF start, H", "#2a78d6"),
    ("onebody_s2pen0.05", "UHF start, H + 0.05 S²", "#eb6834"),
    ("onebody_singlet", "singlet-projected UHF start, H", "#1baf7a"),
    ("onebody_singlet_s2pen0.05", "singlet-projected UHF start, H + 0.05 S²", "#9b59d0"),
]
INK, MUTED, GRID = "#1f1f1e", "#6b6a64", "#e4e3de"
MHA = 1e3


def style(ax, ylog=True):
    if ylog:
        ax.set_yscale("symlog", linthresh=0.01, linscale=0.6)
        ax.axhline(0, color=INK, lw=0.9, ls=":")
    ax.grid(color=GRID, lw=0.5)
    ax.tick_params(colors=MUTED, labelsize=8)
    for s in ax.spines.values():
        s.set_color(GRID)


BASES = sys.argv[1].split(",") if len(sys.argv) > 1 else ["lmo", "lno"]
fig, axes = plt.subplots(len(BASES), 4, figsize=(21, 4.7 * len(BASES)), squeeze=False)
notes, used, zoom_rows = [], set(), []
for row, basis in enumerate(BASES):
    eax, sax, vax, _ = axes[row]
    zoom_rows.append([])
    for key, label, colour in RUNS:
        p = os.path.join(HERE, "sv_results", f"{basis}_2.30_{key}.json")
        if not os.path.exists(p):
            continue
        used.add(key)
        d = json.load(open(p))
        fci, h = d["E_FCI"], d["history"]
        E, V = np.array(h["E"]), np.array(h["var"])
        it = np.arange(len(E))
        eax.plot(it, (E - fci) * MHA, color=colour, lw=1.4)
        sax.plot(it, h["fidelity"], color=colour, lw=1.3)
        sax.plot(it, np.array(h["S2"]) / 3, ":", color=colour, lw=1.3)
        x, y = V[1:], E[1:]
        best, _ = select_adaptive_tail(x, y, MIN_FIT_POINTS)
        k = best["num_take"]
        vax.plot(x[:-k], (y[:-k] - fci) * MHA, "o", color=colour, ms=2.4, mfc="none", alpha=0.5)
        vax.plot(x[-k:], (y[-k:] - fci) * MHA, "o", color=colour, ms=2.6)
        xs = np.linspace(0, x[-k:].max(), 100)
        vax.plot(xs, (np.polyval(np.polyfit(x[-k:], y[-k:], 1), xs) - fci) * MHA, color=colour, lw=1.3)
        vax.plot(xs, (np.polyval(np.polyfit(x[-k:], y[-k:], 2), xs) - fci) * MHA, color=colour, lw=1.1, ls="--")
        e0 = (best["E0_average"] - fci) * MHA
        vax.plot([0], [e0], "o", ms=8, color=colour, mec=INK, mew=0.8, zorder=5, clip_on=False)
        zoom_rows[-1].append(dict(colour=colour, x=x, y=(y - fci) * MHA, k=k, e0=e0, final_e=(E[-1] - fci) * MHA))
        below1 = next((i for i, e in enumerate(E) if (e - fci) * MHA < 1.0), None)
        notes.append(f"{basis.upper()} {label}: start {(E[0] - fci) * MHA:+.2f} mHa (⟨S²⟩ {h['S2'][0]:.3f}, "
                     f"fidelity {h['fidelity'][0]:.3f}); at {len(E) - 1}: {(E[-1] - fci) * MHA:+.3f} mHa, "
                     f"Var {V[-1]:.2e}, fidelity {h['fidelity'][-1]:.4f}, ⟨S²⟩ {h['S2'][-1]:.4f}; "
                     f"first < 1 mHa at it {below1}; tail E0 {e0:+.3f} mHa (last {k} points)")
    eax.set_title(f"{basis.upper()} 2.3 Å, frozen core, exact DBF: energy", color=INK, fontsize=10, loc="left")
    eax.set_xlabel("iteration", color=MUTED, fontsize=8)
    eax.set_ylabel("⟨H⟩ − E_FCI, mHa (symlog)", color=MUTED, fontsize=8)
    style(eax)
    sax.set_title("fidelity (solid) and ⟨S²⟩/3 (dotted)", color=INK, fontsize=10, loc="left")
    sax.set_xlabel("iteration", color=MUTED, fontsize=8)
    sax.set_ylim(0, 1.02)
    style(sax, ylog=False)
    vax.set_title("extrapolation plane (linear axes)", color=INK, fontsize=10, loc="left")
    vax.set_xlabel("Var(H), Ha²", color=MUTED, fontsize=8)
    vax.set_ylabel("⟨H⟩ − E_FCI, mHa", color=MUTED, fontsize=8)
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
xz, yz = zoom_limits(zoom_rows)
for r in range(len(BASES)):
    draw_zoom(axes[r, 3], zoom_rows[r], xz, yz, INK, MUTED)
    style(axes[r, 3], ylog=False)

handles = [Line2D([], [], color=c, lw=1.5, label=l) for k, l, c in RUNS if k in used]
handles += [Line2D([], [], color=MUTED, marker="o", ms=8, mec=INK, ls="none", label="paper-rule E0 (tail fit)"),
            Line2D([], [], color=MUTED, marker="o", ms=4, lw=1.3, label="right: tail window and linear fit"),
            Line2D([], [], color=MUTED, ls="--", lw=1.1, label="right: quadratic fit"),
            Line2D([], [], color="#efeee9", lw=6, label="zoom: ±1.6 mHa around FCI")]
fig.legend(handles=handles, loc="upper center", ncol=4, frameon=False, fontsize=9, labelcolor=INK)
fig.tight_layout(rect=(0, 0, 1, 0.93))
out = os.path.join(HERE, f"reference_2.30_{'_'.join(BASES)}.png")
fig.savefig(out, dpi=140)
print(out)
print("\n".join(notes))
