"""Plots for the sv_block.jl runs (idea 1c, exact block DBF on frozen-core N2).

    python plot_block.py            # from frozen_core/; writes sv_results/plots/*.png

1. energy_vs_variance.png  each block eigenvalue minus the exact level it ends on,
                           against the block variance V_S of H (log), one panel per run
2. tail_extrapolation.png  the same near V_S = 0 on linear axes, with straight-line fits
                           of the tail extrapolated to V_S = 0 (plan, open question 1)
3. block_variance.png      V_S of H against macro-iteration, all runs
4. levels.png              exact levels per symmetry sector, Ky Fan targets marked

Reference levels: each block state is compared with the exact level of its own symmetry
sector and spin that it ends on (matched on the final iteration), since the flow can end
on a symmetry-protected saddle rather than on the Ky Fan target.
"""
import json
import os

import matplotlib
import matplotlib.ticker
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "sv_results")
OUT = os.path.join(RES, "plots")
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]          # categorical slots 1-4
INK, MUTED, GRID = "#15171c", "#6a6f7d", "#e7e9ee"
SPIN = {0: "singlet", 2: "triplet", 6: "quintet", 12: "septet"}
RUNS = [  # label, title
    ("uno_1.10_block1c_exact", "1.1 Å, H"),
    ("uno_2.30_block1c_exact", "2.3 Å, H"),
    ("uno_2.30_block1c_exact_s2pen0.05", "2.3 Å, H + 0.05 S²"),
    ("uno_2.30_block1c_exact_s2pen0.1", "2.3 Å, H + 0.1 S²"),
]
plt.rcParams.update({"font.family": ["Helvetica", "Arial", "DejaVu Sans"], "font.size": 9,
                     "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
                     "ytick.color": MUTED, "axes.linewidth": 0.8, "legend.frameon": False})


def load(label):
    p = os.path.join(RES, label + ".json")
    return json.load(open(p)) if os.path.exists(p) else None


def run_data(d):
    """per-iteration block eigenvalues (mHa above FCI), V_S of H, and the matched references"""
    h, f = d["history"], d["E_FCI"]
    lam = (np.array(h["lam"]) - f) * 1e3
    vs = np.array(h.get("VS_H", h["VS"]))
    s2 = h["S2"][-1]
    cos = list(d["levels"].values())                 # sector of |0> first, then |a>
    owner = [0] + [1] * (lam.shape[1] - 1)
    refs, names = [], []
    for i, (s, c) in enumerate(zip(s2, owner)):
        same = [l for l in cos[c] if abs(l["S2"] - s) < 0.5]
        if same:
            l = min(same, key=lambda l: abs(l["E"] - (lam[-1, i] / 1e3 + f)))
            refs.append((l["E"] - f) * 1e3)
            names.append("S0" if abs(refs[-1]) < 1e-6 else f"{SPIN.get(round(l['S2']), '?')} {refs[-1]:.2f}")
        else:                                        # spin-mixed: compare with the Ky Fan target
            refs.append((d["targets"][i]["E"] - f) * 1e3)
            names.append("S0, spin-mixed" if abs(refs[-1]) < 1e-6 else f"spin-mixed, vs {refs[-1]:.2f}")
    return lam, vs, np.array(refs), names, s2


def style(ax):
    ax.grid(color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def energy_vs_variance(runs):
    # one row per run, shared axes, legends outside on the right
    fig, axes = plt.subplots(len(runs), 1, figsize=(9, 2.5 * len(runs) + 0.8), sharex=True, sharey=True,
                             squeeze=False)
    ticks = [-1000, -100, -10, -1, -0.1, 0, 0.01, 0.1, 1, 10, 100, 1000]
    for ax, (label, title, d) in zip(axes[:, 0], runs):
        lam, vs, refs, names, s2 = run_data(d)
        for i in range(lam.shape[1]):
            err = lam[:, i] - refs[i]
            ax.plot(vs, err, "-", color=SERIES[i], lw=1.6, label=f"{names[i]}   ⟨S²⟩ {s2[i]:.2f}")
            ax.plot(vs[-1], err[-1], "o", color=SERIES[i], ms=5.5, mec="white", mew=0.8, zorder=5)
            ax.plot(vs[0], err[0], "o", color=SERIES[i], ms=5.5, mfc="white", mew=1.2, zorder=5)
        ax.set_xscale("log")
        ax.set_yscale("symlog", linthresh=0.01, linscale=0.6)
        ax.axhline(0, color=INK, lw=0.8)
        ax.axhspan(-1.6, 1.6, color="#eef0f4", lw=0, zorder=0)
        ax.set_title(title, fontsize=10, color=INK, loc="left")
        ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=8, handlelength=1.4,
                  title="state the block ends on", title_fontsize=7.5, alignment="left")
        ax.set_yticks(ticks)
        ax.set_yticklabels(["−1000", "−100", "−10", "−1", "−0.1", "0", "0.01", "0.1", "1", "10", "100", "1000"],
                           fontsize=7.5)
        style(ax)
    axes[0, 0].invert_xaxis()                       # shared: start on the left, converged on the right
    axes[-1, 0].set_xlabel("block variance $V_S$ of H (Ha², log)")
    fig.supylabel("block eigenvalue − exact level (mHa, symlog)", fontsize=9, color=INK)
    fig.suptitle("Block states converging: energy error against block variance, converging to the right\n"
                 "hollow dot = start, filled dot = last iteration, grey band = ±1.6 mHa",
                 x=0.01, ha="left", fontsize=10, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return fig


def tail_extrapolation(runs, frac=0.4):
    fig, axes = plt.subplots(1, len(runs), figsize=(3.6 * len(runs), 3.4), squeeze=False)
    rows = []
    for ax, (label, title, d) in zip(axes[0], runs):
        lam, vs, refs, names, s2 = run_data(d)
        n = len(vs)
        tail = slice(int(n * (1 - frac)), n)
        x = vs[tail]
        for i in range(lam.shape[1]):
            y = lam[tail, i] - refs[i]
            p = np.polyfit(x, y, 1)
            xs = np.linspace(0, x.max(), 50)
            ax.plot(x, y, "o", color=SERIES[i], ms=2.2, label=names[i])
            ax.plot(xs, np.polyval(p, xs), "--", color=SERIES[i], lw=1)
            ax.plot(0, p[1], "D", color=SERIES[i], ms=5, mec=INK, mew=0.6, clip_on=False, zorder=5)
            rows.append((title, names[i], y[-1], p[1]))
        ax.axhline(0, color=INK, lw=0.8)
        ax.set_xlim(left=0)
        ax.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
        ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(4))
        ax.set_title(f"{title}: last {int(frac * 100)}% of iterations", fontsize=10, color=INK, loc="left")
        ax.set_xlabel("block variance $V_S$ of H (Ha²)")
        ax.legend(loc="upper left", fontsize=7.5, handlelength=1)
        style(ax)
    axes[0, 0].set_ylabel("block eigenvalue − exact level (mHa)")
    fig.suptitle("Near convergence: straight-line fit of the tail, extrapolated to $V_S$ = 0 (diamonds)",
                 x=0.01, ha="left", fontsize=10, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    return fig, rows


def block_variance(all_runs):
    panels = [("1.1 Å", [("uno_1.10_block1c_exact", "exact ranking"), ("uno_1.10_block1c_d1c", "d1c ranking")]),
              ("2.3 Å", [("uno_2.30_block1c_exact", "exact ranking"), ("uno_2.30_block1c_d1c", "d1c ranking"),
                         ("uno_2.30_block1c_exact_s2pen0.05", "exact, H + 0.05 S²"),
                         ("uno_2.30_block1c_exact_s2pen0.1", "exact, H + 0.1 S²")])]
    colours = {"exact ranking": SERIES[0], "d1c ranking": SERIES[1], "exact, H + 0.05 S²": SERIES[2],
               "exact, H + 0.1 S²": SERIES[3]}
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.3), sharey=True)
    for ax, (title, runs) in zip(axes, panels):
        for label, name in runs:
            d = all_runs.get(label)
            if d is None:
                continue
            h = d["history"]
            v = np.array(h.get("VS_H", h["VS"]))
            ax.plot(np.arange(len(v)), v, color=colours[name], lw=1.6, label=name)
        ax.set_yscale("log")
        ax.set_title(title, fontsize=10, color=INK, loc="left")
        ax.set_xlabel("macro-iteration (50 rotations each)")
        ax.legend(fontsize=8)
        style(ax)
    axes[0].set_ylabel("block variance $V_S$ of H (Ha²)")
    fig.tight_layout()
    return fig


def levels():
    # exact levels in mHa above S0: (energy, spin label, degeneracy, Ky Fan target components for bare H)
    data = [("1.1 Å", 400, [
                [(0, "S0", 1, 1)],
                [(297.174, "triplet ³Πg", 2, 2), (347.384, "singlet ¹Πg", 2, 0)]],
             ["next 622.6 quintet ×3", "next 722.6 quintet"]),
            ("2.3 Å", 30, [
                [(0, "S0", 1, 1), (8.995, "quintet", 3, 0)],
                [(2.842, "triplet T1", 1, 1), (20.136, "septet", 3, 1)]],
             ["next 120.0 quintet ×3", "next 120.6 triplet ×2;\n227.5 singlet S1"])]
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8))
    for ax, (title, top, cols, notes) in zip(axes, data):
        for ci, col in enumerate(cols):
            x0 = ci * 1.6
            for E, name, deg, tgt in col:
                ax.plot([x0, x0 + 0.6], [E, E], color=INK, lw=2.2, solid_capstyle="round")
                ax.text(x0 + 0.68, E, f"{name}  {E:g}" + (f"  ×{deg}" if deg > 1 else ""), va="center", fontsize=8,
                        color=INK)
                for k in range(tgt):
                    ax.plot(x0 - 0.12 - 0.13 * k, E, "D", color="#4a3dc2", ms=5)
            ax.text(x0, top * 1.06, "sector of |0⟩" if ci == 0 else "sector of HOMO→LUMO", fontsize=8.5, color=INK)
            ax.text(x0, top * 0.99, "↑ " + notes[ci], fontsize=7.5, color=MUTED, va="top")
        ax.set_xlim(-0.5, 3.6)
        ax.set_ylim(-top * 0.04, top * 1.12)
        ax.set_xticks([])
        ax.set_ylabel("mHa above S0")
        ax.set_title(f"{title} (scale 0–{top} mHa)", fontsize=10, color=INK, loc="left")
        style(ax)
        ax.spines["bottom"].set_visible(False)
    fig.suptitle("Exact levels per symmetry sector; diamonds = Ky Fan target of the bare-H flow",
                 x=0.01, ha="left", fontsize=10, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    return fig


def main():
    os.makedirs(OUT, exist_ok=True)
    all_runs = {}
    for label in ["uno_1.10_block1c_exact", "uno_1.10_block1c_d1c", "uno_2.30_block1c_exact", "uno_2.30_block1c_d1c",
                  "uno_2.30_block1c_exact_s2pen0.05", "uno_2.30_block1c_exact_s2pen0.1"]:
        d = load(label)
        if d is not None:
            all_runs[label] = d
    runs = [(l, t, all_runs[l]) for l, t in RUNS if l in all_runs]
    energy_vs_variance(runs).savefig(os.path.join(OUT, "energy_vs_variance.png"), dpi=170)
    conv = [r for r in runs if r[0] in ("uno_1.10_block1c_exact", "uno_2.30_block1c_exact_s2pen0.05",
                                         "uno_2.30_block1c_exact_s2pen0.1")]
    fig, rows = tail_extrapolation(conv)
    fig.savefig(os.path.join(OUT, "tail_extrapolation.png"), dpi=170)
    block_variance(all_runs).savefig(os.path.join(OUT, "block_variance.png"), dpi=170)
    levels().savefig(os.path.join(OUT, "levels.png"), dpi=170)
    print("tail fits (mHa): run | state | last error | extrapolated to V_S = 0")
    for r in rows:
        print(f"  {r[0]:<20} {r[1]:<28} {r[2]:+9.4f} {r[3]:+9.4f}")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
