"""Plots for the sv_block.jl runs (idea 1c, exact block DBF on frozen-core N2).

    python plot_block.py            # from frozen_core/; writes sv_results/plots/*.png
    python plot_block.py --fidelity # only n2_fidelity.png (every N2 block run) and n2_fidelity_kets.png (old vs new kets)

1. energies_vs_variance.png  the block eigenvalues against the block variance V_S of H
                             (log; each run starts on the right), exact levels as horizontal lines
2. energies_zoom.png         one panel per run and state: the eigenvalue landing on its exact line
3. error_vs_variance.png     block eigenvalue minus the exact level it ends on, symlog
4. tail_extrapolation.png    the error near V_S = 0 on linear axes, with straight-line fits
                             of the tail extrapolated to V_S = 0 (plan, open question 1)
5. block_variance.png        V_S of H against macro-iteration, all runs
6. levels.png                exact levels per symmetry sector, Ky Fan targets marked

Reference levels: each block state is compared with the exact level of its own symmetry
sector and spin that it ends on (matched on the final iteration), since the flow can end
on a symmetry-protected saddle rather than on the Ky Fan target.
"""
import json
import os
import sys

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


def nz(x, eps):
    """x, with values that round to zero printed as +0 rather than -0"""
    return 0.0 if abs(x) < eps else x


def load(label):
    p = os.path.join(RES, label + ".json")
    return json.load(open(p)) if os.path.exists(p) else None


def run_data(d):
    """per-iteration block eigenvalues (mHa above FCI), V_S of H, and the matched references"""
    h, f = d["history"], d["E_FCI"]
    lam = (np.array(h["lam"]) - f) * 1e3
    vs = np.array(h.get("VS_H", h["VS"]))
    keep = vs > 1e-14                                # rows past full convergence add nothing on a log axis
    lam, vs = lam[keep], vs[keep]
    s2 = h["S2"][-1]
    # sector of |0> first (its lowest level is FCI), then the sector of |a>
    cos = sorted(d["levels"].values(), key=lambda lv: min(abs(l["E"] - f) for l in lv))
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


def exact_lines(d, refs):
    """exact levels to draw for a run: the levels its states end on plus its Ky Fan targets (mHa above FCI)"""
    f = d["E_FCI"]
    allv = [l for c in d["levels"].values() for l in c]
    out = []
    for e in list(refs) + [(t["E"] - f) * 1e3 for t in d["targets"]]:
        if any(abs(e - o[0]) < 1e-3 for o in out):
            continue
        l = min(allv, key=lambda l: abs((l["E"] - f) * 1e3 - e))
        is_ref = any(abs(e - r) < 1e-3 for r in refs)
        name = "S0" if abs(e) < 1e-6 else f"{SPIN.get(round(l['S2']), '?')} {e:.2f}"
        out.append((e, name if is_ref else name + " (Ky Fan target, missed)", is_ref))
    return sorted(out)


def energies_vs_variance(runs):
    """the block eigenvalues themselves against V_S, with the exact levels as horizontal lines"""
    ncol = min(2, len(runs))
    nrow = (len(runs) + ncol - 1) // ncol
    fig, axes = plt.subplots(nrow, ncol, figsize=(5.5 * ncol, 3.8 * nrow + 0.4), squeeze=False)
    for ax, (label, title, d) in zip(axes.flat, runs):
        lam, vs, refs, names, s2 = run_data(d)
        lines = exact_lines(d, refs)
        top = max(lam[0].max(), max(e for e, _, _ in lines)) * 1.08
        bot = -0.06 * top
        for e, name, is_ref in lines:
            ax.axhline(e, color=INK if is_ref else MUTED, lw=0.9, ls="--" if is_ref else ":", zorder=1)
        # labels for the exact lines at the right edge, dodged so close levels stay readable
        ys = [e for e, _, _ in lines]
        gap = 0.045 * (top - bot)
        for k in range(1, len(ys)):
            ys[k] = max(ys[k], ys[k - 1] + gap)
        for (e, name, is_ref), y in zip(lines, ys):
            ax.annotate("exact " + name, xy=(1.0, e), xycoords=("axes fraction", "data"), xytext=(1.01, y),
                        textcoords=("axes fraction", "data"), fontsize=7.5, va="center",
                        color=INK if is_ref else MUTED, arrowprops=dict(arrowstyle="-", color=GRID, lw=0.6))
        for i in range(lam.shape[1]):
            ax.plot(vs, lam[:, i], "-", color=SERIES[i], lw=1.6, label=f"{names[i]}   ⟨S²⟩ {nz(s2[i], 5e-3):.2f}", zorder=3)
            ax.plot(vs[0], lam[0, i], "o", color=SERIES[i], ms=5.5, mfc="white", mew=1.2, zorder=5)
            ax.plot(vs[-1], lam[-1, i], "o", color=SERIES[i], ms=5.5, mec="white", mew=0.8, zorder=5)
        ax.set_xscale("log")
        ax.set_ylim(bot, top)
        ax.set_title(title, fontsize=10, color=INK, loc="left")
        ax.set_xlabel("block variance $V_S$ of H (Ha², log)")
        ax.set_ylabel("energy (mHa above exact S0)")
        ax.legend(loc="upper left", fontsize=7.5, handlelength=1.4, title="block eigenvalues",
                  title_fontsize=7.5, alignment="left")
        style(ax)
    fig.suptitle("Block eigenvalues against block variance; exact energies dashed\n"
                 "starts on the right (hollow dots), converges to the left (filled dots)",
                 x=0.01, ha="left", fontsize=10, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.94 if nrow > 1 else 0.88), w_pad=9)
    return fig


def energies_zoom(runs, windows=None, rights=None):
    """one small panel per run and state: the energy landing on its exact line.  windows / rights (per state)
    fix the y half-window and the right x edge, e.g. to match another figure; returns (fig, windows, rights)"""
    nS = 3
    used_w, used_r = [], []
    fig, axes = plt.subplots(len(runs), nS, figsize=(11, 2.5 * len(runs) + 0.9), squeeze=False)
    data = [run_data(d) for _, _, d in runs]
    if windows:                                       # same y range in every panel, for a fair comparison
        W = max(windows)
    else:
        W = max(max(abs(lam[len(lam) // 2, i] - refs[i]), 4 * abs(lam[-1, i] - refs[i]), 0.02)
                for lam, _, refs, _, _ in data for i in range(nS))
    for r, (label, title, d) in enumerate(runs):
        lam, vs, refs, names, s2 = data[r]
        n = len(vs)
        for i in range(nS):
            ax = axes[r, i]
            err = lam[:, i] - refs[i]
            w = W
            ax.axhline(0, color=INK, lw=1, ls="--", zorder=1)
            ax.plot(vs, err, "-", color=SERIES[i], lw=1.6, zorder=3)
            ax.plot(vs[-1], err[-1], "o", color=SERIES[i], ms=5.5, mec="white", mew=0.8, zorder=5)
            ax.set_xscale("log")
            ax.set_ylim(-0.25 * w, 1.15 * w)
            # x range: from where the curve last enters the frame to the end of the run
            inside = np.abs(err) <= 1.15 * w
            k = n - 1
            while k > 0 and inside[k - 1]:
                k -= 1
            right = rights[i] if rights else min(vs.max() * 1.5, vs[max(k - 1, 0)] * 3)
            ax.set_xlim(vs.min() / 1.5, right)
            used_w.append(w); used_r.append(right)
    # same x range in every panel too: the widest of the per-panel crops
    lefts = [ax.get_xlim()[0] for ax in axes.flat]
    for ax in axes.flat:
        ax.set_xlim(min(lefts), max(used_r))
    used_r = [max(used_r)] * len(used_r)
    for r in range(len(runs)):
        lam, vs, refs, names, s2 = data[r]
        title = runs[r][1]
        for i in range(nS):
            ax = axes[r, i]
            err = lam[:, i] - refs[i]
            ax.set_title(f"{title}: {names[i]}", fontsize=8.5, color=INK, loc="left")
            ax.text(0.98, 0.94, f"last: {nz(err[-1], 5e-5):+.4f} mHa\n⟨S²⟩ {nz(s2[i], 5e-3):.2f}", transform=ax.transAxes,
                    ha="right", va="top", fontsize=7.5, color=INK,
                    bbox=dict(facecolor="white", edgecolor="none", alpha=0.85, pad=1.5))
            ax.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(4))
            ax.ticklabel_format(axis="y", useOffset=False)
            ax.tick_params(labelsize=7.5)
            style(ax)
            if r == len(runs) - 1:
                ax.set_xlabel("$V_S$ of H (Ha², log)")
            if i == 0:
                ax.set_ylabel("energy − exact (mHa)")
    fig.suptitle("Zoom on each state: error from that state's exact energy (dashed line = 0)\n"
                 "starts on the right, converges to the left; all panels share the same x and y ranges",
                 x=0.01, ha="left", fontsize=10, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return fig, used_w, used_r


def error_vs_variance(runs):
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
    axes[-1, 0].set_xlabel("block variance $V_S$ of H (Ha², log)")
    fig.supylabel("block eigenvalue − exact level (mHa, symlog)", fontsize=9, color=INK)
    fig.suptitle("Energy error of each block state against block variance; starts on the right, converges to the left\n"
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


FLOW_LS = (0, (1.2, 1.6))           # dotted, for the full flow when drawn next to greedy (solid)


def compare_energies(greedy, flow, title):
    """the doc's energies-vs-variance figure with greedy (solid) and the full flow (dotted) on the same axes"""
    fig, ax = plt.subplots(figsize=(6.6, 4.4))
    lam_g, vs_g, refs, names, s2 = run_data(greedy)
    lam_f, vs_f, _, _, _ = run_data(flow)
    lines = exact_lines(greedy, refs)
    top = max(lam_g[0].max(), max(e for e, _, _ in lines)) * 1.08
    bot = -0.06 * top
    for e, name, is_ref in lines:
        ax.axhline(e, color=INK, lw=0.9, ls="--", zorder=1)
        ax.annotate("exact " + name, xy=(1.01, e), xycoords=("axes fraction", "data"), fontsize=7.5, va="center",
                    color=INK)
    for i in range(lam_g.shape[1]):
        ax.plot(vs_g, lam_g[:, i], "-", color=SERIES[i], lw=1.6, zorder=3, label=names[i])
        ax.plot(vs_f, lam_f[:, i], ls=FLOW_LS, color=SERIES[i], lw=2.2, zorder=4)
        ax.plot(vs_g[0], lam_g[0, i], "o", color=SERIES[i], ms=5.5, mfc="white", mew=1.2, zorder=5)
        ax.plot(vs_g[-1], lam_g[-1, i], "o", color=SERIES[i], ms=5.5, mec="white", mew=0.8, zorder=5)
        ax.plot(vs_f[-1], lam_f[-1, i], "D", color=SERIES[i], ms=5, mec="white", mew=0.8, zorder=5)
    ax.set_xscale("log")
    ax.set_ylim(bot, top)
    ax.set_title(title, fontsize=10, color=INK, loc="left")
    ax.set_xlabel("block variance $V_S$ of H (Ha², log)")
    ax.set_ylabel("energy (mHa above exact S0)")
    from matplotlib.lines import Line2D
    states = ax.legend(loc="upper left", fontsize=7.5, title="state", title_fontsize=7.5, alignment="left")
    ax.add_artist(states)
    ax.legend(handles=[Line2D([], [], color=INK, lw=1.6, ls="-", marker="o", ms=4, label="greedy vDBF (200 iterations)"),
                       Line2D([], [], color=INK, lw=2.2, ls=FLOW_LS, marker="D", ms=4, label="full flow"),
                       Line2D([], [], color=INK, lw=0.9, ls="--", label="exact")],
              loc="lower left", bbox_to_anchor=(0.0, 0.17), fontsize=7.5, title="method", title_fontsize=7.5,
              alignment="left")
    style(ax)
    fig.suptitle("Greedy vDBF and the full flow on the same axes; exact energies dashed\n"
                 "both start on the right (hollow dots); filled dot = greedy end, diamond = full-flow end",
                 x=0.01, ha="left", fontsize=10, color=INK)
    fig.tight_layout(rect=(0, 0, 0.9, 0.86))
    return fig


def compare_zoom(greedy, flow, title, windows, rights):
    """the doc's per-state zoom with greedy (solid) and the full flow (dotted) in the same panels"""
    lam_g, vs_g, refs, names, s2 = run_data(greedy)
    lam_f, vs_f, _, _, _ = run_data(flow)
    nS = lam_g.shape[1]
    fig, axes = plt.subplots(1, nS, figsize=(11, 3.4), squeeze=False)
    for i in range(nS):
        ax, w = axes[0, i], max(windows)
        ax.axhline(0, color=INK, lw=1, ls="--", zorder=1)
        ax.plot(vs_g, lam_g[:, i] - refs[i], "-", color=SERIES[i], lw=1.6, zorder=3, label="greedy vDBF")
        ax.plot(vs_f, lam_f[:, i] - refs[i], ls=FLOW_LS, color=SERIES[i], lw=2.2, zorder=4, label="full flow")
        ax.plot(vs_g[-1], lam_g[-1, i] - refs[i], "o", color=SERIES[i], ms=5.5, mec="white", mew=0.8, zorder=5)
        ax.set_xscale("log")
        ax.set_ylim(-0.25 * w, 1.15 * w)
        ax.set_xlim(vs_f.min() / 1.5, rights[i])
        ax.set_title(f"{title}: {names[i]}", fontsize=8.5, color=INK, loc="left")
        ax.text(0.98, 0.94, f"greedy, last: {nz(lam_g[-1, i] - refs[i], 5e-5):+.4f} mHa", transform=ax.transAxes,
                ha="right", va="top", fontsize=7.5, color=INK,
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.85, pad=1.5))
        ax.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(4))
        ax.ticklabel_format(axis="y", useOffset=False)
        ax.tick_params(labelsize=7.5)
        ax.set_xlabel("$V_S$ of H (Ha², log)")
        style(ax)
        if i == 0:
            ax.set_ylabel("energy − exact (mHa)")
            ax.legend(loc="center left", fontsize=7.5)
    fig.suptitle("Zoom on each state: error from that state's exact energy, greedy vDBF (solid) and the full flow (dotted)\n"
                 "same windows as the separate zoom figures; greedy stops at its last iteration, the full flow continues",
                 x=0.01, ha="left", fontsize=10, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.84))
    return fig


# ---------------------------------------------------------------- per-state variance extrapolation
NEG_WINDOW = 2e-4      # Ha²; fit window [0, NEG_WINDOW] for a series whose last variance is negative


def window_fit(x, y, factor=2.0):
    """linear fit of y against x over the window [v, factor v], v = the last x (the run's final variance).
    If v <= 0 (a truncation-corrected variance can turn negative) the window is [0, NEG_WINDOW] instead,
    negative points left out.  E0 = the linear intercept; uncertainty = half its difference from a quadratic
    fit over the same window.  Returns None when the window holds fewer than 3 points."""
    v = x[-1]
    if v > 0:
        lo, hi = v, factor * v
    else:
        lo, hi = 0.0, NEG_WINDOW
    sel = (x >= lo) & (x <= hi)
    k = int(sel.sum())
    if k < 3:
        return None
    lin = np.polyfit(x[sel], y[sel], 1)
    quad = np.polyfit(x[sel], y[sel], 2) if k >= 5 else None
    half = abs(lin[1] - quad[2]) / 2 if quad is not None else float("nan")
    return dict(e0=lin[1], half=half, sel=sel, lin=lin, v=v, k=k, lo=lo, hi=hi)


def state_extrapolation(entries, refs, names, title, grid=False, factor=2.0):
    """per state: energy against that state's own variance, fitted linearly over the window [v, 2v] with v the
    state's variance at the last iteration, and extrapolated to zero variance.
    entries: [(name, lam (iterations x states, mHa above FCI), var (iterations x states, Ha²), marker, linestyle)].
    Iteration 0 is left out.  Returns (fig, rows); rows hold (series, state, last error, E0, ±, v, #points)."""
    nS = len(refs)
    nr = len(entries) if grid else 1                  # grid: one row per series, each panel on its own axes
    fig, axes = plt.subplots(nr, nS, figsize=(11.5, 3.3 * nr + 0.6), squeeze=False)
    rows = []
    for r_, i in [(r_, i) for r_ in range(nr) for i in range(nS)]:
        ax = axes[r_, i]
        ax.axhline(0, color=INK, lw=1, ls="--", zorder=1)
        xmax = 0.0
        for name, lam, var, marker, ls in (entries[r_:r_ + 1] if grid else entries):
            x, y = var[1:, i], lam[1:, i] - refs[i]
            fit = window_fit(x, y, factor)
            if fit is None:
                ax.text(0.03, 0.96, f"{name}: fewer than 3 points in the window,\nno extrapolation", transform=ax.transAxes,
                        ha="left", va="top", fontsize=7.5, color=MUTED)
                rows.append((name, names[i], y[-1], float("nan"), float("nan"), x[-1], 0))
                ok = x > 0                           # still show where the run went
                if ok.any():
                    ax.plot(x[ok], y[ok], marker, color=SERIES[i], ms=2.5, mec="none", alpha=0.3)
                    ax.plot(x[-1], y[-1], marker, color=SERIES[i], ms=6, mec=INK, mew=0.6, zorder=5)
                    xmax = max(xmax, 1.05 * x[ok].max())
                    lo, hi = min(y[ok].min(), 0), max(y[ok].max(), 0)
                    ax.set_ylim(lo - 0.08 * (hi - lo + 1e-9), hi + 0.12 * (hi - lo + 1e-9))
                continue
            v, sel, lo, hi = fit["v"], fit["sel"], fit["lo"], fit["hi"]
            ctx = (x > 0) & (x <= 2 * hi)             # context: points up to twice the window's top, lighter
            ax.plot(x[ctx & ~sel], y[ctx & ~sel], marker, color=SERIES[i], ms=2.5, mec="none", alpha=0.3)
            wlabel = "[v, 2v]" if v > 0 else f"[0, {hi:.1e}] (last variance {v:.0e} < 0)"
            ax.plot(x[sel], y[sel], marker, color=SERIES[i], ms=3.2, mec="none",
                    label=f"{name}: {fit['k']} pts in {wlabel}")
            xs = np.linspace(0, hi, 50)
            ax.plot(xs, np.polyval(fit["lin"], xs), ls, color=SERIES[i], lw=1.1)
            ax.axvspan(lo, hi, color=GRID, alpha=0.5, lw=0, zorder=0)
            # error bar clipped to the panel (it can be far larger than the data range); the marker itself is not
            ax.errorbar(0, fit["e0"], yerr=0 if np.isnan(fit["half"]) else fit["half"], fmt="none", ecolor=SERIES[i],
                        capsize=3, zorder=5)
            ax.plot(0, fit["e0"], marker, color=SERIES[i], ms=6, mec=INK, mew=0.6, clip_on=False, zorder=6)
            rows.append((name, names[i], y[-1], fit["e0"], fit["half"], v, fit["k"]))
            xmax = max(xmax, 2 * hi)
            yy = y[ctx]
            lo, hi = min(yy.min(), fit["e0"], 0), max(yy.max(), fit["e0"], 0)
            ax.set_ylim(lo - 0.08 * (hi - lo + 1e-9), hi + 0.12 * (hi - lo + 1e-9))
        if xmax > 0:
            ax.set_xlim(0, xmax)
        ax.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
        ax.set_title(f"{title}: {names[i]}" + (f", {entries[r_][0]}" if grid else ""), fontsize=9, color=INK, loc="left")
        ax.set_xlabel("energy variance of this state (Ha²)")
        ax.tick_params(labelsize=7.5)
        style(ax)
        if i == 0:
            ax.set_ylabel("energy − exact (mHa)")
        if ax.get_legend_handles_labels()[0]:
            ax.legend(fontsize=7, loc="upper left")
    fig.suptitle("Per-state extrapolation: each block state's energy against its own variance\n"
                 "linear fit over [v, 2v] (shaded), v = the state's variance at the last iteration ([0, 2e-4] if v < 0); "
                 "marker at 0 = extrapolated energy, error bar = half the difference from a quadratic fit",
                 x=0.01, ha="left", fontsize=10, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.86 if nr == 1 else 0.94))
    return fig, rows


# ---------------------------------------------------------------- Pauli-propagation runs (run_block_pp.jl)
def pp_series(pp):
    """V_S reported by the run, raw and truncation-corrected block eigenvalues (mHa above FCI)"""
    h, f = pp["history"], pp["E_FCI"]
    return (np.array(h["VS"]), (np.array(h["lam"]) - f) * 1e3, (np.array(h["lam_corr"]) - f) * 1e3)


def replay_series(rep, pp):
    """true eigenvalues from replay_block.jl, placed at the run's own V_S of those iterations"""
    if rep is None:
        return None, None
    vs = np.array(pp["history"]["VS"])
    it = [r["iter"] for r in rep["rows"]]
    return vs[it], (np.array([r["lam"] for r in rep["rows"]]) - rep["E_FCI"]) * 1e3


def pp_energies(greedy, pp, rep, title):
    """the doc's energies-vs-variance figure for a truncated run: corrected (solid), raw (thin), true (dots)"""
    fig, ax = plt.subplots(figsize=(6.6, 4.4))
    _, _, refs, names, _ = run_data(greedy)
    vs, raw, cor = pp_series(pp)
    xt, tru = replay_series(rep, pp)
    lines = exact_lines(greedy, refs)
    top = max(cor[0].max(), max(e for e, _, _ in lines)) * 1.08
    for e, name, _ in lines:
        ax.axhline(e, color=INK, lw=0.9, ls="--", zorder=1)
        ax.annotate("exact " + name, xy=(1.01, e), xycoords=("axes fraction", "data"), fontsize=7.5, va="center",
                    color=INK)
    for i in range(cor.shape[1]):
        ax.plot(vs, cor[:, i], "-", color=SERIES[i], lw=1.6, zorder=3, label=names[i])
        ax.plot(vs, raw[:, i], "-", color=SERIES[i], lw=0.8, alpha=0.45, zorder=2)
        ax.plot(vs[0], cor[0, i], "o", color=SERIES[i], ms=5.5, mfc="white", mew=1.2, zorder=5)
        ax.plot(vs[-1], cor[-1, i], "o", color=SERIES[i], ms=5.5, mec="white", mew=0.8, zorder=5)
        if tru is not None:
            ax.plot(xt, tru[:, i], "o", color=SERIES[i], ms=2.6, mec="none", zorder=4)
    ax.set_xscale("log")
    ax.set_ylim(-0.06 * top, top)
    ax.set_title(title, fontsize=10, color=INK, loc="left")
    ax.set_xlabel("block variance $V_S$ of H (Ha², log)")
    ax.set_ylabel("energy (mHa above exact S0)")
    ax.legend(loc="upper left", fontsize=7.5, title="corrected block eigenvalues", title_fontsize=7.5, alignment="left")
    style(ax)
    fig.suptitle("Pauli propagation: block eigenvalues against block variance; exact energies dashed\n"
                 "solid = corrected for truncation, thin = raw, dots = true energy of the circuit (exact replay)",
                 x=0.01, ha="left", fontsize=10, color=INK)
    fig.tight_layout(rect=(0, 0, 0.9, 0.86))
    return fig


def pp_zoom(greedy, pps, title):
    """per state, zoomed on the exact energy: each truncated run's corrected (solid) and raw (thin) eigenvalues
    and its true energies (dots), plus the exact greedy run (grey).  pps: [(name, pp, rep, linestyle)]"""
    lam_g, vs_g, refs, names, _ = run_data(greedy)
    nS = lam_g.shape[1]
    fig, axes = plt.subplots(1, nS, figsize=(11, 3.6), squeeze=False)
    ws = []
    for i in range(nS):
        ax = axes[0, i]
        errs = [abs(lam_g[-1, i] - refs[i])]
        lo_x = vs_g.min()
        for name, pp, rep, ls in pps:
            vs, raw, cor = pp_series(pp)
            xt, tru = replay_series(rep, pp)
            ax.plot(vs, cor[:, i] - refs[i], ls=ls, color=SERIES[i], lw=1.6, zorder=3, label=f"{name}, corrected")
            ax.plot(vs, raw[:, i] - refs[i], ls=ls, color=SERIES[i], lw=0.8, alpha=0.45, zorder=2, label=f"{name}, raw")
            if tru is not None:
                ax.plot(xt, tru[:, i] - refs[i], "o", color=SERIES[i], ms=3, mec="none", zorder=4,
                        label=f"{name}, true (replay)")
                errs.append(abs(tru[-1, i] - refs[i]))
            errs += [abs(cor[-1, i] - refs[i]), abs(raw[-1, i] - refs[i])]
            lo_x = min(lo_x, vs.min())
        ax.plot(vs_g, lam_g[:, i] - refs[i], "-", color=MUTED, lw=1.2, zorder=2, label="exact greedy (no truncation)")
        ax.axhline(0, color=INK, lw=1, ls="--", zorder=1)
        ws.append(max(2.5 * max(errs), 0.05))
        ax.set_xscale("log")
        ax.set_xlim(lo_x / 1.5, 0.1)
        ax.set_title(f"{title}: {names[i]}", fontsize=8.5, color=INK, loc="left")
        ax.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(5))
        ax.ticklabel_format(axis="y", useOffset=False)
        ax.tick_params(labelsize=7.5)
        ax.set_xlabel("$V_S$ of H (Ha², log)")
        style(ax)
        if i == 0:
            ax.set_ylabel("energy − exact (mHa)")
    W = max(ws)                                       # same y range in every panel, for a fair comparison
    for ax in axes[0]:
        ax.set_ylim(-0.5 * W, 1.2 * W)
    axes[0, -1].legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize=7.5)
    fig.suptitle("Zoom on each state: error from that state's exact energy, Pauli propagation against the exact greedy run\n"
                 "starts on the right, converges to the left (x axis cut at V_S = 0.1)",
                 x=0.01, ha="left", fontsize=10, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.86))
    return fig


def greedy_vs_flow(methods, refs, names, title):
    """per state: block eigenvalue against V_S for greedy vDBF and the full flow (top row, energies with the
    exact energy dashed; bottom row, error on a log scale).  methods: [(name, data, colour, linestyle)]"""
    nS = len(refs)
    fig, axes = plt.subplots(2, nS, figsize=(4.0 * nS, 6.6), squeeze=False)
    for i in range(nS):
        top, bot = axes[0, i], axes[1, i]
        lo, hi = refs[i], refs[i]
        for name, d, colour, ls in methods:
            h = d["history"]
            lam = (np.array(h["lam"])[:, i] - d["E_FCI"]) * 1e3
            vs = np.array(h.get("VS_H", h["VS"]))
            keep = vs > 1e-14                     # converged flow rows add nothing on a log axis
            top.plot(vs[keep], lam[keep], ls, color=colour, lw=1.6, label=name)
            err = np.abs(lam[keep] - refs[i])
            bot.plot(vs[keep], np.maximum(err, 1e-9), ls, color=colour, lw=1.6, label=name)
            hi = max(hi, lam[0])
        top.axhline(refs[i], color=INK, lw=1, ls="--", zorder=1)
        top.set_ylim(lo - 0.05 * (hi - lo + 1), hi + 0.05 * (hi - lo + 1))
        top.set_title(f"{names[i]} (exact {abs(refs[i]) if abs(refs[i]) < 5e-3 else refs[i]:.2f} mHa, dashed)",
                      fontsize=9.5, color=INK, loc="left")
        bot.set_title(f"{names[i]}: absolute error", fontsize=9.5, color=INK, loc="left")
        for ax in (top, bot):
            ax.set_xscale("log")
            ax.set_xlabel("block variance $V_S$ of H (Ha², log)")
            style(ax)
        top.set_xlim(2e-6, 3)                    # where the methods differ; the bottom row shows the full range
        bot.set_yscale("log")
        bot.set_ylim(1e-6, 1e3)
        bot.axhline(1.6, color=MUTED, lw=0.8, ls=":")
        if i == 0:
            top.set_ylabel("energy (mHa above exact S0)")
            bot.set_ylabel("abs(block eigenvalue − exact) (mHa)")
            top.legend(fontsize=7.5, loc="upper left")
    fig.suptitle(f"{title}: greedy vDBF against the full double-bracket flow, both exact (no truncation)\n"
                 "each method starts on the right ($V_S$ = 0.93) and converges to the left; top row cropped to $V_S$ > 2e-6; dotted line = 1.6 mHa",
                 x=0.01, ha="left", fontsize=10, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    return fig


# ---------------------------------------------------------------- S^2 penalty at 1.1 A (sv_block.jl --s2-penalty)
PEN = ["0.01", "0.02", "0.05", "0.1"]
PEN_COL = {"0": INK, "0.01": "#9ec5f4", "0.02": "#5b9de6", "0.05": "#2a78d6", "0.1": "#16457f"}   # one hue, light -> dark


def penalty_runs():
    """[(λ string, run)] for the bare-H greedy run and the H + λ S^2 runs at 1.1 A that exist"""
    out = [("0", load("uno_1.10_block1c_exact"))]
    out += [(l, load(f"uno_1.10_block1c_exact_s2pen{l}")) for l in PEN]
    return [(l, d) for l, d in out if d is not None]


def penalty_figures(runs):
    """1.1 A: error of each block state (from the exact level with its spin), one line per λ, panels per state.
    Figure 1: against iteration, with a second row of the ratio to the bare-H run.  Figure 2: against V_S of H
    (starts on the right, converges to the left).  x and y ranges shared by all panels of a row."""
    data = [(l, d, run_data(d)) for l, d in runs]
    base = next(r for l, d, r in data if l == "0")
    rows = []
    f1, ax1 = plt.subplots(2, 3, figsize=(11.5, 6.0), squeeze=False, gridspec_kw=dict(height_ratios=[1.6, 1]))
    f2, ax2 = plt.subplots(1, 3, figsize=(11.5, 3.7), squeeze=False)
    lo_y, hi_y = np.inf, 0.0
    for l, d, (lam, vs, refs, names, s2) in data:
        it = np.arange(len(lam))
        for i in range(3):
            err = lam[:, i] - refs[i]
            kw = dict(color=PEN_COL[l], lw=1.7 if l == "0" else 1.3, ls=(0, (4, 2)) if l == "0" else "-",
                      zorder=4 if l == "0" else 3, label=("bare H" if l == "0" else f"λ = {l}"))
            ax1[0, i].plot(it, np.abs(err), **kw)
            ax2[0, i].plot(vs, np.abs(err), **kw)
            if l != "0":
                b = base[0][:, i] - base[2][i]
                n = min(len(b), len(err))
                ax1[1, i].plot(it[1:n], err[1:n] / b[1:n], **kw)
            lo_y, hi_y = min(lo_y, abs(err[-1])), max(hi_y, abs(err[min(3, len(err) - 1)]))
            rows.append((l, i, names[i], s2[i], err[-1],
                         *[next((int(it[k]) for k in range(len(err)) if np.all(np.abs(err[k:]) < t)), None)
                           for t in (1.6, 0.1, 0.03)]))
    slot = ["S0 (|0⟩ sector)", "³Πg (|a⟩, |b⟩ sector)", "¹Πg (|a⟩, |b⟩ sector)"]
    for i in range(3):
        for ax in (ax1[0, i], ax2[0, i]):
            ax.set_yscale("log")
            ax.set_ylim(lo_y / 2, hi_y * 1.5)
            ax.set_title(slot[i], fontsize=9, color=INK, loc="left")
            ax.tick_params(labelsize=7.5)
            style(ax)
        ax1[0, i].legend(fontsize=7, loc="upper right")
        ax2[0, i].legend(fontsize=7, loc="lower right")
        ax1[1, i].axhline(1, color=INK, lw=1, ls=(0, (4, 2)))
        ax1[1, i].set_ylim(0.85, 1.15)
        ax1[1, i].set_xlabel("iteration (50 rotations each)")
        ax1[1, i].tick_params(labelsize=7.5)
        style(ax1[1, i])
        ax2[0, i].set_xscale("log")
        ax2[0, i].set_xlabel("block variance $V_S$ of H (Ha², log)")
    ax1[0, 0].set_ylabel("energy − exact (mHa, log)")
    ax1[1, 0].set_ylabel("error / error of bare H")
    ax2[0, 0].set_ylabel("energy − exact (mHa, log)")
    f1.suptitle("1.1 Å, greedy vDBF (exact): flow on H + λ S² against bare H (dashed), energies of H\n"
                "top: error of each block state from its exact level; bottom: that error divided by the bare-H error "
                "at the same iteration (below 1 = penalty ahead)", x=0.01, ha="left", fontsize=10, color=INK)
    f1.subplots_adjust(left=0.07, right=0.985, bottom=0.09, top=0.88, wspace=0.22, hspace=0.12)
    for ax in ax1[0]:
        ax.tick_params(labelbottom=False)
    f2.suptitle("1.1 Å, the same runs against the block variance of H: starts on the right, converges to the left",
                x=0.01, ha="left", fontsize=10, color=INK)
    f2.subplots_adjust(left=0.07, right=0.985, bottom=0.15, top=0.84, wspace=0.22)
    return (f1, f2), rows


def penalty_230():
    """2.3 A, where the start is spin-contaminated: S0's energy above FCI and its <S^2> against iteration,
    bare H against λ = 0.05 and 0.1"""
    runs = [("0", load("uno_2.30_block1c_exact")), ("0.05", load("uno_2.30_block1c_exact_s2pen0.05")),
            ("0.1", load("uno_2.30_block1c_exact_s2pen0.1"))]
    runs = [(l, d) for l, d in runs if d is not None]
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 3.6))
    for l, d in runs:
        h, f = d["history"], d["E_FCI"]
        e = (np.array(h["lam"])[:, 0] - f) * 1e3
        s2 = np.array(h["S2"])[:, 0]
        kw = dict(color=PEN_COL[l], lw=1.7 if l == "0" else 1.4, ls=(0, (4, 2)) if l == "0" else "-",
                  label="bare H" if l == "0" else f"λ = {l}")
        axes[0].plot(np.arange(len(e)), np.maximum(e, 1e-6), **kw)
        axes[1].plot(np.arange(len(s2)), s2, **kw)
    axes[0].set_yscale("log")
    axes[0].set_ylabel("S0 energy − FCI (mHa, log)")
    axes[1].set_ylabel("⟨S²⟩ of the S0 block state")
    axes[1].axhline(0, color=INK, lw=0.8)
    for ax in axes:
        ax.set_xlabel("iteration (50 rotations each)")
        ax.legend(fontsize=7.5, loc="upper right")
        ax.tick_params(labelsize=7.5)
        style(ax)
    fig.suptitle("2.3 Å (stretched, spin-contaminated start): the S0 block state with and without the penalty",
                 x=0.01, ha="left", fontsize=10, color=INK)
    fig.subplots_adjust(left=0.08, right=0.985, bottom=0.15, top=0.86, wspace=0.25)
    return fig


# ---------------------------------------------------------------- Pauli propagation on H + λ S^2 at 1.1 A
SS1 = np.array([0.0, 2.0, 0.0])                       # S(S+1) of the targets S0, 3Πg, 1Πg


def pp_pen_series(pp, rep, lam_s2):
    """a PP block run on H + λ S^2 (or bare H, λ = 0) and its replay, as per-state arrays in the order
    (S0, 3Πg, 1Πg): estimates = flow-operator eigenvalue - λ S(S+1) (mHa above FCI), variances of the flow
    operator.  The order per iteration comes from the replay's <S^2> (the triplet is the state with the
    largest <S^2>, S0 the lowest); bare-H runs keep the sorted order, where 3Πg < 1Πg already."""
    h, f = pp["history"], pp["E_FCI"]
    n = len(h["lam"])
    order = np.tile(np.arange(3), (n, 1))
    if lam_s2 != 0:
        rows = {r["iter"]: r for r in rep["rows"]}
        last = None
        for it in range(n - 1, -1, -1):                # iterations without a replay row take the next row's order
            r = rows.get(it)
            if r is not None:
                lam = np.array(r["lam"])
                s0 = int(np.argmin(lam))
                rest = [k for k in range(3) if k != s0]
                t = max(rest, key=lambda k: r["S2"][k])
                last = [s0, t, [k for k in rest if k != t][0]]
            if last is not None:
                order[it] = last
    pick = lambda a: np.take_along_axis(np.asarray(a), order, axis=1)
    shift = lam_s2 * 1e3 * SS1
    out = dict(vs=np.array(h["VS"]), it=np.arange(n),
               raw=(pick(h["lam"]) - f) * 1e3 - shift, cor=(pick(h["lam_corr"]) - f) * 1e3 - shift,
               var=pick(h["var"]), var_corr=pick(h["var_corr"]))
    if rep is not None:
        its = np.array([r["iter"] for r in rep["rows"]])
        o = order[its]
        tl = lambda key: np.take_along_axis(np.array([r[key] for r in rep["rows"]]), o, axis=1)
        out.update(t_it=its, tru=(tl("lam") - rep["E_FCI"]) * 1e3 - shift, t_var=tl("var"),
                   t_vs=np.array([r["VS"] for r in rep["rows"]]))
        if lam_s2 != 0:
            out.update(tru_H=(tl("lam_H") - rep["E_FCI"]) * 1e3, t_S2=tl("S2"))
    return out


def pp_penalty_trajectories(runs, refs, names):
    """per state, error against that state's own variance (the plane the extrapolation is done in),
    one row per estimate (true, corrected, raw), one line per run; x and y shared within a row"""
    kinds = [("true (exact replay)", "tru", "t_var"), ("corrected", "cor", "var_corr"), ("raw", "raw", "var")]
    fig, axes = plt.subplots(3, 3, figsize=(11.5, 9.0), squeeze=False)
    for r_, (kname, ek, vk) in enumerate(kinds):
        X, Y = 0.0, [0.0, 0.0]
        for lab, col, ls, s in runs:
            if ek not in s:
                continue
            for i in range(3):
                x, y = s[vk][1:, i], s[ek][1:, i] - refs[i]
                k = x > 0
                axes[r_, i].plot(x[k], y[k], ls=ls, color=col, lw=1.4 if ek != "tru" else 1.1,
                                 marker="o" if ek == "tru" else None, ms=2.2, label=lab)
                late = slice(len(x) // 3, None)          # ranges from the last two thirds of each run
                X = max(X, x[late][x[late] > 0].max())
                Y = [min(Y[0], y[late].min()), max(Y[1], y[late].max())]
        pad = 0.08 * (Y[1] - Y[0])
        for i in range(3):
            ax = axes[r_, i]
            ax.axhline(0, color=INK, lw=0.9, ls=(0, (4, 2)), zorder=1)
            ax.set_xlim(0, X * 1.05)
            ax.set_ylim(Y[0] - pad, Y[1] + pad)
            ax.set_title(f"{kname}: {names[i]}", fontsize=9, color=INK, loc="left")
            ax.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
            ax.tick_params(labelsize=7.5)
            style(ax)
            if r_ == 2:
                ax.set_xlabel("variance of this state (Ha²)")
        axes[r_, 0].set_ylabel("energy − exact (mHa)")
        axes[r_, 2].legend(fontsize=7, loc="upper left")
    fig.suptitle("1.1 Å, Pauli propagation at cutoff 10⁻⁵: each state's energy against its own variance, without and "
                 "with the S² penalty\npenalty runs: eigenvalue of H + λS² minus λS(S+1), variance of H + λS²; "
                 "x and y shared within a row; the run goes from right to left towards (0, 0)",
                 x=0.01, ha="left", fontsize=10, color=INK)
    fig.subplots_adjust(left=0.07, right=0.985, bottom=0.06, top=0.9, wspace=0.22, hspace=0.35)
    return fig


# ---------------------------------------------------------------- Pauli propagation on H + λ S^2 at 2.3 A
def pp_penalty_230():
    """2.3 A, Pauli propagation at cutoff 1e-5, bare H against H + λ S^2, against iteration.  The three block
    states are kept in the run's own energy order (with the penalty that order is S0 < T1 < triplet).
    Rows: the true energy of H of the circuit's states (exact replay), the run's corrected estimate (penalty runs:
    eigenvalue of H + λS² minus λS(S+1) of the target, S(S+1) = 0, 2, 2), and the circuit's <S^2>.
    Energies in mHa above exact S0, exact levels dashed, the exact greedy run on H + 0.05 S² (no truncation) in grey.
    Returns (fig, rows), rows = (run, state, best true error, its iteration, true error at the end,
    corrected error at the end, <S^2> at the end), errors from the run's own target level."""
    runs = []
    for l in ("0", "0.02", "0.05"):
        lab = "uno_2.30_blockpp_c1e-5" + ("" if l == "0" else f"_s2pen{l}")
        pp, rep = load(lab), load(lab + "_replay")
        if pp is None or rep is None:
            continue
        lam, f, rr = float(l), pp["E_FCI"], rep["rows"]
        shift = lam * 1e3 * np.array([0.0, 2.0, 2.0])
        runs.append(dict(l=l, it=np.array([r["iter"] for r in rr]), s2=np.array([r["S2"] for r in rr]),
                         tru=(np.array([r["lam_H" if lam else "lam"] for r in rr]) - f) * 1e3,
                         cor=(np.array(pp["history"]["lam_corr"]) - f) * 1e3 - shift))
    if not runs:
        return None, []
    tg = {}
    for key, lab in (("0", "uno_2.30_block1c_exact"), ("pen", "uno_2.30_block1c_exact_s2pen0.05")):
        d = load(lab)
        tg[key] = (np.array([t["E"] for t in d["targets"]]) - d["E_FCI"]) * 1e3
    ex = load("uno_2.30_block1c_exact_s2pen0.05")
    ex_lam = (np.array(ex["history"]["lam"]) - ex["E_FCI"]) * 1e3
    ex_s2 = np.array(ex["history"]["S2"])
    levels = [[(tg["0"][0], "S0")], [(tg["0"][1], "T1")],
              [(tg["0"][2], "septet (bare-H target)"), (tg["pen"][2], "triplet (penalty target)")]]
    s2_lines = [[0.0], [2.0], [2.0]]
    fig, axes = plt.subplots(3, 3, figsize=(11.5, 8.8), squeeze=False, gridspec_kw=dict(height_ratios=[1.3, 1.3, 1]))
    title = ["lowest block state (target S0)", "second (target T1)", "third (target: see the dashed lines)"]
    col = dict(PEN_COL, **{"0.05": PEN_COL["0.1"]})     # only 0.02 and 0.05 here: spread them further apart
    rows, lo, hi = [], 0.0, 0.0
    for i in range(3):
        g = dict(color=MUTED, lw=1.1, alpha=0.7, label="no truncation, λ = 0.05")
        for r_ in (0, 1):
            axes[r_, i].plot(np.arange(len(ex_lam)), ex_lam[:, i], **g)
        axes[2, i].plot(np.arange(len(ex_s2)), ex_s2[:, i], **g)
        for s in runs:
            kw = dict(color=col[s["l"]], lw=1.6 if s["l"] == "0" else 1.3, ls=(0, (4, 2)) if s["l"] == "0" else "-",
                      label="bare H" if s["l"] == "0" else f"λ = {s['l']}")
            axes[0, i].plot(s["it"], s["tru"][:, i], **kw)
            axes[1, i].plot(np.arange(len(s["cor"])), s["cor"][:, i], **kw)
            axes[2, i].plot(s["it"], s["s2"][:, i], **kw)
            late = s["it"] >= 20
            lo = min(lo, s["tru"][late, i].min(), s["cor"][20:, i].min())
            hi = max(hi, s["tru"][late, i].max(), s["cor"][20:, i].max())
            t = tg["0" if s["l"] == "0" else "pen"][i]
            err = s["tru"][1:, i] - t
            k = int(np.argmin(err))
            rows.append(("bare H" if s["l"] == "0" else f"λ = {s['l']}", i, err[k], int(s["it"][1:][k]), err[-1],
                         s["cor"][-1, i] - t, s["s2"][-1, i]))
        for e, name in levels[i]:
            for r_ in (0, 1):
                axes[r_, i].axhline(e, color=INK, lw=0.9, ls=(0, (4, 2)), zorder=1)
                axes[r_, i].annotate(name, xy=(0.99, e), xycoords=("axes fraction", "data"), xytext=(0, 2),
                                     textcoords="offset points", ha="right", va="bottom", fontsize=7, color=INK)
        for v in s2_lines[i]:
            axes[2, i].axhline(v, color=INK, lw=0.9, ls=(0, (4, 2)), zorder=1)
    pad = 0.06 * (hi - lo)
    for i in range(3):
        for r_ in range(3):
            ax = axes[r_, i]
            ax.set_xlim(0, 500)
            ax.tick_params(labelsize=7.5)
            style(ax)
            if r_ < 2:
                ax.set_ylim(lo - pad, hi + pad)
        axes[0, i].set_title(title[i], fontsize=9, color=INK, loc="left")
        axes[2, i].set_ylim(-0.2, 6.5)
        axes[2, i].set_xlabel("iteration (50 rotations each)")
    axes[0, 0].set_ylabel("true energy of H\n(mHa above exact S0)")
    axes[1, 0].set_ylabel("corrected estimate\n(mHa above exact S0)")
    axes[2, 0].set_ylabel("true ⟨S²⟩")
    axes[0, 0].legend(fontsize=7.5, loc="upper right")
    fig.suptitle("2.3 Å, Pauli propagation at cutoff 10⁻⁵, without and with the S² penalty, against iteration\n"
                 "top: true energy of the circuit's block states (exact replay); middle: the run's corrected estimate; "
                 "bottom: the circuit's ⟨S²⟩ (septet = 12, off scale); exact levels dashed",
                 x=0.01, ha="left", fontsize=10, color=INK)
    fig.subplots_adjust(left=0.08, right=0.985, bottom=0.06, top=0.9, wspace=0.2, hspace=0.25)
    return fig, rows


def pp230_extrapolation():
    """2.3 A, cutoff 1e-5: the per-state [v, 2v] extrapolation (state_extrapolation, one row per series) for each
    run.  True = H energy and H variance of the circuit's block states (exact replay); corrected and raw =
    eigenvalue of H + λS² minus λS(S+1) of the target (0, 2, 2), with the run's variances of H + λS².
    States in the run's energy order, errors from each run's own targets.  Returns [(λ string, fig, rows)]."""
    tg = {}
    for key, lab in (("0", "uno_2.30_block1c_exact"), ("pen", "uno_2.30_block1c_exact_s2pen0.05")):
        d = load(lab)
        tg[key] = (np.array([t["E"] for t in d["targets"]]) - d["E_FCI"]) * 1e3
    out = []
    for l in ("0", "0.02", "0.05"):
        lab = "uno_2.30_blockpp_c1e-5" + ("" if l == "0" else f"_s2pen{l}")
        pp, rep = load(lab), load(lab + "_replay")
        if pp is None or rep is None:
            continue
        lam, f, h, rr = float(l), pp["E_FCI"], pp["history"], rep["rows"]
        shift = lam * 1e3 * np.array([0.0, 2.0, 2.0])
        kE, kV = ("lam_H", "var_H") if lam else ("lam", "var")
        ent = [("true (exact replay)", (np.array([r[kE] for r in rr]) - f) * 1e3, np.array([r[kV] for r in rr]), "o", "-"),
               ("corrected", (np.array(h["lam_corr"]) - f) * 1e3 - shift, np.array(h["var_corr"]), "^", "--"),
               ("raw", (np.array(h["lam"]) - f) * 1e3 - shift, np.array(h["var"]), "v", ":")]
        name = "bare H" if l == "0" else f"λ = {l}"
        fig, rows = state_extrapolation(ent, tg["0" if l == "0" else "pen"], ["S0", "T1", "third"],
                                        f"2.3 Å, {name}", grid=True)
        out.append((l, fig, rows))
    return out


# ---------------------------------------------------------------- subspace fidelity (all N2 block runs)
def level_names(targets, f):
    """'singlet 0, triplet 2.84, ...' for the fidelity targets, degenerate levels merged (×2)"""
    groups = []
    for t in targets:
        if groups and abs(t["Eopt"] - groups[-1][0]["Eopt"]) < 1e-6:
            groups[-1].append(t)
        else:
            groups.append([t])
    out = []
    for g in groups:
        e = (g[0]["E"] - f) * 1e3
        name = SPIN.get(round(g[0]["S2"]), "?") + (" S0" if abs(e) < 1e-6 else f" {e:.2f}")
        out.append(name + (f" ×{len(g)}" if len(g) > 1 else ""))
    return groups, out


def fidelity_figure():
    """F = |P_low P_block|^2 / 3 against iteration, P_low the 3 lowest levels of the flow operator over all
    symmetry sectors (fid in sv_block.jl and replay_block.jl).  Rows: 1.1 and 2.3 A; columns: H, H + 0.02 S²,
    H + 0.05 S².  Exact greedy run and the Pauli-propagation circuits (exact replay).  Returns (fig, rows),
    rows = (R, λ, run, F at the end, largest F, weight of each lowest level in the block at the end, best F)."""
    pts, lams = ("1.10", "2.30"), ("0", "0.02", "0.05")
    series = [("exact greedy (no truncation)", "block1c_exact", INK, "-", 1.7),
              ("Pauli prop. 10⁻⁵, true circuit", "blockpp_c1e-5", SERIES[0], "-", 1.4),
              ("Pauli prop. 10⁻⁶, true circuit", "blockpp_c1e-6", SERIES[1], (0, (5, 2)), 1.4)]
    fig, axes = plt.subplots(2, 3, figsize=(11.5, 6.6), sharey=True)
    rows = []
    for r_, R in enumerate(pts):
        for c_, l in enumerate(lams):
            ax = axes[r_, c_]
            suf = "" if l == "0" else f"_s2pen{l}"
            info = None
            for name, kind, col, ls, lw in series:
                lab = f"uno_{R}_{kind}{suf}"
                if kind.startswith("blockpp"):
                    d = load(lab + "_replay1") or load(lab + "_replay")
                    if d is None or "fid" not in d["rows"][0]:
                        continue
                    it = np.array([r["iter"] for r in d["rows"]])
                    F = np.array([r["fid"] for r in d["rows"]])
                    w = np.array(d["rows"][-1]["fid_states"])
                else:
                    d = load(lab)
                    if d is None or "fid" not in d["history"]:
                        continue
                    F = np.array(d["history"]["fid"])
                    it = np.arange(len(F))
                    w = np.array(d["history"]["fid_states"][-1])
                info = info or d
                ax.plot(it, F, color=col, ls=ls, lw=lw, label=name)
                groups, _ = level_names(d["fid_targets"], d["E_FCI"])
                k, ws = 0, []
                for g in groups:
                    ws.append(w[k:k + len(g)].sum())
                    k += len(g)
                rows.append((R, l, name, F[-1], F.max(), ws, d["fid_best"]))
            ax.axhline(0.9, color=MUTED, lw=0.9, ls=(0, (4, 2)), zorder=1)
            ax.annotate("0.9", xy=(1.0, 0.9), xycoords=("axes fraction", "data"), xytext=(-2, 2),
                        textcoords="offset points", ha="right", va="bottom", fontsize=7, color=MUTED)
            if info is not None:
                best = info["fid_best"]
                fin = [r for r in rows if r[0] == R and r[1] == l]
                if best > 0.999 and fin and max(r[3] for r in fin) < 0.7:
                    ax.annotate("stuck at 2/3: the block holds only one of the ×2 pair", xy=(1.0, 0.62),
                                xycoords="axes fraction", xytext=(-2, 0), textcoords="offset points", ha="right",
                                va="top", fontsize=7, color=INK)
                if best < 0.999:
                    ax.axhline(best, color=INK, lw=0.9, ls=":", zorder=1)
                    ax.annotate(f"most these block states can reach: {best:.2f}", xy=(1.0, best),
                                xycoords=("axes fraction", "data"), xytext=(-2, -3), textcoords="offset points",
                                ha="right", va="top", fontsize=7, color=INK)
                _, names = level_names(info["fid_targets"], info["E_FCI"])
                op = "H" if l == "0" else f"H + {l} S²"
                ax.set_title(f"{R.rstrip('0')} Å, {op}\nlowest 3 (mHa): " + ", ".join(names), fontsize=8.5,
                             color=INK, loc="left")
            ax.set_xlim(0, 200 if R == "1.10" else 500)
            ax.set_ylim(0, 1.02)
            ax.tick_params(labelsize=7.5)
            style(ax)
            if r_ == 1:
                ax.set_xlabel("iteration (50 rotations each)")
        axes[r_, 0].set_ylabel("fidelity F with the 3 lowest levels")
    axes[0, 0].legend(fontsize=7.5, loc="lower right")
    fig.suptitle("N₂: is the block spanning the 3 lowest exact levels?  "
                 r"$F = \|P_{\rm low}\,P_{\rm block}\|^2/3$, with $P_{\rm low}$ over all symmetry sectors of the flow "
                 "operator (H, or H + λS²)\nthe flow never leaves a block state's symmetry sector, so F is capped at "
                 "(lowest levels the block's sectors can hold) / 3, dotted",
                 x=0.01, ha="left", fontsize=10, color=INK)
    fig.subplots_adjust(left=0.07, right=0.985, bottom=0.08, top=0.83, wspace=0.12, hspace=0.45)
    return fig, rows


def main():
    os.makedirs(OUT, exist_ok=True)
    # greedy vDBF vs the full flow at 1.1 A, against the same exact levels
    g = [("greedy vDBF, exact ranking", "uno_1.10_block1c_exact", SERIES[0], "-"),
         ("greedy vDBF, D1c ranking", "uno_1.10_block1c_d1c", SERIES[1], "-"),
         ("full flow, D = block projector", "uno_1.10_flow1c_projector", SERIES[2], "--"),
         ("full flow, D = D1c", "uno_1.10_flow1c_d1c", SERIES[3], "--")]
    methods = [(n, load(l), c, ls) for n, l, c, ls in g if load(l) is not None]
    if len(methods) > 1 and methods[0][1] is not None:
        _, _, refs, names, _ = run_data(load("uno_1.10_block1c_exact"))
        greedy_vs_flow(methods, refs, ["S0", "³Πg", "¹Πg"], "N2, 1.1 Å").savefig(
            os.path.join(OUT, "n2_110_greedy_vs_flow.png"), dpi=170)
    all_runs = {}
    for label in ["uno_1.10_block1c_exact", "uno_1.10_block1c_d1c", "uno_2.30_block1c_exact", "uno_2.30_block1c_d1c",
                  "uno_2.30_block1c_exact_s2pen0.05", "uno_2.30_block1c_exact_s2pen0.1"]:
        d = load(label)
        if d is not None:
            all_runs[label] = d
    runs = [(l, t, all_runs[l]) for l, t in RUNS if l in all_runs]
    energies_vs_variance(runs).savefig(os.path.join(OUT, "energies_vs_variance.png"), dpi=170)
    energies_zoom(runs)[0].savefig(os.path.join(OUT, "energies_zoom.png"), dpi=170)
    r110 = [r for r in runs if r[0] == "uno_1.10_block1c_exact"]       # 1.1 A alone, for the explainer
    if r110:
        energies_vs_variance(r110).savefig(os.path.join(OUT, "n2_110_energies.png"), dpi=170)
        fig, w110, x110 = energies_zoom(r110)
        fig.savefig(os.path.join(OUT, "n2_110_zoom.png"), dpi=170)
        # the same two figures for the full double-bracket flow; zoom panels use the greedy figure's windows
        for tag, title in (("projector", "1.1 Å, full flow, D = block projector"), ("d1c", "1.1 Å, full flow, D = D1c")):
            d = load(f"uno_1.10_flow1c_{tag}")
            if d is None:
                continue
            fl = [(f"uno_1.10_flow1c_{tag}", title, d)]
            energies_vs_variance(fl).savefig(os.path.join(OUT, f"n2_110_flow_{tag}_energies.png"), dpi=170)
            energies_zoom(fl, w110, x110)[0].savefig(os.path.join(OUT, f"n2_110_flow_{tag}_zoom.png"), dpi=170)

        # both on the same axes: greedy (exact ranking) and the full flow, with section 4's D1c when available
        flow = load("uno_1.10_flow1c_d1c") or load("uno_1.10_flow1c_projector")
        if flow is not None:
            compare_energies(r110[0][2], flow, "1.1 Å").savefig(os.path.join(OUT, "n2_110_compare_energies.png"), dpi=170)
            compare_zoom(r110[0][2], flow, "1.1 Å", w110, x110).savefig(os.path.join(OUT, "n2_110_compare_zoom.png"),
                                                                      dpi=170)
        # per-state extrapolation: exact greedy, plus the Pauli-propagation runs (corrected energy vs corrected
        # variance, as in the ground-state protocol) when they have per-state variances
        g = r110[0][2]
        _, _, refs, _, _ = run_data(g)
        sn = ["S0", "³Πg", "¹Πg"]
        ent = [("exact greedy", (np.array(g["history"]["lam"]) - g["E_FCI"]) * 1e3, np.array(g["history"]["var"]), "o", "-")]
        fig, xrows = state_extrapolation(ent, refs, sn, "1.1 Å")
        fig.savefig(os.path.join(OUT, "n2_110_extrap_exact.png"), dpi=170)
        for cut, mk, ls in (("1e-5", "^", "--"), ("1e-6", "s", "-.")):
            pp = load(f"uno_1.10_blockpp_c{cut}")
            if pp is not None and "var_corr" in pp["history"]:
                h = pp["history"]
                ent.append((f"Pauli prop. {cut}", (np.array(h["lam_corr"]) - pp["E_FCI"]) * 1e3, np.array(h["var_corr"]), mk, ls))
        if len(ent) > 1:
            fig, xrows = state_extrapolation(ent, refs, sn, "1.1 Å")
            fig.savefig(os.path.join(OUT, "n2_110_extrap_all.png"), dpi=170)
        print("per-state extrapolation over [v, 2v] (mHa vs exact): run | state | last | extrapolated ± | v | points")
        for r in xrows:
            print(f"  {r[0]:<18} {r[1]:<5} {r[2]:+9.4f} {r[3]:+9.4f} ± {r[4]:.4f}   {r[5]:.2e}  {r[6]}")
        # Pauli propagation at 1.1 A: the doc's figures per cutoff, then the cutoffs together
        pps = []
        for cut, ls in (("1e-5", "-"), ("1e-6", (0, (5, 2)))):
            pp = load(f"uno_1.10_blockpp_c{cut}")
            if pp is None:
                continue
            rep = load(f"uno_1.10_blockpp_c{cut}_replay")
            pps.append((f"cutoff {cut}", pp, rep, ls))
            pp_energies(r110[0][2], pp, rep, f"1.1 Å, Pauli propagation, cutoff {cut}").savefig(
                os.path.join(OUT, f"n2_110_pp_c{cut}_energies.png"), dpi=170)
            pp_zoom(r110[0][2], [pps[-1]], "1.1 Å").savefig(os.path.join(OUT, f"n2_110_pp_c{cut}_zoom.png"), dpi=170)
            if "var_corr" in pp["history"]:          # per-state extrapolation: corrected, raw, and the true circuit
                h = pp["history"]
                ent = []
                if rep is not None and "var" in rep["rows"][0]:
                    ent.append(("true (exact replay)", (np.array([r["lam"] for r in rep["rows"]]) - rep["E_FCI"]) * 1e3,
                                np.array([r["var"] for r in rep["rows"]]), "o", "-"))
                ent += [("corrected", (np.array(h["lam_corr"]) - pp["E_FCI"]) * 1e3, np.array(h["var_corr"]), "^", "--"),
                        ("raw", (np.array(h["lam"]) - pp["E_FCI"]) * 1e3, np.array(h["var"]), "v", ":")]
                fig, prow = state_extrapolation(ent, refs, sn, f"PP {cut}", grid=True)
                fig.savefig(os.path.join(OUT, f"n2_110_pp_c{cut}_extrap.png"), dpi=170)
                print(f"Pauli propagation {cut}: per-state extrapolation over [v, 2v] (mHa vs exact)")
                for r in prow:
                    print(f"  {r[0]:<20} {r[1]:<5} {r[2]:+9.4f} {r[3]:+9.4f} ± {r[4]:.4f}   v {r[5]:.2e}  {r[6]} pts")
        if len(pps) > 1:
            pp_zoom(r110[0][2], pps, "1.1 Å").savefig(os.path.join(OUT, "n2_110_pp_compare_zoom.png"), dpi=170)
    error_vs_variance(runs).savefig(os.path.join(OUT, "error_vs_variance.png"), dpi=170)
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


def kets_figure():
    """Old block {|0>, |a>, |b>} against new kets chosen from the sectors (and spins) of the 3 lowest levels
    (sv_block.jl --kets): F against iteration, one panel per bond length.  Returns (fig, rows), rows = (R, run,
    λ, F at the end, first iteration with F > 0.9, block energies of H at the end (mHa above S0), <S^2> at the end,
    the 3 lowest levels of the flow operator (E of H, mHa above S0))."""
    pts = [("1.10", "k45_47", "|0⟩, 4→5, 4→7", ["0", "0.05"]), ("2.30", "k3456_35", "|0⟩, 3,4→5,6, 3→5", ["0", "0.02"])]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    rows = []
    for ax, (R, tag, kets, lams) in zip(axes, pts):
        for which, lab, col in (("old", f"uno_{R}_block1c_exact", INK), ("new", f"uno_{R}_block_{tag}", SERIES[0])):
            for l in lams:
                d = load(lab + ("" if l == "0" else f"_s2pen{l}"))
                if d is None or "fid" not in d["history"]:
                    continue
                F = np.array(d["history"]["fid"])
                name = ("|0⟩, |a⟩, |b⟩" if which == "old" else kets) + ("" if l == "0" else f", H + {l} S²")
                ax.plot(np.arange(len(F)), F, color=col, lw=1.6, ls="-" if l == "0" else (0, (4, 2)), label=name)
                f = d["E_FCI"]
                hit = np.nonzero(F > 0.9)[0]
                rows.append((R, which, l, F[-1], int(hit[0]) if len(hit) else None,
                             (np.array(d["history"]["lam"][-1]) - f) * 1e3, np.array(d["history"]["S2"][-1]),
                             [(t["E"] - f) * 1e3 for t in d["fid_targets"]]))
        ax.axhline(0.9, color=MUTED, lw=0.9, ls=(0, (4, 2)), zorder=1)
        ax.set_xlim(0, 200 if R == "1.10" else 500)
        ax.set_ylim(0, 1.02)
        ax.set_title(f"{R.rstrip('0')} Å", fontsize=10, color=INK, loc="left")
        ax.set_xlabel("iteration (50 rotations each)")
        ax.legend(fontsize=7.5, loc="lower right")
        ax.tick_params(labelsize=7.5)
        style(ax)
    axes[0].set_ylabel("fidelity F with the 3 lowest levels")
    fig.suptitle("Exact greedy block DBF: the old kets |0⟩, |a⟩, |b⟩ (black) against kets chosen from the sectors of the "
                 "3 lowest levels (blue)\nqubits flipped from the reference, all α; F against the 3 lowest levels of the "
                 "flow operator (H, or H + λS²); dashed grey: 0.9", x=0.01, ha="left", fontsize=10, color=INK)
    fig.subplots_adjust(left=0.07, right=0.985, bottom=0.13, top=0.82, wspace=0.08)
    return fig, rows


def fidelity_main():
    os.makedirs(OUT, exist_ok=True)
    fig, rows = fidelity_figure()
    fig.savefig(os.path.join(OUT, "n2_fidelity.png"), dpi=170)
    print("fidelity with the 3 lowest levels: R | λ | run | F at the end | largest F | weight of each lowest level "
          "in the block at the end | most F these block states allow")
    for r in rows:
        print(f"  {r[0]} {r[1]:<5} {r[2]:<32} {r[3]:.3f}  {r[4]:.3f}  [{', '.join(f'{x:.3f}' for x in r[5])}]  {r[6]:.3f}")
    fig, rows = kets_figure()
    fig.savefig(os.path.join(OUT, "n2_fidelity_kets.png"), dpi=170)
    print("old against new kets: R | kets | λ | F at the end | first iteration F > 0.9 | block energies (mHa above S0) "
          "| <S^2> | 3 lowest levels")
    for r in rows:
        print(f"  {r[0]} {r[1]:<4} {r[2]:<5} {r[3]:.4f}  {str(r[4]):>4}  [{', '.join(f'{x:.3f}' for x in r[5])}]  "
              f"[{', '.join(f'{x:.2f}' for x in r[6])}]  [{', '.join(f'{x:.3f}' for x in r[7])}]")


if __name__ == "__main__":
    if "--fidelity" in sys.argv:
        fidelity_main()
    elif "--penalty" in sys.argv:
        os.makedirs(OUT, exist_ok=True)
        (fi, fv), prow = penalty_figures(penalty_runs())
        fi.savefig(os.path.join(OUT, "n2_110_s2pen_iter.png"), dpi=170)
        fv.savefig(os.path.join(OUT, "n2_110_s2pen_vs.png"), dpi=170)
        penalty_230().savefig(os.path.join(OUT, "n2_230_s2pen_s0.png"), dpi=170)
        # Pauli propagation at cutoff 1e-5, bare H against the penalty runs
        g = load("uno_1.10_block1c_exact")
        _, _, refs, _, _ = run_data(g)
        sn = ["S0", "³Πg", "¹Πg"]
        ppr = [("bare H", INK, (0, (4, 2)), pp_pen_series(load("uno_1.10_blockpp_c1e-5"),
                                                          load("uno_1.10_blockpp_c1e-5_replay1")
                                                          or load("uno_1.10_blockpp_c1e-5_replay"), 0.0))]
        xrows = []
        for l in ("0.02", "0.05"):
            pp, rep = load(f"uno_1.10_blockpp_c1e-5_s2pen{l}"), load(f"uno_1.10_blockpp_c1e-5_s2pen{l}_replay")
            if pp is None or rep is None:
                continue
            s = pp_pen_series(pp, rep, float(l))
            ppr.append((f"λ = {l}", PEN_COL[l], "-", s))
        if len(ppr) > 1:
            pp_penalty_trajectories(ppr, refs, sn).savefig(os.path.join(OUT, "n2_110_pp_s2pen_traj.png"), dpi=170)
            for lab, _, _, s in ppr:
                ent = [("true (exact replay)", s["tru"], s["t_var"], "o", "-"),
                       ("corrected", s["cor"], s["var_corr"], "^", "--"), ("raw", s["raw"], s["var"], "v", ":")]
                fig, pprow = state_extrapolation(ent, refs, sn, f"PP 1e-5, {lab}", grid=True)
                if lab != "bare H":
                    fig.savefig(os.path.join(OUT, f"n2_110_pp_c1e-5_s2pen{lab.split()[-1]}_extrap.png"), dpi=170)
                plt.close(fig)
                xrows += [(lab,) + r for r in pprow]
                if "tru_H" in s:
                    print(f"{lab}: true <S^2> at the end {np.round(s['t_S2'][-1], 4)}; true H energy - exact "
                          f"{np.round(s['tru_H'][-1] - refs, 4)}; estimate - exact {np.round(s['tru'][-1] - refs, 4)}")
            print("PP 1e-5 extrapolation over [v, 2v] (mHa vs exact): run | series | state | last | extrapolated ± | v | pts")
            for r in xrows:
                print(f"  {r[0]:<8} {r[1]:<20} {r[2]:<4} {r[3]:+9.4f} {r[4]:+9.4f} ± {r[5]:.4f}  {r[6]:.2e} {r[7]}")
        fig, rows230 = pp_penalty_230()
        if fig is not None:
            fig.savefig(os.path.join(OUT, "n2_230_pp_s2pen.png"), dpi=170)
            print("2.3 Å PP 1e-5 (mHa from the run's own target): run | state | best true (iteration) | true at the end"
                  " | corrected at the end | <S^2> at the end")
            for r in rows230:
                print(f"  {r[0]:<8} {r[1]}  {r[2]:+8.2f} ({r[3]:3d})  {r[4]:+8.2f}  {r[5]:+8.2f}  {r[6]:.2f}")
        print("2.3 Å PP 1e-5 extrapolation over [v, 2v] (mHa vs target): run | series | state | last | extrapolated ± | v | pts")
        for l, fig, rows in pp230_extrapolation():
            fig.savefig(os.path.join(OUT, f"n2_230_pp_c1e-5{'' if l == '0' else '_s2pen' + l}_extrap.png"), dpi=170)
            plt.close(fig)
            for r in rows:
                print(f"  {l:<5} {r[0]:<20} {r[1]:<6} {r[2]:+9.3f} {r[3]:+9.3f} ± {r[4]:.3f}  {r[5]:.2e} {r[6]}")
        print("λ | slot | state | <S^2> | last err (mHa) | stays below 1.6 / 0.1 / 0.03 mHa from iteration")
        for r in prow:
            print(f"  {r[0]:<5} {r[1]} {r[2]:<16} {r[3]:6.3f}  {r[4]:.2e}   {r[5]} / {r[6]} / {r[7]}")
    else:
        main()
