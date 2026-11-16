"""Plots for the block vDBF runs on the open Heisenberg chain (run_block_heisenberg.jl), against DMRG.

    python plot_heisenberg.py [N] [--cuts 1e-5,...]   # from code/heisenberg/; reads results/heis_N<N>_block_c*.json
                                                     # default: the runs whose corrected variances stayed positive

Energies are in units of J; differences in mJ (10^-3 J).  The three block states are
S0 (Sz = 0), the lowest triplet's Ms = 0 component (Sz = 0) and its Ms = +1 component (Sz = 1);
DMRG gives E0, ET_m0 and ET_m1 (results/dmrg_N<N>.json).  Raw = from the truncated operator;
corrected = minus the accumulated truncation change of the 3x3 block (as in run_block_pp.jl).

    heis_N<N>_energies.png   block eigenvalues against the block variance V_S, DMRG energies dashed (symlog y)
    heis_N<N>_zoom.png       per state, error from DMRG, raw and corrected per cutoff, common y range
    heis_N<N>_gap.png        the two spin-gap estimates (triplet - S0) against V_S, DMRG gap dashed
    heis_N<N>_position.png   overlaps of the DMRG states with the block determinants at every position
    heis_N<N>_extrap.png     per-state extrapolation over [v, 2v], raw and corrected, cutoffs whose
                             corrected variances stayed positive (healthy())
"""
import glob
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker  # noqa: E402
import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "n2_pes", "frozen_core"))
import plot_block as pb  # noqa: E402  (palette, style, window_fit)

RES = os.path.join(HERE, "results")
OUT = os.path.join(RES, "plots")
SERIES, INK, MUTED, GRID = pb.SERIES, pb.INK, pb.MUTED, pb.GRID
NAMES = ["S0", "T, Ms = 0", "T, Ms = +1"]
CUT_LS = {"1e-4": (0, (5, 2)), "1e-5": "-", "1e-6": (0, (1, 1.5))}
TOL = 0.01                     # target: error within 1% of the state's own energy, |E - E_DMRG| <= 0.01 |E_DMRG|
BAND = "#c4c8cf"               # grey band marking that target


def tol_band(ax, lo, hi, label=True):
    """grey band [lo, hi] (mJ): the 1% target"""
    ax.axhspan(lo, hi, color=BAND, alpha=0.4, lw=0, zorder=0,
               label=f"within {TOL:.0%} of the exact energy" if label else None)


def load_runs(N):
    runs = {}
    for f in sorted(glob.glob(os.path.join(RES, f"heis_N{N}_block_c*.json"))):
        d = json.load(open(f))
        runs[d["settings"]["coeff"]] = d
    return runs


def series(d, refs):
    h = d["history"]
    e = lambda k: (np.array(h[k]) - refs[None, :]) * 1e3          # mJ from the matching DMRG level
    return np.array(h["VS"]), e("lam"), e("lam_corr"), np.array(h["var"]), np.array(h["var_corr"])


def energies(runs, dm, N):
    fig, ax = plt.subplots(figsize=(6.8, 4.4))
    e0 = dm["E0"]
    for cut, d in runs.items():
        h = d["history"]
        vs, lam = np.array(h["VS"]), (np.array(h["lam_corr"]) - e0) * 1e3
        for i in range(3):
            ax.plot(vs, lam[:, i], ls=CUT_LS.get(cut, "-"), color=SERIES[i], lw=1.5,
                    label=f"{NAMES[i]}, cutoff {cut}" if True else None)
            ax.plot(vs[0], lam[0, i], "o", color=SERIES[i], ms=5, mfc="white", mew=1.1)
            ax.plot(vs[-1], lam[-1, i], "o", color=SERIES[i], ms=5, mec="white", mew=0.8)
    g = (dm["ET_m1"] - e0) * 1e3
    tol_band(ax, -TOL * abs(e0) * 1e3, g + TOL * abs(dm["ET_m1"]) * 1e3)
    for e, name in ((0.0, "DMRG E0"), ((dm["ET_m1"] - e0) * 1e3, f"DMRG triplet ({(dm['ET_m1'] - e0) * 1e3:.2f})")):
        ax.axhline(e, color=INK, lw=0.9, ls="--", zorder=1)
        ax.annotate(name, xy=(1.01, e), xycoords=("axes fraction", "data"), fontsize=7.5, va="center", color=INK)
    ax.set_xscale("log")
    ax.set_yscale("symlog", linthresh=20, linscale=0.8)
    ax.set_xlabel("block variance $V_S$ (J², log)")
    ax.set_ylabel("energy − DMRG E0 (mJ, symlog)")
    ax.set_title(f"Heisenberg chain, N = {N}, open", fontsize=10, color=INK, loc="left")
    ax.legend(fontsize=7, loc="lower left", ncol=1)
    pb.style(ax)
    fig.suptitle("Block eigenvalues (corrected) against block variance; DMRG energies dashed\n"
                 "starts on the right (hollow dots), converges to the left",
                 x=0.01, ha="left", fontsize=10, color=INK)
    fig.subplots_adjust(left=0.12, right=0.80, bottom=0.12, top=0.80)
    return fig


def zoom(runs, refs, N):
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.7), squeeze=False)
    W, lo_x = 0.0, np.inf
    for cut, d in runs.items():
        vs, raw, cor, _, _ = series(d, refs)
        n = len(vs)
        for i in range(3):
            ax = axes[0, i]
            ls = CUT_LS.get(cut, "-")
            ax.plot(vs, cor[:, i], ls=ls, color=SERIES[i], lw=1.6, label=f"cutoff {cut}, corrected")
            ax.plot(vs, raw[:, i], ls=ls, color=SERIES[i], lw=0.8, alpha=0.45, label=f"cutoff {cut}, raw")
            W = max(W, abs(cor[-1, i]) * 3, abs(raw[-1, i]) * 3, abs(cor[n // 2, i]))
        lo_x = min(lo_x, vs.min())
    for i, ax in enumerate(axes[0]):
        ax.axhline(0, color=INK, lw=1, ls="--", zorder=1)
        w = TOL * abs(refs[i]) * 1e3
        tol_band(ax, -w, w, label=(i == 2))
        W = max(W, 1.1 * w / 0.4)
        ax.set_xscale("log")
        ax.set_xlim(lo_x / 1.1, 3.0)
        ax.set_xticks([0.8, 1.0, 1.5, 2.0, 3.0])
        ax.xaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%g"))
        ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        ax.set_ylim(-0.4 * W, 1.2 * W)
        ax.set_title(f"N = {N}: {NAMES[i]}", fontsize=9, color=INK, loc="left")
        ax.set_xlabel("$V_S$ (J², log)")
        ax.tick_params(labelsize=7.5)
        pb.style(ax)
    axes[0, 0].set_ylabel("energy − DMRG (mJ)")
    axes[0, -1].legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize=7.5)
    fig.suptitle("Zoom on each state: error from the matching DMRG level; all panels share x and y ranges\n"
                 "solid = corrected for truncation, thin = raw; grey = within 1% of the state's exact energy; "
                 "starts on the right, converges to the left",
                 x=0.01, ha="left", fontsize=10, color=INK)
    fig.subplots_adjust(left=0.06, right=0.84, bottom=0.16, top=0.78, wspace=0.22)
    return fig


def gap(runs, dm, N):
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    g_ref = (dm["ET_m1"] - dm["E0"]) * 1e3
    top = 4 * g_ref
    for cut, d in runs.items():
        h = d["history"]
        vs = np.array(h["VS"])
        lc = np.array(h["lam_corr"]) * 1e3
        late = lc[len(lc) // 2:, 1:] - lc[len(lc) // 2:, :1]      # gap estimates over the second half of the run
        top = max(top, 1.15 * late.max())
        for key, lw, alpha, tag in (("lam_corr", 1.6, 1.0, "corrected"), ("lam", 0.8, 0.45, "raw")):
            lam = np.array(h[key]) * 1e3
            for i, col in ((1, SERIES[1]), (2, SERIES[2])):
                ax.plot(vs, lam[:, i] - lam[:, 0], ls=CUT_LS.get(cut, "-"), color=col, lw=lw, alpha=alpha,
                        label=f"{NAMES[i]} − S0, cutoff {cut}, {tag}")
    ax.axhline(g_ref, color=INK, lw=1, ls="--")
    ax.annotate(f"DMRG gap {g_ref:.2f}", xy=(1.01, g_ref), xycoords=("axes fraction", "data"), fontsize=7.5,
                va="center", color=INK)
    ax.set_xscale("log")
    lo_x = min(np.min(d["history"]["VS"]) for d in runs.values())
    ax.set_xlim(lo_x / 1.1, 3.0)                                   # same x range as the zoom figure
    ax.set_xticks([0.8, 1.0, 1.5, 2.0, 3.0])
    ax.xaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%g"))
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_ylim(0, top)
    ax.set_xlabel("block variance $V_S$ (J², log)")
    ax.set_ylabel("spin gap estimate (mJ)")
    ax.set_title(f"Heisenberg chain, N = {N}: lowest triplet − S0", fontsize=10, color=INK, loc="left")
    ax.legend(fontsize=6.5, loc="lower right", bbox_to_anchor=(1.0, 0.15))
    pb.style(ax)
    fig.tight_layout(rect=(0, 0, 0.88, 1))
    return fig


def healthy(d):
    """a run whose truncation correction still makes sense: every state's corrected variance is positive at the end.
    A negative corrected variance (and a corrected energy below the exact one) means the dropped terms have grown
    too large for the first-order correction, and neither series can be extrapolated."""
    return min(d["history"]["var_corr"][-1]) > 0


def extrap(runs, refs, N):
    """per state, energy against its own variance, linear fit over [v, 2v], v = last variance; raw (hollow) and
    corrected (filled) on the same panel, one row per usable cutoff, x and y ranges shared by all panels"""
    cuts = [c for c, d in runs.items() if healthy(d)]
    fig, axes = plt.subplots(len(cuts), 3, figsize=(11.5, 3.4 * len(cuts) + 1.0), squeeze=False)
    X = 1.25 * max(2 * x[-1] for d in (runs[c] for c in cuts) for k in ("var", "var_corr")
                   for x in np.array(d["history"][k]).T)               # common x: up to just past the widest window
    rows, Y = [], [0.0, 0.0]
    for r_, cut in enumerate(cuts):
        vs, raw, cor, var, varc = series(runs[cut], refs)
        for i in range(3):
            ax = axes[r_, i]
            ax.axhline(0, color=INK, lw=1, ls="--", zorder=1)
            txt = []
            for kind, y_all, x_all, mfc, ls in (("corrected", cor, varc, SERIES[i], "-"),
                                                 ("raw", raw, var, "white", (0, (4, 2)))):
                x, y = x_all[1:, i], y_all[1:, i]
                fit = pb.window_fit(x, y)
                if fit is None:
                    rows.append((cut, kind, NAMES[i], y[-1], np.nan, np.nan, x[-1], 0))
                    continue
                sel, lo, hi = fit["sel"], fit["lo"], fit["hi"]
                ctx = (x > 0) & (x <= X)
                ax.plot(x[ctx & ~sel], y[ctx & ~sel], "o", color=SERIES[i], mfc=mfc, ms=2.4, mew=0.6, alpha=0.3)
                ax.plot(x[sel], y[sel], "o", color=SERIES[i], mfc=mfc, ms=3.0, mew=0.7, label=f"{kind} ({fit['k']} pts)")
                xs = np.linspace(0, hi, 50)
                ax.plot(xs, np.polyval(fit["lin"], xs), ls=ls, color=SERIES[i], lw=1.1)
                ax.axvspan(lo, hi, color=SERIES[i], alpha=0.05, lw=0, zorder=0)
                ax.errorbar(0, fit["e0"], yerr=0 if np.isnan(fit["half"]) else fit["half"], fmt="o", color=SERIES[i],
                            mfc=mfc, ms=6, mec=SERIES[i] if kind == "raw" else INK, mew=1.0, capsize=3, clip_on=False,
                            zorder=5)
                rows.append((cut, kind, NAMES[i], y[-1], fit["e0"], fit["half"], fit["v"], fit["k"]))
                txt.append(f"{kind}: last {y[-1]:+.0f}, extrapolated {fit['e0']:+.0f} ± {fit['half']:.0f}")
                Y = [min(Y[0], y[ctx].min(), fit["e0"] - fit["half"]), max(Y[1], y[ctx].max(), fit["e0"] + fit["half"])]
            w = TOL * abs(refs[i]) * 1e3
            tol_band(ax, -w, w, label=(r_ == 0 and i == 0))
            Y = [min(Y[0], -w), max(Y[1], w)]
            ax.text(0.98, 0.04, "\n".join(txt), transform=ax.transAxes, ha="right", va="bottom", fontsize=7, color=INK,
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=GRID, lw=0.6))
            ax.set_title(f"cutoff {cut}: {NAMES[i]}", fontsize=9, color=INK, loc="left")
            ax.set_xlabel("energy variance of this state (J²)")
            ax.tick_params(labelsize=7.5)
            ax.legend(fontsize=7, loc="upper left")
            pb.style(ax)
        axes[r_, 0].set_ylabel("energy − DMRG (mJ)")
    pad = 0.06 * (Y[1] - Y[0])
    for ax in axes.flat:
        ax.set_xlim(0, X)
        ax.set_ylim(Y[0] - pad, Y[1] + pad)
    skipped = [c for c in runs if c not in cuts]
    fig.suptitle(f"Per-state extrapolation, N = {N}: each state's energy against its own variance, linear fit over [v, 2v] "
                 "(tinted), v = its last variance\nfilled = corrected for truncation, hollow = raw; marker at 0 = "
                 "extrapolated energy, error bar = half the difference from a quadratic fit;\ngrey = within 1% of the state's exact energy; "
                 "all panels share x and y"
                 + (f"\ncutoff {', '.join(skipped)} left out: its corrected variance turned negative (truncation "
                    "correction broke down)" if skipped else ""),
                 x=0.01, ha="left", fontsize=9.5, color=INK)
    fig.subplots_adjust(left=0.065, right=0.985, bottom=0.13 / len(cuts), top=1 - 0.24 / len(cuts), wspace=0.22,
                        hspace=0.45)
    return fig, rows


def position(N):
    """position_check.jl: overlaps of the exact (DMRG) states with the block determinants placed anywhere on
    the chain, and the Ms = +1 magnetisation; the run's choice (bond N/2, N/2+1 and site N/2) marked"""
    f = os.path.join(RES, f"position_N{N}.json")
    if not os.path.exists(f):
        return None
    d = json.load(open(f))
    c = d["centre"]
    ff = np.array(d["flipflop_S0"])
    si = [(i + 1, w) for i, w in enumerate(d["single_T1"]) if w is not None]
    m = np.array(d["Sz"])
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.5))
    ax = axes[0]
    ax.plot(np.arange(1, N) + 0.5, ff, "o", color=SERIES[0], ms=2.6, mec="none")
    ax.plot(c + 0.5, ff[c - 1], "o", color=INK, ms=6, mfc="none", mew=1.2, label=f"used: bond {c}–{c + 1}")
    ax.set_yscale("log")
    ax.set_xlabel("bond position")
    ax.set_title("|f⟩ (pair exchanged) on S0", fontsize=9, color=INK, loc="left")
    ax.set_ylabel("weight of the exact state on it")
    ax = axes[1]
    x, w = np.array([a for a, _ in si]), np.array([b for _, b in si])
    ax.plot(x, w, "o", color=SERIES[2], ms=2.6, mec="none")
    ax.plot(c, w[list(x).index(c)], "o", color=INK, ms=6, mfc="none", mew=1.2, label=f"used: site {c}")
    ax.set_yscale("log")
    ax.set_xlabel("flipped site (↓ → ↑)")
    ax.set_title("|s⟩ (single flip) on T, Ms = +1", fontsize=9, color=INK, loc="left")
    ax = axes[2]
    ax.axhline(0, color=INK, lw=0.8)
    ax.plot(np.arange(1, N + 1), m, "-", color=SERIES[1], lw=0.8)
    pair = np.array(d["Sz_pairs"])
    ax.plot(np.arange(1, N, 2) + 0.5, pair, "-", color=INK, lw=1.6, label="pair sums (smooth part)")
    ax.axvline(c + 0.5, color=MUTED, lw=0.8, ls=(0, (3, 2)))
    ax.set_xlabel("site")
    ax.set_title("$\\langle S^z_i \\rangle$ in T, Ms = +1", fontsize=9, color=INK, loc="left")
    for a in axes:
        a.legend(fontsize=7.5, loc="upper center" if a is axes[2] else "lower right", framealpha=0.92)
        a.tick_params(labelsize=7.5)
        pb.style(a)
    fig.suptitle(f"Where the target states live on the open chain, N = {N} (DMRG): weight of each exact state on the block "
                 "determinant placed at each position\nopen circle = the position the block runs use (the centre); "
                 "Néel = ↑ on odd sites", x=0.01, ha="left", fontsize=10, color=INK)
    fig.subplots_adjust(left=0.07, right=0.985, bottom=0.15, top=0.8, wspace=0.28)
    return fig


def main():
    args = sys.argv[1:]
    cuts = args[args.index("--cuts") + 1].split(",") if "--cuts" in args else None
    N = int(args[0]) if args and args[0] != "--cuts" else 100
    os.makedirs(OUT, exist_ok=True)
    dm = json.load(open(os.path.join(RES, f"dmrg_N{N}.json")))
    refs = np.array([dm["E0"], dm["ET_m0"], dm["ET_m1"]])
    runs = load_runs(N)
    if not runs:
        sys.exit("no runs")
    every = runs
    runs = {c: d for c, d in every.items() if (c in cuts if cuts else healthy(d))}
    print("plotting cutoffs", ", ".join(runs), "| left out:", ", ".join(c for c in every if c not in runs) or "none")
    energies(runs, dm, N).savefig(os.path.join(OUT, f"heis_N{N}_energies.png"), dpi=170)
    zoom(runs, refs, N).savefig(os.path.join(OUT, f"heis_N{N}_zoom.png"), dpi=170)
    gap(runs, dm, N).savefig(os.path.join(OUT, f"heis_N{N}_gap.png"), dpi=170)
    fp = position(N)
    if fp is not None:
        fp.savefig(os.path.join(OUT, f"heis_N{N}_position.png"), dpi=170)
    fig, rows = extrap(runs, refs, N)
    fig.savefig(os.path.join(OUT, f"heis_N{N}_extrap.png"), dpi=170)
    g_ref = (dm["ET_m1"] - dm["E0"]) * 1e3
    print(f"DMRG: E0 {dm['E0']:.8f}, gap {g_ref:.4f} mJ")
    for cut, d in every.items():
        h = d["history"]
        it = len(h["lam"]) - 1
        lam, lamc = (np.array(h["lam"][-1]) - refs) * 1e3, (np.array(h["lam_corr"][-1]) - refs) * 1e3
        print(f"cutoff {cut}: iteration {it}, {d['n_rotations']} rotations, {h['terms'][-1]} terms, {h['time'][-1]:.0f} s, "
              f"V_S {h['VS'][-1]:.2e}{'' if d.get('finished') else ' (last checkpoint, run not finished)'}")
        print("   errors vs DMRG (mJ), raw:       " + "  ".join(f"{x:+9.3f}" for x in lam))
        print("   errors vs DMRG (mJ), corrected: " + "  ".join(f"{x:+9.3f}" for x in lamc))
    print("extrapolation over [v, 2v] (mJ vs DMRG): cutoff | series | state | last | extrapolated ± | v | points")
    for r in rows:
        print(f"  {r[0]:<5} {r[1]:<9} {r[2]:<11} {r[3]:+9.3f} {r[4]:+9.3f} ± {r[5]:.3f}   {r[6]:.2e}  {r[7]}")
    print("spin gap from the extrapolated energies (mJ; DMRG %.2f): cutoff | series | Ms = 0 | Ms = +1" % g_ref)
    ext = {(r[0], r[1], r[2]): (r[4], r[5]) for r in rows}
    for cut, kind in sorted({(r[0], r[1]) for r in rows}):
        (s, ds), (t0, dt0), (t1, dt1) = (ext[(cut, kind, n)] for n in NAMES)
        print(f"  {cut:<5} {kind:<9} {g_ref + t0 - s:+8.1f} ± {np.hypot(ds, dt0):.0f}   {g_ref + t1 - s:+8.1f} ± {np.hypot(ds, dt1):.0f}")


if __name__ == "__main__":
    main()
