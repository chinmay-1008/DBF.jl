"""Plots for the block vDBF runs on the open Heisenberg chain (run_block_heisenberg.jl), against DMRG.

    python plot_heisenberg.py [N]          # from code/heisenberg/; reads results/heis_N<N>_block_c*.json

Energies are in units of J; differences in mJ (10^-3 J).  The three block states are
S0 (Sz = 0), the lowest triplet's Ms = 0 component (Sz = 0) and its Ms = +1 component (Sz = 1);
DMRG gives E0, ET_m0 and ET_m1 (results/dmrg_N<N>.json).  Raw = from the truncated operator;
corrected = minus the accumulated truncation change of the 3x3 block (as in run_block_pp.jl).

    heis_N<N>_energies.png   block eigenvalues against the block variance V_S, DMRG energies dashed (symlog y)
    heis_N<N>_zoom.png       per state, error from DMRG, raw and corrected per cutoff, common y range
    heis_N<N>_gap.png        the two spin-gap estimates (triplet - S0) against V_S, DMRG gap dashed
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
import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "n2_pes", "frozen_core"))
import plot_block as pb  # noqa: E402  (palette, style, window_fit)

RES = os.path.join(HERE, "results")
OUT = os.path.join(RES, "plots")
SERIES, INK, MUTED, GRID = pb.SERIES, pb.INK, pb.MUTED, pb.GRID
NAMES = ["S0", "T, Ms = 0", "T, Ms = +1"]
CUT_LS = {"1e-4": (0, (5, 2)), "1e-5": "-", "1e-6": (0, (1, 1.5))}


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
    fig.tight_layout(rect=(0, 0, 0.86, 0.88))
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
        ax.set_xscale("log")
        ax.set_xlim(lo_x / 1.5, 1.0)
        ax.set_ylim(-0.4 * W, 1.2 * W)
        ax.set_title(f"N = {N}: {NAMES[i]}", fontsize=9, color=INK, loc="left")
        ax.set_xlabel("$V_S$ (J², log)")
        ax.tick_params(labelsize=7.5)
        pb.style(ax)
    axes[0, 0].set_ylabel("energy − DMRG (mJ)")
    axes[0, -1].legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize=7.5)
    fig.suptitle("Zoom on each state: error from the matching DMRG level; all panels share x and y ranges\n"
                 "solid = corrected for truncation, thin = raw; starts on the right, converges to the left",
                 x=0.01, ha="left", fontsize=10, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.86))
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
    ax.set_ylim(0, top)
    ax.set_xlabel("block variance $V_S$ (J², log)")
    ax.set_ylabel("spin gap estimate (mJ)")
    ax.set_title(f"Heisenberg chain, N = {N}: lowest triplet − S0", fontsize=10, color=INK, loc="left")
    ax.legend(fontsize=6.5, loc="upper right")
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
                ax.axvspan(lo, hi, color=GRID, alpha=0.35, lw=0, zorder=0)
                ax.errorbar(0, fit["e0"], yerr=0 if np.isnan(fit["half"]) else fit["half"], fmt="o", color=SERIES[i],
                            mfc=mfc, ms=6, mec=SERIES[i] if kind == "raw" else INK, mew=1.0, capsize=3, clip_on=False,
                            zorder=5)
                rows.append((cut, kind, NAMES[i], y[-1], fit["e0"], fit["half"], fit["v"], fit["k"]))
                txt.append(f"{kind}: last {y[-1]:+.0f}, extrapolated {fit['e0']:+.0f} ± {fit['half']:.0f}")
                Y = [min(Y[0], y[ctx].min(), fit["e0"] - fit["half"]), max(Y[1], y[ctx].max(), fit["e0"] + fit["half"])]
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
                 "(shaded), v = its last variance\nfilled = corrected for truncation, hollow = raw; marker at 0 = "
                 "extrapolated energy, error bar = half the difference from a quadratic fit; all panels share x and y"
                 + (f"\ncutoff {', '.join(skipped)} left out: its corrected variance turned negative (truncation "
                    "correction broke down)" if skipped else ""),
                 x=0.01, ha="left", fontsize=9.5, color=INK)
    fig.subplots_adjust(left=0.065, right=0.985, bottom=0.13 / len(cuts), top=1 - 0.2 / len(cuts), wspace=0.22,
                        hspace=0.45)
    return fig, rows


def main():
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    os.makedirs(OUT, exist_ok=True)
    dm = json.load(open(os.path.join(RES, f"dmrg_N{N}.json")))
    refs = np.array([dm["E0"], dm["ET_m0"], dm["ET_m1"]])
    runs = load_runs(N)
    if not runs:
        sys.exit("no runs")
    energies(runs, dm, N).savefig(os.path.join(OUT, f"heis_N{N}_energies.png"), dpi=170)
    zoom(runs, refs, N).savefig(os.path.join(OUT, f"heis_N{N}_zoom.png"), dpi=170)
    gap(runs, dm, N).savefig(os.path.join(OUT, f"heis_N{N}_gap.png"), dpi=170)
    fig, rows = extrap(runs, refs, N)
    fig.savefig(os.path.join(OUT, f"heis_N{N}_extrap.png"), dpi=170)
    g_ref = (dm["ET_m1"] - dm["E0"]) * 1e3
    print(f"DMRG: E0 {dm['E0']:.8f}, gap {g_ref:.4f} mJ")
    for cut, d in runs.items():
        h = d["history"]
        it = len(h["lam"]) - 1
        lam, lamc = (np.array(h["lam"][-1]) - refs) * 1e3, (np.array(h["lam_corr"][-1]) - refs) * 1e3
        print(f"cutoff {cut}: iteration {it}, {d['n_rotations']} rotations, {h['terms'][-1]} terms, {h['time'][-1]:.0f} s, "
              f"V_S {h['VS'][-1]:.2e}{'' if d.get('finished') else ' (running)'}")
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
